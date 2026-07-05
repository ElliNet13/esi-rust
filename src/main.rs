// Eternal Computer Virtual Machine with Framebuffer Support
//
// Safe-Rust port of the original C implementation. No `unsafe` blocks are
// used anywhere in this file. The one place the C code depended on raw
// pointer aliasing -- writing a `struct timespec` on top of `mem[64]` via
// a pointer cast -- has been reimplemented with explicit integer
// arithmetic that reproduces the same little-endian byte layout instead.
//
// As in the original, there is no memory bounds checking beyond what the
// language gives you for free: out-of-range indices will panic (safe
// Rust's "undefined behavior" is a controlled panic, not memory
// corruption), rather than being silently ignored.

use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Mod};
use sdl3::video::WindowSurfaceRef;

const FB_WIDTH: usize = 800;
const FB_HEIGHT: usize = 512;
const FB_SIZE: usize = FB_WIDTH * FB_HEIGHT * 4; // bytes
const MEM_SIZE: usize = 3 << 27; // number of 32-bit words (~1.6 GB)
const FB_ADDR_BYTES: usize = 0x6000_0000 - FB_SIZE; // byte address
const FB_ADDR_WORDS: usize = FB_ADDR_BYTES / 4; // word index into `mem`
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(1000 / 30); // 30 FPS
const TIMER_INTERRUPT_THRESHOLD: i64 = 800_000;

struct Vm {
    mem: Vec<i32>,
    pc: usize,
    timer: i64,
}

impl Vm {
    fn new() -> Self {
        Vm {
            mem: vec![0i32; MEM_SIZE],
            pc: 0,
            timer: 0,
        }
    }

    /// Load a binary image into memory as little-endian 32-bit words,
    /// matching `fread(mem, 4, MEM_SIZE, f)` on a little-endian host.
    fn load_program(&mut self, path: &str) -> io::Result<()> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        for (i, chunk) in bytes.chunks(4).enumerate() {
            if i >= self.mem.len() {
                break;
            }
            let mut word_bytes = [0u8; 4];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            self.mem[i] = i32::from_le_bytes(word_bytes);
        }
        Ok(())
    }

    /// Fetch the next instruction word, following one level of pointer
    /// indirection, exactly as the C `fetch()` function did.
    fn fetch(&mut self) -> i32 {
        let raw = self.mem[self.pc];
        self.pc += 1;
        if raw & 1 != 0 {
            self.mem[(raw / 4) as usize] / 4
        } else {
            raw / 4
        }
    }

    /// Store the current wall-clock time into mem[64..68], reproducing the
    /// byte layout the C code got "for free" by casting `&mem[64]` to a
    /// `struct timespec *` (two 8-byte fields: tv_sec, tv_nsec) on a
    /// 64-bit little-endian machine. Each 8-byte field becomes two
    /// consecutive 32-bit words: low half then high half.
    fn update_clock(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        let sec = now.as_secs() as i64;
        let nsec = now.subsec_nanos() as i64;

        self.mem[64] = (sec & 0xFFFF_FFFF) as i32;
        self.mem[65] = ((sec >> 32) & 0xFFFF_FFFF) as i32;
        self.mem[66] = (nsec & 0xFFFF_FFFF) as i32;
        self.mem[67] = ((nsec >> 32) & 0xFFFF_FFFF) as i32;
    }
}

/// Drain every SDL event currently queued (cheap no-op when the queue is
/// empty). Keyboard events are queued for the guest program to consume via
/// `kb_poll`; window-close and the Ctrl+C key combo set `quit_flag` so the
/// main loop can stop promptly, independent of whether the guest happens to
/// be reading the keyboard right now.
fn drain_sdl_events(
    event_pump: &mut sdl3::EventPump,
    pending_keys: &mut VecDeque<i32>,
    quit_flag: &AtomicBool,
) {
    while let Some(event) = event_pump.poll_event() {
        match event {
            Event::Quit { .. } => {
                quit_flag.store(true, Ordering::SeqCst);
            }
            Event::KeyDown {
                scancode: Some(sc),
                keycode,
                keymod,
                ..
            } => {
                if keycode == Some(Keycode::C) && keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD) {
                    quit_flag.store(true, Ordering::SeqCst);
                }
                pending_keys.push_back(sc as i32);
            }
            Event::KeyUp {
                scancode: Some(sc), ..
            } => {
                pending_keys.push_back(-(sc as i32));
            }
            _ => {}
        }
    }
}

/// Pop one pending key event for the guest program.
/// Returns: positive = key-down scancode, negative = key-up, 0 = none.
fn kb_poll(pending_keys: &mut VecDeque<i32>) -> i32 {
    pending_keys.pop_front().unwrap_or(0)
}

/// Copy the VM's framebuffer region into the SDL surface's pixel buffer,
/// via the surface's safe pixel-locking API rather than a raw memcpy.
fn copy_framebuffer_to_surface(mem: &[i32], surface: &mut WindowSurfaceRef) {
    surface.with_lock_mut(|pixels: &mut [u8]| {
        let copy_len = FB_SIZE.min(pixels.len());
        let word_count = copy_len / 4;
        for i in 0..word_count {
            let bytes = mem[FB_ADDR_WORDS + i].to_le_bytes();
            let offset = i * 4;
            pixels[offset..offset + 4].copy_from_slice(&bytes);
        }
    });
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <binary>", args.get(0).map(String::as_str).unwrap_or("esi_vm"));
        std::process::exit(1);
    }

    let mut vm = Vm::new();
    vm.load_program(&args[1])?;

    let sdl_context = sdl3::init().expect("failed to init SDL3");
    let video_subsystem = sdl_context.video().expect("failed to init video subsystem");
    let mut event_pump = sdl_context.event_pump().expect("failed to get event pump");

    let window = video_subsystem
        .window("ESI Virtual Machine", FB_WIDTH as u32, FB_HEIGHT as u32)
        .build()
        .expect("failed to create window");

    let quit_flag = Arc::new(AtomicBool::new(false));
    {
        // Handles the case where the terminal (not the SDL window) still
        // has focus when Ctrl+C is pressed, so it arrives as a real SIGINT
        // instead of an SDL key event.
        let quit_flag = Arc::clone(&quit_flag);
        ctrlc::set_handler(move || {
            quit_flag.store(true, Ordering::SeqCst);
        })
        .expect("failed to install Ctrl+C handler");
    }
    let mut pending_keys: VecDeque<i32> = VecDeque::new();

    let mut last_render = Instant::now() - DISPLAY_UPDATE_INTERVAL;
    let stdout = io::stdout();
    let mut stdout_lock = BufWriter::new(stdout.lock());
    let mut event_poll_counter: u32 = 0;

    // Translated from the original `do { ... } while (c);` loop: run the
    // body at least once, then stop when the last-fetched `c` is zero.
    loop {
        // Draining SDL's event queue involves a lock/syscall even when
        // empty; doing it every single VM instruction (potentially
        // hundreds of millions per second) was a real cost. Every 4096
        // instructions is still far more often than a human can react,
        // so window-close/Ctrl+C/keyboard input all stay responsive.
        event_poll_counter = event_poll_counter.wrapping_add(1);
        if event_poll_counter & 0xFFF == 0 {
            drain_sdl_events(&mut event_pump, &mut pending_keys, &quit_flag);
            if quit_flag.load(Ordering::SeqCst) {
                break;
            }
        }

        let a = vm.fetch();
        let b = vm.fetch();
        let c = vm.fetch();

        if a == -1 {
            // Read scan code from keyboard
            vm.mem[b as usize] = kb_poll(&mut pending_keys);
        } else if b == -1 {
            // Write text to console (low byte of mem[a], matching a raw
            // single-byte `write()` of a little-endian int)
            let byte = (vm.mem[a as usize] as u32 & 0xFF) as u8;
            stdout_lock.write_all(&[byte])?;
        } else {
            // Subleq operation
            if a == 64 {
                vm.update_clock();
            }
            let new_b = vm.mem[b as usize].wrapping_sub(vm.mem[a as usize]);
            vm.mem[b as usize] = new_b;
            if new_b <= 0 {
                vm.pc = c as usize;
            }

            if vm.mem[0] != 0 {
                vm.timer += 1;
                if vm.timer > TIMER_INTERRUPT_THRESHOLD {
                    // Update display
                    let now = Instant::now();
                    if now.duration_since(last_render) >= DISPLAY_UPDATE_INTERVAL {
                        if let Ok(mut surface) = window.surface(&event_pump) {
                            copy_framebuffer_to_surface(&vm.mem, &mut surface);
                            let _ = surface.update_window();
                        }
                        stdout_lock.flush()?;
                        last_render = now;
                    }
                    // Trigger interrupt
                    vm.mem[1] = (vm.pc as i32).wrapping_mul(4);
                    vm.pc = (vm.mem[0] / 4) as usize;
                    vm.timer = 0;
                }
            }
        }

        if c == 0 {
            break;
        }
    }

    stdout_lock.flush()?;
    Ok(())
}