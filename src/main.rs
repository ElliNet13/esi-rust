// Eternal Computer Virtual Machine with Framebuffer Support
//
// Safe-Rust port of the original C implementation. No `unsafe` blocks are
// used anywhere in this file. The one place the C code depended on raw
// pointer aliasing -- writing a `struct timespec` on top of `mem[64]` via
// a pointer cast -- has been reimplemented with explicit integer
// arithmetic that reproduces the same little-endian byte layout instead.
//
// Unlike the original, every memory access derived from program-controlled
// values (the `a`, `b`, `c` operands, and jump/interrupt targets) is
// bounds-checked explicitly. A corrupted or malicious boot image will
// produce a clear, descriptive error instead of an out-of-bounds panic or,
// as in the original C, undefined behavior / a segfault.

use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fmt;
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

/// A program-controlled memory access (an operand, jump target, or
/// interrupt vector) landed outside the VM's address space.
#[derive(Debug)]
struct OutOfBoundsError {
    context: &'static str,
    index: i64,
    len: usize,
}

impl fmt::Display for OutOfBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "out-of-bounds memory access ({}): word index {} is not in 0..{} \
             -- the boot image is likely corrupted or malformed",
            self.context, self.index, self.len
        )
    }
}

impl Error for OutOfBoundsError {}

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

    /// Validate and convert a word index coming from program-controlled
    /// data (as opposed to indices we manage ourselves, like fixed
    /// low-memory slots, which are always valid by construction).
    fn checked_index(&self, idx: i32, context: &'static str) -> Result<usize, OutOfBoundsError> {
        let idx = idx as i64;
        if idx < 0 || idx as u64 >= self.mem.len() as u64 {
            Err(OutOfBoundsError {
                context,
                index: idx,
                len: self.mem.len(),
            })
        } else {
            Ok(idx as usize)
        }
    }

    fn read(&self, idx: i32, context: &'static str) -> Result<i32, OutOfBoundsError> {
        let i = self.checked_index(idx, context)?;
        Ok(self.mem[i])
    }

    fn write(&mut self, idx: i32, value: i32, context: &'static str) -> Result<(), OutOfBoundsError> {
        let i = self.checked_index(idx, context)?;
        self.mem[i] = value;
        Ok(())
    }

    /// Set the program counter to a program-controlled jump/interrupt
    /// target, rejecting anything outside the VM's address space.
    fn set_pc(&mut self, target: i32, context: &'static str) -> Result<(), OutOfBoundsError> {
        self.pc = self.checked_index(target, context)?;
        Ok(())
    }

    /// Load a binary image into memory as little-endian 32-bit words,
    /// matching `fread(mem, 4, MEM_SIZE, f)` on a little-endian host.
    /// Returns the number of words that didn't fit, if any, so the caller
    /// can warn about truncation instead of silently dropping data.
    fn load_program(&mut self, path: &str) -> io::Result<usize> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let total_words = bytes.len().div_ceil(4);
        let mut truncated = 0usize;

        for (i, chunk) in bytes.chunks(4).enumerate() {
            if i >= self.mem.len() {
                truncated = total_words - i;
                break;
            }
            let mut word_bytes = [0u8; 4];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            self.mem[i] = i32::from_le_bytes(word_bytes);
        }
        Ok(truncated)
    }

    /// Fetch the next instruction word, following one level of pointer
    /// indirection, exactly as the C `fetch()` function did. `pc` is
    /// managed entirely by the VM itself (never set from unchecked
    /// program data -- see `set_pc`), so indexing it directly is safe;
    /// the indirect operand it points at, however, is program-controlled
    /// and is bounds-checked.
    fn fetch(&mut self) -> Result<i32, OutOfBoundsError> {
        if self.pc >= self.mem.len() {
            return Err(OutOfBoundsError {
                context: "instruction fetch (pc ran off the end of memory)",
                index: self.pc as i64,
                len: self.mem.len(),
            });
        }
        let raw = self.mem[self.pc];
        self.pc += 1;
        if raw & 1 != 0 {
            let addr = raw / 4;
            let val = self.read(addr, "indirect operand fetch")?;
            Ok(val / 4)
        } else {
            Ok(raw / 4)
        }
    }

    /// Store the current wall-clock time into mem[64..68], reproducing the
    /// byte layout the C code got "for free" by casting `&mem[64]` to a
    /// `struct timespec *` (two 8-byte fields: tv_sec, tv_nsec) on a
    /// 64-bit little-endian machine. Each 8-byte field becomes two
    /// consecutive 32-bit words: low half then high half. Indices 64..68
    /// are always valid since MEM_SIZE is far larger than that.
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
/// Bounds are clamped against both the surface's actual pixel buffer and
/// the VM's memory, so a mismatched surface format can't cause an
/// out-of-range read from `mem`.
fn copy_framebuffer_to_surface(mem: &[i32], surface: &mut WindowSurfaceRef) {
    surface.with_lock_mut(|pixels: &mut [u8]| {
        let fb_words_available = mem.len().saturating_sub(FB_ADDR_WORDS);
        let copy_len = FB_SIZE.min(pixels.len()).min(fb_words_available * 4);
        let word_count = copy_len / 4;
        for i in 0..word_count {
            let bytes = mem[FB_ADDR_WORDS + i].to_le_bytes();
            let offset = i * 4;
            pixels[offset..offset + 4].copy_from_slice(&bytes);
        }
    });
}

fn main() {
    if let Err(err) = run() {
        eprintln!("esi_vm: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        let prog_name = args.first().map(String::as_str).unwrap_or("esi_vm");
        return Err(format!("Usage: {prog_name} <binary>").into());
    }

    let mut vm = Vm::new();
    let truncated_words = vm
        .load_program(&args[1])
        .map_err(|e| format!("failed to load boot image '{}': {e}", args[1]))?;
    if truncated_words > 0 {
        eprintln!(
            "esi_vm: warning: boot image is larger than VM memory; {truncated_words} \
             trailing word(s) were not loaded"
        );
    }

    let sdl_context = sdl3::init().map_err(|e| format!("failed to init SDL3: {e}"))?;
    let video_subsystem = sdl_context
        .video()
        .map_err(|e| format!("failed to init video subsystem: {e}"))?;
    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| format!("failed to get event pump: {e}"))?;

    let window = video_subsystem
        .window("ESI Virtual Machine", FB_WIDTH as u32, FB_HEIGHT as u32)
        .build()
        .map_err(|e| format!("failed to create window: {e}"))?;

    let quit_flag = Arc::new(AtomicBool::new(false));
    {
        // Handles the case where the terminal (not the SDL window) still
        // has focus when Ctrl+C is pressed, so it arrives as a real SIGINT
        // instead of an SDL key event.
        let quit_flag = Arc::clone(&quit_flag);
        ctrlc::set_handler(move || {
            quit_flag.store(true, Ordering::SeqCst);
        })
        .map_err(|e| format!("failed to install Ctrl+C handler: {e}"))?;
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

        let a = vm.fetch()?;
        let b = vm.fetch()?;
        let c = vm.fetch()?;

        if a == -1 {
            // Read scan code from keyboard
            let key = kb_poll(&mut pending_keys);
            vm.write(b, key, "keyboard read target address")?;
        } else if b == -1 {
            // Write text to console (low byte of mem[a], matching a raw
            // single-byte `write()` of a little-endian int)
            let word = vm.read(a, "console write source address")?;
            let byte = (word as u32 & 0xFF) as u8;
            stdout_lock.write_all(&[byte])?;
        } else {
            // Subleq operation
            if a == 64 {
                vm.update_clock();
            }
            let mem_a = vm.read(a, "subleq operand a")?;
            let mem_b = vm.read(b, "subleq operand b")?;
            let new_b = mem_b.wrapping_sub(mem_a);
            vm.write(b, new_b, "subleq operand b (storing result)")?;
            if new_b <= 0 {
                vm.set_pc(c, "subleq branch target")?;
            }

            // mem[0] and mem[1] are fixed low-memory slots that are
            // always valid given MEM_SIZE, so these two are safe to
            // index directly.
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
                    let interrupt_vector = vm.mem[0] / 4;
                    vm.set_pc(interrupt_vector, "interrupt vector (mem[0])")?;
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