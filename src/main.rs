// Eternal Computer Virtual Machine with Framebuffer Support
//
// Safe-Rust port of the original C implementation. No `unsafe` blocks are
// used anywhere in this file. The one place the C code depended on raw
// pointer aliasing -- writing a `struct timespec` on top of `mem[64]` via
// a pointer cast -- has been reimplemented with explicit integer
// arithmetic that reproduces the same little-endian byte layout instead.
//
// Every memory access derived from program-controlled values (the `a`,
// `b`, `c` operands, and jump/interrupt targets) is bounds-checked
// explicitly. A corrupted or malicious boot image produces a clear,
// descriptive error instead of an out-of-bounds panic or, as in the
// original C, undefined behavior / a segfault.
//
// `--no-gui` skips SDL entirely: no window is created, framebuffer writes
// are left as ordinary memory writes (with a one-time notice the first
// time the boot image touches that region), and keyboard input comes from
// the terminal itself (raw mode, real typed characters) instead of SDL
// scancodes, since there's no window to capture those from.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use sdl3::event::Event;
use sdl3::keyboard::{Keycode, Mod};
use sdl3::video::WindowSurfaceRef;

const FB_WIDTH: usize = 800;
const FB_HEIGHT: usize = 512;
const FB_SIZE: usize = FB_WIDTH * FB_HEIGHT * 4; // bytes
const MEM_SIZE: usize = 3 << 27; // number of 32-bit words (~1.6 GB)
const FB_ADDR_BYTES: usize = 0x6000_0000 - FB_SIZE; // byte address
const FB_ADDR_WORDS: usize = FB_ADDR_BYTES / 4; // word index into `mem`
const FB_WORD_COUNT: usize = FB_SIZE / 4;
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(1000 / 30); // 30 FPS
const TIMER_INTERRUPT_THRESHOLD: i64 = 800_000;
const EVENT_POLL_INTERVAL_INSTRUCTIONS: u32 = 0xFFF; // ~every 4096 instructions

/// Eternal Computer Virtual Machine with framebuffer support.
#[derive(Parser)]
#[command(name = "esi_vm", version, about = "Eternal Computer Virtual Machine")]
struct Cli {
    /// Path to the ESI boot image to run (.bootimage or .bootimage.xz)
    #[arg(default_value = "")]
    binary: String,

    /// Run headless: no window is created, framebuffer writes become
    /// harmless no-ops (a one-time notice is printed the first time the
    /// boot image touches the framebuffer), and keyboard input is read
    /// from the terminal instead of SDL.
    #[arg(long)]
    no_gui: bool,
}

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

/// Lazily-allocated memory: only stores values that have been written to
/// or explicitly read, defaulting to zero for unallocated regions.
struct LazyMemory {
    data: Vec<i32>,
}

impl LazyMemory {
    fn new() -> Self {
        LazyMemory {
            data: Vec::new(),
        }
    }

    fn get(&self, idx: usize) -> i32 {
        self.data.get(idx).copied().unwrap_or(0)
    }

    fn set(&mut self, idx: usize, val: i32) {
        if idx >= self.data.len() {
            self.data.resize(idx + 1, 0);
        }
        self.data[idx] = val;
    }
}

struct Vm {
    mem: LazyMemory,
    pc: usize,
    timer: i64,
}

impl Vm {
    fn new() -> Self {
        Vm {
            mem: LazyMemory::new(),
            pc: 0,
            timer: 0,
        }
    }

    /// Validate and convert a word index coming from program-controlled
    /// data (as opposed to indices we manage ourselves, like fixed
    /// low-memory slots, which are always valid by construction).
    fn checked_index(&self, idx: i32, context: &'static str) -> Result<usize, OutOfBoundsError> {
        let idx = idx as i64;
        if idx < 0 || idx as u64 >= MEM_SIZE as u64 {
            Err(OutOfBoundsError {
                context,
                index: idx,
                len: MEM_SIZE,
            })
        } else {
            Ok(idx as usize)
        }
    }

    fn read(&self, idx: i32, context: &'static str) -> Result<i32, OutOfBoundsError> {
        let i = self.checked_index(idx, context)?;
        Ok(self.mem.get(i))
    }

    fn write(&mut self, idx: i32, value: i32, context: &'static str) -> Result<(), OutOfBoundsError> {
        let i = self.checked_index(idx, context)?;
        self.mem.set(i, value);
        Ok(())
    }

    /// Set the program counter to a program-controlled jump/interrupt
    /// target, rejecting anything outside the VM's address space.
    fn set_pc(&mut self, target: i32, context: &'static str) -> Result<(), OutOfBoundsError> {
        self.pc = self.checked_index(target, context)?;
        Ok(())
    }

    /// Load a binary image from raw bytes into memory as little-endian
    /// 32-bit words. Used for decompressed .xz files and regular files.
    #[allow(dead_code)]
    fn load_program_from_bytes(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let total_words = bytes.len().div_ceil(4);
        let mut truncated = 0usize;

        for (i, chunk) in bytes.chunks(4).enumerate() {
            if i >= MEM_SIZE {
                truncated = total_words - i;
                break;
            }
            let mut word_bytes = [0u8; 4];
            word_bytes[..chunk.len()].copy_from_slice(chunk);
            self.mem.set(i, i32::from_le_bytes(word_bytes));
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
        if self.pc >= MEM_SIZE {
            return Err(OutOfBoundsError {
                context: "instruction fetch (pc ran off the end of memory)",
                index: self.pc as i64,
                len: MEM_SIZE,
            });
        }
        let raw = self.mem.get(self.pc);
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

        self.mem.set(64, (sec & 0xFFFF_FFFF) as i32);
        self.mem.set(65, ((sec >> 32) & 0xFFFF_FFFF) as i32);
        self.mem.set(66, (nsec & 0xFFFF_FFFF) as i32);
        self.mem.set(67, ((nsec >> 32) & 0xFFFF_FFFF) as i32);
    }
}

/// Returns true if `idx` (a word index known to be in-bounds) falls
/// within the memory-mapped framebuffer region.
fn is_framebuffer_word(idx: usize) -> bool {
    idx >= FB_ADDR_WORDS && idx < FB_ADDR_WORDS + FB_WORD_COUNT
}

/// In --no-gui mode, print a one-time notice the first time the boot
/// image touches the framebuffer region, since there's no window to
/// actually display anything. `idx` must already be known to be a valid,
/// non-negative memory index (i.e. came back from a successful
/// `Vm::read`/`Vm::write`).
fn note_framebuffer_access_once(idx: usize, fb_notice_shown: &mut bool) {
    if *fb_notice_shown {
        return;
    }
    if is_framebuffer_word(idx) {
        eprintln!(
            "esi_vm: note: boot image is accessing the framebuffer, but --no-gui is set; \
             there is no window to display it (this message only prints once)"
        );
        *fb_notice_shown = true;
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

/// Drain terminal key events (used in --no-gui mode instead of SDL).
/// Since there's no SDL window to capture scancodes from, this reads
/// actual typed characters from the terminal (which must already be in
/// raw mode). Terminals generally don't report key-release events unless
/// the Kitty keyboard protocol is negotiated, so unlike the SDL path this
/// effectively only produces "key down" values -- a guest program that
/// depends on key-up events will behave differently in this mode.
#[cfg(not(target_arch = "wasm32"))]
fn drain_terminal_events(
    pending_keys: &mut VecDeque<i32>,
    quit_flag: &AtomicBool,
) -> io::Result<()> {
    use crossterm::event::{self, Event as CEvent, KeyCode, KeyEventKind, KeyModifiers};

    while event::poll(Duration::from_secs(0))? {
        if let CEvent::Key(key_event) = event::read()? {
            if key_event.kind == KeyEventKind::Release {
                continue;
            }
            if key_event.code == KeyCode::Char('c') && key_event.modifiers.contains(KeyModifiers::CONTROL) {
                quit_flag.store(true, Ordering::SeqCst);
                continue;
            }
            let value = match key_event.code {
                KeyCode::Char(c) => Some(c as i32),
                KeyCode::Enter => Some(b'\r' as i32),
                KeyCode::Backspace => Some(0x08),
                KeyCode::Tab => Some(b'\t' as i32),
                KeyCode::Esc => Some(0x1B),
                _ => None,
            };
            if let Some(v) = value {
                pending_keys.push_back(v);
            }
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn drain_terminal_events(
    _pending_keys: &mut VecDeque<i32>,
    _quit_flag: &AtomicBool,
) -> io::Result<()> {
    Ok(())
}

/// Pop one pending key event for the guest program.
/// Returns: positive = key-down (SDL scancode, or a typed character's
/// ASCII value in --no-gui mode), negative = key-up (SDL mode only), 0 =
/// none pending.
fn kb_poll(pending_keys: &mut VecDeque<i32>) -> i32 {
    pending_keys.pop_front().unwrap_or(0)
}

/// Copy the VM's framebuffer region into the SDL surface's pixel buffer,
/// via the surface's safe pixel-locking API rather than a raw memcpy.
/// Bounds are clamped against both the surface's actual pixel buffer and
/// the VM's memory, so a mismatched surface format can't cause an
/// out-of-range read from `mem`.
fn copy_framebuffer_to_surface(mem: &LazyMemory, surface: &mut WindowSurfaceRef) {
    surface.with_lock_mut(|pixels: &mut [u8]| {
        let copy_len = FB_SIZE.min(pixels.len());
        let word_count = copy_len / 4;
        for i in 0..word_count {
            let bytes = mem.get(FB_ADDR_WORDS + i).to_le_bytes();
            let offset = i * 4;
            pixels[offset..offset + 4].copy_from_slice(&bytes);
        }
    });
}

/// SDL window/event state, only created when running with a GUI. Keeping
/// the `Sdl` context alongside the subsystems it produced ensures nothing
/// it owns is torn down early.
struct GuiState {
    _sdl_context: sdl3::Sdl,
    event_pump: sdl3::EventPump,
    window: sdl3::video::Window,
}

/// Ensures the terminal is always returned to normal (cooked) mode when
/// this guard is dropped, however the program exits -- normal completion,
/// an early error return, or a panic.
struct RawModeGuard;

impl RawModeGuard {
    #[cfg(not(target_arch = "wasm32"))]
    fn enable() -> io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        Ok(RawModeGuard)
    }

    #[cfg(target_arch = "wasm32")]
    fn enable() -> io::Result<Self> {
        Ok(RawModeGuard)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("esi_vm: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // If no binary argument provided, show file picker
    let binary_path = if cli.binary.is_empty() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = rfd::FileDialog::new()
                .add_filter("bootimage", &["bootimage", "bootimage.xz"])
                .set_title("Select ESI boot image")
                .pick_file();
            match file {
                Some(path) => path,
                None => {
                    eprintln!("esi_vm: no boot image selected");
                    std::process::exit(0);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Err("esi_vm: no boot image specified (file picker not available in WASM)".into());
        }
    } else {
        Path::new(&cli.binary).to_path_buf()
    };

    let binary_path_str = binary_path.to_string_lossy().to_string();

    // Check for .xz extension and decompress if needed
    let data = if binary_path.extension().map_or(false, |ext| ext == "xz") {
        eprintln!("esi_vm: decompressing {}...", binary_path_str);
        let xz_data = std::fs::read(&binary_path)?;
        let mut decompressed = Vec::new();
        let mut reader = std::io::BufReader::new(xz_data.as_slice());
        lzma_rs::xz_decompress(&mut reader, &mut decompressed)
            .map_err(|e| format!("failed to decompress {}: {e}", binary_path_str))?;
        decompressed
    } else {
        std::fs::read(&binary_path)?
    };

    let mut vm = Vm::new();
    let truncated_words = vm
        .load_program_from_bytes(&data)
        .map_err(|e| format!("failed to load boot image '{}': {e}", binary_path_str))?;
    if truncated_words > 0 {
        eprintln!(
            "esi_vm: warning: boot image is larger than VM memory; {truncated_words} \
             trailing word(s) were not loaded"
        );
    } else {
        eprintln!("esi_vm: loaded {}", binary_path_str);
    }

    // Only touch SDL at all when running with a GUI. In --no-gui mode no
    // window is created, so there is nothing to render into.
    let mut gui: Option<GuiState> = if cli.no_gui {
        None
    } else {
        let sdl_context = sdl3::init().map_err(|e| format!("failed to init SDL3: {e}"))?;
        let video_subsystem = sdl_context
            .video()
            .map_err(|e| format!("failed to init video subsystem: {e}"))?;
        let event_pump = sdl_context
            .event_pump()
            .map_err(|e| format!("failed to get event pump: {e}"))?;
        let window = video_subsystem
            .window("ESI Virtual Machine", FB_WIDTH as u32, FB_HEIGHT as u32)
            .build()
            .map_err(|e| format!("failed to create window: {e}"))?;
        Some(GuiState {
            _sdl_context: sdl_context,
            event_pump,
            window,
        })
    };

    // In --no-gui mode, put the terminal in raw mode so keystrokes are
    // delivered immediately (no waiting for Enter, no local echo getting
    // in the way). The guard restores the terminal on any exit path.
    let _raw_mode_guard = if cli.no_gui {
        Some(RawModeGuard::enable().map_err(|e| format!("failed to enable terminal raw mode: {e}"))?)
    } else {
        None
    };

    let quit_flag = Arc::new(AtomicBool::new(false));
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Handles Ctrl+C when nothing else captures it first: in GUI mode,
        // this is the case where the terminal (not the SDL window) still
        // has focus; in --no-gui mode, raw mode normally intercepts
        // Ctrl+C as a plain key event before it ever becomes a signal, so
        // this mostly matters before raw mode is enabled.
        let quit_flag = Arc::clone(&quit_flag);
        ctrlc::set_handler(move || {
            quit_flag.store(true, Ordering::SeqCst);
        })
        .map_err(|e| format!("failed to install Ctrl+C handler: {e}"))?;
    }

    let mut pending_keys: VecDeque<i32> = VecDeque::new();
    let mut fb_notice_shown = false;

    let mut last_render = Instant::now() - DISPLAY_UPDATE_INTERVAL;
    let stdout = io::stdout();
    let mut stdout_lock = BufWriter::new(stdout.lock());
    let mut event_poll_counter: u32 = 0;

    // Translated from the original `do { ... } while (c);` loop: run the
    // body at least once, then stop when the last-fetched `c` is zero.
    loop {
        // Draining the event queue involves a lock/syscall even when
        // empty; doing it every single VM instruction (potentially
        // hundreds of millions per second) was a real cost. Every ~4096
        // instructions is still far more often than a human can react,
        // so window-close/Ctrl+C/keyboard input all stay responsive.
        event_poll_counter = event_poll_counter.wrapping_add(1);
        if event_poll_counter & EVENT_POLL_INTERVAL_INSTRUCTIONS == 0 {
            match &mut gui {
                Some(gui) => drain_sdl_events(&mut gui.event_pump, &mut pending_keys, &quit_flag),
                None => drain_terminal_events(&mut pending_keys, &quit_flag)?,
            }
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

            if gui.is_none() {
                // `a`/`b` are already validated non-negative in-bounds
                // word indices at this point (the reads/write above
                // succeeded), so casting to usize is safe here.
                note_framebuffer_access_once(a as usize, &mut fb_notice_shown);
                note_framebuffer_access_once(b as usize, &mut fb_notice_shown);
            }

            // mem[0] and mem[1] are fixed low-memory slots that are
            // always valid given MEM_SIZE, so these two are safe to
            // index directly.
            if vm.mem.get(0) != 0 {
                vm.timer += 1;
                if vm.timer > TIMER_INTERRUPT_THRESHOLD {
                    let now = Instant::now();
                    if now.duration_since(last_render) >= DISPLAY_UPDATE_INTERVAL {
                        if let Some(gui) = &mut gui {
                            if let Ok(mut surface) = gui.window.surface(&gui.event_pump) {
                                copy_framebuffer_to_surface(&vm.mem, &mut surface);
                                let _ = surface.update_window();
                            }
                        }
                        stdout_lock.flush()?;
                        last_render = now;
                    }
                    // Trigger interrupt
                    vm.mem.set(1, (vm.pc as i32).wrapping_mul(4));
                    let interrupt_vector = vm.mem.get(0) / 4;
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