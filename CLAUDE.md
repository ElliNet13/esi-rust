# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESI VM is an Eternal Computer Virtual Machine written in safe Rust. It executes Subleq bytecode with framebuffer support for graphics output. The project is a port of a C implementation with no unsafe blocks—all memory accesses are bounds-checked.

## Commands

```bash
# Build the project
cargo build

# Build release binary
cargo build --release

# Run tests
cargo test

# Build for WASM (requires SDL3 Emscripten build)
# Set up env first: export SDL3_DIR, PKG_CONFIG_PATH, RUSTFLAGS as shown in NOTETOSELF.txt
cargo build --target wasm32-unknown-emscripten --release
```

## Architecture

The VM implements the Subleq (SUBtract and branch if Less than or EQual) instruction set:

- **Memory**: ~1.6GB (3 << 27 32-bit words), lazy-allocated
- **Framebuffer**: 800x512 pixels at address 0x60000000 (word-mapped)
- **I/O**: Memory-mapped—negative addresses trigger special operations:
  - `a = -1`: Keyboard read (fills `b` with scancode)
  - `b = -1`: Console write (outputs low byte of `mem[a]`)
  - Normal: Subleq operation with conditional branch

**Timer interrupt**: Triggered via `mem[0]` interrupt vector when `mem[0] != 0` and timer exceeds threshold.

**GUI vs headless**: Use `--no-gui` flag to run without SDL—terminal input is used instead of SDL events.

## Key Implementation Details

- Single-file implementation in `src/main.rs`
- `LazyMemory` struct provides sparse memory with on-demand allocation
- All program-controlled memory accesses (operands, jump targets) are explicitly bounds-checked
- No unsafe code anywhere—the C code's pointer aliasing is replaced with integer arithmetic
- WASM builds skip SDL and terminal events entirely

## Development Notes

- Dependencies: `sdl3` (graphics), `clap` (CLI), `ctrlc`/`crossterm` (signals/terminal)
- CI builds Linux (static SDL3) and Windows (dynamic SDL3) releases
- SDL3 is built from source in CI—no Cargo crate dependency