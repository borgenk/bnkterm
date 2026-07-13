//! The bnkterm binary: argument dispatch and the process entry point. Everything
//! of substance lives in the library crate (see `lib.rs`); this is the thin
//! shell that turns flags into a call into it.
//!
//! `--gpu-probe` reports the GPU/dmabuf presentation path
//! without opening one, and `--demo` opens the window on a static styled grid
//! (no shell). Everything else launches the live terminal: a shell on a PTY.
//!
//! The live terminal is silent on stderr but for errors; `-v`/`--verbose` adds the
//! bring-up lines and `--stats` a per-frame timing line (see [`app::Verbosity`]).

use bnkterm::app::{self, Verbosity};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbosity = Verbosity::from_args(&args);
    let result = if args.iter().any(|a| a == "--gpu-probe") {
        app::gpu_probe()
    } else if args.iter().any(|a| a == "--demo") {
        app::run_demo(verbosity)
    } else {
        app::run(verbosity)
    };
    if let Err(e) = result {
        eprintln!("bnkterm: {e}");
        std::process::exit(1);
    }
}
