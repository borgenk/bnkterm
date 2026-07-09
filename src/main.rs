//! The bnkterm binary: argument dispatch and the process entry point. Everything
//! of substance lives in the library crate (see `lib.rs`); this is the thin
//! shell that turns flags into a call into it.
//!
//! `--gpu-probe` reports the GPU/dmabuf presentation path
//! without opening one, and `--demo` opens the window on a static styled grid
//! (no shell). Everything else launches the live terminal: a shell on a PTY.

use bnkterm::app;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.iter().any(|a| a == "--gpu-probe") {
        app::gpu_probe()
    } else if args.iter().any(|a| a == "--demo") {
        app::run_demo()
    } else {
        app::run()
    };
    if let Err(e) = result {
        eprintln!("bnkterm: {e}");
        std::process::exit(1);
    }
}
