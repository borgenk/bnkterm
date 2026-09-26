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
