use bnkterm::app;
use bnkterm::cli::{Invocation, Mode, USAGE};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stats_env = std::env::var_os("BNKTERM_STATS").is_some();
    let invocation = match Invocation::parse(&args, stats_env) {
        Ok(invocation) => invocation,
        Err(arg) => {
            eprintln!("bnkterm: unknown argument {arg:?}; {USAGE}");
            std::process::exit(2);
        }
    };
    let result = match invocation.mode {
        Mode::GpuProbe => app::gpu_probe(),
        Mode::Demo => app::run_demo(invocation.verbosity),
        Mode::Run => app::run(invocation.verbosity),
    };
    if let Err(e) = result {
        eprintln!("bnkterm: {e}");
        std::process::exit(1);
    }
}
