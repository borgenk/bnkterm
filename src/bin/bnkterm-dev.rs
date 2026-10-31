use bnkterm::dev::{capture, gen_stream, gen_tables, perf};
use bnkterm::error::{Error, Result};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let result: Result<()> = if has("--perf-lab") {
        perf::lab::run_lab(&args)
    } else if has("--perf-stages") {
        perf::lab::run_stages(&args)
    } else if has("--perf") {
        perf::gate::run(&args)
    } else if has("--cat-bench") {
        perf::lab::cat_bench(&args)
    } else if has("--render-bench") {
        perf::render::render_bench(&args)
    } else if has("--gen-stream") {
        gen_stream::run(&args)
    } else if has("--capture") {
        capture::run(&args)
    } else if has("--gen-tables") {
        gen_tables::run(&args)
    } else {
        Err(Error::msg(
            "usage: --perf [--save] | --perf-lab | --perf-stages | --cat-bench FILE | \
             --render-bench | --gen-stream | --capture | --gen-tables",
        ))
    };
    if let Err(e) = result {
        eprintln!("bnkterm-dev: {e}");
        std::process::exit(1);
    }
}
