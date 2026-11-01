use bnkterm::dev::{capture, gen_stream, gen_tables, perf};
use bnkterm::error::{Error, Result};

const USAGE: &str = "usage: --perf [--save] | --perf-lab | --perf-stages | --cat-bench FILE | \
                     --render-bench | --gen-stream | --capture | --gen-tables";

fn main() {
    // The mode comes first, and the tool it names reads the rest.
    let mut args = std::env::args().skip(1);
    let result: Result<()> = match args.next().as_deref() {
        Some("--perf") => perf::gate::run(args),
        Some("--perf-lab") => perf::lab::run_lab(args),
        Some("--perf-stages") => perf::lab::run_stages(args),
        Some("--cat-bench") => perf::lab::cat_bench(args),
        Some("--render-bench") => perf::render::render_bench(args),
        Some("--gen-stream") => gen_stream::run(args),
        Some("--capture") => capture::run(args),
        Some("--gen-tables") => gen_tables::run(args),
        _ => Err(Error::msg(USAGE)),
    };
    if let Err(e) = result {
        eprintln!("bnkterm-dev: {e}");
        std::process::exit(1);
    }
}
