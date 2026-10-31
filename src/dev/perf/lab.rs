//! The local lab: whole-pipeline throughput over the generated streams (`--perf-lab`),
//! the same streams split into parser, grid and full-pipeline stages (`--perf-stages`),
//! and decode, parse and grid over a real file (`--cat-bench`). For finding work, not
//! gating it.

use crate::dev::perf::{duration_ns, ensure_parent, BENCH_COLS, BENCH_ROWS};
use crate::dev::{alloc, profile, stream, sys};
use crate::error::{Error, Result};
use crate::gather::BUF_CAP;
use crate::grid::Screen;
use crate::vt::{Params, Parser, Perform};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

const HISTORY_PATH: &str = "data/history.jsonl";

/// Repetitions per lab stream, and the default/max stream size in mebibytes.
const LAB_REPS: u32 = 5;
const LAB_DEFAULT_MIB: usize = 8;
const LAB_MAX_MIB: usize = 512;

/// Timed reps per `--cat-bench` run; the min is the least-noisy estimate, the median
/// the typical.
const CAT_REPS: usize = 10;

/// Throughput over `FILE`: warm the parser, then time [`CAT_REPS`] passes of
/// decode -> parse -> grid (fed in [`BUF_CAP`] slices, as a live PTY delivers),
/// reporting min and median wall time and MB/s with rendering excluded. The file is
/// read once, outside the timed loop, so the number is parse and grid, not disk IO.
pub fn cat_bench(args: &[String]) -> Result<()> {
    let path = args
        .iter()
        .position(|a| a == "--cat-bench")
        .and_then(|i| args.get(i + 1))
        .ok_or_else(|| Error::msg("--cat-bench needs a FILE argument"))?;
    let bytes = std::fs::read(path).map_err(|e| Error::msg(format!("read {path}: {e}")))?;
    if bytes.is_empty() {
        return Err(Error::msg(format!("{path} is empty")));
    }

    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    feed_in_chunks(&mut parser, &mut screen, &bytes); // warm (untimed)

    let mut times = Vec::with_capacity(CAT_REPS);
    for _ in 0..CAT_REPS {
        let start = Instant::now();
        feed_in_chunks(&mut parser, &mut screen, &bytes);
        black_box(&screen);
        times.push(start.elapsed().as_nanos());
    }
    times.sort_unstable();
    let min_ns = times.first().copied().unwrap_or(0);
    let median_ns = times.get(times.len() / 2).copied().unwrap_or(0);
    let mb = bytes.len() as f64 / 1e6;
    let mbps = |ns: u128| if ns == 0 { 0.0 } else { mb / (ns as f64 / 1e9) };

    println!(
        "cat-bench {path}: {} bytes x {CAT_REPS} reps (decode -> parse -> grid; render excluded)",
        bytes.len()
    );
    println!(
        "  min    {:>9.1} ms   {:>9.1} MB/s",
        min_ns as f64 / 1e6,
        mbps(min_ns)
    );
    println!(
        "  median {:>9.1} ms   {:>9.1} MB/s",
        median_ns as f64 / 1e6,
        mbps(median_ns)
    );
    Ok(())
}

/// Feed `bytes` to the parser in [`BUF_CAP`] slices, exactly as the live PTY
/// drain does, so cross-chunk parser state stays on the measured path. The parser
/// carries UTF-8 and escape state across calls, so where a boundary falls never
/// changes the resulting grid, only the realism.
fn feed_in_chunks(parser: &mut Parser, screen: &mut Screen, bytes: &[u8]) {
    for chunk in bytes.chunks(BUF_CAP) {
        parser.advance_bytes(screen, chunk);
    }
}

/// Attribute parser-only, decoded-grid-only, and complete parser-to-grid cost
/// over the same characterized corpora. This is a local lab, not a gate. The
/// grid column is reported only when a stream contains print and execute actions:
/// replaying an invented subset of CSI/OSC semantics would be a mock rather than
/// a measurement.
pub fn run_stages(args: &[String]) -> Result<()> {
    let size_mib = parse_size(args)
        .unwrap_or(LAB_DEFAULT_MIB)
        .clamp(1, LAB_MAX_MIB);
    let target = size_mib * 1024 * 1024;
    let selected = select_streams(args);
    let requested_delivery = parse_delivery(args)?;

    println!(
        "\n{:<18} {:<14} {:>12} {:>12} {:>12}",
        "stream", "delivery", "parser MB/s", "grid MB/s", "full MB/s"
    );
    for name in selected {
        let input = load_lab_input(name, target, requested_delivery)?;
        let parser_ns = measure_parser_stage(&input);
        let (grid_ns, non_text_actions) = measure_grid_stage(&input);
        let full_ns = measure_full_stage(&input);
        let grid = if non_text_actions == 0 {
            format!("{:.1}", rate_mbps(input.bytes.len(), grid_ns))
        } else {
            format!("- ({non_text_actions} seq)")
        };
        println!(
            "{:<18} {:<14} {:>12.1} {:>12} {:>12.1}",
            name,
            input.delivery,
            rate_mbps(input.bytes.len(), parser_ns),
            grid,
            rate_mbps(input.bytes.len(), full_ns),
        );
    }
    Ok(())
}

#[derive(Default)]
struct ParseCounter {
    scalars: usize,
    actions: usize,
}

impl Perform for ParseCounter {
    fn print(&mut self, _c: char) {
        self.scalars = self.scalars.saturating_add(1);
    }

    fn print_run(&mut self, chars: &[char]) {
        self.scalars = self.scalars.saturating_add(chars.len());
    }

    fn print_ascii(&mut self, bytes: &[u8]) {
        self.scalars = self.scalars.saturating_add(bytes.len());
    }

    fn execute(&mut self, _byte: u8) {
        self.actions = self.actions.saturating_add(1);
    }

    fn csi_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _private: u8, _action: u8) {
        self.actions = self.actions.saturating_add(1);
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _byte: u8) {
        self.actions = self.actions.saturating_add(1);
    }

    fn dcs_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _action: u8, _data: &[u8]) {
        self.actions = self.actions.saturating_add(1);
    }

    fn osc_dispatch(&mut self, _data: &[u8], _bel_terminated: bool) {
        self.actions = self.actions.saturating_add(1);
    }
}

#[derive(Clone, Copy)]
enum GridAction {
    Print(char),
    Ascii { start: usize, len: usize },
    Run { start: usize, len: usize },
    Execute(u8),
}

struct GridActions {
    actions: Vec<GridAction>,
    // Payloads live in contiguous arenas: actions retain the parser's callback
    // boundaries without allocating a separate Vec for every captured run.
    ascii: Vec<u8>,
    runs: Vec<char>,
    non_text_actions: usize,
}

impl GridActions {
    fn with_capacity(capacity: usize) -> Self {
        GridActions {
            actions: Vec::new(),
            ascii: Vec::with_capacity(capacity),
            runs: Vec::new(),
            non_text_actions: 0,
        }
    }

    fn observe_sequence(&mut self) {
        self.non_text_actions = self.non_text_actions.saturating_add(1);
    }
}

impl Perform for GridActions {
    fn print(&mut self, c: char) {
        self.actions.push(GridAction::Print(c));
    }

    fn print_ascii(&mut self, bytes: &[u8]) {
        let start = self.ascii.len();
        self.ascii.extend_from_slice(bytes);
        self.actions.push(GridAction::Ascii {
            start,
            len: bytes.len(),
        });
    }

    fn print_run(&mut self, chars: &[char]) {
        let start = self.runs.len();
        self.runs.extend_from_slice(chars);
        self.actions.push(GridAction::Run {
            start,
            len: chars.len(),
        });
    }

    fn execute(&mut self, byte: u8) {
        self.actions.push(GridAction::Execute(byte));
    }

    fn csi_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _private: u8, _action: u8) {
        self.observe_sequence();
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _byte: u8) {
        self.observe_sequence();
    }

    fn dcs_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _action: u8, _data: &[u8]) {
        self.observe_sequence();
    }

    fn osc_dispatch(&mut self, _data: &[u8], _bel_terminated: bool) {
        self.observe_sequence();
    }
}

fn measure_parser_stage(input: &LabInput) -> u64 {
    let mut parser = Parser::new();
    let mut counter = ParseCounter::default();
    feed_lab_input(&mut parser, &mut counter, input);
    let mut best = u64::MAX;
    for _ in 0..LAB_REPS {
        let start = Instant::now();
        feed_lab_input(&mut parser, &mut counter, input);
        black_box((counter.scalars, counter.actions));
        best = best.min(duration_ns(start.elapsed()));
    }
    best
}

fn measure_grid_stage(input: &LabInput) -> (u64, usize) {
    let mut parser = Parser::new();
    let mut decoded = GridActions::with_capacity(input.bytes.len());
    feed_lab_input(&mut parser, &mut decoded, input);
    if decoded.non_text_actions > 0 {
        return (0, decoded.non_text_actions);
    }
    let mut screen = Screen::new(input.cols, input.rows);
    replay_grid_actions(&mut screen, &decoded);
    let mut best = u64::MAX;
    for _ in 0..LAB_REPS {
        let start = Instant::now();
        replay_grid_actions(&mut screen, &decoded);
        black_box(&screen);
        best = best.min(duration_ns(start.elapsed()));
    }
    (best, 0)
}

fn measure_full_stage(input: &LabInput) -> u64 {
    let mut parser = Parser::new();
    let mut screen = Screen::new(input.cols, input.rows);
    feed_lab_input(&mut parser, &mut screen, input);
    let mut best = u64::MAX;
    for _ in 0..LAB_REPS {
        let start = Instant::now();
        feed_lab_input(&mut parser, &mut screen, input);
        black_box(&screen);
        best = best.min(duration_ns(start.elapsed()));
    }
    best
}

fn replay_grid_actions<P: Perform>(performer: &mut P, decoded: &GridActions) {
    for action in &decoded.actions {
        match *action {
            GridAction::Print(c) => performer.print(c),
            GridAction::Ascii { start, len } => {
                let Some(end) = start.checked_add(len) else {
                    continue;
                };
                if let Some(bytes) = decoded.ascii.get(start..end) {
                    performer.print_ascii(bytes);
                }
            }
            GridAction::Run { start, len } => {
                let Some(end) = start.checked_add(len) else {
                    continue;
                };
                if let Some(chars) = decoded.runs.get(start..end) {
                    performer.print_run(chars);
                }
            }
            GridAction::Execute(byte) => performer.execute(byte),
        }
    }
}

fn rate_mbps(bytes: usize, ns: u64) -> f64 {
    if ns == 0 {
        return 0.0;
    }
    let bytes = u32::try_from(bytes)
        .map(f64::from)
        .unwrap_or_else(|_| f64::from(u32::MAX));
    bytes / 1_000_000.0 / std::time::Duration::from_nanos(ns).as_secs_f64()
}

/// Bytes as mebibytes, for the lab's memory columns. Binary units, matching the
/// kernel's own `kB`-that-means-KiB in `smaps_rollup` and the footprint gate's MiB.
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Run the lab: generate each selected stream, drive the whole pipeline over it,
/// print a table, and append the numbers to `data/history.jsonl` keyed by commit.
pub fn run_lab(args: &[String]) -> Result<()> {
    let size_mib = parse_size(args)
        .unwrap_or(LAB_DEFAULT_MIB)
        .clamp(1, LAB_MAX_MIB);
    let target = size_mib * 1024 * 1024;
    let selected = select_streams(args);
    let show_profiles = args.iter().any(|a| a == "--profile");
    let requested_delivery = parse_delivery(args)?;
    eprintln!(
        "bnkterm perf-lab: {size_mib} MiB/stream, {} streams{}",
        selected.len(),
        if alloc::ENABLED {
            ""
        } else {
            " (allocs unmeasured; build --features perf-alloc)"
        }
    );

    let commit = git_head();
    let ts = unix_seconds();
    let mut history = String::new();
    println!(
        "\n{:<18} {:<14} {:>10} {:>10} {:>12} {:>9} {:>9} {:>9}",
        "stream", "delivery", "min ms", "MB/s", "allocs/pass", "held MiB", "rss MiB", "swap MiB"
    );

    for &name in &selected {
        let input = load_lab_input(name, target, requested_delivery)?;
        if show_profiles {
            let profile = profile::StreamProfile::analyze(&input.bytes, &input.chunks);
            println!("\nprofile {name}:\n{profile}");
        }
        let mut screen = Screen::new(input.cols, input.rows);
        let mut parser = Parser::new();
        feed_lab_input(&mut parser, &mut screen, &input); // warm

        alloc::reset();
        feed_lab_input(&mut parser, &mut screen, &input);
        let allocs = alloc::snapshot().allocs;

        let mut best = u64::MAX;
        for _ in 0..LAB_REPS {
            let start = Instant::now();
            feed_lab_input(&mut parser, &mut screen, &input);
            black_box(&screen);
            best = best.min(start.elapsed().as_nanos() as u64);
        }
        let mb = input.bytes.len() as f64 / 1e6;
        let mbps = if best == 0 {
            0.0
        } else {
            mb / (best as f64 / 1e9)
        };
        // Sampled here, with the grid at its steady state and still alive: the
        // scrollback is full, so this is what a session that has been running costs,
        // not what starting one does.
        let held = screen.storage_bytes() as u64;
        let mem = sys::memory();

        println!(
            "{:<18} {:<14} {:>10.1} {:>10.1} {:>12} {:>9.1} {:>9.1} {:>9.1}",
            name,
            input.delivery,
            best as f64 / 1e6,
            mbps,
            if alloc::ENABLED {
                allocs.to_string()
            } else {
                "-".to_string()
            },
            mib(held),
            mib(mem.rss),
            mib(mem.swap)
        );
        let _ = writeln!(
            history,
            "{{\"commit\":\"{commit}\",\"ts\":{ts},\"stream\":\"{name}\",\"bytes\":{},\
             \"delivery\":\"{}\",\"ns\":{best},\"allocs\":{allocs},\"mbps\":{mbps:.1},\
             \"held\":{held},\"rss\":{},\"swap\":{}}}",
            input.bytes.len(),
            input.delivery,
            mem.rss,
            mem.swap
        );
    }

    append(HISTORY_PATH, &history)?;
    eprintln!(
        "appended {} lab result(s) to {HISTORY_PATH}",
        selected.len()
    );
    Ok(())
}

struct LabInput {
    bytes: Vec<u8>,
    chunks: Vec<usize>,
    cols: usize,
    rows: usize,
    delivery: &'static str,
}

fn load_lab_input(
    name: &str,
    target: usize,
    requested_delivery: profile::Delivery,
) -> Result<LabInput> {
    if let Some(bytes) = stream::generate(name, target) {
        let chunks = profile::fixed_chunk_lengths(bytes.len());
        return Ok(LabInput {
            bytes,
            chunks,
            cols: BENCH_COLS,
            rows: BENCH_ROWS,
            delivery: "normalized-64k",
        });
    }
    let recording = profile::load_recording(name, target, requested_delivery)?
        .ok_or_else(|| Error::msg(format!("unknown stream {name}")))?;
    Ok(LabInput {
        bytes: recording.bytes,
        chunks: recording.chunks,
        cols: recording.cols,
        rows: recording.rows,
        delivery: recording.delivery,
    })
}

fn parse_delivery(args: &[String]) -> Result<profile::Delivery> {
    let Some(index) = args.iter().position(|arg| arg == "--delivery") else {
        return Ok(profile::Delivery::Recorded);
    };
    match args.get(index + 1).map(String::as_str) {
        Some("recorded") => Ok(profile::Delivery::Recorded),
        Some("normalized") | Some("64k") => Ok(profile::Delivery::Normalized),
        _ => Err(Error::msg(
            "--delivery needs `recorded`, `normalized`, or `64k`",
        )),
    }
}

/// Feed `input` to `performer` at its recorded parser-call boundaries.
fn feed_lab_input<P: Perform>(parser: &mut Parser, performer: &mut P, input: &LabInput) {
    profile::for_each_chunk(&input.bytes, &input.chunks, |chunk| {
        parser.advance_bytes(performer, chunk);
    });
}

/// The `--size N` argument (MiB), if present and parseable.
fn parse_size(args: &[String]) -> Option<usize> {
    let i = args.iter().position(|a| a == "--size")?;
    args.get(i + 1)?.parse().ok()
}

/// The stream names to run: any explicitly named on the command line, else all.
fn select_streams(args: &[String]) -> Vec<&'static str> {
    let all: Vec<&'static str> = stream::NAMES
        .iter()
        .copied()
        .chain(profile::recording_names())
        .collect();
    let named: Vec<&'static str> = all
        .iter()
        .copied()
        .filter(|n| args.iter().any(|a| a == n))
        .collect();
    if named.is_empty() {
        all
    } else {
        named
    }
}

/// The short commit hash, or "unknown" (read-only; the lab only records it).
fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn append(path: &str, text: &str) -> Result<()> {
    ensure_parent(path)?;
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| Error::msg(format!("open {path}: {e}")))?;
    file.write_all(text.as_bytes())
        .map_err(|e| Error::msg(format!("append {path}: {e}")))
}

#[cfg(test)]
mod tests {
    use crate::dev::perf::lab::*;

    /// A scratch file that removes itself on drop, so a failing assertion never
    /// leaves litter in the temp dir.
    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn write(name: &str, bytes: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("bnkterm_catbench_{}_{name}", std::process::id()));
            std::fs::write(&path, bytes).unwrap();
            TempFile(path)
        }

        fn arg(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn cat_bench_streams_a_multi_chunk_file() {
        // Larger than one BUF_CAP so the chunk loop iterates and the parser
        // resumes across boundaries; mixed ASCII / UTF-8 / escape so every arm of
        // the parser is exercised. Success means all reps ran without error.
        let mut bytes = Vec::new();
        while bytes.len() < BUF_CAP * 2 + 123 {
            bytes.extend_from_slice("héllo \x1b[31mworld\x1b[0m 日本語\n".as_bytes());
        }
        let f = TempFile::write("multi", &bytes);
        assert!(cat_bench(&["--cat-bench".into(), f.arg()]).is_ok());
    }

    #[test]
    fn cat_bench_rejects_missing_arg_empty_and_absent() {
        // No path after the flag.
        assert!(cat_bench(&["--cat-bench".into()]).is_err());
        // A path that does not exist.
        assert!(cat_bench(&["--cat-bench".into(), "/no/such/bnkterm/stream".into()]).is_err());
        // An empty file has nothing to stream.
        let empty = TempFile::write("empty", b"");
        assert!(cat_bench(&["--cat-bench".into(), empty.arg()]).is_err());
    }

    #[test]
    fn grid_stage_replay_preserves_batch_callbacks() {
        let mut captured = GridActions::with_capacity(8);
        captured.print('x');
        captured.print_ascii(b"abc");
        captured.print_run(&['\u{754c}', '\u{03bb}']);
        captured.execute(b'\n');

        let mut replayed = GridActions::with_capacity(8);
        replay_grid_actions(&mut replayed, &captured);

        assert!(matches!(
            replayed.actions.as_slice(),
            [
                GridAction::Print('x'),
                GridAction::Ascii { start: 0, len: 3 },
                GridAction::Run { start: 0, len: 2 },
                GridAction::Execute(b'\n')
            ]
        ));
        assert_eq!(replayed.ascii, b"abc");
        assert_eq!(replayed.runs, ['\u{754c}', '\u{03bb}']);
    }
}
