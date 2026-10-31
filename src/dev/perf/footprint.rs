//! `--perf`: the footprint gate. Each scenario fills the scrollback ring with a corpus and
//! weighs the grid's containers ([`Screen::storage_bytes`]), which is exact and identical
//! on every machine, so it gates against `stats/footprint.txt` at equality.

use crate::dev::perf::{
    corpus, ensure_parent, parse_rows, ratio_pct, read_if_present, BENCH_COLS, BENCH_ROWS,
};
use crate::error::{Error, Result};
use crate::grid::Screen;
use crate::vt::Parser;
use std::collections::HashMap;
use std::fmt::Write as _;

pub(super) const FOOTPRINT_PATH: &str = "stats/footprint.txt";

/// Bytes of stream generated per footprint scenario. One pass of this fills the
/// ring for the denser corpora and two or three do for the rest, so the fill loop
/// never reaches its bound on any corpus here.
const FOOTPRINT_CHUNK: usize = 1024 * 1024;

/// Passes over the generated stream before a scenario gives up trying to fill the
/// ring. It exists only so a future corpus that scrolls slowly (or not at all)
/// cannot hang the gate; the `rows` column shows when it bites.
const FOOTPRINT_PASSES: u32 = 16;

/// One measured footprint: the bytes a filled grid holds, and the history rows it
/// holds them for.
///
/// `rows` is reported but never baselined. It is the sanity column: every scenario
/// should land on a full ring, and two scenarios can only be compared to each other
/// when they did. A change that quietly stopped one of them filling would otherwise
/// read as a large, welcome improvement.
pub(super) struct Footprint {
    name: &'static str,
    bytes: u64,
    rows: usize,
}

/// Measure what a full grid costs for each payload shape, from plain ASCII (the
/// floor: cells and nothing else) through emoji (the combining side table's worst
/// case). Same corpora as the timing scenarios, so a change that trades bytes for
/// speed shows up in both tables of one run.
pub(super) fn measure_footprints() -> Result<Vec<Footprint>> {
    Ok(vec![
        footprint_scenario("mem_ascii", "ascii")?,
        footprint_scenario("mem_styled", "short_runs")?,
        footprint_scenario("mem_mixed", "unicode_mixed")?,
        footprint_scenario("mem_emoji", "emoji")?,
        resized_footprint_scenario("mem_resized", "ascii")?,
    ])
}

/// Fill a fresh grid's scrollback to the ring's cap with `corpus`. Untimed: every
/// footprint is a property of the resulting structure, not of the work that built
/// it, so nothing here needs warming, rounds, or a `black_box`.
fn filled_screen(name: &str) -> Result<Screen> {
    let chunk = corpus(name, FOOTPRINT_CHUNK)?;
    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    let mut passes = 0;
    while screen.scrollback_len() < crate::grid::DEFAULT_SCROLLBACK && passes < FOOTPRINT_PASSES {
        parser.advance_bytes(&mut screen, &chunk);
        passes += 1;
    }
    Ok(screen)
}

/// What a filled grid holds, straight off the fill.
fn footprint_scenario(name: &'static str, corpus: &str) -> Result<Footprint> {
    Ok(weigh(name, &filled_screen(corpus)?))
}

/// The same grid after the window has been widened and pulled back to where it
/// started: one drag of a resize handle, the most ordinary thing a user does to a
/// terminal, and an axis no other scenario can see.
///
/// The content is identical to `mem_ascii` at the end — same corpus, same geometry, same
/// rows — so everything held above it is capacity the widening left behind, and that is
/// permanent: rows reset in place and keep the buffer they have, and steady scrolling
/// recycles those same over-wide buffers. Measured at 20 columns of widening, one round
/// trip.
fn resized_footprint_scenario(name: &'static str, corpus: &str) -> Result<Footprint> {
    let mut screen = filled_screen(corpus)?;
    screen.resize(BENCH_COLS + 20, BENCH_ROWS);
    screen.resize(BENCH_COLS, BENCH_ROWS);
    Ok(weigh(name, &screen))
}

fn weigh(name: &'static str, screen: &Screen) -> Footprint {
    Footprint {
        name,
        bytes: u64::try_from(screen.storage_bytes()).unwrap_or(u64::MAX),
        rows: screen.scrollback_len(),
    }
}

/// Print the footprint table against its baseline and return what grew, saving the
/// new numbers when asked.
pub(super) fn gate_footprints(footprints: &[Footprint], save: bool) -> Result<Vec<&'static str>> {
    let Some(baseline) = load_footprints(FOOTPRINT_PATH)? else {
        write_footprints(FOOTPRINT_PATH, footprints)?;
        print_footprints(footprints, &HashMap::new());
        println!("\nNo footprint baseline existed; wrote {FOOTPRINT_PATH}. Commit it.");
        return Ok(Vec::new());
    };
    let regressed = print_footprints(footprints, &baseline);
    if save {
        write_footprints(FOOTPRINT_PATH, footprints)?;
        return Ok(Vec::new());
    }
    Ok(regressed)
}

/// Print a current-vs-baseline footprint table and return the names that grew.
///
/// Compared at equality, unlike the timing table: the number is summed from container
/// capacities rather than sampled, so it does not drift between runs or machines and a
/// tolerance would only hide small changes. Growth has to be explained or accepted with
/// `--save`.
fn print_footprints(
    footprints: &[Footprint],
    baseline: &HashMap<String, u64>,
) -> Vec<&'static str> {
    let mut sorted: Vec<&Footprint> = footprints.iter().collect();
    sorted.sort_by_key(|f| f.name);
    println!(
        "\n{:<16} {:>12} {:>12} {:>16}   {:>8} {:>7}",
        "footprint", "bytes", "base", "Δ", "MiB", "rows"
    );
    let mut regressed = Vec::new();
    for f in sorted {
        let (base_col, delta_col) = match baseline.get(f.name) {
            Some(&base) if base == f.bytes => (base.to_string(), String::new()),
            Some(&base) => {
                let label = if f.bytes > base { "GREW" } else { "shrank" };
                if f.bytes > base {
                    regressed.push(f.name);
                }
                (
                    base.to_string(),
                    format!("{:+.2}% {label}", ratio_pct(f.bytes, base)),
                )
            }
            None => ("(new)".to_string(), String::new()),
        };
        println!(
            "{:<16} {:>12} {:>12} {:>16}   {:>8.2} {:>7}",
            f.name,
            f.bytes,
            base_col,
            delta_col,
            f.bytes as f64 / (1024.0 * 1024.0),
            f.rows,
        );
    }
    regressed
}

/// Write footprints to `path`, sorted by name for a stable diff. The row count stays
/// out of the file: it is a sanity column, not something to gate on.
fn write_footprints(path: &str, footprints: &[Footprint]) -> Result<()> {
    ensure_parent(path)?;
    let mut sorted: Vec<&Footprint> = footprints.iter().collect();
    sorted.sort_by_key(|f| f.name);
    let mut text = String::from(
        "# bnkterm grid footprint: name, bytes held by a full 120x80 grid (lower is better)\n",
    );
    for f in sorted {
        let _ = writeln!(text, "{}\t{}", f.name, f.bytes);
    }
    std::fs::write(path, text).map_err(|e| Error::msg(format!("write {path}: {e}")))
}

/// Load a footprint baseline into a name -> bytes map, or `None` if it does not exist.
fn load_footprints(path: &str) -> Result<Option<HashMap<String, u64>>> {
    let Some(text) = read_if_present(path)? else {
        return Ok(None);
    };
    Ok(Some(
        parse_rows::<1>(&text, "footprint")?
            .into_iter()
            .map(|(name, [bytes])| (name, bytes))
            .collect(),
    ))
}
