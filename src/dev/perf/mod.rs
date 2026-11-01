//! Performance measurement over the pipeline `bytes -> vt::Parser -> grid::Screen`,
//! and over the render pipeline behind it. No Wayland, no GPU, no PTY, so it runs
//! anywhere.
//!
//! `--perf` is the regression gate: a handful of hot-path microbenchmarks (parse
//! ASCII, UTF-8 and escape-heavy chunks; grid scroll and print; a tab-bar frame; a
//! hyperlink lookup) timed against `stats/baseline.txt`. It writes `data/latest.txt` and exits nonzero
//! when a scenario is slower than the baseline by `REGRESSION_RATIO`. `--save`
//! accepts the current numbers as the new baseline. Built with `perf-alloc` it also
//! counts the allocations a fixed run of `COUNTED_OPS` operations makes and fails on
//! any increase: a count is exact and repeats run to run, so a ratio would only let a
//! scenario's first allocation through. `--save` needs `perf-alloc`, because a baseline
//! without counts would leave them ungated.
//!
//! The same run gates the grid's footprint against `stats/footprint.txt`, in its own
//! table because it is a different kind of number. A time is sampled and noisy, so it
//! gates at a ratio; a footprint is [`Screen::storage_bytes`] summed from container
//! capacities, so it is exact, identical on every machine, and gates at equality.
//! An allocation count cannot see that axis at all — a grid that holds twice the
//! memory while allocating as often reads as unchanged.
//!
//! `--perf-lab` measures whole-pipeline throughput over the generated streams in
//! [`crate::dev::stream`], fed at the live path's 64 KiB parser boundary, reporting
//! MB/s, allocations and memory, appending to `data/history.jsonl`. `--profile` also
//! prints each stream's run, width and action shape outside the timed loop. For
//! finding work rather than gating it.
//!
//! Its three memory columns answer different questions. `held` is
//! [`Screen::storage_bytes`], the grid's own containers: exact and machine-
//! independent. `rss` and `swap` come from [`sys::memory`](crate::dev::sys::memory) and
//! report where the process's pages physically are, which sees what `held` cannot:
//! allocator overhead, pages the kernel has reclaimed. But it is sampled and
//! machine-dependent. Read `rss` against `held`, never alone: it covers the whole
//! process, so it carries the corpus, the fonts and the binary alongside the grid, and
//! only the gap between the two says anything.
//!
//! `--cat-bench FILE` times `CAT_REPS` passes of decode -> parse -> grid over a real
//! file, fed in 64 KiB chunks as a PTY delivers them, reporting min and median wall
//! time and MB/s with rendering excluded.
//!
//! `--render-bench` times the render pipeline headlessly (`grid ->
//! build_display_list -> damage diff -> gpu::build_frame` vertices) per frame, so the
//! render side carries its own number and a fast parser cannot hide a slow renderer.
//! It sizes the CPU cost only; GPU execution and compositor present need the device.
//!
//! Every number here needs a `--release` build. A debug build measures nothing.
//!
//! # Where it lives
//!
//! ```text
//!   gate.rs       --perf: the speed and allocation scenarios, their baseline and table
//!   footprint.rs  --perf: the grid footprint gate that runs alongside them
//!   lab.rs        --perf-lab, --perf-stages and --cat-bench: throughput, for finding work
//!   render.rs     --render-bench: the render pipeline per frame
//! ```
//!
//! This file holds what they share: the bench grid, the frame the scenarios paint, the
//! corpora, and reading and writing the results files.

mod footprint;
pub mod gate;
mod instructions;
pub mod lab;
pub mod render;

use crate::color::Theme;
use crate::dev::stream;
use crate::error::{Error, Result};
use crate::grid::Screen;
use crate::platform::geom::Scale;
use crate::platform::scroll::Scrollbar;
use crate::term_render::{CellMetrics, CursorRender, FrameInputs};
use std::collections::HashMap;

/// The grid every perf harness runs against, matching the reference cat
/// benchmark's default (120 cols x 80 rows) so `--cat-bench` throughput compares
/// cell-for-cell, wrap width and all, not just in order of
/// magnitude. The gate and lab share it so all three measure the same terminal.
const BENCH_COLS: usize = 120;

const BENCH_ROWS: usize = 80;

/// The scrollbar every harness frame paints with: out of sight, so it contributes no
/// commands. A perf scenario measures the grid, not the chrome that floats over it, and
/// a bar mid-fade would make the display list depend on the wall clock.
static HIDDEN_SCROLLBAR: Scrollbar = Scrollbar::hidden();

/// The 8x16 cell the display-list scenarios lay out on, with no fonts loaded.
const METRICS: CellMetrics = CellMetrics {
    size: 16,
    w: 8,
    h: 16,
    baseline: 12,
    ascent: 12,
    descent: 4,
    lock_glyph: true,
};

/// The frame a scenario paints: `screen` at the top-left of `surface` under the default
/// cursor, with nothing selected or hovered and the scrollbar out of sight. A scenario
/// that needs more (a hovered link, a strip above the grid) sets those fields over it.
fn plain_frame<'a>(
    screen: &'a Screen,
    theme: &'a Theme,
    metrics: CellMetrics,
    surface: (i32, i32),
) -> FrameInputs<'a> {
    FrameInputs {
        screen,
        bell: false,
        theme,
        metrics,
        surface,
        origin: (0, 0),
        cursor: CursorRender::default(),
        selection: None,
        hover: None,
        preview: None,
        scale: Scale::ONE,
        scrollbar: &HIDDEN_SCROLLBAR,
    }
}

/// The named stream from [`stream::generate`], or an error naming it. An empty chunk in
/// its place would measure nothing, and pass.
fn corpus(name: &str, bytes: usize) -> Result<Vec<u8>> {
    stream::generate(name, bytes)
        .ok_or_else(|| Error::msg(format!("no stream named {name:?} in dev::stream")))
}

/// Percentage change of `current` over `base` (positive is a regression). Equal
/// values are exactly 0%, so two zeros do not read as a spurious -100%.
fn ratio_pct(current: u64, base: u64) -> f64 {
    if current == base {
        0.0
    } else {
        (current as f64 / base.max(1) as f64 - 1.0) * 100.0
    }
}

/// The text of the file at `path`, or `None` if it does not exist.
fn read_if_present(path: &str) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::msg(format!("read {path}: {e}"))),
    }
}

/// Parse `name<ws>number...` lines into name -> numbers, skipping blank and `#` lines.
/// Every line must hold exactly `N` numbers, so a short line cannot leave a column
/// ungated. `what` names the file in an error.
fn parse_rows<const N: usize>(text: &str, what: &str) -> Result<HashMap<String, [u64; N]>> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let name = fields.next().unwrap_or_default();
        let mut row = [0; N];
        let mut slots = row.iter_mut();
        for field in fields {
            let Some(slot) = slots.next() else {
                return Err(malformed_row(what, line, N));
            };
            *slot = field
                .parse::<u64>()
                .map_err(|_| Error::msg(format!("malformed {what} number: {field:?}")))?;
        }
        if slots.next().is_some() {
            return Err(malformed_row(what, line, N));
        }
        out.insert(name.to_string(), row);
    }
    Ok(out)
}

/// The error for a line that is not a name followed by exactly `numbers` numbers.
fn malformed_row(what: &str, line: &str, numbers: usize) -> Error {
    Error::msg(format!(
        "malformed {what} line {line:?}: want a name and {numbers} numbers"
    ))
}

fn ensure_parent(path: &str) -> Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::msg(format!("create {}: {e}", parent.display())))?;
        }
    }
    Ok(())
}

/// Font size (device px) for the render harness. Once the glyph cache is warm the
/// per-frame cost is dominated by the cell count, not the raster size, so the exact
/// value barely matters.
const RENDER_FONT_PX: u32 = 16;

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
