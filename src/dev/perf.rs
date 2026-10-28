//! Performance measurement over the pipeline `bytes -> vt::Parser -> grid::Screen`,
//! and over the render pipeline behind it. No Wayland, no GPU, no PTY, so it runs
//! anywhere.
//!
//! `--perf` is the regression gate: a handful of hot-path microbenchmarks (parse
//! ASCII, UTF-8 and escape-heavy chunks; grid scroll and print; a tab-bar frame; a
//! hyperlink lookup) timed against `stats/baseline.txt`. It writes `data/latest.txt` and exits nonzero
//! when a scenario is slower than the baseline by [`REGRESSION_RATIO`]. `--save`
//! accepts the current numbers as the new baseline. Built with `perf-alloc` it also
//! counts the allocations a fixed run of [`COUNTED_OPS`] operations makes and fails on
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
//! independent. `rss` and `swap` come from [`sys::memory`] and report where the
//! process's pages physically are, which sees what `held` cannot — allocator overhead,
//! pages the kernel has reclaimed — but is sampled and machine-dependent. Read `rss`
//! against `held`, never alone: it covers the whole process, so it carries the corpus,
//! the fonts and the binary alongside the grid, and only the gap between the two says
//! anything.
//!
//! `--cat-bench FILE` times [`CAT_REPS`] passes of decode -> parse -> grid over a real
//! file, fed in 64 KiB chunks as a PTY delivers them, reporting min and median wall
//! time and MB/s with rendering excluded.
//!
//! `--render-bench` times the render pipeline headlessly (`grid ->
//! build_display_list -> damage diff -> gpu::build_frame` vertices) per frame, so the
//! render side carries its own number and a fast parser cannot hide a slow renderer.
//! It sizes the CPU cost only; GPU execution and compositor present need the device.
//!
//! Every number here needs a `--release` build. A debug build measures nothing.

use crate::dev::{alloc, profile, stream, sys};
use crate::error::{Error, Result};
use crate::gather::BUF_CAP;
use crate::grid::{LinkId, LinkProbe, Screen};
use crate::platform::geom::Scale;
use crate::platform::scroll::Scrollbar;
use crate::platform::verdict::{Refusal, Verdict};
use crate::vt::{Params, Parser, Perform};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

const BASELINE_PATH: &str = "stats/baseline.txt";
const LATEST_PATH: &str = "data/latest.txt";
const HISTORY_PATH: &str = "data/history.jsonl";
const FOOTPRINT_PATH: &str = "stats/footprint.txt";

/// A scenario counts as a regression when it is at least this much slower than the
/// baseline. Wide enough to absorb timing noise, tight enough to catch the
/// order-of-magnitude regressions this exists for.
const REGRESSION_RATIO: f64 = 1.10;

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

/// Bytes per gate parse chunk (one "op").
const CHUNK: usize = 8 * 1024;
/// Timing rounds per gate scenario; the minimum of these is the least-noisy
/// estimate. Inner iterations per round are chosen per scenario.
///
/// Twenty rather than five because the noise is per-process, not per-round: whole runs
/// land slow together, and five rounds regularly put the minimum 10-17% above the floor
/// with no code change. More rounds give the minimum more chances at an uninterrupted
/// window, and twenty still runs the whole gate in about a second.
const GATE_ROUNDS: u32 = 20;

/// Repetitions per lab stream, and the default/max stream size in mebibytes.
const LAB_REPS: u32 = 5;
const LAB_DEFAULT_MIB: usize = 8;
const LAB_MAX_MIB: usize = 512;

/// Timed reps per `--cat-bench` run; the min is the least-noisy estimate, the median
/// the typical.
const CAT_REPS: usize = 10;

/// One measured gate scenario: its best per-op time and, when measured, the
/// allocations [`COUNTED_OPS`] ops make (0 with the counting allocator absent).
struct Sample {
    name: &'static str,
    ns: u64,
    allocs: u64,
}

/// A baseline entry: the recorded time per op and allocation count.
struct Base {
    ns: u64,
    allocs: u64,
}

// ---------------------------------------------------------------------------
// --perf: the speed and allocation gate.
// ---------------------------------------------------------------------------

/// Run the gate: measure speed and footprint, write `latest`, then compare each to
/// (or establish) its baseline. Returns an error when anything regressed and
/// `--save` was not given, so the exit code signals it to a script or CI.
pub fn run(args: &[String]) -> Result<()> {
    let save = args.iter().any(|a| a == "--save");
    eprintln!("bnkterm perf: measuring (build --release for meaningful numbers)...");
    if !alloc::ENABLED {
        eprintln!("  allocations unmeasured; build --features perf-alloc (or `make perf`)");
    }
    let samples = measure()?;
    write_samples(LATEST_PATH, &samples)?;
    let mut regressed = gate_samples(&samples, save)?;
    regressed.extend(gate_footprints(&measure_footprints()?, save)?);

    if save {
        println!("\nUpdated baselines at {BASELINE_PATH} and {FOOTPRINT_PATH}.");
        return Ok(());
    }
    if regressed.is_empty() {
        let exact = if alloc::ENABLED {
            "allocations and footprint"
        } else {
            "footprint"
        };
        println!(
            "\nNo regressions (speed within {:.0}%, {exact} exact). Latest written to {LATEST_PATH}.",
            (REGRESSION_RATIO - 1.0) * 100.0
        );
        Ok(())
    } else {
        Err(Error::msg(format!(
            "perf regression in: {}. Re-run with --save to accept, or fix the regression.",
            regressed.join(", ")
        )))
    }
}

/// Print the timing table against its baseline and return what regressed, saving
/// the new numbers when asked. A first run has nothing to compare against, so it
/// establishes the baseline and reports no regression.
fn gate_samples(samples: &[Sample], save: bool) -> Result<Vec<&'static str>> {
    let Some(baseline) = load_baseline(BASELINE_PATH)? else {
        save_baseline(samples)?;
        print_table(samples, &HashMap::new(), alloc::ENABLED);
        println!("\nNo baseline existed; wrote {BASELINE_PATH}. Commit it to compare future runs.");
        return Ok(Vec::new());
    };
    let regressed = print_table(samples, &baseline, alloc::ENABLED);
    if save {
        save_baseline(samples)?;
        return Ok(Vec::new());
    }
    Ok(regressed)
}

/// Write `samples` as the new baseline. Refused unless this run counted allocations:
/// the baseline would have no counts to compare against, and every later run would
/// leave allocations ungated.
fn save_baseline(samples: &[Sample]) -> Result<()> {
    if !alloc::ENABLED {
        return Err(Error::msg(format!(
            "refusing to write {BASELINE_PATH} without allocation counts; \
             build with --features perf-alloc (`make perf-save` does)"
        )));
    }
    write_samples(BASELINE_PATH, samples)
}

/// The named stream from [`stream::generate`], or an error naming it. An empty chunk in
/// its place would measure nothing, and pass.
fn corpus(name: &str, bytes: usize) -> Result<Vec<u8>> {
    stream::generate(name, bytes)
        .ok_or_else(|| Error::msg(format!("no stream named {name:?} in dev::stream")))
}

/// Measure every gate scenario. Each drives a persistent, pre-warmed grid so
/// the numbers are steady state, not first-touch: the parse and print scenarios
/// then allocate nothing, which the allocation column records and gates.
fn measure() -> Result<Vec<Sample>> {
    let ascii = corpus("ascii", CHUNK)?;
    let unicode = corpus("unicode", CHUNK)?;
    let escape = corpus("escape_heavy", CHUNK)?;

    // A screenful of ASCII preceded by a home, so it overwrites in place without
    // scrolling: isolates the cell-write cost.
    let mut print_chunk = b"\x1b[H".to_vec();
    print_chunk.resize(3 + BENCH_COLS * BENCH_ROWS, b'x');
    // Short lines that force one scroll each: isolates the scroll cost.
    let scroll_chunk = b"line\n".repeat(512);
    // The same scroll on rows that are actually full. `scroll_chunk` writes four columns
    // of a hundred and twenty, so its rows are 3.3% occupied and any cost that scales
    // with how much of a row holds text is nearly invisible to it — a retiring row that
    // must be copied is exactly such a cost. Full lines also pay a screenful of cell
    // writes per scroll, so this is a mixture rather than a pure scroll probe;
    // `grid_print` beside it separates the write cost back out when the two disagree.
    let scroll_dense_chunk = repeated_lines("x", BENCH_COLS, 128);
    // Combining-mark and emoji scroll once the ring is full: the grid's combining side
    // table is the one warmed text path that used to allocate per marked cell. The lines
    // are fixed-shape, so after the warm-up cycles the ring each row's flattened table
    // settles to one capacity and a further identical line reallocates nothing, which is
    // what holds these at zero. The varied `combining` and `emoji` streams exercise the
    // capacity high-water instead.
    let combining_chunk = repeated_lines("a\u{0301}\u{0302}", 40, 24);
    let emoji_chunk = repeated_lines("🇳🇴", 40, 24);
    // A shell marking a prompt (`OSC 133;A`) on every line. Warmed past the number of marks
    // the grid remembers, so every mark here evicts old ones.
    let prompt_chunk = b"\x1b]133;A\x07\r\n".repeat(CHUNK / 10);
    // A Zalgo row: each cell's base followed by more marks than a cell keeps, overwritten
    // in place so the row stays at the cap, about 3,840 marks.
    let mut zalgo_chunk = b"\x1b[H".to_vec();
    let zalgo_cell: String = std::iter::once('a')
        .chain((0..40).filter_map(|i| char::from_u32(0x0300 + i)))
        .collect();
    zalgo_chunk.extend_from_slice(zalgo_cell.repeat(BENCH_COLS).as_bytes());
    // Newlines on a coloured background that walk down from the home row and never reach
    // the bottom margin, so none of them scrolls.
    let mut newline_bg_unit = b"\x1b[44m\x1b[H".to_vec();
    newline_bg_unit.resize(newline_bg_unit.len() + BENCH_ROWS - 1, b'\n');
    let newline_bg_chunk = newline_bg_unit.repeat(CHUNK / newline_bg_unit.len());

    let mut samples = vec![
        stream_scenario("parse_ascii", &ascii, 128),
        stream_scenario("parse_utf8", &unicode, 128),
        stream_scenario("parse_escape", &escape, 128),
        stream_scenario("grid_print", &print_chunk, 512),
        stream_scenario("grid_scroll", &scroll_chunk, 256),
        stream_scenario("grid_scroll_dense", &scroll_dense_chunk, 128),
        stream_scenario("grid_combining", &combining_chunk, 128),
        stream_scenario("grid_emoji", &emoji_chunk, 128),
        stream_scenario("grid_prompts", &prompt_chunk, 128),
        stream_scenario("grid_zalgo", &zalgo_chunk, 8),
        stream_scenario("grid_newline_bg", &newline_bg_chunk, 128),
        frame_tabbar_scenario()?,
        frame_damage_scenario(),
        frame_damage_scroll_scenario(),
        grid_reflow_scenario()?,
        link_preview_scenario()?,
    ];
    for fixture in link_fixtures() {
        samples.push(link_scenario(&fixture)?);
    }
    Ok(samples)
}

/// The display row each hyperlink scenario prints its line on.
const LINK_ROW: usize = 40;

/// A signed download URL, long enough to wrap across three rows of [`BENCH_COLS`].
const LINK_LONG_URL: &str = "https://example.com/releases/download/v0.3.0/\
    bnkterm-v0.3.0-x86_64-unknown-linux-gnu.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256\
    &X-Amz-Credential=EXAMPLE%2F20261010%2Fus-east-1%2Fs3%2Faws4_request\
    &X-Amz-Date=20261010T120000Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host\
    &X-Amz-Signature=5f2b8c0e9a7d4e31b6c8f0a2d4e6b8c0e2f4a6b8d0c2e4f6a8b0c2d4e6f8a0b2";

/// One hyperlink lookup: the line printed at [`LINK_ROW`], the display cell the pointer
/// rests on, and what [`Screen::link_at`] must find there.
struct LinkFixture {
    name: &'static str,
    line: String,
    pointer: (usize, usize),
    url: &'static str,
    /// Whether the pointer's cell is inside an OSC 8 anchor rather than plain text.
    anchored: bool,
    verdict: Verdict,
    /// Display rows the found link spans.
    rows: usize,
    /// Timed lookups per round, sized so each round times about a millisecond.
    iters: u64,
}

/// One fixture per path through [`Screen::link_at`]: a bare URL the text scan finds, an
/// OSC 8 anchor read off its cells, a bare URL soft-wrapped across three rows, and a
/// refused anchor.
fn link_fixtures() -> [LinkFixture; 4] {
    const URL: &str = "https://example.com/docs/terminal/hyperlinks.html";
    const EDITOR: &str = "vscode://file/home/user/src/bnkterm/src/grid/view.rs:533:12";
    [
        LinkFixture {
            name: "link_bare",
            line: format!("  docs: see {URL} for the details"),
            pointer: (LINK_ROW, 30),
            url: URL,
            anchored: false,
            verdict: Verdict::Open,
            rows: 1,
            iters: 2048,
        },
        LinkFixture {
            name: "link_anchor",
            // Labelled with its own target, so the text under the anchor is a URL too.
            line: format!("  docs: see \x1b]8;;{URL}\x1b\\{URL}\x1b]8;;\x1b\\ for the details"),
            pointer: (LINK_ROW, 30),
            url: URL,
            anchored: true,
            verdict: Verdict::Open,
            rows: 1,
            iters: 8192,
        },
        LinkFixture {
            name: "link_wrapped",
            line: format!("  download: {LINK_LONG_URL}"),
            pointer: (LINK_ROW + 1, 60),
            url: LINK_LONG_URL,
            anchored: false,
            verdict: Verdict::Open,
            rows: 3,
            iters: 512,
        },
        LinkFixture {
            name: "link_refused",
            // An editor link, as `rg --hyperlink-format vscode` prints one.
            line: format!(
                "  \x1b]8;;{EDITOR}\x1b\\src/grid/view.rs:533:12\x1b]8;;\x1b\\: pub fn link_at("
            ),
            pointer: (LINK_ROW, 10),
            url: EDITOR,
            anchored: true,
            verdict: Verdict::Refused(Refusal::Scheme),
            rows: 1,
            iters: 8192,
        },
    ]
}

/// A grid holding `fixture`'s line at [`LINK_ROW`] and nothing else, since a lookup reads
/// only the logical line under the pointer.
fn link_screen(fixture: &LinkFixture) -> Screen {
    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let bytes = format!("\x1b[{};1H{}", LINK_ROW + 1, fixture.line);
    Parser::new().advance_bytes(&mut screen, bytes.as_bytes());
    screen
}

/// Look up the link under a resting pointer, which the window does on every cell the
/// pointer enters and after every output turn while it is parked over the grid. A warmed
/// probe reuses its buffers, so the lookup allocates nothing.
///
/// A fixture whose lookup misses its link, or finds it by the other path, is an error
/// rather than a sample, since it would time something else and read as a speedup.
fn link_scenario(fixture: &LinkFixture) -> Result<Sample> {
    let screen = link_screen(fixture);
    let mut probe = LinkProbe::default();
    let (row, col) = fixture.pointer;
    let anchored = screen.cell(row, col).link != LinkId::NONE;
    let rows = screen
        .link_at(row, col, &mut probe)
        .map(|((first, _), (last, _))| last - first + 1);
    if anchored != fixture.anchored
        || probe.url() != fixture.url
        || probe.verdict() != Some(fixture.verdict)
        || rows != Some(fixture.rows)
    {
        return Err(Error::msg(format!(
            "{}: expected {:?} ({:?}) over {} rows (anchored: {}), found {:?} ({:?}) over {rows:?} (anchored: {anchored})",
            fixture.name,
            fixture.url,
            fixture.verdict,
            fixture.rows,
            fixture.anchored,
            probe.url(),
            probe.verdict(),
        )));
    }
    let (ns, allocs) = best_of(GATE_ROUNDS, fixture.iters, || {
        screen.link_at(row, col, &mut probe)
    });
    Ok(Sample {
        name: fixture.name,
        ns,
        allocs,
    })
}

/// Ctrl held over a refused link: write its preview line, as a hover onto it does, then
/// build the 80x24 frame that shows it, with its hidden character escaped and the line
/// cut to the grid, through the window's pooled display-list path. Warm, it allocates
/// nothing: the line goes into a reused buffer and the frame's strings are recycled.
fn link_preview_scenario() -> Result<Sample> {
    use crate::color::Theme;
    use crate::term_render::{
        build_display_list_into, CellMetrics, CellSpan, CursorRender, DisplayListPool, FrameInputs,
        LinkHover,
    };

    const COLS: usize = 80;
    const ROWS: usize = 24;
    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };
    let url = LINK_LONG_URL.replacen("download", "down\u{200B}load", 1);
    let mut screen = Screen::new(COLS, ROWS);
    Parser::new().advance_bytes(&mut screen, format!("\x1b[3Hsee {url} here").as_bytes());
    let mut probe = LinkProbe::default();
    let verdict = Verdict::Refused(Refusal::Hidden);
    let span = screen.link_at(2, 10, &mut probe);
    let Some((start, end)) =
        span.filter(|_| probe.url() == url && probe.verdict() == Some(verdict))
    else {
        return Err(Error::msg(format!(
            "link_preview: expected the long URL refused as hidden, found {:?} ({:?})",
            probe.url(),
            probe.verdict()
        )));
    };
    let hover = LinkHover {
        span: CellSpan { start, end },
        verdict,
    };

    let theme = Theme::default();
    let surface = (COLS as i32 * METRICS.w, ROWS as i32 * METRICS.h);
    let room = crate::term_render::preview_room(COLS, METRICS);
    let mut line = String::new();
    let mut lists = DisplayListPool::default();
    let mut build = || {
        crate::link_preview::write(&mut line, &url, verdict, room);
        let len = {
            let (out, strings) = lists.begin();
            build_display_list_into(
                out,
                strings,
                &FrameInputs {
                    screen: &screen,
                    bell: false,
                    theme: &theme,
                    metrics: METRICS,
                    surface,
                    origin: (0, 0),
                    cursor: CursorRender::default(),
                    selection: None,
                    hover: Some(hover),
                    preview: Some(&line),
                    scale: Scale::ONE,
                    scrollbar: &HIDDEN_SCROLLBAR,
                },
            );
            out.len()
        };
        lists.commit();
        len
    };
    for _ in 0..4 {
        black_box(build());
    }
    let (ns, allocs) = best_of(GATE_ROUNDS, 512, &mut build);
    Ok(Sample {
        name: "link_preview",
        ns,
        allocs,
    })
}

/// One column of an interactive width drag against a full scrollback ring: the grid
/// rewrap ([`Screen::resize`]) the window runs for every painted frame while a resize
/// handle is moving.
///
/// Every other scenario holds the width fixed, and this is the one grid operation whose
/// work is the whole ring rather than the screen: unwrapping and re-laying-out ten
/// thousand rows per configure. The allocation column carries the gate here. A row whose
/// text already fits the new width is handed over untouched, so a warmed drag step
/// allocates a handful of times in total, and a change that goes back to rebuilding rows
/// shows thousands before the timing moves.
///
/// One column, not ten: that is the step a drag delivers, and the shortcut it measures is
/// the one a wide swing cannot take.
fn grid_reflow_scenario() -> Result<Sample> {
    let chunk = corpus("ascii", CHUNK)?;
    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    warm(&mut parser, &mut screen, &chunk);
    // Alternate between two adjacent widths so every timed op is a real reflow (a resize
    // to the width already in force returns without doing anything).
    let mut wide = false;
    let (ns, allocs) = best_of(GATE_ROUNDS, 64, || {
        wide = !wide;
        screen.resize(BENCH_COLS - usize::from(wide), BENCH_ROWS);
        screen.cursor() // an observable result, so the work is not elided
    });
    Ok(Sample {
        name: "grid_reflow",
        ns,
        allocs,
    })
}

/// Diff two display lists differing by a handful of recolored fills, the small
/// changed-frame case the live loop runs every painted frame, through the pooled
/// reused rectangle buffer. Guards the invariant behind the render bench's zero: a warmed
/// damage diff reuses its rectangle buffer and allocates nothing.
fn frame_damage_scenario() -> Sample {
    use crate::platform::geom::Rect;
    use crate::render::display::{damage_into, DrawCmd};

    const CELL: i32 = 8;
    let (w, h) = (BENCH_COLS as i32 * CELL, BENCH_ROWS as i32 * 16);
    // One small fill per row, scattered across the width; the next frame recolors a
    // few, so the diff yields a few small rectangles (well under the full-surface
    // collapse threshold) rather than one big region.
    let old: Vec<DrawCmd> = (0..BENCH_ROWS as i32)
        .map(|r| DrawCmd::Fill {
            rect: Rect {
                x: (r % 40) * CELL,
                y: r * 16,
                w: CELL,
                h: 16,
            },
            color: 0x0011_1111,
        })
        .collect();
    let mut new = old.clone();
    for cmd in new.iter_mut().step_by(20) {
        if let DrawCmd::Fill { color, .. } = cmd {
            *color = 0x0022_2222;
        }
    }

    let mut scratch = Vec::new();
    let mut diff = || {
        damage_into(&old, &new, w, h, &mut scratch);
        scratch.len()
    };
    for _ in 0..4 {
        black_box(diff());
    }
    // A far larger inner count than the other scenarios, because the op is an order of
    // magnitude smaller (~200 ns against 4.5 us for the next smallest). At 512 iterations
    // a round timed a 0.1 ms window, too short to be reproducible; this puts it on the
    // same ~1.5 ms footing as the rest. The time is per op and the allocation count
    // does not depend on the inner count, so the baseline stays comparable either way.
    let (ns, allocs) = best_of(GATE_ROUNDS, 8192, &mut diff);
    Sample {
        name: "frame_damage",
        ns,
        allocs,
    }
}

/// Diff two frames of a full-screen colour scroll: every cell in its own truecolour, as
/// `lolcat` paints, and the second frame one line further on. Each cell is its own run
/// and every run moves, so the diff hands the merge two overlapping rectangles per cell,
/// about 19k, which must still collapse to one full-surface repaint cheaply.
fn frame_damage_scroll_scenario() -> Sample {
    use crate::color::Theme;
    use crate::render::display::{damage_into, DisplayList};
    use crate::term_render::{build_display_list_into, CellMetrics, CursorRender, FrameInputs};

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };
    let surface = (BENCH_COLS as i32 * METRICS.w, BENCH_ROWS as i32 * METRICS.h);
    let theme = Theme::default();
    let frame = |screen: &Screen| {
        let mut out = DisplayList::new();
        build_display_list_into(
            &mut out,
            &mut Vec::new(),
            &FrameInputs {
                screen,
                bell: false,
                theme: &theme,
                metrics: METRICS,
                surface,
                origin: (0, 0),
                cursor: CursorRender::default(),
                selection: None,
                hover: None,
                preview: None,
                scale: Scale::ONE,
                scrollbar: &HIDDEN_SCROLLBAR,
            },
        );
        out
    };
    // No cell shares its colour with its neighbours or the cells above and below it.
    let rainbow = |first: usize, lines: usize| {
        let mut out = Vec::new();
        for n in first..first + lines {
            for c in 0..BENCH_COLS {
                let (r, g, b) = ((n * 8 + c * 4) % 256, (n * 4 + c * 8) % 256, (c * 2) % 256);
                out.extend_from_slice(format!("\x1b[38;2;{r};{g};{b}m#").as_bytes());
            }
            out.extend_from_slice(b"\x1b[0m\r\n");
        }
        out
    };

    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    parser.advance_bytes(&mut screen, &rainbow(0, BENCH_ROWS));
    let old = frame(&screen);
    parser.advance_bytes(&mut screen, &rainbow(BENCH_ROWS, 1));
    let new = frame(&screen);

    let mut scratch = Vec::new();
    let mut diff = || {
        damage_into(&old, &new, surface.0, surface.1, &mut scratch);
        scratch.len()
    };
    for _ in 0..4 {
        black_box(diff());
    }
    let (ns, allocs) = best_of(GATE_ROUNDS, 32, &mut diff);
    Sample {
        name: "frame_damage_scroll",
        ns,
        allocs,
    }
}

/// A fixed-shape stream: `lines` identical rows, each `cells` copies of `cell`
/// followed by a newline. Every row needs the same combining capacity, so a warmed
/// feed against a cycled ring reallocates nothing, unlike the varied lab streams.
fn repeated_lines(cell: &str, cells: usize, lines: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((cell.len() * cells + 1) * lines);
    for _ in 0..lines {
        for _ in 0..cells {
            out.extend_from_slice(cell.as_bytes());
        }
        out.push(b'\n');
    }
    out
}

/// Build a populated 80x24 grid plus a four-tab bar through the window's pooled
/// display-list path. This guards both CPU cost and the stronger invariant: once
/// both list buffers and their strings are warm, one complete frame allocates
/// nothing.
fn frame_tabbar_scenario() -> Result<Sample> {
    use crate::color::Theme;
    use crate::config::TabBarConfig;
    use crate::tab_bar::{self, BarGeom, TabLabel};
    use crate::term_render::{
        build_display_list_into, CellMetrics, CursorRender, DisplayListPool, FrameInputs,
    };

    const COLS: usize = 80;
    const ROWS: usize = 24;
    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    let mut screen = Screen::new(COLS, ROWS);
    let mut parser = Parser::new();
    let mut content = b"\x1b[H".to_vec();
    for row in 0..ROWS {
        let line = format!("row {row:02} populated terminal text\r\n");
        content.extend_from_slice(line.as_bytes());
    }
    parser.advance_bytes(&mut screen, &content);

    let theme = Theme::default();
    let cfg = TabBarConfig::default();
    let surface = (COLS as i32 * METRICS.w, (ROWS as i32 + 1) * METRICS.h);
    let slots = tab_bar::layout(
        COLS,
        &[
            TabLabel {
                title: "edit",
                active: true,
            },
            TabLabel {
                title: "make",
                active: false,
            },
            TabLabel {
                title: "logs",
                active: false,
            },
            TabLabel {
                title: "term",
                active: false,
            },
        ],
        &cfg,
    );
    // The labels are the proportional interface font, so painting the bar needs the
    // fonts to measure advances. Load them once (outside the measured loop); a bar
    // repaint itself allocates nothing.
    let fonts = crate::platform::freetype::Fonts::new(&[METRICS.size])
        .map_err(|e| Error::msg(format!("frame_tabbar needs the default fonts: {e}")))?;
    let bar = BarGeom {
        metrics: METRICS,
        label: METRICS,
        surface_width: surface.0,
        pad: 0,
        y: ROWS as i32 * METRICS.h,
        h: METRICS.h,
    };
    let mut lists = DisplayListPool::default();
    let mut build = || {
        let len = {
            let (out, strings) = lists.begin();
            build_display_list_into(
                out,
                strings,
                &FrameInputs {
                    screen: &screen,
                    bell: false,
                    theme: &theme,
                    metrics: METRICS,
                    surface,
                    origin: (0, METRICS.h),
                    cursor: CursorRender::default(),
                    selection: None,
                    hover: None,
                    preview: None,
                    scale: Scale::ONE,
                    scrollbar: &HIDDEN_SCROLLBAR,
                },
            );
            tab_bar::fill_bar(out, strings, &slots, &bar, &cfg, &fonts, None);
            out.len()
        };
        lists.commit();
        len
    };

    // Warm both alternating buffers and all recycled run strings before best_of's
    // own warm/allocation samples.
    for _ in 0..4 {
        black_box(build());
    }
    let (ns, allocs) = best_of(GATE_ROUNDS, 512, &mut build);
    Ok(Sample {
        name: "frame_tabbar",
        ns,
        allocs,
    })
}

/// Measure feeding `chunk` into a warmed grid. Warmup fills the scrollback ring
/// so a steady scroll recycles row storage (allocating nothing), then reaches a
/// stable state for this chunk; all of it is untimed.
fn stream_scenario(name: &'static str, chunk: &[u8], iters: u64) -> Sample {
    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    warm(&mut parser, &mut screen, chunk);
    let (ns, allocs) = best_of(GATE_ROUNDS, iters, || {
        parser.advance_bytes(&mut screen, chunk);
        screen.cursor() // an observable result, so the work is not elided
    });
    Sample { name, ns, allocs }
}

/// Drive the grid to steady state before timing: enough newlines to fill the
/// scrollback ring (so later scrolls recycle rather than grow), then enough chunk
/// feeds to cycle the whole ring with the chunk's own content.
///
/// The second pass matters for the combining/emoji chunks: a row's combining side
/// table grows the first time that row ever holds a mark, and `Row::reset` then
/// retains the capacity, so only a full pass over the ring reaches the true steady
/// state where a recycled row reallocates nothing. Sixteen feeds touch a fraction
/// of the ring and leave that first-touch growth leaking into the timed op.
fn warm(parser: &mut Parser, screen: &mut Screen, chunk: &[u8]) {
    let newlines = [b'\n'; 256];
    for _ in 0..128 {
        parser.advance_bytes(screen, &newlines); // ~32k newlines > scrollback cap
    }
    // Feed the chunk until it has scrolled past the ring several times over, plus a
    // few to settle. One pass leaves the rows recycled right at the wrap boundary
    // untouched, so a few first-touch growths leak into the timed op; three passes
    // clear that margin cheaply. A chunk that does not scroll (a homed screenful)
    // overwrites in place, so `checked_div` yields `None` and the plain 16 feeds
    // already reach its steady state.
    let lines = chunk.iter().filter(|&&b| b == b'\n').count();
    let feeds = match (3 * crate::grid::DEFAULT_SCROLLBACK).checked_div(lines) {
        Some(passes) => passes + 16,
        None => 16,
    };
    for _ in 0..feeds {
        parser.advance_bytes(screen, chunk);
    }
}

/// The operations each scenario's allocation count covers. Separate from the timing's
/// `rounds * inner`, so retuning a scenario's timing never moves its count. Even, so a
/// scenario that alternates two steps (`grid_reflow`) counts both alike, and long enough
/// that growth amortised over many ops, such as a buffer doubling now and then, shows.
const COUNTED_OPS: u64 = 64;

/// The best (minimum) ns per op over `rounds`, each timing `inner` ops, plus the
/// allocations [`COUNTED_OPS`] ops make in total. The minimum is the least-noisy
/// estimator for a microbenchmark. The count is a total because a per-op average would
/// round an occasional allocation down to zero. It comes from an untimed pass after one
/// warm op, and the counter is per thread (see [`alloc`]), so a parallel sibling never
/// bleeds into it. Under `perf-alloc` the timed rounds still pay the counter on every
/// allocation they make, so an allocating scenario times a little slower there than in a
/// plain build. `black_box` stops the optimizer eliding the work.
fn best_of<T>(rounds: u32, inner: u64, mut f: impl FnMut() -> T) -> (u64, u64) {
    black_box(f());
    alloc::reset();
    for _ in 0..COUNTED_OPS {
        black_box(f());
    }
    let allocs = alloc::snapshot().allocs;

    let mut best = u64::MAX;
    for _ in 0..rounds.max(1) {
        let start = Instant::now();
        for _ in 0..inner.max(1) {
            black_box(f());
        }
        best = best.min((start.elapsed().as_nanos() as u64) / inner.max(1));
    }
    (best, allocs)
}

/// How a measured number compares to its baseline; the one classification behind
/// the columns and the gate, so they agree by construction.
#[derive(Clone, Copy)]
enum Delta {
    Regressed,
    Improved,
    Flat,
}

impl Delta {
    /// A sampled time: only a move of [`REGRESSION_RATIO`] either way counts.
    fn of_time(current: u64, base: u64) -> Delta {
        if current as f64 >= base.max(1) as f64 * REGRESSION_RATIO {
            Delta::Regressed
        } else if base as f64 / current.max(1) as f64 >= REGRESSION_RATIO {
            Delta::Improved
        } else {
            Delta::Flat
        }
    }

    /// A count: exact and the same on every run, so any change counts.
    fn of_count(current: u64, base: u64) -> Delta {
        match current.cmp(&base) {
            Ordering::Greater => Delta::Regressed,
            Ordering::Less => Delta::Improved,
            Ordering::Equal => Delta::Flat,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Delta::Regressed => "REGRESSION",
            Delta::Improved => "improved",
            Delta::Flat => "",
        }
    }

    fn regressed(self) -> bool {
        matches!(self, Delta::Regressed)
    }
}

/// Print a current-vs-baseline table and return the names that regressed. Time is
/// always compared; allocations only when `counted` says this run measured them, so
/// an unmeasured run never reads as a spurious improvement to zero.
fn print_table(
    samples: &[Sample],
    baseline: &HashMap<String, Base>,
    counted: bool,
) -> Vec<&'static str> {
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    let allocs = format!("allocs/{COUNTED_OPS}");
    println!(
        "\n{:<16} {:>11} {:>11} {:>16}   {:>9} {:>8} {:>16}",
        "metric", "ns/op", "base", "Δ", allocs, "base", "Δ"
    );
    let mut regressed = Vec::new();
    for s in sorted {
        let base = baseline.get(s.name);

        let ns_delta = base.map(|b| Delta::of_time(s.ns, b.ns));
        let (base_ns_col, ns_flag) = match (base, ns_delta) {
            (Some(b), Some(delta)) => (
                b.ns.to_string(),
                format!("{:+.1}% {}", ratio_pct(s.ns, b.ns), delta.label()),
            ),
            _ => ("(new)".to_string(), String::new()),
        };

        let alloc_delta = base
            .filter(|_| counted)
            .map(|b| Delta::of_count(s.allocs, b.allocs));
        let (alloc_cur, alloc_base, alloc_col) = if !counted {
            ("-".to_string(), "-".to_string(), String::new())
        } else if let (Some(b), Some(delta)) = (base, alloc_delta) {
            (
                s.allocs.to_string(),
                b.allocs.to_string(),
                count_change(s.allocs, b.allocs, delta),
            )
        } else {
            (s.allocs.to_string(), String::new(), String::new())
        };

        if ns_delta.is_some_and(Delta::regressed) || alloc_delta.is_some_and(Delta::regressed) {
            regressed.push(s.name);
        }

        println!(
            "{:<16} {:>11} {:>11} {:>16}   {:>9} {:>8} {:>16}",
            s.name, s.ns, base_ns_col, ns_flag, alloc_cur, alloc_base, alloc_col
        );
    }
    regressed
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

/// The Δ column for a count: the signed difference, since a percentage of a zero
/// baseline says nothing, then the label. Blank when the count held.
fn count_change(current: u64, base: u64, delta: Delta) -> String {
    let diff = current.abs_diff(base);
    match delta {
        Delta::Regressed => format!("+{diff} {}", delta.label()),
        Delta::Improved => format!("-{diff} {}", delta.label()),
        Delta::Flat => String::new(),
    }
}

/// Write samples to `path`, sorted by name for a stable diff. The allocation
/// column is written only when this run measured it, so a non-`perf-alloc` run
/// never records zeros a later comparison would misread.
fn write_samples(path: &str, samples: &[Sample]) -> Result<()> {
    ensure_parent(path)?;
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    let mut text =
        format!("# bnkterm perf results: name, ns/op, allocations over {COUNTED_OPS} ops (lower is better)\n");
    for s in sorted {
        if alloc::ENABLED {
            let _ = writeln!(text, "{}\t{}\t{}", s.name, s.ns, s.allocs);
        } else {
            let _ = writeln!(text, "{}\t{}", s.name, s.ns);
        }
    }
    std::fs::write(path, text).map_err(|e| Error::msg(format!("write {path}: {e}")))
}

/// Load a baseline into a name -> entry map, or `None` if it does not exist.
fn load_baseline(path: &str) -> Result<Option<HashMap<String, Base>>> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_baseline(&text).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::msg(format!("read {path}: {e}"))),
    }
}

/// Parse `name<ws>ns<ws>allocs`, skipping blank and `#` lines. Every entry needs its
/// allocation count: a baseline without one would leave allocations ungated, so it is
/// malformed rather than read as unmeasured.
fn parse_baseline(text: &str) -> Result<HashMap<String, Base>> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(name), Some(ns), Some(allocs), None) =
            (it.next(), it.next(), it.next(), it.next())
        else {
            return Err(Error::msg(format!(
                "malformed perf line {line:?}: want name, ns/op and allocations"
            )));
        };
        let ns = ns
            .parse::<u64>()
            .map_err(|_| Error::msg(format!("malformed perf number: {ns:?}")))?;
        let allocs = allocs
            .parse::<u64>()
            .map_err(|_| Error::msg(format!("malformed alloc number: {allocs:?}")))?;
        out.insert(name.to_string(), Base { ns, allocs });
    }
    Ok(out)
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

// ---------------------------------------------------------------------------
// --perf: the footprint gate.
// ---------------------------------------------------------------------------

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
struct Footprint {
    name: &'static str,
    bytes: u64,
    rows: usize,
}

/// Measure what a full grid costs for each payload shape, from plain ASCII (the
/// floor: cells and nothing else) through emoji (the combining side table's worst
/// case). Same corpora as the timing scenarios, so a change that trades bytes for
/// speed shows up in both tables of one run.
fn measure_footprints() -> Result<Vec<Footprint>> {
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
fn gate_footprints(footprints: &[Footprint], save: bool) -> Result<Vec<&'static str>> {
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
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::msg(format!("read {path}: {e}"))),
    };
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(name), Some(bytes)) = (it.next(), it.next()) else {
            return Err(Error::msg(format!("malformed footprint line: {line:?}")));
        };
        let bytes = bytes
            .parse::<u64>()
            .map_err(|_| Error::msg(format!("malformed footprint number: {bytes:?}")))?;
        out.insert(name.to_string(), bytes);
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// --cat-bench: in-process throughput over a real file.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// --render-bench: the render pipeline, headless (no GPU, no window).
// ---------------------------------------------------------------------------

/// Font size (device px) for the render harness. Once the glyph cache is warm the
/// per-frame cost is dominated by the cell count, not the raster size, so the exact
/// value barely matters.
const RENDER_FONT_PX: u32 = 16;
/// Frames timed per `--render-bench` run; the minimum is the least-noisy estimate.
const RENDER_FRAMES: u32 = 500;
/// A painted frame happens at most once per compositor vsync (~60 Hz), so a ~1.3 s
/// cat renders ~this many frames. Used only to project the per-frame CPU cost onto
/// the full-terminal render residual.
const CAT_FRAMES_ESTIMATE: f64 = 78.0;

/// `count` distinct printable chars from `start`, in rows of 30 so a screenful holds
/// them without scrolling out of view. Every char is unique, so feeding this to a
/// fresh glyph cache makes every cell a cache miss: the input for measuring the
/// per-glyph rasterise-and-pack allocation the miss path costs.
fn distinct_glyphs(start: u32, count: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let (mut cp, mut made, mut col) = (start, 0, 0);
    while made < count {
        if let Some(c) = char::from_u32(cp) {
            if !c.is_control() && !c.is_whitespace() {
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                made += 1;
                col += 1;
                if col >= 30 {
                    out.push(b'\n');
                    col = 0;
                }
            }
        }
        cp = cp.wrapping_add(1);
    }
    out
}

/// Time the CPU render pipeline in isolation: `grid -> build_display_list -> damage
/// diff -> gpu::build_frame` (vertices), per frame, with no GPU and no window. This
/// gives the render side its own number so a fast parser can never hide a slow
/// renderer. It measures the CPU work every painted frame costs;
/// GPU execution and compositor present are a separate question that needs the
/// device. Built `--features perf-alloc` it also reports allocations per frame —
/// with the display list, its run strings, the vertex buffer, and the damage-diff
/// scratch all pooled across frames, a warmed changed frame allocates nothing.
pub fn render_bench(args: &[String]) -> Result<()> {
    use crate::color::Theme;
    use crate::platform::freetype::Fonts;
    use crate::render::display::damage_into;
    use crate::render::gpu::{build_frame_into, FrameData, GlyphCache};
    use crate::term_render::{
        build_display_list_into, CellMetrics, CursorRender, DisplayListPool, FrameInputs,
    };

    // `--frames N` overrides the default (the smoke test uses a tiny count).
    let frames = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(RENDER_FRAMES);

    let fonts = Fonts::new(&[RENDER_FONT_PX])
        .map_err(|e| Error::msg(format!("render-bench needs a monospace font: {e}")))?;
    let metrics = CellMetrics::from_fonts(&fonts, RENDER_FONT_PX);
    let theme = Theme::default();
    let surface = (metrics.w * BENCH_COLS as i32, metrics.h * BENCH_ROWS as i32);
    let origin = (0, 0);
    let cursor = CursorRender::default();

    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    let mut cache = GlyphCache::default();
    // The window's per-frame render state, reused across frames exactly as the live
    // loop reuses it: the double-buffered list + string pool, the frame data, and the
    // damage-diff scratch.
    let mut lists = DisplayListPool::default();
    let mut frame = FrameData::default();
    let mut damage_scratch: Vec<crate::platform::geom::Rect> = Vec::new();
    // A little over a screenful of real prose per frame, enough to turn the whole
    // grid over — the worst case a fast cat hits, so the diff sees a fully changed
    // screen and every cell is rebuilt.
    let feed = corpus("ascii", BENCH_COLS * BENCH_ROWS * 2)?;

    // Warm: rasterize every glyph and size the pool's buffers, string pool, and the
    // frame's vertex capacity, so the timed loop and the alloc snapshot measure
    // steady-state reuse, not first-touch growth. Each pass turns the grid over so
    // `front`/`back` genuinely differ (a real full-screen diff, like a cat).
    for _ in 0..3 {
        parser.advance_bytes(&mut screen, &feed);
        let (out, strings) = lists.begin();
        build_display_list_into(
            out,
            strings,
            &FrameInputs {
                screen: &screen,
                bell: false,
                theme: &theme,
                metrics,
                surface,
                origin,
                cursor,
                selection: None,
                hover: None,
                preview: None,
                scale: Scale::ONE,
                scrollbar: &HIDDEN_SCROLLBAR,
            },
        );
        damage_into(
            lists.front(),
            lists.back(),
            surface.0,
            surface.1,
            &mut damage_scratch,
        );
        build_frame_into(
            &fonts,
            lists.back(),
            &mut cache,
            crate::render::gpu::TextGamma::default(),
            &mut frame,
        );
        lists.commit();
    }

    // One steady frame's allocation cost. With the list, its run strings, the frame
    // vertices, and the damage-diff scratch all reused, a warmed changed frame
    // allocates nothing.
    parser.advance_bytes(&mut screen, &feed);
    alloc::reset();
    {
        let (out, strings) = lists.begin();
        build_display_list_into(
            out,
            strings,
            &FrameInputs {
                screen: &screen,
                bell: false,
                theme: &theme,
                metrics,
                surface,
                origin,
                cursor,
                selection: None,
                hover: None,
                preview: None,
                scale: Scale::ONE,
                scrollbar: &HIDDEN_SCROLLBAR,
            },
        );
        damage_into(
            lists.front(),
            lists.back(),
            surface.0,
            surface.1,
            &mut damage_scratch,
        );
        build_frame_into(
            &fonts,
            lists.back(),
            &mut cache,
            crate::render::gpu::TextGamma::default(),
            &mut frame,
        );
        black_box((&damage_scratch, &frame));
        lists.commit();
    }
    let allocs = alloc::snapshot().allocs;

    let (mut build_ns, mut diff_ns, mut frame_ns, mut total_ns) =
        (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
    let mut verts = 0usize;
    for _ in 0..frames {
        parser.advance_bytes(&mut screen, &feed); // scroll the grid (untimed)

        let t0 = Instant::now();
        {
            let (out, strings) = lists.begin();
            build_display_list_into(
                out,
                strings,
                &FrameInputs {
                    screen: &screen,
                    bell: false,
                    theme: &theme,
                    metrics,
                    surface,
                    origin,
                    cursor,
                    selection: None,
                    hover: None,
                    preview: None,
                    scale: Scale::ONE,
                    scrollbar: &HIDDEN_SCROLLBAR,
                },
            );
        }
        let t1 = Instant::now();
        damage_into(
            lists.front(),
            lists.back(),
            surface.0,
            surface.1,
            &mut damage_scratch,
        );
        let t2 = Instant::now();
        build_frame_into(
            &fonts,
            lists.back(),
            &mut cache,
            crate::render::gpu::TextGamma::default(),
            &mut frame,
        );
        let t3 = Instant::now();
        verts = frame.vertices.len();
        black_box((&damage_scratch, &frame));

        build_ns = build_ns.min((t1 - t0).as_nanos() as u64);
        diff_ns = diff_ns.min((t2 - t1).as_nanos() as u64);
        frame_ns = frame_ns.min((t3 - t2).as_nanos() as u64);
        total_ns = total_ns.min((t3 - t0).as_nanos() as u64);
        lists.commit();
    }

    // One-cell change: only the home cell differs from the last frame (a cursor
    // blink or a single keystroke), yet the list and every vertex are still rebuilt
    // in full. Alternating the char keeps each rep a real change. Of the three stages
    // only `damage` scales with the edit rather than the screen.
    let (mut edit_list, mut edit_diff, mut edit_frame) = (u64::MAX, u64::MAX, u64::MAX);
    for i in 0..frames {
        parser.advance_bytes(&mut screen, b"\x1b[1;1H");
        parser.advance_bytes(&mut screen, if i % 2 == 0 { b"X" } else { b"Y" });
        let t0 = Instant::now();
        {
            let (out, strings) = lists.begin();
            build_display_list_into(
                out,
                strings,
                &FrameInputs {
                    screen: &screen,
                    bell: false,
                    theme: &theme,
                    metrics,
                    surface,
                    origin,
                    cursor,
                    selection: None,
                    hover: None,
                    preview: None,
                    scale: Scale::ONE,
                    scrollbar: &HIDDEN_SCROLLBAR,
                },
            );
        }
        let t1 = Instant::now();
        damage_into(
            lists.front(),
            lists.back(),
            surface.0,
            surface.1,
            &mut damage_scratch,
        );
        let t2 = Instant::now();
        build_frame_into(
            &fonts,
            lists.back(),
            &mut cache,
            crate::render::gpu::TextGamma::default(),
            &mut frame,
        );
        let t3 = Instant::now();
        black_box((&damage_scratch, &frame));
        edit_list = edit_list.min((t1 - t0).as_nanos() as u64);
        edit_diff = edit_diff.min((t2 - t1).as_nanos() as u64);
        edit_frame = edit_frame.min((t3 - t2).as_nanos() as u64);
        lists.commit();
    }

    // Idle: nothing changed since the on-screen frame (no output, no blink). The list
    // is rebuilt and diffed in full, but the diff comes back empty and the live loop
    // stops there (`app::present::render_frame` early-outs on empty damage *before*
    // building vertices), so an idle frame costs list + diff and never pays the vertex
    // build. Neither list nor front is committed, exactly as that early-out does not:
    // `front` stays the on-screen frame and each pass refills `back` from the pool.
    let (mut idle_list, mut idle_diff, mut idle_rects) = (u64::MAX, u64::MAX, usize::MAX);
    for _ in 0..frames {
        let t0 = Instant::now();
        {
            let (out, strings) = lists.begin();
            build_display_list_into(
                out,
                strings,
                &FrameInputs {
                    screen: &screen,
                    bell: false,
                    theme: &theme,
                    metrics,
                    surface,
                    origin,
                    cursor,
                    selection: None,
                    hover: None,
                    preview: None,
                    scale: Scale::ONE,
                    scrollbar: &HIDDEN_SCROLLBAR,
                },
            );
        }
        let t1 = Instant::now();
        damage_into(
            lists.front(),
            lists.back(),
            surface.0,
            surface.1,
            &mut damage_scratch,
        );
        let t2 = Instant::now();
        idle_rects = damage_scratch.len();
        black_box(&damage_scratch);
        idle_list = idle_list.min((t1 - t0).as_nanos() as u64);
        idle_diff = idle_diff.min((t2 - t1).as_nanos() as u64);
    }

    let us = |ns: u64| ns as f64 / 1000.0;
    println!(
        "render-bench: {BENCH_COLS}x{BENCH_ROWS} grid, {RENDER_FONT_PX}px, {frames} frames \
         (grid -> list -> diff -> vertices; no GPU){}",
        if alloc::ENABLED {
            ""
        } else {
            " (allocs unmeasured; build --features perf-alloc)"
        }
    );
    println!("  build_list   {:>8.1} us/frame", us(build_ns));
    println!("  damage_diff  {:>8.1} us/frame", us(diff_ns));
    println!(
        "  build_frame  {:>8.1} us/frame  ({verts} vertices)",
        us(frame_ns)
    );
    println!("  total        {:>8.1} us/frame", us(total_ns));
    if alloc::ENABLED {
        println!("  allocs       {allocs:>8} per frame (list, strings, vertices, and damage scratch all reused)");
    }
    println!(
        "  1-cell edit  {:>8.1} us list + {:.1} us diff + {:.1} us vertices = {:.1} us \
         (list & vertices rebuilt in full for one changed cell)",
        us(edit_list),
        us(edit_diff),
        us(edit_frame),
        us(edit_list + edit_diff + edit_frame)
    );
    println!(
        "  idle frame   {:>8.1} us list + {:.1} us diff = {:.1} us ({idle_rects} damage rects: \
         the live loop early-outs here, so no vertex build)",
        us(idle_list),
        us(idle_diff),
        us(idle_list + idle_diff)
    );
    println!(
        "  => est. render CPU over a ~1.3 s / ~{CAT_FRAMES_ESTIMATE:.0}-frame cat: {:.0} ms \
         (compare against the ~0.5 s full-terminal residual)",
        us(total_ns) * CAT_FRAMES_ESTIMATE / 1000.0
    );

    // Miss-path allocation, isolated: build the same frame twice, once against a warm
    // cache (all hits) and once against a fresh one (all misses), so the difference is
    // exactly what rasterising and packing a novel glyph costs, with the vertex-buffer
    // growth common to both cancelled out. A warm frame allocates nothing; first-seen
    // glyphs run the rasterise, pack and upload path on every frame.
    if alloc::ENABLED {
        let miss_allocs = |content: &[u8]| -> u64 {
            let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
            Parser::new().advance_bytes(&mut screen, content);
            let mut lists = DisplayListPool::default();
            {
                let (out, strings) = lists.begin();
                build_display_list_into(
                    out,
                    strings,
                    &FrameInputs {
                        screen: &screen,
                        bell: false,
                        theme: &theme,
                        metrics,
                        surface,
                        origin,
                        cursor,
                        selection: None,
                        hover: None,
                        preview: None,
                        scale: Scale::ONE,
                        scrollbar: &HIDDEN_SCROLLBAR,
                    },
                );
            }
            lists.commit();
            let list = lists.front();
            let g = crate::render::gpu::TextGamma::default();
            // Warm cache + fresh frame: measures only the fresh frame's buffer growth.
            let mut warm = GlyphCache::default();
            build_frame_into(&fonts, list, &mut warm, g, &mut FrameData::default());
            let mut fa = FrameData::default();
            alloc::reset();
            build_frame_into(&fonts, list, &mut warm, g, &mut fa);
            let hit = alloc::snapshot().allocs;
            black_box(&fa);
            // Fresh cache + fresh frame: the same growth plus every glyph rasterised.
            let mut cold = GlyphCache::default();
            let mut fb = FrameData::default();
            alloc::reset();
            build_frame_into(&fonts, list, &mut cold, g, &mut fb);
            let miss = alloc::snapshot().allocs;
            black_box(&fb);
            miss.saturating_sub(hit)
        };
        let per = |allocs: u64, n: usize| allocs as f64 / n.max(1) as f64;
        let (scalars, emojis) = (200usize, 80usize);
        let scalar_allocs = miss_allocs(&distinct_glyphs(0x00C0, scalars));
        let emoji_allocs = miss_allocs(&distinct_glyphs(0x1F600, emojis));
        println!(
            "  new scalar   {scalar_allocs:>8} allocs / {scalars} glyphs ({:.1}/glyph)",
            per(scalar_allocs, scalars)
        );
        println!(
            "  new emoji    {emoji_allocs:>8} allocs / {emojis} clusters ({:.1}/cluster)",
            per(emoji_allocs, emojis)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// --perf-lab: whole-pipeline throughput over the generated streams.
// ---------------------------------------------------------------------------

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

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
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
    use crate::dev::perf::*;

    #[test]
    fn allocations_gate_at_equality_and_time_at_a_ratio() {
        let baseline = parse_baseline(
            "first_alloc\t100\t0\none_more\t100\t71\none_fewer\t100\t71\n\
             noisy\t100\t0\nslow\t100\t0\n",
        )
        .unwrap();
        let samples = [
            ("first_alloc", 100, 1),
            ("one_more", 100, 72),
            ("one_fewer", 100, 70),
            ("noisy", 109, 0),
            ("slow", 111, 0),
        ]
        .map(|(name, ns, allocs)| Sample { name, ns, allocs });
        assert_eq!(
            print_table(&samples, &baseline, true),
            ["first_alloc", "one_more", "slow"]
        );
    }

    #[test]
    fn a_baseline_line_without_its_allocation_count_is_refused() {
        let baseline = parse_baseline("# header\n\ngrid_reflow\t610355\t71\n").unwrap();
        let reflow = &baseline["grid_reflow"];
        assert_eq!((reflow.ns, reflow.allocs), (610355, 71));
        for bad in [
            "grid_reflow\t610355",
            "grid_reflow\t610355\t71\t9",
            "grid_reflow\t610355\tx",
            "grid_reflow\tx\t71",
        ] {
            assert!(parse_baseline(bad).is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn a_link_scenario_times_only_the_lookup_it_names() {
        for f in link_fixtures() {
            link_scenario(&LinkFixture { iters: 1, ..f }).unwrap();
        }
        let [bare, anchor, wrapped, refused] = link_fixtures();
        assert!(link_scenario(&LinkFixture {
            anchored: true,
            ..bare
        })
        .is_err());
        assert!(link_scenario(&LinkFixture {
            url: "https://example.com/",
            ..anchor
        })
        .is_err());
        assert!(link_scenario(&LinkFixture { rows: 2, ..wrapped }).is_err());
        assert!(link_scenario(&LinkFixture {
            verdict: Verdict::Open,
            ..refused
        })
        .is_err());
    }

    #[cfg(feature = "perf-alloc")]
    #[test]
    fn an_allocation_every_few_ops_is_counted() {
        let mut calls = 0u64;
        let (_, allocs) = best_of(1, 1, || {
            calls += 1;
            if calls.is_multiple_of(8) {
                black_box(Vec::<u8>::with_capacity(16));
            }
        });
        assert_eq!(allocs, COUNTED_OPS / 8);
    }

    #[cfg(feature = "perf-alloc")]
    #[test]
    fn a_warmed_link_lookup_allocates_nothing() {
        for f in link_fixtures() {
            let s = link_scenario(&LinkFixture { iters: 8, ..f }).unwrap();
            assert_eq!(
                s.allocs, 0,
                "{} allocated {} over {COUNTED_OPS} lookups",
                s.name, s.allocs
            );
        }
    }

    #[cfg(feature = "perf-alloc")]
    #[test]
    fn steady_state_parse_and_print_allocate_nothing() {
        // Checkable only with the counting allocator on (`cargo test --features
        // perf-alloc`): after warmup, feeding a chunk costs zero allocations. Combining
        // and emoji hold at zero because the grid flattens its combining side table into
        // one retained per-row buffer. The counter is per thread (see `alloc`), so this
        // reads only its own op while sibling tests run in parallel.
        for (name, chunk) in [
            ("ascii", stream::generate("ascii", CHUNK).unwrap()),
            ("unicode", stream::generate("unicode", CHUNK).unwrap()),
            ("combining", repeated_lines("a\u{0301}\u{0302}", 40, 24)),
            ("emoji", repeated_lines("🇳🇴", 40, 24)),
        ] {
            let s = stream_scenario("t", &chunk, 8);
            assert_eq!(
                s.allocs, 0,
                "parse {name} allocated {} over {COUNTED_OPS} chunks",
                s.allocs
            );
        }
    }

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

    #[test]
    fn render_bench_runs_without_error() {
        // Drives the real render pipeline headlessly for a couple of frames. Skip
        // where no monospace font is installed (headless CI), like the pty tests do.
        if crate::platform::freetype::Fonts::new(&[RENDER_FONT_PX]).is_err() {
            eprintln!("no monospace font; skipping render_bench smoke test");
            return;
        }
        assert!(render_bench(&["--frames".into(), "2".into()]).is_ok());
    }
}
