//! `--perf`: the speed and allocation gate. Each scenario is a hot-path microbenchmark
//! timed as the best of [`GATE_ROUNDS`] rounds and its allocations counted over
//! [`COUNTED_OPS`] ops, then compared against `stats/baseline.txt`. [`run`] also runs the
//! footprint gate beside it.

use crate::color::Theme;
use crate::dev::alloc;
use crate::dev::perf::footprint::{gate_footprints, measure_footprints, FOOTPRINT_PATH};
use crate::dev::perf::{
    corpus, ensure_parent, parse_rows, plain_frame, ratio_pct, read_if_present, BENCH_COLS,
    BENCH_ROWS, METRICS, RENDER_FONT_PX,
};
use crate::error::{Error, Result};
use crate::grid::{LinkId, LinkProbe, Screen};
use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::platform::verdict::{Refusal, Verdict};
use crate::render::display::{damage_into, DisplayListPool};
use crate::render::gpu::{build_frame_into, FrameData, GlyphCache, TextGamma};
use crate::term_render::{build_display_list_into, CellMetrics, FrameInputs};
use crate::vt::Parser;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

const BASELINE_PATH: &str = "stats/baseline.txt";
const LATEST_PATH: &str = "data/latest.txt";

/// A scenario counts as a regression when it is at least this much slower than the
/// baseline. Wide enough to absorb timing noise, tight enough to catch the
/// order-of-magnitude regressions this exists for.
const REGRESSION_RATIO: f64 = 1.10;

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
        frame_text_unicode_scenario()?,
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
    use crate::term_render::{CellSpan, LinkHover};

    const COLS: usize = 80;
    const ROWS: usize = 24;
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
                    hover: Some(hover),
                    preview: Some(&line),
                    ..plain_frame(&screen, &theme, METRICS, surface)
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
    use crate::render::display::DrawCmd;

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
    use crate::render::display::DisplayList;

    let surface = (BENCH_COLS as i32 * METRICS.w, BENCH_ROWS as i32 * METRICS.h);
    let theme = Theme::default();
    let frame = |screen: &Screen| {
        let mut out = DisplayList::new();
        build_display_list_into(
            &mut out,
            &mut Vec::new(),
            &plain_frame(screen, &theme, METRICS, surface),
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
    use crate::config::TabBarConfig;
    use crate::tab_bar::{self, BarGeom, TabLabel};

    const COLS: usize = 80;
    const ROWS: usize = 24;

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
                marker: None,
            },
            TabLabel {
                title: "make",
                active: false,
                marker: None,
            },
            TabLabel {
                title: "logs",
                active: false,
                marker: None,
            },
            TabLabel {
                title: "term",
                active: false,
                marker: None,
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
                    origin: (0, METRICS.h),
                    ..plain_frame(&screen, &theme, METRICS, surface)
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

/// The vertices for a screen of non-ASCII text: box drawing as `tree` prints it beside
/// accented prose, on every row, so every run takes the segmented path. The display list
/// is built once; only the vertex build is timed.
fn frame_text_unicode_scenario() -> Result<Sample> {
    let fonts = Fonts::new(&[RENDER_FONT_PX])
        .map_err(|e| Error::msg(format!("frame_text_unicode needs the default fonts: {e}")))?;
    let metrics = CellMetrics::from_fonts(&fonts, RENDER_FONT_PX);
    let theme = Theme::default();
    let surface = (metrics.w * BENCH_COLS as i32, metrics.h * BENCH_ROWS as i32);

    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
    let line: String = "├── café/naïve_résumé.txt │ Ærøskøbing, Größe, façade "
        .chars()
        .cycle()
        .take(BENCH_COLS)
        .collect();
    for row in 1..=BENCH_ROWS {
        parser.advance_bytes(&mut screen, format!("\x1b[{row};1H{line}").as_bytes());
    }

    let mut lists = DisplayListPool::default();
    {
        let (out, strings) = lists.begin();
        build_display_list_into(
            out,
            strings,
            &plain_frame(&screen, &theme, metrics, surface),
        );
    }
    let list = lists.back();
    let mut cache = GlyphCache::default();
    let mut frame = FrameData::default();
    let (ns, allocs) = best_of(GATE_ROUNDS, 16, || {
        build_frame_into(&fonts, list, &mut cache, TextGamma::default(), &mut frame);
        frame.vertices.len()
    });
    Ok(Sample {
        name: "frame_text_unicode",
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
    read_if_present(path)?
        .map(|text| parse_baseline(&text))
        .transpose()
}

/// Parse `name<ws>ns<ws>allocs`. Every entry needs its allocation count: a baseline
/// without one would leave allocations ungated, so it is malformed rather than read as
/// unmeasured.
fn parse_baseline(text: &str) -> Result<HashMap<String, Base>> {
    Ok(parse_rows::<2>(text, "perf")?
        .into_iter()
        .map(|(name, [ns, allocs])| (name, Base { ns, allocs }))
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::dev::perf::gate::*;

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
        use crate::dev::stream;

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

    #[cfg(feature = "perf-alloc")]
    #[test]
    fn program_status_never_allocates() {
        use crate::program_status::Records;

        // New ids past the limit, updates with decoded text, then a table filled again
        // after each way records leave.
        let fill: Vec<Vec<u8>> = (0..100)
            .map(|n| format!("state=done:id=job/{n}").into_bytes())
            .collect();
        let updates = [
            &b"state=blocked:id=job/99:msg=QXBwcm92ZSB0aGUgZGVwbG95Pw:title=RVUgV2VzdA"[..],
            b"state=working:id=job/98",
        ];
        let leave = [
            &b"state=clear:id=job/97"[..],
            b"state=clear:id=job",
            b"state=clear",
        ];

        alloc::reset();
        let mut records = Records::default();
        let refill = |records: &mut Records| {
            for report in &fill {
                records.apply(report);
            }
            for report in updates {
                records.apply(report);
            }
        };
        refill(&mut records);
        records.prompt_started();
        records.acknowledge();
        for report in leave {
            refill(&mut records);
            records.apply(report);
        }
        assert_eq!(alloc::snapshot().allocs, 0);
    }
}
