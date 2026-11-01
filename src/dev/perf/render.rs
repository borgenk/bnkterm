//! `--render-bench`: the render pipeline, headless (no GPU, no window). Each frame runs
//! the grid through the display list, the damage diff and the vertex build, timed per
//! stage, and a warmed frame's allocations are counted.

use crate::color::Theme;
use crate::dev::perf::{corpus, duration_ns, plain_frame, BENCH_COLS, BENCH_ROWS, RENDER_FONT_PX};
use crate::dev::{alloc, flags};
use crate::error::{Error, Result};
use crate::grid::Screen;
use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::render::display::{damage_into, DisplayListPool};
use crate::render::gpu::{build_frame_into, FrameData, GlyphCache, TextGamma};
use crate::term_render::{build_display_list_into, CellMetrics};
use crate::vt::Parser;
use std::hint::black_box;
use std::time::Instant;

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

/// The window's per-frame render state, reused across frames exactly as the live loop
/// reuses it: the double-buffered list and its string pool, the glyph cache, the frame
/// data, and the damage-diff scratch.
struct RenderRig<'a> {
    fonts: &'a Fonts,
    theme: Theme,
    metrics: CellMetrics,
    surface: (i32, i32),
    lists: DisplayListPool,
    cache: GlyphCache,
    frame: FrameData,
    damage: Vec<Rect>,
}

impl<'a> RenderRig<'a> {
    /// A rig for the bench grid at `metrics`, with nothing built yet.
    fn new(fonts: &'a Fonts, metrics: CellMetrics) -> Self {
        RenderRig {
            fonts,
            theme: Theme::default(),
            metrics,
            surface: (metrics.w * BENCH_COLS as i32, metrics.h * BENCH_ROWS as i32),
            lists: DisplayListPool::default(),
            cache: GlyphCache::default(),
            frame: FrameData::default(),
            damage: Vec::new(),
        }
    }

    /// Build `screen`'s display list and diff it against the frame on screen, returning
    /// how long each took in nanoseconds. An idle frame stops here, as the live loop does
    /// on empty damage.
    fn list_and_diff(&mut self, screen: &Screen) -> [u64; 2] {
        let t0 = Instant::now();
        {
            let (out, strings) = self.lists.begin();
            let inputs = plain_frame(screen, &self.theme, self.metrics, self.surface);
            build_display_list_into(out, strings, &inputs);
        }
        let t1 = Instant::now();
        let (w, h) = self.surface;
        damage_into(
            self.lists.front(),
            self.lists.back(),
            w,
            h,
            &mut self.damage,
        );
        let t2 = Instant::now();
        [duration_ns(t1 - t0), duration_ns(t2 - t1)]
    }

    /// One painted frame: the list and the diff, then the vertices, then the list becomes
    /// the frame on screen. Returns each stage's time in nanoseconds: list, diff, vertices.
    fn paint(&mut self, screen: &Screen) -> [u64; 3] {
        let [list, diff] = self.list_and_diff(screen);
        let t0 = Instant::now();
        let gamma = TextGamma::default();
        build_frame_into(
            self.fonts,
            self.lists.back(),
            &mut self.cache,
            gamma,
            &mut self.frame,
        );
        let vertices = duration_ns(t0.elapsed());
        black_box((&self.damage, &self.frame));
        self.lists.commit();
        [list, diff, vertices]
    }
}

/// Time the CPU render pipeline in isolation: `grid -> build_display_list -> damage
/// diff -> gpu::build_frame` (vertices), per frame, with no GPU and no window. This
/// gives the render side its own number so a fast parser can never hide a slow
/// renderer. It measures the CPU work every painted frame costs;
/// GPU execution and compositor present are a separate question that needs the
/// device. Built `--features perf-alloc` it also reports allocations per frame —
/// with the display list, its run strings, the vertex buffer, and the damage-diff
/// scratch all pooled across frames, a warmed changed frame allocates nothing.
pub fn render_bench(args: impl IntoIterator<Item = impl AsRef<str>>) -> Result<()> {
    let mut frames = RENDER_FRAMES;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            // The smoke test uses a tiny count.
            "--frames" => frames = flags::parsed("--frames", &mut args)?,
            other => return Err(flags::unknown(other)),
        }
    }
    if frames == 0 {
        return Err(Error::msg("--frames must be positive"));
    }

    let fonts = Fonts::new(&[RENDER_FONT_PX])
        .map_err(|e| Error::msg(format!("render-bench needs a monospace font: {e}")))?;
    let mut rig = RenderRig::new(&fonts, CellMetrics::from_fonts(&fonts, RENDER_FONT_PX));
    let mut screen = Screen::new(BENCH_COLS, BENCH_ROWS);
    let mut parser = Parser::new();
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
        rig.paint(&screen);
    }

    // One steady frame's allocation cost. With the list, its run strings, the frame
    // vertices, and the damage-diff scratch all reused, a warmed changed frame
    // allocates nothing.
    parser.advance_bytes(&mut screen, &feed);
    alloc::reset();
    rig.paint(&screen);
    let allocs = alloc::snapshot().allocs;

    let (mut build_ns, mut diff_ns, mut frame_ns, mut total_ns) =
        (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
    for _ in 0..frames {
        parser.advance_bytes(&mut screen, &feed); // scroll the grid (untimed)
        let [list, diff, vertices] = rig.paint(&screen);
        build_ns = build_ns.min(list);
        diff_ns = diff_ns.min(diff);
        frame_ns = frame_ns.min(vertices);
        total_ns = total_ns.min(list + diff + vertices);
    }
    let verts = rig.frame.vertices.len();

    // One-cell change: only the home cell differs from the last frame (a cursor
    // blink or a single keystroke), yet the list and every vertex are still rebuilt
    // in full. Alternating the char keeps each rep a real change. Of the three stages
    // only `damage` scales with the edit rather than the screen.
    let (mut edit_list, mut edit_diff, mut edit_frame) = (u64::MAX, u64::MAX, u64::MAX);
    for i in 0..frames {
        parser.advance_bytes(&mut screen, b"\x1b[1;1H");
        parser.advance_bytes(&mut screen, if i % 2 == 0 { b"X" } else { b"Y" });
        let [list, diff, vertices] = rig.paint(&screen);
        edit_list = edit_list.min(list);
        edit_diff = edit_diff.min(diff);
        edit_frame = edit_frame.min(vertices);
    }

    // Idle: nothing changed since the on-screen frame (no output, no blink). The list
    // is rebuilt and diffed in full, but the diff comes back empty and the live loop
    // stops there (`app::present::render_frame` early-outs on empty damage *before*
    // building vertices), so an idle frame costs list + diff and never pays the vertex
    // build. Neither list nor front is committed, exactly as that early-out does not:
    // `front` stays the on-screen frame and each pass refills `back` from the pool.
    let (mut idle_list, mut idle_diff, mut idle_rects) = (u64::MAX, u64::MAX, usize::MAX);
    for _ in 0..frames {
        let [list, diff] = rig.list_and_diff(&screen);
        idle_rects = rig.damage.len();
        black_box(&rig.damage);
        idle_list = idle_list.min(list);
        idle_diff = idle_diff.min(diff);
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
                    &plain_frame(&screen, &rig.theme, rig.metrics, rig.surface),
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

#[cfg(test)]
mod tests {
    use crate::dev::perf::render::*;

    #[test]
    fn render_bench_runs_without_error() {
        // Drives the real render pipeline headlessly for a couple of frames. Skip
        // where no monospace font is installed (headless CI), like the pty tests do.
        if crate::platform::freetype::Fonts::new(&[RENDER_FONT_PX]).is_err() {
            eprintln!("no monospace font; skipping render_bench smoke test");
            return;
        }
        assert!(render_bench(["--frames", "2"]).is_ok());
    }
}
