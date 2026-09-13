//! Procedural rasterization of the Box Drawing (U+2500..=U+257F), Block Elements
//! (U+2580..=U+259F), and Braille Patterns (U+2800..=U+28FF) blocks.
//!
//! Fonts cover these ranges unevenly (Consolas lacks the quadrants, so `▛` is
//! tofu), draw them against their own em box rather than the cell (hairline seams
//! between neighbours), and may draw braille's unraised dots as hollow rings.
//! Drawing them at the cell size avoids all three.
//!
//! [`coverage`] is a pure function of `(char, cell_w, cell_h)` returning a top-down
//! `cell_w * cell_h` R8 buffer, the same shape as a FreeType raster. The
//! [`crate::render::gpu`] batcher places it at `left = 0, top = baseline`, so it
//! fills the cell box exactly.
//!
//! ```text
//!   Box Drawing U+2500..=U+257F
//!     ├─ orthogonal arms   ┼ ├ ┏ ╾ …   up/down/left/right ∈ {none,light,heavy}
//!     │                                 → axis-aligned rectangles from edge to center
//!     ├─ dashes            ┄ ┆ ╌ …      → a straight arm broken into 2/3/4 segments
//!     ├─ double lines      ═ ╔ ╬ ╫ …    → two parallel rails per axis; each rail's
//!     │                                   middle segment is drawn only where no arm
//!     │                                   opens, which carves every junction
//!     ├─ rounded corners   ╭ ╮ ╯ ╰      → straight arms and a quarter circle (AA)
//!     └─ diagonals         ╱ ╲ ╳        → a supersampled corner-to-corner line (AA)
//!   Block Elements U+2580..=U+259F
//!     ├─ eighth blocks     ▀ ▁ ▌ ▐ …    → one rectangle, a fraction of the cell
//!     ├─ shades            ░ ▒ ▓        → a uniform partial coverage over the cell
//!     └─ quadrants         ▘ ▙ ▚ ▟ …    → a union of the four cell quarters
//!   Braille Patterns U+2800..=U+28FF
//!     └─ dot matrix        ⠁ ⠂ ⡀ ⣿ …    → one square per raised dot on a 2x4 grid
//! ```
//!
//! Everything but the arcs and diagonals is crisp 0/255 rectangles; those two
//! antialias because a sloped edge stair-steps without it. Complementary blocks
//! partition the cell exactly (`▀` and `▄` cover every row once), so they tile.

/// Whether `ch` is drawn here rather than by the font.
pub fn is_glyph(ch: char) -> bool {
    matches!(ch, '\u{2500}'..='\u{259F}' | '\u{2800}'..='\u{28FF}')
}

/// A `w * h` top-down R8 coverage buffer for `ch` at a `w`-by-`h` pixel cell.
/// `is_glyph(ch)` must hold; a char outside the range returns a blank cell. `w`
/// and `h` are the caller's whole-pixel cell box, so the glyph fills it edge to
/// edge and tiles with its neighbours.
pub fn coverage(ch: char, w: usize, h: usize) -> Vec<u8> {
    let mut cv = Canvas::new(w as i32, h as i32);
    if cv.w > 0 && cv.h > 0 {
        draw(&mut cv, ch);
    }
    cv.px
}

/// A single-channel coverage bitmap under construction: `w * h` bytes, row-major
/// top-down, `px[y * w + x]` the coverage at `(x, y)`.
struct Canvas {
    w: i32,
    h: i32,
    px: Vec<u8>,
}

impl Canvas {
    fn new(w: i32, h: i32) -> Self {
        let n = (w.max(0) * h.max(0)) as usize;
        Self {
            w,
            h,
            px: vec![0; n],
        }
    }

    /// Overwrite the half-open box `[x0, x1) x [y0, y1)` with `v`, clipped to the
    /// canvas. Overwrite (not blend) is what the double-line hollowing relies on:
    /// a later erase pass writes `0` straight over an earlier fill.
    fn rect(&mut self, x0: i32, x1: i32, y0: i32, y1: i32, v: u8) {
        let x0 = x0.max(0);
        let y0 = y0.max(0);
        let x1 = x1.min(self.w);
        let y1 = y1.min(self.h);
        let mut y = y0;
        while y < y1 {
            let row = (y * self.w) as usize;
            let mut x = x0;
            while x < x1 {
                self.px[row + x as usize] = v;
                x += 1;
            }
            y += 1;
        }
    }

    /// Raise the coverage at `(x, y)` to `v` (max-blend). The antialiased arcs and
    /// diagonals accumulate this way so a `╳`'s two strokes take the brighter of
    /// their overlapping subpixels instead of clobbering.
    fn blend(&mut self, x: i32, y: i32, v: u8) {
        if x < 0 || y < 0 || x >= self.w || y >= self.h {
            return;
        }
        let i = (y * self.w + x) as usize;
        if v > self.px[i] {
            self.px[i] = v;
        }
    }
}

/// The band of `t` pixels centered on `c`: `[c - t/2, c - t/2 + t)`. A 1px stroke
/// sits exactly on `c`; a 2px stroke straddles it biased up/left, the standard
/// choice so a light rule lands on a stable row across every cell.
fn band(c: i32, t: i32) -> (i32, i32) {
    let s = c - t / 2;
    (s, s + t)
}

/// `round(dim * num / den)` in integers: a fraction of a cell dimension in whole
/// pixels. `frac(h, 4, 8)` is the half-height boundary both `▀` and the quadrants
/// use, so blocks and quadrants share edges and tile.
fn frac(dim: i32, num: i32, den: i32) -> i32 {
    (dim * num + den / 2) / den
}

/// A light rule's thickness for this cell: about a ninth of the smaller cell
/// dimension, at least one pixel. Cells are taller than wide, so the width
/// dominates and a normal-size cell yields a crisp 1px rule, a 2x HiDPI cell 2px.
fn light(w: i32, h: i32) -> i32 {
    ((w.min(h) as f32 * 0.11).round() as i32).max(1)
}

/// A heavy rule's thickness: roughly a quarter of the smaller cell dimension, and
/// always at least one pixel bolder than [`light`] so the two read as distinct.
fn heavy(w: i32, h: i32) -> i32 {
    ((w.min(h) as f32 * 0.24).round() as i32).max(light(w, h) + 1)
}

/// Route `ch` to its family. Grouped by the ranges in the module header; a char
/// the ranges miss (there are a few unassigned points) leaves the cell blank.
fn draw(cv: &mut Canvas, ch: char) {
    let c = ch as u32;
    match c {
        0x2500..=0x257F => draw_box(cv, c),
        0x2580..=0x259F => draw_block(cv, c),
        0x2800..=0x28FF => draw_braille(cv, c),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Box drawing (U+2500..=U+257F)
// ---------------------------------------------------------------------------

/// One orthogonal arm's weight. The value is a thickness selector, not a pixel
/// count; [`arms`] turns it into [`light`]/[`heavy`] pixels for the cell.
#[derive(Clone, Copy, PartialEq, Eq)]
enum W {
    None,
    Light,
    Heavy,
}

impl W {
    fn px(self, w: i32, h: i32) -> i32 {
        match self {
            W::None => 0,
            W::Light => light(w, h),
            W::Heavy => heavy(w, h),
        }
    }
}

fn draw_box(cv: &mut Canvas, c: u32) {
    use W::{Heavy as H, Light as L, None as N};
    // [up, down, left, right] weights for the orthogonal-arm glyphs. Listed in
    // codepoint order; the verbosity is the point, each glyph names its arms.
    let arm = |u, d, l, r| Some((u, d, l, r));
    let spec = match c {
        0x2500 => arm(N, N, L, L), // ─
        0x2501 => arm(N, N, H, H), // ━
        0x2502 => arm(L, L, N, N), // │
        0x2503 => arm(H, H, N, N), // ┃
        0x250C => arm(N, L, N, L), // ┌
        0x250D => arm(N, L, N, H), // ┍
        0x250E => arm(N, H, N, L), // ┎
        0x250F => arm(N, H, N, H), // ┏
        0x2510 => arm(N, L, L, N), // ┐
        0x2511 => arm(N, L, H, N), // ┑
        0x2512 => arm(N, H, L, N), // ┒
        0x2513 => arm(N, H, H, N), // ┓
        0x2514 => arm(L, N, N, L), // └
        0x2515 => arm(L, N, N, H), // ┕
        0x2516 => arm(H, N, N, L), // ┖
        0x2517 => arm(H, N, N, H), // ┗
        0x2518 => arm(L, N, L, N), // ┘
        0x2519 => arm(L, N, H, N), // ┙
        0x251A => arm(H, N, L, N), // ┚
        0x251B => arm(H, N, H, N), // ┛
        0x251C => arm(L, L, N, L), // ├
        0x251D => arm(L, L, N, H), // ┝
        0x251E => arm(H, L, N, L), // ┞
        0x251F => arm(L, H, N, L), // ┟
        0x2520 => arm(H, H, N, L), // ┠
        0x2521 => arm(H, L, N, H), // ┡
        0x2522 => arm(L, H, N, H), // ┢
        0x2523 => arm(H, H, N, H), // ┣
        0x2524 => arm(L, L, L, N), // ┤
        0x2525 => arm(L, L, H, N), // ┥
        0x2526 => arm(H, L, L, N), // ┦
        0x2527 => arm(L, H, L, N), // ┧
        0x2528 => arm(H, H, L, N), // ┨
        0x2529 => arm(H, L, H, N), // ┩
        0x252A => arm(L, H, H, N), // ┪
        0x252B => arm(H, H, H, N), // ┫
        0x252C => arm(N, L, L, L), // ┬
        0x252D => arm(N, L, H, L), // ┭
        0x252E => arm(N, L, L, H), // ┮
        0x252F => arm(N, L, H, H), // ┯
        0x2530 => arm(N, H, L, L), // ┰
        0x2531 => arm(N, H, H, L), // ┱
        0x2532 => arm(N, H, L, H), // ┲
        0x2533 => arm(N, H, H, H), // ┳
        0x2534 => arm(L, N, L, L), // ┴
        0x2535 => arm(L, N, H, L), // ┵
        0x2536 => arm(L, N, L, H), // ┶
        0x2537 => arm(L, N, H, H), // ┷
        0x2538 => arm(H, N, L, L), // ┸
        0x2539 => arm(H, N, H, L), // ┹
        0x253A => arm(H, N, L, H), // ┺
        0x253B => arm(H, N, H, H), // ┻
        0x253C => arm(L, L, L, L), // ┼
        0x253D => arm(L, L, H, L), // ┽
        0x253E => arm(L, L, L, H), // ┾
        0x253F => arm(L, L, H, H), // ┿
        0x2540 => arm(H, L, L, L), // ╀
        0x2541 => arm(L, H, L, L), // ╁
        0x2542 => arm(H, H, L, L), // ╂
        0x2543 => arm(H, L, H, L), // ╃
        0x2544 => arm(H, L, L, H), // ╄
        0x2545 => arm(L, H, H, L), // ╅
        0x2546 => arm(L, H, L, H), // ╆
        0x2547 => arm(H, L, H, H), // ╇
        0x2548 => arm(L, H, H, H), // ╈
        0x2549 => arm(H, H, H, L), // ╉
        0x254A => arm(H, H, L, H), // ╊
        0x254B => arm(H, H, H, H), // ╋
        0x2574 => arm(N, N, L, N), // ╴
        0x2575 => arm(L, N, N, N), // ╵
        0x2576 => arm(N, N, N, L), // ╶
        0x2577 => arm(N, L, N, N), // ╷
        0x2578 => arm(N, N, H, N), // ╸
        0x2579 => arm(H, N, N, N), // ╹
        0x257A => arm(N, N, N, H), // ╺
        0x257B => arm(N, H, N, N), // ╻
        0x257C => arm(N, N, L, H), // ╼
        0x257D => arm(L, H, N, N), // ╽
        0x257E => arm(N, N, H, L), // ╾
        0x257F => arm(H, L, N, N), // ╿
        _ => None,
    };
    if let Some((u, d, l, r)) = spec {
        arms(cv, u, d, l, r);
        return;
    }
    // Dashes: a straight light/heavy line broken into 2, 3, or 4 segments.
    if let Some((horizontal, weight, count)) = dash_spec(c) {
        dashes(cv, horizontal, weight.px(cv.w, cv.h), count);
        return;
    }
    // Double and mixed single/double lines, then the curves.
    match c {
        0x2550..=0x256C => draw_double(cv, c),
        0x256D => arc(cv, cv.w, cv.h), // ╭ down+right
        0x256E => arc(cv, 0, cv.h),    // ╮ down+left
        0x256F => arc(cv, 0, 0),       // ╯ up+left
        0x2570 => arc(cv, cv.w, 0),    // ╰ up+right
        0x2571 => diagonal(cv, true),  // ╱
        0x2572 => diagonal(cv, false), // ╲
        0x2573 => {
            diagonal(cv, true);
            diagonal(cv, false);
        } // ╳
        _ => {}
    }
}

/// Draw the four orthogonal arms of a box-drawing glyph. Each arm runs from its
/// cell edge to the center, so a left and right arm of equal weight fuse into one
/// rule and any two perpendicular arms meet cleanly at the middle. Weights are
/// independent: `╼` (light left, heavy right) thickens across the center.
fn arms(cv: &mut Canvas, u: W, d: W, l: W, r: W) {
    let (w, h) = (cv.w, cv.h);
    let (cx, cy) = (w / 2, h / 2);
    let (tu, td, tl, tr) = (u.px(w, h), d.px(w, h), l.px(w, h), r.px(w, h));
    if tl > 0 {
        let (y0, y1) = band(cy, tl);
        cv.rect(0, cx + 1, y0, y1, 255);
    }
    if tr > 0 {
        let (y0, y1) = band(cy, tr);
        cv.rect(cx, w, y0, y1, 255);
    }
    if tu > 0 {
        let (x0, x1) = band(cx, tu);
        cv.rect(x0, x1, 0, cy + 1, 255);
    }
    if td > 0 {
        let (x0, x1) = band(cx, td);
        cv.rect(x0, x1, cy, h, 255);
    }
}

/// `(horizontal, weight, dash_count)` for the dash glyphs, or `None`. Triple and
/// quadruple dashes sit non-contiguously in the block, so the mapping is explicit.
fn dash_spec(c: u32) -> Option<(bool, W, i32)> {
    Some(match c {
        0x2504 => (true, W::Light, 3),  // ┄
        0x2505 => (true, W::Heavy, 3),  // ┅
        0x2506 => (false, W::Light, 3), // ┆
        0x2507 => (false, W::Heavy, 3), // ┇
        0x2508 => (true, W::Light, 4),  // ┈
        0x2509 => (true, W::Heavy, 4),  // ┉
        0x250A => (false, W::Light, 4), // ┊
        0x250B => (false, W::Heavy, 4), // ┋
        0x254C => (true, W::Light, 2),  // ╌
        0x254D => (true, W::Heavy, 2),  // ╍
        0x254E => (false, W::Light, 2), // ╎
        0x254F => (false, W::Heavy, 2), // ╏
        _ => return None,
    })
}

/// A dashed rule: the center line split into `count` inked segments with a gap in
/// each, so it reads as a dashed line at any cell size. Dashes are not meant to
/// meet a neighbour, so unlike a solid rule they need not reach the cell edges.
fn dashes(cv: &mut Canvas, horizontal: bool, t: i32, count: i32) {
    if t <= 0 || count <= 0 {
        return;
    }
    let (w, h) = (cv.w, cv.h);
    let span = if horizontal { w } else { h };
    let (b0, b1) = band(if horizontal { h / 2 } else { w / 2 }, t);
    for k in 0..count {
        let s0 = k * span / count;
        let s1 = (k + 1) * span / count;
        let gap = (((s1 - s0) as f32) * 0.33).round() as i32;
        let (a, b) = (s0 + gap / 2, s1 - gap / 2);
        if horizontal {
            cv.rect(a, b, b0, b1, 255);
        } else {
            cv.rect(b0, b1, a, b, 255);
        }
    }
}

/// Draw a double or mixed single/double junction (U+2550..=U+256C).
///
/// A double line is two parallel rails a gap apart. The junction is where the
/// rails cross, and getting its openings right is the whole difficulty: a corner
/// keeps its outer rails long and its inner rails short; a tee opens one rail to
/// let the branch through; the cross opens all four to leave a hole. Rather than
/// special-case each, place the four rails (top/bottom rows, left/right columns)
/// and give each three candidate segments, of which the middle one is drawn only
/// when the perpendicular double arm is *absent*:
///
/// ```text
///   left/right rails, as rows RT & RB          A rail's three segments:
///       ┌────┬────┐   RT                        [edge→node][node→node][node→edge]
///   top/bottom     │                            the middle [node→node] segment
///   rails, as   ┌──┼──┐                          closes the hole on that side, so
///   columns     │  │  │                          it is drawn only when no arm
///   CL & CR     └──┴──┘   RB                      opens that way (u/d/l/r != 2)
/// ```
///
/// That single rule makes corners, tees, and the cross all fall out: the cross
/// (`╬`, every arm present) drops all four middle segments and leaves the central
/// hole; a corner (`╔`) keeps the two middle segments on its closed sides, fusing
/// its outer rails. Single arms in a mixed glyph (e.g. `╞`) are then laid down as
/// one center rule, extended to bridge the double's rails.
fn draw_double(cv: &mut Canvas, c: u32) {
    let idx = (c - 0x2550) as usize;
    // [up, down, left, right], each 0 = none, 1 = single, 2 = double.
    let [u, d, l, r] = DOUBLE_ARMS[idx];
    let (w, h) = (cv.w, cv.h);
    let (cx, cy) = (w / 2, h / 2);
    let t = light(w, h);
    // Rails sit a gap apart, the gap equal to the rail width so a double reads as
    // rail-gap-rail across three light widths.
    let off = t;
    let (cl0, cl1) = band(cx - off, t); // left rail columns
    let (cr0, cr1) = band(cx + off, t); // right rail columns
    let (rt0, rt1) = band(cy - off, t); // top rail rows
    let (rb0, rb1) = band(cy + off, t); // bottom rail rows
    let (u2, d2, l2, r2) = (u == 2, d == 2, l == 2, r == 2);

    // Horizontal rails exist when either horizontal arm is double.
    if l2 || r2 {
        for &(y0, y1, open_absent) in &[(rt0, rt1, u2), (rb0, rb1, d2)] {
            if l2 {
                cv.rect(0, cl1, y0, y1, 255); // stub to the left edge
            }
            if !open_absent {
                cv.rect(cl0, cr1, y0, y1, 255); // middle: closes the hole this side
            }
            if r2 {
                cv.rect(cr0, w, y0, y1, 255); // stub to the right edge
            }
        }
    }
    // Vertical rails exist when either vertical arm is double.
    if u2 || d2 {
        for &(x0, x1, open_absent) in &[(cl0, cl1, l2), (cr0, cr1, r2)] {
            if u2 {
                cv.rect(x0, x1, 0, rt1, 255); // stub to the top edge
            }
            if !open_absent {
                cv.rect(x0, x1, rt0, rb1, 255); // middle: closes the hole this side
            }
            if d2 {
                cv.rect(x0, x1, rb0, h, 255); // stub to the bottom edge
            }
        }
    }

    // Single arms as one center rule, reaching a rail-span past center so they
    // meet both rails of a crossing double line.
    let reach = off + t;
    let (vx0, vx1) = band(cx, t);
    let (hy0, hy1) = band(cy, t);
    if u == 1 {
        cv.rect(vx0, vx1, 0, cy + reach + 1, 255);
    }
    if d == 1 {
        cv.rect(vx0, vx1, cy - reach, h, 255);
    }
    if l == 1 {
        cv.rect(0, cx + reach + 1, hy0, hy1, 255);
    }
    if r == 1 {
        cv.rect(cx - reach, w, hy0, hy1, 255);
    }
}

/// Arm weights for U+2550..=U+256C in `[up, down, left, right]`, each
/// `0` none / `1` single / `2` double. Indexed by `codepoint - 0x2550`.
const DOUBLE_ARMS: [[u8; 4]; 29] = [
    [0, 0, 2, 2], // 2550 ═
    [2, 2, 0, 0], // 2551 ║
    [0, 1, 0, 2], // 2552 ╒
    [0, 2, 0, 1], // 2553 ╓
    [0, 2, 0, 2], // 2554 ╔
    [0, 1, 2, 0], // 2555 ╕
    [0, 2, 1, 0], // 2556 ╖
    [0, 2, 2, 0], // 2557 ╗
    [1, 0, 0, 2], // 2558 ╘
    [2, 0, 0, 1], // 2559 ╙
    [2, 0, 0, 2], // 255A ╚
    [1, 0, 2, 0], // 255B ╛
    [2, 0, 1, 0], // 255C ╜
    [2, 0, 2, 0], // 255D ╝
    [1, 1, 0, 2], // 255E ╞
    [2, 2, 0, 1], // 255F ╟
    [2, 2, 0, 2], // 2560 ╠
    [1, 1, 2, 0], // 2561 ╡
    [2, 2, 1, 0], // 2562 ╢
    [2, 2, 2, 0], // 2563 ╣
    [0, 1, 2, 2], // 2564 ╤
    [0, 2, 1, 1], // 2565 ╥
    [0, 2, 2, 2], // 2566 ╦
    [1, 0, 2, 2], // 2567 ╧
    [2, 0, 1, 1], // 2568 ╨
    [2, 0, 2, 2], // 2569 ╩
    [1, 1, 2, 2], // 256A ╪
    [2, 2, 1, 1], // 256B ╫
    [2, 2, 2, 2], // 256C ╬
];

/// Join two straight arms with a constant-width quarter circle. The arm centers
/// come from the same pixel bands as the straight neighbours, including the
/// half-pixel offset of odd stroke widths. Each arm keeps a full pixel of straight
/// coverage at the cell edge so antialiasing the bend cannot dim the seam.
/// Cells too small for the bend use a square junction.
fn arc(cv: &mut Canvas, corner_x: i32, corner_y: i32) {
    let (w, h) = (cv.w, cv.h);
    let t = light(w, h);
    let (x0, x1) = band(w / 2, t);
    let (y0, y1) = band(h / 2, t);
    let cx = (x0 + x1) as f32 / 2.0;
    let cy = (y0 + y1) as f32 / 2.0;
    // Share a radius across orientations even when the pixel band is off-center.
    let r = (cx.min(w as f32 - cx).min(cy).min(h as f32 - cy) - 1.0).max(0.0);
    let hw = t as f32 / 2.0;
    if r <= hw {
        let u = if corner_y == 0 { W::Light } else { W::None };
        let d = if corner_y == 0 { W::None } else { W::Light };
        let l = if corner_x == 0 { W::Light } else { W::None };
        let right = if corner_x == 0 { W::None } else { W::Light };
        arms(cv, u, d, l, right);
        return;
    }
    let direction_x = if corner_x == 0 { -1.0 } else { 1.0 };
    let direction_y = if corner_y == 0 { -1.0 } else { 1.0 };
    const SS: i32 = 4;
    for y in 0..h {
        for x in 0..w {
            let mut hits = 0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f32 + (sx as f32 + 0.5) / SS as f32;
                    let py = y as f32 + (sy as f32 + 0.5) / SS as f32;
                    let dx = r - (px - cx) * direction_x;
                    let dy = r - (py - cy) * direction_y;
                    let distance = if dx < 0.0 {
                        (py - cy).abs()
                    } else if dy < 0.0 {
                        (px - cx).abs()
                    } else {
                        ((dx * dx + dy * dy).sqrt() - r).abs()
                    };
                    if distance <= hw {
                        hits += 1;
                    }
                }
            }
            if hits > 0 {
                let v = (hits * 255 / (SS * SS)) as u8;
                cv.blend(x, y, v);
            }
        }
    }
}

/// A corner-to-corner diagonal stroke, `rising` is bottom-left→top-right (`╱`),
/// else top-left→bottom-right (`╲`). Light thickness, supersampled for a smooth
/// edge. `╳` is the two drawn over each other (max-blended, so the crossing does
/// not double).
fn diagonal(cv: &mut Canvas, rising: bool) {
    let (w, h) = (cv.w, cv.h);
    let (wf, hf) = (w as f32, h as f32);
    // Segment endpoints A→B in pixel space.
    let (ax, ay, bx, by) = if rising {
        (0.0, hf, wf, 0.0)
    } else {
        (0.0, 0.0, wf, hf)
    };
    let (ex, ey) = (bx - ax, by - ay);
    let len2 = ex * ex + ey * ey;
    let half = light(w, h) as f32 / 2.0;
    const SS: i32 = 4;
    for y in 0..h {
        for x in 0..w {
            let mut hits = 0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f32 + (sx as f32 + 0.5) / SS as f32;
                    let py = y as f32 + (sy as f32 + 0.5) / SS as f32;
                    // Distance from the point to the (infinite, len2>0) line
                    // through A→B; the segment spans the whole cell diagonal so
                    // clamping to endpoints is unnecessary.
                    let t = ((px - ax) * ex + (py - ay) * ey) / len2;
                    let projx = ax + t * ex;
                    let projy = ay + t * ey;
                    let dd = (px - projx).powi(2) + (py - projy).powi(2);
                    if dd <= half * half {
                        hits += 1;
                    }
                }
            }
            if hits > 0 {
                let v = (hits * 255 / (SS * SS)) as u8;
                cv.blend(x, y, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block elements (U+2580..=U+259F)
// ---------------------------------------------------------------------------

fn draw_block(cv: &mut Canvas, c: u32) {
    let (w, h) = (cv.w, cv.h);
    match c {
        0x2580 => cv.rect(0, w, 0, frac(h, 1, 2), 255), // ▀ upper half
        0x2581..=0x2587 => {
            // ▁▂▃▄▅▆▇ lower one-eighth .. seven-eighths
            let n = (c - 0x2580) as i32; // 1..=7
            cv.rect(0, w, h - frac(h, n, 8), h, 255);
        }
        0x2588 => cv.rect(0, w, 0, h, 255), // █ full
        0x2589..=0x258F => {
            // ▉▊▋▌▍▎▏ left seven-eighths .. one-eighth
            let n = 8 - (c - 0x2588) as i32; // 7..=1
            cv.rect(0, frac(w, n, 8), 0, h, 255);
        }
        0x2590 => cv.rect(w - frac(w, 1, 2), w, 0, h, 255), // ▐ right half
        0x2591 => cv.rect(0, w, 0, h, 64),                  // ░ light shade
        0x2592 => cv.rect(0, w, 0, h, 128),                 // ▒ medium shade
        0x2593 => cv.rect(0, w, 0, h, 192),                 // ▓ dark shade
        0x2594 => cv.rect(0, w, 0, frac(h, 1, 8), 255),     // ▔ upper one-eighth
        0x2595 => cv.rect(w - frac(w, 1, 8), w, 0, h, 255), // ▕ right one-eighth
        0x2596..=0x259F => quadrants(cv, c),
        _ => {}
    }
}

/// The quadrant block elements (U+2596..=U+259F): each is a union of the four cell
/// quarters. Quarter edges use the same half-cell fractions the half blocks do,
/// so a quadrant abuts a half block seamlessly. Bit order below: UL, UR, LL, LR.
fn quadrants(cv: &mut Canvas, c: u32) {
    let (w, h) = (cv.w, cv.h);
    let hx = frac(w, 1, 2);
    let hy = frac(h, 1, 2);
    let mask = match c {
        0x2596 => 0b0010, // ▖ lower left
        0x2597 => 0b0001, // ▗ lower right
        0x2598 => 0b1000, // ▘ upper left
        0x2599 => 0b1011, // ▙ upper left + lower left + lower right
        0x259A => 0b1001, // ▚ upper left + lower right
        0x259B => 0b1110, // ▛ upper left + upper right + lower left
        0x259C => 0b1101, // ▜ upper left + upper right + lower right
        0x259D => 0b0100, // ▝ upper right
        0x259E => 0b0110, // ▞ upper right + lower left
        0x259F => 0b0111, // ▟ upper right + lower left + lower right
        _ => 0,
    };
    if mask & 0b1000 != 0 {
        cv.rect(0, hx, 0, hy, 255); // UL
    }
    if mask & 0b0100 != 0 {
        cv.rect(hx, w, 0, hy, 255); // UR
    }
    if mask & 0b0010 != 0 {
        cv.rect(0, hx, hy, h, 255); // LL
    }
    if mask & 0b0001 != 0 {
        cv.rect(hx, w, hy, h, 255); // LR
    }
}

// ---------------------------------------------------------------------------
// Braille patterns (U+2800..=U+28FF)
// ---------------------------------------------------------------------------

/// `(bit, column, row)` of each dot on the 2x4 grid. Dots 7 and 8 (`0x40`, `0x80`)
/// are the bottom row.
const BRAILLE_DOTS: [(u32, i32, i32); 8] = [
    (0x01, 0, 0),
    (0x02, 0, 1),
    (0x04, 0, 2),
    (0x08, 1, 0),
    (0x10, 1, 1),
    (0x20, 1, 2),
    (0x40, 0, 3),
    (0x80, 1, 3),
];

/// One square per raised dot, nothing else. The codepoint's low byte is the dot mask.
fn draw_braille(cv: &mut Canvas, c: u32) {
    let grid = BrailleGrid::new(cv.w, cv.h);
    let raised = c & 0xFF;
    for (bit, col, row) in BRAILLE_DOTS {
        if raised & bit != 0 {
            let (x, y) = grid.dot_at(col, row);
            cv.rect(x, x + grid.dot, y, y + grid.dot, 255);
        }
    }
}

/// Dot size and spacing for one braille cell: an even split of the cell, then the
/// pixels lost to rounding go, in order, to a dot of at least 1px, a margin of at
/// least 1px (so adjacent cells' dots never touch), a wider gap, wider margins, and a
/// bigger dot.
struct BrailleGrid {
    dot: i32,
    x_margin: i32,
    x_gap: i32,
    y_margin: i32,
    y_gap: i32,
}

impl BrailleGrid {
    fn new(w: i32, h: i32) -> Self {
        let mut dot = (w / 4).min(h / 8);
        let mut x_gap = w / 4;
        let mut y_gap = h / 8;
        let mut x_margin = x_gap / 2;
        let mut y_margin = y_gap / 2;
        let mut x_left = w - 2 * x_margin - 2 * dot - x_gap;
        let mut y_left = h - 2 * y_margin - 4 * dot - 3 * y_gap;
        if dot == 0 && x_left >= 2 && y_left >= 4 {
            dot = 1;
            x_left -= 2;
            y_left -= 4;
        }
        if x_margin == 0 && x_left >= 2 {
            x_margin = 1;
            x_left -= 2;
        }
        if y_margin == 0 && y_left >= 2 {
            y_margin = 1;
            y_left -= 2;
        }
        if x_left >= 1 {
            x_gap += 1;
            x_left -= 1;
        }
        if y_left >= 3 {
            y_gap += 1;
            y_left -= 3;
        }
        if x_left >= 2 {
            x_margin += 1;
            x_left -= 2;
        }
        if y_left >= 2 {
            y_margin += 1;
            y_left -= 2;
        }
        if x_left >= 2 && y_left >= 4 {
            dot += 1;
        }
        Self {
            dot,
            x_margin,
            x_gap,
            y_margin,
            y_gap,
        }
    }

    /// The top-left pixel of the dot in `col` (0 or 1) and `row` (0 to 3).
    fn dot_at(&self, col: i32, row: i32) -> (i32, i32) {
        (
            self.x_margin + col * (self.dot + self.x_gap),
            self.y_margin + row * (self.dot + self.y_gap),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::render::boxdraw::*;

    /// Coverage at `(x, y)` in a `w`-wide buffer.
    fn at(buf: &[u8], w: usize, x: usize, y: usize) -> u8 {
        buf[y * w + x]
    }

    /// The number of contiguous inked runs in a line of coverage. A double rule
    /// shows as two runs (its rails) regardless of how many pixels wide each is.
    fn runs(inked: impl Iterator<Item = bool>) -> usize {
        let mut n = 0;
        let mut prev = false;
        for v in inked {
            if v && !prev {
                n += 1;
            }
            prev = v;
        }
        n
    }

    #[test]
    fn is_glyph_covers_exactly_the_three_blocks() {
        assert!(is_glyph('\u{2500}')); // first box drawing
        assert!(is_glyph('\u{259F}')); // last block element
        assert!(is_glyph('█'));
        assert!(is_glyph('▛'));
        assert!(is_glyph('\u{2800}')); // blank braille pattern
        assert!(is_glyph('\u{28FF}')); // all eight dots
        assert!(!is_glyph('\u{24FF}'));
        assert!(!is_glyph('\u{25A0}')); // black square, just past the range
        assert!(!is_glyph('\u{27FF}')); // just before braille
        assert!(!is_glyph('\u{2900}')); // just after it
        assert!(!is_glyph('A'));
    }

    #[test]
    fn coverage_is_exactly_cell_sized() {
        for ch in ['█', '─', '╬', '╭', '╱', '▟', '░', '⠂', '⣿'] {
            let buf = coverage(ch, 9, 20);
            assert_eq!(buf.len(), 9 * 20, "{ch} must fill a 9x20 cell");
        }
    }

    #[test]
    fn a_degenerate_cell_never_panics() {
        for ch in ['█', '┼', '╬', '╭', '╳', '▚', '⣿'] {
            assert!(coverage(ch, 0, 0).is_empty());
            assert_eq!(coverage(ch, 1, 1).len(), 1);
        }
    }

    #[test]
    fn full_block_inks_every_pixel() {
        let buf = coverage('█', 10, 18);
        assert!(buf.iter().all(|&v| v == 255));
    }

    #[test]
    fn left_and_right_half_partition_the_cell() {
        let (w, h) = (10, 18);
        let left = coverage('▌', w, h);
        let right = coverage('▐', w, h);
        // Every pixel is inked by exactly one of the two halves: no gap, no
        // overlap, so `▌▐` reconstructs a full block.
        for i in 0..w * h {
            let l = left[i] == 255;
            let r = right[i] == 255;
            assert!(l ^ r, "pixel {i} must belong to exactly one half");
        }
    }

    #[test]
    fn upper_and_lower_half_partition_the_cell() {
        let (w, h) = (10, 18);
        let up = coverage('▀', w, h);
        let down = coverage('▄', w, h);
        for i in 0..w * h {
            assert!(
                (up[i] == 255) ^ (down[i] == 255),
                "pixel {i} exactly one half"
            );
        }
    }

    #[test]
    fn left_eighths_increase_monotonically() {
        let (w, h) = (16, 18);
        let ink = |ch| coverage(ch, w, h).iter().filter(|&&v| v == 255).count();
        // ▏ (1/8) < ▎ < ▍ < ▌ (1/2) < ▊ < ▉ (7/8) < █ (full)
        let widths = ['▏', '▎', '▍', '▌', '▊', '▉', '█'].map(ink);
        for pair in widths.windows(2) {
            assert!(pair[0] < pair[1], "eighth blocks must grow: {widths:?}");
        }
    }

    #[test]
    fn shades_are_uniform_and_ordered() {
        let (w, h) = (8, 12);
        let light = coverage('░', w, h);
        let medium = coverage('▒', w, h);
        let dark = coverage('▓', w, h);
        assert!(light.iter().all(|&v| v == light[0]), "░ is uniform");
        assert!(
            light[0] < medium[0] && medium[0] < dark[0],
            "shade ordering"
        );
        assert!(light[0] > 0 && dark[0] < 255, "shades are partial coverage");
    }

    /// The four glyphs of the Claude logo the font was missing (`▛▜▝▘`) plus the
    /// ones it had (`▐█▌`): every one must now produce ink, none a blank tofu.
    #[test]
    fn every_logo_glyph_inks() {
        for ch in ['▐', '▛', '█', '▜', '▌', '▝', '▘'] {
            let buf = coverage(ch, 9, 20);
            assert!(buf.iter().any(|&v| v > 0), "{ch} must not be blank");
        }
    }

    #[test]
    fn upper_left_quadrant_fills_one_corner() {
        let (w, h) = (10, 16);
        let buf = coverage('▘', w, h); // upper-left quadrant
        assert_eq!(at(&buf, w, 0, 0), 255, "top-left inked");
        assert_eq!(at(&buf, w, w - 1, 0), 0, "top-right blank");
        assert_eq!(at(&buf, w, 0, h - 1), 0, "bottom-left blank");
        assert_eq!(at(&buf, w, w - 1, h - 1), 0, "bottom-right blank");
    }

    #[test]
    fn quadrants_of_a_cell_partition_it() {
        // ▘ (UL) + ▝ (UR) + ▖ (LL) + ▗ (LR) must tile to a full block.
        let (w, h) = (11, 15); // odd dims to exercise the rounding
        let mut acc = vec![0u8; w * h];
        for ch in ['▘', '▝', '▖', '▗'] {
            let buf = coverage(ch, w, h);
            for i in 0..w * h {
                if buf[i] == 255 {
                    assert_eq!(acc[i], 0, "quadrants must not overlap at {i}");
                    acc[i] = 255;
                }
            }
        }
        assert!(
            acc.iter().all(|&v| v == 255),
            "quadrants must cover every pixel"
        );
    }

    #[test]
    fn horizontal_rule_reaches_both_edges_on_the_center_row() {
        let (w, h) = (12, 20);
        let buf = coverage('─', w, h);
        let cy = h / 2;
        assert_eq!(at(&buf, w, 0, cy), 255, "reaches the left edge");
        assert_eq!(at(&buf, w, w - 1, cy), 255, "reaches the right edge");
        // A neighbouring `─` shares the same row, so the rule is continuous.
        assert_eq!(at(&buf, w, w / 2, 0), 0, "no ink at the top");
    }

    #[test]
    fn vertical_rule_reaches_top_and_bottom() {
        let (w, h) = (12, 20);
        let buf = coverage('│', w, h);
        let cx = w / 2;
        assert_eq!(at(&buf, w, cx, 0), 255, "reaches the top");
        assert_eq!(at(&buf, w, cx, h - 1), 255, "reaches the bottom");
    }

    #[test]
    fn cross_has_all_four_arms() {
        let (w, h) = (14, 20);
        let buf = coverage('┼', w, h);
        let (cx, cy) = (w / 2, h / 2);
        assert_eq!(at(&buf, w, 0, cy), 255, "left arm");
        assert_eq!(at(&buf, w, w - 1, cy), 255, "right arm");
        assert_eq!(at(&buf, w, cx, 0), 255, "up arm");
        assert_eq!(at(&buf, w, cx, h - 1), 255, "down arm");
    }

    #[test]
    fn corner_reaches_only_its_two_edges() {
        // ┌ (down + right): ink at the right and bottom edges, none left or top.
        let (w, h) = (14, 20);
        let buf = coverage('┌', w, h);
        let (cx, cy) = (w / 2, h / 2);
        assert_eq!(at(&buf, w, w - 1, cy), 255, "right arm present");
        assert_eq!(at(&buf, w, cx, h - 1), 255, "down arm present");
        assert_eq!(at(&buf, w, 0, cy), 0, "no left arm");
        assert_eq!(at(&buf, w, cx, 0), 0, "no up arm");
    }

    #[test]
    fn heavy_rule_is_thicker_than_light() {
        let (w, h) = (16, 24);
        let ink = |ch| coverage(ch, w, h).iter().filter(|&&v| v == 255).count();
        assert!(ink('━') > ink('─'), "heavy horizontal is bolder");
        assert!(ink('┃') > ink('│'), "heavy vertical is bolder");
    }

    #[test]
    fn double_horizontal_has_two_separated_rails() {
        // `═` down a column must show two inked rails with a clear gap between.
        for (w, h) in [(12, 24), (18, 36)] {
            let buf = coverage('═', w, h);
            let x = w / 2;
            let rails = runs((0..h).map(|y| at(&buf, w, x, y) == 255));
            assert_eq!(rails, 2, "two rails at column {x} of a {w}x{h} ═");
            assert_eq!(at(&buf, w, x, h / 2), 0, "the gap sits on the center row");
        }
    }

    #[test]
    fn double_vertical_has_two_separated_rails() {
        for (w, h) in [(16, 20), (24, 30)] {
            let buf = coverage('║', w, h);
            let y = h / 2;
            let rails = runs((0..w).map(|x| at(&buf, w, x, y) == 255));
            assert_eq!(rails, 2, "two rails across row {y} of a {w}x{h} ║");
            assert_eq!(
                at(&buf, w, w / 2, y),
                0,
                "the gap sits on the center column"
            );
        }
    }

    #[test]
    fn double_cross_keeps_a_hole_in_the_middle() {
        // The defining feature of `╬`: the exact center is a hole, not ink.
        let (w, h) = (16, 20);
        let buf = coverage('╬', w, h);
        assert_eq!(at(&buf, w, w / 2, h / 2), 0, "center of ╬ is open");
        // But every edge is still reached (all four double arms present); the ink
        // on an edge sits on the rails, not the center row/column.
        assert!((0..h).any(|y| at(&buf, w, 0, y) != 0), "left edge inked");
        assert!(
            (0..h).any(|y| at(&buf, w, w - 1, y) != 0),
            "right edge inked"
        );
        assert!((0..w).any(|x| at(&buf, w, x, 0) != 0), "top edge inked");
        assert!(
            (0..w).any(|x| at(&buf, w, x, h - 1) != 0),
            "bottom edge inked"
        );
        // Two rails cross each edge.
        assert_eq!(
            runs((0..w).map(|x| at(&buf, w, x, 0) != 0)),
            2,
            "two rails top"
        );
    }

    #[test]
    fn double_corner_joins_outer_rails_and_opens_the_others() {
        // ╔ (double down + right): reaches the right and bottom edges, and the top
        // and left edges stay completely clear (no up or left arm).
        let (w, h) = (16, 20);
        let buf = coverage('╔', w, h);
        assert!(
            (0..h).any(|y| at(&buf, w, w - 1, y) != 0),
            "right edge inked"
        );
        assert!(
            (0..w).any(|x| at(&buf, w, x, h - 1) != 0),
            "bottom edge inked"
        );
        assert!(
            (0..w).all(|x| at(&buf, w, x, 0) == 0),
            "top edge fully clear"
        );
        assert!(
            (0..h).all(|y| at(&buf, w, 0, y) == 0),
            "left edge fully clear"
        );
        // The outer rails fuse into one continuous corner: the outer-corner pixel
        // (top rail row, left rail column) is inked.
        let off = light(w as i32, h as i32) as usize;
        let (cx, cy) = (w / 2, h / 2);
        assert_ne!(
            at(&buf, w, cx - off, cy - off),
            0,
            "outer corner of ╔ is fused"
        );
    }

    #[test]
    fn tiny_rounded_corners_keep_the_square_junction() {
        for w in 1..=3 {
            for h in 1..=3 {
                for (rounded, square) in [('╭', '┌'), ('╮', '┐'), ('╯', '┘'), ('╰', '└')]
                {
                    assert_eq!(
                        coverage(rounded, w, h),
                        coverage(square, w, h),
                        "{rounded} keeps its arms in {w}x{h}"
                    );
                }
            }
        }
    }

    #[test]
    fn rounded_corners_match_their_straight_neighbours() {
        // Compare the complete seam, including clear pixels: checking only the
        // brightest pixel would miss a displaced or wider antialiased stroke.
        for w in 6..=32 {
            for h in 6..=48 {
                let horizontal = coverage('─', w, h);
                let vertical = coverage('│', w, h);
                for (ch, right, down) in [
                    ('╭', true, true),
                    ('╮', false, true),
                    ('╯', false, false),
                    ('╰', true, false),
                ] {
                    let corner = coverage(ch, w, h);
                    let x = if right { w - 1 } else { 0 };
                    let y = if down { h - 1 } else { 0 };
                    for row in 0..h {
                        assert_eq!(
                            at(&corner, w, x, row),
                            at(&horizontal, w, w - 1 - x, row),
                            "{ch} horizontal join at row {row} in {w}x{h}"
                        );
                        assert_eq!(
                            at(&corner, w, w - 1 - x, row),
                            0,
                            "{ch} closed horizontal edge at row {row} in {w}x{h}"
                        );
                    }
                    for col in 0..w {
                        assert_eq!(
                            at(&corner, w, col, y),
                            at(&vertical, w, col, h - 1 - y),
                            "{ch} vertical join at column {col} in {w}x{h}"
                        );
                        assert_eq!(
                            at(&corner, w, col, h - 1 - y),
                            0,
                            "{ch} closed vertical edge at column {col} in {w}x{h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn arc_and_diagonal_produce_ink() {
        for ch in ['╭', '╮', '╯', '╰', '╱', '╲', '╳'] {
            let buf = coverage(ch, 14, 20);
            assert!(buf.iter().any(|&v| v > 0), "{ch} must draw a stroke");
            // Curves/diagonals antialias, so some partial-coverage pixel exists.
            assert!(
                buf.iter().any(|&v| v > 0 && v < 255),
                "{ch} must be antialiased"
            );
        }
    }

    #[test]
    fn rising_diagonal_favors_the_rising_direction() {
        // `╱` runs bottom-left → top-right: inked near the bottom-left corner,
        // clear near the top-left.
        let (w, h) = (16, 16);
        let buf = coverage('╱', w, h);
        assert_ne!(at(&buf, w, 0, h - 1), 0, "bottom-left inked");
        assert_eq!(at(&buf, w, 0, 0), 0, "top-left clear");
    }

    /// The inked pixels' bounding box as half-open `(x0, y0, x1, y1)`, or `None` for
    /// a blank buffer.
    fn ink_box(buf: &[u8], w: usize) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for (i, _) in buf.iter().enumerate().filter(|&(_, &v)| v != 0) {
            let (x, y) = (i % w, i / w);
            bounds = Some(match bounds {
                None => (x, y, x + 1, y + 1),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1)),
            });
        }
        bounds
    }

    fn braille(bits: u32) -> char {
        char::from_u32(0x2800 + bits).unwrap()
    }

    #[test]
    fn the_blank_braille_pattern_draws_nothing() {
        for (w, h) in [(9, 20), (18, 40)] {
            assert!(coverage('\u{2800}', w, h).iter().all(|&v| v == 0));
        }
    }

    /// Dots 7 and 8 hold the two highest bits but sit in the bottom row.
    #[test]
    fn each_braille_dot_lands_in_its_unicode_position() {
        let (w, h) = (9, 20);
        let dot_box =
            |bits: u32| ink_box(&coverage(braille(bits), w, h), w).expect("a raised dot inks");
        // Each column, top to bottom.
        for column in [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]] {
            let boxes = column.map(dot_box);
            for pair in boxes.windows(2) {
                assert_eq!(pair[0].0, pair[1].0, "one column: {boxes:?}");
                assert!(pair[0].3 <= pair[1].1, "each dot below the last: {boxes:?}");
            }
        }
        // Each row, left to right.
        for (left, right) in [(0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80)] {
            let (l, r) = (dot_box(left), dot_box(right));
            assert_eq!(l.1, r.1, "dots {left:#04x} and {right:#04x} share a row");
            assert!(l.2 <= r.0, "dot {right:#04x} is right of dot {left:#04x}");
        }
        // Each dot is a solid square with no ink outside it.
        for bits in [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80] {
            let buf = coverage(braille(bits), w, h);
            let (x0, y0, x1, y1) = dot_box(bits);
            assert_eq!(x1 - x0, y1 - y0, "dot {bits:#04x} is square");
            let inked = buf.iter().filter(|&&v| v != 0).count();
            assert_eq!(
                inked,
                (x1 - x0) * (y1 - y0),
                "dot {bits:#04x} is solid, no stray ink"
            );
            assert!(
                buf.iter().all(|&v| v == 0 || v == 255),
                "dot {bits:#04x} is crisp"
            );
        }
    }

    /// Each of the 256 patterns is the pixelwise union of its single-dot rasters.
    #[test]
    fn every_braille_pattern_is_the_union_of_its_dots() {
        for (w, h) in [(9, 20), (18, 40), (7, 15)] {
            let dots: Vec<Vec<u8>> = (0..8)
                .map(|bit| coverage(braille(1 << bit), w, h))
                .collect();
            for bits in 0u32..=0xFF {
                let buf = coverage(braille(bits), w, h);
                for i in 0..w * h {
                    let raised = (0..8).any(|bit| bits & (1 << bit) != 0 && dots[bit][i] == 255);
                    let want = if raised { 255 } else { 0 };
                    assert_eq!(
                        buf[i],
                        want,
                        "{} at pixel {i} of a {w}x{h} cell",
                        braille(bits)
                    );
                }
            }
        }
    }

    /// `⣿` inks exactly eight whole dots at every size, none clipped or overlapping.
    /// Cells smaller than 2x4 stay blank.
    #[test]
    fn the_braille_grid_fits_every_cell_without_overlap() {
        for w in 1..=40 {
            for h in 1..=80 {
                let dot = BrailleGrid::new(w as i32, h as i32).dot as usize;
                assert_eq!(dot > 0, w >= 2 && h >= 4, "dot size {dot} at {w}x{h}");
                let ink = coverage('⣿', w, h).iter().filter(|&&v| v == 255).count();
                assert_eq!(
                    ink,
                    8 * dot * dot,
                    "⣿ is eight whole, disjoint dots at {w}x{h}"
                );
            }
        }
    }

    /// From 5x10 up (at least as tall as wide), dots keep 1px from the cell edge and
    /// from each other, so dots in neighbouring cells never fuse.
    #[test]
    fn braille_dots_stay_clear_of_the_cell_edges_and_each_other() {
        for w in 5..=40 {
            for h in w.max(10)..=100 {
                let buf = coverage('⣿', w, h);
                let clear = |x: usize, y: usize| at(&buf, w, x, y) == 0;
                assert!(
                    (0..w).all(|x| clear(x, 0) && clear(x, h - 1)),
                    "⣿ touches the top or bottom edge of a {w}x{h} cell"
                );
                assert!(
                    (0..h).all(|y| clear(0, y) && clear(w - 1, y)),
                    "⣿ touches the left or right edge of a {w}x{h} cell"
                );
                let (x, y) = BrailleGrid::new(w as i32, h as i32).dot_at(0, 0);
                let (x, y) = (x as usize, y as usize);
                let across = runs((0..w).map(|c| !clear(c, y)));
                let down = runs((0..h).map(|r| !clear(x, r)));
                assert_eq!(across, 2, "two separate dots across a {w}x{h} cell");
                assert_eq!(down, 4, "four separate dots down a {w}x{h} cell");
            }
        }
    }

    /// Pins the layout at a few cell sizes.
    #[test]
    fn the_braille_layout_is_pinned() {
        // (w, h) -> (dot, x_margin, x_gap, y_margin, y_gap)
        let cases = [
            ((9, 20), (2, 1, 3, 1, 3)),
            ((18, 40), (4, 2, 5, 3, 6)),
            ((8, 17), (2, 1, 2, 1, 2)),
            ((7, 15), (1, 1, 2, 2, 2)),
        ];
        for ((w, h), want) in cases {
            let g = BrailleGrid::new(w, h);
            let got = (g.dot, g.x_margin, g.x_gap, g.y_margin, g.y_gap);
            assert_eq!(got, want, "braille layout of a {w}x{h} cell");
        }
    }

    #[test]
    fn coverage_never_panics_over_the_whole_range() {
        for c in (0x2500u32..=0x259F).chain(0x2800..=0x28FF) {
            if let Some(ch) = char::from_u32(c) {
                for (w, h) in [(1, 1), (3, 5), (9, 20), (16, 32)] {
                    let buf = coverage(ch, w, h);
                    assert_eq!(buf.len(), w * h);
                }
            }
        }
    }
}
