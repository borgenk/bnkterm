//! The terminal's painter: it walks the visible grid and produces a
//! backend-agnostic [`DisplayList`], the value the damage differ and the GPU
//! batcher both consume. This is the seam between the terminal core and the
//! renderer: the core never touches Vulkan, it emits drawing
//! primitives, and everything downstream (`display::damage`, `gpu::build_frame`)
//! is a pure function of that list.
//!
//! ```text
//!   grid::Screen ──build_display_list──▶ DisplayList ──┬─ gpu::build_frame ─▶ pixels
//!   (+ Theme, CellMetrics, cursor,                     └─ damage(old,new) ─▶ [Rect]
//!    selection)
//! ```
//!
//! # Fixed pitch is the whole game
//!
//! A terminal is a grid: the cell at `(row, col)` occupies exactly the pixel box
//! `[ox + col*w, ox + (col+1)*w) x [oy + row*h, oy + (row+1)*h)`, where `(ox, oy)`
//! is the window's content origin (the padding inset the caller passes; the whole
//! grid slides by it as one rigid block). The cursor, the selection, and the
//! glyphs must all agree on where that box is or the display corrupts (the cursor
//! "drifts from the glyphs"). They agree by
//! *routing every coordinate through `cell_x`/`cell_y`* and never accumulating a font's
//! (fractional) advance: text goes out as [`DrawCmd::Cells`], whose contract is
//! exactly "cluster `i` draws at `x + i*cell_w`". The one measurement that keys
//! the whole grid, `cell_w`, is the advance of a reference glyph from the regular
//! face rounded to a whole pixel; the grid decides *which* column a rune lands in
//! (via `width::width`), so measurement and placement can never disagree.
//!
//! # Where the text sits in the cell
//!
//! The vertical half of the same story, and the one that is easy to get wrong: a
//! row's glyphs are seated on `cell_y(row) + metrics.baseline`, where `baseline` is
//! the distance from the cell's *top edge* down to the baseline — never the ascent.
//! The two differ by the font's line gap (Consolas asks for 2-3px of it at terminal
//! sizes), and spending that gap under the text instead of over it rides every glyph
//! up against the cell's ceiling: capitals crowd the top edge, descenders float clear
//! of the bottom, and the block cursor no longer reads as a box drawn around the
//! character. [`crate::platform::freetype::Metrics`] derives the number and explains
//! why it lands where it does; here it is simply the one seat every row uses, and the
//! one a box-drawing glyph is packed against in [`crate::render::gpu`], so the two
//! agree pixel for pixel.
//!
//! # What a frame is made of
//!
//! Per painted row, in stacking order (the list order the damage diff and the GPU
//! rely on): the base background fill (once, whole surface), then per-cell
//! background runs where a cell's background differs from the theme's, then the
//! foreground as fixed-pitch [`DrawCmd::Cells`] runs (with wide glyphs and
//! astral-plane runes broken out into their own [`DrawCmd::Text`] so the pitch
//! stays uniform), then underline/strike rules and the hovered link's rule, and
//! finally the cursor on top. Everything past the base fill is emitted only where it
//! is not the default, so an idle screen of mostly-blank cells produces a short list
//! and an unchanged frame diffs to nothing.
//!
//! The two overlays the app hands in — the selection band and the hovered hyperlink's
//! underline — are both a [`CellSpan`], and both resolve to *one inclusive column
//! range per row* (`span_cols`) before anything per-cell happens. That is a
//! performance rule, not a stylistic one: `Painter::cell` runs about four times per
//! cell, so an overlay tested there is tested a quarter of a million times a frame to
//! serve a span that is usually empty. A 13% frame regression established it.

use crate::color::{Ground, Rgb, Theme};
use crate::grid::{AbsRow, Attrs, Cell, RowEpoch, Screen, UnderlineStyle};
use crate::platform::freetype::{FaceKey, FontStyle, Fonts};
use crate::platform::geom::{Rect, Scale};
use crate::platform::grapheme;
use crate::platform::pixel;
use crate::platform::scroll::{self, Scrollbar};
use crate::render::display::{DisplayList, DrawCmd, Fade, RoundedCorners};

/// The glyph whose advance defines the monospace cell width. `M` is the classic
/// full-width reference; on a genuine monospace face every glyph shares it.
const REFERENCE_GLYPH: char = 'M';

/// The selection highlight background: a muted steel blue that stays legible
/// under the default fg.
const SELECTION_BG: Rgb = Rgb::new(0x41, 0x57, 0x76);

/// The width of a bar (`DECSCUSR 5/6`) cursor, in pixels.
const BAR_CURSOR_WIDTH: i32 = 2;

/// The padlock drawn in place of the cursor while the tty is taking a password: `nf-fa-lock`
/// in the Nerd Fonts private-use area, the same codepoint wezterm and ghostty typeset, so
/// bnkterm's lock is *their* lock rather than a lookalike.
///
/// Being private-use, it is only there if a symbols font is: the Nerd Fonts range is the one
/// part of the fallback chain that a bare system genuinely may not have (`make deps` installs
/// it). So nothing may assume it — [`CellMetrics::lock_glyph`] says whether this machine can
/// draw it, and the cursor falls back to the shape the child asked for when it cannot. A
/// password cursor that renders as an empty tofu box would be worse than no password cursor
/// at all: it is unexplained rather than merely absent.
pub const LOCK_GLYPH: char = '\u{f023}';

/// The overlay scrollbar down the grid's right edge, in *logical* pixels (they pass
/// through [`Scale::px`], so they come out the same physical size at any DPI).
///
/// The bar rests as a thin indicator and grows to [`SCROLL_WIDE`] when the pointer
/// enters its zone: [`SCROLL_NEAR`] of slack to the left of the lane, so it meets the
/// pointer before the pointer has to aim. The thumb is right-aligned in the lane and
/// grows leftward, staying anchored to the window edge, which is where a maximized
/// window's pointer lands.
///
/// The window padding is thinner than the bar and there is no text margin for the
/// slider to hide in, so the expanded slider laps over the last column. That is the price
/// of an *overlay*: reserving a gutter would cost a column and reflow the child on every
/// appearance. The bar is only that wide while the pointer is on it, and only visible at
/// all while there is history to scroll.
const SCROLL_THIN: i32 = 4;
const SCROLL_WIDE: i32 = 10;
const SCROLL_NEAR: i32 = 24;
/// Gap between the lane and the window's right edge.
const SCROLL_INSET: i32 = 3;
/// Shortest the thumb gets, so even a 100k-line scrollback leaves something to grab.
const SCROLL_MIN_THUMB: i32 = 30;

/// The colour the whole surface is filled with: the theme background, lifted a third of
/// the way toward the foreground while the bell is ringing.
///
/// It lives here, and not inline in the painter, because the GPU clear must use the very
/// same colour as the display list's base fill — two places computing "the background"
/// separately is exactly how a flash tears at the edges.
pub fn bell_background(theme: &Theme, bell: bool) -> Rgb {
    if bell {
        theme.bg.mix(theme.fg, 1, 3)
    } else {
        theme.bg
    }
}

/// The bar's ink, and the coverages it is tinted onto the theme background at: the
/// resting thumb, the thumb once the pointer has lifted it, and the trough behind it.
/// The fade is these falling to zero (the display list has no alpha; see
/// [`crate::platform::pixel::tint`]).
const SCROLL_INK: u32 = 0x00cf_d8e3;
const SCROLL_THUMB_COVER: f32 = 0.20;
const SCROLL_THUMB_HOVER_COVER: f32 = 0.36;
const SCROLL_TROUGH_COVER: f32 = 0.05;

/// Fixed monospace cell metrics in whole pixels: the pitch the entire grid is
/// laid out on. `size` is the pixel size the faces were opened at, carried so the
/// painter can name the [`FaceKey`] a run draws in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellMetrics {
    pub size: u32,
    /// Cell width: one column's advance.
    pub w: i32,
    /// Cell height: the baseline-to-baseline line height.
    pub h: i32,
    /// Pixels from the top of a cell down to the baseline: where the text sits in
    /// the cell. This is *not* the ascent (see [`Metrics`](crate::platform::freetype::Metrics)) — the font's line gap
    /// lies between the two, and mistaking one for the other rides the text up
    /// against the cell's top edge.
    pub baseline: i32,
    /// The face's ascent: ink above the baseline. A reference for decorations
    /// (a strike is a fraction of it), never the cell's top.
    pub ascent: i32,
    /// The face's descent: ink below the baseline, which the baseline seats on the
    /// cell's bottom edge.
    pub descent: i32,
    /// Whether the font stack can draw [`LOCK_GLYPH`], the password padlock.
    ///
    /// Not a measurement, but it rides here because it is the same question as the rest of
    /// this struct — *what can this font stack do in a cell* — asked of the same stack at
    /// the same moment, and it is re-asked whenever a font change re-measures the rest. It
    /// is read where the cursor is painted, which is exactly where the metrics already are.
    pub lock_glyph: bool,
}

impl CellMetrics {
    /// Measure the cell box from the opened fonts at `size`: the line height for
    /// the row pitch and the regular face's reference-glyph advance for the column
    /// pitch, each rounded to a whole pixel (a fractional pitch is what drifts a
    /// grid). Width and height are clamped to at least one pixel so a degenerate
    /// face can never produce a zero-area cell.
    pub fn from_fonts(fonts: &Fonts, size: u32) -> Self {
        let m = fonts.metrics(size);
        let advance = fonts
            .face(size, FontStyle::Regular)
            .advance(REFERENCE_GLYPH);
        CellMetrics {
            size,
            w: (advance.round() as i32).max(1),
            h: m.line_height.max(1),
            baseline: m.baseline,
            ascent: m.ascent,
            descent: m.descent,
            lock_glyph: fonts.covers(
                FaceKey::Prose {
                    size,
                    style: FontStyle::Regular,
                },
                LOCK_GLYPH,
            ),
        }
    }

    /// The vertical box for the proportional interface face at `size`: its own
    /// ascent/descent/line-height, so chrome text (the tab bar) baselines on the
    /// sans ink box. `w` carries a nominal glyph advance for callers that want one,
    /// but proportional UI text is placed by each glyph's real advance, not this
    /// pitch, so it is not a cell width.
    pub fn from_ui(fonts: &Fonts, size: u32) -> Self {
        let m = fonts.ui_metrics(size);
        let advance = fonts.ui_face(size, FontStyle::Regular).advance('n');
        CellMetrics {
            size,
            w: (advance.round() as i32).max(1),
            h: m.line_height.max(1),
            baseline: m.baseline,
            ascent: m.ascent,
            descent: m.descent,
            // The chrome never draws a cursor, and lock coverage comes from the fallback
            // chain, not this face, so it resolves the same either way. Measuring it
            // rather than hardcoding `false` keeps the two constructors from disagreeing.
            lock_glyph: fonts.covers(
                FaceKey::Ui {
                    size,
                    style: FontStyle::Regular,
                },
                LOCK_GLYPH,
            ),
        }
    }

    /// The largest `(cols, rows)` grid that fits a `width` x `height` pixel
    /// surface, at least 1x1. The app uses this to size the grid on a resize.
    pub fn columns_rows(self, width: i32, height: i32) -> (usize, usize) {
        let cols = (width / self.w).max(1) as usize;
        let rows = (height / self.h).max(1) as usize;
        (cols, rows)
    }
}

/// A cursor's drawn shape. The child picks the first three through `DECSCUSR`; the
/// app picks [`Lock`](CursorShape::Lock), which is not a terminal state at all but a
/// property of the tty underneath it (see [`TtyMode`](crate::pty::TtyMode)), and so
/// overrides whatever style the child last asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CursorShape {
    /// A filled cell that inverts the glyph under it (the default).
    Block,
    /// A thin vertical bar at the cell's left edge.
    Bar,
    /// A thin rule along the cell's baseline.
    Underline,
    /// A padlock: the tty is line-editing with echo off, so whatever is being typed
    /// here is a secret (a `sudo`/`ssh`/`passwd` prompt).
    Lock,
}

/// How to paint the cursor this frame. `visible` folds the grid's `DECTCEM` state
/// together with the app's blink phase, so the painter draws the cursor exactly
/// when it should be lit. An unfocused window draws a hollow block, the xterm
/// convention for "this window does not have the keyboard".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CursorRender {
    pub shape: CursorShape,
    pub visible: bool,
    pub focused: bool,
}

impl Default for CursorRender {
    fn default() -> Self {
        CursorRender {
            shape: CursorShape::Block,
            visible: true,
            focused: true,
        }
    }
}

/// A linear text selection, in absolute `(row, col)` cells. `anchor` is where the drag
/// began and `head` where it is now, either order; a cell falls in the selection when it
/// lies between them in reading order (whole rows in the middle, partial rows at the
/// ends). The painter paints the whole geometric span, blank cells included (see
/// `Painter::selection_cols`), so dragging over the empty area below the prompt
/// highlights it; the copy still trims each line's trailing blanks (see
/// [`crate::grid::Screen::selection_text`]), so the paint is deliberately wider than what
/// lands on the clipboard.
///
/// [`AbsRow`] and not a display row, so that a printing child cannot drag the selection
/// off the text it was made over: output scrolls the grid, the rows keep their ids, and
/// the highlight stays on the words the user picked. `epoch` is the identity regime the
/// ids were minted in — see [`RowEpoch`] for what ends one, and
/// `TerminalCore::prune_selection` for who checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Selection {
    pub anchor: (AbsRow, usize),
    pub head: (AbsRow, usize),
    pub epoch: RowEpoch,
}

impl Selection {
    /// `(start, end)` in reading order, so `start <= end` row-major.
    pub fn ordered(self) -> ((AbsRow, usize), (AbsRow, usize)) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }
}

/// An inclusive span of display cells in reading order: every cell from `start` to
/// `end` row-major is inside it, so a span may run off the right edge of one row and
/// resume at the left of the next. That is the shape of the hovered hyperlink, whose
/// text is contiguous across a soft wrap (see [`crate::grid::Screen::link_at`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CellSpan {
    pub start: (usize, usize),
    pub end: (usize, usize),
}

/// The inputs a frame build reads: the grid and its theme, the fixed cell metrics,
/// the `surface`/`origin` geometry (`origin` is the grid's top-left pixel, the window
/// padding; the background still fills the whole `surface`, so the inset shows as a
/// margin), and the transient cursor, selection, and hovered link. Grouped so the
/// pooled and one-shot builders share one parameter and the window/bench assemble it
/// in one place. Pure data (no fonts, no GPU), so a test asserts the exact primitives
/// a grid state produces.
pub struct FrameInputs<'a> {
    pub screen: &'a Screen,
    pub theme: &'a Theme,
    pub metrics: CellMetrics,
    pub surface: (i32, i32),
    pub origin: (i32, i32),
    pub cursor: CursorRender,
    pub selection: Option<Selection>,
    /// The hyperlink under the pointer, which paints underlined. The app finds it
    /// (the grid holds the text); the painter only draws it.
    pub hover: Option<CellSpan>,
    /// The display scale, so the scrollbar's logical pixel constants land at the right
    /// device size. [`Scale::ONE`] in a test or a headless build.
    pub scale: Scale,
    /// The bell is ringing: paint the background lifted toward the foreground for the
    /// moment the flash lasts. A terminal bell on a modern desktop is something you
    /// *see*; nobody wants a beep out of a terminal in 2026.
    pub bell: bool,
    /// The overlay scrollbar, as lit and as wide as its last tick left it. The bar
    /// carries no geometry: the painter derives that from the grid and the screen's
    /// [`Screen::scroll_extent`] each frame.
    pub scrollbar: &'a Scrollbar,
}

/// The three rectangles the overlay scrollbar owns down the right edge of the grid: the
/// lane its thumb travels, the band a press grabs it in, and the zone whose pointer
/// wakes it into a full slider. Shared by the painter and the window's pointer
/// hit-testing, so the bar is grabbable exactly where it is drawn.
///
/// ```text
///                     window's right edge ─┐
///     │        zone            │  grab  │  │
///     │                        │ track  │  │
///     ├────────────────────────┼────────┤  │
///     │  ....................  │  ███   │  │  ← thumb, right-aligned in the track
/// ```
pub struct ScrollLane {
    /// The thumb's lane: the full height of the grid, inset from the window's right edge.
    pub track: Rect,
    /// Where a press takes the bar: the lane plus the inset out to the window's edge, so
    /// a click at the very edge of a maximized window still lands on it.
    pub grab: Rect,
    /// Where the pointer expands the bar: the lane plus a margin of slack to its left.
    pub zone: Rect,
}

/// The column the scrollbar hangs off: the grid's visible rows, running out to the
/// *surface's* right edge rather than the grid's. The window padding is thinner than the
/// bar, and a pointer flung at the edge of a maximized window has to land on it.
///
/// Both the painter and the window's hit-test call this, so the bar cannot be drawn
/// anywhere other than where it is grabbable.
pub fn scroll_column(
    surface: (i32, i32),
    origin: (i32, i32),
    metrics: CellMetrics,
    rows: usize,
) -> Rect {
    Rect {
        x: origin.0,
        y: origin.1,
        w: (surface.0 - origin.0).max(0),
        h: rows as i32 * metrics.h,
    }
}

/// Split the right edge of the grid column `view` into the scrollbar's lane, its grab
/// band, and its proximity zone.
pub fn scroll_lane(view: Rect, scale: Scale) -> ScrollLane {
    let wide = scale.px(SCROLL_WIDE);
    let inset = scale.px(SCROLL_INSET);
    let right = view.x + view.w;
    let track = Rect {
        x: right - inset - wide,
        y: view.y,
        w: wide,
        h: view.h,
    };
    let from = |x: i32| Rect {
        x,
        y: view.y,
        w: (right - x).max(0),
        h: view.h,
    };
    ScrollLane {
        track,
        grab: from(track.x),
        zone: from(track.x - scale.px(SCROLL_NEAR)),
    }
}

/// The shortest thumb the bar will draw, in device pixels. The window's drag hit-test
/// needs the same floor the painter used, or a grab would miss the thumb it can see.
pub fn scroll_min_thumb(scale: Scale) -> i32 {
    scale.px(SCROLL_MIN_THUMB)
}

/// Build the frame's display list into `out` (cleared first), drawing every run's
/// text from `strings` — a pool of buffers retired by a previous frame — so a steady
/// stream of frames rebuilds the list without allocating. Both are owned by a
/// [`DisplayListPool`] across frames; `out` keeps its capacity and the strings are
/// recycled, so only a first frame (or a grown grid) touches the allocator.
pub fn build_display_list_into(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    inputs: &FrameInputs,
) {
    out.clear();
    let mut painter = Painter {
        screen: inputs.screen,
        theme: inputs.theme,
        bell: inputs.bell,
        metrics: inputs.metrics,
        origin: inputs.origin,
        selection: inputs.selection,
        hover: inputs.hover,
        scale: inputs.scale,
        scrollbar: inputs.scrollbar,
        list: out,
        strings,
    };
    painter.background(inputs.surface);
    let (cols, rows) = inputs.screen.dimensions();
    for row in 0..rows {
        painter.background_row(row, cols);
        painter.foreground_row(row, cols);
    }
    painter.cursor(inputs.cursor, cols, rows);
    painter.push_scrollbar(inputs.surface, rows);
}

/// Build a fresh display list, allocating its vector and run strings. The one-shot
/// path for tests and callers that do not keep frame-to-frame state; the windowed
/// render loop uses [`build_display_list_into`] with a [`DisplayListPool`] to stay
/// allocation-free in steady state.
pub fn build_display_list(inputs: &FrameInputs) -> DisplayList {
    let mut out = DisplayList::new();
    let mut strings = Vec::new();
    build_display_list_into(&mut out, &mut strings, inputs);
    out
}

/// Reusable storage for the per-frame display list, so a steady stream of frames
/// rebuilds it without touching the allocator. Two buffers let the damage differ
/// compare the on-screen frame against the freshly built one; a pool of string
/// buffers salvaged from retired [`DrawCmd`]s feeds the next build.
///
/// ```text
///   begin()  ─ salvage back's run strings ─▶ pool, clear back
///   fill back (build_display_list_into, drawing text from the pool)
///   damage(front, back) ─▶ regions to repaint
///   commit() ─ swap front/back once the frame is presented
/// ```
#[derive(Default)]
pub struct DisplayListPool {
    /// The two lists; `front` indexes the on-screen one, `front ^ 1` the scratch.
    buffers: [DisplayList; 2],
    front: usize,
    /// String buffers reclaimed from retired commands, cleared and ready to refill.
    strings: Vec<String>,
}

impl DisplayListPool {
    /// Recycle the back buffer's run strings into the pool and empty it (keeping its
    /// capacity), then hand back that empty buffer and the pool to fill.
    pub fn begin(&mut self) -> (&mut DisplayList, &mut Vec<String>) {
        let back = self.front ^ 1;
        let buf = &mut self.buffers[back];
        let strings = &mut self.strings;
        for cmd in buf.drain(..) {
            if let Some(s) = cmd.into_text_buf() {
                strings.push(s);
            }
        }
        (buf, strings)
    }

    /// The on-screen list (the previous frame): the differ's `old` side.
    pub fn front(&self) -> &DisplayList {
        &self.buffers[self.front]
    }

    /// The freshly built list (this frame): the differ's `new` side.
    pub fn back(&self) -> &DisplayList {
        &self.buffers[self.front ^ 1]
    }

    /// Promote the built back buffer to the on-screen list, once it is presented.
    pub fn commit(&mut self) {
        self.front ^= 1;
    }

    /// Forget both frames (a resize wiped the buffers): recycle their strings and
    /// clear them, so the next diff sees an empty prev and repaints in full.
    pub fn reset(&mut self) {
        let strings = &mut self.strings;
        for buf in &mut self.buffers {
            for cmd in buf.drain(..) {
                if let Some(s) = cmd.into_text_buf() {
                    strings.push(s);
                }
            }
        }
        self.front = 0;
    }
}

/// What a fixed-pitch run is uniform in: its resolved foreground, the face it draws
/// in, and the two rules the painter lays over it. Built by [`Painter::run_style`],
/// which says why both the run's extent and its decorations come from this one value.
///
/// Two fields, and both are cheap on purpose: the painter compares this once per cell in
/// the run scan, so it is a colour and one masked `u16` rather than a set of unpacked
/// flags. Building the unpacked form here cost the frame 14% on the gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RunStyle {
    /// The resolved foreground the run's glyphs and rules are drawn in (reverse and dim
    /// already applied, which is why neither bit appears in `key`).
    fg: Rgb,
    /// The face and rule bits, masked and normalised by [`Attrs::run_key`]. Kept packed
    /// rather than unpacked because this is the per-cell comparison; the rules are read
    /// back out of it once per run, in [`Painter::push_decorations`].
    key: Attrs,
}

/// The per-frame builder: the shared inputs plus the list being appended to. One
/// method per concern keeps [`build_display_list`] readable.
struct Painter<'a> {
    screen: &'a Screen,
    theme: &'a Theme,
    /// The bell is mid-flash; the base background lifts for it.
    bell: bool,
    metrics: CellMetrics,
    /// The grid's top-left pixel in the surface: the window padding inset. Every
    /// cell coordinate is measured from here (via [`Self::cell_x`]/[`Self::cell_y`]),
    /// so the whole grid rides the same rigid offset and can never drift from it.
    origin: (i32, i32),
    selection: Option<Selection>,
    /// The hovered hyperlink's cells, underlined by [`Self::push_hover_rule`].
    hover: Option<CellSpan>,
    /// The display scale, so the scrollbar's logical constants land at the right size.
    scale: Scale,
    /// The overlay scrollbar's fade/expand state, drawn by [`Self::scrollbar`].
    scrollbar: &'a Scrollbar,
    /// The list being appended to, owned by the caller's [`DisplayListPool`] and
    /// reused across frames.
    list: &'a mut DisplayList,
    /// Recycled string buffers to draw run/glyph text from, so a steady frame's
    /// text costs no allocation.
    strings: &'a mut Vec<String>,
}

impl Painter<'_> {
    /// A cleared string buffer from the recycle pool, or a fresh empty one when the
    /// pool is dry. Every run and glyph text is drawn from here so the frame reuses
    /// the buffers the previous frame retired.
    fn take_string(&mut self) -> String {
        take_string(self.strings)
    }

    /// Recycle a buffer already large enough for the coming run when possible.
    fn take_string_with_capacity(&mut self, min_capacity: usize) -> String {
        take_string_with_capacity(self.strings, min_capacity)
    }

    /// The left pixel of column `col`, from the content origin.
    fn cell_x(&self, col: usize) -> i32 {
        self.origin.0 + col as i32 * self.metrics.w
    }

    /// The top pixel of row `row`, from the content origin.
    fn cell_y(&self, row: usize) -> i32 {
        self.origin.1 + row as i32 * self.metrics.h
    }

    /// The baseline y for `row`: the row's top plus the cell's baseline offset.
    fn baseline(&self, row: usize) -> i32 {
        self.cell_y(row) + self.metrics.baseline
    }

    /// The base background: one fill of the theme background covering the whole
    /// surface (including any partial cell at the right/bottom edge). Every
    /// default-background cell is then just this fill showing through, so the
    /// per-cell background pass emits nothing for them.
    fn background(&mut self, surface: (i32, i32)) {
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: 0,
                y: 0,
                w: surface.0.max(0),
                h: surface.1.max(0),
            },
            color: bell_background(self.theme, self.bell).to_u32(),
        });
    }

    /// Per-cell backgrounds for one row, coalesced into runs of equal colour so a
    /// solid band is one fill. A run equal to the theme background is skipped (the
    /// base fill already covers it). A wide glyph's spacer carries the leader's
    /// colours, so it coalesces with the leader with no special case here.
    fn background_row(&mut self, row: usize, cols: usize) {
        let m = self.metrics;
        // The selected span is one inclusive column range per row, so the per-cell
        // test below stays O(1) and the whole band coalesces into one fill.
        let sel = self.selection_cols(row);
        let selected = |col: usize| sel.is_some_and(|(a, b)| a <= col && col <= b);
        let mut col = 0;
        while col < cols {
            let bg = self.cell_bg(row, col, selected(col));
            let start = col;
            col += 1;
            while col < cols && self.cell_bg(row, col, selected(col)) == bg {
                col += 1;
            }
            if bg != self.theme.bg {
                let x = self.cell_x(start);
                let y = self.cell_y(row);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y,
                        w: (col - start) as i32 * m.w,
                        h: m.h,
                    },
                    color: bg.to_u32(),
                });
            }
        }
    }

    /// The foreground glyphs and text decorations for one row. Consecutive
    /// single-width cells that share a foreground colour, face style, and
    /// underline/strike state batch into one fixed-pitch [`DrawCmd::Cells`] run;
    /// a wide glyph (its leader) and an astral-plane rune break out into their own
    /// [`DrawCmd::Text`] so the run's one-cluster-per-cell contract holds. See the
    /// module header for why the pitch, not the advance, drives placement.
    fn foreground_row(&mut self, row: usize, cols: usize) {
        let baseline = self.baseline(row);
        let mut col = 0;
        while col < cols {
            let cell = self.cell(row, col);
            if cell.is_wide_spacer() {
                // An orphaned spacer (its leader scrolled off the left) draws
                // nothing; the leader owns the glyph.
                col += 1;
                continue;
            }
            if cell.is_wide_leader() || !cells_safe(cell.rune) {
                // Wide glyphs and astral runes stand alone at their exact column,
                // so their non-uniform advance can never shift a following cell.
                self.push_glyph(row, col, cell, baseline);
                col += if cell.is_wide_leader() { 2 } else { 1 };
                continue;
            }
            col = self.push_run(row, col, cols, cell, baseline);
        }
        // Over the glyphs, like a run's own rules.
        self.push_hover_rule(row);
    }

    /// Emit a fixed-pitch run starting at `col`: the maximal span of single-width,
    /// same-style cells. The [`DrawCmd::Cells`] text is trimmed to the last inked
    /// cell (trailing blanks in the run draw nothing, so carrying them would only
    /// cost allocation and coarsen the damage diff), while underline/strike rules
    /// span the full styled run, because a styled trailing space still shows its
    /// rule (xterm draws it).
    ///
    /// Returns the run's end column (exclusive), which is where the caller's scan
    /// resumes.
    ///
    /// The run's extent is discovered *while* its text is built, in one walk. Finding
    /// the end first and then filling the string reads more clearly, but it is two
    /// passes over the same cells — the extent scan resolving a colour and a style per
    /// cell, the fill re-reading every one of them — to emit a single run. Scanning
    /// once and copying once is the same rule the grid's own hot paths follow.
    fn push_run(
        &mut self,
        row: usize,
        col: usize,
        cols: usize,
        first: Cell,
        baseline: i32,
    ) -> usize {
        let m = self.metrics;
        // The run breaks on foreground, not background, so the first cell's background
        // stands in for the run when weighting the glyph anti-aliasing; a same-fg run
        // over a mixed background is rare (reverse video and selection tint uniformly).
        let (_, bg) = self.resolve(first, false);
        let run = self.run_style(first);
        // A run cannot outlast its row, so ask the pool for that much and let every
        // run settle on one capacity: the pool probes from the back for the first
        // buffer that fits, and uniform sizes are what make that probe hit first try.
        // It is a hint either way — combining marks push more chars than there are
        // cells, so no bound here is exact.
        let mut text = self.take_string_with_capacity(cols - col);
        // Bytes and cell count up to and including the last cell with ink, so a
        // trailing blank never lands in the emitted text.
        let mut inked_bytes = 0;
        let mut inked_cells = 0;
        let mut end = col;
        while end < cols {
            // The caller already read the first cell to decide this was a run at all.
            let cell = if end == col {
                first
            } else {
                self.cell(row, end)
            };
            if end > col {
                // A wide glyph or an astral rune stands alone, and anything the run can
                // only draw one of ends it: past here is a different command.
                if cell.is_wide_leader() || cell.is_wide_spacer() || !cells_safe(cell.rune) {
                    break;
                }
                if self.run_style(cell) != run {
                    break;
                }
            }
            let hidden = cell.attrs.contains(Attrs::HIDDEN);
            text.push(if hidden { ' ' } else { cell.rune });
            let has_marks = !hidden && !self.marks_empty(row, end);
            if has_marks {
                text.extend(self.marks(row, end));
            }
            if !hidden && (cell.rune != ' ' || has_marks) {
                inked_bytes = text.len();
                inked_cells = end - col + 1;
            }
            end += 1;
        }
        let x = self.cell_x(col);
        if inked_cells > 0 {
            text.truncate(inked_bytes);
            push_owned_cells(
                self.list,
                text,
                x,
                baseline,
                inked_cells,
                m,
                FaceKey::Prose {
                    size: m.size,
                    style: style_of(run.key),
                },
                run.fg.to_u32(),
                bg.to_u32(),
            );
        } else {
            // A blank run emits no command, so explicitly return its scratch
            // buffer; otherwise it is dropped and reallocated next frame. Clear it
            // first: the pool's invariant is that its buffers are empty (every
            // buffer salvaged from a retired command enters via `into_text_buf`,
            // which clears it), and the next taker appends without clearing.
            text.clear();
            self.strings.push(text);
        }
        self.push_decorations(run, x, (end - col) as i32 * m.w, baseline);
        end
    }

    /// Everything about a cell that the painter's per-run output depends on, and so
    /// everything a run may not batch across.
    ///
    /// One value serves both questions — where the run ends, and what its single set of
    /// decorations is drawn from — because deriving them separately is how a run comes to
    /// batch across a difference it cannot then draw. What is deliberately *not* here is
    /// what a run can absorb: a cell's background (a run is one span of glyphs over
    /// whatever ground the background pass laid down) and its `HIDDEN` bit (a concealed
    /// cell goes into the text as a space and keeps the run going).
    fn run_style(&self, cell: Cell) -> RunStyle {
        RunStyle {
            fg: self.cell_fg(cell),
            key: cell.attrs.run_key(),
        }
    }

    /// One underline, in the shape the cell asked for (`SGR 4:n`).
    ///
    /// Every style is built from the one primitive the display list already has, a filled
    /// rectangle, rather than by adding a shader or a new `DrawCmd`. A dash is a rect; a
    /// double is two; a curly is a staircase of them.
    ///
    /// The staircase is the only one that costs anything, so it is stepped in whole
    /// *pixels of period*, not per pixel: a wave with a period of roughly a third of a
    /// cell reads as a squiggle at any size a terminal is legible at, and a run of 20
    /// underlined cells costs a few dozen rects rather than several hundred. Underlines
    /// are drawn per *run* of same-attribute cells, not per cell, so an editor squiggling
    /// one diagnostic is a single call with one width.
    fn push_underline(&mut self, style: UnderlineStyle, x: i32, w: i32, baseline: i32, fg: Rgb) {
        let m = self.metrics;
        let color = fg.to_u32();
        let thickness = self.rule_thickness();
        let base = self.underline_rect(x, w, baseline);

        match style {
            UnderlineStyle::Single => self.list.push(DrawCmd::Fill { rect: base, color }),
            UnderlineStyle::Double => {
                // Two rules, with a gap of at least a pixel, sitting inside the space one
                // underline would have had so the second never collides with the row below.
                let gap = thickness.max(1);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        h: thickness,
                        ..base
                    },
                    color,
                });
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        y: base.y + thickness + gap,
                        h: thickness,
                        ..base
                    },
                    color,
                });
            }
            UnderlineStyle::Dotted | UnderlineStyle::Dashed => {
                // A dash and its gap, tiled across the run. Dotted is a short dash; the
                // difference is only the duty cycle.
                let (dash, gap) = if style == UnderlineStyle::Dotted {
                    (thickness, thickness)
                } else {
                    ((m.w / 3).max(2), (m.w / 4).max(2))
                };
                let mut at = base.x;
                let end = base.x + base.w;
                while at < end {
                    self.list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: at,
                            w: dash.min(end - at),
                            ..base
                        },
                        color,
                    });
                    at += dash + gap;
                }
            }
            UnderlineStyle::Curly => {
                // A triangle wave, drawn as a staircase of short rects. The amplitude is
                // the underline's own thickness, which keeps the squiggle inside the
                // descent at every font size instead of biting into the row below.
                let step = (m.w / 3).max(2);
                let amplitude = thickness;
                let mut at = base.x;
                let end = base.x + base.w;
                let mut up = true;
                while at < end {
                    let y = if up { base.y } else { base.y + amplitude };
                    self.list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: at,
                            y,
                            w: step.min(end - at),
                            h: thickness,
                        },
                        color,
                    });
                    at += step;
                    up = !up;
                }
            }
        }
    }

    /// The box an underline rule occupies: `w` pixels from `x`, just below `baseline`.
    /// Shared by the SGR underline and the hovered link's rule, so the two can never
    /// end up at different heights.
    fn underline_rect(&self, x: i32, w: i32, baseline: i32) -> Rect {
        Rect {
            x,
            y: baseline + (self.metrics.descent / 2).max(1),
            w,
            h: self.rule_thickness(),
        }
    }

    /// The underline and strike rules for a run, drawn as solid fills spanning the
    /// run's width in the run's foreground colour.
    fn push_decorations(&mut self, run: RunStyle, x: i32, run_w: i32, baseline: i32) {
        if run.key.contains(Attrs::UNDERLINE) {
            self.push_underline(run.key.underline_style(), x, run_w, baseline, run.fg);
        }
        if run.key.contains(Attrs::STRIKE) {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y: baseline - self.metrics.ascent / 3,
                    w: run_w,
                    h: self.rule_thickness(),
                },
                color: run.fg.to_u32(),
            });
        }
    }

    /// How thick a decoration rule is drawn, from the font size. Every rule the painter
    /// draws — underline, strike, the hovered link's — asks here, so they cannot come out
    /// at different weights on the same text.
    fn rule_thickness(&self) -> i32 {
        (self.metrics.size as i32 / 12).max(1)
    }

    /// One cell's glyph as a standalone [`DrawCmd::Text`] at its exact column,
    /// used for wide glyphs and astral runes (base rune plus any combining marks).
    /// A hidden or blank cell draws no glyph, but still takes its decorations: an
    /// underline runs under the cells it covers whether or not they have ink, the
    /// same rule [`Self::push_run`] follows for a styled trailing space.
    fn push_glyph(&mut self, row: usize, col: usize, cell: Cell, baseline: i32) {
        let m = self.metrics;
        let x = self.cell_x(col);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let (fg, bg) = self.resolve(cell, false);
        let inked = !cell.attrs.contains(Attrs::HIDDEN)
            && (cell.rune != ' ' || !self.marks_empty(row, col));
        if inked {
            let mut text = self.take_string();
            text.push(cell.rune);
            text.extend(self.marks(row, col));
            self.list.push(DrawCmd::Text {
                bounds: text_bounds(x, baseline, width_cells * m.w, m),
                x,
                baseline,
                face: FaceKey::Prose {
                    size: m.size,
                    style: style_of(cell.attrs),
                },
                color: fg.to_u32(),
                bg: bg.to_u32(),
                fade: None,
                text,
            });
        }
        // A wide glyph is drawn standalone, so it never rides a run's rule; without
        // this it would be the one gap in an underlined span (an SGR 4 CJK character,
        // or a hovered link with one in its path).
        let run = self.run_style(cell);
        self.push_decorations(run, x, width_cells * m.w, baseline);
    }

    /// The cursor, drawn last so it stacks over the cell it sits on. Nothing is
    /// drawn when the cursor is hidden or its position is off the grid. A block
    /// cursor on a wide glyph covers both halves and, when focused, re-stamps the
    /// glyph in the cell background so it reads as inverted.
    fn cursor(&mut self, cursor: CursorRender, cols: usize, rows: usize) {
        if !cursor.visible {
            return;
        }
        let (cr, mut cc) = self.screen.cursor();
        // The live cursor row maps to `cr + view_offset` on screen; when scrolled
        // far enough into history it falls below the window and is not drawn.
        let Some(dr) = cr.checked_add(self.screen.view_offset()) else {
            return;
        };
        if dr >= rows || cc >= cols {
            return;
        }
        // A cursor parked on a wide spacer belongs to the leader on its left.
        if self.cell(dr, cc).is_wide_spacer() && cc > 0 {
            cc -= 1;
        }
        let cell = self.cell(dr, cc);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let m = self.metrics;
        let x = self.cell_x(cc);
        let y = self.cell_y(dr);
        let color = self.theme.cursor.to_u32();
        let w = width_cells * m.w;
        match cursor.shape {
            CursorShape::Block => self.block_cursor(dr, cc, cursor.focused),
            // The padlock stands in for the cursor, in the cursor's colour and on the cell's
            // own background: no block behind it, and it looks the same focused or not. It
            // is a status light rather than a caret style, and "this window is waiting for a
            // password" is worth reading from across the desk.
            //
            // Without a symbols font there is no padlock to draw, and a tofu box is worse
            // than no lock at all, so the cursor quietly stays whatever the child asked for.
            CursorShape::Lock if m.lock_glyph => {
                // The lock is the whole message, so it gets a clean cell: paint out whatever
                // is under the cursor (blank at a real password prompt, but nothing promises
                // where the cursor is parked) and stand the glyph on that background.
                let (_, behind) = self.resolve(cell, false);
                self.list.push(DrawCmd::Fill {
                    rect: Rect { x, y, w, h: m.h },
                    color: behind.to_u32(),
                });
                let baseline = self.baseline(dr);
                let mut text = self.take_string();
                text.push(LOCK_GLYPH);
                self.list.push(DrawCmd::Text {
                    bounds: text_bounds(x, baseline, w, m),
                    x,
                    baseline,
                    face: FaceKey::Prose {
                        size: m.size,
                        style: FontStyle::Regular,
                    },
                    color,
                    bg: behind.to_u32(),
                    fade: None,
                    text,
                });
            }
            CursorShape::Lock => self.block_cursor(dr, cc, cursor.focused),
            CursorShape::Bar => self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y,
                    w: BAR_CURSOR_WIDTH,
                    h: m.h,
                },
                color,
            }),
            CursorShape::Underline => {
                let thickness = (m.size as i32 / 8).max(2);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y: y + m.h - thickness,
                        w: width_cells * m.w,
                        h: thickness,
                    },
                    color,
                });
            }
        }
    }

    /// The overlay scrollbar down the grid's right edge: a faint trough (only once the
    /// pointer has grown the bar into a slider) and the thumb, both tinted onto the theme
    /// background by how lit the bar is. A bar out of sight, or a screen with no history
    /// behind it (which includes every alt screen), draws nothing.
    fn push_scrollbar(&mut self, surface: (i32, i32), rows: usize) {
        let lit = self.scrollbar.lit();
        if lit <= 0.0 {
            return;
        }
        let view = scroll_column(surface, self.origin, self.metrics, rows);
        let track = scroll_lane(view, self.scale).track;
        let (content, viewport) = self.screen.scroll_extent();
        let Some(full) = scroll::thumb(
            track,
            viewport,
            content,
            self.screen.scroll_position(),
            scroll_min_thumb(self.scale),
        ) else {
            return;
        };
        let behind = self.theme.bg.to_u32();
        let grown = self.scrollbar.wide();
        // The bar grows out of the window edge: the thumb keeps its right edge and widens
        // leftward, and the trough appears under it as it goes.
        if grown > 0.0 {
            self.list.push(DrawCmd::Fill {
                rect: track,
                color: pixel::tint(behind, SCROLL_INK, SCROLL_TROUGH_COVER * grown * lit),
            });
        }
        let (thin, wide) = (self.scale.px(SCROLL_THIN), self.scale.px(SCROLL_WIDE));
        let w = thin + ((wide - thin) as f32 * grown).round() as i32;
        let rect = Rect {
            x: full.x + full.w - w,
            w,
            ..full
        };
        // The thumb lifts under the pointer, so a bar that is merely reporting a scroll
        // reads quieter than one asking to be grabbed.
        let cover = SCROLL_THUMB_COVER + (SCROLL_THUMB_HOVER_COVER - SCROLL_THUMB_COVER) * grown;
        self.list.push(DrawCmd::RoundRect {
            rect,
            color: pixel::tint(behind, SCROLL_INK, cover * lit),
            radius: w / 2,
            corners: RoundedCorners::Both,
        });
    }

    /// Re-draw the glyph beneath a focused block cursor in the cell's background
    /// colour, so the character shows as a cutout in the cursor block.
    fn stamp_inverted_glyph(&mut self, row: usize, col: usize, cell: Cell, width_cells: i32) {
        if cell.rune == ' ' && self.marks_empty(row, col) {
            return;
        }
        let m = self.metrics;
        let x = self.cell_x(col);
        let baseline = self.baseline(row);
        let mut text = self.take_string();
        text.push(cell.rune);
        text.extend(self.marks(row, col));
        // The inverted glyph takes the cell's own background (reverse honoured),
        // ignoring any selection so the cursor stays legible over a selection. It is
        // stamped over the cursor block, so that colour is the background the glyph
        // anti-aliasing is weighted against.
        let (_, ink) = self.resolve(cell, false);
        self.list.push(DrawCmd::Text {
            bounds: text_bounds(x, baseline, width_cells * m.w, m),
            x,
            baseline,
            face: FaceKey::Prose {
                size: m.size,
                style: style_of(cell.attrs),
            },
            color: ink.to_u32(),
            bg: self.theme.cursor.to_u32(),
            fade: None,
            text,
        });
    }

    /// The block cursor: filled with the glyph inverted out of it when the window has
    /// focus, a hollow outline when it does not (the xterm convention). Shared with the
    /// lock, which falls back to it on a cell too small to draw a padlock in.
    fn block_cursor(&mut self, row: usize, col: usize, focused: bool) {
        let m = self.metrics;
        let cell = self.cell(row, col);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let (x, y) = (self.cell_x(col), self.cell_y(row));
        if !focused {
            self.hollow_block(x, y, width_cells * m.w);
            return;
        }
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x,
                y,
                w: width_cells * m.w,
                h: m.h,
            },
            color: self.theme.cursor.to_u32(),
        });
        self.stamp_inverted_glyph(row, col, cell, width_cells);
    }

    /// A hollow block outline (unfocused window) as four thin edge fills.
    fn hollow_block(&mut self, x: i32, y: i32, w: i32) {
        let m = self.metrics;
        let color = self.theme.cursor.to_u32();
        let t = 1;
        let edges = [
            Rect { x, y, w, h: t }, // top
            Rect {
                x,
                y: y + m.h - t,
                w,
                h: t,
            }, // bottom
            Rect { x, y, w: t, h: m.h }, // left
            Rect {
                x: x + w - t,
                y,
                w: t,
                h: m.h,
            }, // right
        ];
        for rect in edges {
            self.list.push(DrawCmd::Fill { rect, color });
        }
    }

    /// The cell shown at display `(row, col)`, honouring the scroll offset (the
    /// live cell when pinned to the bottom). Every content path goes through this
    /// so scrollback and the live screen paint identically.
    ///
    /// Deliberately does nothing else. This is the painter's hottest funnel — four
    /// calls per cell, across the background runs, the run scan, and the glyph pass —
    /// so anything tested here is tested a quarter of a million times a frame. An
    /// earlier cut of the hovered link handed the cell out with `UNDERLINE` already
    /// set, which reads nicely and cost 13% of the frame; the rule is drawn from a
    /// per-row column span instead (see [`Self::push_hover_rule`]).
    fn cell(&self, row: usize, col: usize) -> Cell {
        self.screen.view_cell(row, col)
    }

    /// Combining marks at display `(row, col)`, honouring the scroll offset, in
    /// arrival order; empty if none.
    fn marks(&self, row: usize, col: usize) -> impl Iterator<Item = char> + '_ {
        self.screen.view_marks(row, col)
    }

    /// The resolved foreground colour of a cell (reverse and dim applied).
    ///
    /// Deliberately not [`Self::resolve`]`(cell, false).0`. Both grounds resolve
    /// through the theme, and the painter's two per-cell scans each want exactly one
    /// of them — the background scan the background, the run scan the foreground — so
    /// resolving the pair and dropping half doubled the theme lookups on the hottest
    /// loops it has. [`Self::resolve`] stays for the callers that genuinely want both,
    /// which run once per run or per row, not per cell.
    fn cell_fg(&self, cell: Cell) -> Rgb {
        // Reverse means the foreground is drawn from the cell's *background* colour;
        // dim then darkens whichever one it landed on, exactly as `resolve` dims after
        // it swaps.
        let fg = if cell.attrs.contains(Attrs::REVERSE) {
            cell.bg.resolve(self.theme, Ground::Background)
        } else {
            cell.fg.resolve(self.theme, Ground::Foreground)
        };
        if cell.attrs.contains(Attrs::DIM) {
            dim(fg)
        } else {
            fg
        }
    }

    /// The resolved background colour of cell `(row, col)`; `selected` (precomputed
    /// by the caller from the content-trimmed row span) swaps in the selection tint.
    ///
    /// A selected cell takes the tint whatever it holds, so that case resolves nothing
    /// and does not even read the cell. Dim never reaches a background: `resolve`
    /// applies it to the foreground after reverse has already swapped the two.
    fn cell_bg(&self, row: usize, col: usize, selected: bool) -> Rgb {
        if selected {
            return SELECTION_BG;
        }
        self.cell_bg_of(self.cell(row, col))
    }

    /// The resolved background a cell asks for, before selection has its say.
    fn cell_bg_of(&self, cell: Cell) -> Rgb {
        if cell.attrs.contains(Attrs::REVERSE) {
            cell.fg.resolve(self.theme, Ground::Foreground)
        } else {
            cell.bg.resolve(self.theme, Ground::Background)
        }
    }

    /// The inclusive column span of `row` covered by a reading-order [`CellSpan`]: the
    /// first row from its start column to the right edge, whole rows in between, and
    /// the last row from the left edge to its end column. `None` when `row` lies
    /// outside it. One inclusive range per row is what keeps the per-cell test that
    /// consumes it O(1), and both things the painter overlays — the selection band and
    /// the hovered link's rule — are this same shape.
    fn span_cols(&self, span: CellSpan, row: usize) -> Option<(usize, usize)> {
        self.row_in_span(
            row >= span.start.0 && row <= span.end.0,
            row == span.start.0,
            row == span.end.0,
            span.start.1,
            span.end.1,
        )
    }

    /// One row's inclusive column range inside a reading-order span: from the start
    /// column on the row the span opens on, to the end column on the row it closes on,
    /// edge to edge on every row between. `None` when the row is outside it.
    ///
    /// Takes the three row comparisons already made rather than the span itself, because
    /// the painter's two overlays index rows differently — the hovered link by display
    /// row, the selection by [`AbsRow`], since it must not drift when the child prints —
    /// and only the comparisons differ. The clamping rule is the part that must not, so
    /// it lives here once.
    fn row_in_span(
        &self,
        inside: bool,
        at_start: bool,
        at_end: bool,
        start_col: usize,
        end_col: usize,
    ) -> Option<(usize, usize)> {
        if !inside {
            return None;
        }
        let last_col = self.screen.dimensions().0.saturating_sub(1);
        let first = if at_start { start_col } else { 0 };
        let last = if at_end { end_col } else { last_col }.min(last_col);
        Some((first, last))
    }

    /// The inclusive column span of `row` the selection highlights: the whole
    /// geometric span the drag covers, blank cells included, so dragging across the
    /// empty area below the prompt highlights it, the way xterm/wezterm/ghostty do.
    /// The paint is deliberately wider than [`Screen::selection_text`] copies: it
    /// shows the drag for feedback while the copy trims trailing blanks.
    ///
    /// The selection is held in absolute rows, so it is resolved against the display
    /// here, at paint time. That is also where the clipping comes from for free: a
    /// selection whose top has scrolled above the window simply has no display row
    /// inside it until the part you can see, so the visible part still highlights.
    fn selection_cols(&self, row: usize) -> Option<(usize, usize)> {
        let (start, end) = self.selection?.ordered();
        let abs = self.screen.abs_row(row);
        self.row_in_span(
            abs >= start.0 && abs <= end.0,
            abs == start.0,
            abs == end.0,
            start.1,
            end.1,
        )
    }

    /// The underline under the hovered hyperlink where it crosses `row`, drawn as one
    /// rule over its columns. Nothing on a row the link does not reach, so an ordinary
    /// frame pays a single `None` check per row for the whole feature.
    ///
    /// It takes the colour of the link's first cell on the row, so the rule reads as
    /// part of the text it underlines rather than as chrome laid over it.
    fn push_hover_rule(&mut self, row: usize) {
        let Some(hover) = self.hover else {
            return;
        };
        let Some((first, last)) = self.span_cols(hover, row) else {
            return;
        };
        let fg = self.resolve(self.cell(row, first), false).0;
        let rect = self.underline_rect(
            self.cell_x(first),
            (last + 1 - first) as i32 * self.metrics.w,
            self.baseline(row),
        );
        self.list.push(DrawCmd::Fill {
            rect,
            color: fg.to_u32(),
        });
    }

    /// Resolve a cell's `(fg, bg)` to concrete colours: reverse swaps the two, dim
    /// darkens the foreground, and a selected cell takes the selection background.
    fn resolve(&self, cell: Cell, selected: bool) -> (Rgb, Rgb) {
        let mut fg = cell.fg.resolve(self.theme, Ground::Foreground);
        let mut bg = cell.bg.resolve(self.theme, Ground::Background);
        if cell.attrs.contains(Attrs::REVERSE) {
            std::mem::swap(&mut fg, &mut bg);
        }
        if cell.attrs.contains(Attrs::DIM) {
            fg = dim(fg);
        }
        if selected {
            bg = SELECTION_BG;
        }
        (fg, bg)
    }

    /// Whether cell `(row, col)` has no combining marks (a blank base rune with no
    /// marks draws nothing).
    fn marks_empty(&self, row: usize, col: usize) -> bool {
        !self.screen.view_has_marks(row, col)
    }
}

/// Append one proportional text run (chrome/UI text) at `x` on `baseline` in
/// `face`, drawing the owned string from the pool so a steady repaint allocates
/// nothing. Unlike [`push_cell_text`] the glyphs advance by the font's own metrics
/// rather than a fixed cell pitch, so this is for a UI-sans label, never grid text;
/// `run_w` is the run's already-measured pixel width, used only for the damage
/// bounds. `fade` ramps the run's ink away over a span of `x` (see [`Fade`]), for a
/// truncated label that dissolves rather than ending on an ellipsis. An empty run
/// pushes nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_text_run(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    text: &str,
    x: i32,
    run_w: i32,
    baseline: i32,
    face: FaceKey,
    color: u32,
    bg: u32,
    fade: Option<Fade>,
    metrics: CellMetrics,
) {
    if text.is_empty() {
        return;
    }
    let mut owned = take_string_with_capacity(strings, text.len());
    owned.push_str(text);
    out.push(DrawCmd::Text {
        bounds: text_bounds(x, baseline, run_w, metrics),
        x,
        baseline,
        face,
        color,
        bg,
        fade,
        text: owned,
    });
}

/// Append styled cell text at a fixed-pitch origin, recycling every command's
/// owned string from `strings`. Single-column clusters coalesce into `Cells`;
/// wide or astral clusters break out as `Text` at their exact cell position so
/// they can never desynchronise the fixed-pitch run that follows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_cell_text(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    text: &str,
    x: i32,
    baseline: i32,
    metrics: CellMetrics,
    face: FaceKey,
    color: u32,
    bg: u32,
) {
    /// The run being accumulated: where it starts, how many cells it covers, and the
    /// pooled buffer its text is going into. One value rather than a `Option<String>`
    /// beside a counter, so "there is a run" and "the run has storage" cannot disagree.
    struct Pending {
        start_cell: usize,
        cells: usize,
        text: String,
    }
    let flush = |out: &mut DisplayList, run: Pending| {
        push_owned_cells(
            out,
            run.text,
            x + run.start_cell as i32 * metrics.w,
            baseline,
            run.cells,
            metrics,
            face,
            color,
            bg,
        );
    };

    let mut pen_cells = 0usize;
    let mut run: Option<Pending> = None;
    for (_, cluster) in grapheme::graphemes(text) {
        let width = display_cluster_width(cluster).max(1);
        if width == 1 && cluster.chars().all(cells_safe) {
            let pending = run.get_or_insert_with(|| Pending {
                start_cell: pen_cells,
                cells: 0,
                text: take_string_with_capacity(strings, text.len()),
            });
            pending.text.push_str(cluster);
            pending.cells += 1;
        } else {
            if let Some(pending) = run.take() {
                flush(out, pending);
            }
            let mut glyph = take_string_with_capacity(strings, cluster.len());
            glyph.push_str(cluster);
            let glyph_x = x + pen_cells as i32 * metrics.w;
            out.push(DrawCmd::Text {
                bounds: text_bounds(glyph_x, baseline, width as i32 * metrics.w, metrics),
                x: glyph_x,
                baseline,
                face,
                color,
                bg,
                fade: None,
                text: glyph,
            });
        }
        pen_cells += width;
    }
    if let Some(pending) = run.take() {
        flush(out, pending);
    }
}

/// Display columns occupied by one already-segmented grapheme cluster.
///
/// One line, because the rule itself now lives in `width.rs` beside the scalar width —
/// the grid decides which cells a cluster occupies and the renderer decides how wide to
/// draw it, and the two measuring differently is how a cursor drifts from its glyphs.
pub(crate) fn display_cluster_width(cluster: &str) -> usize {
    usize::from(crate::width::cluster_width(cluster))
}

fn take_string(strings: &mut Vec<String>) -> String {
    strings.pop().unwrap_or_default()
}

/// Pull a pooled buffer that already holds `min_capacity`, so filling the run
/// never reallocates (a plain `pop` + `reserve` would realloc whenever the popped
/// buffer was the wrong size, and that shows up as an allocation in the perf
/// gate). Scanning from the back finds the most-recently-retired buffer first, so
/// in steady state — where retired runs are all similar sizes — the first probe
/// hits and this is O(1); the O(n) scan only bites when no buffer is big enough,
/// which is exactly the case where reallocating would otherwise cost more.
fn take_string_with_capacity(strings: &mut Vec<String>, min_capacity: usize) -> String {
    if let Some(index) = strings
        .iter()
        .rposition(|string| string.capacity() >= min_capacity)
    {
        return strings.swap_remove(index);
    }
    let mut string = take_string(strings);
    string.reserve(min_capacity);
    string
}

#[allow(clippy::too_many_arguments)]
fn push_owned_cells(
    out: &mut DisplayList,
    text: String,
    x: i32,
    baseline: i32,
    cells: usize,
    metrics: CellMetrics,
    face: FaceKey,
    color: u32,
    bg: u32,
) {
    out.push(DrawCmd::Cells {
        bounds: text_bounds(x, baseline, cells as i32 * metrics.w, metrics),
        x,
        baseline,
        cell_w: metrics.w,
        face,
        color,
        bg,
        text,
    });
}

/// The run's cell box, padded for bearing: a glyph's ink can spill past its cell
/// (an italic tail, a box-drawing overhang, an accent riding above the ascent), so
/// the damage and clip rectangle is grown to never shear a glyph's edge. The box is
/// measured from the cell, not from ascent-to-descent: a box-drawing glyph fills its
/// cell exactly, and clipping to the ink box would shave its top edge.
fn text_bounds(x: i32, baseline: i32, run_w: i32, m: CellMetrics) -> Rect {
    let pad_x = (m.size as i32 / 2).max(2);
    let pad_y = (m.size as i32 / 4).max(1);
    Rect {
        x: x - pad_x,
        y: baseline - m.baseline - pad_y,
        w: run_w.max(0) + 2 * pad_x,
        h: m.h + 2 * pad_y,
    }
}

/// Whether a rune is safe to batch into a fixed-pitch [`DrawCmd::Cells`] run.
///
/// The batcher recovers cell boundaries by segmenting the run's text and using each
/// cluster's index as the pen multiplier, so the run must hold one cluster per cell. Two
/// families break that and are excluded here:
///
/// - **astral scalars**, the regional indicators and emoji that pair up into flags and
///   ZWJ sequences;
/// - **the mergeable BMP scalars** ([`grapheme::joins_across_cells`]) — Indic spacing
///   vowel signs, `Prepend`, and the trailing consonant of a conjunct. These are a full
///   column wide on the grid *and* joinable by UAX #29, which is the exact combination
///   the fixed-pitch path cannot represent.
///
/// Everything else a segmenter would merge is zero-width and never occupies a cell at
/// all: it lives in the grid's combining-mark side list, and the run text picks it up
/// with the cell it belongs to.
///
/// An excluded rune is drawn standalone, the same treatment a wide rune already gets.
fn cells_safe(rune: char) -> bool {
    (rune as u32) <= 0xFFFF && !grapheme::joins_across_cells(rune)
}

/// The font style a cell's bold/italic attributes select. Takes the attributes rather
/// than the cell so a run can ask it of its own [`Attrs::run_key`], which is where the
/// two bits it reads are preserved.
fn style_of(attrs: Attrs) -> FontStyle {
    match (attrs.contains(Attrs::BOLD), attrs.contains(Attrs::ITALIC)) {
        (true, true) => FontStyle::BoldItalic,
        (true, false) => FontStyle::Bold,
        (false, true) => FontStyle::Italic,
        (false, false) => FontStyle::Regular,
    }
}

/// Darken a colour to two-thirds intensity for the SGR dim (faint) attribute.
/// Done in `u16` so the doubling can never overflow a channel.
fn dim(c: Rgb) -> Rgb {
    let scale = |v: u8| ((u16::from(v) * 2) / 3) as u8;
    Rgb::new(scale(c.r), scale(c.g), scale(c.b))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::color::Color;

    /// Synthetic metrics: a 10x20 cell so column/row math reads off by eye, with a
    /// 2px line gap (ascent 14 + descent 4 < h 20), like every real font ships. The
    /// gap sits above the ascent, so the baseline is 16, *not* the ascent: any test
    /// that seats text by `M.ascent` is asserting the bug this metric exists to catch.
    const M: CellMetrics = CellMetrics {
        size: 16,
        w: 10,
        h: 20,
        baseline: 16,
        ascent: 14,
        descent: 4,
        lock_glyph: true,
    };

    fn feed(s: &mut Screen, bytes: &[u8]) {
        let mut p = crate::vt::Parser::new();
        p.advance_bytes(s, bytes);
    }

    /// `cell_fg` and `cell_bg` each resolve one ground where [`Painter::resolve`]
    /// resolves both, which is only sound if they arrive at the same colours it
    /// would. Reverse and dim interact (dim lands on the foreground *after* reverse
    /// has swapped the grounds, so it must never darken a background), and selection
    /// overrides a background outright, so the agreement is asserted over every
    /// combination of the attributes involved rather than argued for in a comment.
    #[test]
    fn the_split_grounds_resolve_exactly_as_the_pair_does() {
        let screen = Screen::new(1, 1);
        let theme = Theme::default();
        let bar = Scrollbar::hidden();
        let (mut list, mut strings) = (DisplayList::new(), Vec::new());
        let painter = Painter {
            screen: &screen,
            theme: &theme,
            bell: false,
            metrics: M,
            origin: (0, 0),
            selection: None,
            hover: None,
            scale: Scale::ONE,
            scrollbar: &bar,
            list: &mut list,
            strings: &mut strings,
        };

        let colors = [
            Color::Default,
            Color::Ansi(3),
            Color::Indexed(200),
            Color::Rgb(10, 20, 30),
        ];
        let attr_sets = [
            Attrs::empty(),
            Attrs::REVERSE,
            Attrs::DIM,
            Attrs::REVERSE | Attrs::DIM,
            Attrs::BOLD | Attrs::REVERSE | Attrs::DIM,
        ];
        for fg in colors {
            for bg in colors {
                for attrs in attr_sets {
                    let cell = Cell {
                        rune: 'x',
                        fg,
                        bg,
                        attrs,
                        ..Cell::default()
                    };
                    let (want_fg, want_bg) = painter.resolve(cell, false);
                    assert_eq!(
                        painter.cell_fg(cell),
                        want_fg,
                        "foreground diverged for {fg:?} on {bg:?} with {attrs:?}"
                    );
                    assert_eq!(
                        painter.cell_bg_of(cell),
                        want_bg,
                        "background diverged for {fg:?} on {bg:?} with {attrs:?}"
                    );
                    // And the reason `cell_bg` can answer a selected cell without
                    // resolving anything at all.
                    assert_eq!(
                        painter.resolve(cell, true).1,
                        SELECTION_BG,
                        "a selected cell takes the tint whatever it holds"
                    );
                }
            }
        }
    }

    /// A bar at rest, for the frames that are not about the scrollbar: it is out of
    /// sight, so it adds nothing to the list they assert over.
    static HIDDEN_BAR: Scrollbar = Scrollbar::hidden();

    /// The frame inputs for `s`: default theme, a surface exactly the grid's size, no
    /// origin inset, no selection, no hovered link, the cursor hidden and the scrollbar
    /// out of sight (so a test about content is not perturbed by either). A case varies
    /// one field with struct update syntax: `FrameInputs { selection, ..inputs(&s, &t) }`.
    fn inputs<'a>(s: &'a Screen, theme: &'a Theme) -> FrameInputs<'a> {
        FrameInputs {
            screen: s,
            bell: false,
            theme,
            metrics: M,
            surface: (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            origin: (0, 0),
            cursor: CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            selection: None,
            hover: None,
            scale: Scale::ONE,
            scrollbar: &HIDDEN_BAR,
        }
    }

    /// A bar a scroll has lit, ticked past its fade-in so it draws at full strength but
    /// has not been grown by a pointer (`wide` is still 0: a thin indicator, no trough).
    fn lit_bar() -> Scrollbar {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.flash(start);
        for f in 1..=8 {
            bar.tick(start + Duration::from_millis(f * 16));
        }
        assert_eq!((bar.lit(), bar.wide()), (1.0, 0.0), "lit, still thin");
        bar
    }

    /// The frame `s` paints with `bar` down its edge.
    fn list_with_bar(s: &Screen, bar: &Scrollbar) -> DisplayList {
        build_display_list(&FrameInputs {
            scrollbar: bar,
            ..inputs(s, &Theme::default())
        })
    }

    /// The scrollbar's thumb: the only rounded rectangle this painter ever draws.
    fn thumb_of(list: &[DrawCmd]) -> Option<Rect> {
        list.iter().find_map(|c| match c {
            DrawCmd::RoundRect { rect, .. } => Some(*rect),
            _ => None,
        })
    }

    /// Build a list for a screen with the default theme, no selection, cursor
    /// hidden (so the tests that are about content are not perturbed by it).
    fn list_of(s: &Screen) -> DisplayList {
        build_display_list(&inputs(s, &Theme::default()))
    }

    /// A selection over the display cells `a`..`b`, resolved to absolute rows the way a
    /// pointer drag resolves one.
    fn sel(s: &Screen, a: (usize, usize), b: (usize, usize)) -> Selection {
        Selection {
            anchor: (s.abs_row(a.0), a.1),
            head: (s.abs_row(b.0), b.1),
            epoch: s.row_epoch(),
        }
    }

    /// [`list_of`] with a selection over it.
    fn list_selecting(s: &Screen, selection: Selection) -> DisplayList {
        build_display_list(&FrameInputs {
            selection: Some(selection),
            ..inputs(s, &Theme::default())
        })
    }

    /// [`list_of`] with a hovered hyperlink over it.
    fn list_hovering(s: &Screen, hover: CellSpan) -> DisplayList {
        build_display_list(&FrameInputs {
            hover: Some(hover),
            ..inputs(s, &Theme::default())
        })
    }

    fn cells_runs(list: &[DrawCmd]) -> Vec<(i32, i32, String)> {
        list.iter()
            .filter_map(|c| match c {
                DrawCmd::Cells {
                    x, baseline, text, ..
                } => Some((*x, *baseline, text.clone())),
                _ => None,
            })
            .collect()
    }

    fn fills(list: &[DrawCmd]) -> Vec<(Rect, u32)> {
        list.iter()
            .filter_map(|c| match c {
                DrawCmd::Fill { rect, color } => Some((*rect, *color)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn blank_screen_is_just_the_background() {
        let s = Screen::new(4, 2);
        let list = list_of(&s);
        // One base fill covering the whole surface, nothing else.
        assert_eq!(
            fills(&list),
            vec![(
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 40
                },
                Theme::default().bg.to_u32()
            )]
        );
        assert!(cells_runs(&list).is_empty(), "no glyphs on a blank screen");
    }

    #[test]
    fn plain_text_is_one_fixed_pitch_run_per_line() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"hello");
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![(0, M.baseline, "hello".to_string())],
            "the whole line batches into one run at column 0, seated on the baseline"
        );
    }

    #[test]
    fn a_colour_change_splits_the_run() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"ab\x1b[31mcd");
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![
                (0, M.baseline, "ab".to_string()),
                (2 * M.w, M.baseline, "cd".to_string()),
            ],
            "the run breaks where the colour changes; the second starts at its column"
        );
    }

    #[test]
    fn trailing_blanks_emit_no_run() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"hi");
        let runs = cells_runs(&list_of(&s));
        // "hi" then eight blanks: the blanks are default cells, so only "hi" is a
        // run and there are no stray blank runs.
        assert_eq!(runs, vec![(0, M.baseline, "hi".to_string())]);
    }

    #[test]
    fn a_coloured_background_is_a_fill_band() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[41mXX"); // red background, two cells
        let f = fills(&list_of(&s));
        let red = Color::Ansi(1).resolve(&Theme::default(), Ground::Background);
        // Base fill first, then the two-cell red band.
        assert_eq!(f[0].1, Theme::default().bg.to_u32());
        assert_eq!(
            f[1],
            (
                Rect {
                    x: 0,
                    y: 0,
                    w: 2 * M.w,
                    h: M.h
                },
                red.to_u32()
            ),
            "adjacent same-bg cells coalesce into one band"
        );
    }

    #[test]
    fn reverse_video_swaps_foreground_and_background() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"\x1b[7mA");
        let list = list_of(&s);
        let t = Theme::default();
        // The cell background becomes the default fg (a fill), and the glyph is
        // drawn in the default bg.
        let band = fills(&list)
            .into_iter()
            .find(|(r, _)| r.x == 0 && r.w == M.w)
            .expect("a reverse cell paints its background");
        assert_eq!(band.1, t.fg.to_u32());
        let run = cells_runs(&list).into_iter().next().expect("glyph run");
        // Foreground colour lives on the Cells command, not returned by helper;
        // re-read it here.
        if let DrawCmd::Cells { color, .. } = list
            .iter()
            .find(|c| matches!(c, DrawCmd::Cells { .. }))
            .unwrap()
        {
            assert_eq!(*color, t.bg.to_u32(), "reversed glyph uses the default bg");
        }
        assert_eq!(run.2, "A");
    }

    #[test]
    fn wide_char_is_broken_out_and_the_spacer_is_skipped() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, "a漢b".as_bytes());
        let list = list_of(&s);
        // 'a' at col 0 (run), 漢 standalone Text at col 1 (two cells wide), 'b' at
        // col 3 (the spacer at col 2 drew nothing).
        let texts: Vec<(i32, String)> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { x, text, .. } => Some((*x, text.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec![(M.w, "漢".to_string())]);
        assert_eq!(
            cells_runs(&list),
            vec![
                (0, M.baseline, "a".to_string()),
                (3 * M.w, M.baseline, "b".to_string()),
            ],
            "'b' lands at column 3, past the wide glyph's spacer"
        );
    }

    #[test]
    fn combining_mark_rides_with_its_base_in_the_run() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, "e\u{0301}x".as_bytes()); // é (e + combining acute), then x
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![(0, M.baseline, "e\u{0301}x".to_string())],
            "the mark stays glued to its base; both cells share one run"
        );
    }

    #[test]
    fn underline_adds_a_rule_under_the_run() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[4mok");
        let list = list_of(&s);
        let t = Theme::default();
        let rule = fills(&list)
            .into_iter()
            .find(|(r, _)| r.y > M.baseline) // below the baseline
            .expect("an underline rule");
        assert_eq!(rule.0.x, 0);
        assert_eq!(rule.0.w, 2 * M.w, "the rule spans the whole run");
        assert_eq!(rule.1, t.fg.to_u32());
    }

    #[test]
    fn each_underline_style_draws_a_different_shape() {
        // The point of the whole feature: an editor squiggling an error must not come out
        // looking like an editor underlining a link. Every style is built from the one
        // primitive the display list has (a filled rect), so the difference is in how many
        // and where.
        let under = |seq: &[u8]| {
            let mut s = Screen::new(6, 1);
            let mut p = crate::vt::Parser::new();
            p.advance_bytes(&mut s, seq);
            p.advance_bytes(&mut s, b"ok"); // two cells, so a run is 2 * M.w wide
            let t = Theme::default();
            let list = list_of(&s);
            fills(&list)
                .into_iter()
                .filter(|(r, c)| r.y > M.baseline && *c == t.fg.to_u32())
                .map(|(r, _)| r)
                .collect::<Vec<_>>()
        };

        // Single: one rule spanning the run.
        let single = under(b"\x1b[4m");
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].w, 2 * M.w);

        // Double: two rules, one below the other, both spanning the run.
        let double = under(b"\x1b[4:2m");
        assert_eq!(double.len(), 2);
        assert_eq!(double[0].w, 2 * M.w);
        assert_eq!(double[1].w, 2 * M.w);
        assert!(double[1].y > double[0].y, "the second sits below the first");

        // Curly: a staircase, alternating between two heights.
        let curly = under(b"\x1b[4:3m");
        assert!(curly.len() > 2, "a squiggle is more than a line");
        let heights: Vec<i32> = curly.iter().map(|r| r.y).collect();
        assert!(
            heights.windows(2).all(|w| w[0] != w[1]),
            "it goes up and down: {heights:?}"
        );
        assert_eq!(
            heights
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            2,
            "between exactly two heights, so it stays inside the descent"
        );

        // Dotted and dashed: a run of marks with gaps, at one height.
        for seq in [&b"\x1b[4:4m"[..], &b"\x1b[4:5m"[..]] {
            let dashes = under(seq);
            assert!(dashes.len() > 1, "{seq:?} is more than one mark");
            assert!(dashes.iter().all(|r| r.y == dashes[0].y), "{seq:?} is flat");
            assert!(
                dashes.iter().all(|r| r.w < 2 * M.w),
                "{seq:?} has gaps in it"
            );
        }

        // And the cost stays bounded: a squiggle under two cells is a handful of rects,
        // not one per pixel. The damage diff walks this list every painted frame.
        assert!(curly.len() <= 8, "{} rects for two cells", curly.len());
    }

    #[test]
    fn a_change_of_underline_shape_ends_the_run() {
        // Two diagnostics of different kinds side by side: an error squiggled curly, a hint
        // dotted. A run carries *one* set of decorations, drawn from its first cell, so a
        // run that batches across a change of shape draws the whole span in the first
        // shape and the hint comes out squiggled. Same fg, same face, both underlined —
        // every other reason to break the run is absent, which is exactly why the shape has
        // to be one of them.
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[4:3maa\x1b[4:4mbb");
        let list = list_of(&s);
        assert_eq!(
            cells_runs(&list)
                .iter()
                .map(|(x, _, text)| (*x, text.clone()))
                .collect::<Vec<_>>(),
            vec![(0, "aa".to_string()), (2 * M.w, "bb".to_string())],
            "the run ends where the shape changes"
        );

        // And the two halves really are drawn as different shapes.
        let t = Theme::default();
        let rules: Vec<Rect> = fills(&list)
            .into_iter()
            .filter(|(r, c)| r.y > M.baseline && *c == t.fg.to_u32())
            .map(|(r, _)| r)
            .collect();
        let curly: Vec<i32> = rules
            .iter()
            .filter(|r| r.x < 2 * M.w)
            .map(|r| r.y)
            .collect();
        let dotted: Vec<i32> = rules
            .iter()
            .filter(|r| r.x >= 2 * M.w)
            .map(|r| r.y)
            .collect();
        assert!(
            curly.len() > 2 && curly.windows(2).all(|w| w[0] != w[1]),
            "the error stays a squiggle: {curly:?}"
        );
        assert!(
            dotted.len() > 1 && dotted.iter().all(|y| *y == dotted[0]),
            "and the hint stays flat: {dotted:?}"
        );
    }

    #[test]
    fn block_cursor_fills_its_cell_and_inverts_the_glyph() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X"); // cursor now rests at column 1 (after X)
        s.move_to(0, 0); // park it on the glyph
        let t = Theme::default();
        let list = build_display_list(&FrameInputs {
            cursor: CursorRender::default(),
            ..inputs(&s, &t)
        });
        // The last commands are the cursor block then the inverted glyph.
        let cursor_fill = fills(&list)
            .into_iter()
            .find(|(r, c)| r.x == 0 && r.w == M.w && *c == t.cursor.to_u32())
            .expect("a filled cursor block");
        assert_eq!(
            cursor_fill.0,
            Rect {
                x: 0,
                y: 0,
                w: M.w,
                h: M.h
            }
        );
        // The inverted 'X' is a Text in the background colour, drawn last.
        match list.last().unwrap() {
            DrawCmd::Text { text, color, .. } => {
                assert_eq!(text, "X");
                assert_eq!(*color, t.bg.to_u32(), "the cursor glyph is inverted");
            }
            other => panic!("expected the inverted glyph last, got {other:?}"),
        }
    }

    #[test]
    fn a_nonzero_origin_insets_the_grid_but_not_the_background() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0); // park the cursor on the glyph
        let t = Theme::default();
        let (ox, oy) = (5, 5);
        let list = build_display_list(&FrameInputs {
            origin: (ox, oy),
            cursor: CursorRender::default(),
            ..inputs(&s, &t)
        });
        // The base fill still covers the whole surface: the inset is a margin of
        // background around the grid, not a smaller canvas.
        assert_eq!(
            fills(&list)[0],
            (
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 20
                },
                t.bg.to_u32()
            )
        );
        // Glyph run and cursor block both ride the origin, so they stay aligned:
        // the run at (ox, oy + ascent), the cursor block at (ox, oy).
        assert_eq!(
            cells_runs(&list),
            vec![(ox, oy + M.baseline, "X".to_string())],
            "the run is shifted by the content origin"
        );
        let cursor_fill = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == t.cursor.to_u32())
            .expect("a filled cursor block");
        assert_eq!(
            cursor_fill.0,
            Rect {
                x: ox,
                y: oy,
                w: M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn bar_and_underline_cursors_do_not_invert() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let t = Theme::default();
        for shape in [CursorShape::Bar, CursorShape::Underline] {
            let list = build_display_list(&FrameInputs {
                cursor: CursorRender {
                    shape,
                    ..CursorRender::default()
                },
                ..inputs(&s, &t)
            });
            // No inverted glyph: the only Text/Cells for 'X' is the normal one.
            let glyphs = list
                .iter()
                .filter(|c| matches!(c, DrawCmd::Cells { .. } | DrawCmd::Text { .. }))
                .count();
            assert_eq!(glyphs, 1, "{shape:?} leaves the glyph alone");
        }
    }

    /// A `Lock` cursor over a cell holding 'X', at `metrics`. Returns the display list.
    fn lock_frame(metrics: CellMetrics, t: &Theme) -> DisplayList {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0); // park the cursor on the glyph
        build_display_list(&FrameInputs {
            metrics,
            cursor: CursorRender {
                shape: CursorShape::Lock,
                ..CursorRender::default()
            },
            ..inputs(&s, t)
        })
    }

    #[test]
    fn the_lock_typesets_the_padlock_glyph_on_a_cleared_cell() {
        // bnkterm's lock is *their* lock: the same Nerd Fonts codepoint wezterm and ghostty
        // typeset, not a lookalike drawn from rectangles. It stands in the cursor's colour
        // on the cell's own background, with no block behind it to swallow it, and the cell
        // is cleared first so whatever the cursor is parked on cannot show through.
        let t = Theme::default();
        let list = lock_frame(M, &t);

        let cell = Rect {
            x: 0,
            y: 0,
            w: M.w,
            h: M.h,
        };
        assert_eq!(
            fills(&list).last(),
            Some(&(cell, t.bg.to_u32())),
            "the cell is cleared to its own background before the glyph lands"
        );
        match list.last() {
            Some(DrawCmd::Text {
                text,
                color,
                bg,
                x,
                face,
                ..
            }) => {
                assert_eq!(
                    text,
                    &LOCK_GLYPH.to_string(),
                    "the padlock, not a drawn shape"
                );
                assert_eq!(*color, t.cursor.to_u32(), "in the cursor's colour");
                assert_eq!(*bg, t.bg.to_u32());
                assert_eq!(*x, 0, "in the cursor's cell");
                assert_eq!(
                    *face,
                    FaceKey::Prose {
                        size: M.size,
                        style: FontStyle::Regular
                    }
                );
            }
            other => panic!("expected the padlock glyph last, got {other:?}"),
        }
    }

    #[test]
    fn a_font_stack_without_the_padlock_falls_back_to_an_ordinary_cursor() {
        // The padlock is private-use, so a machine with no symbols font simply has no such
        // glyph. Typesetting it anyway would draw a tofu box: a cursor that is not merely
        // absent but actively confusing, on exactly the machines least equipped to explain
        // it. So the lock is silently declined and the caret stays what the child asked for.
        let t = Theme::default();
        let bare = CellMetrics {
            lock_glyph: false,
            ..M
        };
        let list = lock_frame(bare, &t);

        assert!(
            !list.iter().any(|c| matches!(
                c,
                DrawCmd::Text { text, .. } if text.contains(LOCK_GLYPH)
            )),
            "no padlock is typeset when the font stack has none"
        );
        // An ordinary focused block: filled cell, glyph inverted out of it.
        assert!(
            fills(&list).contains(&(
                Rect {
                    x: 0,
                    y: 0,
                    w: M.w,
                    h: M.h
                },
                t.cursor.to_u32()
            )),
            "the cursor falls back to a block, not to nothing"
        );
        match list.last() {
            Some(DrawCmd::Text { text, color, .. }) => {
                assert_eq!(text, "X", "the inverted glyph of a normal block cursor");
                assert_eq!(*color, t.bg.to_u32());
            }
            other => panic!("expected the inverted glyph last, got {other:?}"),
        }
    }

    #[test]
    fn the_lock_looks_the_same_focused_or_not() {
        // The lock is a status light, not a caret style: "this window is waiting for a
        // password" is worth reading from across the desk, focused or not.
        let t = Theme::default();
        let focused = lock_frame(M, &t);
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let unfocused = build_display_list(&FrameInputs {
            cursor: CursorRender {
                shape: CursorShape::Lock,
                focused: false,
                ..CursorRender::default()
            },
            ..inputs(&s, &t)
        });
        assert_eq!(
            focused.last(),
            unfocused.last(),
            "the padlock does not hollow out when the window loses focus"
        );
    }

    #[test]
    fn hidden_cursor_draws_nothing_extra() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let hidden = list_of(&s);
        // Same as a plain render: base fill + the one glyph run.
        assert_eq!(hidden.len(), 2);
    }

    /// The underline rules in `list`: the fills that sit below the baseline.
    fn rules(list: &DisplayList) -> Vec<Rect> {
        fills(list)
            .into_iter()
            .filter(|(r, _)| r.y > M.baseline)
            .map(|(r, _)| r)
            .collect()
    }

    #[test]
    fn a_hovered_link_is_underlined_across_exactly_its_cells() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"a http://x");
        // The link occupies cols 2..=9; the painter is told so and rules those cells.
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 2),
                end: (0, 9),
            },
        );
        assert_eq!(
            rules(&list),
            vec![Rect {
                x: 2 * M.w,
                y: M.baseline + (M.descent / 2).max(1),
                w: 8 * M.w,
                h: (M.size as i32 / 12).max(1),
            }],
            "one rule, starting at the link and spanning exactly its cells"
        );
        // The row's text is untouched: the rule is drawn over the glyphs, so hovering
        // never disturbs how they batch.
        assert_eq!(
            cells_runs(&list),
            cells_runs(&list_of(&s)),
            "the glyph runs are the same hovered or not"
        );
    }

    #[test]
    fn a_hovered_link_is_ruled_over_its_wide_glyphs_too() {
        // One rule over the whole span, so a wide glyph inside a link is covered like
        // any other cell rather than leaving a gap where the run breaks around it.
        let mut s = Screen::new(6, 1);
        feed(&mut s, "ab漢c".as_bytes()); // 漢 is wide: cols 2..=3
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 0),
                end: (0, 4),
            },
        );
        assert_eq!(rules(&list).len(), 1, "one rule, not one per run");
        assert_eq!(rules(&list)[0].x, 0);
        assert_eq!(
            rules(&list)[0].w,
            5 * M.w,
            "including the wide glyph's cells"
        );
    }

    #[test]
    fn a_hovered_link_wraps_onto_the_next_row() {
        // A link that runs off the right margin is ruled on both rows: to the edge on
        // the first, from the edge on the second.
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"ab https://x"); // wraps after col 5
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 3),
                end: (1, 5),
            },
        );
        let rules = rules(&list);
        assert_eq!(rules.len(), 2, "one rule per row: {rules:?}");
        assert_eq!(rules[0].x, 3 * M.w, "the first row starts at the link");
        assert_eq!(rules[0].w, 3 * M.w, "and runs to the right edge");
        assert_eq!(rules[1].x, 0, "the second row starts at the left edge");
        assert_eq!(rules[1].w, 6 * M.w, "and runs to where the link ends");
    }

    #[test]
    fn no_hover_means_no_underline() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"a http://x");
        assert!(
            rules(&list_of(&s)).is_empty(),
            "an unhovered link is plain text"
        );
    }

    #[test]
    fn a_wide_glyph_takes_its_sgr_underline() {
        // A wide glyph is drawn standalone, outside any run, so it draws its own rules
        // or none at all. Nothing to do with links: `\x1b[4m漢` must underline.
        let mut s = Screen::new(4, 1);
        feed(&mut s, "\x1b[4m漢".as_bytes());
        let rules = rules(&list_of(&s));
        assert_eq!(rules.len(), 1, "the wide glyph is underlined: {rules:?}");
        assert_eq!(rules[0].x, 0);
        assert_eq!(rules[0].w, 2 * M.w, "the rule spans both its cells");
    }

    #[test]
    fn selection_paints_the_selected_cells() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"abcdef");
        let list = list_selecting(&s, sel(&s, (0, 1), (0, 3)));
        // Columns 1..=3 get the selection background as one band.
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band");
        assert_eq!(
            band.0,
            Rect {
                x: M.w,
                y: 0,
                w: 3 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn selection_highlights_the_full_span_including_trailing_blanks() {
        // "abc" in a 6-wide row, selected edge to edge. The highlight spans all six
        // cells, the trailing blanks (cols 3..=5) included, matching wezterm/ghostty;
        // the copy still trims them (see grid::selection_text), so paint is wider.
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"abc");
        let list = list_selecting(&s, sel(&s, (0, 0), (0, 5)));
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band");
        assert_eq!(
            band.0,
            Rect {
                x: 0,
                y: 0,
                w: 6 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn selection_over_blank_cells_paints_the_full_band() {
        // Dragging across an empty region highlights the whole geometric span (the
        // "select the empty space below the prompt" behaviour): one full-width band,
        // even though the row holds no glyphs and would copy nothing.
        let s = Screen::new(6, 1);
        let list = list_selecting(&s, sel(&s, (0, 0), (0, 5)));
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band over the blank cells");
        assert_eq!(
            band.0,
            Rect {
                x: 0,
                y: 0,
                w: 6 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn a_selection_scrolled_off_the_top_still_paints_the_part_that_shows() {
        // The selection is held in absolute rows and resolved against the display at
        // paint time, so clipping is not a special case: rows above the window simply
        // have no display row inside the span, and the visible tail still highlights.
        let mut s = Screen::new(4, 2);
        feed(&mut s, b"aa\r\nbb"); // both rows live
        let selection = sel(&s, (0, 0), (1, 3)); // both rows selected
        feed(&mut s, b"\r\ncc"); // "aa" scrolls into history, out of the window

        let bands: Vec<Rect> = fills(&list_selecting(&s, selection))
            .into_iter()
            .filter(|(_, c)| *c == SELECTION_BG.to_u32())
            .map(|(r, _)| r)
            .collect();
        assert_eq!(
            bands.len(),
            1,
            "one band, for the one visible row: {bands:?}"
        );
        assert_eq!(
            bands[0],
            Rect {
                x: 0,
                y: 0,
                w: 4 * M.w,
                h: M.h
            },
            "the surviving row of the selection, now at the top of the window"
        );
    }

    #[test]
    fn a_selection_follows_its_rows_as_output_scrolls_the_grid() {
        // The painted band tracks the content: the same selection, after one row scrolls
        // into history, highlights the row where that text now is.
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"aa\r\nbb\r\ncc");
        let selection = sel(&s, (1, 0), (1, 3)); // the middle row, "bb"
        let before = fills(&list_selecting(&s, selection))
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a band");
        assert_eq!(before.0.y, M.h, "row 1 to start with");

        feed(&mut s, b"\r\ndd"); // everything shifts up one row

        let after = fills(&list_selecting(&s, selection))
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("still a band");
        assert_eq!(after.0.y, 0, "and the highlight moved up with the text");
    }

    #[test]
    fn an_unchanged_frame_diffs_to_no_damage() {
        // The heart of the damage claim: rebuild the same grid twice and the diff
        // is empty, so an idle terminal presents no draw work.
        let mut s = Screen::new(20, 5);
        feed(&mut s, b"\x1b[32mhello\x1b[0m world");
        let a = list_of(&s);
        let b = list_of(&s);
        let (w, h) = (20 * M.w, 5 * M.h);
        assert!(
            crate::render::display::damage(&a, &b, w, h).is_empty(),
            "an unchanged frame emits no damage"
        );
    }

    #[test]
    fn typing_one_cell_damages_only_its_line() {
        let mut s = Screen::new(20, 3);
        feed(&mut s, b"line one\r\nline two\r\nline three");
        let before = list_of(&s);
        // Change one cell on the middle line.
        s.move_to(1, 0);
        feed(&mut s, b"X");
        let after = list_of(&s);
        let (w, h) = (20 * M.w, 3 * M.h);
        let d = crate::render::display::damage(&before, &after, w, h);
        assert!(!d.is_empty(), "the edit is visible");
        // Every damaged rectangle is confined to the middle row's cell band, give or
        // take the bearing pad a run's bounds carries for overhanging ink.
        let pad = (M.size as i32 / 4).max(1);
        for r in d {
            assert!(
                r.y >= M.h - pad && r.y + r.h <= 2 * M.h + pad,
                "damage {r:?} stays on the edited line"
            );
        }
    }

    #[test]
    fn cell_metrics_from_a_real_face_are_a_sane_box() {
        // The one font-touching entry point: the measured cell must be a positive
        // box with the baseline inside it, or the whole grid is malformed. Uses
        // the installed monospace family (available wherever the render tests run).
        let fonts = crate::platform::freetype::Fonts::new(&[16]).expect("a monospace face");
        let m = CellMetrics::from_fonts(&fonts, 16);
        assert_eq!(m.size, 16);
        assert!(m.w >= 1 && m.h >= 1, "a non-degenerate cell: {m:?}");
        assert!(
            m.ascent > 0 && m.descent >= 0,
            "baseline inside the cell: {m:?}"
        );
        assert!(
            m.ascent + m.descent <= m.h,
            "ink height fits the line height: {m:?}"
        );
        // The line gap rides above the text, so the descent lands on the cell floor
        // and the baseline sits at or below the ascent. A cell whose baseline *is*
        // the ascent has hoisted the text into the cell's ceiling.
        assert_eq!(
            m.baseline,
            m.h - m.descent,
            "the descent seats on the cell's bottom edge: {m:?}"
        );
        assert!(m.baseline >= m.ascent, "the gap sits above the text: {m:?}");
        // The grid-fit inverse divides the surface back into whole cells.
        assert_eq!(m.columns_rows(m.w * 80, m.h * 24), (80, 24));
    }

    #[test]
    fn a_run_is_seated_on_the_baseline_and_bounded_by_its_cells() {
        // The bug this pins: seating a run on the ascent (14) instead of the baseline
        // (16) rides every glyph 2px up in its cell, so capitals crowd the cell's top
        // edge while descenders float clear of its bottom. Row 1 catches an off-by-one
        // that row 0 would hide, since its cell top is not the surface's.
        let mut s = Screen::new(4, 2);
        feed(&mut s, b"Ty\r\nTy");
        let list = list_of(&s);
        assert_eq!(
            cells_runs(&list),
            vec![
                (0, M.baseline, "Ty".to_string()),
                (0, M.h + M.baseline, "Ty".to_string()),
            ],
            "each run rides its own cell's baseline"
        );
        // A run's bounds is its cell band, padded: a box-drawing glyph fills its cell
        // to the edge, so bounds measured from the ink box would shear its top row.
        let pad_y = (M.size as i32 / 4).max(1);
        let bounds: Vec<Rect> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Cells { bounds, .. } => Some(*bounds),
                _ => None,
            })
            .collect();
        for (row, b) in bounds.iter().enumerate() {
            let cell_top = row as i32 * M.h;
            assert!(
                b.y <= cell_top && b.y + b.h >= cell_top + M.h,
                "row {row}: bounds {b:?} must cover the whole cell band, padded by {pad_y}"
            );
        }
    }

    #[test]
    fn astral_rune_is_drawn_standalone() {
        let mut s = Screen::new(6, 1);
        // U+1D400 MATHEMATICAL BOLD CAPITAL A: astral, width 1, must not batch.
        feed(&mut s, "a\u{1D400}b".as_bytes());
        let list = list_of(&s);
        let has_astral_text = list
            .iter()
            .any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "\u{1D400}"));
        assert!(
            has_astral_text,
            "the astral rune breaks out into its own Text"
        );
    }

    #[test]
    fn a_scrolled_view_paints_scrollback_not_the_live_screen() {
        // Two rows visible, older lines in history; scrolled up, the painter must
        // draw the history rows, not the live bottom.
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        s.scroll_view_up(2); // to the top: "one" over "two"
        let runs = cells_runs(&list_of(&s));
        let texts: Vec<&str> = runs.iter().map(|(_, _, t)| t.as_str()).collect();
        assert!(texts.contains(&"one"), "history row is painted: {texts:?}");
        assert!(
            !texts.contains(&"four"),
            "the live bottom is not: {texts:?}"
        );
    }

    #[test]
    fn the_cursor_is_hidden_when_scrolled_out_of_view() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour"); // cursor on the live bottom row
        s.scroll_view_up(2); // live rows scrolled below the window
        let t = Theme::default();
        let list = build_display_list(&FrameInputs {
            cursor: CursorRender::default(), // visible + focused
            ..inputs(&s, &t)
        });
        // No inverted glyph and no cursor block colour: the cursor sits off-view.
        let cursor_color = Theme::default().cursor.to_u32();
        assert!(
            !fills(&list).iter().any(|(_, c)| *c == cursor_color),
            "the cursor is not drawn while viewing history"
        );
    }

    /// A 6x2 screen with four lines fed through it, so two rows sit in history and the
    /// view can scroll: content 4 lines, viewport 2, max scroll 2.
    fn scrollable_screen() -> Screen {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        assert_eq!(
            s.scroll_extent(),
            (4, 2),
            "two lines of history behind two rows"
        );
        s
    }

    #[test]
    fn a_bar_out_of_sight_draws_nothing_even_while_scrolled() {
        let mut s = scrollable_screen();
        s.scroll_view_up(2);
        // The bar is state, not a function of the scroll position: until something lights
        // it, viewing history draws no chrome at all.
        assert_eq!(thumb_of(&list_of(&s)), None);
    }

    #[test]
    fn a_lit_bar_draws_a_thumb_in_its_lane() {
        let s = scrollable_screen();
        let list = list_with_bar(&s, &lit_bar());
        let thumb = thumb_of(&list).expect("a lit, scrollable screen draws a thumb");
        // It is drawn where a press would grab it, which is the whole contract between
        // the painter and the window's hit-test.
        let view = scroll_column(
            (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            (0, 0),
            M,
            s.dimensions().1,
        );
        let lane = scroll_lane(view, Scale::ONE);
        assert!(
            thumb.x >= lane.grab.x && thumb.x + thumb.w <= lane.grab.x + lane.grab.w,
            "thumb {thumb:?} sits inside the grab band {:?}",
            lane.grab
        );
        // Thin at rest: no pointer has grown it, so it has not reached the full slider.
        assert_eq!(thumb.w, Scale::ONE.px(SCROLL_THIN));
    }

    #[test]
    fn no_bar_when_there_is_no_history_to_scroll() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo"); // fills the screen, nothing retired into history
        assert_eq!(s.scroll_extent(), (2, 2), "content fits the viewport");
        assert_eq!(
            thumb_of(&list_with_bar(&s, &lit_bar())),
            None,
            "a lit bar still draws nothing when there is nothing to scroll"
        );
    }

    #[test]
    fn the_alt_screen_never_shows_a_bar() {
        let mut s = scrollable_screen();
        feed(&mut s, b"\x1b[?1049h"); // switch to the alt screen
        assert!(s.is_alt());
        assert_eq!(
            s.scroll_extent(),
            (2, 2),
            "the alt screen keeps no history, so it reports as unscrollable"
        );
        assert_eq!(thumb_of(&list_with_bar(&s, &lit_bar())), None);
    }

    #[test]
    fn the_thumb_travels_the_track_from_the_oldest_line_to_the_live_bottom() {
        let mut s = scrollable_screen();
        let bar = lit_bar();
        let view = scroll_column(
            (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            (0, 0),
            M,
            s.dimensions().1,
        );
        let track = scroll_lane(view, Scale::ONE).track;

        // Pinned to the live bottom (view_offset 0): the thumb's bottom meets the track's.
        let bottom = thumb_of(&list_with_bar(&s, &bar)).expect("scrollable");
        assert_eq!(bottom.y + bottom.h, track.y + track.h);

        // Scrolled all the way back: its top meets the track's top.
        s.scroll_view_to_top();
        let top = thumb_of(&list_with_bar(&s, &bar)).expect("scrollable");
        assert_eq!(top.y, track.y);
    }

    // ---- damage ------------------------------------------------------------
    //
    // The display list is only half the frame: what the compositor is actually asked to
    // repaint is `display::damage(previous, current)`. Nothing here asserted that until
    // now, and the two ways it can be wrong fail in opposite, equally silent directions.
    // **Under-damage** leaves a stale pixel on screen — the grid is right and the window
    // is lying. **Over-damage** is invisible in every correctness test there is and only
    // shows up as a machine running hot: repaint the whole surface for a blinking cursor
    // and the frame costs what a full redraw costs, forever, and no test goes red.

    /// The rectangles the compositor would be asked to repaint between two states of the
    /// same screen.
    fn damage_of(before: &Screen, after: &Screen) -> Vec<Rect> {
        let theme = Theme::default();
        let old = build_display_list(&inputs(before, &theme));
        let new = build_display_list(&inputs(after, &theme));
        let (w, h) = inputs(after, &theme).surface;
        crate::render::display::damage(&old, &new, w, h)
    }

    /// Total area of a damage set, in pixels.
    fn damaged_area(rects: &[Rect]) -> i64 {
        rects.iter().map(|r| i64::from(r.w) * i64::from(r.h)).sum()
    }

    /// Whether the damage covers every pixel of the cell at `(row, col)`. Under-damage is
    /// the failure that leaves a stale glyph on screen, and it shows up first at a cell's
    /// edges: a rectangle one pixel short of the cell still covers every interior sample,
    /// so a stepped grid would miss the very row or column that was left stale. Every
    /// pixel, boundary row and column included, is the point.
    fn covers_cell(rects: &[Rect], row: usize, col: usize) -> bool {
        let (x0, y0) = (col as i32 * M.w, row as i32 * M.h);
        (x0..x0 + M.w).all(|x| {
            (y0..y0 + M.h).all(|y| {
                rects
                    .iter()
                    .any(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
            })
        })
    }

    fn screen_after(cols: usize, rows: usize, bytes: &[u8]) -> Screen {
        let mut s = Screen::new(cols, rows);
        let mut p = crate::vt::Parser::new();
        p.advance_bytes(&mut s, bytes);
        s
    }

    /// A frame identical to the one before it asks the compositor for nothing.
    ///
    /// The cheapest and most load-bearing case: an idle terminal must not repaint. If
    /// this ever fails, bnkterm is burning a GPU on a screen that is not changing.
    #[test]
    fn an_unchanged_screen_damages_nothing() {
        for bytes in [
            &b""[..],
            b"hello",
            b"\x1b[1;31mcoloured\x1b[0m\r\nlines",
            "wide \u{4e00} and \u{1f980}".as_bytes(),
        ] {
            let a = screen_after(20, 4, bytes);
            let b = screen_after(20, 4, bytes);
            assert_eq!(
                damage_of(&a, &b),
                vec![],
                "{:?}",
                String::from_utf8_lossy(bytes)
            );
        }
    }

    /// One character typed damages that character, and does not repaint the window.
    ///
    /// Both halves matter. Covering the cell is correctness; *not* covering the whole
    /// surface is the perf regression no other test in this repo can see.
    #[test]
    fn typing_one_character_damages_about_one_cell() {
        let before = screen_after(20, 4, b"hello");
        let after = screen_after(20, 4, b"hellox");
        let rects = damage_of(&before, &after);

        assert!(!rects.is_empty(), "the new glyph must be repainted");
        assert!(
            covers_cell(&rects, 0, 5),
            "the cell that changed is covered"
        );

        // The painter emits a text run per row, so a row's worth of damage is the
        // honest bound here — a screen's worth is not.
        let surface = i64::from(20 * M.w) * i64::from(4 * M.h);
        let row = i64::from(20 * M.w) * i64::from(M.h);
        let area = damaged_area(&rects);
        assert!(
            area <= row * 2,
            "typing one character damaged {area}px of a {surface}px surface"
        );
        assert!(area < surface, "a keystroke must not repaint the window");
    }

    /// An edit far down the screen does not drag the rows above it into the damage.
    #[test]
    fn an_edit_damages_only_the_row_it_touches() {
        let before = screen_after(20, 6, b"aaa\r\nbbb\r\nccc\r\nddd");
        let after = screen_after(20, 6, b"aaa\r\nbbb\r\nccc\r\nddX");
        let rects = damage_of(&before, &after);

        assert!(covers_cell(&rects, 3, 2), "the changed cell is covered");
        // Nothing on the untouched rows: row 0 is three rows away from the edit.
        assert!(
            !rects.iter().any(|r| r.y < M.h),
            "row 0 was repainted for an edit on row 3: {rects:?}"
        );
    }

    /// Changing a colour damages the text it recoloured, not the text beside it.
    #[test]
    fn a_recolour_damages_the_run_it_recoloured() {
        let before = screen_after(20, 2, b"\x1b[31mred\x1b[0m plain");
        let after = screen_after(20, 2, b"\x1b[32mred\x1b[0m plain");
        let rects = damage_of(&before, &after);
        assert!(!rects.is_empty(), "a recolour is a change");
        assert!(covers_cell(&rects, 0, 0), "the recoloured text is covered");
        assert!(
            !rects.iter().any(|r| r.y >= M.h),
            "row 1 is untouched and must not repaint: {rects:?}"
        );
    }

    /// A scroll damages the text that moved, and only collapses to a full repaint when
    /// enough of the surface actually changed to make repainting whole the cheaper job.
    ///
    /// The pleasant surprise here, and worth pinning before someone "optimises" it away:
    /// scrolling a screen of *short* lines does not repaint the window. The painter emits
    /// a text run per row, so only the runs damage, and a screenful of one-character rows
    /// costs a column strip rather than a screen.
    #[test]
    fn a_scroll_damages_the_text_that_moved() {
        let surface = i64::from(20 * M.w) * i64::from(4 * M.h);

        // Short lines: every row's text changed, but the text is one column wide.
        let before = screen_after(20, 4, b"a\r\nb\r\nc\r\nd");
        let after = screen_after(20, 4, b"a\r\nb\r\nc\r\nd\r\ne");
        let rects = damage_of(&before, &after);
        for row in 0..4 {
            assert!(
                covers_cell(&rects, row, 0),
                "row {row} moved and is covered"
            );
        }
        assert!(
            damaged_area(&rects) * 4 < surface,
            "scrolling one-column rows repainted {}px of a {surface}px surface",
            damaged_area(&rects)
        );

        // Full lines: now most of the surface really has changed, and one rectangle
        // beats clipping four.
        let full = |tail: &str| -> Screen {
            let body: String = [
                "aaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbb",
                "cccccccccccccccccccc",
                "dddddddddddddddddddd",
            ]
            .join("\r\n");
            screen_after(20, 4, format!("{body}{tail}").as_bytes())
        };
        let rects = damage_of(&full(""), &full("\r\neeeeeeeeeeeeeeeeeeee"));
        assert_eq!(rects.len(), 1, "one rectangle, not four: {rects:?}");
        assert_eq!(damaged_area(&rects), surface, "the whole surface");
    }

    /// Every cell in a painted frame lands on its own column, for the scripts a
    /// fixed-pitch run cannot batch.
    ///
    /// A `Cells` run is drawn by segmenting its text and multiplying the cluster *index*
    /// by the cell width, so a run holding a scalar that a segmenter merges leftward
    /// silently paints every following cell one column short. That is not a subtle
    /// artefact: for Devanagari it was every line of text, with the background fill, the
    /// selection band and the cursor all still on the true grid.
    ///
    /// Checked by walking the display list and asking where each cluster actually lands,
    /// rather than by asserting the internal predicate, so the property survives a change
    /// in how the exclusion is implemented.
    #[test]
    fn every_cell_paints_on_its_own_column_in_every_script() {
        let theme = Theme::default();
        // One line per script, each ending in an ASCII letter: the letter is what visibly
        // walks left when a cluster merge eats a column.
        for (script, text) in [
            ("Devanagari (SpacingMark)", "कीx"),
            ("Devanagari (conjunct)", "क्षx"),
            ("Bengali", "কীx"),
            ("Gurmukhi", "ਕੀx"),
            ("Gujarati", "કીx"),
            ("Oriya", "କୀx"),
            ("Tamil", "கீx"),
            ("Telugu", "కీx"),
            ("Kannada", "ಕೀx"),
            ("Malayalam", "കീx"),
        ] {
            let screen = screen_after(12, 2, text.as_bytes());
            let list = build_display_list(&inputs(&screen, &theme));

            // Where the painter puts each cluster: one entry per column it draws at.
            let mut pens: Vec<i32> = Vec::new();
            for cmd in list.iter() {
                match cmd {
                    DrawCmd::Cells {
                        x, cell_w, text, ..
                    } => {
                        for (i, _) in grapheme::graphemes(text.as_str()).enumerate() {
                            pens.push(x + i as i32 * cell_w);
                        }
                    }
                    DrawCmd::Text { x, .. } => pens.push(*x),
                    _ => {}
                }
            }
            pens.sort_unstable();
            pens.dedup();

            // The grid's own answer, from the cells it filled.
            let cell_w = M.w;
            let expected: Vec<i32> = (0..12)
                .filter(|&c| screen.cell(0, c).rune != ' ')
                .map(|c| pens[0] + (c as i32 - first_inked_col(&screen)) * cell_w)
                .collect();
            assert_eq!(
                pens,
                expected,
                "{script}: {} clusters painted at {pens:?}, grid wants {expected:?}",
                pens.len()
            );
        }
    }

    /// The column of the first cell holding anything, for anchoring a placement check.
    fn first_inked_col(screen: &Screen) -> i32 {
        (0..12)
            .find(|&c| screen.cell(0, c).rune != ' ')
            .unwrap_or(0) as i32
    }

    /// The damage never lies by omission: whatever changed on the grid is covered.
    ///
    /// A property rather than a case list — it compares the damage against the cells
    /// that actually differ, over a structured stream, so it holds for operations nobody
    /// thought to write a case for. Under-damage is the failure that leaves the window
    /// showing something the terminal no longer believes.
    /// A cell reduced to what the painter can actually show.
    ///
    /// The painter's own rule (`push_run`, `push_glyph`): a `HIDDEN` cell is drawn as a
    /// space and is never inked, so its rune cannot be observed and neither can its
    /// foreground — unless `REVERSE` is also set, which swaps the foreground into the
    /// *background* and makes it visible again. `SGR 8` is what a program masking a
    /// password field sends, and two such cells differing only in the character they
    /// conceal are, correctly, the same picture.
    ///
    /// Nothing else is normalised. The point of the property is to catch damage that
    /// omits a real change, so anything that might be visible stays in the comparison.
    fn painted(cell: Cell) -> Cell {
        if !cell.attrs.contains(Attrs::HIDDEN) {
            return cell;
        }
        let reverse = cell.attrs.contains(Attrs::REVERSE);
        // With no glyph to shape, bold and italic have nothing to act on, and `HIDDEN`
        // itself says only "draw no ink". `DIM` acts on the foreground, which is drawn
        // only when `REVERSE` has moved it into the background.
        let mut attrs = cell.attrs;
        attrs.remove(Attrs::HIDDEN | Attrs::BOLD | Attrs::ITALIC);
        if !reverse {
            attrs.remove(Attrs::DIM);
        }
        Cell {
            rune: ' ',
            fg: if reverse { cell.fg } else { Color::Default },
            attrs,
            ..cell
        }
    }

    #[test]
    fn damage_covers_every_cell_that_changed() {
        let theme = Theme::default();
        let mut stream = crate::fuzz::Stream::new(0x0DA3_40E0_1234_5678);
        let (cols, rows) = (12, 5);

        let mut screen = Screen::new(cols, rows);
        let mut parser = crate::vt::Parser::new();
        for _ in 0..400 {
            let before: Vec<Cell> = (0..rows)
                .flat_map(|r| (0..cols).map(move |c| (r, c)))
                .map(|(r, c)| screen.cell(r, c))
                .collect();
            let old = build_display_list(&inputs(&screen, &theme));

            parser.advance_bytes(&mut screen, &stream.bytes(24));

            let new = build_display_list(&inputs(&screen, &theme));
            let (w, h) = inputs(&screen, &theme).surface;
            let rects = crate::render::display::damage(&old, &new, w, h);

            for r in 0..rows {
                for c in 0..cols {
                    let changed = painted(screen.cell(r, c)) != painted(before[r * cols + c]);
                    if changed {
                        assert!(
                            covers_cell(&rects, r, c),
                            "cell ({r},{c}) changed and was not repainted: {rects:?}"
                        );
                    }
                }
            }
        }
    }
}
