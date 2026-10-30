//! The terminal's painter: it walks the visible grid and produces a
//! backend-agnostic [`DisplayList`], the value the damage differ and the GPU
//! batcher both consume. This is the seam between the terminal core and the
//! renderer: the core never touches Vulkan, it emits drawing
//! primitives, and everything downstream (`display::damage_into`,
//! `gpu::build_frame_into`) is a pure function of that list.
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
//! foreground as fixed-pitch [`DrawCmd::Cells`] runs (with wide glyphs broken out
//! into their own [`DrawCmd::Text`], and a rune that could join its neighbour's
//! cluster drawn as a run of one, so the pitch stays uniform), then underline/strike
//! rules and the hovered link's rule, and finally the cursor on top. Over the whole
//! grid come the overlay scrollbar and, while Ctrl is held over a link, the plate
//! that previews its target. Everything past the base fill is emitted only where it
//! is not the default, so an idle screen of mostly-blank cells produces a short list
//! and an unchanged frame diffs to nothing.
//!
//! The two overlays the app hands in — the selection band and the hovered hyperlink's
//! underline — both cover a [`CellSpan`], and both resolve to *one inclusive column
//! range per row* (`span_cols`) before anything per-cell happens. That is a
//! performance rule, not a stylistic one: `Painter::cell` runs about four times per
//! cell, so an overlay tested there is tested a quarter of a million times a frame to
//! serve a span that is usually empty. A 13% frame regression established it.
//!
//! # Where it lives
//!
//! This file holds the types a frame is built from, [`build_display_list_into`], the
//! per-cell vocabulary every pass shares (cell geometry, colour resolution, what a cell
//! shows), and the text helpers the chrome draws with. The passes are split by what they
//! paint:
//!
//! ```text
//!   rows.rs       backgrounds, fixed-pitch runs, underlines and strikes, the selection
//!                 band, and the hovered link's rule
//!   cursor.rs     the cursor in each shape, and the password padlock
//!   scrollbar.rs  the overlay scrollbar and the lane it is grabbed in
//!   preview.rs    the plate that previews a hovered link's target
//! ```

mod cursor;
mod preview;
mod rows;
mod scrollbar;

use crate::color::{Ground, Rgb, Theme};
use crate::grid::{AbsRow, Attrs, Cell, RowEpoch, Screen};
use crate::panel::Panel;
use crate::platform::freetype::{FaceKey, FontStyle, Fonts, Metrics};
use crate::platform::geom::{Rect, Scale};
use crate::platform::grapheme;
use crate::platform::scroll::Scrollbar;
use crate::platform::verdict::Verdict;
use crate::render::display::{DisplayList, DrawCmd, Fade};
use crate::width;

/// The glyph whose advance defines the monospace cell width. `M` is the classic
/// full-width reference; on a genuine monospace face every glyph shares it.
const REFERENCE_GLYPH: char = 'M';

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
    /// the cell. This is *not* the ascent (see [`Metrics`]) — the font's line gap
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
    /// Measure the cell box from the opened fonts at `size`: the line height for the row
    /// pitch and the regular face's reference-glyph advance for the column pitch.
    pub fn from_fonts(fonts: &Fonts, size: u32) -> Self {
        let key = FaceKey::Prose {
            size,
            style: FontStyle::Regular,
        };
        let advance = fonts.face_for(key).advance(REFERENCE_GLYPH);
        let lock_glyph = fonts.covers(key, LOCK_GLYPH);
        Self::measure(size, fonts.metrics(size), advance, lock_glyph)
    }

    /// The vertical box for the proportional interface face at `size`: its own
    /// ascent/descent/line-height, so chrome text (the tab bar) baselines on the
    /// sans ink box. `w` carries a nominal glyph advance for callers that want one,
    /// but proportional UI text is placed by each glyph's real advance, not this
    /// pitch, so it is not a cell width.
    pub fn from_ui(fonts: &Fonts, size: u32) -> Self {
        let key = FaceKey::Ui {
            size,
            style: FontStyle::Regular,
        };
        let advance = fonts.face_for(key).advance('n');
        // The chrome never draws a cursor, and lock coverage comes from the fallback
        // chain, not this face, so it resolves the same either way. Measuring it rather
        // than hardcoding `false` keeps the two constructors from disagreeing.
        let lock_glyph = fonts.covers(key, LOCK_GLYPH);
        Self::measure(size, fonts.ui_metrics(size), advance, lock_glyph)
    }

    /// The cell box the two constructors share: the font's own vertical metrics, the
    /// given advance rounded to a whole pixel for the column pitch (a fractional pitch is
    /// what drifts a grid), and both pitches floored at one pixel so a degenerate face
    /// can never produce a zero-area cell.
    fn measure(size: u32, m: Metrics, advance: f32, lock_glyph: bool) -> Self {
        CellMetrics {
            size,
            w: (advance.round() as i32).max(1),
            h: m.line_height.max(1),
            baseline: m.baseline,
            ascent: m.ascent,
            descent: m.descent,
            lock_glyph,
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
/// [`crate::grid::Screen::selection_text_into`]), so the paint is deliberately wider than
/// what lands on the clipboard.
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

/// The hyperlink under the pointer: the cells its text covers, and whether it opens.
/// A link that opens is ruled solid, a refused one dashed and dimmed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LinkHover {
    pub span: CellSpan,
    pub verdict: Verdict,
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
    pub hover: Option<LinkHover>,
    /// The hovered link's preview line ([`crate::link_preview`]), shown while Ctrl is
    /// held: drawn on a plate beside [`Self::hover`], over the grid and its scrollbar.
    /// `None` draws nothing.
    pub preview: Option<&'a str>,
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
    /// The shortest the thumb may draw, so a deep scrollback still leaves something to
    /// grab.
    min_thumb: i32,
}

/// Build the frame's display list into `out` (cleared first), drawing every run's
/// text from `strings` (a pool of buffers retired by a previous frame), so a steady
/// stream of frames rebuilds the list without allocating. Both are owned by a
/// [`DisplayListPool`](crate::render::display::DisplayListPool) across frames; `out`
/// keeps its capacity and the strings are recycled, so only a first frame (or a grown
/// grid) touches the allocator.
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
    if let Some(text) = inputs.preview {
        painter.push_link_preview(text, cols, rows);
    }
}

/// Build a fresh display list, allocating its vector and run strings. The one-shot
/// path for tests; the windowed render loop uses [`build_display_list_into`] with a
/// [`DisplayListPool`](crate::render::display::DisplayListPool) to stay allocation-free
/// in steady state.
#[cfg(test)]
pub fn build_display_list(inputs: &FrameInputs) -> DisplayList {
    let mut out = DisplayList::new();
    let mut strings = Vec::new();
    build_display_list_into(&mut out, &mut strings, inputs);
    out
}

/// What one cell actually shows: the rune to draw, whether its combining marks go with
/// it, and whether any of that is ink.
///
/// The painter draws a cell's glyph from three places (the batched run, the wide
/// glyph, and the cursor's inverted stamp), and every one of them has to conceal an
/// `SGR 8` cell identically. Three copies of that rule is exactly how the third came to
/// be missing it: a TUI masking a password with `SGR 8` rather than tty echo-off showed
/// the most recently typed character under the cursor, one keystroke at a time.
///
/// So it is stated once and asked for twice, by [`Painter::push_wide_glyph`] and
/// [`Painter::stamp_inverted_glyph`], the two that build a one-cell string. The batched
/// run keeps the rule spelled out flat instead, and that exception is measured rather
/// than assumed: constructing this per cell in the run scan cost `frame_tabbar` +3-6%,
/// unmoved by `#[inline(always)]`. The comment at that loop says so, and
/// `a_concealed_cell_is_still_concealed_under_the_cursor` compares whole display lists,
/// so both copies are pinned by one test.
///
/// Built by [`Painter::shown`], which is where the marks lookup happens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shown {
    /// The rune to draw. A concealed cell shows a space, whatever it holds.
    rune: char,
    /// Whether the cell's combining marks are drawn with it. A concealed cell's are not:
    /// an accent surviving the conceal would leak which character is under it.
    marks: bool,
    /// Whether any ink lands at all. A blank cell with no marks draws nothing, and
    /// neither does a concealed one — but the cell still takes its decorations, since an
    /// underline runs beneath cells whether or not they carry ink.
    inked: bool,
}

/// The per-frame builder: the shared inputs plus the list being appended to. One
/// method per concern keeps [`build_display_list_into`] readable.
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
    /// The hovered hyperlink, underlined by [`Self::push_hover_rule`].
    hover: Option<LinkHover>,
    /// The display scale, so the scrollbar's logical constants land at the right size.
    scale: Scale,
    /// The overlay scrollbar's fade/expand state, drawn by [`Self::push_scrollbar`].
    scrollbar: &'a Scrollbar,
    /// The list being appended to, owned by the caller's
    /// [`DisplayListPool`](crate::render::display::DisplayListPool) and reused across
    /// frames.
    list: &'a mut DisplayList,
    /// Recycled string buffers to draw run/glyph text from, so a steady frame's
    /// text costs no allocation.
    strings: &'a mut Vec<String>,
}

impl Painter<'_> {
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

    /// What a cell shows once `SGR 8` has had its say. See [`Shown`] for why this is one
    /// function rather than a rule each drawing path restates.
    ///
    /// The concealed arm returns without consulting the marks side table, which is what
    /// keeps this the same single lookup per cell that the run scan already paid for.
    fn shown(&self, row: usize, col: usize, cell: Cell) -> Shown {
        if cell.attrs.contains(Attrs::HIDDEN) {
            return Shown {
                rune: ' ',
                marks: false,
                inked: false,
            };
        }
        let marks = !self.marks_empty(row, col);
        Shown {
            rune: cell.rune,
            marks,
            inked: cell.rune != ' ' || marks,
        }
    }

    /// Push one glyph as a [`DrawCmd::Text`] spanning `width_cells` columns from `x`.
    /// The shape every standalone glyph takes: a wide character, the glyph stamped
    /// inside a block cursor, and the lock all differ only in their colours and in what
    /// they put in `text` (which the caller filled from [`take_string`]).
    fn push_glyph_text(
        &mut self,
        x: i32,
        baseline: i32,
        width_cells: i32,
        style: FontStyle,
        (color, bg): (Rgb, Rgb),
        text: String,
    ) {
        let m = self.metrics;
        self.list.push(DrawCmd::Text {
            bounds: text_bounds(x, baseline, width_cells * m.w, m),
            x,
            baseline,
            face: FaceKey::Prose {
                size: m.size,
                style,
            },
            color: color.to_u32(),
            bg: bg.to_u32(),
            fade: None,
            text,
        });
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
    /// Resolved on its own, not as half of a pair: both grounds go through the theme,
    /// and the painter's two per-cell scans each want exactly one of them (the
    /// background scan the background, the run scan the foreground), so resolving both
    /// and dropping half doubled the theme lookups on the hottest loops it has.
    fn cell_fg(&self, cell: Cell) -> Rgb {
        // Reverse means the foreground is drawn from the cell's *background* colour;
        // dim then darkens whichever one it landed on.
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

    /// The resolved background a cell asks for, before selection has its say.
    fn cell_bg_of(&self, cell: Cell) -> Rgb {
        if cell.attrs.contains(Attrs::REVERSE) {
            cell.fg.resolve(self.theme, Ground::Foreground)
        } else {
            cell.bg.resolve(self.theme, Ground::Background)
        }
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

/// How many cells of a link's preview fit across a grid `cols` wide, inside the plate's
/// padding: what [`crate::link_preview::write`] fits its line to.
pub(crate) fn preview_room(cols: usize, m: CellMetrics) -> usize {
    if m.w <= 0 {
        return 0;
    }
    usize::try_from(Panel::snug(m).cols_within(cols as i32 * m.w)).unwrap_or(0)
}

/// Append styled cell text at a fixed-pitch origin, recycling every command's
/// owned string from `strings`. Single-column clusters coalesce into `Cells`, and
/// one [`cells_safe`] excludes is a `Cells` of its own; wide clusters break out as
/// `Text` at their exact cell position so they can never desynchronise the
/// fixed-pitch run that follows.
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
        let width = usize::from(width::cluster_width(cluster)).max(1);
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
            if width == 1 {
                let one = Pending {
                    start_cell: pen_cells,
                    cells: 1,
                    text: glyph,
                };
                flush(out, one);
            } else {
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
        }
        pen_cells += width;
    }
    if let Some(pending) = run.take() {
        flush(out, pending);
    }
}

/// A cleared string buffer from the recycle pool, or a fresh empty one when the pool is
/// dry. Every run and glyph text is drawn from here so the frame reuses the buffers the
/// previous frame retired.
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
/// An excluded rune is drawn as a run of one cell.
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

    use crate::color::Color;

    use crate::term_render::*;
    use crate::testutil::{feed, screen_after};

    /// Synthetic metrics: a 10x20 cell so column/row math reads off by eye, with a
    /// 2px line gap (ascent 14 + descent 4 < h 20), like every real font ships. The
    /// gap sits above the ascent, so the baseline is 16, *not* the ascent: any test
    /// that seats text by `M.ascent` is asserting the bug this metric exists to catch.
    pub(super) const M: CellMetrics = CellMetrics {
        size: 16,
        w: 10,
        h: 20,
        baseline: 16,
        ascent: 14,
        descent: 4,
        lock_glyph: true,
    };

    /// A bar at rest, for the frames that are not about the scrollbar: it is out of
    /// sight, so it adds nothing to the list they assert over.
    pub(super) static HIDDEN_BAR: Scrollbar = Scrollbar::hidden();

    /// The frame inputs for `s`: default theme, a surface exactly the grid's size, no
    /// origin inset, no selection, no hovered link, the cursor hidden and the scrollbar
    /// out of sight (so a test about content is not perturbed by either). A case varies
    /// one field with struct update syntax: `FrameInputs { selection, ..inputs(&s, &t) }`.
    pub(super) fn inputs<'a>(s: &'a Screen, theme: &'a Theme) -> FrameInputs<'a> {
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
            preview: None,
            scale: Scale::ONE,
            scrollbar: &HIDDEN_BAR,
        }
    }

    /// Build a list for a screen with the default theme, no selection, cursor
    /// hidden (so the tests that are about content are not perturbed by it).
    pub(super) fn list_of(s: &Screen) -> DisplayList {
        build_display_list(&inputs(s, &Theme::default()))
    }

    /// [`list_of`] with a hovered hyperlink that opens over it.
    pub(super) fn list_hovering(s: &Screen, span: CellSpan) -> DisplayList {
        list_hovering_verdict(s, span, Verdict::Open)
    }

    /// [`list_of`] with a hovered hyperlink over it, judged `verdict`.
    pub(super) fn list_hovering_verdict(
        s: &Screen,
        span: CellSpan,
        verdict: Verdict,
    ) -> DisplayList {
        build_display_list(&FrameInputs {
            hover: Some(LinkHover { span, verdict }),
            ..inputs(s, &Theme::default())
        })
    }

    pub(super) fn cells_runs(list: &[DrawCmd]) -> Vec<(i32, i32, String)> {
        list.iter()
            .filter_map(|c| match c {
                DrawCmd::Cells {
                    x, baseline, text, ..
                } => Some((*x, *baseline, text.clone())),
                _ => None,
            })
            .collect()
    }

    pub(super) fn fills(list: &[DrawCmd]) -> Vec<(Rect, u32)> {
        list.iter()
            .filter_map(|c| match c {
                DrawCmd::Fill { rect, color } => Some((*rect, *color)),
                _ => None,
            })
            .collect()
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

    /// Panel text follows the grid's rule: a single-width cluster a run cannot batch is
    /// a run of one, and only a wide one is `Text`.
    #[test]
    fn cell_text_draws_a_single_width_cluster_in_its_cell() {
        let mut out = DisplayList::new();
        let face = FaceKey::Prose {
            size: M.size,
            style: FontStyle::Regular,
        };
        let text = "a\u{1F321}b\u{6F22}";
        push_cell_text(
            &mut out,
            &mut Vec::new(),
            text,
            0,
            M.baseline,
            M,
            face,
            0,
            0,
        );
        let drawn: Vec<(bool, i32, &str)> = out
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Cells { x, text, .. } => Some((true, *x, text.as_str())),
                DrawCmd::Text { x, text, .. } => Some((false, *x, text.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            drawn,
            [
                (true, 0, "a"),
                (true, M.w, "\u{1F321}"),
                (true, 2 * M.w, "b"),
                (false, 3 * M.w, "\u{6F22}"),
            ],
            "(is Cells, x, text)"
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

    /// A cell reduced to what the painter can actually show.
    ///
    /// A concealed cell is drawn as a space and is never inked, so neither its rune nor
    /// its foreground can be observed — unless `REVERSE` is also set, which swaps the
    /// foreground into the *background* and makes it visible again. `SGR 8` is what a
    /// program masking a password field sends, and two such cells differing only in the
    /// character they conceal are, correctly, the same picture.
    ///
    /// Deliberately an independent restatement of [`Shown`] rather than a call to it: a
    /// property that derives its expectation from the code under test proves only that the
    /// code agrees with itself. It has to be kept in step by hand, and the day the two
    /// disagree is a question worth being asked.
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

    /// The damage never lies by omission: whatever changed on the grid is covered.
    ///
    /// A property rather than a case list — it compares the damage against the cells that
    /// actually differ, over a structured stream, so it holds for operations nobody
    /// thought to write a case for. Under-damage is the failure that leaves the window
    /// showing something the terminal no longer believes.
    #[test]
    fn damage_covers_every_cell_that_changed() {
        let theme = Theme::default();
        let mut stream = crate::dev::fuzz::Stream::new(0x0DA3_40E0_1234_5678);
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
