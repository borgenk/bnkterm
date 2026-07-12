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
//! "drifts from the glyphs"). We get that for free by
//! *routing every coordinate through `cell_x`/`cell_y`* and never accumulating a font's
//! (fractional) advance: text goes out as [`DrawCmd::Cells`], whose contract is
//! exactly "cluster `i` draws at `x + i*cell_w`". The one measurement that keys
//! the whole grid, `cell_w`, is the advance of a reference glyph from the regular
//! face rounded to a whole pixel; the grid decides *which* column a rune lands in
//! (via `width::width`), so measurement and placement can never disagree.
//!
//! # What a frame is made of
//!
//! Per painted row, in stacking order (the list order the damage diff and the GPU
//! rely on): the base background fill (once, whole surface), then per-cell
//! background runs where a cell's background differs from the theme's, then the
//! foreground as fixed-pitch [`DrawCmd::Cells`] runs (with wide glyphs and
//! astral-plane runes broken out into their own [`DrawCmd::Text`] so the pitch
//! stays uniform), then underline/strike rules, and finally the cursor on top.
//! Everything past the base fill is emitted only where it is not the default, so
//! an idle screen of mostly-blank cells produces a short list and an unchanged
//! frame diffs to nothing.

use crate::color::{Ground, Rgb, Theme};
use crate::grid::{Attrs, Cell, Screen};
use crate::platform::freetype::{FaceKey, FontStyle, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd};

/// The glyph whose advance defines the monospace cell width. `M` is the classic
/// full-width reference; on a genuine monospace face every glyph shares it.
const REFERENCE_GLYPH: char = 'M';

/// The selection highlight background: a muted steel blue that stays legible
/// under the default fg.
const SELECTION_BG: Rgb = Rgb::new(0x41, 0x57, 0x76);

/// The width of a bar (`DECSCUSR 5/6`) cursor, in pixels.
const BAR_CURSOR_WIDTH: i32 = 2;

/// The scroll-position indicator drawn on the right edge while viewing history:
/// its width in pixels and its colour (a soft steel grey, dim so it never fights
/// the text).
const SCROLL_INDICATOR_WIDTH: i32 = 4;
const SCROLL_INDICATOR_COLOR: u32 = 0x0055_6070;

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
    /// Pixels from the top of a cell down to the baseline.
    pub ascent: i32,
    /// Pixels from the baseline to the bottom of a cell.
    pub descent: i32,
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
            ascent: m.ascent,
            descent: m.descent,
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
            ascent: m.ascent,
            descent: m.descent,
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

/// A cursor's drawn shape. The style escape (`DECSCUSR`) is not parsed yet
/// (phase 3+), so the app chooses this; the grid only says whether the cursor is
/// visible at all (`DECTCEM`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CursorShape {
    /// A filled cell that inverts the glyph under it (the default).
    Block,
    /// A thin vertical bar at the cell's left edge.
    Bar,
    /// A thin rule along the cell's baseline.
    Underline,
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

/// A linear text selection over the visible grid, in `(row, col)` cell
/// coordinates. `anchor` is where the drag began and `head` where it is now,
/// either order; a cell falls in the selection when it lies between them in reading
/// order (whole rows in the middle, partial rows at the ends). The painter paints
/// the whole geometric span, blank cells included (see `Painter::selection_cols`),
/// so dragging over the empty area below the prompt highlights it; the copy still
/// trims each line's trailing blanks (see [`crate::grid::Screen::selection_text`]),
/// so the paint is deliberately wider than what lands on the clipboard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Selection {
    pub anchor: (usize, usize),
    pub head: (usize, usize),
}

impl Selection {
    /// `(start, end)` in reading order, so `start <= end` row-major.
    fn ordered(self) -> ((usize, usize), (usize, usize)) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }
}

/// The inputs a frame build reads: the grid and its theme, the fixed cell metrics,
/// the `surface`/`origin` geometry (`origin` is the grid's top-left pixel, the window
/// padding; the background still fills the whole `surface`, so the inset shows as a
/// margin), and the transient cursor and selection. Grouped so the pooled and
/// one-shot builders share one parameter and the window/bench assemble it in one
/// place. Pure data (no fonts, no GPU), so a test asserts the exact primitives a
/// grid state produces.
pub struct FrameInputs<'a> {
    pub screen: &'a Screen,
    pub theme: &'a Theme,
    pub metrics: CellMetrics,
    pub surface: (i32, i32),
    pub origin: (i32, i32),
    pub cursor: CursorRender,
    pub selection: Option<Selection>,
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
        metrics: inputs.metrics,
        origin: inputs.origin,
        selection: inputs.selection,
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
    painter.scroll_indicator(inputs.surface, rows);
}

/// Build a fresh display list, allocating its vector and run strings. The one-shot
/// path for tests and callers that do not keep frame-to-frame state; the windowed
/// render loop uses [`build_display_list_into`] with a [`DisplayListPool`] to stay
/// allocation-free in steady state.
pub fn build_display_list(
    screen: &Screen,
    theme: &Theme,
    metrics: CellMetrics,
    surface: (i32, i32),
    origin: (i32, i32),
    cursor: CursorRender,
    selection: Option<Selection>,
) -> DisplayList {
    let mut out = DisplayList::new();
    let mut strings = Vec::new();
    build_display_list_into(
        &mut out,
        &mut strings,
        &FrameInputs {
            screen,
            theme,
            metrics,
            surface,
            origin,
            cursor,
            selection,
        },
    );
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

/// The per-frame builder: the shared inputs plus the list being appended to. One
/// method per concern keeps [`build_display_list`] readable.
struct Painter<'a> {
    screen: &'a Screen,
    theme: &'a Theme,
    metrics: CellMetrics,
    /// The grid's top-left pixel in the surface: the window padding inset. Every
    /// cell coordinate is measured from here (via [`Self::cell_x`]/[`Self::cell_y`]),
    /// so the whole grid rides the same rigid offset and can never drift from it.
    origin: (i32, i32),
    selection: Option<Selection>,
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

    /// The baseline y for `row`: the row's top plus the face ascent.
    fn baseline(&self, row: usize) -> i32 {
        self.cell_y(row) + self.metrics.ascent
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
            color: self.theme.bg.to_u32(),
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
            self.push_run(row, col, cols, cell, baseline);
            col = self.run_end(row, col, cols, cell);
        }
    }

    /// Emit a fixed-pitch run starting at `col`: the maximal span of single-width,
    /// same-style cells. The [`DrawCmd::Cells`] text is trimmed to the last inked
    /// cell (trailing blanks in the run draw nothing, so carrying them would only
    /// cost allocation and coarsen the damage diff), while underline/strike rules
    /// span the full styled run, because a styled trailing space still shows its
    /// rule (xterm draws it).
    fn push_run(&mut self, row: usize, col: usize, cols: usize, first: Cell, baseline: i32) {
        let m = self.metrics;
        // The run breaks on foreground, not background, so the first cell's background
        // stands in for the run when weighting the glyph anti-aliasing; a same-fg run
        // over a mixed background is rare (reverse video and selection tint uniformly).
        let (fg, bg) = self.resolve(first, false);
        let style = style_of(first);
        let end = self.run_end(row, col, cols, first);
        // A recycled buffer (from the pool) usually already has the capacity a run
        // needs; reserving covers a fresh one and any run longer than last frame's,
        // so an all-ASCII run (the common case) fills without reallocating.
        let mut text = self.take_string_with_capacity(end - col);
        text.reserve(end - col);
        // Bytes and cell count up to and including the last cell with ink, so a
        // trailing blank never lands in the emitted text.
        let mut inked_bytes = 0;
        let mut inked_cells = 0;
        for c in col..end {
            let cell = self.cell(row, c);
            let hidden = cell.attrs.contains(Attrs::HIDDEN);
            text.push(if hidden { ' ' } else { cell.rune });
            let marks = (!hidden).then(|| self.marks(row, c)).flatten();
            if let Some(marks) = marks {
                text.extend(marks);
            }
            if !hidden && (cell.rune != ' ' || marks.is_some_and(|m| !m.is_empty())) {
                inked_bytes = text.len();
                inked_cells = c - col + 1;
            }
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
                    style,
                },
                fg.to_u32(),
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
        self.push_decorations(first, x, (end - col) as i32 * m.w, baseline, fg);
    }

    /// The underline and strike rules for a run, drawn as solid fills spanning the
    /// run's width in the run's foreground colour.
    fn push_decorations(&mut self, cell: Cell, x: i32, run_w: i32, baseline: i32, fg: Rgb) {
        let m = self.metrics;
        let thickness = (m.size as i32 / 12).max(1);
        if cell.attrs.contains(Attrs::UNDERLINE) {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y: baseline + (m.descent / 2).max(1),
                    w: run_w,
                    h: thickness,
                },
                color: fg.to_u32(),
            });
        }
        if cell.attrs.contains(Attrs::STRIKE) {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y: baseline - m.ascent / 3,
                    w: run_w,
                    h: thickness,
                },
                color: fg.to_u32(),
            });
        }
    }

    /// One cell's glyph as a standalone [`DrawCmd::Text`] at its exact column,
    /// used for wide glyphs and astral runes (base rune plus any combining marks).
    /// A hidden or blank cell draws nothing.
    fn push_glyph(&mut self, row: usize, col: usize, cell: Cell, baseline: i32) {
        if cell.attrs.contains(Attrs::HIDDEN) || (cell.rune == ' ' && self.marks_empty(row, col)) {
            return;
        }
        let m = self.metrics;
        let x = self.cell_x(col);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let (fg, bg) = self.resolve(cell, false);
        let mut text = self.take_string();
        text.push(cell.rune);
        if let Some(marks) = self.marks(row, col) {
            text.extend(marks);
        }
        self.list.push(DrawCmd::Text {
            bounds: text_bounds(x, baseline, width_cells * m.w, m),
            x,
            baseline,
            face: FaceKey::Prose {
                size: m.size,
                style: style_of(cell),
            },
            color: fg.to_u32(),
            bg: bg.to_u32(),
            text,
        });
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
        match cursor.shape {
            CursorShape::Block if cursor.focused => {
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y,
                        w: width_cells * m.w,
                        h: m.h,
                    },
                    color,
                });
                self.stamp_inverted_glyph(dr, cc, cell, width_cells);
            }
            CursorShape::Block => self.hollow_block(x, y, width_cells * m.w),
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

    /// A right-edge indicator of where the view sits in the scrollback, drawn only
    /// while scrolled up. Its height and position are proportional to the window's
    /// slice of the whole history-plus-screen, like a scrollbar thumb, so a glance
    /// says how far back you are.
    fn scroll_indicator(&mut self, surface: (i32, i32), rows: usize) {
        if !self.screen.is_scrolled() {
            return;
        }
        let (sw, sh) = surface;
        let total = self.screen.scrollback_len() + rows; // all lines, history + screen
        if total == 0 || sh <= 0 || sw <= 0 {
            return;
        }
        // The first visible line's index into that whole, and the window height.
        let top = self.screen.scrollback_len() - self.screen.view_offset();
        let y = (top as i64 * sh as i64 / total as i64) as i32;
        // A minimum thumb height so it stays grabbable/visible on a deep history.
        let min_thumb = (sh / 20).max(8);
        let h = ((rows as i64 * sh as i64 / total as i64) as i32).max(min_thumb);
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: sw - SCROLL_INDICATOR_WIDTH,
                y: y.min(sh - h).max(0),
                w: SCROLL_INDICATOR_WIDTH,
                h,
            },
            color: SCROLL_INDICATOR_COLOR,
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
        if let Some(marks) = self.marks(row, col) {
            text.extend(marks);
        }
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
                style: style_of(cell),
            },
            color: ink.to_u32(),
            bg: self.theme.cursor.to_u32(),
            text,
        });
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

    /// The exclusive end column of the run beginning at `col`: it stops at the
    /// first cell whose style differs, or that is wide or astral (those are drawn
    /// standalone), or the end of the row.
    fn run_end(&self, row: usize, col: usize, cols: usize, first: Cell) -> usize {
        let fg = self.cell_fg(first);
        let style = style_of(first);
        let ul = first.attrs.contains(Attrs::UNDERLINE);
        let st = first.attrs.contains(Attrs::STRIKE);
        let mut end = col + 1;
        while end < cols {
            let c = self.cell(row, end);
            if c.is_wide_leader() || c.is_wide_spacer() || !cells_safe(c.rune) {
                break;
            }
            if self.cell_fg(c) != fg
                || style_of(c) != style
                || c.attrs.contains(Attrs::UNDERLINE) != ul
                || c.attrs.contains(Attrs::STRIKE) != st
            {
                break;
            }
            end += 1;
        }
        end
    }

    /// The cell shown at display `(row, col)`, honouring the scroll offset (the
    /// live cell when pinned to the bottom). Every content path goes through this
    /// so scrollback and the live screen paint identically.
    fn cell(&self, row: usize, col: usize) -> Cell {
        self.screen.view_cell(row, col)
    }

    /// Combining marks at display `(row, col)`, honouring the scroll offset.
    fn marks(&self, row: usize, col: usize) -> Option<&[char]> {
        self.screen.view_marks(row, col)
    }

    /// The resolved foreground colour of a cell (reverse and dim applied).
    fn cell_fg(&self, cell: Cell) -> Rgb {
        self.resolve(cell, false).0
    }

    /// The resolved background colour of cell `(row, col)`; `selected` (precomputed
    /// by the caller from the content-trimmed row span) swaps in the selection tint.
    fn cell_bg(&self, row: usize, col: usize, selected: bool) -> Rgb {
        self.resolve(self.cell(row, col), selected).1
    }

    /// The inclusive column span of `row` the selection highlights: the whole
    /// geometric span the drag covers, blank cells included. A linear selection
    /// paints the first row from its start column to the row's right edge, every
    /// whole middle row edge to edge, and the last row from the left edge to its end
    /// column, so dragging across the empty area below the prompt highlights it, the
    /// way xterm/wezterm/ghostty do. `None` only when `row` lies outside the
    /// selection. The paint is deliberately wider than [`Screen::selection_text`]
    /// copies: it shows the drag for feedback while the copy trims trailing blanks.
    fn selection_cols(&self, row: usize) -> Option<(usize, usize)> {
        let (start, end) = self.selection?.ordered();
        if row < start.0 || row > end.0 {
            return None;
        }
        let last_col = self.screen.dimensions().0.saturating_sub(1);
        let first = if row == start.0 { start.1 } else { 0 };
        let last = if row == end.0 { end.1 } else { last_col }.min(last_col);
        Some((first, last))
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
        self.marks(row, col).is_none_or(|m| m.is_empty())
    }
}

/// Append one proportional text run (chrome/UI text) at `x` on `baseline` in
/// `face`, drawing the owned string from the pool so a steady repaint allocates
/// nothing. Unlike [`push_cell_text`] the glyphs advance by the font's own metrics
/// rather than a fixed cell pitch, so this is for a UI-sans label, never grid text;
/// `run_w` is the run's already-measured pixel width, used only for the damage
/// bounds. An empty run pushes nothing.
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
    let mut pen_cells = 0usize;
    let mut run_start = 0usize;
    let mut run_cells = 0usize;
    let mut run: Option<String> = None;

    for (_, cluster) in grapheme::graphemes(text) {
        let width = display_cluster_width(cluster).max(1);
        let safe = width == 1 && cluster.chars().all(|c| (c as u32) <= 0xffff);
        if safe {
            if run_cells == 0 {
                run_start = pen_cells;
                run = Some(take_string_with_capacity(strings, text.len()));
            }
            run.as_mut().expect("run starts above").push_str(cluster);
            run_cells += 1;
        } else {
            if run_cells > 0 {
                push_owned_cells(
                    out,
                    run.take().expect("nonempty run has storage"),
                    x + run_start as i32 * metrics.w,
                    baseline,
                    run_cells,
                    metrics,
                    face,
                    color,
                    bg,
                );
                run_cells = 0;
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
                text: glyph,
            });
        }
        pen_cells += width;
    }
    if run_cells > 0 {
        push_owned_cells(
            out,
            run.expect("nonempty run has storage"),
            x + run_start as i32 * metrics.w,
            baseline,
            run_cells,
            metrics,
            face,
            color,
            bg,
        );
    }
}

/// Display columns occupied by one already-segmented grapheme cluster. Combining
/// sequences inherit their base width; emoji keycaps and flags are two cells.
pub(crate) fn display_cluster_width(cluster: &str) -> usize {
    let mut chars = cluster.chars();
    let Some(first) = chars.next() else {
        return 0;
    };
    if cluster.contains('\u{20e3}')
        || cluster.contains('\u{fe0f}')
        || (matches!(first, '\u{1f1e6}'..='\u{1f1ff}') && chars.next().is_some())
    {
        return 2;
    }
    cluster.chars().map(crate::width::width).max().unwrap_or(0) as usize
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

/// The bearing padding around a run: a glyph's ink can spill
/// past its advance box (an italic tail, a box-drawing overhang), so the damage
/// and clip rectangle is padded to never shear a glyph's edge.
fn text_bounds(x: i32, baseline: i32, run_w: i32, m: CellMetrics) -> Rect {
    let pad_x = (m.size as i32 / 2).max(2);
    let pad_y = (m.size as i32 / 4).max(1);
    Rect {
        x: x - pad_x,
        y: baseline - m.ascent - pad_y,
        w: run_w.max(0) + 2 * pad_x,
        h: m.ascent + m.descent + 2 * pad_y,
    }
}

/// Whether a rune is safe to batch into a fixed-pitch [`DrawCmd::Cells`] run:
/// every Basic-Multilingual-Plane scalar is, because none of them are regional
/// indicators or emoji that a grapheme segmenter would merge across cell
/// boundaries (those all live in the astral planes). An astral rune is drawn
/// standalone so one cell always maps to one cluster in a run. Indic conjuncts
/// and other segmentation exotica are the 5% we deliberately punt.
fn cells_safe(rune: char) -> bool {
    (rune as u32) <= 0xFFFF
}

/// The font style a cell's bold/italic attributes select.
fn style_of(cell: Cell) -> FontStyle {
    match (
        cell.attrs.contains(Attrs::BOLD),
        cell.attrs.contains(Attrs::ITALIC),
    ) {
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
    use super::*;
    use crate::color::Color;

    /// Synthetic metrics: a 10x20 cell so column/row math reads off by eye.
    const M: CellMetrics = CellMetrics {
        size: 16,
        w: 10,
        h: 20,
        ascent: 15,
        descent: 5,
    };

    fn feed(s: &mut Screen, bytes: &[u8]) {
        let mut p = crate::vt::Parser::new();
        p.advance_bytes(s, bytes);
    }

    /// Build a list for a screen with the default theme, no selection, cursor
    /// hidden (so the tests that are about content are not perturbed by it).
    fn list_of(s: &Screen) -> DisplayList {
        build_display_list(
            s,
            &Theme::default(),
            M,
            (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            None,
        )
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
            vec![(0, M.ascent, "hello".to_string())],
            "the whole line batches into one run at column 0, baseline = ascent"
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
                (0, M.ascent, "ab".to_string()),
                (2 * M.w, M.ascent, "cd".to_string()),
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
        assert_eq!(runs, vec![(0, M.ascent, "hi".to_string())]);
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
                (0, M.ascent, "a".to_string()),
                (3 * M.w, M.ascent, "b".to_string()),
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
            vec![(0, M.ascent, "e\u{0301}x".to_string())],
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
            .find(|(r, _)| r.y > M.ascent) // below the baseline
            .expect("an underline rule");
        assert_eq!(rule.0.x, 0);
        assert_eq!(rule.0.w, 2 * M.w, "the rule spans the whole run");
        assert_eq!(rule.1, t.fg.to_u32());
    }

    #[test]
    fn block_cursor_fills_its_cell_and_inverts_the_glyph() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X"); // cursor now rests at column 1 (after X)
        s.move_to(0, 0); // park it on the glyph
        let t = Theme::default();
        let list = build_display_list(&s, &t, M, (40, 20), (0, 0), CursorRender::default(), None);
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
        let list = build_display_list(&s, &t, M, (40, 20), (ox, oy), CursorRender::default(), None);
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
            vec![(ox, oy + M.ascent, "X".to_string())],
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
        for shape in [CursorShape::Bar, CursorShape::Underline] {
            let list = build_display_list(
                &s,
                &Theme::default(),
                M,
                (40, 20),
                (0, 0),
                CursorRender {
                    shape,
                    ..CursorRender::default()
                },
                None,
            );
            // No inverted glyph: the only Text/Cells for 'X' is the normal one.
            let glyphs = list
                .iter()
                .filter(|c| matches!(c, DrawCmd::Cells { .. } | DrawCmd::Text { .. }))
                .count();
            assert_eq!(glyphs, 1, "{shape:?} leaves the glyph alone");
        }
    }

    #[test]
    fn hidden_cursor_draws_nothing_extra() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let hidden = build_display_list(
            &s,
            &Theme::default(),
            M,
            (40, 20),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            None,
        );
        // Same as a plain render: base fill + the one glyph run.
        assert_eq!(hidden.len(), 2);
    }

    #[test]
    fn selection_paints_the_selected_cells() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"abcdef");
        let list = build_display_list(
            &s,
            &Theme::default(),
            M,
            (60, 20),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            Some(Selection {
                anchor: (0, 1),
                head: (0, 3),
            }),
        );
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
        let list = build_display_list(
            &s,
            &Theme::default(),
            M,
            (60, 20),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            Some(Selection {
                anchor: (0, 0),
                head: (0, 5),
            }),
        );
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
        let list = build_display_list(
            &s,
            &Theme::default(),
            M,
            (60, 20),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            Some(Selection {
                anchor: (0, 0),
                head: (0, 5),
            }),
        );
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
        // Every damaged rectangle is confined to the middle row's band.
        for r in d {
            assert!(
                r.y >= M.h - M.ascent && r.y + r.h <= 2 * M.h + M.descent,
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
        // The grid-fit inverse divides the surface back into whole cells.
        assert_eq!(m.columns_rows(m.w * 80, m.h * 24), (80, 24));
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
        let runs = cells_runs(&build_display_list(
            &s,
            &Theme::default(),
            M,
            (6 * M.w, 2 * M.h),
            (0, 0),
            CursorRender {
                visible: false,
                ..CursorRender::default()
            },
            None,
        ));
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
        let list = build_display_list(
            &s,
            &Theme::default(),
            M,
            (6 * M.w, 2 * M.h),
            (0, 0),
            CursorRender::default(), // visible + focused
            None,
        );
        // No inverted glyph and no cursor block colour: the cursor sits off-view.
        let cursor_color = Theme::default().cursor.to_u32();
        assert!(
            !fills(&list).iter().any(|(_, c)| *c == cursor_color),
            "the cursor is not drawn while viewing history"
        );
    }

    #[test]
    fn a_scroll_indicator_shows_only_while_scrolled() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        let surface = (6 * M.w, 2 * M.h);
        let indicator = |s: &Screen| {
            build_display_list(
                s,
                &Theme::default(),
                M,
                surface,
                (0, 0),
                CursorRender {
                    visible: false,
                    ..CursorRender::default()
                },
                None,
            )
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == SCROLL_INDICATOR_COLOR))
        };
        assert!(!indicator(&s), "no indicator when pinned to the bottom");
        s.scroll_view_up(1);
        assert!(indicator(&s), "an indicator appears while viewing history");
    }
}
