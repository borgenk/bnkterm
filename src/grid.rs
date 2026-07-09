//! The screen model: the cell grid, cursor, scrollback, and the operations the
//! VT parser drives. This is the type `bytes -> vt::Parser -> grid::Screen` feeds,
//! and where terminal *correctness* lives. The parser holds no grid state and the
//! grid holds no parser state: this file knows only semantic operations (print a
//! rune, move the cursor, erase, scroll), never bytes.
//!
//! Structure, chosen for the access pattern (the structure is the
//! optimization):
//!
//! ```text
//!   Screen ── primary: Buffer ─┐
//!         └── alt:     Buffer ─┴─ each Buffer is a ring of Rows:
//!
//!     scrollback (VecDeque<Row>)   visible (VecDeque<Row>, len == rows)
//!     ┌───┬───┬───┐               ┌───┬───┬───┬───┐
//!     │ … │ … │old│  ◀── scroll ──│row│row│row│row│
//!     └───┴───┴───┘               └───┴───┴───┴───┘
//! ```
//!
//! A scroll rotates *row headers* (a pointer move), never the cells: a
//! full-screen scroll is a `pop_front` + `push_back` (O(1)); a region scroll
//! moves only the headers inside the region. The evicted row's cell buffer is
//! recycled into the new blank row, so a steady scroll allocates nothing.
//!
//! Two shapes of the cell earn a comment because they corrupt a screen when
//! gotten wrong:
//!
//!   * Wide characters (CJK, most emoji) occupy two columns. The left column
//!     carries the rune and the `WIDE_LEADER` attr; the right column is a
//!     `WIDE_SPACER` placeholder the cursor steps over and the renderer skips.
//!     Overwriting either half cleans up its orphaned partner (`write_cell`).
//!
//!   * Combining marks (an accent after its base, a Hangul jamo stack) are rare,
//!     so a `Cell` stores only the base rune inline and extra marks overflow into
//!     a small per-`Row` side list keyed by column. It travels with the row when
//!     it scrolls, and the common all-single-codepoint row pays nothing.

use crate::color::Color;
use crate::mouse::{MouseMode, MouseProtocol};
use crate::vt::Perform;
use crate::width::width;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;

/// Default lines of scrollback the primary screen keeps. Overridable by config
/// later (phase 4); the alternate screen keeps none.
const DEFAULT_SCROLLBACK: usize = 10_000;

/// Columns between default tab stops.
const TAB_WIDTH: usize = 8;

/// A cell's rendition attributes: the SGR styles plus the two layout bits that
/// mark a wide character's halves and the one that records a soft-wrapped line.
/// A bitfield newtype (no `bitflags` crate) so it is one `u16`, `Copy`, and
/// cheap to compare in the damage diff.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Attrs(u16);

impl Attrs {
    pub const BOLD: Attrs = Attrs(1 << 0);
    pub const DIM: Attrs = Attrs(1 << 1);
    pub const ITALIC: Attrs = Attrs(1 << 2);
    pub const UNDERLINE: Attrs = Attrs(1 << 3);
    pub const REVERSE: Attrs = Attrs(1 << 4);
    pub const STRIKE: Attrs = Attrs(1 << 5);
    /// SGR 8 (conceal): the cell keeps its rune but the renderer draws it in the
    /// background color, so it is invisible yet still selectable and copyable.
    pub const HIDDEN: Attrs = Attrs(1 << 6);
    /// The left half of a wide (2-column) character; carries the rune.
    pub const WIDE_LEADER: Attrs = Attrs(1 << 7);
    /// The right half of a wide character; a placeholder the cursor skips.
    pub const WIDE_SPACER: Attrs = Attrs(1 << 8);
    /// Set on a row's last cell when autowrap carried its line onto the next
    /// row, so reflow and selection can tell a soft wrap from a hard newline.
    pub const WRAPPED: Attrs = Attrs(1 << 9);

    /// All flags, in bit order, with their names, for `Debug` and for tests.
    const ALL: [(Attrs, &'static str); 10] = [
        (Attrs::BOLD, "BOLD"),
        (Attrs::DIM, "DIM"),
        (Attrs::ITALIC, "ITALIC"),
        (Attrs::UNDERLINE, "UNDERLINE"),
        (Attrs::REVERSE, "REVERSE"),
        (Attrs::STRIKE, "STRIKE"),
        (Attrs::HIDDEN, "HIDDEN"),
        (Attrs::WIDE_LEADER, "WIDE_LEADER"),
        (Attrs::WIDE_SPACER, "WIDE_SPACER"),
        (Attrs::WRAPPED, "WRAPPED"),
    ];

    pub const fn empty() -> Self {
        Attrs(0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every flag in `other` is set here.
    pub const fn contains(self, other: Attrs) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Attrs) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Attrs) {
        self.0 &= !other.0;
    }

    /// Set or clear `other` by a boolean, the shape SGR wants (e.g. bold on 1,
    /// off on 22).
    pub fn set(&mut self, other: Attrs, on: bool) {
        if on {
            self.insert(other);
        } else {
            self.remove(other);
        }
    }
}

impl std::ops::BitOr for Attrs {
    type Output = Attrs;
    fn bitor(self, rhs: Attrs) -> Attrs {
        Attrs(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Attrs {
    fn bitor_assign(&mut self, rhs: Attrs) {
        self.0 |= rhs.0;
    }
}

/// Lists the set flags (`Attrs(BOLD|ITALIC)`), so a grid dump in a golden test
/// reads like the styling it represents instead of a hex mask.
impl fmt::Debug for Attrs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Attrs(")?;
        if self.is_empty() {
            f.write_str("-")?;
        } else {
            let mut first = true;
            for (flag, name) in Attrs::ALL {
                if self.contains(flag) {
                    if !first {
                        f.write_str("|")?;
                    }
                    f.write_str(name)?;
                    first = false;
                }
            }
        }
        f.write_str(")")
    }
}

/// One grid cell: the base rune of its grapheme cluster, its foreground and
/// background colors, and its rendition attributes. Small, `Copy`, and `Eq` so
/// the per-frame damage diff can compare two screenfuls cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cell {
    /// The base rune. Combining marks, when present, live in the `Row`'s side
    /// list keyed by column (see the module header).
    pub rune: char,
    pub fg: Color,
    pub bg: Color,
    pub attrs: Attrs,
}

impl Cell {
    /// The blank cell: a space in the default colors with no attributes. This is
    /// the grid's initial fill; an erase writes a space in the *current* bg.
    pub const BLANK: Cell = Cell {
        rune: ' ',
        fg: Color::Default,
        bg: Color::Default,
        attrs: Attrs::empty(),
    };

    /// A cell carrying `rune` in the default colors, no attributes. Handy for
    /// tests and for plain printable output before any SGR is seen.
    pub fn new(rune: char) -> Self {
        Cell {
            rune,
            ..Cell::BLANK
        }
    }

    /// Whether this cell is the right-half placeholder of a wide character.
    pub fn is_wide_spacer(self) -> bool {
        self.attrs.contains(Attrs::WIDE_SPACER)
    }

    /// Whether this cell is the left half (the rune) of a wide character.
    pub fn is_wide_leader(self) -> bool {
        self.attrs.contains(Attrs::WIDE_LEADER)
    }
}

impl Default for Cell {
    fn default() -> Self {
        Cell::BLANK
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cell({:?}, fg={:?}, bg={:?}, {:?})",
            self.rune, self.fg, self.bg, self.attrs
        )
    }
}

/// The cursor: a position plus the deferred-wrap flag that implements xterm's
/// last-column rule. `pending_wrap` is set after a glyph lands in the final
/// column; the wrap to the next line is delayed until the next glyph actually
/// arrives, so writing exactly `cols` characters does not scroll.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Cursor {
    row: usize,
    col: usize,
    pending_wrap: bool,
}

/// The current graphic rendition (SGR state): the colors and attributes printed
/// cells receive. `Copy` so a save/restore (DECSC/DECRC) is a plain move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Pen {
    fg: Color,
    bg: Color,
    attrs: Attrs,
}

impl Default for Pen {
    fn default() -> Self {
        Pen {
            fg: Color::Default,
            bg: Color::Default,
            attrs: Attrs::empty(),
        }
    }
}

/// A saved cursor (DECSC): position, pen, and origin mode, restored by DECRC.
#[derive(Clone, Copy)]
struct Saved {
    cursor: Cursor,
    pen: Pen,
    origin: bool,
}

/// One row of cells plus its rare combining marks. Cells are contiguous, so a
/// row is cheap to scan and write. Combining marks live in a tiny side list keyed
/// by column so they ride along when the row scrolls and cost nothing on the
/// common all-single-codepoint row.
#[derive(Clone, Debug)]
struct Row {
    cells: Vec<Cell>,
    combining: Vec<(usize, Vec<char>)>,
}

impl Row {
    fn filled(cols: usize, cell: Cell) -> Self {
        Row {
            cells: vec![cell; cols],
            combining: Vec::new(),
        }
    }

    /// Reset every cell to `blank` and drop combining marks, reusing the existing
    /// allocation so a recycled scroll row never allocates.
    fn reset(&mut self, cols: usize, blank: Cell) {
        if self.cells.len() == cols {
            self.cells.iter_mut().for_each(|c| *c = blank);
        } else {
            self.cells.clear();
            self.cells.resize(cols, blank);
        }
        self.combining.clear();
    }

    fn marks_at(&self, col: usize) -> Option<&[char]> {
        self.combining
            .iter()
            .find(|(c, _)| *c == col)
            .map(|(_, m)| m.as_slice())
    }

    fn add_mark(&mut self, col: usize, mark: char) {
        match self.combining.iter_mut().find(|(c, _)| *c == col) {
            Some((_, marks)) => marks.push(mark),
            None => self.combining.push((col, vec![mark])),
        }
    }

    fn clear_marks(&mut self, col: usize) {
        self.combining.retain(|(c, _)| *c != col);
    }

    /// Grow or shrink the row to `new_cols`, padding with blanks or truncating.
    /// Truncation drops any combining marks past the new edge and blanks a wide
    /// glyph's leader whose spacer just fell off, so no half of a wide cell is
    /// ever left dangling. This is the column half of a terminal resize; it does
    /// not re-wrap soft-wrapped content (see [`Buffer::resize`]).
    fn resize_cols(&mut self, new_cols: usize) {
        let old = self.cells.len();
        if new_cols < old {
            self.cells.truncate(new_cols);
            self.combining.retain(|(c, _)| *c < new_cols);
            if let Some(last) = self.cells.last_mut() {
                if last.is_wide_leader() {
                    *last = Cell::BLANK;
                }
            }
        } else if new_cols > old {
            self.cells.resize(new_cols, Cell::BLANK);
        }
    }
}

/// One screen buffer: the visible rows (a ring so a scroll rotates row headers,
/// never cells), the scrollback ring behind it, the cursor, the DECSTBM scroll
/// region, and the tab stops. Ring mechanics live here; the terminal *semantics*
/// (what each escape means) live on `Screen`.
struct Buffer {
    cols: usize,
    rows: usize,
    /// Visible rows, always exactly `rows` long.
    lines: VecDeque<Row>,
    /// History above the screen, oldest at the front, capped at `scrollback_limit`.
    scrollback: VecDeque<Row>,
    scrollback_limit: usize,
    cursor: Cursor,
    saved: Option<Saved>,
    /// DECSTBM scroll region, inclusive, within `0..rows`.
    scroll_top: usize,
    scroll_bottom: usize,
    /// Tab stops; `tabs[c]` is true where a horizontal tab lands.
    tabs: Vec<bool>,
}

impl Buffer {
    fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let mut lines = VecDeque::with_capacity(rows);
        for _ in 0..rows {
            lines.push_back(Row::filled(cols, Cell::BLANK));
        }
        Buffer {
            cols,
            rows,
            lines,
            scrollback: VecDeque::with_capacity(scrollback_limit),
            scrollback_limit,
            cursor: Cursor::default(),
            saved: None,
            scroll_top: 0,
            scroll_bottom: rows - 1,
            tabs: default_tabs(cols),
        }
    }

    fn line(&self, row: usize) -> Option<&Row> {
        self.lines.get(row)
    }

    fn cell(&self, row: usize, col: usize) -> Cell {
        self.lines
            .get(row)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// The row shown at display position `display_row` when the view is scrolled
    /// `offset` lines up into scrollback. The scrollback and the live lines form
    /// one virtual column of rows; the window of `rows` starts `offset` lines
    /// above the live top. `offset` is assumed already clamped to `0..=scrollback
    /// .len()`. Returns `None` past the end (a short final window).
    fn view_row(&self, display_row: usize, offset: usize) -> Option<&Row> {
        let base = self.scrollback.len().checked_sub(offset)?;
        let idx = base + display_row;
        if idx < self.scrollback.len() {
            self.scrollback.get(idx)
        } else {
            self.lines.get(idx - self.scrollback.len())
        }
    }

    fn set_raw(&mut self, row: usize, col: usize, cell: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            if let Some(slot) = r.cells.get_mut(col) {
                *slot = cell;
            }
        }
    }

    /// Write `cell` at (row, col), first breaking any wide pair it straddles so a
    /// half-overwritten wide glyph never leaves an orphan on screen, and dropping
    /// any combining marks the overwritten cell carried.
    fn write_cell(&mut self, row: usize, col: usize, cell: Cell) {
        let existing = self.cell(row, col);
        if existing.is_wide_leader() {
            self.set_raw(row, col + 1, Cell::BLANK);
        } else if existing.is_wide_spacer() && col > 0 {
            self.set_raw(row, col - 1, Cell::BLANK);
            if let Some(r) = self.lines.get_mut(row) {
                r.clear_marks(col - 1);
            }
        }
        self.set_raw(row, col, cell);
        if let Some(r) = self.lines.get_mut(row) {
            r.clear_marks(col);
        }
    }

    /// Scroll `[top, bottom]` up by `n`, feeding `blank` rows in at the bottom.
    /// When `to_scrollback` and the region reaches the top of the screen, the
    /// rows leaving the top are retained in scrollback; otherwise they are
    /// discarded. Either way the row storage is recycled, so a steady scroll
    /// allocates nothing once scrollback is full.
    fn scroll_up_range(
        &mut self,
        top: usize,
        bottom: usize,
        n: usize,
        blank: Cell,
        to_scrollback: bool,
    ) {
        if top > bottom || bottom >= self.rows {
            return;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(top) else {
                return;
            };
            let mut recycled = if to_scrollback && top == 0 && self.scrollback_limit > 0 {
                self.scrollback.push_back(leaving);
                if self.scrollback.len() > self.scrollback_limit {
                    self.scrollback
                        .pop_front()
                        .unwrap_or_else(|| Row::filled(self.cols, blank))
                } else {
                    Row::filled(self.cols, blank)
                }
            } else {
                leaving
            };
            recycled.reset(self.cols, blank);
            self.lines.insert(bottom, recycled);
        }
    }

    /// Scroll `[top, bottom]` down by `n`, feeding `blank` rows in at the top.
    /// Never touches scrollback (only content leaving the top of the full screen
    /// enters history).
    fn scroll_down_range(&mut self, top: usize, bottom: usize, n: usize, blank: Cell) {
        if top > bottom || bottom >= self.rows {
            return;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(bottom) else {
                return;
            };
            let mut recycled = leaving;
            recycled.reset(self.cols, blank);
            self.lines.insert(top, recycled);
        }
    }

    /// Push a row off the top into scrollback, evicting the oldest if the ring is
    /// full. A no-op when this buffer keeps no history (the alt screen), so its
    /// shrunk-away rows are simply dropped.
    fn push_history(&mut self, row: Row) {
        if self.scrollback_limit == 0 {
            return;
        }
        self.scrollback.push_back(row);
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
        }
    }

    /// Resize the buffer to `new_cols` x `new_rows`. Columns truncate or pad every
    /// row (visible and scrollback); rows grow by appending blanks at the bottom
    /// and shrink by dropping rows below the cursor first, then scrolling the top
    /// into scrollback so the cursor stays on screen. The scroll region resets to
    /// the full screen (as xterm does on resize) and the cursor is re-clamped.
    ///
    /// Deliberately does NOT re-wrap soft-wrapped lines: full-screen programs
    /// repaint on `SIGWINCH`, so what matters is a correctly sized, uncorrupted
    /// grid, not re-flowed history. Scrollback re-wrap is a later nicety.
    fn resize(&mut self, new_cols: usize, new_rows: usize) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);

        if new_cols != self.cols {
            for r in self.lines.iter_mut().chain(self.scrollback.iter_mut()) {
                r.resize_cols(new_cols);
            }
            self.cols = new_cols;
            self.tabs = default_tabs(new_cols);
            if self.cursor.col >= new_cols {
                self.cursor.col = new_cols - 1;
            }
        }

        match new_rows.cmp(&self.rows) {
            Ordering::Greater => {
                for _ in 0..(new_rows - self.rows) {
                    self.lines.push_back(Row::filled(new_cols, Cell::BLANK));
                }
            }
            Ordering::Less => {
                let mut excess = self.rows - new_rows;
                // Rows strictly below the cursor go first (they hold nothing the
                // user is looking at); only then does the top scroll into history.
                let below = self.rows.saturating_sub(self.cursor.row + 1);
                let from_bottom = excess.min(below);
                for _ in 0..from_bottom {
                    self.lines.pop_back();
                }
                excess -= from_bottom;
                for _ in 0..excess {
                    if let Some(top) = self.lines.pop_front() {
                        self.push_history(top);
                    }
                    self.cursor.row = self.cursor.row.saturating_sub(1);
                }
            }
            Ordering::Equal => {}
        }

        self.rows = new_rows;
        self.scroll_top = 0;
        self.scroll_bottom = new_rows - 1;
        if self.cursor.row >= new_rows {
            self.cursor.row = new_rows - 1;
        }
        self.cursor.pending_wrap = false;
    }

    fn clear_line_full(&mut self, row: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            r.cells.iter_mut().for_each(|c| *c = blank);
            r.combining.clear();
        }
    }

    /// Blank cells `[start, end)` of `row`, leaving the rest untouched.
    fn clear_line_range(&mut self, row: usize, start: usize, end: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let hi = end.min(r.cells.len());
            if start < hi {
                if let Some(slice) = r.cells.get_mut(start..hi) {
                    slice.iter_mut().for_each(|c| *c = blank);
                }
                r.combining.retain(|(c, _)| *c < start || *c >= hi);
            }
        }
    }

    /// ICH: shift `[col, len)` right by `n`, blanking the `n` opened cells; cells
    /// pushed past the right edge are lost.
    fn insert_blanks(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.cells.copy_within(col..len - n, col + n);
            if let Some(slice) = r.cells.get_mut(col..col + n) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|(c, _)| *c < col || *c + n < len);
            r.combining.iter_mut().for_each(|(c, _)| {
                if *c >= col {
                    *c += n
                }
            });
        }
    }

    /// DCH: shift `[col+n, len)` left by `n`, blanking the `n` cells at the right.
    fn delete_chars(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.cells.copy_within(col + n..len, col);
            if let Some(slice) = r.cells.get_mut(len - n..len) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|(c, _)| *c < col || *c >= col + n);
            r.combining.iter_mut().for_each(|(c, _)| {
                if *c >= col + n {
                    *c -= n
                }
            });
        }
    }

    fn clear_all(&mut self, blank: Cell) {
        for row in 0..self.rows {
            self.clear_line_full(row, blank);
        }
    }
}

/// A designated character set. bnkterm supports ASCII and the DEC Special
/// Graphics (line-drawing) set, which is what box-drawing TUIs rely on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charset {
    Ascii,
    DecSpecialGraphics,
}

/// The cursor's drawn shape, set by `DECSCUSR` (`CSI Ps SP q`). The renderer maps
/// this to how it paints the cursor; whether it blinks is tracked separately.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CursorStyle {
    #[default]
    Block,
    Underline,
    Bar,
}

/// The terminal grid: primary and alternate buffers, the active graphic rendition
/// (pen), and the terminal-wide modes. This is the type the VT parser drives (via
/// the `Perform` impl below); the renderer reads the visible cells back out.
/// Positions in its API are 0-based (the parser converts the 1-based numbers
/// escape sequences carry).
pub struct Screen {
    primary: Buffer,
    alt: Buffer,
    on_alt: bool,
    pen: Pen,
    autowrap: bool,        // DECAWM (?7)
    origin_mode: bool,     // DECOM (?6)
    cursor_visible: bool,  // DECTCEM (?25)
    insert_mode: bool,     // IRM (4)
    bracketed_paste: bool, // ?2004
    app_cursor_keys: bool, // DECCKM (?1)
    keypad_app: bool,      // DECKPAM/DECKPNM, for the input encoder (phase 3)
    /// The two designated charsets and whether GL points at G1 (SO) not G0 (SI).
    g0: Charset,
    g1: Charset,
    gl_is_g1: bool,
    title: String,
    /// How many lines the *view* is scrolled up into the primary scrollback (0 =
    /// pinned to the live bottom). A display concern, not terminal state: writes
    /// always land on the live screen; only what the renderer shows shifts. The
    /// alt screen has no scrollback, so it forces this to 0.
    view_offset: usize,
    /// Mouse reporting the child asked for (`?1000`/`?1002`/`?1003`/`?1006`); the
    /// app reads it to decide whether a pointer event goes to the child or drives
    /// local selection/scroll.
    mouse: MouseMode,
    /// The cursor shape and whether it blinks, from `DECSCUSR`. Defaults to a
    /// steady block: xterm's default is a *blinking* block, but every modern
    /// terminal opens steady (commonly via a `cursor-style-blink = false` setting),
    /// so that is bnkterm's power-on default. A program can still request blink with
    /// `DECSCUSR`.
    cursor_style: CursorStyle,
    cursor_blink: bool,
    /// Bytes to write back to the child in answer to a query (DA, DSR). The grid
    /// holds no PTY, so it queues its replies as data here; the app drains them
    /// after each parse and writes them to the master fd. Kept tiny (queries are
    /// rare), reused across parses.
    responses: Vec<u8>,
}

impl Screen {
    pub fn new(cols: usize, rows: usize) -> Self {
        Screen {
            primary: Buffer::new(cols, rows, DEFAULT_SCROLLBACK),
            alt: Buffer::new(cols, rows, 0),
            on_alt: false,
            pen: Pen::default(),
            autowrap: true,
            origin_mode: false,
            cursor_visible: true,
            insert_mode: false,
            bracketed_paste: false,
            app_cursor_keys: false,
            keypad_app: false,
            g0: Charset::Ascii,
            g1: Charset::Ascii,
            gl_is_g1: false,
            title: String::new(),
            view_offset: 0,
            mouse: MouseMode::default(),
            cursor_style: CursorStyle::Block,
            cursor_blink: false,
            responses: Vec::new(),
        }
    }

    // ---- accessors ----------------------------------------------------------

    pub fn dimensions(&self) -> (usize, usize) {
        let b = self.active();
        (b.cols, b.rows)
    }

    pub fn is_alt(&self) -> bool {
        self.on_alt
    }

    /// The cursor position as `(row, col)`, 0-based.
    pub fn cursor(&self) -> (usize, usize) {
        let c = self.active().cursor;
        (c.row, c.col)
    }

    pub fn cursor_visible(&self) -> bool {
        self.cursor_visible
    }

    pub fn bracketed_paste(&self) -> bool {
        self.bracketed_paste
    }

    /// The mouse reporting the child asked for (off when it wants none).
    pub fn mouse_mode(&self) -> MouseMode {
        self.mouse
    }

    /// The cursor shape the child selected via `DECSCUSR` (default block).
    pub fn cursor_style(&self) -> CursorStyle {
        self.cursor_style
    }

    /// Whether the child asked the cursor to blink (default false; see
    /// `cursor_blink`).
    pub fn cursor_blinks(&self) -> bool {
        self.cursor_blink
    }

    /// Take the bytes queued to write back to the child (DA/DSR answers), leaving
    /// the queue empty. Empty in the common case (no query since the last parse),
    /// so the app can call it after every parse cheaply.
    pub fn take_responses(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.responses)
    }

    pub fn app_cursor_keys(&self) -> bool {
        self.app_cursor_keys
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    /// The cell at visible `(row, col)`; a blank cell if out of range.
    pub fn cell(&self, row: usize, col: usize) -> Cell {
        self.active().cell(row, col)
    }

    /// Combining marks attached to visible `(row, col)`, if any.
    pub fn marks_at(&self, row: usize, col: usize) -> Option<&[char]> {
        self.active().line(row).and_then(|r| r.marks_at(col))
    }

    // ---- scrollback viewport ------------------------------------------------

    /// How many lines the view is scrolled up into scrollback (0 = live bottom).
    /// Always 0 on the alt screen, which keeps no history.
    pub fn view_offset(&self) -> usize {
        if self.on_alt {
            0
        } else {
            self.view_offset
        }
    }

    /// Whether the view is currently showing history rather than the live bottom.
    pub fn is_scrolled(&self) -> bool {
        self.view_offset() > 0
    }

    /// How many lines of scrollback the primary screen holds (for a scroll-position
    /// indicator).
    pub fn scrollback_len(&self) -> usize {
        self.primary.scrollback.len()
    }

    /// Scroll the view up into history by `n` lines, clamped to the top. A no-op on
    /// the alt screen.
    pub fn scroll_view_up(&mut self, n: usize) {
        if self.on_alt {
            return;
        }
        self.view_offset = (self.view_offset + n).min(self.primary.scrollback.len());
    }

    /// Scroll the view back down toward the live bottom by `n` lines.
    pub fn scroll_view_down(&mut self, n: usize) {
        self.view_offset = self.view_offset.saturating_sub(n);
    }

    /// Jump the view to the oldest scrollback line.
    pub fn scroll_view_to_top(&mut self) {
        if !self.on_alt {
            self.view_offset = self.primary.scrollback.len();
        }
    }

    /// Pin the view back to the live bottom (what output and fresh input do).
    pub fn scroll_view_to_bottom(&mut self) {
        self.view_offset = 0;
    }

    /// The cell shown at display `(row, col)` honouring the scroll offset: the live
    /// cell when pinned to the bottom, else the scrollback row scrolled into view.
    pub fn view_cell(&self, row: usize, col: usize) -> Cell {
        let off = self.view_offset();
        if off == 0 {
            return self.active().cell(row, col);
        }
        self.active()
            .view_row(row, off)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// Combining marks at display `(row, col)` honouring the scroll offset.
    pub fn view_marks(&self, row: usize, col: usize) -> Option<&[char]> {
        let off = self.view_offset();
        if off == 0 {
            return self.marks_at(row, col);
        }
        self.active()
            .view_row(row, off)
            .and_then(|r| r.marks_at(col))
    }

    /// The text of a linear selection over the current view, `a`..`b` inclusive in
    /// display `(row, col)` cells (either order). Rows join with `\n`, except a
    /// soft-wrapped row joins with nothing (the two display rows are one logical
    /// line, so a selection across a wrap copies as unbroken text). Trailing
    /// blanks on a row are dropped, wide spacers skipped, combining marks kept.
    pub fn selection_text(&self, a: (usize, usize), b: (usize, usize)) -> String {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        let (cols, rows) = self.dimensions();
        let last_row = end.0.min(rows.saturating_sub(1));
        let mut out = String::new();
        for row in start.0..=last_row {
            let first = if row == start.0 { start.1 } else { 0 };
            let last = if row == end.0 {
                end.1
            } else {
                cols.saturating_sub(1)
            };
            let mut line = String::new();
            for col in first..=last.min(cols.saturating_sub(1)) {
                let cell = self.view_cell(row, col);
                if cell.is_wide_spacer() {
                    continue;
                }
                line.push(cell.rune);
                if let Some(marks) = self.view_marks(row, col) {
                    line.extend(marks);
                }
            }
            out.push_str(line.trim_end_matches(' '));
            if row != last_row {
                // A soft wrap continues the same logical line: no newline.
                let wrapped = self
                    .view_cell(row, cols.saturating_sub(1))
                    .attrs
                    .contains(Attrs::WRAPPED);
                if !wrapped {
                    out.push('\n');
                }
            }
        }
        out
    }

    /// A visible row as text: base runes with their combining marks, wide spacers
    /// skipped. For selection/copy later, and for readable test assertions.
    pub fn row_string(&self, row: usize) -> String {
        let b = self.active();
        let Some(r) = b.line(row) else {
            return String::new();
        };
        let mut s = String::new();
        for (col, cell) in r.cells.iter().enumerate() {
            if cell.is_wide_spacer() {
                continue;
            }
            s.push(cell.rune);
            if let Some(marks) = r.marks_at(col) {
                s.extend(marks);
            }
        }
        s
    }

    /// The whole visible screen as text, one row per line, trailing blanks
    /// trimmed. A debugging and golden-diff convenience.
    pub fn dump(&self) -> String {
        let (_, rows) = self.dimensions();
        (0..rows)
            .map(|r| {
                let mut line = self.row_string(r);
                let trimmed = line.trim_end();
                line.truncate(trimmed.len());
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A complete, human-diffable snapshot of the terminal state: dimensions,
    /// cursor, mode flags, the visible grid as text, and every non-default cell's
    /// colors and attributes. The golden-oracle tests assert this field-for-field,
    /// so a regression reads like a screenshot with a style key beneath it. A
    /// blank default cell is the omission: it shows as a space in `grid:` and
    /// carries no `style:` line.
    pub fn snapshot(&self) -> String {
        use std::fmt::Write;
        let (cols, rows) = self.dimensions();
        let (cr, cc) = self.cursor();
        let mut out = String::new();
        let _ = writeln!(out, "size {cols}x{rows}");
        let _ = writeln!(out, "cursor {cr},{cc}");
        let _ = writeln!(
            out,
            "flags alt={} autowrap={} origin={} insert={} cursor_vis={} \
             bracketed={} appcursor={} keypad={}",
            self.on_alt,
            self.autowrap,
            self.origin_mode,
            self.insert_mode,
            self.cursor_visible,
            self.bracketed_paste,
            self.app_cursor_keys,
            self.keypad_app,
        );
        let _ = writeln!(out, "title {:?}", self.title);
        out.push_str("grid:\n");
        for r in 0..rows {
            let _ = writeln!(out, "|{}|", self.row_string(r));
        }
        out.push_str("style:\n");
        for r in 0..rows {
            for c in 0..cols {
                let cell = self.cell(r, c);
                if cell.fg != Color::Default || cell.bg != Color::Default || !cell.attrs.is_empty()
                {
                    let _ = writeln!(
                        out,
                        "r{r}c{c} {:?} fg={:?} bg={:?} {:?}",
                        cell.rune, cell.fg, cell.bg, cell.attrs
                    );
                }
            }
        }
        out
    }

    // ---- internal helpers ---------------------------------------------------

    fn active(&self) -> &Buffer {
        if self.on_alt {
            &self.alt
        } else {
            &self.primary
        }
    }

    fn active_mut(&mut self) -> &mut Buffer {
        if self.on_alt {
            &mut self.alt
        } else {
            &mut self.primary
        }
    }

    /// The cell an erase or scroll fills with: a space in the current background
    /// (background-color erase, as xterm does), default foreground, no attrs.
    fn blank_cell(&self) -> Cell {
        Cell {
            rune: ' ',
            fg: Color::Default,
            bg: self.pen.bg,
            attrs: Attrs::empty(),
        }
    }

    // ---- printing -----------------------------------------------------------

    /// Print one scalar value at the cursor, applying wide-character and wrap
    /// semantics. Combining marks (width 0) attach to the preceding cell.
    pub fn print(&mut self, c: char) {
        let cw = usize::from(width(c));
        if cw == 0 {
            self.put_combining(c);
            return;
        }
        let cols = self.active().cols;
        if cols == 0 {
            return;
        }
        // Deferred wrap: a prior glyph filled the last column (xterm last-column
        // rule); perform the wrap now, before placing this glyph.
        if self.active().cursor.pending_wrap {
            self.wrap_line();
        }
        // A wide glyph with a single column left cannot fit.
        if cw == 2 && self.active().cursor.col + 1 >= cols {
            if self.autowrap {
                self.wrap_line();
            } else {
                // No autowrap: back up so it overwrites the last two columns.
                self.active_mut().cursor.col = cols.saturating_sub(2);
            }
        }

        let pen = self.pen;
        let insert = self.insert_mode;
        let (row, col) = {
            let cur = self.active().cursor;
            (cur.row, cur.col)
        };
        if insert {
            let blank = self.blank_cell();
            self.active_mut().insert_blanks(row, col, cw, blank);
        }
        let leader = Cell {
            rune: c,
            fg: pen.fg,
            bg: pen.bg,
            attrs: if cw == 2 {
                pen.attrs | Attrs::WIDE_LEADER
            } else {
                pen.attrs
            },
        };
        {
            let b = self.active_mut();
            b.write_cell(row, col, leader);
            if cw == 2 {
                let spacer = Cell {
                    rune: ' ',
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: pen.attrs | Attrs::WIDE_SPACER,
                };
                b.write_cell(row, col + 1, spacer);
            }
        }

        let autowrap = self.autowrap;
        let b = self.active_mut();
        let cols = b.cols;
        if b.cursor.col + cw >= cols {
            b.cursor.col = cols.saturating_sub(1);
            b.cursor.pending_wrap = autowrap;
        } else {
            b.cursor.col += cw;
            b.cursor.pending_wrap = false;
        }
    }

    /// Attach a zero-width combining mark to the base cell to the left of where
    /// the next glyph would land, composing onto a wide glyph's leader (not its
    /// spacer). Dropped only when there is no cell to attach to (column 0).
    fn put_combining(&mut self, mark: char) {
        let cols = self.active().cols;
        let cur = self.active().cursor;
        let (row, mut col) = if cur.pending_wrap {
            (cur.row, cols.saturating_sub(1))
        } else if cur.col > 0 {
            (cur.row, cur.col - 1)
        } else {
            return;
        };
        if self.active().cell(row, col).is_wide_spacer() && col > 0 {
            col -= 1;
        }
        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.add_mark(col, mark);
        }
    }

    /// The soft wrap: mark the row being left as `WRAPPED`, then move to the
    /// start of the next line (scrolling if at the bottom of the region).
    fn wrap_line(&mut self) {
        let cols = self.active().cols;
        let row = self.active().cursor.row;
        if cols > 0 {
            let last = cols - 1;
            if let Some(r) = self.active_mut().lines.get_mut(row) {
                if let Some(cell) = r.cells.get_mut(last) {
                    cell.attrs.insert(Attrs::WRAPPED);
                }
            }
        }
        self.active_mut().cursor.pending_wrap = false;
        self.line_feed();
        self.carriage_return();
    }

    // ---- C0 controls --------------------------------------------------------

    pub fn carriage_return(&mut self) {
        let b = self.active_mut();
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
    }

    /// LF / IND: move down one row, scrolling the region up at the bottom margin.
    pub fn line_feed(&mut self) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        if b.cursor.row == b.scroll_bottom {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            b.scroll_up_range(top, bottom, 1, blank, top == 0);
        } else if b.cursor.row + 1 < b.rows {
            b.cursor.row += 1;
        }
    }

    /// RI: move up one row, scrolling the region down at the top margin.
    pub fn reverse_index(&mut self) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        if b.cursor.row == b.scroll_top {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            b.scroll_down_range(top, bottom, 1, blank);
        } else if b.cursor.row > 0 {
            b.cursor.row -= 1;
        }
    }

    /// NEL: carriage return plus line feed.
    pub fn next_line(&mut self) {
        self.line_feed();
        self.carriage_return();
    }

    pub fn backspace(&mut self) {
        let b = self.active_mut();
        if b.cursor.col > 0 {
            b.cursor.col -= 1;
        }
        b.cursor.pending_wrap = false;
    }

    /// HT: advance to the next tab stop, or the last column.
    pub fn tab(&mut self) {
        let b = self.active_mut();
        let mut c = b.cursor.col + 1;
        while c < b.cols && !b.tabs.get(c).copied().unwrap_or(false) {
            c += 1;
        }
        b.cursor.col = c.min(b.cols.saturating_sub(1));
        b.cursor.pending_wrap = false;
    }

    /// HTS: set a tab stop at the cursor column.
    pub fn set_tab_stop(&mut self) {
        let b = self.active_mut();
        let col = b.cursor.col;
        if let Some(stop) = b.tabs.get_mut(col) {
            *stop = true;
        }
    }

    /// TBC: clear the tab stop at the cursor (mode 0) or all stops (mode 3).
    pub fn clear_tab_stop(&mut self, mode: u16) {
        let b = self.active_mut();
        match mode {
            3 => b.tabs.iter_mut().for_each(|t| *t = false),
            _ => {
                let col = b.cursor.col;
                if let Some(stop) = b.tabs.get_mut(col) {
                    *stop = false;
                }
            }
        }
    }

    // ---- cursor movement ----------------------------------------------------

    pub fn move_up(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        let limit = if b.cursor.row >= b.scroll_top {
            b.scroll_top
        } else {
            0
        };
        b.cursor.row = b.cursor.row.saturating_sub(n).max(limit);
        b.cursor.pending_wrap = false;
    }

    pub fn move_down(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        let limit = if b.cursor.row <= b.scroll_bottom {
            b.scroll_bottom
        } else {
            b.rows - 1
        };
        b.cursor.row = (b.cursor.row + n).min(limit);
        b.cursor.pending_wrap = false;
    }

    pub fn move_forward(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        b.cursor.col = (b.cursor.col + n).min(b.cols.saturating_sub(1));
        b.cursor.pending_wrap = false;
    }

    pub fn move_back(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        b.cursor.col = b.cursor.col.saturating_sub(n);
        b.cursor.pending_wrap = false;
    }

    /// CUP / HVP: move to `(row, col)`, 0-based, honoring origin mode (which
    /// makes rows relative to and bounded by the scroll region).
    pub fn move_to(&mut self, row: usize, col: usize) {
        let origin = self.origin_mode;
        let b = self.active_mut();
        b.cursor.row = if origin {
            (b.scroll_top + row).min(b.scroll_bottom)
        } else {
            row.min(b.rows - 1)
        };
        b.cursor.col = col.min(b.cols - 1);
        b.cursor.pending_wrap = false;
    }

    /// CHA / HPA: move to an absolute column.
    pub fn move_to_col(&mut self, col: usize) {
        let b = self.active_mut();
        b.cursor.col = col.min(b.cols - 1);
        b.cursor.pending_wrap = false;
    }

    /// VPA: move to a row (origin-aware, like `move_to`).
    pub fn move_to_row(&mut self, row: usize) {
        let origin = self.origin_mode;
        let b = self.active_mut();
        b.cursor.row = if origin {
            (b.scroll_top + row).min(b.scroll_bottom)
        } else {
            row.min(b.rows - 1)
        };
        b.cursor.pending_wrap = false;
    }

    // ---- erase --------------------------------------------------------------

    /// EL: erase in line. 0 = cursor to end, 1 = start to cursor, 2 = whole line.
    pub fn erase_line(&mut self, mode: u16) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col, cols) = (b.cursor.row, b.cursor.col, b.cols);
        let (start, end) = match mode {
            1 => (0, col + 1),
            2 => (0, cols),
            _ => (col, cols),
        };
        b.clear_line_range(row, start, end, blank);
        b.cursor.pending_wrap = false;
    }

    /// ED: erase in display. 0 = cursor to end, 1 = start to cursor, 2 = all,
    /// 3 = all plus scrollback.
    pub fn erase_display(&mut self, mode: u16) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col, rows, cols) = (b.cursor.row, b.cursor.col, b.rows, b.cols);
        match mode {
            1 => {
                for r in 0..row {
                    b.clear_line_full(r, blank);
                }
                b.clear_line_range(row, 0, col + 1, blank);
            }
            2 | 3 => {
                b.clear_all(blank);
                if mode == 3 {
                    b.scrollback.clear();
                }
            }
            _ => {
                b.clear_line_range(row, col, cols, blank);
                for r in (row + 1)..rows {
                    b.clear_line_full(r, blank);
                }
            }
        }
        b.cursor.pending_wrap = false;
    }

    /// ECH: erase `n` cells from the cursor, without shifting the rest.
    pub fn erase_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.clear_line_range(row, col, col + n.max(1), blank);
    }

    // ---- insert / delete ----------------------------------------------------

    /// ICH: insert `n` blanks at the cursor, shifting the rest of the line right.
    pub fn insert_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.insert_blanks(row, col, n.max(1), blank);
    }

    /// DCH: delete `n` cells at the cursor, shifting the rest of the line left.
    pub fn delete_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.delete_chars(row, col, n.max(1), blank);
    }

    /// IL: insert `n` blank lines at the cursor row, within the scroll region.
    pub fn insert_lines(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        if b.cursor.row < b.scroll_top || b.cursor.row > b.scroll_bottom {
            return;
        }
        let (top, bottom) = (b.cursor.row, b.scroll_bottom);
        b.scroll_down_range(top, bottom, n.max(1), blank);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
    }

    /// DL: delete `n` lines at the cursor row, within the scroll region.
    pub fn delete_lines(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        if b.cursor.row < b.scroll_top || b.cursor.row > b.scroll_bottom {
            return;
        }
        let (top, bottom) = (b.cursor.row, b.scroll_bottom);
        b.scroll_up_range(top, bottom, n.max(1), blank, false);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
    }

    // ---- scrolling ----------------------------------------------------------

    /// DECSTBM: set the scroll region to `[top, bottom]` (0-based, inclusive).
    /// An empty or inverted region resets to the full screen. Homes the cursor.
    pub fn set_scroll_region(&mut self, top: usize, bottom: usize) {
        {
            let b = self.active_mut();
            let bottom = bottom.min(b.rows - 1);
            if top < bottom {
                b.scroll_top = top;
                b.scroll_bottom = bottom;
            } else {
                b.scroll_top = 0;
                b.scroll_bottom = b.rows - 1;
            }
        }
        self.move_to(0, 0);
    }

    /// SU: scroll the region up `n` lines (content moves up; no scrollback).
    pub fn scroll_up(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        b.scroll_up_range(top, bottom, n.max(1), blank, false);
    }

    /// SD: scroll the region down `n` lines (content moves down).
    pub fn scroll_down(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        b.scroll_down_range(top, bottom, n.max(1), blank);
    }

    // ---- save / restore -----------------------------------------------------

    /// DECSC: save the cursor, pen, and origin mode.
    pub fn save_cursor(&mut self) {
        let (pen, origin) = (self.pen, self.origin_mode);
        let b = self.active_mut();
        b.saved = Some(Saved {
            cursor: b.cursor,
            pen,
            origin,
        });
    }

    /// DECRC: restore what DECSC saved, or home the cursor if nothing was saved.
    pub fn restore_cursor(&mut self) {
        let saved = self.active().saved;
        match saved {
            Some(s) => {
                self.pen = s.pen;
                self.origin_mode = s.origin;
                let b = self.active_mut();
                b.cursor = s.cursor;
                b.cursor.row = b.cursor.row.min(b.rows - 1);
                b.cursor.col = b.cursor.col.min(b.cols - 1);
            }
            None => self.move_to(0, 0),
        }
    }

    // ---- rendition (SGR) ----------------------------------------------------

    /// Apply an SGR sequence, updating the pen. An empty parameter list is a
    /// reset (SGR 0). Handles ANSI-16, bright, 256 (`38;5;n`), and truecolor
    /// (`38;2;r;g;b`) for both foreground (38) and background (48).
    pub fn sgr(&mut self, params: &[u16]) {
        if params.is_empty() {
            self.pen = Pen::default();
            return;
        }
        let mut i = 0;
        while i < params.len() {
            let p = params[i];
            match p {
                0 => self.pen = Pen::default(),
                1 => self.pen.attrs.insert(Attrs::BOLD),
                2 => self.pen.attrs.insert(Attrs::DIM),
                3 => self.pen.attrs.insert(Attrs::ITALIC),
                4 => self.pen.attrs.insert(Attrs::UNDERLINE),
                7 => self.pen.attrs.insert(Attrs::REVERSE),
                8 => self.pen.attrs.insert(Attrs::HIDDEN),
                9 => self.pen.attrs.insert(Attrs::STRIKE),
                21 => self.pen.attrs.remove(Attrs::BOLD),
                22 => self.pen.attrs.remove(Attrs::BOLD | Attrs::DIM),
                23 => self.pen.attrs.remove(Attrs::ITALIC),
                24 => self.pen.attrs.remove(Attrs::UNDERLINE),
                27 => self.pen.attrs.remove(Attrs::REVERSE),
                28 => self.pen.attrs.remove(Attrs::HIDDEN),
                29 => self.pen.attrs.remove(Attrs::STRIKE),
                30..=37 => self.pen.fg = Color::Ansi((p - 30) as u8),
                38 => {
                    let (color, advance) = parse_ext_color(&params[i..]);
                    if let Some(color) = color {
                        self.pen.fg = color;
                    }
                    i += advance;
                }
                39 => self.pen.fg = Color::Default,
                40..=47 => self.pen.bg = Color::Ansi((p - 40) as u8),
                48 => {
                    let (color, advance) = parse_ext_color(&params[i..]);
                    if let Some(color) = color {
                        self.pen.bg = color;
                    }
                    i += advance;
                }
                49 => self.pen.bg = Color::Default,
                90..=97 => self.pen.fg = Color::Ansi((p - 90 + 8) as u8),
                100..=107 => self.pen.bg = Color::Ansi((p - 100 + 8) as u8),
                _ => {}
            }
            i += 1;
        }
    }

    // ---- modes --------------------------------------------------------------

    /// Set or reset a mode. `private` distinguishes the DEC private modes
    /// (`?`-prefixed, like DECAWM) from the ANSI modes (like IRM).
    pub fn set_mode(&mut self, mode: u16, private: bool, enable: bool) {
        if private {
            match mode {
                1 => self.app_cursor_keys = enable,
                6 => {
                    self.origin_mode = enable;
                    self.move_to(0, 0);
                }
                7 => self.autowrap = enable,
                25 => self.cursor_visible = enable,
                47 | 1047 => self.switch_alt(enable),
                1049 => {
                    if enable {
                        self.save_cursor();
                        self.switch_alt(true);
                    } else {
                        self.switch_alt(false);
                        self.restore_cursor();
                    }
                }
                2004 => self.bracketed_paste = enable,
                // Mouse reporting: the ?1000/?1002/?1003 levels are mutually
                // exclusive (enabling one, or disabling any, sets the level);
                // ?1006 is the orthogonal SGR encoding flag.
                1000 => self.set_mouse_protocol(MouseProtocol::Press, enable),
                1002 => self.set_mouse_protocol(MouseProtocol::ButtonEvent, enable),
                1003 => self.set_mouse_protocol(MouseProtocol::AnyEvent, enable),
                1006 => self.mouse.sgr = enable,
                _ => {}
            }
        } else if mode == 4 {
            self.insert_mode = enable;
        }
    }

    /// Set or clear a mouse-reporting level. Disabling any level turns reporting
    /// off (programs toggle a single level, so this is the common, correct case).
    fn set_mouse_protocol(&mut self, protocol: MouseProtocol, enable: bool) {
        self.mouse.protocol = if enable { protocol } else { MouseProtocol::Off };
    }

    /// DECSCUSR: the `Ps` argument selects both the shape and whether it blinks
    /// (odd/zero blink, even steady). An unknown `Ps` is ignored.
    fn set_cursor_style(&mut self, ps: u16) {
        let (style, blink) = match ps {
            0 | 1 => (CursorStyle::Block, true),
            2 => (CursorStyle::Block, false),
            3 => (CursorStyle::Underline, true),
            4 => (CursorStyle::Underline, false),
            5 => (CursorStyle::Bar, true),
            6 => (CursorStyle::Bar, false),
            _ => return,
        };
        self.cursor_style = style;
        self.cursor_blink = blink;
    }

    /// DA (Send Device Attributes): answer a program's "what are you?" probe.
    /// Primary (`CSI c`) reports a VT100 with the Advanced Video Option, the
    /// conservative, universally-understood identity (real capabilities come from
    /// `TERM`); secondary (`CSI > c`) gives a benign version triple. Answering at
    /// all is the point: a program that queries and gets nothing can hang.
    fn device_attributes(&mut self, private: u8) {
        match private {
            0 => self.respond(b"\x1b[?1;2c"),
            b'>' => self.respond(b"\x1b[>0;0;0c"),
            _ => {}
        }
    }

    /// DSR (Device Status Report): `5 n` asks if we are OK (yes), `6 n` asks for
    /// the cursor position (CPR). The `?6 n` private form is the extended report
    /// (DECXCPR) some programs use. The position is 1-based and origin-mode aware.
    fn device_status(&mut self, params: &[u16], private: u8) {
        let ps = params.first().copied().unwrap_or(0);
        match (private, ps) {
            (0, 5) => self.respond(b"\x1b[0n"),
            (0, 6) => {
                let (row, col) = self.report_position();
                self.respond(b"\x1b[");
                push_decimal(&mut self.responses, row);
                self.responses.push(b';');
                push_decimal(&mut self.responses, col);
                self.responses.push(b'R');
            }
            (b'?', 6) => {
                let (row, col) = self.report_position();
                self.respond(b"\x1b[?");
                push_decimal(&mut self.responses, row);
                self.responses.push(b';');
                push_decimal(&mut self.responses, col);
                self.responses.extend_from_slice(b";1R");
            }
            _ => {}
        }
    }

    /// The cursor position for a CPR, 1-based, relative to the scroll region when
    /// origin mode is on (as the report expects).
    fn report_position(&self) -> (u32, u32) {
        let (mut row, col) = self.cursor();
        if self.origin_mode {
            row = row.saturating_sub(self.active().scroll_top);
        }
        (row as u32 + 1, col as u32 + 1)
    }

    /// Queue bytes to be written back to the child.
    fn respond(&mut self, bytes: &[u8]) {
        self.responses.extend_from_slice(bytes);
    }

    /// Enter or leave the alternate screen. Entering clears it and homes the
    /// cursor; the primary buffer is untouched, so leaving reveals it intact.
    fn switch_alt(&mut self, enable: bool) {
        if enable == self.on_alt {
            return;
        }
        // Entering or leaving the alt screen shows a live view, never stale
        // history; the alt screen has no scrollback to scroll anyway.
        self.view_offset = 0;
        if enable {
            let blank = self.blank_cell();
            self.alt.clear_all(blank);
            self.alt.cursor = Cursor::default();
            self.alt.scroll_top = 0;
            self.alt.scroll_bottom = self.alt.rows - 1;
            self.alt.saved = None;
            self.on_alt = true;
        } else {
            self.on_alt = false;
        }
    }

    // ---- misc ---------------------------------------------------------------

    /// RIS: hard reset to the power-on state, keeping the current dimensions.
    pub fn reset(&mut self) {
        let (cols, rows) = self.dimensions();
        *self = Screen::new(cols, rows);
    }

    /// Resize both screens to `cols` x `rows` (clamped to at least 1x1). The app
    /// calls this when the window's pixel size divided by the cell size yields a
    /// new grid; it then sends the child the matching `TIOCSWINSZ`. Content is
    /// preserved and re-clamped, not re-wrapped (see [`Buffer::resize`]).
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let cols = cols.max(1);
        let rows = rows.max(1);
        self.primary.resize(cols, rows);
        self.alt.resize(cols, rows);
        // Scrollback was reindexed; a stale offset would point at the wrong rows.
        self.view_offset = 0;
    }

    /// OSC 0/2: set the window title.
    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    /// DECKPAM/DECKPNM keypad mode, read by the input encoder (phase 3).
    pub fn keypad_app(&self) -> bool {
        self.keypad_app
    }

    /// DECALN: fill the whole screen with 'E' and home the cursor (a vttest
    /// alignment pattern; useful for confirming glyph placement early).
    pub fn decaln(&mut self) {
        let cell = Cell::new('E');
        let b = self.active_mut();
        for row in 0..b.rows {
            if let Some(r) = b.lines.get_mut(row) {
                r.cells.iter_mut().for_each(|c| *c = cell);
                r.combining.clear();
            }
        }
        b.cursor = Cursor::default();
    }

    /// The charset GL currently maps through (G0 or G1).
    fn active_charset(&self) -> Charset {
        if self.gl_is_g1 {
            self.g1
        } else {
            self.g0
        }
    }

    /// Translate a printable char through the active charset (DEC Special
    /// Graphics remaps `_`..`~` to line-drawing glyphs; ASCII is identity).
    fn map_glyph(&self, c: char) -> char {
        match self.active_charset() {
            Charset::Ascii => c,
            Charset::DecSpecialGraphics => dec_special_graphics(c),
        }
    }
}

/// Tab stops every [`TAB_WIDTH`] columns (the terminal default).
fn default_tabs(cols: usize) -> Vec<bool> {
    (0..cols).map(|c| c % TAB_WIDTH == 0).collect()
}

/// Clamp an SGR color component (`0..=255`) to a byte; the parser already bounds
/// parameters, so this only guards a malformed stream, never a valid one.
fn sgr_component(v: u16) -> u8 {
    v.min(255) as u8
}

/// Parse a `38`/`48` extended-color introducer at the front of `params`,
/// returning the color and how many *extra* parameters it consumed (0 if the
/// form is unrecognized). Handles `;5;n` (indexed) and `;2;r;g;b` (truecolor).
fn parse_ext_color(params: &[u16]) -> (Option<Color>, usize) {
    match params.get(1).copied() {
        Some(5) => {
            let n = params.get(2).copied().unwrap_or(0);
            (Some(Color::Indexed(sgr_component(n))), 2)
        }
        Some(2) => {
            let r = sgr_component(params.get(2).copied().unwrap_or(0));
            let g = sgr_component(params.get(3).copied().unwrap_or(0));
            let b = sgr_component(params.get(4).copied().unwrap_or(0));
            (Some(Color::Rgb(r, g, b)), 4)
        }
        _ => (None, 0),
    }
}

/// The charset an `ESC ( F` / `ESC ) F` designation selects. Only `0` (DEC
/// Special Graphics) differs from ASCII in what we support; `B` (ASCII) and any
/// other designation map to ASCII.
fn charset_from(byte: u8) -> Charset {
    match byte {
        b'0' => Charset::DecSpecialGraphics,
        _ => Charset::Ascii,
    }
}

/// The DEC Special Graphics glyph for a byte in `0x60..=0x7e` (the line-drawing
/// set); any other char passes through unchanged. This is the box-drawing
/// coverage `less`, `mc`, and framed TUIs depend on.
fn dec_special_graphics(c: char) -> char {
    match c {
        '`' => '◆',
        'a' => '▒',
        'b' => '␉',
        'c' => '␌',
        'd' => '␍',
        'e' => '␊',
        'f' => '°',
        'g' => '±',
        'h' => '␤',
        'i' => '␋',
        'j' => '┘',
        'k' => '┐',
        'l' => '┌',
        'm' => '└',
        'n' => '┼',
        'o' => '⎺',
        'p' => '⎻',
        'q' => '─',
        'r' => '⎼',
        's' => '⎽',
        't' => '├',
        'u' => '┤',
        'v' => '┴',
        'w' => '┬',
        'x' => '│',
        'y' => '≤',
        'z' => '≥',
        '{' => 'π',
        '|' => '≠',
        '}' => '£',
        '~' => '·',
        _ => c,
    }
}

/// A CSI numeric parameter, or 0 when absent.
fn csi_arg(params: &[u16], i: usize) -> u16 {
    params.get(i).copied().unwrap_or(0)
}

/// A CSI count parameter (for cursor moves, repeat counts): absent or 0 means 1.
fn csi_count(params: &[u16], i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1))
}

/// A 1-based CSI position parameter as a 0-based index (default 1 maps to 0).
fn csi_index(params: &[u16], i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1)) - 1
}

/// Append `n` as decimal ASCII (for building query responses), allocation-free.
fn push_decimal(out: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&tmp[i..]);
}

/// The VT interpretation: turn the parser's syntactic callbacks into the grid
/// operations above. This is the "grid interprets the actions" seam; the parser
/// (`vt.rs`) stays free of any of this meaning.
impl Perform for Screen {
    fn print(&mut self, c: char) {
        let mapped = self.map_glyph(c);
        // The inherent `Screen::print` (wide-char / wrap / combining) takes
        // priority over this trait method, so this is not a recursive call.
        self.print(mapped);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            0x08 => self.backspace(),        // BS
            0x09 => self.tab(),              // HT
            0x0a..=0x0c => self.line_feed(), // LF, VT, FF
            0x0d => self.carriage_return(),  // CR
            0x0e => self.gl_is_g1 = true,    // SO (shift out to G1)
            0x0f => self.gl_is_g1 = false,   // SI (shift in to G0)
            _ => {}                          // BEL, NUL, ...: nothing to draw
        }
    }

    fn csi_dispatch(&mut self, params: &[u16], intermediates: &[u8], private: u8, action: u8) {
        // DECSCUSR (`CSI Ps SP q`) is the one CSI-with-intermediate we act on; the
        // rest are dropped rather than misreading the final byte.
        if intermediates == [b' '] && action == b'q' {
            self.set_cursor_style(params.first().copied().unwrap_or(0));
            return;
        }
        if !intermediates.is_empty() {
            return;
        }
        match action {
            b'A' => self.move_up(csi_count(params, 0)),
            b'B' => self.move_down(csi_count(params, 0)),
            b'C' => self.move_forward(csi_count(params, 0)),
            b'D' => self.move_back(csi_count(params, 0)),
            b'E' => {
                self.move_down(csi_count(params, 0));
                self.carriage_return();
            }
            b'F' => {
                self.move_up(csi_count(params, 0));
                self.carriage_return();
            }
            b'G' | b'`' => self.move_to_col(csi_index(params, 0)),
            b'H' | b'f' => self.move_to(csi_index(params, 0), csi_index(params, 1)),
            b'd' => self.move_to_row(csi_index(params, 0)),
            b'I' => {
                for _ in 0..csi_count(params, 0) {
                    self.tab();
                }
            }
            b'J' => self.erase_display(csi_arg(params, 0)),
            b'K' => self.erase_line(csi_arg(params, 0)),
            b'L' => self.insert_lines(csi_count(params, 0)),
            b'M' => self.delete_lines(csi_count(params, 0)),
            b'@' => self.insert_chars(csi_count(params, 0)),
            b'P' => self.delete_chars(csi_count(params, 0)),
            b'X' => self.erase_chars(csi_count(params, 0)),
            b'S' => self.scroll_up(csi_count(params, 0)),
            b'T' => self.scroll_down(csi_count(params, 0)),
            b'r' => {
                let (_, rows) = self.dimensions();
                let top = csi_index(params, 0);
                let bottom = match csi_arg(params, 1) {
                    0 => rows - 1,
                    v => usize::from(v) - 1,
                };
                self.set_scroll_region(top, bottom);
            }
            b'm' if private == 0 => self.sgr(params),
            b'h' => {
                let dec = private == b'?';
                for &m in params {
                    self.set_mode(m, dec, true);
                }
            }
            b'l' => {
                let dec = private == b'?';
                for &m in params {
                    self.set_mode(m, dec, false);
                }
            }
            b's' if private == 0 && params.is_empty() => self.save_cursor(),
            b'u' if private == 0 && params.is_empty() => self.restore_cursor(),
            b'g' => self.clear_tab_stop(csi_arg(params, 0)),
            b'c' => self.device_attributes(private),
            b'n' => self.device_status(params, private),
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8) {
        match intermediates.first().copied() {
            None => match byte {
                b'c' => self.reset(),            // RIS
                b'D' => self.line_feed(),        // IND
                b'E' => self.next_line(),        // NEL
                b'M' => self.reverse_index(),    // RI
                b'7' => self.save_cursor(),      // DECSC
                b'8' => self.restore_cursor(),   // DECRC
                b'=' => self.keypad_app = true,  // DECKPAM
                b'>' => self.keypad_app = false, // DECKPNM
                _ => {}
            },
            Some(b'(') => self.g0 = charset_from(byte), // designate G0
            Some(b')') => self.g1 = charset_from(byte), // designate G1
            Some(b'#') if byte == b'8' => self.decaln(), // DECALN
            _ => {}
        }
    }

    fn osc_dispatch(&mut self, data: &[u8]) {
        // OSC Ps ; Pt — we handle 0 (icon + title) and 2 (title).
        let mut parts = data.splitn(2, |&b| b == b';');
        let ps = parts.next().unwrap_or(&[]);
        let pt = parts.next().unwrap_or(&[]);
        if ps == b"0" || ps == b"2" {
            self.set_title(String::from_utf8_lossy(pt).into_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_layout_is_pinned() {
        // A screenful plus scrollback is a lot of cells; growth is a decision,
        // not an accident. char(4) + Color(4) + Color(4) + Attrs(2), aligned to
        // char's 4 bytes, rounds to 16.
        assert_eq!(std::mem::size_of::<Attrs>(), 2);
        assert_eq!(std::mem::size_of::<Cell>(), 16);
        assert_eq!(std::mem::align_of::<Cell>(), 4);
    }

    #[test]
    fn blank_is_a_default_space() {
        assert_eq!(Cell::BLANK, Cell::default());
        assert_eq!(Cell::BLANK.rune, ' ');
        assert_eq!(Cell::BLANK.fg, Color::Default);
        assert!(Cell::BLANK.attrs.is_empty());
    }

    #[test]
    fn attrs_insert_remove_set_roundtrip() {
        let mut a = Attrs::empty();
        assert!(a.is_empty());
        a.insert(Attrs::BOLD);
        a.insert(Attrs::ITALIC);
        assert!(a.contains(Attrs::BOLD));
        assert!(a.contains(Attrs::ITALIC));
        assert!(!a.contains(Attrs::UNDERLINE));
        // contains(A|B) is all-of, not any-of.
        assert!(a.contains(Attrs::BOLD | Attrs::ITALIC));
        assert!(!a.contains(Attrs::BOLD | Attrs::UNDERLINE));
        a.remove(Attrs::BOLD);
        assert!(!a.contains(Attrs::BOLD));
        a.set(Attrs::ITALIC, false);
        assert!(a.is_empty());
    }

    #[test]
    fn every_flag_is_a_distinct_bit() {
        // No two flags share a bit, and none is zero (which would alias empty).
        let mut seen = 0u16;
        for (flag, _) in Attrs::ALL {
            assert_ne!(flag.0, 0);
            assert_eq!(seen & flag.0, 0, "overlapping attr bit");
            seen |= flag.0;
        }
    }

    #[test]
    fn attrs_debug_lists_flags() {
        assert_eq!(format!("{:?}", Attrs::empty()), "Attrs(-)");
        assert_eq!(format!("{:?}", Attrs::BOLD), "Attrs(BOLD)");
        // Listed in bit order regardless of insertion order.
        let a = Attrs::ITALIC | Attrs::BOLD;
        assert_eq!(format!("{a:?}"), "Attrs(BOLD|ITALIC)");
    }

    #[test]
    fn wide_halves_report_their_role() {
        let mut leader = Cell::new('世');
        leader.attrs.insert(Attrs::WIDE_LEADER);
        assert!(leader.is_wide_leader());
        assert!(!leader.is_wide_spacer());

        let mut spacer = Cell::BLANK;
        spacer.attrs.insert(Attrs::WIDE_SPACER);
        assert!(spacer.is_wide_spacer());
        assert!(!spacer.is_wide_leader());
    }

    // ---- Screen ------------------------------------------------------------

    fn print_str(s: &mut Screen, text: &str) {
        for c in text.chars() {
            s.print(c);
        }
    }

    #[test]
    fn print_advances_and_reads_back() {
        let mut s = Screen::new(10, 3);
        print_str(&mut s, "hi");
        assert_eq!(s.cursor(), (0, 2));
        assert_eq!(s.cell(0, 0).rune, 'h');
        assert_eq!(s.cell(0, 1).rune, 'i');
        assert_eq!(s.row_string(0).trim_end(), "hi");
    }

    #[test]
    fn pending_wrap_is_the_last_column_rule() {
        // Exactly `cols` characters must fill the row without wrapping; the wrap
        // is deferred until the next glyph arrives.
        let mut s = Screen::new(4, 3);
        print_str(&mut s, "abcd");
        // Cursor pinned to the last column, wrap pending, still on row 0.
        assert_eq!(s.cursor(), (0, 3));
        assert_eq!(s.row_string(0).trim_end(), "abcd");
        // The next glyph triggers the wrap and lands on row 1.
        s.print('e');
        assert_eq!(s.cursor(), (1, 1));
        assert_eq!(s.cell(1, 0).rune, 'e');
        // The wrapped row is flagged for reflow/selection.
        assert!(s.cell(0, 3).attrs.contains(Attrs::WRAPPED));
    }

    #[test]
    fn no_autowrap_pins_at_the_last_column() {
        let mut s = Screen::new(4, 3);
        s.set_mode(7, true, false); // DECAWM off
        print_str(&mut s, "abcdef");
        // Everything past the edge overwrites the last cell; no wrap.
        assert_eq!(s.cursor(), (0, 3));
        assert_eq!(s.cell(0, 3).rune, 'f');
        assert_eq!(s.row_string(1).trim_end(), "");
    }

    #[test]
    fn wide_char_takes_two_cells() {
        let mut s = Screen::new(10, 3);
        s.print('世');
        assert!(s.cell(0, 0).is_wide_leader());
        assert_eq!(s.cell(0, 0).rune, '世');
        assert!(s.cell(0, 1).is_wide_spacer());
        assert_eq!(s.cursor(), (0, 2));
        // row_string skips the spacer, so it reads as one glyph.
        assert_eq!(s.row_string(0).trim_end(), "世");
    }

    #[test]
    fn wide_char_at_right_edge_wraps_whole() {
        // Two columns wide, one column left: the glyph must move to the next row
        // rather than split across the edge.
        let mut s = Screen::new(3, 3);
        print_str(&mut s, "ab"); // cursor now at col 2 (last), pending wrap
        s.print('世');
        // The wide glyph lands on row 1, not straddling the edge of row 0.
        assert_eq!(s.cell(1, 0).rune, '世');
        assert!(s.cell(1, 0).is_wide_leader());
        assert!(s.cell(1, 1).is_wide_spacer());
        assert_eq!(s.cursor(), (1, 2));
    }

    #[test]
    fn overwriting_a_wide_half_cleans_its_partner() {
        let mut s = Screen::new(10, 3);
        s.print('世'); // leader at 0, spacer at 1
        s.move_to(0, 0);
        s.print('x'); // overwrite the leader
        assert_eq!(s.cell(0, 0).rune, 'x');
        assert!(!s.cell(0, 0).is_wide_leader());
        // The orphaned spacer is cleared to a blank.
        assert!(!s.cell(0, 1).is_wide_spacer());
        assert_eq!(s.cell(0, 1), Cell::BLANK);
    }

    #[test]
    fn combining_mark_attaches_to_base() {
        let mut s = Screen::new(10, 3);
        s.print('e');
        s.print('\u{0301}'); // combining acute accent
                             // The base cell is unchanged; the mark is in the side table.
        assert_eq!(s.cell(0, 0).rune, 'e');
        assert_eq!(s.cursor(), (0, 1)); // zero-width, cursor did not advance
        assert_eq!(s.marks_at(0, 0), Some(['\u{0301}'].as_slice()));
        assert_eq!(s.row_string(0).trim_end(), "e\u{0301}");
    }

    #[test]
    fn combining_mark_after_wide_char_hits_the_leader() {
        let mut s = Screen::new(10, 3);
        s.print('か'); // wide: leader at 0, spacer at 1, cursor at 2
        s.print('\u{3099}'); // combining voiced sound mark -> が
                             // The mark composes onto the leader (col 0), not the spacer (col 1).
        assert_eq!(s.marks_at(0, 0), Some(['\u{3099}'].as_slice()));
        assert_eq!(s.marks_at(0, 1), None);
    }

    #[test]
    fn carriage_return_and_line_feed() {
        let mut s = Screen::new(10, 3);
        print_str(&mut s, "ab");
        s.carriage_return();
        assert_eq!(s.cursor(), (0, 0));
        s.line_feed();
        assert_eq!(s.cursor(), (1, 0));
    }

    #[test]
    fn line_feed_at_bottom_scrolls_into_scrollback() {
        let mut s = Screen::new(4, 2);
        print_str(&mut s, "aa");
        s.line_feed();
        s.carriage_return();
        print_str(&mut s, "bb");
        s.line_feed(); // at bottom: scroll up
        s.carriage_return();
        print_str(&mut s, "cc");
        // "aa" scrolled off, "bb"/"cc" are the two visible rows.
        assert_eq!(s.row_string(0).trim_end(), "bb");
        assert_eq!(s.row_string(1).trim_end(), "cc");
    }

    #[test]
    fn cursor_moves_clamp_to_bounds() {
        let mut s = Screen::new(5, 4);
        s.move_to(2, 3);
        assert_eq!(s.cursor(), (2, 3));
        s.move_up(10);
        assert_eq!(s.cursor(), (0, 3));
        s.move_back(10);
        assert_eq!(s.cursor(), (0, 0));
        s.move_down(10);
        assert_eq!(s.cursor(), (3, 0));
        s.move_forward(10);
        assert_eq!(s.cursor(), (3, 4));
        // Out-of-range CUP clamps rather than trapping.
        s.move_to(99, 99);
        assert_eq!(s.cursor(), (3, 4));
    }

    #[test]
    fn origin_mode_confines_addressing_to_the_region() {
        let mut s = Screen::new(10, 10);
        s.set_scroll_region(2, 5);
        s.set_mode(6, true, true); // DECOM on; also homes cursor to region top
        assert_eq!(s.cursor(), (2, 0));
        s.move_to(0, 0); // row 0 is relative to the region top (row 2)
        assert_eq!(s.cursor(), (2, 0));
        s.move_to(99, 0); // clamps to the region bottom, not the screen bottom
        assert_eq!(s.cursor(), (5, 0));
    }

    #[test]
    fn erase_line_modes() {
        let mut s = Screen::new(6, 2);
        print_str(&mut s, "abcdef");
        s.move_to(0, 3);
        s.erase_line(0); // cursor to end
        assert_eq!(s.row_string(0).trim_end(), "abc");
        print_str(&mut s, "");
        s.move_to(0, 0);
        print_str(&mut s, "abcdef");
        s.move_to(0, 3);
        s.erase_line(1); // start to cursor inclusive
        assert_eq!(s.cell(0, 0).rune, ' ');
        assert_eq!(s.cell(0, 3).rune, ' ');
        assert_eq!(s.cell(0, 4).rune, 'e');
        s.erase_line(2);
        assert_eq!(s.row_string(0).trim_end(), "");
    }

    #[test]
    fn erase_display_below_and_all() {
        let mut s = Screen::new(4, 3);
        print_str(&mut s, "aa");
        s.line_feed();
        s.carriage_return();
        print_str(&mut s, "bb");
        s.line_feed();
        s.carriage_return();
        print_str(&mut s, "cc");
        s.move_to(1, 0);
        s.erase_display(0); // cursor to end of screen
        assert_eq!(s.row_string(0).trim_end(), "aa");
        assert_eq!(s.row_string(1).trim_end(), "");
        assert_eq!(s.row_string(2).trim_end(), "");
        s.erase_display(2);
        assert_eq!(s.dump().trim(), "");
    }

    #[test]
    fn insert_and_delete_chars() {
        let mut s = Screen::new(6, 1);
        print_str(&mut s, "abcdef");
        s.move_to(0, 1);
        s.delete_chars(2); // remove "bc"
        assert_eq!(s.row_string(0).trim_end(), "adef");
        s.move_to(0, 1);
        s.insert_chars(2); // open two blanks after 'a'
        assert_eq!(s.cell(0, 1).rune, ' ');
        assert_eq!(s.cell(0, 2).rune, ' ');
        assert_eq!(s.cell(0, 3).rune, 'd');
    }

    #[test]
    fn erase_chars_does_not_shift() {
        let mut s = Screen::new(6, 1);
        print_str(&mut s, "abcdef");
        s.move_to(0, 2);
        s.erase_chars(2);
        assert_eq!(s.cell(0, 2).rune, ' ');
        assert_eq!(s.cell(0, 3).rune, ' ');
        assert_eq!(s.cell(0, 4).rune, 'e');
    }

    #[test]
    fn insert_delete_lines_respect_scroll_region() {
        // The classic scroll-region interaction: IL/DL scroll only within
        // [top, bottom], leaving lines outside the region alone.
        let mut s = Screen::new(3, 5);
        for (r, text) in ["00", "11", "22", "33", "44"].iter().enumerate() {
            s.move_to(r, 0);
            print_str(&mut s, text);
        }
        s.set_scroll_region(1, 3); // rows 1..=3
        s.move_to(1, 0);
        s.insert_lines(1); // push rows 1,2 down to 2,3; row 4 (outside) untouched
        assert_eq!(s.row_string(0).trim_end(), "00");
        assert_eq!(s.row_string(1).trim_end(), ""); // new blank line
        assert_eq!(s.row_string(2).trim_end(), "11");
        assert_eq!(s.row_string(3).trim_end(), "22");
        assert_eq!(s.row_string(4).trim_end(), "44"); // below the region, intact
        s.move_to(1, 0);
        s.delete_lines(1); // pull rows 2,3 back up
        assert_eq!(s.row_string(1).trim_end(), "11");
        assert_eq!(s.row_string(2).trim_end(), "22");
        assert_eq!(s.row_string(4).trim_end(), "44");
    }

    #[test]
    fn scroll_up_down_within_region() {
        let mut s = Screen::new(3, 4);
        for (r, text) in ["00", "11", "22", "33"].iter().enumerate() {
            s.move_to(r, 0);
            print_str(&mut s, text);
        }
        s.scroll_up(1);
        assert_eq!(s.row_string(0).trim_end(), "11");
        assert_eq!(s.row_string(3).trim_end(), "");
        s.scroll_down(1);
        assert_eq!(s.row_string(0).trim_end(), "");
        assert_eq!(s.row_string(1).trim_end(), "11");
    }

    #[test]
    fn reverse_index_scrolls_at_top() {
        let mut s = Screen::new(3, 3);
        s.move_to(0, 0);
        print_str(&mut s, "aa");
        s.move_to(1, 0);
        print_str(&mut s, "bb");
        s.move_to(0, 0);
        s.reverse_index(); // at the top margin: scroll down
        assert_eq!(s.row_string(0).trim_end(), ""); // new blank line at top
        assert_eq!(s.row_string(1).trim_end(), "aa");
        assert_eq!(s.row_string(2).trim_end(), "bb");
    }

    #[test]
    fn tabs_land_on_stops() {
        let mut s = Screen::new(20, 1);
        s.tab();
        assert_eq!(s.cursor(), (0, 8));
        s.tab();
        assert_eq!(s.cursor(), (0, 16));
        s.clear_tab_stop(3); // clear all
        s.move_to(0, 0);
        s.tab();
        assert_eq!(s.cursor(), (0, 19)); // no stops: go to last column
    }

    #[test]
    fn sgr_sets_the_pen_and_print_applies_it() {
        let mut s = Screen::new(10, 1);
        s.sgr(&[1, 31]); // bold, red foreground
        s.print('x');
        let c = s.cell(0, 0);
        assert!(c.attrs.contains(Attrs::BOLD));
        assert_eq!(c.fg, Color::Ansi(1));
        s.sgr(&[0]); // reset
        s.print('y');
        let c = s.cell(0, 1);
        assert!(c.attrs.is_empty());
        assert_eq!(c.fg, Color::Default);
    }

    #[test]
    fn sgr_extended_colors() {
        let mut s = Screen::new(10, 1);
        s.sgr(&[38, 5, 200]); // 256-color foreground
        assert_eq!(s.pen.fg, Color::Indexed(200));
        s.sgr(&[48, 2, 10, 20, 30]); // truecolor background
        assert_eq!(s.pen.bg, Color::Rgb(10, 20, 30));
        s.sgr(&[90]); // bright black foreground -> ANSI 8
        assert_eq!(s.pen.fg, Color::Ansi(8));
    }

    #[test]
    fn alt_screen_is_separate_and_restores() {
        let mut s = Screen::new(6, 2);
        print_str(&mut s, "main");
        s.set_mode(1049, true, true); // enter alt, save cursor
        assert!(s.is_alt());
        assert_eq!(s.dump().trim(), ""); // alt starts cleared
        print_str(&mut s, "alt");
        assert_eq!(s.row_string(0).trim_end(), "alt");
        s.set_mode(1049, true, false); // leave alt, restore cursor
        assert!(!s.is_alt());
        assert_eq!(s.row_string(0).trim_end(), "main"); // primary intact
        assert_eq!(s.cursor(), (0, 4)); // cursor restored to after "main"
    }

    #[test]
    fn save_and_restore_cursor() {
        let mut s = Screen::new(10, 5);
        s.move_to(2, 4);
        s.sgr(&[1]);
        s.save_cursor();
        s.move_to(0, 0);
        s.sgr(&[0]);
        s.restore_cursor();
        assert_eq!(s.cursor(), (2, 4));
        assert!(s.pen.attrs.contains(Attrs::BOLD));
    }

    #[test]
    fn reset_returns_to_power_on_state() {
        let mut s = Screen::new(6, 3);
        s.sgr(&[31]);
        print_str(&mut s, "junk");
        s.set_scroll_region(1, 2);
        s.reset();
        assert_eq!(s.dump().trim(), "");
        assert_eq!(s.cursor(), (0, 0));
        assert_eq!(s.pen.fg, Color::Default);
        assert_eq!(s.dimensions(), (6, 3));
    }

    #[test]
    fn steady_scroll_is_allocation_free_after_warmup() {
        // Once scrollback is full, scrolling must recycle row storage. We can at
        // least assert the ring stays bounded and correct over many scrolls.
        let mut s = Screen::new(4, 3);
        for i in 0..50_000u32 {
            s.carriage_return();
            let d = char::from(b'0' + (i % 10) as u8);
            s.print(d);
            s.line_feed();
        }
        // Never grows past the visible height; the bottom holds the last digit.
        assert_eq!(s.dimensions(), (4, 3));
        assert_eq!(s.primary.lines.len(), 3);
        assert!(s.primary.scrollback.len() <= DEFAULT_SCROLLBACK);
    }

    // ---- end-to-end: bytes through the parser into the grid ----------------

    fn feed(s: &mut Screen, bytes: &[u8]) {
        let mut p = crate::vt::Parser::new();
        p.advance_bytes(s, bytes);
    }

    #[test]
    fn sgr_colors_through_the_parser() {
        let mut s = Screen::new(10, 2);
        feed(&mut s, b"\x1b[1;31mhi\x1b[0mok");
        assert_eq!(s.cell(0, 0).rune, 'h');
        assert!(s.cell(0, 0).attrs.contains(Attrs::BOLD));
        assert_eq!(s.cell(0, 0).fg, Color::Ansi(1));
        // After the reset, 'o' carries the default rendition again.
        assert_eq!(s.cell(0, 2).rune, 'o');
        assert!(s.cell(0, 2).attrs.is_empty());
        assert_eq!(s.cell(0, 2).fg, Color::Default);
    }

    #[test]
    fn cursor_addressing_and_erase_through_the_parser() {
        let mut s = Screen::new(10, 3);
        feed(&mut s, b"line1\r\nline2");
        feed(&mut s, b"\x1b[H"); // home
        assert_eq!(s.cursor(), (0, 0));
        feed(&mut s, b"\x1b[2;3Hx"); // row 2, col 3 (1-based)
        assert_eq!(s.cell(1, 2).rune, 'x');
        feed(&mut s, b"\x1b[2J"); // erase whole display
        assert_eq!(s.dump().trim(), "");
    }

    #[test]
    fn wrap_via_parser_flags_the_row() {
        let mut s = Screen::new(3, 3);
        feed(&mut s, b"abcde");
        assert_eq!(s.row_string(0).trim_end(), "abc");
        assert_eq!(s.row_string(1).trim_end(), "de");
        assert!(s.cell(0, 2).attrs.contains(Attrs::WRAPPED));
    }

    #[test]
    fn box_drawing_charset() {
        let mut s = Screen::new(10, 1);
        // ESC ( 0 designates G0 = DEC Special Graphics; lqk -> ┌─┐; ESC ( B back.
        feed(&mut s, b"\x1b(0lqk\x1b(Bx");
        assert_eq!(s.cell(0, 0).rune, '┌');
        assert_eq!(s.cell(0, 1).rune, '─');
        assert_eq!(s.cell(0, 2).rune, '┐');
        assert_eq!(s.cell(0, 3).rune, 'x');
    }

    #[test]
    fn shift_out_in_switches_charset() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"\x1b)0"); // designate G1 = special graphics
        feed(&mut s, b"a\x0eq\x0fb"); // 'a', SO, 'q'->─, SI, 'b'
        assert_eq!(s.cell(0, 0).rune, 'a');
        assert_eq!(s.cell(0, 1).rune, '─');
        assert_eq!(s.cell(0, 2).rune, 'b');
    }

    #[test]
    fn alt_screen_via_parser() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"main");
        feed(&mut s, b"\x1b[?1049h");
        assert!(s.is_alt());
        feed(&mut s, b"alt");
        assert_eq!(s.row_string(0).trim_end(), "alt");
        feed(&mut s, b"\x1b[?1049l");
        assert_eq!(s.row_string(0).trim_end(), "main");
    }

    #[test]
    fn osc_sets_title_through_the_parser() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"\x1b]0;my title\x07");
        assert_eq!(s.title(), "my title");
        feed(&mut s, b"\x1b]2;other\x1b\\");
        assert_eq!(s.title(), "other");
    }

    #[test]
    fn scroll_region_index_via_parser() {
        let mut s = Screen::new(4, 4);
        feed(&mut s, b"\x1b[2;3r"); // region rows 2..=3 (1-based) -> index 1..=2
        feed(&mut s, b"\x1b[2;1HAA");
        feed(&mut s, b"\x1b[3;1HBB");
        feed(&mut s, b"\x1b[3;1H\n"); // LF at the region bottom scrolls the region
        assert_eq!(s.row_string(1).trim_end(), "BB");
        assert_eq!(s.row_string(2).trim_end(), "");
    }

    #[test]
    fn decaln_fills_with_e() {
        let mut s = Screen::new(3, 2);
        feed(&mut s, b"\x1b#8");
        assert_eq!(s.row_string(0), "EEE");
        assert_eq!(s.row_string(1), "EEE");
        assert_eq!(s.cursor(), (0, 0));
    }

    #[test]
    fn wide_and_combining_via_parser() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, "aか\u{3099}b".as_bytes()); // 'a', wide か + combining mark, 'b'
        assert_eq!(s.cell(0, 0).rune, 'a');
        assert_eq!(s.cell(0, 1).rune, 'か');
        assert!(s.cell(0, 1).is_wide_leader());
        assert_eq!(s.marks_at(0, 1), Some(['\u{3099}'].as_slice()));
        assert_eq!(s.cell(0, 3).rune, 'b');
    }

    #[test]
    fn random_bytes_into_screen_keep_invariants() {
        // The whole pipeline under fuzz: ~1.2 MB of arbitrary bytes through the
        // parser into a real Screen must never panic and must keep the grid's
        // structural invariants (a seeded LCG, no crate).
        let mut s = Screen::new(24, 8);
        let mut p = crate::vt::Parser::new();
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut buf = [0u8; 4096];
        for _ in 0..300 {
            for b in buf.iter_mut() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (seed >> 33) as u8;
            }
            p.advance_bytes(&mut s, &buf);
            let (cols, rows) = s.dimensions();
            assert_eq!((cols, rows), (24, 8));
            let (cr, cc) = s.cursor();
            assert!(cr < rows && cc < cols, "cursor {cr},{cc} out of bounds");
            assert_eq!(s.primary.lines.len(), rows);
            assert_eq!(s.alt.lines.len(), rows);
            assert!(s.primary.scrollback.len() <= DEFAULT_SCROLLBACK);
        }
    }

    #[test]
    fn resize_keeps_the_line_count_and_content() {
        let mut s = Screen::new(10, 3);
        feed(&mut s, b"abc\r\ndef\r\nghi");
        s.resize(6, 5);
        assert_eq!(s.dimensions(), (6, 5));
        // Existing rows keep their text; growth adds blank rows at the bottom.
        assert_eq!(s.row_string(0).trim_end(), "abc");
        assert_eq!(s.row_string(2).trim_end(), "ghi");
        assert_eq!(s.row_string(4).trim_end(), "");
        // Both screens track the new size.
        assert_eq!(s.primary.lines.len(), 5);
        assert_eq!(s.alt.lines.len(), 5);
    }

    #[test]
    fn narrowing_truncates_and_widening_pads() {
        let mut s = Screen::new(8, 1);
        feed(&mut s, b"abcdefgh");
        s.resize(4, 1);
        assert_eq!(s.row_string(0).trim_end(), "abcd");
        s.resize(8, 1);
        // Widening pads with blanks; the truncated tail is gone, not restored.
        assert_eq!(s.cell(0, 4).rune, ' ');
        assert_eq!(s.dimensions(), (8, 1));
    }

    #[test]
    fn shrinking_rows_scrolls_the_cursor_line_into_view() {
        // Cursor on the last row: shrinking must scroll the top away (into
        // scrollback), not drop the cursor's line off the bottom.
        let mut s = Screen::new(6, 4);
        feed(&mut s, b"r0\r\nr1\r\nr2\r\nr3");
        let (cr, _) = s.cursor();
        assert_eq!(cr, 3);
        s.resize(6, 2);
        assert_eq!(s.dimensions(), (6, 2));
        let (cr, _) = s.cursor();
        assert!(cr < 2, "cursor stayed on screen");
        // The last-written line is still visible; the top scrolled into history.
        assert_eq!(s.row_string(1).trim_end(), "r3");
        assert!(!s.primary.scrollback.is_empty());
    }

    #[test]
    fn resize_clamps_the_cursor_and_resets_the_scroll_region() {
        let mut s = Screen::new(10, 10);
        feed(&mut s, b"\x1b[3;8r"); // DECSTBM: region rows 3..8
        feed(&mut s, b"\x1b[9;9H"); // cursor near the old bottom-right
        s.resize(5, 5);
        let (cr, cc) = s.cursor();
        assert!(
            cr < 5 && cc < 5,
            "cursor {cr},{cc} clamped into the new grid"
        );
        // A resize resets DECSTBM to the full screen (xterm behavior): a full-
        // screen scroll now moves all five rows.
        assert_eq!((s.primary.scroll_top, s.primary.scroll_bottom), (0, 4));
    }

    #[test]
    fn scrolling_up_reveals_scrollback_rows() {
        // Four lines on a two-row screen: "a","b" scroll into history, "c","d"
        // stay live. The view walks up through the combined history-plus-screen.
        let mut s = Screen::new(5, 2);
        feed(&mut s, b"a\r\nb\r\nc\r\nd");
        assert_eq!(s.scrollback_len(), 2);
        // Pinned to the bottom: the live rows.
        assert_eq!(s.view_cell(0, 0).rune, 'c');
        assert_eq!(s.view_cell(1, 0).rune, 'd');
        // One line up: "b" over "c".
        s.scroll_view_up(1);
        assert!(s.is_scrolled());
        assert_eq!(s.view_cell(0, 0).rune, 'b');
        assert_eq!(s.view_cell(1, 0).rune, 'c');
        // Two lines up: the very top, "a" over "b".
        s.scroll_view_up(1);
        assert_eq!(s.view_cell(0, 0).rune, 'a');
        assert_eq!(s.view_cell(1, 0).rune, 'b');
        // Back to the bottom.
        s.scroll_view_to_bottom();
        assert!(!s.is_scrolled());
        assert_eq!(s.view_cell(0, 0).rune, 'c');
    }

    #[test]
    fn scroll_is_clamped_to_the_history() {
        let mut s = Screen::new(5, 2);
        feed(&mut s, b"a\r\nb\r\nc\r\nd"); // two lines of scrollback
        s.scroll_view_up(999);
        assert_eq!(s.view_offset(), 2, "cannot scroll above the oldest line");
        s.scroll_view_down(999);
        assert_eq!(s.view_offset(), 0, "cannot scroll below the live bottom");
        s.scroll_view_to_top();
        assert_eq!(s.view_offset(), 2);
    }

    #[test]
    fn alt_screen_has_no_scrollback_view() {
        let mut s = Screen::new(5, 2);
        feed(&mut s, b"a\r\nb\r\nc\r\nd");
        s.scroll_view_up(2);
        feed(&mut s, b"\x1b[?1049h"); // enter the alt screen
        assert_eq!(s.view_offset(), 0, "the alt screen pins to the bottom");
        s.scroll_view_up(5);
        assert!(!s.is_scrolled(), "scrolling is inert on the alt screen");
    }

    #[test]
    fn leaving_the_alt_screen_resets_the_view() {
        let mut s = Screen::new(5, 2);
        feed(&mut s, b"a\r\nb\r\nc\r\nd");
        s.scroll_view_up(2);
        feed(&mut s, b"\x1b[?1049h\x1b[?1049l"); // enter then leave the alt screen
        assert_eq!(s.view_offset(), 0, "back to a live view, not stale history");
    }

    #[test]
    fn resize_pins_the_view_to_the_bottom() {
        let mut s = Screen::new(5, 3);
        feed(&mut s, b"a\r\nb\r\nc\r\nd\r\ne");
        s.scroll_view_up(2);
        assert!(s.is_scrolled());
        s.resize(8, 4);
        assert_eq!(s.view_offset(), 0, "a resize reindexes history and re-pins");
    }

    #[test]
    fn selection_text_extracts_and_joins_rows() {
        let mut s = Screen::new(10, 3);
        feed(&mut s, b"hello\r\nworld\r\nfoo");
        // A single-row partial selection: cols 0..=4 of row 0.
        assert_eq!(s.selection_text((0, 0), (0, 4)), "hello");
        // Across rows: end of row 0 through part of row 1, joined by a newline;
        // trailing blanks on the first row are trimmed.
        assert_eq!(s.selection_text((0, 0), (1, 2)), "hello\nwor");
        // A mid-row start.
        assert_eq!(s.selection_text((0, 2), (0, 4)), "llo");
    }

    #[test]
    fn selection_joins_a_soft_wrapped_line() {
        // A line longer than the width soft-wraps; selecting across the wrap must
        // copy it as one unbroken logical line (no newline at the wrap point).
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
        assert!(s.cell(0, 3).attrs.contains(Attrs::WRAPPED));
        assert_eq!(s.selection_text((0, 0), (1, 1)), "abcdef");
    }

    #[test]
    fn selection_reads_scrollback_when_scrolled() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        s.scroll_view_up(2); // top: "one" over "two"
        assert_eq!(s.selection_text((0, 0), (0, 2)), "one");
    }

    #[test]
    fn device_attributes_and_status_queries_get_answers() {
        let mut s = Screen::new(80, 24);
        // Primary DA.
        feed(&mut s, b"\x1b[c");
        assert_eq!(s.take_responses(), b"\x1b[?1;2c");
        assert!(s.take_responses().is_empty(), "the queue drains");
        // Secondary DA.
        feed(&mut s, b"\x1b[>c");
        assert_eq!(s.take_responses(), b"\x1b[>0;0;0c");
        // DSR status: OK.
        feed(&mut s, b"\x1b[5n");
        assert_eq!(s.take_responses(), b"\x1b[0n");
    }

    #[test]
    fn cursor_position_report_is_one_based() {
        let mut s = Screen::new(80, 24);
        feed(&mut s, b"\x1b[3;10H"); // move to row 3, col 10 (1-based)
        feed(&mut s, b"\x1b[6n"); // request CPR
        assert_eq!(s.take_responses(), b"\x1b[3;10R");
        // The extended (?6n) form carries a trailing page number.
        feed(&mut s, b"\x1b[?6n");
        assert_eq!(s.take_responses(), b"\x1b[?3;10;1R");
    }

    #[test]
    fn decscusr_sets_the_cursor_shape_and_blink() {
        let mut s = Screen::new(10, 2);
        // Power-on default is a steady block (see `cursor_blink`).
        assert_eq!(s.cursor_style(), CursorStyle::Block);
        assert!(!s.cursor_blinks());
        feed(&mut s, b"\x1b[4 q"); // steady underline
        assert_eq!(s.cursor_style(), CursorStyle::Underline);
        assert!(!s.cursor_blinks());
        feed(&mut s, b"\x1b[5 q"); // blinking bar
        assert_eq!(s.cursor_style(), CursorStyle::Bar);
        assert!(s.cursor_blinks());
        feed(&mut s, b"\x1b[0 q"); // DECSCUSR 0: a blinking block (its own default)
        assert_eq!(s.cursor_style(), CursorStyle::Block);
        assert!(s.cursor_blinks());
    }

    #[test]
    fn mouse_modes_track_the_dec_private_toggles() {
        use crate::mouse::MouseProtocol;
        let mut s = Screen::new(10, 4);
        assert!(!s.mouse_mode().reports(), "off by default");
        feed(&mut s, b"\x1b[?1000h");
        assert_eq!(s.mouse_mode().protocol, MouseProtocol::Press);
        feed(&mut s, b"\x1b[?1002h"); // a higher level supersedes
        assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
        feed(&mut s, b"\x1b[?1006h"); // SGR is an orthogonal encoding flag
        assert!(s.mouse_mode().sgr);
        assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
        feed(&mut s, b"\x1b[?1002l"); // disabling turns reporting off
        assert!(!s.mouse_mode().reports());
        assert!(s.mouse_mode().sgr, "the encoding flag is independent");
    }

    #[test]
    fn narrowing_blanks_a_dangling_wide_leader() {
        // A wide glyph whose spacer is cut off must not leave a half-cell behind.
        let mut s = Screen::new(4, 1);
        feed(&mut s, "aa\u{6f22}".as_bytes()); // 'a','a', then 漢 as leader+spacer at cols 2,3
        assert!(s.cell(0, 2).is_wide_leader());
        s.resize(3, 1); // drops the spacer at col 3, leaving the leader at col 2
        assert!(
            !s.cell(0, 2).is_wide_leader(),
            "the orphaned leader was blanked"
        );
        assert_eq!(s.cell(0, 2).rune, ' ');
    }
}
