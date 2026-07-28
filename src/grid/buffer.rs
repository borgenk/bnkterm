//! The ring: what a screen is *made of*, and the mechanics of moving rows through it.
//!
//! A [`Buffer`] is one screen's worth of storage — the visible rows, the history behind
//! them, the cursor, the DECSTBM margins, the tab stops — plus every operation that moves
//! or rewrites rows without knowing what any escape sequence means. The semantics live on
//! [`Screen`] (see the sibling modules); the invariants of the storage live here.
//!
//! ```text
//!     scrollback (VecDeque<Row>)   visible (VecDeque<Row>, len == rows)
//!     ┌───┬───┬───┐               ┌───┬───┬───┬───┐
//!     │ … │ … │old│  ◀── scroll ──│row│row│row│row│
//!     └───┴───┴───┘               └───┴───┴───┴───┘
//! ```
//!
//! A scroll rotates *row headers* (a pointer move), never the cells: a full-screen scroll
//! is a `pop_front` + `push_back` (O(1)); a region scroll moves only the headers inside
//! the region. The evicted row's cell buffer is recycled into the new blank row, so a
//! steady scroll allocates nothing.
//!
//! Fields here are `pub(super)`, which means *visible to the rest of `grid`, private
//! outside it*. `Screen`'s operations are split across sibling modules and all of them
//! work directly on this representation, so the grid shares it internally and exposes
//! none of it: `grid` itself is the encapsulation boundary, not this file.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::*;

/// Columns between default tab stops.
pub(super) const TAB_WIDTH: usize = 8;

/// The widest a single logical line may grow before reflow treats the boundary as a
/// hard break. It bounds both the intermediate buffer and the per-line rewrap cost
/// against a child that prints megabytes with no newline: a line longer than this is
/// rewrapped in independent blocks that never re-join across the seam. wezterm uses the
/// same 1024; the visible cost is a seam every 1024 columns on such a line, invisible on
/// any real output. See [`Buffer::reflow`].
pub(super) const MAX_LOGICAL_COLS: usize = 1024;

/// How many columns the tab-stop table covers, regardless of how wide the screen
/// currently is.
///
/// The table is deliberately *not* geometry, and sizing it to the width was a bug: `TBC
/// 3` ("clear every stop") can only clear the columns the table has, so a later widening
/// would seed fresh defaults into columns the program had explicitly emptied, and the
/// stops it deleted would come back. Covering a fixed span from the start means "every"
/// means every, whatever the window does afterwards. xterm's array is width-independent
/// (1024 columns) for the same reason.
pub(super) const MAX_TAB_COLUMNS: usize = 1024;

/// The most combining marks a single cell retains. Unicode's Stream-Safe Text
/// format caps a grapheme at 30, so this sits just above it: real accented and
/// stacked text is never clipped, but an unbounded Zalgo stream cannot grow one
/// cell's side table (or the render string it feeds) without limit.
pub(super) const MAX_COMBINING_PER_CELL: usize = 32;

/// What a scroll did to the rows it moved, which is all the rest of the terminal needs to
/// know about it.
///
/// - `pushed`: rows that *retired into history* off the top of the screen. A scrolled-back
///   viewport must follow them up ([`Screen::follow_history`]).
/// - `renumbered`: the ids stopped naming the same lines, so the [`RowEpoch`] must end.
///
/// The two are independent, and the reason is worth stating once. Rows live in one stream,
/// `scrollback ++ lines`, and an [`AbsRow`] is a position in it. A row retiring into
/// history does not move within that stream — it goes from the front of `lines` to the
/// back of `scrollback`, which is the very same slot — so a *full-screen* scroll keeps
/// every id: it only appends a blank at the far end.
///
/// ```text
///   stream:  [ h0 h1 | r0 r1 r2 ]        r0 retires, blank appends at the end
///            [ h0 h1   r0 | r1 r2 __ ]   every surviving row kept its slot
/// ```
///
/// Anything else that moves rows *inside* the stream renumbers it, and there are two such
/// moves. A row can be torn out of the middle — a scroll inside a `DECSTBM` region, a
/// `DL`, any scroll on the historyless alt screen, or any scroll *down* — and everything
/// past the hole shifts. Or a top-anchored region can stop short of the last row, in which
/// case the blank that fills in behind the scroll is inserted mid-stream and shifts every
/// row *below the region*, even though nothing about those rows moved on screen:
///
/// ```text
///   region 0..=1 of a 4-row screen, r0 retiring into history
///   stream:  [ h0 | r0 r1 r2 r3 ]
///            [ h0   r0 | r1 __ r2 r3 ]   ← the blank landed mid-stream:
///                            ↑↑  r2 and r3 each shifted up an id, sitting still
/// ```
///
/// That last case is the subtle one: it pushes to history *and* renumbers, so a program
/// holding a status line below a scroll region would silently drag a selection onto the
/// blank if only `pushed` were reported.
#[derive(Clone, Copy, Default)]
pub(super) struct Scrolled {
    pub(super) pushed: usize,
    pub(super) renumbered: bool,
}

/// The cursor: a position plus the deferred-wrap flag that implements xterm's
/// last-column rule. `pending_wrap` is set after a glyph lands in the final
/// column; the wrap to the next line is delayed until the next glyph actually
/// arrives, so writing exactly `cols` characters does not scroll.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Cursor {
    pub(super) row: usize,
    pub(super) col: usize,
    pub(super) pending_wrap: bool,
}

/// The current graphic rendition (SGR state): the colors and attributes printed
/// cells receive, plus the OSC 8 hyperlink they fall inside. `Copy` so a
/// save/restore (DECSC/DECRC) is a plain move.
///
/// The link rides on the pen because that is exactly what it is: a mode the child
/// turns on, prints under, and turns off, no different from bold. It is *not* an SGR
/// attribute, though, so `SGR 0` (reset) must leave it alone — closing a hyperlink is
/// `OSC 8 ; ; ST` and nothing else, and a program that resets colors mid-anchor (as
/// any colored `ls` listing does) still expects the anchor to hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Pen {
    pub(super) fg: Color,
    pub(super) bg: Color,
    pub(super) attrs: Attrs,
    pub(super) link: LinkId,
}

impl Pen {
    /// SGR 0: default colors, no attributes, and the hyperlink left exactly as it was.
    /// See the type's header for why the link survives a rendition reset — a colored
    /// `ls --hyperlink` listing emits `SGR 0` between entries *inside* an open anchor,
    /// so clearing the link here would break the most common OSC 8 producer there is.
    pub(super) fn reset_rendition(&mut self) {
        // Destructured rather than `..Pen::default()` so that adding a field to `Pen`
        // is a compile error here: whoever adds it has to say whether SGR 0 clears it.
        let Pen {
            fg,
            bg,
            attrs,
            link: _,
        } = Pen::default();
        self.fg = fg;
        self.bg = bg;
        self.attrs = attrs;
    }
}

impl Default for Pen {
    fn default() -> Self {
        Pen {
            fg: Color::Default,
            bg: Color::Default,
            attrs: Attrs::empty(),
            link: LinkId::NONE,
        }
    }
}

/// A saved cursor (DECSC): position, pen, and origin mode, restored by DECRC.
#[derive(Clone, Copy)]
pub(super) struct Saved {
    pub(super) cursor: Cursor,
    pub(super) pen: Pen,
    pub(super) origin: bool,
    /// The character set shift state: the G0 and G1 designations and which of them GL
    /// is currently mapped to. DECSC saves it and DECRC restores it, which is not
    /// optional decoration — see [`Screen::save_cursor`].
    pub(super) charsets: Charsets,
}

/// The character set shift state: what G0 and G1 are designated as, and which one GL
/// reads through (SI selects G0, SO selects G1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Charsets {
    pub(super) g0: Charset,
    pub(super) g1: Charset,
    pub(super) gl_is_g1: bool,
}

/// One combining mark riding along with the base cell in `col`, in arrival order.
/// Flattened into the row's [`Row::combining`] list rather than owned per marked
/// cell, so the whole row's marks live in one reusable heap buffer (see the field
/// docs for why that matters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CombiningMark {
    pub(super) col: usize,
    pub(super) ch: char,
}

/// One row of cells plus its rare combining marks. Cells are contiguous, so a
/// row is cheap to scan and write. Combining marks live in a tiny side list keyed
/// by column so they ride along when the row scrolls and cost nothing on the
/// common all-single-codepoint row.
#[derive(Clone, Debug)]
pub(super) struct Row {
    pub(super) cells: Vec<Cell>,
    /// The row's combining marks, flattened: one `(col, ch)` pair per mark rather
    /// than an owned `Vec<char>` per marked cell. Flattening keeps the side table a
    /// single heap buffer that [`Row::reset`] clears in place, so a recycled scroll
    /// row reuses its capacity and a warmed combining/emoji line allocates nothing.
    /// A column's marks read back by filtering on `col`, which preserves the order
    /// they arrived in.
    pub(super) combining: Vec<CombiningMark>,
    /// Autowrap carried this line onto the next row: the two are one logical line,
    /// so a selection spanning them copies as unbroken text and a triple-click takes
    /// both. A hard newline leaves this false.
    ///
    /// It lives on the row rather than on the row's last cell because it describes the
    /// *line*, and a cell is addressed by a column that a resize moves: widening would
    /// strand the flag mid-row and narrowing would truncate it away, and either one
    /// silently breaks a wrapped line in half on the next copy. We do not re-wrap on
    /// resize ([`Buffer::resize`]), so nothing would put it back.
    ///
    /// The invariant: true only while the text that wrapped is still the text in the
    /// final column. Every write or erase reaching that column clears it.
    pub(super) wrapped: bool,
}

impl Row {
    pub(super) fn filled(cols: usize, cell: Cell) -> Self {
        Row {
            cells: vec![cell; cols],
            combining: Vec::new(),
            wrapped: false,
        }
    }

    /// Reset every cell to `blank` and drop combining marks, reusing the existing
    /// allocation so a recycled scroll row never allocates. The wrap link goes with
    /// the content: a recycled row must not inherit one and glue two unrelated lines
    /// together.
    pub(super) fn reset(&mut self, cols: usize, blank: Cell) {
        if self.cells.len() == cols {
            self.cells.iter_mut().for_each(|c| *c = blank);
        } else {
            self.cells.clear();
            self.cells.resize(cols, blank);
        }
        self.combining.clear();
        self.wrapped = false;
    }

    /// The heap this row holds: its cells and its combining side table, at
    /// **capacity** rather than length, because capacity is what is actually held.
    /// A recycled row keeps both allocations by design (see [`Row::reset`]), so the
    /// difference is the whole point of measuring it.
    pub(super) fn storage_bytes(&self) -> usize {
        self.cells.capacity() * std::mem::size_of::<Cell>()
            + self.combining.capacity() * std::mem::size_of::<CombiningMark>()
    }

    /// Blank a wide glyph straddling the boundary *before* `col`, so an operation that
    /// cuts the row there cannot leave half a glyph behind.
    ///
    /// A pair straddles the cut exactly when the cell at `col` is a spacer: its leader is
    /// the cell before it, on the far side of the line about to be drawn.
    ///
    /// ```text
    ///   split before col 2        the pair straddles the cut, so both halves go
    ///   ┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
    ///   │ a │ 一────▶│ b │   ──▶  │ a │   │   │ b │
    ///   └───┴───┴─┃─┴───┘         └───┴───┴───┴───┘
    ///             ┃ the cut
    /// ```
    ///
    /// Every shift is a pair of cuts (see [`Buffer::insert_blanks`] and
    /// [`Buffer::delete_chars`]), which is why this is stated once here rather than
    /// open-coded per operation: each one gets it independently wrong otherwise.
    pub(super) fn split_wide_at(&mut self, col: usize, blank: Cell) {
        if !self.cells.get(col).is_some_and(|c| c.is_wide_spacer()) {
            return;
        }
        let Some(lead) = col.checked_sub(1) else {
            return;
        };
        if let Some(slice) = self.cells.get_mut(lead..=col) {
            slice.iter_mut().for_each(|c| *c = blank);
        }
        self.combining.retain(|m| m.col != lead && m.col != col);
    }

    /// The combining marks attached to `col`, in arrival order; empty if the cell
    /// carries none. Yields `char` so a caller appends them straight onto a run
    /// string without materializing a temporary collection.
    pub(super) fn marks(&self, col: usize) -> impl Iterator<Item = char> + '_ {
        self.combining
            .iter()
            .filter(move |m| m.col == col)
            .map(|m| m.ch)
    }

    /// Whether `col` carries any combining mark. Cheaper than `marks(col).next()` at
    /// the ink test, and reads clearly there.
    pub(super) fn has_marks(&self, col: usize) -> bool {
        self.combining.iter().any(|m| m.col == col)
    }

    pub(super) fn add_mark(&mut self, col: usize, mark: char) {
        // Clamp how many marks one cell can stack. Unicode's Stream-Safe format caps a
        // grapheme at 30 combining marks; a hostile stream (Zalgo) piles on thousands,
        // which would grow this row's side table and every render string it feeds
        // without bound. Beyond the cap the extra marks are dropped: invisible clutter
        // no reader needs, and the bound keeps retained memory and render work finite.
        if self.combining.iter().filter(|m| m.col == col).count() >= MAX_COMBINING_PER_CELL {
            return;
        }
        self.combining.push(CombiningMark { col, ch: mark });
    }

    pub(super) fn clear_marks(&mut self, col: usize) {
        self.combining.retain(|m| m.col != col);
    }

    /// Grow or shrink the row to `new_cols`, padding with blanks or truncating.
    /// Truncation drops any combining marks past the new edge and blanks a wide
    /// glyph's leader whose spacer just fell off, so no half of a wide cell is
    /// ever left dangling. [`Row::wrapped`] is line state, not cell state, so it
    /// survives untouched. This is the column half of a terminal resize; it does
    /// not re-wrap soft-wrapped content (see [`Buffer::resize`]).
    pub(super) fn resize_cols(&mut self, new_cols: usize) {
        let old = self.cells.len();
        if new_cols < old {
            self.cells.truncate(new_cols);
            self.combining.retain(|m| m.col < new_cols);
            if let Some(last) = self.cells.last_mut() {
                if last.is_wide_leader() {
                    *last = Cell::BLANK;
                }
            }
        } else if new_cols > old {
            self.cells.resize(new_cols, Cell::BLANK);
        }
    }

    /// Refill this row in place from `cells` (at most `cols` of them), padding to `cols`
    /// with blanks, and set the soft-wrap link. Reuses the existing cell allocation so a
    /// reflow recycles rows instead of allocating fresh. Combining marks are cleared; the
    /// caller re-adds any the new columns carry.
    pub(super) fn refill(&mut self, cols: usize, cells: &[Cell], wrapped: bool) {
        self.cells.clear();
        self.cells.extend_from_slice(cells);
        self.cells.resize(cols, Cell::BLANK);
        self.combining.clear();
        self.wrapped = wrapped;
    }
}

/// A logical line reassembled from a run of soft-wrapped physical rows: the flat cell
/// sequence, with its combining marks re-keyed from per-row columns to the logical
/// column. This is reflow's intermediate — [`Buffer::reflow`] unwraps the stream into
/// these, then rewraps them at the new width. It carries no "was this a hard newline"
/// flag on purpose: a line ends either at a real newline or at [`MAX_LOGICAL_COLS`], and
/// both cases rewrap the same way (the last row is `wrapped = false`), so the seam a cap
/// break leaves behind is indistinguishable from a newline and will not re-join on a
/// later reflow — which is exactly the bound the cap is there to keep.
#[derive(Default)]
pub(super) struct LogicalLine {
    pub(super) cells: Vec<Cell>,
    pub(super) combining: Vec<CombiningMark>,
}

impl LogicalLine {
    /// Drop trailing default-blank cells (and any marks that sat on them). A cell with a
    /// non-default background or a combining mark is content, not padding, and stops the
    /// trim: a coloured prompt bar drawn with spaces, or a marked blank, must survive a
    /// reflow rather than be pulled up into the line above it. A wide glyph's spacer is
    /// not `Cell::BLANK` (it carries `WIDE_SPACER`), so a trailing wide pair is never
    /// half-trimmed.
    pub(super) fn trim_trailing_blanks(&mut self) {
        let mut end = self.cells.len();
        while end > 0
            && self.cells[end - 1] == Cell::BLANK
            && !self.combining.iter().any(|m| m.col == end - 1)
        {
            end -= 1;
        }
        self.cells.truncate(end);
        self.combining.retain(|m| m.col < end);
    }
}

/// Take a row from the recycle `pool` (or allocate one when it is empty) and fill it from
/// `cells` padded to `cols`, with the given soft-wrap link and no combining marks. This is
/// reflow's row factory: draining the old stream leaves a pool of `Row`s whose cell
/// allocations this reuses, so a rewrap does not allocate a row per line.
pub(super) fn take_row(pool: &mut Vec<Row>, cols: usize, cells: &[Cell], wrapped: bool) -> Row {
    let mut row = pool.pop().unwrap_or_else(|| Row::filled(cols, Cell::BLANK));
    row.refill(cols, cells, wrapped);
    row
}

/// One screen buffer: the visible rows (a ring so a scroll rotates row headers,
/// never cells), the scrollback ring behind it, the cursor, the DECSTBM scroll
/// region, and the tab stops. Ring mechanics live here; the terminal *semantics*
/// (what each escape means) live on `Screen`.
pub(super) struct Buffer {
    pub(super) cols: usize,
    pub(super) rows: usize,
    /// Visible rows, always exactly `rows` long.
    pub(super) lines: VecDeque<Row>,
    /// History above the screen, oldest at the front, capped at `scrollback_limit`.
    pub(super) scrollback: VecDeque<Row>,
    pub(super) scrollback_limit: usize,
    /// How many rows have left the front of the stream for good: evicted from a full
    /// ring, or dropped when `ED 3` cleared the history. It is the offset that turns an
    /// [`AbsRow`] into an index into `scrollback ++ lines`, and the reason an id is
    /// never reused: the front of the stream only ever moves forward.
    pub(super) evicted: u64,
    pub(super) cursor: Cursor,
    /// What DECSC last stored *for this buffer*. Each screen owns its own slot, and it
    /// outlives a trip to the other screen and back: a program may save on the alt
    /// screen, leave, return, and restore, and it expects to find what it saved.
    pub(super) saved: Option<Saved>,
    /// DECSTBM scroll region, inclusive, within `0..rows`.
    pub(super) scroll_top: usize,
    pub(super) scroll_bottom: usize,
    /// Tab stops; `tabs[c]` is true where a horizontal tab lands.
    pub(super) tabs: Vec<bool>,
}

impl Buffer {
    pub(super) fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
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
            scrollback: VecDeque::with_capacity(scrollback_capacity(scrollback_limit)),
            scrollback_limit,
            evicted: 0,
            cursor: Cursor::default(),
            saved: None,
            scroll_top: 0,
            scroll_bottom: rows - 1,
            tabs: default_tabs(cols),
        }
    }

    pub(super) fn line(&self, row: usize) -> Option<&Row> {
        self.lines.get(row)
    }

    pub(super) fn cell(&self, row: usize, col: usize) -> Cell {
        self.lines
            .get(row)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// The heap this buffer holds: every row's cells and marks, the two ring
    /// allocations the row headers sit in, and the tab-stop table.
    ///
    /// Both rings count at **capacity**, which is not a rounding detail: a ring is
    /// filled by pushing past its limit and popping the front back off, so it doubles
    /// on the push that crosses the cap and then holds the doubled allocation forever.
    /// Measuring `len` would hide exactly the kind of waste this number exists to find.
    pub(super) fn storage_bytes(&self) -> usize {
        let ring = |d: &VecDeque<Row>| {
            d.capacity() * std::mem::size_of::<Row>()
                + d.iter().map(Row::storage_bytes).sum::<usize>()
        };
        ring(&self.lines) + ring(&self.scrollback) + self.tabs.capacity()
    }

    /// The row at `idx` in this buffer's stream — history first (oldest at 0), then the
    /// live screen — which is the one column of rows every other row lookup is a view
    /// onto. `None` past the live bottom.
    pub(super) fn stream_row(&self, idx: usize) -> Option<&Row> {
        if idx < self.scrollback.len() {
            self.scrollback.get(idx)
        } else {
            self.lines.get(idx - self.scrollback.len())
        }
    }

    /// The stream index of `abs`, or `None` once it has been evicted off the front.
    pub(super) fn stream_index(&self, abs: AbsRow) -> Option<usize> {
        usize::try_from(abs.0.checked_sub(self.evicted)?).ok()
    }

    /// The row `abs` names, or `None` if it has aged out of history or does not exist
    /// yet (an id past the live bottom).
    pub(super) fn abs_row(&self, abs: AbsRow) -> Option<&Row> {
        self.stream_row(self.stream_index(abs)?)
    }

    /// The id of the row at stream index `idx`.
    pub(super) fn abs_of(&self, idx: usize) -> AbsRow {
        AbsRow(
            self.evicted
                .saturating_add(idx.try_into().unwrap_or(u64::MAX)),
        )
    }

    /// One past the newest row's id: the live bottom of the stream.
    pub(super) fn abs_end(&self) -> AbsRow {
        self.abs_of(self.scrollback.len() + self.rows)
    }

    /// The row shown at display position `display_row` when the view is scrolled
    /// `offset` lines up into scrollback. The window of `rows` starts `offset` lines
    /// above the live top. `offset` is assumed already clamped to `0..=scrollback
    /// .len()`. Returns `None` past the end (a short final window).
    pub(super) fn view_row(&self, display_row: usize, offset: usize) -> Option<&Row> {
        let base = self.scrollback.len().checked_sub(offset)?;
        self.stream_row(base + display_row)
    }

    /// Write `cell` at (row, col) with no wide-pair or combining-mark bookkeeping.
    /// Replacing a row's final cell replaces the text that wrapped out of it, so the
    /// line stops there: the wrap link goes (autowrap sets it again if the new text
    /// wraps in its turn).
    pub(super) fn set_raw(&mut self, row: usize, col: usize, cell: Cell) {
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        let Some(slot) = r.cells.get_mut(col) else {
            return;
        };
        *slot = cell;
        if col + 1 == r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Write `cell` at (row, col), first breaking any wide pair it straddles so a
    /// half-overwritten wide glyph never leaves an orphan on screen, and dropping
    /// any combining marks the overwritten cell carried.
    pub(super) fn write_cell(&mut self, row: usize, col: usize, cell: Cell) {
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

    /// Fill `[start_col, start_col + run.len())` on `row` with `run`'s bytes as
    /// single-width cells sharing `pen` — the bulk equivalent of one
    /// [`write_cell`](Self::write_cell) per byte, but with the row (a `VecDeque`
    /// index) resolved once instead of three times per cell. This is the ASCII
    /// print hot path, so the redundant per-cell work is what it strips.
    ///
    /// The wide-pair break only has to look at the two edges: a wide character
    /// wholly inside the run has both halves overwritten (no orphan), so only a
    /// pair the run *straddles* can leave a partner outside it — a spacer at the
    /// left edge orphans its leader to the left, a leader at the right edge orphans
    /// its spacer to the right. Combining marks under the run drop in a single
    /// retain. The `bulk_ascii_matches_per_char_under_fuzz` test pins this against
    /// the byte-at-a-time path.
    pub(super) fn fill_ascii_run(&mut self, row: usize, start_col: usize, run: &[u8], pen: Pen) {
        let cols = self.cols;
        let end_col = start_col + run.len();
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        // Left edge: overwriting a wide spacer orphans its leader one cell left.
        if start_col > 0 && r.cells.get(start_col).is_some_and(|c| c.is_wide_spacer()) {
            if let Some(slot) = r.cells.get_mut(start_col - 1) {
                *slot = Cell::BLANK;
            }
            r.clear_marks(start_col - 1);
        }
        // Right edge: overwriting a wide leader orphans its spacer one cell right.
        if end_col < cols && r.cells.get(end_col - 1).is_some_and(|c| c.is_wide_leader()) {
            if let Some(slot) = r.cells.get_mut(end_col) {
                *slot = Cell::BLANK;
            }
        }
        for (k, &byte) in run.iter().enumerate() {
            if let Some(slot) = r.cells.get_mut(start_col + k) {
                *slot = Cell {
                    rune: char::from(byte),
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: pen.attrs,
                    link: pen.link,
                };
            }
        }
        r.combining
            .retain(|m| m.col < start_col || m.col >= end_col);
        // As in `set_raw`: a run reaching the final column replaces whatever wrapped out
        // of it. The caller re-wraps this row if the run itself runs off the edge.
        if end_col >= r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Fill one row segment from decoded width-1/2 scalars. Widths were already
    /// classified by [`Screen::print_text_run`], and the caller guarantees the
    /// complete segment fits on this row. Like [`Self::fill_ascii_run`], this
    /// resolves the row once, repairs only wide pairs straddling the two edges,
    /// and removes combining marks under the whole overwritten interval once.
    pub(super) fn fill_text_run(
        &mut self,
        row: usize,
        start_col: usize,
        chars: &[char],
        widths: &[u8],
        pen: Pen,
    ) {
        let columns = widths
            .iter()
            .fold(0usize, |sum, width| sum.saturating_add(usize::from(*width)));
        let end_col = start_col.saturating_add(columns).min(self.cols);
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        if start_col > 0
            && r.cells
                .get(start_col)
                .is_some_and(|cell| cell.is_wide_spacer())
        {
            if let Some(slot) = r.cells.get_mut(start_col - 1) {
                *slot = Cell::BLANK;
            }
            r.clear_marks(start_col - 1);
        }
        if end_col < r.cells.len()
            && r.cells
                .get(end_col.saturating_sub(1))
                .is_some_and(|cell| cell.is_wide_leader())
        {
            if let Some(slot) = r.cells.get_mut(end_col) {
                *slot = Cell::BLANK;
            }
        }

        let mut col = start_col;
        for (&c, &cell_width) in chars.iter().zip(widths) {
            let wide = cell_width == 2 && col.saturating_add(1) < r.cells.len();
            if let Some(slot) = r.cells.get_mut(col) {
                *slot = Cell {
                    rune: c,
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: if wide {
                        pen.attrs | Attrs::WIDE_LEADER
                    } else {
                        pen.attrs
                    },
                    link: pen.link,
                };
            }
            if wide {
                if let Some(slot) = r.cells.get_mut(col.saturating_add(1)) {
                    *slot = Cell {
                        rune: ' ',
                        fg: pen.fg,
                        bg: pen.bg,
                        attrs: pen.attrs | Attrs::WIDE_SPACER,
                        link: pen.link,
                    };
                }
            }
            col = col.saturating_add(usize::from(cell_width));
        }
        r.combining
            .retain(|mark| mark.col < start_col || mark.col >= end_col);
        if end_col >= r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Where a row-addressing sequence (CUP, HVP, VPA) actually lands `row`.
    ///
    /// Under origin mode (DECOM) row numbers are relative to the scroll region *and*
    /// bounded by it, so a program addressing past the bottom margin stays inside the
    /// window it asked for; otherwise they are absolute and bounded by the screen. Stated
    /// once because two sequences ask it and a program that moves by CUP and by VPA
    /// interchangeably must not find them disagreeing.
    pub(super) fn origin_row(&self, row: usize, origin: bool) -> usize {
        if origin {
            self.scroll_top.saturating_add(row).min(self.scroll_bottom)
        } else {
            row.min(self.rows.saturating_sub(1))
        }
    }

    /// Mint default tab stops for columns that have just come into existence.
    ///
    /// A widen only ever *adds*: the stops below the old width are state a program set,
    /// and a window resize is not a program asking to lose them (see the note in
    /// [`Self::resize`]). Columns that have never existed are the one case where fresh
    /// defaults are right, because nothing has ever said anything about them. Both the
    /// clamping resize and the reflow grow the table, so the rule lives here.
    pub(super) fn grow_tabs(&mut self, new_cols: usize) {
        for c in self.tabs.len()..new_cols {
            self.tabs.push(c % TAB_WIDTH == 0);
        }
    }

    /// Park the cursor after a print that filled columns up to `end_col` (exclusive).
    ///
    /// This is xterm's last-column rule, and it is stated here once because all three
    /// print paths — the scalar, the bulk ASCII run, and the decoded text run — end by
    /// asking it. When the text ran to the right edge the cursor stays *on* the last
    /// column and the wrap is only deferred (under DECAWM), so printing exactly `cols`
    /// characters does not scroll; anywhere short of the edge the cursor simply lands
    /// there and any deferred wrap is cancelled.
    pub(super) fn advance_cursor(&mut self, end_col: usize, autowrap: bool) {
        if end_col >= self.cols {
            self.cursor.col = self.cols.saturating_sub(1);
            self.cursor.pending_wrap = autowrap;
        } else {
            self.cursor.col = end_col;
            self.cursor.pending_wrap = false;
        }
    }

    /// Scroll `[top, bottom]` up by `n`, feeding `blank` rows in at the bottom.
    /// When `to_scrollback` and the region reaches the top of the screen, the
    /// rows leaving the top are retained in scrollback; otherwise they are
    /// discarded. Either way the row storage is recycled, so a steady scroll
    /// allocates nothing once scrollback is full.
    ///
    /// Reports what left, as a [`Scrolled`]: rows that *retired into history* keep their
    /// [`AbsRow`] and pull a scrolled-back viewport along with them
    /// ([`Screen::follow_history`]); rows that were *discarded* are torn out of the
    /// middle of the stream, shifting every id below them, which ends the
    /// [`RowEpoch`]. The push count is a count of pushes, not of net growth: once the
    /// ring is full every push also evicts, and that difference is what makes a view
    /// parked at the top hold still while history slides out from under it.
    pub(super) fn scroll_up_range(
        &mut self,
        top: usize,
        bottom: usize,
        n: usize,
        blank: Cell,
        to_scrollback: bool,
    ) -> Scrolled {
        let mut out = Scrolled::default();
        if top > bottom || bottom >= self.rows {
            return out;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(top) else {
                return out;
            };
            let mut recycled = if to_scrollback && top == 0 && self.scrollback_limit > 0 {
                out.pushed += 1;
                self.scrollback.push_back(leaving);
                if self.scrollback.len() > self.scrollback_limit {
                    self.evicted += 1;
                    self.scrollback
                        .pop_front()
                        .unwrap_or_else(|| Row::filled(self.cols, blank))
                } else {
                    Row::filled(self.cols, blank)
                }
            } else {
                // Torn out of the middle of the stream: everything past the hole shifts.
                out.renumbered = true;
                leaving
            };
            recycled.reset(self.cols, blank);
            self.lines.insert(bottom, recycled);
        }
        // A retiring row keeps its slot in the stream, but the blank replacing it is
        // inserted at the region's bottom margin. Only when that is the last row of the
        // screen does it land at the end of the stream and leave the ids alone; short of
        // it, every row below the region shifts up one while sitting perfectly still.
        if out.pushed > 0 && bottom + 1 < self.rows {
            out.renumbered = true;
        }
        out
    }

    /// Scroll `[top, bottom]` down by `n`, feeding `blank` rows in at the top.
    /// Never touches scrollback (only content leaving the top of the full screen
    /// enters history).
    ///
    /// Every row it moves is [`Scrolled::renumbered`]: the content slides *down* while the
    /// ids stay where they are, so the row that was `AbsRow(n)` now holds what used to be
    /// above it, and the row pushed off the bottom of the region is gone. That shifts the
    /// content out from under anything holding an id, which ends the [`RowEpoch`] — the
    /// mirror image of a region scroll up.
    pub(super) fn scroll_down_range(
        &mut self,
        top: usize,
        bottom: usize,
        n: usize,
        blank: Cell,
    ) -> Scrolled {
        let mut out = Scrolled::default();
        if top > bottom || bottom >= self.rows {
            return out;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(bottom) else {
                return out;
            };
            out.renumbered = true;
            let mut recycled = leaving;
            recycled.reset(self.cols, blank);
            self.lines.insert(top, recycled);
        }
        out
    }

    /// Push a row off the top into scrollback, evicting the oldest if the ring is
    /// full. A no-op when this buffer keeps no history (the alt screen), so its
    /// shrunk-away rows are simply dropped.
    pub(super) fn push_history(&mut self, row: Row) {
        if self.scrollback_limit == 0 {
            return;
        }
        self.scrollback.push_back(row);
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
            self.evicted += 1;
        }
    }

    /// Drop the whole history (`ED 3`). The rows are gone from the front of the stream
    /// like any eviction, so the *live* rows keep the ids they already had: an id names
    /// a line, and nothing here moved a line.
    pub(super) fn clear_history(&mut self) {
        self.evicted += u64::try_from(self.scrollback.len()).unwrap_or(u64::MAX);
        self.scrollback.clear();
    }

    /// Resize the buffer to `new_cols` x `new_rows`. Columns truncate or pad every row
    /// (visible and scrollback); rows grow by pulling recent history back down onto the
    /// screen (bottom stays pinned, scrollback fills the new space above) and only pad
    /// blanks below once history runs out, and shrink by dropping rows below the cursor
    /// first, then scrolling the top into scrollback so the cursor stays on screen. The
    /// scroll region resets to the full screen (as xterm does on resize) and the cursor is
    /// re-clamped.
    ///
    /// Deliberately does NOT re-wrap soft-wrapped lines on a width change — that is
    /// [`Buffer::reflow`]'s job, called instead of this when the width changes. This is the
    /// same-width path (a height change or a scale-driven metrics change), where re-wrapping
    /// would be wasted work.
    pub(super) fn resize(&mut self, new_cols: usize, new_rows: usize) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);

        if new_cols != self.cols {
            for r in self.lines.iter_mut().chain(self.scrollback.iter_mut()) {
                r.resize_cols(new_cols);
            }
            self.cols = new_cols;
            // Tab stops are not geometry: they are state a program set, and a window
            // resize is not a program asking to lose it. `HTS` at column 3 has to still
            // be there after a drag, so the table is never rebuilt — it already covers
            // [`MAX_TAB_COLUMNS`], and only a screen wider than that needs more.
            self.grow_tabs(new_cols);
            if self.cursor.col >= new_cols {
                self.cursor.col = new_cols - 1;
            }
        }

        match new_rows.cmp(&self.rows) {
            Ordering::Greater => {
                let mut grow = new_rows - self.rows;
                // Growing pulls recent history back down onto the screen so the bottom line
                // stays pinned to the window's bottom edge and the new space reveals
                // scrollback above, as xterm and wezterm do — not blank rows shoved in below
                // the content. The stream `scrollback ++ lines` is unchanged, only its split
                // point moves, so every row keeps its id. Blank rows fill in below only once
                // history runs out.
                while grow > 0 {
                    let Some(row) = self.scrollback.pop_back() else {
                        break;
                    };
                    self.lines.push_front(row);
                    self.cursor.row += 1;
                    grow -= 1;
                }
                for _ in 0..grow {
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

    /// Reflow the buffer to `new_cols` x `new_rows`, re-wrapping soft-wrapped lines to the
    /// new width instead of clamping them. This is the widen-rejoins / narrow-splits
    /// behaviour: a `docker ps` line that wrapped narrow pulls back together when the
    /// window widens, and a long line breaks cleanly when it narrows. Only the primary
    /// buffer calls this; the alt screen keeps [`Buffer::resize`], because a full-screen
    /// program owns that canvas and repaints on `SIGWINCH`.
    ///
    /// The cursor and the scrolled view are preserved by recording each as a logical
    /// `(line, offset)` before the rewrap and locating it again after. `view_offset` is
    /// how far the view is scrolled up into this buffer's scrollback (0 = pinned to the
    /// live bottom); the returned value is that same view re-derived for the new geometry.
    ///
    /// ```text
    ///   unwrap                      rewrap to new_cols        re-split (cursor on screen)
    ///   physical rows ──▶ logical lines ──▶ physical rows ──▶ scrollback + live screen
    ///        (wrapped flag              (chunks of new_cols,      (bottom new_rows are
    ///         joins the runs)            wide glyph never split)   live; rest is history)
    /// ```
    ///
    /// The cursor and the scrolled view are carried across internally. The returned
    /// [`RowRemap`] carries everything anchored by [`AbsRow`] from the outside — a text
    /// selection and the OSC 133 prompt marks — by translating each old id to where its cell
    /// now sits; the caller ([`Screen::resize`]) applies it.
    ///
    /// `frozen_from` is the old stream index where the live prompt region begins (the last
    /// idle shell prompt). Rows at or below it are **clamped**, not rewrapped: the shell
    /// repaints its prompt on `SIGWINCH` and clears by its own row count, so re-wrapping a
    /// full-width prompt bar to a different number of rows would leave the extra row as
    /// uncleared residue above it. Pass the old row count to reflow everything (no prompt to
    /// protect).
    pub(super) fn reflow(
        &mut self,
        new_cols: usize,
        new_rows: usize,
        view_offset: usize,
        frozen_from: usize,
    ) -> (usize, RowRemap) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);
        let old_evicted = self.evicted;
        let old_scrollback = self.scrollback.len();

        // Anchors, as absolute indices into the current stream (`scrollback ++ lines`).
        // The cursor always sits on a live row; the view top matters only when scrolled.
        let cursor_idx = self.scrollback.len() + self.cursor.row;
        let cursor_col = self.cursor.col;
        let track_view = view_offset > 0;
        let view_idx = self.scrollback.len().saturating_sub(view_offset);

        // Drain the whole stream into one Vec so every Row is ours to recycle.
        let mut old_rows: Vec<Row> = Vec::with_capacity(self.scrollback.len() + self.lines.len());
        old_rows.extend(self.scrollback.drain(..));
        old_rows.extend(self.lines.drain(..));
        let old_count = old_rows.len();
        let frozen_from = frozen_from.min(old_count);
        // Split off the live prompt region; the head is what actually reflows.
        let frozen_rows: Vec<Row> = old_rows.split_off(frozen_from);

        // --- unwrap the reflowed head: join soft-wrapped runs into logical lines, recording
        // for every old row the logical line it joined and the offset its first column sits
        // at (`old`), so any id can later be resolved to a logical position ---
        let mut lines: Vec<LogicalLine> = Vec::new();
        let mut old: Vec<(usize, usize)> = Vec::with_capacity(old_rows.len());
        let mut cur = LogicalLine::default();
        let mut pending = false; // `cur` has rows not yet closed into `lines`
        for (idx, row) in old_rows.iter().enumerate() {
            old.push((lines.len(), cur.cells.len()));
            let base = cur.cells.len();
            cur.cells.extend_from_slice(&row.cells);
            for m in &row.combining {
                cur.combining.push(CombiningMark {
                    col: base + m.col,
                    ch: m.ch,
                });
            }
            // A soft-wrapped row whose continuation begins with a wide glyph left its last
            // column blank because the pair could not fit there (the printer's last-column
            // rule). That blank is wrap padding, not text: drop it so a widen rejoins the glyph
            // flush against the text before it instead of preserving it as a phantom space.
            if row.wrapped
                && cur.cells.last() == Some(&Cell::BLANK)
                && old_rows
                    .get(idx + 1)
                    .and_then(|next| next.cells.first())
                    .is_some_and(|c| c.is_wide_leader())
            {
                cur.cells.pop();
            }
            pending = true;
            // Close on a hard newline, or when the run reaches the cap (a forced seam).
            if !row.wrapped || cur.cells.len() >= MAX_LOGICAL_COLS {
                cur.trim_trailing_blanks();
                lines.push(std::mem::take(&mut cur));
                pending = false;
            }
        }
        // A trailing wrapped run with no terminator cannot arise (the bottom live row is
        // never wrapped past the screen), but close any remainder rather than lose it.
        if pending {
            cur.trim_trailing_blanks();
            lines.push(cur);
        }

        // Empty rows below the cursor are screen padding, not content. Drop the trailing ones
        // so the reflow lets the content rise to fill the screen from the bottom, instead of
        // stranding it above blank lines (and needlessly into scrollback) when a narrow
        // multiplies the row count. Only when nothing is frozen: with a live prompt below, the
        // cursor is down in it and the blank rows above are the shell's own spacing.
        if frozen_rows.is_empty() {
            let cursor_line = old.get(cursor_idx).map(|a| a.0).unwrap_or(0);
            let mut keep = lines.len();
            while keep > cursor_line + 1 && lines[keep - 1].cells.is_empty() {
                keep -= 1;
            }
            lines.truncate(keep);
        }

        // --- rewrap the head: lay each logical line into new_cols-wide rows, recycling the
        // drained rows, and record each line's first new index and every new row's start ---
        let mut pool = old_rows;
        let mut new_stream: Vec<Row> = Vec::with_capacity(old_count);
        let mut line_first: Vec<usize> = Vec::with_capacity(lines.len() + 1);
        let mut new_start: Vec<usize> = Vec::with_capacity(old_count);

        for line in lines.iter() {
            line_first.push(new_stream.len());
            let cells = &line.cells;
            if cells.is_empty() {
                // An empty logical line is a bare newline: it still shows one blank row.
                new_start.push(0);
                new_stream.push(take_row(&mut pool, new_cols, &[], false));
            } else {
                let mut i = 0;
                while i < cells.len() {
                    let mut end = (i + new_cols).min(cells.len());
                    // Never split a wide glyph from its spacer at the wrap: if a leader
                    // would be this row's last column with its spacer on the next, push
                    // the whole pair down (the printer's own last-column rule).
                    if end < cells.len() && cells[end - 1].is_wide_leader() {
                        end -= 1;
                    }
                    // Degenerate new_cols == 1 against a wide glyph: place it alone rather
                    // than loop forever.
                    if end <= i {
                        end = i + 1;
                    }
                    // A wide glyph whose spacer cannot share its row (only at new_cols == 1)
                    // is stored clipped: the leader alone with its WIDE_LEADER attr stripped,
                    // and its spacer dropped, exactly as the printer records a wide glyph at a
                    // single column. Leaving the leader marked would strand it without the
                    // spacer the grid's invariant demands, and copying the spacer would orphan
                    // it onto the next row.
                    let clip_wide = cells[i].is_wide_leader() && end == i + 1;
                    let next_i =
                        if clip_wide && cells.get(i + 1).is_some_and(|c| c.is_wide_spacer()) {
                            i + 2
                        } else {
                            end
                        };
                    let wrapped = next_i < cells.len();
                    let mut row = if clip_wide {
                        let mut clipped = cells[i];
                        clipped.attrs.remove(Attrs::WIDE_LEADER);
                        take_row(&mut pool, new_cols, &[clipped], wrapped)
                    } else {
                        take_row(&mut pool, new_cols, &cells[i..end], wrapped)
                    };
                    for m in &line.combining {
                        if m.col >= i && m.col < end {
                            row.combining.push(CombiningMark {
                                col: m.col - i,
                                ch: m.ch,
                            });
                        }
                    }
                    new_start.push(i);
                    new_stream.push(row);
                    i = next_i;
                }
            }
        }
        line_first.push(new_stream.len());

        // The live prompt region follows, clamped to the new width one row per row so its row
        // count is exactly what the shell's redraw expects.
        let reflowed_len = new_stream.len();
        for mut row in frozen_rows {
            row.resize_cols(new_cols);
            new_stream.push(row);
        }

        let remap = RowRemap {
            evicted: old_evicted,
            new_cols,
            lines_len: lines.len(),
            old,
            line_first,
            new_start,
            frozen_from,
            reflowed_len,
        };

        // --- re-split: the live screen is the bottom new_rows rows, the rest is history,
        // the cursor clamped into the screen ---
        let total = new_stream.len();
        let (cursor_new_idx, cursor_new_col) = remap
            .locate(cursor_idx, cursor_col)
            .unwrap_or((total.saturating_sub(1), 0));
        let view_new = track_view
            .then(|| remap.locate(view_idx, 0).map(|(i, _)| i))
            .flatten();
        // The live screen is a window of new_rows rows over the rewrapped stream. Pin it to the
        // bottom so the most recent output shows, but never above the cursor: a narrow can push
        // more rows below the cursor than fit, and the cursor must stay live because output
        // writes there. When that happens the window slides up to the cursor and the rows below
        // it that no longer fit are dropped, as a height shrink drops rows below the cursor to
        // keep it on screen.
        let live_top = total.saturating_sub(new_rows).min(cursor_new_idx);
        let live_end = live_top.saturating_add(new_rows).min(total);

        let mut lines_out: VecDeque<Row> = VecDeque::with_capacity(new_rows);
        // Sized for the ring this will become, not just the rows going into it: a
        // narrow can rewrap more history than the limit holds (it is trimmed just
        // below), and the steady state after that is a full ring (see
        // [`scrollback_capacity`]). Either way the deque must never have to grow.
        let mut scrollback_out: VecDeque<Row> =
            VecDeque::with_capacity(live_top.max(scrollback_capacity(self.scrollback_limit)));
        for (idx, row) in new_stream.into_iter().enumerate() {
            if idx < live_top {
                scrollback_out.push_back(row);
            } else if idx < live_end {
                lines_out.push_back(row);
            }
            // idx >= live_end: a row below the cursor the shrunk screen cannot hold; dropped.
        }
        while lines_out.len() < new_rows {
            lines_out.push_back(take_row(&mut pool, new_cols, &[], false));
        }

        self.scrollback = scrollback_out;
        self.lines = lines_out;
        self.cols = new_cols;
        self.rows = new_rows;
        self.cursor.row = cursor_new_idx.saturating_sub(live_top).min(new_rows - 1);
        self.cursor.col = cursor_new_col.min(new_cols - 1);
        // Preserve pending-wrap only if the cursor still sits in the last column; anywhere
        // else there is now room, so the deferred wrap no longer applies.
        self.cursor.pending_wrap = self.cursor.pending_wrap && self.cursor.col == new_cols - 1;
        // Carry the DECSC saved cursor across the reflow the same way. This is the piece
        // alacritty leaves out (it only clamps the column) and the reason a shell that saves
        // its cursor at the prompt and restores it on `SIGWINCH` — the shell-integration path
        // — lands its redraw right instead of on a stale row. See `Screen::resize`.
        if let Some(saved) = self.saved.as_mut() {
            let old_idx = old_scrollback + saved.cursor.row;
            match remap.locate(old_idx, saved.cursor.col) {
                Some((idx, col)) => {
                    saved.cursor.row = idx.saturating_sub(live_top).min(new_rows - 1);
                    saved.cursor.col = col;
                    saved.cursor.pending_wrap =
                        saved.cursor.pending_wrap && saved.cursor.col == new_cols - 1;
                }
                None => {
                    saved.cursor.row = saved.cursor.row.min(new_rows - 1);
                    saved.cursor.col = saved.cursor.col.min(new_cols - 1);
                }
            }
        }
        self.scroll_top = 0;
        self.scroll_bottom = new_rows - 1;
        self.grow_tabs(new_cols);

        // The rewrap can multiply the row count past the ring's cap (narrowing); evict the
        // oldest, as `push_history` does, so scrollback stays bounded.
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
            self.evicted = self.evicted.saturating_add(1);
        }

        // The view anchor named content by logical position; place it back at display row
        // 0. If the rewrap pulled it down into the live screen, pin to the bottom.
        let scrollback_len = self.scrollback.len();
        let new_view = match view_new {
            Some(idx) => live_top.saturating_sub(idx).min(scrollback_len),
            None => 0,
        };
        (new_view, remap)
    }

    /// Overwrite every cell of `row` with `fill` (a blank for the erases, an 'E' for
    /// DECALN), dropping its marks and its wrap link.
    pub(super) fn clear_line_full(&mut self, row: usize, fill: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            r.cells.iter_mut().for_each(|c| *c = fill);
            r.combining.clear();
            r.wrapped = false;
        }
    }

    /// Blank cells `[start, end)` of `row`, leaving the rest untouched. A range that
    /// reaches the final column erases the text that wrapped, so the line ends here.
    ///
    /// The range grows over a wide glyph's other half at either edge, and that is not
    /// politeness — it is the only answer that can be drawn. A wide glyph is a leader
    /// carrying the rune plus a spacer holding its second column, and neither half means
    /// anything alone:
    ///
    /// ```text
    ///   erase just the leader        erase just the spacer
    ///   ┌───┬───┬───┐                ┌───┬───┬───┐
    ///   │ a │   │▒▒▒│                │ a │ 一────▶│      the leader still draws two
    ///   └───┴───┴───┘                └───┴───┴───┘      columns, over the erased cell
    ///           ↑ a spacer holding
    ///             a column for a rune
    ///             that is gone
    /// ```
    ///
    /// So landing on a spacer takes its leader, and ending on a leader takes its spacer.
    pub(super) fn clear_line_range(&mut self, row: usize, start: usize, end: usize, blank: Cell) {
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        let len = r.cells.len();
        let (mut lo, mut hi) = (start, end.min(len));
        if lo >= hi {
            return;
        }
        if r.cells.get(lo).is_some_and(|c| c.is_wide_spacer()) {
            lo = lo.saturating_sub(1);
        }
        if r.cells.get(hi - 1).is_some_and(|c| c.is_wide_leader()) {
            hi = (hi + 1).min(len);
        }
        if let Some(slice) = r.cells.get_mut(lo..hi) {
            slice.iter_mut().for_each(|c| *c = blank);
        }
        r.combining.retain(|m| m.col < lo || m.col >= hi);
        if hi == len {
            r.wrapped = false;
        }
    }

    /// ICH: shift `[col, len)` right by `n`, blanking the `n` opened cells; cells
    /// pushed past the right edge are lost, including whatever wrapped out of the
    /// final column, so the wrap link goes with it.
    ///
    /// The shift cuts the row twice — at the cursor, where the move begins, and at the
    /// last cell that survives it, where the right edge eats the rest — and a wide glyph
    /// straddling either cut would lose half of itself (see [`Row::split_wide_at`]).
    pub(super) fn insert_blanks(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.split_wide_at(col, blank);
            r.split_wide_at(len - n, blank);
            r.cells.copy_within(col..len - n, col + n);
            if let Some(slice) = r.cells.get_mut(col..col + n) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|m| m.col < col || m.col + n < len);
            r.combining.iter_mut().for_each(|m| {
                if m.col >= col {
                    m.col += n
                }
            });
            r.wrapped = false;
        }
    }

    /// DCH: shift `[col+n, len)` left by `n`, blanking the `n` cells at the right.
    /// Blanking the tail ends the line there, so the wrap link goes too.
    ///
    /// Two cuts again, mirroring ICH's: at the cursor, where the deletion begins, and at
    /// the first cell pulled in over it (see [`Row::split_wide_at`]).
    pub(super) fn delete_chars(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.split_wide_at(col, blank);
            r.split_wide_at(col + n, blank);
            r.cells.copy_within(col + n..len, col);
            if let Some(slice) = r.cells.get_mut(len - n..len) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|m| m.col < col || m.col >= col + n);
            r.combining.iter_mut().for_each(|m| {
                if m.col >= col + n {
                    m.col -= n
                }
            });
            r.wrapped = false;
        }
    }

    pub(super) fn clear_all(&mut self, fill: Cell) {
        for row in 0..self.rows {
            self.clear_line_full(row, fill);
        }
    }
}

/// How many row slots a scrollback ring holding `limit` rows must be allocated: one
/// **past** the limit, and that spare slot is worth half a megabyte.
///
/// A row retires by pushing onto the back and popping the front back off, so a full
/// ring is momentarily `limit + 1` rows long. Against a `limit`-sized allocation that
/// push is one too many: the deque doubles, the pop brings the length back down, and
/// the doubled header array is held for the rest of the session. At the default 10k
/// ring that is 20000 slots kept for 10000 rows, 0.53 MiB of nothing, on every
/// terminal, forever. It is stated here rather than at the two allocation sites
/// because [`Buffer::reflow`] rebuilds the ring and silently reintroduced it on every
/// width change.
///
/// ```text
///   cap = limit      push ─▶ len = limit+1 ─▶ REALLOC to 2*limit ─▶ pop ─▶ len = limit
///   cap = limit + 1  push ─▶ len = limit+1 ─▶ fits ──────────────▶ pop ─▶ len = limit
/// ```
pub(super) fn scrollback_capacity(limit: usize) -> usize {
    limit.saturating_add(1)
}

/// The default table: a stop every [`TAB_WIDTH`] columns, across [`MAX_TAB_COLUMNS`] or
/// the screen's width, whichever is more.
pub(super) fn default_tabs(cols: usize) -> Vec<bool> {
    (0..cols.max(MAX_TAB_COLUMNS))
        .map(|c| c % TAB_WIDTH == 0)
        .collect()
}
