//! Reading the grid: every question answered without changing anything.
//!
//! Three kinds of reader, and the distinction between the first two is load-bearing:
//!
//!   * **Display coordinates** — what is on screen *right now*, honouring how far the
//!     viewport is scrolled back ([`Screen::view_cell`], [`Screen::view_offset`]). The
//!     painter works here.
//!   * **Stream coordinates** — [`AbsRow`], naming a line of everything the child has
//!     ever printed, so a selection survives output scrolling under it
//!     ([`Screen::abs_row`], [`Screen::display_row`]). Anything stored across output
//!     works here.
//!   * **Derived spans** — the word, the logical line, the hyperlink under a cell, and
//!     the text a selection would copy.
//!
//! Mixing the first two is a bug with a name in this codebase: `abs_row` resolves a
//! *display* row, so handing it a live cursor row silently offsets the answer by the
//! scroll position (see [`Screen::cursor_abs_row`], which exists for exactly that).

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::buffer::{Row, Scrolled};
use super::*;

impl Screen {
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
        self.cursor_appearance.style
    }

    /// Whether the child asked the cursor to blink (default false; see
    /// [`CursorAppearance`]).
    pub fn cursor_blinks(&self) -> bool {
        self.cursor_appearance.blink
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

    /// Combining marks attached to visible `(row, col)`, in arrival order; empty if
    /// none.
    pub fn marks_at(&self, row: usize, col: usize) -> impl Iterator<Item = char> + '_ {
        self.active()
            .line(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
    }

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

    /// How many lines of scrollback the primary screen holds.
    pub fn scrollback_len(&self) -> usize {
        self.primary.scrollback.len()
    }

    /// Lines of history the *view* can scroll through: the primary's scrollback, or none
    /// at all on the alt screen, which does not show it (the history is still there
    /// behind it, which is why this is not [`Self::scrollback_len`]).
    ///
    /// Every scrollbar question is answered from here, so "the alt screen does not
    /// scroll" is stated once and the extent, the position, and a thumb drop cannot
    /// disagree about it.
    pub(super) fn scrollable_history(&self) -> i32 {
        if self.on_alt {
            0
        } else {
            self.scrollback_len() as i32
        }
    }

    /// What the scrollbar needs to size its thumb, in lines: how much content there is to
    /// scroll through (the history plus the screen), and how much of it is on screen.
    ///
    /// On the alt screen the content is exactly the viewport, so it reports as
    /// unscrollable — which is what hides the bar there, with no special case in the
    /// painter or the pointer.
    pub fn scroll_extent(&self) -> (i32, i32) {
        let rows = self.dimensions().1 as i32;
        (self.scrollable_history() + rows, rows)
    }

    /// How far down the whole history-plus-screen the view sits, in lines: 0 at the
    /// oldest line kept, the full history at the live bottom.
    ///
    /// This is [`Self::view_offset`] read the other way round. The offset counts lines
    /// *up* from the live bottom, because that is the direction the user scrolls back; a
    /// thumb travels *down* a track, so the scrollbar wants the complement.
    pub fn scroll_position(&self) -> i32 {
        self.scrollable_history() - self.view_offset() as i32
    }

    /// Put the view where a scrollbar thumb dropped it: `position` lines down the whole
    /// history-plus-screen, the inverse of [`Self::scroll_position`]. Clamped, so a drag
    /// flung past either end of the track rests at the oldest line or the live bottom.
    pub fn scroll_view_to(&mut self, position: i32) {
        let history = self.scrollable_history();
        self.view_offset = (history - position).clamp(0, history) as usize;
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

    /// Pin the view back to the live bottom (what fresh input does).
    pub fn scroll_view_to_bottom(&mut self) {
        self.view_offset = 0;
    }

    /// Keep a scrolled-back view anchored to the content it is showing after `pushed`
    /// rows left the screen for scrollback. The offset counts lines above the live
    /// bottom, so every row entering history moves the text under the user's eye one
    /// line further up; the offset has to grow with it or the view drifts. Output
    /// therefore never yanks the view back to the bottom: only the user does, by
    /// typing, pasting, or asking for the bottom.
    ///
    /// A view already at the live bottom stays there — `view_offset == 0` *is* what
    /// "follow the tail" means. The clamp to `scrollback.len()` is the full-ring case:
    /// history evicts from the front as fast as it grows, so a view parked at the very
    /// top holds while the oldest lines slide out from under it (as xterm does).
    pub(super) fn follow_history(&mut self, pushed: usize) {
        if pushed == 0 || self.view_offset == 0 {
            return;
        }
        self.view_offset = (self.view_offset + pushed).min(self.primary.scrollback.len());
    }

    /// Settle the two things a scroll leaves behind: the viewport follows the rows that
    /// retired into history, and a scroll that renumbered the stream ends the
    /// [`RowEpoch`] (see [`Scrolled`] for both, and for why a scroll can do both at once).
    ///
    /// Every scroll goes through here, so those two rules are stated once.
    pub(super) fn after_scroll(&mut self, scrolled: Scrolled) {
        self.follow_history(scrolled.pushed);
        if scrolled.renumbered {
            self.break_row_identity();
        }
    }

    /// The regime the grid's [`AbsRow`] ids currently belong to. A holder of row ids
    /// (the selection) keeps the epoch it minted them in and drops them when this
    /// changes, rather than resolving stale ids against a renumbered grid.
    pub fn row_epoch(&self) -> RowEpoch {
        self.epoch
    }

    /// End the current identity regime: the rows the outstanding ids named are gone or
    /// renumbered. See [`RowEpoch`] for the (short) list of things that do this.
    pub(super) fn break_row_identity(&mut self) {
        self.epoch = self.epoch.next();
        // The marks hold row ids, and the ids no longer name the lines they named. Keeping
        // them would mean offering to jump to a prompt that is not there any more — the
        // same reason the selection is dropped here, and the same fix.
        self.prompts.clear();
    }

    /// The id of the line currently shown at display `row`.
    ///
    /// Display row 0 sits `view_offset` lines above the live screen, so it is that far
    /// back into history; add the rows already evicted off the front and the display row
    /// itself, and the result names a line rather than a position.
    pub fn abs_row(&self, row: usize) -> AbsRow {
        let b = self.active();
        let top = b.scrollback.len().saturating_sub(self.view_offset());
        b.abs_of(top + row)
    }

    /// The id of the line the cursor sits on, which is where the child is printing.
    ///
    /// Not the same question as [`Self::abs_row`], and the difference is the whole reason
    /// this exists: that one resolves a *display* row, which moves with the viewport, and
    /// this one resolves a *live* row, which does not. Output lands on the live screen
    /// whatever the user has scrolled to, so anything naming the row a byte was printed on
    /// (an OSC 133 mark) has to ask this. Asking the display instead puts the answer
    /// `view_offset` lines back in history the moment someone scrolls up to read.
    pub(super) fn cursor_abs_row(&self) -> AbsRow {
        let b = self.active();
        b.abs_of(b.scrollback.len() + b.cursor.row)
    }

    /// Where `abs` sits on screen right now, or `None` when it is not in the visible
    /// band — scrolled off above or below it, or aged out of history entirely. The
    /// painter uses the `None` to clip: a selection whose top has scrolled away still
    /// paints the part you can see.
    pub fn display_row(&self, abs: AbsRow) -> Option<usize> {
        let b = self.active();
        let top = b.scrollback.len().saturating_sub(self.view_offset());
        let idx = b.stream_index(abs)?;
        let row = idx.checked_sub(top)?;
        (row < b.rows).then_some(row)
    }

    /// Whether `abs` still names a line that exists: not yet evicted off the front of
    /// history, not past the live bottom. (A row can exist without being on screen.)
    pub fn row_exists(&self, abs: AbsRow) -> bool {
        let b = self.active();
        abs >= b.abs_of(0) && abs < b.abs_end()
    }

    /// The cell at absolute `(row, col)`, blank when the row no longer exists. This is
    /// the selection's view of the grid: it reads the content the user picked, whatever
    /// the viewport has since done.
    pub(super) fn abs_cell(&self, row: AbsRow, col: usize) -> Cell {
        self.active()
            .abs_row(row)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// Combining marks at absolute `(row, col)`.
    pub(super) fn abs_marks(&self, row: AbsRow, col: usize) -> impl Iterator<Item = char> + '_ {
        self.active()
            .abs_row(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
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

    /// Combining marks at display `(row, col)` honouring the scroll offset, in
    /// arrival order; empty if none.
    pub fn view_marks(&self, row: usize, col: usize) -> impl Iterator<Item = char> + '_ {
        self.view_line(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
    }

    /// Whether display `(row, col)` carries any combining mark, honouring the scroll
    /// offset. The ink test uses this instead of `view_marks(..).next()` so a caller
    /// can ask the yes/no without starting an iterator it will not drain.
    pub fn view_has_marks(&self, row: usize, col: usize) -> bool {
        self.view_line(row).is_some_and(|r| r.has_marks(col))
    }

    /// The row shown at display `row` honouring the scroll offset: the live row when
    /// pinned to the bottom, else the scrollback row scrolled into view.
    pub(super) fn view_line(&self, row: usize) -> Option<&Row> {
        let off = self.view_offset();
        if off == 0 {
            self.active().line(row)
        } else {
            self.active().view_row(row, off)
        }
    }

    /// The text of a linear selection, `a`..`b` inclusive in absolute `(row, col)`
    /// cells (either order). Rows join with `\n`, except a soft-wrapped row joins with
    /// nothing (the two rows are one logical line, so a selection across a wrap copies
    /// as unbroken text). Trailing blanks on a row are dropped, wide spacers skipped,
    /// combining marks kept.
    ///
    /// Absolute rows, so it copies the lines the user picked no matter where the
    /// viewport has drifted to since — including lines that have scrolled out of sight
    /// entirely. Rows that have aged out of history read blank.
    pub fn selection_text(&self, a: (AbsRow, usize), b: (AbsRow, usize)) -> String {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        let cols = self.dimensions().0;
        let last_col = cols.saturating_sub(1);
        let buf = self.active();
        // Clip to the rows that still exist: the head of an old selection may have aged
        // out of history, and its tail cannot run past the live bottom.
        let start_row = start.0.max(buf.abs_of(0));
        let Some(end_row) = buf.abs_end().prev().map(|last| end.0.min(last)) else {
            return String::new();
        };
        let mut out = String::new();
        let mut row = start_row;
        while row <= end_row {
            let first = if row == start.0 { start.1 } else { 0 };
            let last = if row == end.0 { end.1 } else { last_col };
            // Append this row straight onto the shared buffer rather than into a fresh
            // per-row `String`, then drop its trailing spaces in place — no further back
            // than where the row began, so an earlier row's content is untouched.
            let row_start = out.len();
            for col in first..=last.min(last_col) {
                let cell = self.abs_cell(row, col);
                if cell.is_wide_spacer() {
                    continue;
                }
                out.push(cell.rune);
                out.extend(self.abs_marks(row, col));
            }
            while out.len() > row_start && out.ends_with(' ') {
                out.pop();
            }
            if row != end_row {
                // A soft wrap continues the same logical line: no newline.
                if !self.wraps(row) {
                    out.push('\n');
                }
            }
            let Some(next) = row.next() else { break };
            row = next;
        }
        out
    }

    /// Whether absolute `row` soft-wrapped into the next one, so the two are one
    /// logical line. False for a row that has aged out of history.
    pub(super) fn wraps(&self, row: AbsRow) -> bool {
        self.active().abs_row(row).is_some_and(|r| r.wrapped)
    }

    /// Whether the line on the live screen at display `row` soft-wrapped into the next.
    pub(super) fn row_wraps(&self, row: usize) -> bool {
        self.active().line(row).is_some_and(|r| r.wrapped)
    }

    /// The inclusive cell range of the word at display `(row, col)`, for a
    /// double-click selection. A word is a maximal run of non-boundary runes (see
    /// [`is_word_boundary`]); a wide glyph's spacer continues its leader's word.
    /// Clicking a boundary cell (whitespace, a bracket) selects just that cell.
    pub fn word_at(&self, row: AbsRow, col: usize) -> ((AbsRow, usize), (AbsRow, usize)) {
        let cols = self.dimensions().0;
        if col >= cols {
            return ((row, col), (row, col));
        }
        let is_word = |c: usize| {
            let cell = self.abs_cell(row, c);
            cell.is_wide_spacer() || !is_word_boundary(cell.rune)
        };
        if !is_word(col) {
            return ((row, col), (row, col));
        }
        let mut start = col;
        while start > 0 && is_word(start - 1) {
            start -= 1;
        }
        let mut end = col;
        while end + 1 < cols && is_word(end + 1) {
            end += 1;
        }
        ((row, start), (row, end))
    }

    /// The inclusive cell range of the whole logical line at absolute `row`, for a
    /// triple-click selection: it spans every row a soft wrap joined (the `WRAPPED` flag
    /// on a row's last cell continues it into the next), edge to edge.
    ///
    /// The walk runs over the stream, not the visible band, so triple-clicking the first
    /// row on screen still takes the part of the line that wrapped in from above it.
    pub fn line_at(&self, row: AbsRow) -> ((AbsRow, usize), (AbsRow, usize)) {
        let cols = self.dimensions().0;
        let last_col = cols.saturating_sub(1);
        let b = self.active();
        let (first_row, end_row) = (b.abs_of(0), b.abs_end());
        let mut start = row;
        while let Some(above) = start.prev() {
            if start <= first_row || !self.wraps(above) {
                break;
            }
            start = above;
        }
        let mut end = row;
        while let Some(below) = end.next() {
            if below >= end_row || !self.wraps(end) {
                break;
            }
            end = below;
        }
        ((start, 0), (end, last_col))
    }

    /// The display rows the logical line at display `row` spans, clipped to the visible
    /// band. A hyperlink is a *hover* concern — re-probed on every pointer motion, never
    /// stored across output, and underlined row by row on screen — so unlike a selection
    /// it works in display rows and stops at the edge of what is drawn, even where the
    /// logical line runs on past it. The clamps are safe because `row` is on screen and
    /// the line contains it: a bound that does not resolve is off the band on that side.
    pub(super) fn line_rows_on_screen(&self, row: usize) -> (usize, usize) {
        let rows = self.dimensions().1;
        let ((first, _), (last, _)) = self.line_at(self.abs_row(row));
        (
            self.display_row(first).unwrap_or(0),
            self.display_row(last).unwrap_or(rows.saturating_sub(1)),
        )
    }

    /// The hyperlink under display `(row, col)`: the inclusive cell range its text
    /// occupies, in reading order, with the URL left in `probe` for
    /// [`LinkProbe::url`] to read. `None` when that cell is not inside one.
    ///
    /// The scan runs over the whole *logical* line (soft wraps joined, the same walk
    /// [`Self::line_at`] does), because a URL printed near the right margin routinely
    /// continues on the next display row, and half a URL is not a URL. So the range
    /// returned is contiguous in reading order and may span rows — the first row from
    /// its start column to the right edge, then whole rows, then the last row to its
    /// end column — which is exactly the geometry the painter already highlights for a
    /// selection.
    ///
    /// Wide glyphs are handled at both ends: probing a spacer probes its leader (the
    /// right half of a character is the character), and a range ending on a leader is
    /// widened over its spacer, so the underline never stops half a glyph short.
    ///
    /// An OSC 8 anchor short-circuits all of that. When the cell carries a [`LinkId`],
    /// the child has *told* us these cells are this link, so there is nothing to infer:
    /// the extent is the run of cells sharing the id and the URL is the one it interned.
    /// The text scan is only ever the fallback for output that was never marked up.
    pub fn link_at(
        &self,
        row: usize,
        col: usize,
        probe: &mut LinkProbe,
    ) -> Option<((usize, usize), (usize, usize))> {
        let (cols, _) = self.dimensions();
        if col >= cols {
            return None;
        }
        // The right half of a wide glyph belongs to the glyph on its left.
        let col = if self.view_cell(row, col).is_wide_spacer() {
            col.saturating_sub(1)
        } else {
            col
        };

        probe.text.clear();
        probe.runes.clear();
        probe.url.clear();

        let id = self.view_cell(row, col).link;
        if id.is_set() {
            probe.url.push_str(self.links.url(id)?);
            return Some(self.anchor_span(row, col, id));
        }
        // Byte offset of the probed cell's rune, recorded as that rune is pushed.
        let mut at = None;
        let (first, last) = self.line_rows_on_screen(row);
        for r in first..=last {
            for c in 0..cols {
                let cell = self.view_cell(r, c);
                if cell.is_wide_spacer() {
                    continue;
                }
                if (r, c) == (row, col) {
                    at = Some(probe.text.len());
                }
                probe.runes.push((probe.text.len(), (r, c)));
                probe.text.push(cell.rune);
                probe.text.extend(self.view_marks(r, c));
            }
        }

        let found = crate::platform::link::find_at(&probe.text, at?)?;
        // `found.end` is exclusive, so the last rune inside the link is the one
        // covering the byte before it.
        let start = probe.cell_at(found.start)?;
        let mut end = probe.cell_at(found.end.checked_sub(1)?)?;
        probe.url.push_str(probe.text.get(found)?);
        if self.view_cell(end.0, end.1).is_wide_leader() && end.1 + 1 < cols {
            end.1 += 1;
        }
        Some((start, end))
    }

    /// The extent of the OSC 8 anchor `id` around `(row, col)`: the unbroken run of
    /// cells carrying that same id, walked in reading order across the whole logical
    /// line, so an anchor that soft-wraps at the margin underlines whole.
    ///
    /// The run — rather than "every cell with this id" — is what makes two anchors that
    /// happen to share a URL (and so an id) stay two links: the plain cells between
    /// them break the run. It is also why an ICH-opened gap splits an anchor, which is
    /// the honest reading once the label is no longer contiguous.
    ///
    /// ```text
    ///        col 0                     cols-1
    ///   row  ┌───────────────────────────────┐
    ///    r   │ $ ls  [docs····················│  ← id 7 starts, hits the margin
    ///   r+1  │····/spec.md]  README.md       │  ← id 7 continues, then plain cells
    ///        └───────────────────────────────┘
    ///          the span runs start → end in reading order, exactly the geometry
    ///          `push_hover_rule` already knows how to underline row by row.
    /// ```
    pub(super) fn anchor_span(
        &self,
        row: usize,
        col: usize,
        id: LinkId,
    ) -> ((usize, usize), (usize, usize)) {
        let (cols, _) = self.dimensions();
        let (first, last) = self.line_rows_on_screen(row);
        // Walk the logical line as one flat sequence of cells, so crossing a soft wrap
        // needs no special case: cell `n` is at (first + n / cols, n % cols).
        let at = |n: usize| self.view_cell(first + n / cols, n % cols);
        let cell_of = |n: usize| (first + n / cols, n % cols);
        let total = (last - first + 1) * cols;
        let here = (row - first) * cols + col;

        let mut start = here;
        while start > 0 && at(start - 1).link == id {
            start -= 1;
        }
        let mut end = here;
        while end + 1 < total && at(end + 1).link == id {
            end += 1;
        }
        (cell_of(start), cell_of(end))
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
            s.extend(r.marks(col));
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
    ///
    /// A row that soft-wrapped into the next is flagged `wrapped` beside its text.
    /// That is not cosmetic: the flag is what a copy across the wrap depends on, and
    /// it is invisible in the grid text (both a soft wrap and a hard newline just end
    /// the row), so without it a regression could not be seen here at all.
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
            let cont = if self.row_wraps(r) { " wrapped" } else { "" };
            let _ = writeln!(out, "|{}|{cont}", self.row_string(r));
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
        // What we said back. Half of what a terminal *does* is answer questions, and none
        // of it shows up on the grid — a wrong reply to `OSC 11` leaves the screen
        // pixel-identical and the editor's colours wrong. So a golden test that cannot
        // see the replies cannot see half the terminal. Only emitted when there are any,
        // so a fixture that asks nothing looks exactly as it did before.
        if !self.responses.is_empty() {
            out.push_str("replies:\n");
            for reply in split_replies(&self.responses) {
                let _ = writeln!(out, "{}", escape_reply(reply));
            }
        }
        out
    }
}

/// Whether a rune bounds a double-click word selection: whitespace, or one of the
/// brackets/quotes that fence a token. The set matches the common terminal default
/// (wezterm/xterm), so double-clicking selects a path or URL whole but stops at a
/// delimiter. A blank cell's rune is a space, so it bounds a word too.
pub(super) fn is_word_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']' | '(' | ')' | '"' | '\'' | '`')
}

/// Split the reply buffer into individual replies, so a snapshot shows one per line.
///
/// Every reply we emit begins with `ESC`, which makes the boundary almost unambiguous —
/// the exception being the `ESC \` that *terminates* a DCS reply, which opens nothing and
/// belongs to the reply in front of it.
pub(super) fn split_replies(responses: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &byte) in responses.iter().enumerate() {
        let terminator = responses.get(i + 1) == Some(&b'\\');
        if byte == 0x1b && i > start && !terminator {
            if let Some(reply) = responses.get(start..i) {
                out.push(reply);
            }
            start = i;
        }
    }
    if let Some(reply) = responses.get(start..) {
        if !reply.is_empty() {
            out.push(reply);
        }
    }
    out
}

/// One reply, readable: the control bytes named rather than printed, so a golden diff
/// reads like the sequence it is instead of a hex dump.
pub(super) fn escape_reply(reply: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for &byte in reply {
        match byte {
            0x1b => out.push_str("<ESC>"),
            0x07 => out.push_str("<BEL>"),
            0x20..=0x7e => out.push(char::from(byte)),
            _ => {
                let _ = write!(out, "<{byte:#04x}>");
            }
        }
    }
    out
}
