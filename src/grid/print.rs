//! Putting characters in cells: the print paths, wide glyphs, and grapheme clustering.
//!
//! Three roads to the same place, fastest first, and they must agree exactly — the
//! `bulk_ascii_matches_per_char_under_fuzz` and split-invariance tests pin that:
//!
//! ```text
//!   print_ascii_run   a run of 0x20..=0x7e, one row-fill per row         (the hot path)
//!   print_text_run    decoded scalars, width-classified once, in batches
//!   print / print_width   one scalar, the semantic oracle for every awkward case
//! ```
//!
//! The awkward cases are why the scalar path stays the oracle: a wide glyph with one
//! column left (wrap, or back up under `DECAWM` off), a combining mark attaching to the
//! cell behind the cursor, and a grapheme cluster growing a cell from one column to two
//! *after* its base was already placed. Every path ends at
//! [`Buffer::advance_cursor`](crate::grid::buffer), which states xterm's last-column rule once.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::*;

impl Screen {
    /// Print one scalar value at the cursor, applying wide-character and wrap
    /// semantics. Combining marks (width 0) attach to the preceding cell.
    pub fn print(&mut self, c: char) {
        // Under `?2027` a scalar that continues the cluster in the cell behind us joins
        // it, whatever its own width — that is the whole difference between counting
        // scalars and counting characters. The rocket of an astronaut is two columns wide
        // on its own and no columns wide as part of the astronaut.
        if self.grapheme_clustering && self.extend_cluster(c) {
            return;
        }
        self.print_width(c, usize::from(width(c)));
    }

    /// Place one scalar whose width is already known. Decoded batches use this
    /// for complex fallbacks so width-table searches are never repeated.
    fn print_width(&mut self, c: char, cw: usize) {
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

        let pen = self.packed_pen();
        let insert = self.insert_mode;
        let (row, col) = {
            let cur = self.active().cursor;
            (cur.row, cur.col)
        };
        if insert {
            let blank = self.blank_cell();
            self.active_mut().insert_blanks(row, col, cw, blank);
        }
        // A wide glyph needs two columns and there is exactly one screen narrow enough to
        // deny it: a single-column grid, where the wrap has nowhere to go and the back-up
        // for no-autowrap lands on the same cell. Marking it a leader there would set the
        // one invariant this grid keeps — a leader always has its spacer — against a
        // spacer that cannot exist. It goes in as an ordinary cell instead: clipped, and
        // structurally sound.
        let fits_wide = cw == 2 && col + 1 < cols;
        let width = if fits_wide {
            CellWidth::Leader
        } else {
            CellWidth::Narrow
        };
        let leader = PackedCell::new(c, width, pen.style, pen.link);
        {
            let b = self.active_mut();
            b.write_cell(row, col, leader);
            if fits_wide {
                // The spacer carries the link too, so the run the hover probe walks
                // never breaks in the middle of a wide glyph.
                let spacer = PackedCell::new(' ', CellWidth::Spacer, pen.style, pen.link);
                b.write_cell(row, col + 1, spacer);
            }
        }

        let autowrap = self.autowrap;
        let b = self.active_mut();
        let end_col = b.cursor.col.saturating_add(cw);
        b.advance_cursor(end_col, autowrap);
        self.last_printed = Some(c);
        if self.grapheme_clustering {
            self.set_cluster_anchor(row, col);
        }
    }

    /// Remember the cell a continuing scalar would join.
    ///
    /// `#[inline(never)]`, and that is not a style choice. `print` is the per-character hot
    /// path, and inlining this into it grew the function past whatever threshold the
    /// compiler lays code out around: the escape-heavy benchmark lost 10% to it *while the
    /// mode was switched off and this never ran once*. Code that cannot execute still costs
    /// what it displaces.
    #[inline(never)]
    fn set_cluster_anchor(&mut self, row: usize, col: usize) {
        let cursor = self.active().cursor;
        self.cluster_anchor = Some(ClusterAnchor {
            row,
            col,
            after: (cursor.row, cursor.col),
        });
    }

    /// Bulk-write a run of printable ASCII (each width 1) at the cursor. Equivalent
    /// to calling [`print`](Self::print) once per byte, but hoisting the per-char
    /// width lookup, wrap check, and cursor math out of the inner loop: a screenful
    /// of plain text becomes a few row-fills instead of thousands of single prints.
    ///
    /// Caller guarantees (upheld by `<Screen as Perform>::print_ascii`): every byte
    /// is `0x20..=0x7e`, the active charset is identity ASCII, and insert mode is
    /// off, so no per-char glyph mapping, wide-cell, or shift handling is needed.
    /// `write_cell` still runs per cell, so wide-pair and combining-mark cleanup at
    /// the run's edges is preserved.
    pub(super) fn print_ascii_run(&mut self, bytes: &[u8]) {
        let cols = self.active().cols;
        if cols == 0 {
            return;
        }
        let pen = self.packed_pen();
        let autowrap = self.autowrap;

        let mut rest = bytes;
        while !rest.is_empty() {
            // Take any deferred wrap the previous cell left before placing more.
            if self.active().cursor.pending_wrap {
                self.wrap_line();
            }
            let (row, start_col) = {
                let c = self.active().cursor;
                (c.row, c.col)
            };
            // Fill to the end of the row, or until the run ends. `room >= 1`: the
            // cursor column is always `< cols`, and a wrap just reset it to 0.
            let room = cols - start_col;
            let take = room.min(rest.len());
            let (run, tail) = rest.split_at(take);
            self.active_mut().fill_ascii_run(row, start_col, run, pen);
            rest = tail;
            self.active_mut().advance_cursor(start_col + take, autowrap);
        }
        // REP repeats the last character printed, and the bulk path prints too.
        self.last_printed = bytes.last().map(|&b| char::from(b));
    }

    /// Consume already-decoded text in bounded pieces, classifying every scalar
    /// once and writing maximal positive-width prefixes a row at a time. Width-0
    /// marks and awkward right-edge wide glyphs go through [`Self::print`], which
    /// remains the semantic oracle for complex placement.
    ///
    /// The scratch widths are fixed and initialized: no allocation and no
    /// uninitialized-memory `unsafe`. Its capacity is only a work quantum;
    /// callers may supply an arbitrarily long slice without changing semantics.
    pub(super) fn print_text_run(&mut self, chars: &[char]) {
        const BATCH: usize = 128;

        let mut rest = chars;
        let mut widths = [0u8; BATCH];
        while !rest.is_empty() {
            let take = rest.len().min(BATCH);
            let Some(chunk) = rest.get(..take) else {
                return;
            };
            let Some(chunk_widths) = widths.get_mut(..take) else {
                return;
            };
            let Some(&first) = chunk.first() else {
                return;
            };
            let first_width = width(first);
            if first_width == 0 {
                self.print_scalar_partitioned_run(chunk);
                rest = rest.get(take..).unwrap_or_default();
                continue;
            }
            if let Some(slot) = chunk_widths.first_mut() {
                *slot = first_width;
            }
            for (slot, &c) in chunk_widths.iter_mut().skip(1).zip(chunk.iter().skip(1)) {
                *slot = width(c);
            }
            let zero_width = chunk_widths
                .iter()
                .filter(|&&cell_width| cell_width == 0)
                .count();
            if zero_width.saturating_mul(2) >= chunk.len() {
                self.print_scalar_partitioned_run(chunk);
            } else {
                self.print_classified_run(chunk, chunk_widths);
            }
            rest = rest.get(take..).unwrap_or_default();
        }
    }

    /// Preserve the established shape for a run dominated by complex marks:
    /// non-ASCII scalars use the scalar oracle, while each printable ASCII scalar
    /// retains the same byte-style callback the parser used before batching.
    /// Mark-heavy text normally alternates one base with one or more marks, so
    /// collecting spans here only adds a second scan and a scratch copy.
    fn print_scalar_partitioned_run(&mut self, chars: &[char]) {
        for &c in chars {
            if c.is_ascii() {
                let byte = u8::try_from(u32::from(c)).unwrap_or(b'?');
                self.print_ascii_run(std::slice::from_ref(&byte));
            } else {
                self.print_width(c, usize::from(width(c)));
            }
        }
    }

    fn print_classified_run(&mut self, chars: &[char], widths: &[u8]) {
        let mut at = 0usize;
        while let (Some(&c), Some(&classified)) = (chars.get(at), widths.get(at)) {
            if classified == 0 {
                self.print_width(c, 0);
                at += 1;
                continue;
            }
            let cols = self.active().cols;
            if cols == 0 {
                return;
            }
            if self.active().cursor.pending_wrap {
                self.wrap_line();
            }
            let (row, start_col) = {
                let cursor = self.active().cursor;
                (cursor.row, cursor.col)
            };
            let room = cols.saturating_sub(start_col);
            let mut end = at;
            let mut columns = 0usize;
            while let Some(&cell_width) = widths.get(end) {
                if cell_width == 0 {
                    break;
                }
                let next = columns.saturating_add(usize::from(cell_width));
                if next > room {
                    break;
                }
                columns = next;
                end += 1;
            }

            // A two-cell glyph with one column remaining needs the full scalar
            // edge policy (wrap first, or back up under no-autowrap).
            if end == at {
                self.print_width(c, usize::from(classified));
                at += 1;
                continue;
            }

            let Some(segment) = chars.get(at..end) else {
                return;
            };
            let Some(segment_widths) = widths.get(at..end) else {
                return;
            };
            let pen = self.packed_pen();
            self.active_mut()
                .fill_text_run(row, start_col, segment, segment_widths, pen);

            let autowrap = self.autowrap;
            self.active_mut()
                .advance_cursor(start_col.saturating_add(columns), autowrap);
            self.last_printed = segment.last().copied();
            at = end;
        }
    }

    /// Try to join `c` onto the grapheme cluster already in the cell behind the cursor.
    /// Returns whether it did, in which case the caller has nothing left to do.
    ///
    /// This is where `?2027` actually lives. Everything else about the mode is
    /// bookkeeping; the decision is here, and it is made one scalar at a time because that
    /// is how a terminal receives them. There is no lookahead: when the woman arrives we
    /// do not know whether a ZWJ and a rocket are coming, so she is printed as herself,
    /// and the ZWJ and the rocket *join her* when they turn up. A cluster is never held
    /// back waiting to see whether it is finished — a program that writes half an emoji and
    /// crashes must still leave half an emoji on screen.
    ///
    /// The anchor carries where the cursor was when the cluster last grew, and that one
    /// check is the *whole* invalidation rule. If the cursor is not where the cluster left
    /// it, the cluster is over.
    ///
    /// Nothing else needs to say so. A control byte moves the cursor; a run of ASCII moves
    /// the cursor; a cursor motion sequence moves the cursor — so every one of them breaks
    /// the anchor by construction, and none of them needs a hook here. The hooks were
    /// written first and then measured: they sat in `execute` and in the bulk-ASCII print
    /// run, which are the two hottest functions in the terminal, and cost the
    /// escape-heavy benchmark 7.5% *while the mode was switched off*. They were also
    /// redundant. Both facts point the same way.
    #[inline(never)]
    fn extend_cluster(&mut self, c: char) -> bool {
        if !self.grapheme_clustering {
            return false;
        }
        let cursor = self.active().cursor;
        let anchor = match self.cluster_anchor {
            Some(a) if a.after == (cursor.row, cursor.col) => Some(a),
            // The cursor moved, or there is nothing to continue: whatever run of text we
            // were in has ended, and this scalar starts a fresh cluster.
            _ => {
                self.cluster.reset();
                self.cluster_anchor = None;
                None
            }
        };
        // The machine is fed every scalar, whether it joins or starts something.
        let joins = !self.cluster.breaks_before(c);
        let Some(anchor) = anchor else {
            return false;
        };
        if !joins {
            return false;
        }
        self.grow_cluster(anchor, c);
        true
    }

    /// Add `c` to the cluster at `anchor`, and widen the cell if the cluster has grown
    /// from one column to two.
    ///
    /// The widening is the fiddly half, and it is unavoidable: a cluster's width is not
    /// known until it ends. `☀` is one column, and `☀️` — the very same character with an
    /// emoji presentation selector after it — is two. The base was already placed in a
    /// narrow cell by the time the selector arrived, so the cell has to grow under it. The
    /// same goes for a flag: a regional indicator is narrow on its own and a pair of them
    /// is an emoji.
    ///
    /// At the right margin there is nowhere to grow into, and the cluster stays narrow
    /// rather than wrapping: a character that has already been drawn cannot be moved to the
    /// next line without the cursor arithmetic on the far end of the pty disagreeing about
    /// where everything after it went.
    #[inline(never)]
    fn grow_cluster(&mut self, anchor: ClusterAnchor, c: char) {
        let (row, col) = (anchor.row, anchor.col);
        let cols = self.active().cols;
        let was = self.cluster_text(row, col);
        let before = crate::width::cluster_width(&was);

        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.add_mark(col, c);
        }
        let now = self.cluster_text(row, col);
        let after = crate::width::cluster_width(&now);

        // One column to two: promote the cell to a wide leader and lay a spacer beside it,
        // the same shape a wide character has had all along, so nothing downstream has to
        // learn a new one.
        if before < 2 && after == 2 && col + 1 < cols {
            let leader = self.active().cell(row, col);
            let spacer = PackedCell::new(' ', CellWidth::Spacer, leader.style_id(), leader.link);
            let b = self.active_mut();
            // The column being grown into is not empty just because the cluster was
            // narrow: it may hold the leader of the *next* wide pair, whose spacer would
            // be stranded at `col + 2` by a raw write. Go through `write_cell` so that
            // pair is broken the way any other overwriting print breaks one, and so the
            // marks the displaced cell carried go with it. This runs first: it can blank
            // `col` when the neighbour is a spacer, and the promotion below writes the
            // cluster back over it.
            b.write_cell(row, col + 1, spacer);
            b.set_raw(row, col, leader.with_width(CellWidth::Leader));
            b.cursor.col = (col + 2).min(cols.saturating_sub(1));
            b.cursor.pending_wrap = col + 2 >= cols;
        }
        let cursor = self.active().cursor;
        self.cluster_anchor = Some(ClusterAnchor {
            row,
            col,
            after: (cursor.row, cursor.col),
        });
        self.last_printed = Some(c);
    }

    /// The full text of the cluster in a cell: its base rune and every mark that has
    /// joined it. This is what both the width rule and the shaper are handed, so the
    /// number of columns it takes and the glyph drawn in them come from the same string.
    pub(super) fn cluster_text(&self, row: usize, col: usize) -> String {
        let mut out = String::new();
        let b = self.active();
        out.push(b.cell(row, col).rune());
        if let Some(r) = b.line(row) {
            out.extend(r.marks(col));
        }
        out
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

    /// The soft wrap: link the row being left to the next one, then move to the
    /// start of that next line (scrolling if at the bottom of the region).
    fn wrap_line(&mut self) {
        let row = self.active().cursor.row;
        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.wrapped = true;
        }
        self.active_mut().cursor.pending_wrap = false;
        self.line_feed();
        self.carriage_return();
    }
}
