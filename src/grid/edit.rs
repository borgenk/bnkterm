//! Editing the grid: the operations an escape sequence asks for, and the modes that
//! change what they mean.
//!
//! This is the bulk of the VT semantics — the C0 controls, cursor motion, the erases,
//! the insert/delete shifts, the scrolls, DECSC/DECRC, SGR, the mode set/reset, and
//! resize. It knows nothing about bytes (that is `perform`) and nothing about storage
//! mechanics (that is `buffer`); it turns one named operation into calls on the active
//! buffer.
//!
//! The recurring subtlety is the **deferred wrap**. `pending_wrap` is a claim that the
//! *next* glyph belongs on the next row on behalf of text already in the last column, so
//! every operator that rewrites the cells around the cursor has to cancel it — ICH, DCH,
//! ECH, IL and DL all do, and each one says so where it does it.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::buffer::{Charsets, Cursor, Pen, Saved, Scrolled};
use super::*;

impl Screen {
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
        let mut scrolled = Scrolled::default();
        if b.cursor.row == b.scroll_bottom {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            scrolled = b.scroll_up_range(top, bottom, 1, blank, top == 0);
        } else if b.cursor.row + 1 < b.rows {
            b.cursor.row += 1;
        }
        self.after_scroll(scrolled);
    }

    /// RI: move up one row, scrolling the region down at the top margin.
    pub fn reverse_index(&mut self) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        let mut scrolled = Scrolled::default();
        if b.cursor.row == b.scroll_top {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            scrolled = b.scroll_down_range(top, bottom, 1, blank);
        } else if b.cursor.row > 0 {
            b.cursor.row -= 1;
        }
        self.after_scroll(scrolled);
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

    /// REP (`CSI Ps b`): print the last character again, `n` times.
    ///
    /// ncurses reaches for this to paint a run of one character without sending it a
    /// thousand times, so it turns up in real output more than its obscurity suggests.
    /// It repeats the last *printed* character and nothing else: a control byte, an escape
    /// sequence, or a fresh line all leave nothing to repeat, and then it does nothing —
    /// which is the behaviour a program relies on when it uses REP right after a newline
    /// and expects no output rather than a screenful of the last thing on the line above.
    pub fn repeat_last(&mut self, n: usize) {
        let Some(c) = self.last_printed else {
            return;
        };
        for _ in 0..n {
            self.print(c);
        }
    }

    /// CBT: move back `n` tab stops, stopping at column 0. The mirror of [`tab`](Self::tab).
    pub fn back_tab(&mut self, n: usize) {
        let b = self.active_mut();
        for _ in 0..n.max(1) {
            let Some(mut c) = b.cursor.col.checked_sub(1) else {
                break;
            };
            while c > 0 && !b.tabs.get(c).copied().unwrap_or(false) {
                c -= 1;
            }
            b.cursor.col = c;
        }
        b.cursor.pending_wrap = false;
    }

    /// TBC: clear the tab stop at the cursor (mode 0, the default) or every stop
    /// (mode 3).
    ///
    /// Every other parameter is ignored, which is the whole behavior and not a
    /// shortcut. ECMA-48 also defines 1, 2, 4 and 5 in terms of *line* tab stops, which
    /// no terminal in use has; xterm implements 0 and 3 and drops the rest, and vttest's
    /// tab screen checks exactly that by sending `CSI 1 g` and `CSI 2 g` at a live tab
    /// stop and requiring the stop to survive. Treating an unknown parameter as 0 would
    /// silently delete it.
    pub fn clear_tab_stop(&mut self, mode: u16) {
        let b = self.active_mut();
        match mode {
            0 => {
                let col = b.cursor.col;
                if let Some(stop) = b.tabs.get_mut(col) {
                    *stop = false;
                }
            }
            3 => b.tabs.iter_mut().for_each(|t| *t = false),
            _ => {}
        }
    }

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
        b.cursor.row = b.origin_row(row, origin);
        b.cursor.col = col.min(b.cols.saturating_sub(1));
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
        b.cursor.row = b.origin_row(row, origin);
        b.cursor.pending_wrap = false;
    }

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
                    b.clear_history();
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
        if mode == 2 || mode == 3 {
            // The display the user was looking at (and may have selected) is gone.
            self.break_row_identity();
        }
        if mode == 3 {
            // The history the view was scrolled into no longer exists, and an offset
            // pointing past the end of an empty ring is not a view of anything.
            self.view_offset = 0;
        }
    }

    /// ECH: erase `n` cells from the cursor, without shifting the rest.
    /// ECH: blank `n` cells from the cursor without shifting anything.
    ///
    /// Cancels a deferred wrap, as every operator that rewrites the cells around the
    /// cursor must: the flag is a claim that the next glyph belongs on the next row *on
    /// behalf of the text already here*, and this just erased that text. Honouring it
    /// afterwards would wrap for a glyph that no longer exists. Unconditional, and
    /// before any bounds check refuses the erase itself — a rule about the cursor does
    /// not wait on whether the cells were in range.
    pub fn erase_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.clear_line_range(row, col, col + n.max(1), blank);
    }

    /// ICH: insert `n` blanks at the cursor, shifting the rest of the line right.
    pub fn insert_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        // Cancels a deferred wrap: ICH shifts the text the flag was deferring for out
        // from under the cursor (see `erase_chars`).
        b.cursor.pending_wrap = false;
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.insert_blanks(row, col, n.max(1), blank);
    }

    /// DCH: delete `n` cells at the cursor, shifting the rest of the line left.
    pub fn delete_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        // Cancels a deferred wrap, as ICH does and for the same reason.
        b.cursor.pending_wrap = false;
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
        let scrolled = b.scroll_down_range(top, bottom, n.max(1), blank);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
        self.after_scroll(scrolled);
    }

    /// DL: delete `n` lines at the cursor row, within the scroll region.
    ///
    /// The deleted rows are discarded, never retained, even with the cursor at row 0.
    /// This is a deliberate pick in a genuine split: xterm and alacritty push them into
    /// history (xterm's `DeleteLine` saves when `cur_row == 0`), ghostty does not. DL is
    /// an editing command, not a scroll — the application is *removing* those lines, and
    /// a shell redrawing a multi-line prompt at the top of the screen should not dribble
    /// prompt fragments into the scrollback. If a real program is ever found to depend on
    /// the xterm behavior, this is the line to change.
    pub fn delete_lines(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        if b.cursor.row < b.scroll_top || b.cursor.row > b.scroll_bottom {
            return;
        }
        let (top, bottom) = (b.cursor.row, b.scroll_bottom);
        let scrolled = b.scroll_up_range(top, bottom, n.max(1), blank, false);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
        self.after_scroll(scrolled);
    }

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

    /// SU: scroll the region up `n` lines (content moves up, blanks feed in at the
    /// bottom). Rows leaving the *top of the screen* enter history, exactly as they do
    /// for a line feed at the bottom margin: xterm retains them whenever the top margin
    /// is 0 (`xtermScroll`'s `top_marg == 0`), and alacritty and ghostty agree. A region
    /// with a top margin discards instead — those rows never touch the top of the
    /// screen, so they were never history.
    pub fn scroll_up(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        let scrolled = b.scroll_up_range(top, bottom, n.max(1), blank, top == 0);
        self.after_scroll(scrolled);
    }

    /// SD: scroll the region down `n` lines (content moves down).
    pub fn scroll_down(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        let scrolled = b.scroll_down_range(top, bottom, n.max(1), blank);
        self.after_scroll(scrolled);
    }

    /// DECSC: save the cursor, pen, origin mode, and character set shift state.
    ///
    /// The charsets are part of the saved state by the VT spec's own list ("cursor
    /// position, graphic rendition, character set shift state, state of wrap flag,
    /// state of origin mode, state of selective erase"), and leaving them out is
    /// visible immediately: a shell that saves the cursor, switches to the line-drawing
    /// set to paint a frame, then restores, is telling the terminal to put the ASCII
    /// mapping back. Without that, every letter it prints afterwards comes out as box
    /// glyphs.
    pub fn save_cursor(&mut self) {
        let (pen, origin, charsets) = (self.pen, self.origin_mode, self.charsets());
        let b = self.active_mut();
        b.saved = Some(Saved {
            cursor: b.cursor,
            pen,
            origin,
            charsets,
        });
    }

    /// DECRC: restore what DECSC saved, or home the cursor if nothing was saved.
    pub fn restore_cursor(&mut self) {
        let saved = self.active().saved;
        match saved {
            Some(s) => {
                self.pen = s.pen;
                self.origin_mode = s.origin;
                self.set_charsets(s.charsets);
                let b = self.active_mut();
                b.cursor = s.cursor;
                b.cursor.row = b.cursor.row.min(b.rows - 1);
                b.cursor.col = b.cursor.col.min(b.cols - 1);
            }
            None => self.move_to(0, 0),
        }
    }

    pub(super) fn charsets(&self) -> Charsets {
        Charsets {
            g0: self.g0,
            g1: self.g1,
            gl_is_g1: self.gl_is_g1,
        }
    }

    pub(super) fn set_charsets(&mut self, c: Charsets) {
        self.g0 = c.g0;
        self.g1 = c.g1;
        self.gl_is_g1 = c.gl_is_g1;
    }

    /// Apply an SGR sequence, updating the pen. An empty parameter list is a reset
    /// (SGR 0). Handles ANSI-16, bright, 256 (`38;5;n` and `38:5:n`), and truecolor
    /// (`38;2;r;g;b` and `38:2::r:g:b`) for both foreground (38) and background (48).
    ///
    /// The rule for everything we do not implement is **consume, then ignore**, and it
    /// is load-bearing rather than pedantic. An extended-colour introducer owns the
    /// parameters that follow it, so failing to consume `58;2;255;0;0` would not merely
    /// skip an underline colour: the `2` would be read as *dim*, the `0` as *reset*, and
    /// the rest of the line would come out in the wrong style. Skipping a parameter we
    /// do not understand is only safe once we know how many belong to it.
    pub fn sgr(&mut self, params: &Params) {
        if params.is_empty() {
            self.pen.reset_rendition();
            return;
        }
        let mut i = 0;
        while i < params.len() {
            let Some(param) = params.get(i) else { break };
            let p = param.first().copied().unwrap_or(0);
            match p {
                0 => self.pen.reset_rendition(),
                1 => self.pen.attrs.insert(Attrs::BOLD),
                2 => self.pen.attrs.insert(Attrs::DIM),
                3 => self.pen.attrs.insert(Attrs::ITALIC),
                // `4` is underline; `4:n` picks a style (1 single, 2 double, 3 curly,
                // 4 dotted, 5 dashed) and `4:0` turns it off. We draw only a solid
                // underline for now, so a style other than 0 is an underline of some
                // kind and that is what the cell records. Storing *which* style needs a
                // wider `Attrs` and a cell field; until then a curly underline shows up
                // as a straight one, which is the right kind of wrong.
                // `4` underlines; `4:n` says with what shape. `4:0` is the one
                // sub-parameter of 4 that is not an underline at all.
                4 => match param.get(1).copied() {
                    Some(0) => {
                        self.pen.attrs.remove(Attrs::UNDERLINE);
                        self.pen.attrs.set_underline_style(UnderlineStyle::Single);
                    }
                    style => {
                        // An unknown style is still an underline: better a straight line
                        // where a program wanted a fancy one than no mark at all.
                        let style = match style {
                            Some(2) => UnderlineStyle::Double,
                            Some(3) => UnderlineStyle::Curly,
                            Some(4) => UnderlineStyle::Dotted,
                            Some(5) => UnderlineStyle::Dashed,
                            _ => UnderlineStyle::Single,
                        };
                        self.pen.attrs.insert(Attrs::UNDERLINE);
                        self.pen.attrs.set_underline_style(style);
                    }
                },
                7 => self.pen.attrs.insert(Attrs::REVERSE),
                8 => self.pen.attrs.insert(Attrs::HIDDEN),
                9 => self.pen.attrs.insert(Attrs::STRIKE),
                21 => self.pen.attrs.remove(Attrs::BOLD),
                22 => self.pen.attrs.remove(Attrs::BOLD | Attrs::DIM),
                23 => self.pen.attrs.remove(Attrs::ITALIC),
                24 => {
                    self.pen.attrs.remove(Attrs::UNDERLINE);
                    self.pen.attrs.set_underline_style(UnderlineStyle::Single);
                }
                27 => self.pen.attrs.remove(Attrs::REVERSE),
                28 => self.pen.attrs.remove(Attrs::HIDDEN),
                29 => self.pen.attrs.remove(Attrs::STRIKE),
                30..=37 => self.pen.fg = Color::Ansi((p - 30) as u8),
                38 => {
                    let (color, advance) = parse_ext_color(param, params, i);
                    if let Some(color) = color {
                        self.pen.fg = color;
                    }
                    i += advance;
                }
                39 => self.pen.fg = Color::Default,
                40..=47 => self.pen.bg = Color::Ansi((p - 40) as u8),
                48 => {
                    let (color, advance) = parse_ext_color(param, params, i);
                    if let Some(color) = color {
                        self.pen.bg = color;
                    }
                    i += advance;
                }
                49 => self.pen.bg = Color::Default,
                // Underline colour. We have nowhere on a cell to put it, but it takes
                // the same arguments as 38/48, so it must be consumed all the same.
                58 => {
                    let (_, advance) = parse_ext_color(param, params, i);
                    i += advance;
                }
                90..=97 => self.pen.fg = Color::Ansi((p - 90 + 8) as u8),
                100..=107 => self.pen.bg = Color::Ansi((p - 100 + 8) as u8),
                _ => {}
            }
            i += 1;
        }
    }

    /// Whether we implement `mode`, and if so whether it is currently on. `None` means
    /// we do not know it, which is exactly what DECRQM needs to hear and the one thing
    /// it must never be told falsely.
    ///
    /// Every mode [`set_mode`](Self::set_mode) acts on appears here, and nothing else
    /// does. If you teach the terminal a new mode, teach this at the same time — the two
    /// lists disagreeing is how a terminal ends up claiming a feature it does not have.
    pub(super) fn mode_state(&self, mode: u16, private: bool) -> Option<bool> {
        if !private {
            return match mode {
                4 => Some(self.insert_mode), // IRM
                _ => None,
            };
        }
        Some(match mode {
            1 => self.app_cursor_keys,
            6 => self.origin_mode,
            7 => self.autowrap,
            25 => self.cursor_visible,
            47 | 1047 | 1049 => self.on_alt,
            1000 => self.mouse.protocol == MouseProtocol::Press,
            1002 => self.mouse.protocol == MouseProtocol::ButtonEvent,
            1003 => self.mouse.protocol == MouseProtocol::AnyEvent,
            1006 => self.mouse.sgr,
            12 => self.cursor_appearance.blink,
            1004 => self.focus_events,
            2048 => self.in_band_resize,
            2004 => self.bracketed_paste,
            2026 => self.synchronized,
            2027 => self.grapheme_clustering,
            _ => return None,
        })
    }

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
                // `?12` blinks the cursor without saying anything about its shape, which
                // is the one thing DECSCUSR cannot do: its parameters bind the two
                // together. A program that wants a blinking bar and finds a steady block
                // has to be able to say only "blink".
                12 => self.cursor_appearance.blink = enable,
                1004 => self.focus_events = enable,
                2048 => self.set_in_band_resize(enable),
                2004 => self.bracketed_paste = enable,
                2026 => self.synchronized = enable,
                2027 => {
                    self.grapheme_clustering = enable;
                    // Whatever was on screen was measured the other way; start clean.
                    self.cluster.reset();
                    self.cluster_anchor = None;
                }
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
    pub(super) fn set_mouse_protocol(&mut self, protocol: MouseProtocol, enable: bool) {
        self.mouse.protocol = if enable { protocol } else { MouseProtocol::Off };
    }

    /// DECSCUSR: the `Ps` argument selects both the shape and whether it blinks (odd
    /// blinks, even is steady). `Ps = 0` asks for the terminal's own default, so it
    /// restores bnkterm's power-on look rather than xterm's documented blinking block
    /// (see [`CursorAppearance`]). An unknown `Ps` is ignored.
    pub(super) fn set_cursor_style(&mut self, ps: u16) {
        self.cursor_appearance = match ps {
            0 => CursorAppearance::default(),
            1 => CursorAppearance::new(CursorStyle::Block, true),
            2 => CursorAppearance::new(CursorStyle::Block, false),
            3 => CursorAppearance::new(CursorStyle::Underline, true),
            4 => CursorAppearance::new(CursorStyle::Underline, false),
            5 => CursorAppearance::new(CursorStyle::Bar, true),
            6 => CursorAppearance::new(CursorStyle::Bar, false),
            _ => return,
        };
    }

    /// Enter or leave the alternate screen. Entering clears it and carries the cursor
    /// across unchanged (it is one cursor shared by both buffers, see below); the
    /// primary buffer is untouched, so leaving reveals it intact.
    fn switch_alt(&mut self, enable: bool) {
        if enable == self.on_alt {
            return;
        }
        // Entering or leaving the alt screen shows a live view, never stale
        // history; the alt screen has no scrollback to scroll anyway. The rows now
        // belong to a different buffer, so no id minted against the old one survives.
        self.view_offset = 0;
        self.break_row_identity();
        if enable {
            let blank = self.blank_cell();
            // The cursor carries across the switch rather than homing: the alt screen
            // is a different set of cells, not a different cursor. `?1049h` is where
            // this shows — it saves the cursor, switches, and clears, and a program
            // that then restores expects to land where it started, not at the origin.
            // The deferred-wrap flag rides along with it for the same reason.
            let cursor = self.active().cursor;
            self.alt.clear_all(blank);
            self.alt.cursor = cursor;
            self.alt.scroll_top = 0;
            self.alt.scroll_bottom = self.alt.rows - 1;
            self.on_alt = true;
        } else {
            // And back the same way. The cursor is one cursor: xterm keeps a single
            // position across both buffers, so leaving the alt screen has to carry it
            // home as much as entering it did. Carrying it only one way would be a
            // cursor that is shared going in and per-buffer coming out, which is the
            // half-and-half model this is here to remove.
            //
            // `?1049l` restores a saved cursor immediately after this and so cannot
            // tell; `?47l` and `?1047l` do not, and are what a program using the alt
            // screen without the save/restore pair sends.
            let cursor = self.alt.cursor;
            self.primary.cursor = cursor;
            self.on_alt = false;
        }
    }

    /// RIS: hard reset to the power-on state, keeping the current dimensions.
    pub fn reset(&mut self) {
        let (cols, rows) = self.dimensions();
        let epoch = self.epoch.next();
        *self = Screen::new(cols, rows);
        // A fresh `Screen` starts at epoch zero, which would make ids minted before the
        // reset look current again. Identity moves forward across a reset, never back.
        self.epoch = epoch;
    }

    /// DECSTR (`CSI ! p`): soft reset. Puts the *settings* back to their defaults while
    /// leaving the screen, the scrollback and the cursor position alone — which is why
    /// programs reach for it instead of RIS: it recovers a sane terminal without
    /// throwing away what is on it.
    ///
    /// One deliberate departure from the DEC table, which lists autowrap as reset ("No
    /// autowrap"). **xterm leaves autowrap on**, and a terminal that stops wrapping
    /// after a soft reset breaks the very shell the program handed control back to.
    /// xterm.js once implemented the letter of the spec here and had to undo it. When
    /// the spec and xterm disagree, programs were written against xterm.
    ///
    /// The saved cursor goes too (DECSC is reset to "home position"), so a stray DECRC
    /// afterwards cannot restore a cursor from before the reset.
    pub fn soft_reset(&mut self) {
        self.pen = Pen::default();
        self.cursor_visible = true;
        self.insert_mode = false;
        self.origin_mode = false;
        self.autowrap = true; // xterm, not DEC. See above.
        self.app_cursor_keys = false;
        self.keypad_app = false;
        self.g0 = Charset::Ascii;
        self.g1 = Charset::Ascii;
        self.gl_is_g1 = false;
        let b = self.active_mut();
        b.saved = None;
        // The margins go back to the full screen, but *without* homing the cursor the
        // way an explicit DECSTBM would: a soft reset is not a cursor move.
        b.scroll_top = 0;
        b.scroll_bottom = b.rows.saturating_sub(1);
        b.cursor.pending_wrap = false;
    }

    /// Resize both screens to `cols` x `rows` (clamped to at least 1x1). The app calls
    /// this when the window's pixel size divided by the cell size yields a new grid; it
    /// then sends the child the matching `TIOCSWINSZ`.
    ///
    /// A width change re-wraps the primary screen and its scrollback ([`Buffer::reflow`]),
    /// so a widened window pulls soft-wrapped lines back together and a narrowed one breaks
    /// them; the cursor and a scrolled-up view are carried across. A height-only change
    /// keeps the cheap path — no rewrap is needed when the width is the same, and it grows by
    /// pulling history back onto the screen. The alt screen never reflows: a full-screen
    /// program owns it and repaints on `SIGWINCH`.
    ///
    /// Anchored state follows the text where it can. The primary's own OSC 133 prompt marks
    /// are translated internally; the returned [`ResizeEffect`] tells the caller how to carry
    /// its text selection (translate, keep, or drop). The row-id epoch ends on a reflow (or
    /// any alt-screen resize), so any holder that is *not* translated is still dropped by the
    /// usual prune — the translation is the addition, not a removal of that safety net.
    pub fn resize(&mut self, cols: usize, rows: usize) -> ResizeEffect {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let remap = if cols == self.primary.cols {
            // Height-only (or a no-op): re-clamp rows (a grow pulls history back down) and
            // re-pin the view. The primary's ids stay put, so prompts need no translation.
            self.primary.resize(cols, rows);
            self.view_offset = 0;
            None
        } else {
            // Width changed: rewrap the primary, re-deriving the view, and translate the
            // primary's prompt marks through the reflow.
            //
            // The live prompt is left for the shell to repaint: freeze from the last idle
            // prompt mark (one with no command output yet — the input the user is sitting at)
            // down to the bottom, never below the cursor. Everything above it, including a
            // finished command's output, still reflows.
            let old_count = self.primary.scrollback.len() + self.primary.rows;
            let cursor_idx = self.primary.scrollback.len() + self.primary.cursor.row;
            let frozen_from = self
                .prompts
                .iter()
                .rev()
                .find(|p| p.output.is_none())
                .and_then(|p| self.primary.stream_index(p.row))
                .map(|idx| idx.min(cursor_idx))
                .unwrap_or(old_count);
            let view = self.view_offset();
            let (new_view, remap) = self.primary.reflow(cols, rows, view, frozen_from);
            self.view_offset = new_view;
            self.prompts.retain_mut(|p| match remap.row(p.row) {
                Some(row) => {
                    p.row = row;
                    p.output = p.output.and_then(|o| remap.row(o));
                    true
                }
                None => false,
            });
            Some(remap)
        };
        self.alt.resize(cols, rows);

        if self.on_alt {
            // A selection was over the alt screen, which only clamps and renumbers: there is
            // nothing to translate it against, so end the epoch and tell the caller to drop.
            self.epoch = self.epoch.next();
            ResizeEffect::Reset
        } else if let Some(remap) = remap {
            // Primary width reflow: the ids were renumbered but are translatable. End the
            // epoch (the safety net for any holder not translated) and hand back the map.
            self.epoch = self.epoch.next();
            ResizeEffect::Reflowed(remap)
        } else {
            // Height-only on the primary: the ids stayed put, so a selection stands as-is.
            ResizeEffect::Stable
        }
    }

    /// DECALN: fill the whole screen with 'E', reset the margins to the extremes of the
    /// page, and home the cursor (a vttest alignment pattern; useful for confirming
    /// glyph placement early).
    ///
    /// The margins and the cursor are not incidental: DECALN's own definition is that it
    /// "sets the margins to the extremes of the page, and moves the cursor to the home
    /// position", so a screen left under a stale scroll region after an alignment test
    /// would scroll inside it.
    pub fn decaln(&mut self) {
        let b = self.active_mut();
        b.clear_all(Cell::new('E'));
        b.scroll_top = 0;
        b.scroll_bottom = b.rows.saturating_sub(1);
        b.cursor = Cursor::default();
    }

    /// DECKPAM/DECKPNM keypad mode, read by the input encoder (phase 3).
    pub fn keypad_app(&self) -> bool {
        self.keypad_app
    }

    /// Whether the bell has rung since this was last asked, clearing it. The grid has no
    /// opinion on what a bell should *do* — that is the app's call, and on a modern
    /// desktop it is a visual mark rather than a beep.
    pub fn take_bell(&mut self) -> bool {
        std::mem::take(&mut self.bell)
    }

    /// Whether the child is mid-frame under synchronized output (`?2026`), and would
    /// rather we showed nothing than showed it half-drawn.
    pub fn synchronized(&self) -> bool {
        self.synchronized
    }

    /// The palette in force. The renderer borrows this at paint time; a program can have
    /// changed any of it through `OSC 4/10/11/12`.
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// The window's pixel size, which the in-band resize report carries alongside the
    /// cell count. The grid does not otherwise care about pixels; it is told.
    pub fn set_pixel_size(&mut self, width: u32, height: u32) {
        self.pixel_size = (width, height);
    }
}

/// Clamp an SGR color component (`0..=255`) to a byte; the parser already bounds
/// parameters, so this only guards a malformed stream, never a valid one.
pub(super) fn sgr_component(v: u16) -> u8 {
    v.min(255) as u8
}

/// Parse the extended-colour introducer (`38`, `48`, `58`) at parameter `at`,
/// returning the colour and how many *extra parameters* it consumed.
///
/// The arguments come in one of two spellings, and a terminal has to read both:
///
/// ```text
///   38 ; 2 ; r ; g ; b     five parameters   (xterm's, and the common one)
///   38 : 2 : : r : g : b   one parameter, six sub-parameters (ITU T.416)
///   38 : 2 : r : g : b     one parameter, five sub-parameters (seen in the wild)
/// ```
///
/// The empty slot in the T.416 form is a colour-space id, which nobody uses and we
/// ignore. The sub-parameter form consumes no *extra* parameters — everything it needs
/// is inside the one it started in — which is why the return value counts parameters
/// and not values.
///
/// An unrecognized form consumes nothing: better to skip one parameter than to swallow
/// a `1` that was meant to turn on bold.
pub(super) fn parse_ext_color(param: &[u16], params: &Params, at: usize) -> (Option<Color>, usize) {
    // The sub-parameter form: the arguments ride inside this parameter.
    if param.len() > 1 {
        return match param.get(1).copied() {
            Some(5) => {
                let n = param.get(2).copied().unwrap_or(0);
                (Some(Color::Indexed(sgr_component(n))), 0)
            }
            Some(2) => {
                // With a colour-space id (6 values) the channels start one later than
                // without it (5 values). Anything shorter is malformed; read what is
                // there and let the missing channels default to zero.
                let base = if param.len() >= 6 { 3 } else { 2 };
                let r = sgr_component(param.get(base).copied().unwrap_or(0));
                let g = sgr_component(param.get(base + 1).copied().unwrap_or(0));
                let b = sgr_component(param.get(base + 2).copied().unwrap_or(0));
                (Some(Color::Rgb(r, g, b)), 0)
            }
            _ => (None, 0),
        };
    }

    // The parameter form: the arguments are the parameters that follow.
    match params.value(at + 1) {
        5 if params.len() > at + 1 => {
            let n = params.value(at + 2);
            (Some(Color::Indexed(sgr_component(n))), 2)
        }
        2 if params.len() > at + 1 => {
            let r = sgr_component(params.value(at + 2));
            let g = sgr_component(params.value(at + 3));
            let b = sgr_component(params.value(at + 4));
            (Some(Color::Rgb(r, g, b)), 4)
        }
        _ => (None, 0),
    }
}
