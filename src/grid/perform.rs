//! The parser-to-grid seam: syntax becomes semantics here and nowhere else.
//!
//! `vt::Parser` emits purely syntactic callbacks — this printed, this control byte, this
//! CSI with these parameters — and this module is the single place that decides what each
//! one *means*, by calling the operations the sibling modules provide. That keeps the
//! parser free of grid state and the grid free of byte handling, which is what makes each
//! independently testable.
//!
//! The rule that earns its own file: **a private marker selects a different sequence, it
//! does not modify the ANSI sequence sharing its final byte.** Letting one fall through is
//! how `CSI ? Ps r` (XTRESTORE) came to execute as DECSTBM, resetting the scroll region
//! and homing the cursor, and how `CSI ? Pi ; Pa ; Pv S` (a sixel query) came to execute
//! as SU and scroll the screen out from under the program that asked. So a marker and a
//! final byte are matched *together*, and anything unrecognised is dropped. Dropping an
//! unknown sequence is always safe; guessing is not.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::*;

impl Screen {
    /// The charset GL currently maps through (G0 or G1).
    pub(super) fn active_charset(&self) -> Charset {
        if self.gl_is_g1 {
            self.g1
        } else {
            self.g0
        }
    }

    /// Translate a printable char through the active charset (DEC Special
    /// Graphics remaps `_`..`~` to line-drawing glyphs; ASCII is identity).
    pub(super) fn map_glyph(&self, c: char) -> char {
        match self.active_charset() {
            Charset::Ascii => c,
            Charset::DecSpecialGraphics => dec_special_graphics(c),
        }
    }

    /// The private-marker CSI space: `CSI ? …` (DEC), `CSI > …`, `CSI = …`, `CSI < …`.
    ///
    /// A marker is not a modifier on the ANSI sequence that shares its final byte, it
    /// selects a *different sequence*. Letting one fall through to its ANSI namesake is
    /// how `CSI ? Ps r` (XTRESTORE, "restore saved private modes") used to execute as
    /// DECSTBM — resetting the scroll region and homing the cursor — and how
    /// `CSI ? Pi ; Pa ; Pv S` (XTSMGRAPHICS, a sixel query) used to execute as SU and
    /// scroll the screen out from under the program that asked.
    ///
    /// So the rule here is: match a marker and a final byte *together*, and drop
    /// anything else. Dropping an unknown sequence is always safe. Guessing is not.
    fn csi_private(&mut self, params: &Params, private: u8, action: u8) {
        match (private, action) {
            (b'?', b'h') => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), true, true);
                }
            }
            (b'?', b'l') => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), true, false);
                }
            }
            (b'?', b'n') => self.device_status(params, b'?'),
            (b'>', b'c') => self.device_attributes(b'>'),
            (b'>', b'm') => self.xtmodkeys(params),
            (b'>', b'q') => self.xtversion(),
            // The kitty keyboard protocol: `?` query, `=` set, `>` push, `<` pop. It
            // shares SCORC's final byte, and only the marker tells them apart — which is
            // exactly why this space has to be matched as a pair.
            (_, b'u') => self.kitty_keyboard(params, private),
            _ => {}
        }
    }
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

    fn print_run(&mut self, chars: &[char]) {
        if self.insert_mode || self.grapheme_clustering || self.active_charset() != Charset::Ascii {
            for &c in chars {
                let mapped = self.map_glyph(c);
                self.print(mapped);
            }
            return;
        }
        self.print_text_run(chars);
    }

    fn print_ascii(&mut self, bytes: &[u8]) {
        // The bulk write assumes each byte is its own glyph placed one column
        // apart. That holds only under the identity (ASCII) charset, outside
        // insert mode, and outside grapheme clustering; DEC Special Graphics remaps
        // each byte to a line-drawing glyph, insert mode shifts the row per char, and
        // under `?2027` a byte is not its own glyph at all — it may be the *base* of a
        // cluster that the next scalar joins and widens. `#` is one column; `#️⃣` is the
        // same byte, two columns, and one keycap. The bulk path places cells without
        // anchoring a cluster, so a run through it leaves nothing for a following VS16
        // to attach to and the keycap silently loses half its width.
        //
        // The cost is that `?2027` turns off the bulk ASCII path, which is a real
        // slowdown for a program that opts in. That is the right way round: the mode is
        // off by default, so nothing pays for it unasked, and a program that asks for
        // cluster semantics is asking for exactly the thing the fast path cannot do.
        if self.insert_mode || self.grapheme_clustering || self.active_charset() != Charset::Ascii {
            for &b in bytes {
                let mapped = self.map_glyph(char::from(b));
                self.print(mapped);
            }
            return;
        }
        self.print_ascii_run(bytes);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            0x08 => self.backspace(),        // BS
            0x09 => self.tab(),              // HT
            0x0a..=0x0c => self.line_feed(), // LF, VT, FF
            0x0d => self.carriage_return(),  // CR
            0x0e => self.gl_is_g1 = true,    // SO (shift out to G1)
            0x0f => self.gl_is_g1 = false,   // SI (shift in to G0)
            0x07 => self.bell = true,        // BEL
            _ => {}                          // NUL, XON/XOFF, ...: nothing to draw
        }
        // Whatever it did, it was not printing: there is nothing left for REP to repeat.
        self.last_printed = None;
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], private: u8, action: u8) {
        // The sequences carrying an intermediate byte, and *only* those: the overwhelming
        // majority of CSIs have none, and they should not pay four slice comparisons to
        // discover it. Each is matched on the intermediate, the private marker *and* the
        // final byte together — see `csi_private` for what happens when a final byte is
        // trusted on its own.
        if !intermediates.is_empty() {
            match (intermediates, private, action) {
                ([b'!'], 0, b'p') => self.soft_reset(), // DECSTR
                ([b'$'], 0, b'p') => self.report_mode(params.value(0), false), // DECRQM
                ([b'$'], b'?', b'p') => self.report_mode(params.value(0), true), // DECRQM
                ([b' '], 0, b'q') => self.set_cursor_style(params.value(0)), // DECSCUSR
                // Anything else with an intermediate is dropped rather than misread.
                _ => {}
            }
            return;
        }
        // A private marker selects a different sequence space; it does not modify the
        // ANSI sequence that happens to share the final byte. Splitting them here is
        // what stops `CSI ? Ps r` (XTRESTORE) from executing as DECSTBM.
        if private != 0 {
            self.csi_private(params, private, action);
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
            b'Z' => self.back_tab(csi_count(params, 0)), // CBT
            b'b' => self.repeat_last(csi_count(params, 0)), // REP
            b'J' => self.erase_display(csi_arg(params, 0)),
            b'K' => self.erase_line(csi_arg(params, 0)),
            b'L' => self.insert_lines(csi_count(params, 0)),
            b'M' => self.delete_lines(csi_count(params, 0)),
            b'@' => self.insert_chars(csi_count(params, 0)),
            b'P' => self.delete_chars(csi_count(params, 0)),
            b'X' => self.erase_chars(csi_count(params, 0)),
            b'S' => self.scroll_up(csi_count(params, 0)),
            // `CSI Ps T` is SD, but `CSI Ps;Ps;Ps;Ps;Ps T` is xterm's highlight mouse
            // tracking — a different function that merely shares a final byte. bnkterm
            // does not implement it, and *ignoring* it is the whole of the right answer:
            // reading it as a scroll would jerk the page every time a program asked to
            // track the mouse, which is worse than not answering at all.
            b'T' if params.len() == 5 => {}
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
            b'm' => self.sgr(params),
            b'h' => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), false, true);
                }
            }
            b'l' => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), false, false);
                }
            }
            b's' if params.is_empty() => self.save_cursor(),
            b'u' if params.is_empty() => self.restore_cursor(),
            b'g' => self.clear_tab_stop(csi_arg(params, 0)),
            b'c' => self.device_attributes(0),
            b'n' => self.device_status(params, 0),
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8) {
        match intermediates.first().copied() {
            None => match byte {
                b'c' => self.reset(),            // RIS
                b'D' => self.line_feed(),        // IND
                b'E' => self.next_line(),        // NEL
                b'H' => self.set_tab_stop(),     // HTS
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

    fn dcs_dispatch(&mut self, _params: &Params, intermediates: &[u8], action: u8, data: &[u8]) {
        // Both of these are questions, and both were previously swallowed whole: DCS had
        // no callback at all, so a program that asked one waited for an answer that was
        // never coming.
        match (intermediates, action) {
            ([b'$'], b'q') => self.decrqss(data), // "what is this setting?"
            ([b'+'], b'q') => self.xtgettcap(data), // "what does your terminfo say?"
            _ => {}
        }
    }

    fn osc_dispatch(&mut self, data: &[u8], bel_terminated: bool) {
        // OSC Ps ; Pt
        let mut parts = data.splitn(2, |&b| b == b';');
        let ps = parts.next().unwrap_or(&[]);
        let pt = parts.next().unwrap_or(&[]);
        match ps {
            b"0" | b"2" => self.set_title(String::from_utf8_lossy(pt).into_owned()),
            b"4" => self.osc_palette(pt, bel_terminated),
            b"8" => self.set_hyperlink(pt),
            b"10" => self.osc_named_color(NamedColor::Foreground, pt, bel_terminated),
            b"11" => self.osc_named_color(NamedColor::Background, pt, bel_terminated),
            b"12" => self.osc_named_color(NamedColor::Cursor, pt, bel_terminated),
            b"7" => self.osc_cwd(pt),
            b"52" => self.osc_clipboard(pt),
            b"104" => self.osc_reset_palette(pt),
            b"133" => self.osc_shell_mark(pt),
            b"110" => self.theme.fg = Theme::default().fg,
            b"111" => self.theme.bg = Theme::default().bg,
            b"112" => self.theme.cursor = Theme::default().cursor,
            _ => {}
        }
    }
}

/// A CSI numeric parameter, or 0 when absent.
pub(super) fn csi_arg(params: &Params, i: usize) -> u16 {
    params.value(i)
}

/// A CSI count parameter (for cursor moves, repeat counts): absent or 0 means 1.
pub(super) fn csi_count(params: &Params, i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1))
}

/// A 1-based CSI position parameter as a 0-based index (default 1 maps to 0).
pub(super) fn csi_index(params: &Params, i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1)) - 1
}

/// The charset an `ESC ( F` / `ESC ) F` designation selects. Only `0` (DEC
/// Special Graphics) differs from ASCII in what we support; `B` (ASCII) and any
/// other designation map to ASCII.
pub(super) fn charset_from(byte: u8) -> Charset {
    match byte {
        b'0' => Charset::DecSpecialGraphics,
        _ => Charset::Ascii,
    }
}

/// The DEC Special Graphics glyph for a byte in `0x60..=0x7e` (the line-drawing
/// set); any other char passes through unchanged. This is the box-drawing
/// coverage `less`, `mc`, and framed TUIs depend on.
pub(super) fn dec_special_graphics(c: char) -> char {
    match c {
        // 0x5F is "Blank" in DEC's own chart, not an underscore: the set replaces the
        // ASCII glyph at every position it defines, and this is the one that maps to
        // nothing visible. Leaving it through as `_` puts an underscore in the middle
        // of any line-drawing output that happens to contain one.
        '_' => ' ',
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
