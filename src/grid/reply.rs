//! What the terminal says back.
//!
//! Half of what a terminal *does* is answer questions, and none of it shows up on the
//! grid: a wrong reply to `OSC 11` leaves the screen pixel-identical and the editor's
//! colours wrong. The grid holds no PTY, so every answer is appended to
//! [`Screen::responses`] and the app writes it to the master fd after each parse.
//!
//! ```text
//!   CSI c / CSI > c      DA      "what are you?"
//!   CSI 5n / CSI 6n      DSR     "are you there / where is the cursor?"
//!   CSI ? Ps $ p         DECRQM  "do you know this mode, and is it on?"
//!   DCS $ q … ST         DECRQSS "what is this setting set to?"
//!   DCS + q … ST         XTGETTCAP "what does your terminfo say?"
//!   CSI > 0 q            XTVERSION "which terminal are you?"
//!   CSI … u              kitty keyboard query/set/push/pop
//! ```
//!
//! One rule governs all of it: **answer honestly or not at all**. A mode we do not
//! implement must report "unrecognised" rather than "off", because "off" invites the
//! program to switch it on and depend on it; an unknown DECRQSS is refused rather than
//! answered with an invented setting. A lie here is worse than the silence it replaces.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::*;

impl Screen {
    /// The bytes queued to write back to the child (DA/DSR answers). Empty in the
    /// common case (no query since the last parse). Borrowed rather than taken so the
    /// caller writes it straight through and then [`clear_responses`](Self::clear_responses)
    /// in place, keeping the queue's capacity instead of leaving a zero-cap vec the
    /// next reply must regrow.
    pub fn responses(&self) -> &[u8] {
        &self.responses
    }

    /// Drop the queued reply bytes after they have been written, retaining capacity.
    pub fn clear_responses(&mut self) {
        self.responses.clear();
    }

    /// Take the queued reply bytes, leaving the queue empty. A read-and-clear
    /// convenience for tests; the app writes via [`responses`](Self::responses) and
    /// [`clear_responses`](Self::clear_responses) instead, to keep the capacity.
    #[cfg(test)]
    pub fn take_responses(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.responses)
    }

    /// Whether there is room to begin another answer.
    ///
    /// Reply *generation* has to be bounded here, because the child controls how many
    /// questions it asks and nothing guarantees anything is draining the answers. A
    /// 64 KiB batch of `\x1b[c` is roughly 21,800 device-attributes queries and about
    /// 150 KB of replies, minted from a file the user merely `cat`s — and `cat` never
    /// reads its own stdin. `OSC_MAX` bounds one sequence's payload and says nothing
    /// about how many sequences arrive.
    ///
    /// Checked *before* an answer is built, never during, which is what makes
    /// [`RESPONSE_MAX`] a soft cap: a reply already under way runs to completion and may
    /// carry the buffer a few tens of bytes past it. That is the point. Several replies
    /// are assembled in pieces (a prefix through [`respond`](Self::respond), then decimal
    /// fields pushed straight onto the buffer), and cutting one in half would deliver a
    /// malformed escape sequence into the child's input — strictly worse than silence,
    /// because a program waiting on an answer merely times out, while a program handed a
    /// broken one may act on it.
    fn can_reply(&self) -> bool {
        self.responses.len() < RESPONSE_MAX
    }

    /// Queue bytes to be written back to the child, if there is budget for another
    /// answer. See [`can_reply`](Self::can_reply) for why the cap exists and why it is
    /// checked per answer rather than per byte.
    pub(super) fn respond(&mut self, bytes: &[u8]) {
        if !self.can_reply() {
            return;
        }
        self.responses.extend_from_slice(bytes);
    }

    /// DECRQM (`CSI ? Ps $ p`, and `CSI Ps $ p` for the ANSI modes): "do you know this
    /// mode, and is it on?" Answered with DECRPM, `CSI ? Ps ; Pm $ y`.
    ///
    /// This is how anything we implement ever gets *used*. nvim asks five of these
    /// before it draws a single character, and a terminal that stays silent is told
    /// nothing and assumes nothing — so a mode we add and never report is a mode nobody
    /// will ever turn on.
    ///
    /// The values are 0 not recognised, 1 set, 2 reset, 3 permanently set, 4 permanently
    /// reset. Answering *honestly* is the whole job: a mode we do not implement must
    /// report **0**, never 2. Reporting 2 says "I know that mode, and it is currently
    /// off", which invites the program to switch it on and then depend on it. A lie here
    /// is worse than the silence it replaces.
    pub(super) fn report_mode(&mut self, mode: u16, private: bool) {
        if !self.can_reply() {
            return;
        }
        let state = match self.mode_state(mode, private) {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        };
        self.respond(if private { b"\x1b[?" } else { b"\x1b[" });
        push_decimal(&mut self.responses, u32::from(mode));
        self.responses.push(b';');
        push_decimal(&mut self.responses, state);
        self.respond(b"$y");
    }

    /// DA (Send Device Attributes): answer a program's "what are you?" probe.
    /// Primary (`CSI c`) reports a VT100 with the Advanced Video Option, the
    /// conservative, universally-understood identity (real capabilities come from
    /// `TERM`); secondary (`CSI > c`) gives a benign version triple. Answering at
    /// all is the point: a program that queries and gets nothing can hang.
    pub(super) fn device_attributes(&mut self, private: u8) {
        match private {
            0 => self.respond(b"\x1b[?1;2c"),
            b'>' => self.respond(b"\x1b[>0;0;0c"),
            _ => {}
        }
    }

    /// DSR (Device Status Report): `5 n` asks if we are OK (yes), `6 n` asks for
    /// the cursor position (CPR). The `?6 n` private form is the extended report
    /// (DECXCPR) some programs use. The position is 1-based and origin-mode aware.
    pub(super) fn device_status(&mut self, params: &Params, private: u8) {
        if !self.can_reply() {
            return;
        }
        let ps = params.value(0);
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
    pub(super) fn report_position(&self) -> (u32, u32) {
        let (mut row, col) = self.cursor();
        if self.origin_mode {
            row = row.saturating_sub(self.active().scroll_top);
        }
        (row as u32 + 1, col as u32 + 1)
    }

    /// `?2048`: report the terminal's size in band, as an escape sequence, rather than
    /// only through SIGWINCH.
    ///
    /// The signal is not enough on its own. It reaches the process group on *this* side of
    /// the pty and nothing else, so a program on the far end of an ssh hop, or behind
    /// tmux, learns nothing — and a program that has just been handed a pty has no way to
    /// ask. Enabling the mode reports the size immediately for exactly that reason: the
    /// first report is the one that answers "what am I attached to?".
    pub(super) fn set_in_band_resize(&mut self, enable: bool) {
        self.in_band_resize = enable;
        if enable {
            self.report_size();
        }
    }

    /// `CSI 48 ; rows ; cols ; height ; width t`, the in-band size report. Silent unless
    /// the child asked for it (`?2048`), which the app calls on every resize.
    pub fn report_size(&mut self) {
        if !self.in_band_resize {
            return;
        }
        if !self.can_reply() {
            return;
        }
        let (cols, rows) = self.dimensions();
        let (w, h) = self.pixel_size;
        self.respond(b"\x1b[48;");
        push_decimal(&mut self.responses, rows as u32);
        self.responses.push(b';');
        push_decimal(&mut self.responses, cols as u32);
        self.responses.push(b';');
        push_decimal(&mut self.responses, h);
        self.responses.push(b';');
        push_decimal(&mut self.responses, w);
        self.responses.push(b't');
    }

    /// Tell the child the window gained or lost focus (`CSI I` / `CSI O`), if it asked to
    /// be told (`?1004`). Silent otherwise: a program that never enabled focus reporting
    /// would read these as stray key presses.
    pub fn report_focus(&mut self, focused: bool) {
        if !self.focus_events {
            return;
        }
        self.respond(if focused { b"\x1b[I" } else { b"\x1b[O" });
    }

    /// XTVERSION (`CSI > 0 q`): "what terminal are you?" Answered `DCS > | bnkterm <ver> ST`.
    ///
    /// This does not get us onto anyone's allowlist — the CLIs that gate features on a
    /// terminal's name match it against a fixed list we are not on, and we do not intend
    /// to impersonate someone to get on it (see the `TERM_PROGRAM` note in `app.rs`). But
    /// it is what a terminal is *supposed* to say when asked, and the only way a program
    /// could recognise us deliberately rather than not at all.
    pub(super) fn xtversion(&mut self) {
        self.respond(b"\x1bP>|bnkterm ");
        self.respond(env!("CARGO_PKG_VERSION").as_bytes());
        self.respond(b"\x1b\\");
    }

    /// DECRQSS (`DCS $ q <setting> ST`): "what is this setting currently set to?" The
    /// answer is the sequence that would *reproduce* it — `DCS 1 $ r 0 m ST` says "SGR is
    /// at its default" — so a program can save a setting, change it, and put it back
    /// without knowing what it was.
    ///
    /// An unknown request is answered `DCS 0 $ r ST`, which means "I do not support that
    /// query". Answering 1 with an empty or invented setting would be worse than silence:
    /// the program would take the reply at face value and restore garbage.
    pub(super) fn decrqss(&mut self, setting: &[u8]) {
        if !self.can_reply() {
            return;
        }
        match setting {
            b"m" => {
                // SGR. We report the *pen*, which is what a program restoring a rendition
                // needs; the attribute bits are spelled out in the order xterm uses.
                self.respond(b"\x1bP1$r0");
                let attrs = self.pen.attrs;
                for (bit, code) in [
                    (Attrs::BOLD, b"1".as_slice()),
                    (Attrs::DIM, b"2"),
                    (Attrs::ITALIC, b"3"),
                    // The underline carries its shape with it. Reporting a bare `4` for a
                    // curly one answers a program's "what is this set to?" with a
                    // different setting: it saves the reply, changes the style, replays
                    // it, and its squiggle has quietly become a straight line. The whole
                    // point of DECRQSS is that the answer reproduces the state.
                    (
                        Attrs::UNDERLINE,
                        Self::underline_sgr(attrs.underline_style()),
                    ),
                    (Attrs::REVERSE, b"7"),
                    (Attrs::HIDDEN, b"8"),
                    (Attrs::STRIKE, b"9"),
                ] {
                    if attrs.contains(bit) {
                        self.responses.push(b';');
                        self.respond(code);
                    }
                }
                self.push_sgr_color(true);
                self.push_sgr_color(false);
                self.respond(b"m\x1b\\");
            }
            b"r" => {
                // DECSTBM, the scroll region, 1-based as the sequence that sets it.
                let (top, bottom) = {
                    let b = self.active();
                    (b.scroll_top, b.scroll_bottom)
                };
                self.respond(b"\x1bP1$r");
                push_decimal(&mut self.responses, top as u32 + 1);
                self.responses.push(b';');
                push_decimal(&mut self.responses, bottom as u32 + 1);
                self.respond(b"r\x1b\\");
            }
            _ => self.respond(b"\x1bP0$r\x1b\\"),
        }
    }

    /// The SGR parameter that sets an underline of this shape, for a DECRQSS reply.
    ///
    /// Plain underline reports as `4` rather than `4:1`, because that is what everything
    /// has always written and a program restoring it should get back what it sent.
    pub(super) fn underline_sgr(style: UnderlineStyle) -> &'static [u8] {
        match style {
            UnderlineStyle::Single => b"4",
            UnderlineStyle::Double => b"4:2",
            UnderlineStyle::Curly => b"4:3",
            UnderlineStyle::Dotted => b"4:4",
            UnderlineStyle::Dashed => b"4:5",
        }
    }

    /// One colour of the pen, as the SGR parameters that would set it, appended to a
    /// DECRQSS reply. Nothing is appended for a default colour: `SGR 0` already said it.
    pub(super) fn push_sgr_color(&mut self, foreground: bool) {
        let color = if foreground { self.pen.fg } else { self.pen.bg };
        let base = if foreground { 30 } else { 40 };
        match color {
            Color::Default => {}
            Color::Ansi(i) if i < 8 => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + i));
            }
            Color::Ansi(i) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 60 + (i & 7)));
            }
            Color::Indexed(i) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 8));
                self.respond(b";5;");
                push_decimal(&mut self.responses, u32::from(i));
            }
            Color::Rgb(r, g, b) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 8));
                self.respond(b";2;");
                push_decimal(&mut self.responses, u32::from(r));
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(g));
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(b));
            }
        }
    }

    /// XTGETTCAP (`DCS + q <hex-encoded names> ST`): "what does your terminfo say about
    /// these capabilities?" Names and values travel hex-encoded, which is how a value
    /// containing an escape sequence survives the trip.
    ///
    /// This is the query that lets a program learn what we can do *without* a terminfo
    /// entry installed for us — which matters, because we ship none and claim
    /// `xterm-256color`, so a terminfo lookup describes xterm rather than bnkterm.
    ///
    /// An unknown capability is answered with `DCS 0 + r <name> ST`, which is the
    /// protocol's way of saying "I do not have that" — and it is a real answer, not a
    /// silence, so the program stops waiting.
    pub(super) fn xtgettcap(&mut self, data: &[u8]) {
        if !self.can_reply() {
            return;
        }
        for name in data.split(|&b| b == b';') {
            let Some(decoded) = hex_decode(name) else {
                self.respond(b"\x1bP0+r\x1b\\");
                continue;
            };
            let value: Option<&[u8]> = match decoded.as_slice() {
                b"TN" | b"name" => Some(b"bnkterm".as_slice()),
                b"Co" | b"colors" => Some(b"256".as_slice()),
                // Truecolor. The `RGB` capability is how a program learns it can send
                // `38;2;r;g;b` and mean it; `COLORTERM=truecolor` says the same thing to
                // the programs that read the environment instead.
                b"RGB" => Some(b"8/8/8".as_slice()),
                _ => None,
            };
            match value {
                Some(value) => {
                    // The name goes back exactly as it arrived: it is already hex on the
                    // wire, and re-encoding it here would hex the hex.
                    self.respond(b"\x1bP1+r");
                    self.respond(name);
                    self.responses.push(b'=');
                    hex_encode(value, &mut self.responses);
                    self.respond(b"\x1b\\");
                }
                None => {
                    self.respond(b"\x1bP0+r");
                    self.respond(name);
                    self.respond(b"\x1b\\");
                }
            }
        }
    }

    /// The kitty keyboard protocol's four control sequences, all sharing the final byte
    /// `u` and distinguished by their private marker:
    ///
    /// ```text
    ///   CSI ? u                 query   ─▶ reply CSI ? <flags> u
    ///   CSI = <flags> ; <mode> u set     (mode 1 replace, 2 set bits, 3 clear bits)
    ///   CSI > <flags> u          push
    ///   CSI < <count> u          pop
    /// ```
    ///
    /// The request is masked to [`KittyFlags::SUPPORTED`] on the way in, so the query
    /// reports what bnkterm will really do rather than what was asked for. A program
    /// that wants key-release events and reads back that it is not getting them can
    /// fall back; one that is told yes and then never sees a release would hang.
    pub(super) fn kitty_keyboard(&mut self, params: &Params, private: u8) {
        if !self.can_reply() {
            return;
        }
        match private {
            b'?' => {
                self.respond(b"\x1b[?");
                let flags = self.kitty_flags().bits();
                push_decimal(&mut self.responses, flags);
                self.respond(b"u");
            }
            b'=' => {
                let flags = KittyFlags::from_request(params.value(0));
                let mode = if params.len() > 1 { params.value(1) } else { 1 };
                let current = self.kitty_flags();
                let next = current.apply(flags, mode);
                match self.kitty_stack.last_mut() {
                    Some(top) => *top = next,
                    // Setting flags with nothing pushed still has to take effect, so the
                    // set becomes the stack's first entry.
                    None => self.kitty_stack.push(next),
                }
            }
            b'>' => {
                let flags = KittyFlags::from_request(params.value(0));
                if self.kitty_stack.len() >= KITTY_STACK_LIMIT {
                    self.kitty_stack.remove(0);
                }
                self.kitty_stack.push(flags);
            }
            b'<' => {
                let count = usize::from(
                    if params.is_empty() {
                        1
                    } else {
                        params.value(0)
                    }
                    .max(1),
                );
                let keep = self.kitty_stack.len().saturating_sub(count);
                self.kitty_stack.truncate(keep);
            }
            _ => {}
        }
    }

    /// The kitty keyboard flags in force: the top of the stack, or none when the child
    /// has pushed nothing and the legacy encoding applies.
    pub fn kitty_flags(&self) -> KittyFlags {
        self.kitty_stack.last().copied().unwrap_or(KittyFlags::NONE)
    }

    /// The `modifyOtherKeys` level the child asked for.
    pub fn modify_other_keys(&self) -> ModifyOtherKeys {
        self.modify_other_keys
    }

    /// XTMODKEYS (`CSI > Pp ; Pv m`). `Pp = 4` is `modifyOtherKeys`, the only resource
    /// we implement; the others (`modifyCursorKeys`, `modifyFunctionKeys`) only shuffle
    /// encodings we already emit in their standard form. Omitting `Pv` resets, which is
    /// how xterm defines it and how a program turns the mode back off on exit.
    pub(super) fn xtmodkeys(&mut self, params: &Params) {
        // An absent `Pp` reads as 0, so the empty case is refused by this test too.
        if params.value(0) != 4 {
            return;
        }
        self.modify_other_keys = if params.len() > 1 {
            ModifyOtherKeys::from_param(params.value(1))
        } else {
            ModifyOtherKeys::Off
        };
    }

    /// Close a reply that opened with `OSC`, with the same terminator the request used.
    /// A client that asked with BEL may only be listening for BEL.
    pub(super) fn end_osc(&mut self, bel: bool) {
        if bel {
            self.responses.push(0x07);
        } else {
            self.respond(b"\x1b\\");
        }
    }
}

/// Append `n` as decimal ASCII (for building query responses), allocation-free.
pub(super) fn push_decimal(out: &mut Vec<u8>, mut n: u32) {
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

/// Decode the hex that XTGETTCAP encodes capability names in. `None` on anything that is
/// not an even run of hex digits, which the caller answers as "I do not have that"
/// rather than guessing at.
pub(super) fn hex_decode(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() || !input.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for pair in input.chunks_exact(2) {
        let hi = (*pair.first()? as char).to_digit(16)?;
        let lo = (*pair.get(1)? as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

/// Encode a capability value as the hex XTGETTCAP expects, which is how a value holding
/// an escape sequence survives being sent inside one.
pub(super) fn hex_encode(input: &[u8], out: &mut Vec<u8>) {
    for &byte in input {
        for nibble in [byte >> 4, byte & 0x0f] {
            out.push(match nibble {
                0..=9 => b'0' + nibble,
                _ => b'a' + (nibble - 10),
            });
        }
    }
}
