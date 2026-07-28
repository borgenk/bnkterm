//! The OSC handlers: the sequences by which the child tells the terminal about
//! something other than the grid.
//!
//! ```text
//!   OSC 0/2   the window title           OSC 8    a hyperlink opens/closes
//!   OSC 4     a palette entry            OSC 7    the shell's working directory
//!   OSC 10/11/12  fg / bg / cursor       OSC 133  where a prompt and its output begin
//!   OSC 104   reset the palette          OSC 52   put this on the clipboard
//! ```
//!
//! Two of these are more than decoration. `OSC 133` is what turns an undifferentiated
//! river of text into a structure the terminal can act on (jump to a prompt, and — the
//! load-bearing use — know where the live prompt is so a reflow does not fight the
//! shell's own redraw). `OSC 52` is how a program with no compositor connection reaches
//! the clipboard, and its *read* half is refused on purpose: see
//! [`Screen::osc_clipboard`].
//!
//! Everything here parses attacker-controlled bytes, so the payload decoders at the
//! bottom (percent, base64) refuse malformed input rather than improvising on it.

// The grid's shared vocabulary: the types in `grid/mod.rs` and the imports it makes.
// `Screen`'s operations are split across these sibling modules, so each one works on
// the same definitions rather than re-importing them piecemeal.
use super::reply::push_decimal;
use super::*;

impl Screen {
    /// OSC 0/2: set the window title.
    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    /// OSC 8: open or close a hyperlink on the pen. `pt` is everything after the `8;`,
    /// which the spec shapes as `params ; URI`:
    ///
    /// ```text
    ///   ESC ] 8 ; ; https://example.com  ESC \   the anchor opens
    ///   text the user sees                       these cells carry the link
    ///   ESC ] 8 ; ;                       ESC \   an empty URI closes it
    /// ```
    ///
    /// `params` is a `key=value:key=value` list whose one defined key is `id=`, a hint
    /// that two separate runs are one logical link. We ignore it: interning by URL
    /// already gives identical URLs one identity, which is the only thing the hint was
    /// for, and honouring an attacker-chosen id would let a child fuse two unrelated
    /// runs into one hover target.
    ///
    /// The URI is vetted **here**, when the anchor opens, not at click time. That is
    /// the rule [`crate::platform::browser::OPENABLE_SCHEMES`] states: a link we
    /// decorate and then refuse to follow is worse than one we never decorated. So a
    /// `javascript:` or `data:` anchor is not a link at all — it prints as plain text
    /// and never underlines, rather than underlining and then doing nothing. Anything
    /// we cannot make sense of (bad UTF-8, an unopenable scheme, an empty URI) closes
    /// whatever anchor was open, which is also the safe reading of a malformed
    /// sequence.
    pub(super) fn set_hyperlink(&mut self, pt: &[u8]) {
        let mut fields = pt.splitn(2, |&b| b == b';');
        let _params = fields.next();
        let uri = fields.next().unwrap_or(&[]);
        let opened = std::str::from_utf8(uri)
            .ok()
            .filter(|uri| crate::platform::browser::can_open(uri));
        self.pen.link = match opened {
            Some(uri) => self.intern_link(uri),
            None => LinkId::NONE,
        };
    }

    /// The id for `url`, collecting dead ids first if the table has filled up. Falling
    /// all the way through to [`LinkId::NONE`] means 65534 *live* links are on screen
    /// and in scrollback at once, at which point the newest simply is not clickable —
    /// a degradation, never a failure, and the text still reads.
    pub(super) fn intern_link(&mut self, url: &str) -> LinkId {
        // Too long to ever store: turned away here rather than in the table, because
        // failing there would send us through a full mark-and-sweep over every cell in
        // both buffers to make room that no amount of collecting can make.
        if url.len() > LINK_URL_MAX {
            return LinkId::NONE;
        }
        if let Some(id) = self.links.intern(url) {
            return id;
        }
        // The table is full, of ids or of bytes. A sweep is worth doing once — and worth
        // *not* repeating until something could have died since, or a child printing
        // anchors into a full table buys a walk over every cell in both buffers with
        // each one. See `LinkTable::swept_dry_at`.
        let grid_at = (self.epoch, self.primary.evicted);
        if self.links.swept_dry_at == Some(grid_at) {
            return LinkId::NONE;
        }
        self.collect_links();
        match self.links.intern(url) {
            Some(id) => id,
            None => {
                self.links.swept_dry_at = Some(grid_at);
                LinkId::NONE
            }
        }
    }

    /// Reclaim the ids of hyperlinks no cell carries any more.
    ///
    /// Links die when the cells citing them do: scrollback evicts its oldest rows, the
    /// alt screen is thrown away wholesale on exit, an erase blanks a line. Without
    /// this, a session that prints an unbounded stream of *distinct* links (an
    /// `ls --hyperlink` of a large tree, run again and again) would burn through the
    /// `u16` id space and quietly stop linking anything until the terminal restarted.
    ///
    /// A textbook mark-and-sweep, run only when [`LinkTable::intern`] reports the table
    /// full — so the walk over every cell is an *event*, once per 65534 distinct links,
    /// and never a per-frame cost. The roots are every cell in both buffers (visible
    /// and scrollback) plus the pen and both saved cursors, because an anchor can be
    /// open with no text printed under it yet.
    pub(super) fn collect_links(&mut self) {
        // `live[i]` speaks for `LinkId(i)`; slot 0 is `LinkId::NONE` and is never a URL.
        let mut live = vec![false; self.links.urls.len() + 1];
        let mark = |id: LinkId, live: &mut Vec<bool>| {
            if let Some(slot) = live.get_mut(usize::from(id.0)) {
                *slot = true;
            }
        };
        for buf in [&self.primary, &self.alt] {
            for row in buf.scrollback.iter().chain(buf.lines.iter()) {
                for cell in &row.cells {
                    mark(cell.link, &mut live);
                }
            }
            if let Some(saved) = &buf.saved {
                mark(saved.pen.link, &mut live);
            }
        }
        mark(self.pen.link, &mut live);

        let remap = self.links.compact(&live);
        let renumber = |id: LinkId| {
            remap
                .get(usize::from(id.0))
                .copied()
                .unwrap_or(LinkId::NONE)
        };
        for buf in [&mut self.primary, &mut self.alt] {
            for row in buf.scrollback.iter_mut().chain(buf.lines.iter_mut()) {
                for cell in &mut row.cells {
                    cell.link = renumber(cell.link);
                }
            }
            if let Some(saved) = &mut buf.saved {
                saved.pen.link = renumber(saved.pen.link);
            }
        }
        self.pen.link = renumber(self.pen.link);
    }

    /// `OSC 10/11/12`: set or *query* the default foreground, background, or cursor
    /// colour.
    ///
    /// The query is the reason this matters, and it is the most useful reply a terminal
    /// gives. An editor asks `OSC 11 ; ? ST` to find out whether it is sitting on a dark
    /// or a light background, and picks its whole colour scheme from the answer. A
    /// terminal that stays silent does not get a default — it gets a *guess*, and half
    /// the time the guess is wrong and every colour in the editor is subtly off.
    ///
    /// A program may batch several requests in one sequence (`OSC 10 ; ? ; ? ST` asks for
    /// the foreground and then the background), so the payload is a list, and each field
    /// steps to the next colour in the sequence fg → bg → cursor.
    pub(super) fn osc_named_color(&mut self, first: NamedColor, pt: &[u8], bel: bool) {
        let mut which = Some(first);
        for field in pt.split(|&b| b == b';') {
            let Some(target) = which else { break };
            let slot = match target {
                NamedColor::Foreground => &mut self.theme.fg,
                NamedColor::Background => &mut self.theme.bg,
                NamedColor::Cursor => &mut self.theme.cursor,
            };
            if field == b"?" {
                let (color, code) = (*slot, osc_color_code(target));
                self.respond(b"\x1b]");
                push_decimal(&mut self.responses, code);
                self.responses.push(b';');
                color::write_x11_color(color, &mut self.responses);
                self.end_osc(bel);
            } else if let Some(color) = color::parse_x11_color(field) {
                *slot = color;
            }
            which = match target {
                NamedColor::Foreground => Some(NamedColor::Background),
                NamedColor::Background => Some(NamedColor::Cursor),
                NamedColor::Cursor => None,
            };
        }
    }

    /// `OSC 4 ; index ; spec` sets a palette entry; `OSC 4 ; index ; ?` queries one.
    /// Several pairs may ride in one sequence, which is how a theme script repaints the
    /// whole palette in a single write.
    pub(super) fn osc_palette(&mut self, pt: &[u8], bel: bool) {
        let mut fields = pt.split(|&b| b == b';');
        while let (Some(index), Some(spec)) = (fields.next(), fields.next()) {
            let Some(index) = parse_u8(index) else {
                continue;
            };
            if spec == b"?" {
                self.respond(b"\x1b]4;");
                push_decimal(&mut self.responses, u32::from(index));
                self.responses.push(b';');
                color::write_x11_color(self.theme.indexed(index), &mut self.responses);
                self.end_osc(bel);
            } else if let Some(color) = color::parse_x11_color(spec) {
                self.theme.set_indexed(index, color);
            }
        }
    }

    /// `OSC 104`: reset the named palette entries, or the whole palette when the payload
    /// is empty.
    pub(super) fn osc_reset_palette(&mut self, pt: &[u8]) {
        if pt.is_empty() {
            let (fg, bg, cursor) = (self.theme.fg, self.theme.bg, self.theme.cursor);
            self.theme.reset_palette();
            // `OSC 104` is about the *indexed* palette; the three named colours have
            // their own resets (110/111/112) and must survive this one.
            self.theme.fg = fg;
            self.theme.bg = bg;
            self.theme.cursor = cursor;
            return;
        }
        for field in pt.split(|&b| b == b';') {
            if let Some(index) = parse_u8(field) {
                self.theme.reset_indexed(index);
            }
        }
    }

    /// `OSC 7 ; file://<host>/<path>`: the shell reports its working directory.
    ///
    /// The path is percent-encoded, because a directory may contain any byte a filename
    /// may contain, including the `;` that separates OSC fields. A malformed escape is
    /// left as written rather than guessed at.
    pub(super) fn osc_cwd(&mut self, pt: &[u8]) {
        let path = match pt.strip_prefix(b"file://") {
            // Strip the host: it is the machine the shell is on, which we cannot do
            // anything useful with, and the path begins at the slash that follows it.
            Some(rest) => match rest.iter().position(|&b| b == b'/') {
                Some(i) => rest.get(i..).unwrap_or(&[]),
                None => return,
            },
            // A bare path, which some shells send. Accept it.
            None if pt.starts_with(b"/") => pt,
            None => return,
        };
        self.cwd = Some(percent_decode(path));
    }

    /// `OSC 133`: the shell tells us where a prompt is, where the command's output starts,
    /// and how it ended.
    ///
    /// ```text
    ///   OSC 133 ; A ST          a prompt starts here
    ///   OSC 133 ; B ST          the prompt ends and what you type begins
    ///   OSC 133 ; C ST          the command's output starts here
    ///   OSC 133 ; D ; 1 ST      the command finished, and it failed
    /// ```
    ///
    /// Without this a terminal sees an undifferentiated river of text and can only offer
    /// you a scrollbar. With it, it knows where one command ended and the next began, and
    /// everything good follows from that: jump back to the last prompt, select a command's
    /// output, mark the ones that failed.
    pub(super) fn osc_shell_mark(&mut self, pt: &[u8]) {
        let mut fields = pt.split(|&b| b == b';');
        let kind = fields.next().unwrap_or(&[]);
        match kind {
            b"A" => {
                let row = self.cursor_abs_row();
                // A shell that redraws its prompt in place (zsh does, on every keystroke
                // with syntax highlighting) re-marks the same row. It is the same prompt.
                if self.prompts.last().map(|p| p.row) == Some(row) {
                    return;
                }
                if self.prompts.len() >= PROMPT_LIMIT {
                    self.prompts.remove(0);
                }
                self.prompts.push(Prompt {
                    row,
                    output: None,
                    exit: None,
                });
            }
            b"C" => {
                let row = self.cursor_abs_row();
                if let Some(p) = self.prompts.last_mut() {
                    p.output.get_or_insert(row);
                }
            }
            b"D" => {
                // The exit code is optional: a shell that reports only "it finished" is
                // still telling us something worth knowing.
                let exit = fields
                    .next()
                    .and_then(|f| std::str::from_utf8(f).ok())
                    .and_then(|f| f.trim().parse::<i32>().ok());
                if let Some(p) = self.prompts.last_mut() {
                    p.exit = exit.or(p.exit);
                }
            }
            // `B` (the prompt ends, input begins) we parse and do not need: nothing we
            // offer keys off it, and inventing a use for it would be inventing a feature.
            _ => {}
        }
    }

    /// The prompts the shell has marked, oldest first.
    pub fn prompts(&self) -> &[Prompt] {
        &self.prompts
    }

    /// Scroll so the prompt nearest above (or below) the top of the view comes into sight,
    /// and answer whether one was found.
    ///
    /// This is the payoff, and it is the thing a scrollbar can never do: a scrollbar knows
    /// how far you have come, and this knows *what* you have come past. Ten screens of a
    /// build log scroll by in one keystroke because the terminal knows where the command
    /// that produced them started.
    ///
    /// The prompt lands at the top of the screen, which is where you want it: the command
    /// and everything it printed are then below it, in the order they happened. A prompt
    /// close to the live bottom cannot be lifted that far — there is not enough text under
    /// it to fill the screen — so it comes to rest as high as the history allows, which is
    /// the same thing every scrollbar in the world does at the end of its travel.
    pub fn scroll_to_prompt(&mut self, backward: bool) -> bool {
        let top = self.abs_row(0);
        let target = if backward {
            self.prompts.iter().rev().find(|p| p.row < top)
        } else {
            self.prompts.iter().find(|p| p.row > top)
        };
        let Some(row) = target.map(|p| p.row) else {
            return false;
        };
        let b = self.active();
        let Some(index) = b.stream_index(row) else {
            return false;
        };
        // The offset is measured back from the live top of the screen, which is where the
        // scrollback ends.
        let offset = b.scrollback.len().saturating_sub(index);
        self.view_offset = offset.min(self.scrollback_len());
        true
    }

    /// The working directory the shell last reported (`OSC 7`).
    pub fn reported_cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }

    /// `OSC 52 ; <selection> ; <base64>`: the child puts text on the clipboard.
    ///
    /// This is how a program reaches the system clipboard when it cannot reach the
    /// compositor itself — `nvim` over SSH, tmux's copy mode, Claude Code's `/copy`. The
    /// terminal is the only process in the pipeline that *is* on your desktop, so it has
    /// to be the one to do it.
    ///
    /// # Reads are refused, and that is deliberate
    ///
    /// The protocol also defines a *read* (`OSC 52 ; c ; ?`), which answers with the
    /// clipboard's contents. We do not implement it, and will not.
    ///
    /// A terminal cannot tell which program printed a byte. Anything that reaches your
    /// screen can send this — a `cat` of a hostile file, a log line, a compiler error
    /// quoting attacker-controlled text — and the reply goes straight back down the pty
    /// to whatever is reading it. That turns "display some text" into "exfiltrate the
    /// user's clipboard", which routinely holds passwords. xterm ships reads disabled for
    /// exactly this reason and it is the right call.
    ///
    /// The *write* is a smaller version of the same problem (a hostile file can clobber
    /// your clipboard, which is annoying but not a disclosure), and it is the half every
    /// real program actually needs, so it stays.
    pub(super) fn osc_clipboard(&mut self, pt: &[u8]) {
        let mut fields = pt.splitn(2, |&b| b == b';');
        let selection = fields.next().unwrap_or(&[]);
        let payload = fields.next().unwrap_or(&[]);
        if payload == b"?" {
            return; // A read. See above: never.
        }
        let Some(text) = base64_decode(payload) else {
            return;
        };
        // The X11 selection letters. An empty field means the clipboard, and a program
        // may name several at once (`OSC 52 ; pc ; …`), so honour each one it lists.
        let letters = if selection.is_empty() {
            &b"c"[..]
        } else {
            selection
        };
        let mut clipboard = false;
        let mut primary = false;
        for &letter in letters {
            match letter {
                b'c' => clipboard = true,
                b'p' | b's' => primary = true,
                _ => {}
            }
        }
        if clipboard {
            self.clipboard_writes
                .push((ClipboardTarget::Clipboard, text.clone()));
        }
        if primary {
            self.clipboard_writes.push((ClipboardTarget::Primary, text));
        }
    }

    /// Take the clipboard writes the child has asked for, leaving the queue empty. The
    /// app drains this after each parse batch, exactly as it drains query replies.
    pub fn take_clipboard_writes(&mut self) -> Vec<(ClipboardTarget, Vec<u8>)> {
        std::mem::take(&mut self.clipboard_writes)
    }
}

/// The OSC number that names each colour, for building a reply.
pub(super) fn osc_color_code(which: NamedColor) -> u32 {
    match which {
        NamedColor::Foreground => 10,
        NamedColor::Background => 11,
        NamedColor::Cursor => 12,
    }
}

/// A decimal palette index from an OSC field. `None` for anything that is not a plain
/// number in `0..=255`, which the caller then skips.
pub(super) fn parse_u8(field: &[u8]) -> Option<u8> {
    if field.is_empty() || field.len() > 3 {
        return None;
    }
    let mut n: u32 = 0;
    for &b in field {
        let digit = (b as char).to_digit(10)?;
        n = n * 10 + digit;
    }
    u8::try_from(n).ok()
}

/// Percent-decode a URL path, which is how `OSC 7` carries a directory name.
///
/// It has to be encoded: a filename may contain any byte except `/` and NUL — including
/// the `;` that separates OSC fields and the bytes that would terminate the sequence
/// outright. A `%` that is not followed by two hex digits is a literal `%`, which is what
/// a shell that forgot to encode one produces, and guessing otherwise would mangle a
/// directory that is merely unusual rather than malformed.
///
/// Invalid UTF-8 becomes U+FFFD rather than being refused: a path we cannot render is
/// still a path we can show *something* for, and the alternative is silently having no cwd.
pub(super) fn percent_decode(input: &[u8]) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut i = 0;
    while let Some(&byte) = input.get(i) {
        let decoded = (byte == b'%')
            .then(|| {
                let hi = (*input.get(i + 1)? as char).to_digit(16)?;
                let lo = (*input.get(i + 2)? as char).to_digit(16)?;
                Some((hi * 16 + lo) as u8)
            })
            .flatten();
        match decoded {
            Some(b) => {
                bytes.push(b);
                i += 3;
            }
            None => {
                bytes.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Decode base64, the encoding `OSC 52` wraps clipboard text in. `None` for anything
/// malformed, which the caller drops.
///
/// Hand-rolled, and easily: base64 is a fixed 4-to-3 byte fold with no lengths, no
/// offsets and no allocation decisions taken from the input, which is exactly the shape
/// of thing the dependency policy says to write rather than depend on. (Image decoding
/// is the counter-example, and the reason the policy exists at all.)
///
/// Strict about what it accepts: the child writes this. Whitespace is skipped (mail-era
/// encoders wrap lines), padding is optional, and anything else is a refusal rather than
/// a guess — a decoder that improvises on bad input is how you smuggle bytes past it.
pub(super) fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3 + 3);
    let mut quad = [0u8; 4];
    let mut n = 0;
    let mut padding = 0;
    for &byte in input {
        if byte.is_ascii_whitespace() {
            continue;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                padding += 1;
                if padding > 2 {
                    return None;
                }
                0
            }
            _ => return None,
        };
        // Padding may only ever be trailing: `A=AA` is malformed, not a clever encoding.
        if padding > 0 && byte != b'=' {
            return None;
        }
        quad[n] = value;
        n += 1;
        if n == 4 {
            let triple = (u32::from(quad[0]) << 18)
                | (u32::from(quad[1]) << 12)
                | (u32::from(quad[2]) << 6)
                | u32::from(quad[3]);
            out.push((triple >> 16) as u8);
            if padding < 2 {
                out.push((triple >> 8) as u8);
            }
            if padding < 1 {
                out.push(triple as u8);
            }
            n = 0;
        }
    }
    match n {
        // A clean end, padded or exactly aligned.
        0 => Some(out),
        // Unpadded input, which the spec allows: finish the partial group.
        //
        // Only when it really is unpadded. `=` exists to *complete* a four-character
        // group, so a group that ended early cannot have carried any — and the loop above
        // counts a `=` into `n` like any other character, which is what made `AA=` land
        // here and decode to two bytes where the same input without the `=` gives one.
        // A length that cannot occur must be refused, not rounded to the nearest one that
        // can: this decoder's whole contract is that it never improvises on bad input.
        2 | 3 if padding == 0 => {
            let triple =
                (u32::from(quad[0]) << 18) | (u32::from(quad[1]) << 12) | (u32::from(quad[2]) << 6);
            out.push((triple >> 16) as u8);
            if n == 3 {
                out.push((triple >> 8) as u8);
            }
            Some(out)
        }
        // One leftover character encodes six bits of nothing, and a padded partial group
        // is a length base64 cannot produce.
        _ => None,
    }
}
