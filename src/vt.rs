//! The VT parser: a byte-oriented state machine that turns a terminal stream
//! into semantic actions, and nothing more. It holds no grid state; it emits
//! callbacks on a [`Perform`] trait that a consumer (`grid::Screen`) interprets.
//! That seam keeps the parser a pure `bytes -> actions` function, independently
//! testable against a recording oracle, and the grid a pure actions interpreter.
//!
//! The state machine follows Paul Flo Williams' DEC VT500 model
//! (<https://vt100.net/emu/dec_ansi_parser>):
//!
//! ```text
//!   ground ──ESC──▶ escape ──[──▶ csi_entry ──▶ csi_param ──▶ (dispatch) ──▶ ground
//!     │ print         │  ]              │ ; digits   │ final 0x40-0x7e
//!     │ execute       │  ▼ intermediate ▼            │
//!     ▼              esc_dispatch      csi_intermediate / csi_ignore
//!   (UTF-8 decode)     │  ]──▶ osc_string ──BEL/ST──▶ (title)
//!                      P/X/^/_──▶ string swallow (DCS/SOS/PM/APC: no-op in v1)
//! ```
//!
//! Hostile-input posture (a shell can print anything): parameters and
//! intermediates are bounded and saturating, the OSC buffer is capped, every
//! byte in every state has a defined transition, and there is no recursion, no
//! `unwrap`, and no indexing that can trap. Malformed UTF-8 yields U+FFFD, never
//! a panic. DCS/SOS/PM/APC strings are recognized and safely swallowed; bnkterm
//! implements no DCS in v1.

/// Maximum CSI parameters retained. Bounded so hostile input cannot grow state;
/// extra parameters past this are dropped, not an error. (The plan floated 16;
/// 32 gives headroom for long combined-SGR sequences real apps emit.)
const MAX_PARAMS: usize = 32;

/// Maximum intermediate bytes (0x20-0x2f) retained; a third makes the sequence
/// ignored, matching the reference parser.
const MAX_INTERMEDIATES: usize = 2;

/// Cap on the OSC string buffer (e.g. a window title); bytes past it are dropped.
const OSC_MAX: usize = 4096;

/// Cap on a DCS payload. The sequences we answer (DECRQSS, XTGETTCAP) carry a handful of
/// bytes; anything approaching this is not one of them.
const DCS_MAX: usize = 4096;

/// Length of the leading run of printable ASCII (`0x20..=0x7e`) in `bytes`.
///
/// A byte `b` is printable iff `b.wrapping_sub(0x20) <= 0x5e`: one unsigned compare
/// that rejects `b < 0x20` (wraps high), `b == 0x7f` (DEL, `0x5f`), and `b >= 0x80`
/// (`>= 0x60`) at once. We sweep eight bytes at a time with SWAR and fall to that
/// scalar test at the first word that isn't all-printable (and for the < 8-byte
/// tail). Within a word, a byte is non-printable iff it is `>= 0x80` (`& HI`), or
/// `< 0x20` (subtracting `0x20` borrows a high bit — a valid existence test once the
/// high bits are known clear), or `== 0x7f` (adding one carries into the high bit,
/// and cannot cross a byte boundary while every byte is `< 0x80`). The three terms
/// are OR-ed: the word is all-printable iff the result is zero. Endianness does not
/// matter, each byte is tested independently, so a native-order load is fine.
fn printable_run_len(bytes: &[u8]) -> usize {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    const LO_0X20: u64 = 0x2020_2020_2020_2020; // LO * 0x20

    let mut i = 0;
    while i + 8 <= bytes.len() {
        // `get`/`try_from` cannot fail here (the window is exactly eight bytes); the
        // `else` arms just keep the hot path panic-free.
        let Some(chunk) = bytes.get(i..i + 8) else {
            break;
        };
        let Ok(word) = <[u8; 8]>::try_from(chunk) else {
            break;
        };
        let x = u64::from_ne_bytes(word);
        let nonprintable = (x & HI) | (x.wrapping_sub(LO_0X20) & HI) | (x.wrapping_add(LO) & HI);
        if nonprintable != 0 {
            break;
        }
        i += 8;
    }
    // The word the SWAR loop stopped on, plus any tail shorter than eight bytes.
    while let Some(&b) = bytes.get(i) {
        if b.wrapping_sub(0x20) > 0x5e {
            break;
        }
        i += 1;
    }
    i
}

/// The actions the parser emits. A consumer implements this to interpret the
/// stream; `grid::Screen` does so to drive the terminal, and tests do so to
/// record the action sequence. Kept low-level (raw params, not pre-interpreted
/// commands) so no per-sequence allocation is needed and the parser stays purely
/// syntactic. The parser is generic over the implementor, so calls monomorphize.
pub trait Perform {
    /// A printable character (already UTF-8 decoded).
    fn print(&mut self, c: char);
    /// A run of printable ASCII bytes (each `0x20..=0x7e`, width 1), in order. The
    /// default prints them one at a time, so an implementor need not handle it; one
    /// that can write cells in bulk (`grid::Screen`) overrides this to skip the
    /// per-char width lookup and wrap math. `advance_bytes` emits it for plain-text
    /// runs, which is the common case in a terminal stream.
    fn print_ascii(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.print(char::from(b));
        }
    }
    /// A C0/C1 control byte to act on (BS, HT, LF, CR, ...).
    fn execute(&mut self, byte: u8);
    /// A complete CSI sequence: its numeric parameters, intermediate bytes, the
    /// private-marker byte if any (`?`/`<`/`=`/`>`, else 0), and the final byte.
    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], private: u8, action: u8);
    /// A complete escape sequence (not CSI/OSC): its intermediates and final byte.
    fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8);
    /// A complete DCS: the prologue's parameters, intermediates and final byte, plus the
    /// payload that followed it. `DCS $ q m ST` arrives as `([], [b'$'], b'q', b"m")`.
    ///
    /// The default does nothing, so an implementor that has no use for DCS (the test
    /// recorder, a headless consumer) is not made to care.
    fn dcs_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _action: u8, _data: &[u8]) {
    }
    /// A complete OSC string (the bytes between `ESC ]` and its terminator), and which
    /// terminator ended it: `true` for BEL (xterm's convention), `false` for ST.
    ///
    /// The terminator matters because an OSC *query* is answered with an OSC, and a
    /// client that sent BEL may only be looking for BEL. Mirroring what we were sent is
    /// the one answer that is right for both kinds of client.
    fn osc_dispatch(&mut self, data: &[u8], bel_terminated: bool);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Ground,
    Escape,
    EscapeIntermediate,
    CsiEntry,
    CsiParam,
    CsiIntermediate,
    CsiIgnore,
    OscString,
    /// DCS: the parameter/intermediate prologue, laid out exactly like a CSI's.
    DcsEntry,
    DcsParam,
    DcsIntermediate,
    /// DCS: past the final byte, passing the payload through to the string buffer.
    DcsPassthrough,
    /// A DCS we cannot use (too many intermediates, a malformed prologue): swallowed to
    /// the terminator so the payload cannot be mistaken for text.
    DcsIgnore,
    /// SOS/PM/APC: recognized and swallowed until ST. Nothing we implement rides on them
    /// (APC is the kitty graphics transport, and we draw no images).
    StringIgnore,
}

/// The parameters of a CSI sequence.
///
/// A parameter is **not** a number: it is a list of colon-separated *sub-parameters*,
/// of which the number everyone thinks of is merely the first. `SGR 4` is "underline";
/// `SGR 4:3` is "underline, style 3 (curly)". `SGR 38;2;255;0;0` and
/// `SGR 38:2::255:0:0` both say "foreground = red", the first as five parameters and
/// the second as one parameter with five sub-parameters. Both forms are in the wild —
/// which one a program emits depends on what its terminfo told it — so a terminal has
/// to read both.
///
/// ```text
///   CSI 4:3 ; 38:2::255:0:0 m
///       └┬┘   └──────┬─────┘
///        │           └── one parameter: [38, 2, 0, 255, 0, 0]
///        └────────────── one parameter: [4, 3]
/// ```
///
/// So `Params` is a flat value buffer plus the length of each parameter, and iterating
/// it yields `&[u16]` slices rather than numbers. Everything is fixed-size and
/// saturating: the child controls this input, and a sequence with ten thousand
/// sub-parameters must cost us nothing.
///
/// An omitted sub-parameter is zero, which is what the standard means by "default":
/// the empty slot in `38:2::255:0:0` is the (unused) colour-space id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Params {
    /// Every value of every parameter, laid end to end.
    values: [u16; MAX_PARAMS],
    /// Where each parameter's values begin in `values`, and how many it owns. The
    /// start is stored rather than derived: reading parameter `i` is the hot path (a
    /// CSI dispatch reads two or three of them, and SGR walks all of them), and
    /// summing the preceding lengths to find it would make that a quadratic walk.
    ///
    /// # Three optimizations that did not work
    ///
    /// This shape costs about 10% of the escape-heavy benchmark against the flat
    /// `[u16; 32]` it replaced. That is the price of parsing sub-parameters at all, and it
    /// bought a real bug: a colon used to make the parser drop the entire sequence, so an
    /// editor's curly underline took its diagnostic *colours* down with it.
    ///
    /// Three attempts to win it back, each measured by building two trees identical but
    /// for the change and running them interleaved on the same machine:
    ///
    /// 1. **One `bounds[i]..bounds[i + 1]` array instead of `starts` + `lens`.** Tidier,
    ///    and one store per parameter instead of two. **3.5% slower** (23.5k vs 22.7k).
    /// 2. **A flat fast path**, using `values[i]` directly until a colon is seen and
    ///    promoting only then. Restores the original single store. **12% slower** (25.5k).
    /// 3. **A `firsts[]` array**, so the first value of a parameter is one load rather than
    ///    a start-lookup plus a load. **7% slower** (24.3k).
    ///
    /// The lesson is in what they have in common: every one of them made *writing* a
    /// parameter cheaper and *reading* one dearer, and a terminal reads parameters far more
    /// often than it writes them. The stores were never where the time went. If you come
    /// back to this, come back with a profiler and start on the read path — and measure two
    /// trees against each other, because the benchmark moves 10% on code placement alone.
    starts: [u8; MAX_PARAMS],
    lens: [u8; MAX_PARAMS],
    /// Parameters, and values used.
    num: usize,
    total: usize,
}

impl Params {
    pub fn len(&self) -> usize {
        self.num
    }

    pub fn is_empty(&self) -> bool {
        self.num == 0
    }

    /// Parameter `i` with its sub-parameters; the first element is the value a caller
    /// that does not care about sub-parameters wants.
    ///
    /// Inlined: a CSI dispatch reads two or three parameters and SGR walks every one of
    /// them, so this sits directly on the escape-sequence hot path.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u16]> {
        if i >= self.num {
            return None;
        }
        let start = usize::from(*self.starts.get(i)?);
        let len = usize::from(*self.lens.get(i)?);
        self.values.get(start..start + len)
    }

    /// The first value of parameter `i`, or 0 when it is absent — the "default" every
    /// CSI parameter has.
    #[inline]
    pub fn value(&self, i: usize) -> u16 {
        // Not `get(i).first()`: the overwhelmingly common parameter is a single value,
        // and going through the slice makes the compiler re-derive the range for it.
        if i >= self.num {
            return 0;
        }
        match self.starts.get(i) {
            Some(&start) => self.values.get(usize::from(start)).copied().unwrap_or(0),
            None => 0,
        }
    }

    /// Each parameter in order, as a slice of its sub-parameters.
    pub fn iter(&self) -> impl Iterator<Item = &[u16]> {
        (0..self.num).filter_map(|i| self.get(i))
    }

    fn clear(&mut self) {
        self.num = 0;
        self.total = 0;
    }

    /// Finish the value being accumulated and add it to the parameter under
    /// construction. Silently dropped once the buffer is full, so a hostile sequence
    /// truncates rather than growing.
    #[inline]
    fn push_value(&mut self, value: u16, cur_len: &mut u8) {
        if let Some(slot) = self.values.get_mut(self.total) {
            *slot = value;
            self.total += 1;
            *cur_len = cur_len.saturating_add(1);
        }
    }

    /// Close the parameter under construction, recording where its values start.
    #[inline]
    fn push_param(&mut self, cur_len: &mut u8) {
        let len = *cur_len;
        *cur_len = 0;
        if len == 0 || self.num >= MAX_PARAMS {
            return;
        }
        let start = (self.total as u8).saturating_sub(len);
        if let (Some(s), Some(l)) = (self.starts.get_mut(self.num), self.lens.get_mut(self.num)) {
            *s = start;
            *l = len;
            self.num += 1;
        }
    }
}

/// The parser. Feed it bytes with [`Parser::advance`] (one byte) or
/// [`Parser::advance_bytes`] (a chunk); it calls back into a [`Perform`].
pub struct Parser {
    state: State,
    params: Params,
    /// The value being accumulated, and how many values the parameter under
    /// construction has taken so far.
    cur_param: u16,
    cur_len: u8,
    param_started: bool,
    intermediates: [u8; MAX_INTERMEDIATES],
    num_intermediates: usize,
    private: u8,
    ignore: bool,
    osc: Vec<u8>,
    /// The DCS payload (the bytes after the final byte, before ST), and the final byte
    /// that named the sequence. Separate from `osc` so the two cannot be confused; both
    /// are allocated once and reused, and both are capped.
    dcs: Vec<u8>,
    dcs_final: u8,
    /// UTF-8 decode: continuation bytes still expected, the value so far, and the
    /// smallest legal value for this length (to reject overlong encodings).
    utf8_remaining: u8,
    utf8_char: u32,
    utf8_min: u32,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            state: State::Ground,
            params: Params::default(),
            cur_param: 0,
            cur_len: 0,
            param_started: false,
            intermediates: [0; MAX_INTERMEDIATES],
            num_intermediates: 0,
            private: 0,
            ignore: false,
            osc: Vec::new(),
            dcs: Vec::new(),
            dcs_final: 0,
            utf8_remaining: 0,
            utf8_char: 0,
            utf8_min: 0,
        }
    }

    /// Feed a chunk of bytes. Reading the PTY in large chunks and handing the
    /// parser a slice (not a byte at a time) is the ingestion fast path.
    ///
    /// The bulk of a terminal stream is plain text, and in `Ground` a printable
    /// ASCII byte prints one char and changes no state. So when we are in `Ground`
    /// with no partial UTF-8 pending, we scan the whole run of printable ASCII with
    /// a SWAR sweep ([`printable_run_len`]) and print it directly, skipping the
    /// per-byte state-machine dispatch. Every other byte still goes through the full
    /// [`advance`](Self::advance) path, so behavior is byte-for-byte identical, this
    /// is purely a faster road for the common case (guarded by the golden suite and
    /// the 2M-byte fuzz test, which must stay green).
    pub fn advance_bytes<P: Perform>(&mut self, performer: &mut P, bytes: &[u8]) {
        let mut rest = bytes;
        while let Some(&first) = rest.first() {
            if self.state == State::Ground
                && self.utf8_remaining == 0
                && first.wrapping_sub(0x20) <= 0x5e
            {
                // `first` is printable, so the run is at least one byte; `rest`
                // strictly shrinks and the loop terminates.
                let n = printable_run_len(rest);
                performer.print_ascii(&rest[..n]);
                rest = &rest[n..];
            } else {
                self.advance(performer, first);
                rest = &rest[1..];
            }
        }
    }

    /// Feed one byte.
    pub fn advance<P: Perform>(&mut self, p: &mut P, byte: u8) {
        // Mid-UTF-8 (only possible in Ground): consume a continuation byte, or,
        // if this is not one, emit the replacement char and reprocess `byte`.
        if self.utf8_remaining > 0 {
            if byte & 0xc0 == 0x80 {
                self.utf8_char = (self.utf8_char << 6) | u32::from(byte & 0x3f);
                self.utf8_remaining -= 1;
                if self.utf8_remaining == 0 {
                    p.print(self.finish_utf8());
                }
                return;
            }
            self.utf8_remaining = 0;
            p.print('\u{FFFD}');
            // fall through: handle `byte` fresh
        }

        // CAN/SUB abort any sequence to ground; ESC (re)starts one, ending an OSC
        // string first. These act in every state.
        match byte {
            0x18 | 0x1a => {
                self.state = State::Ground;
                return;
            }
            0x1b => {
                if self.state == State::OscString {
                    // ESC ends the string; the ST that follows is consumed by the escape
                    // state as a no-op final byte. Not BEL, so an answer uses ST.
                    self.finish_osc(p, false);
                } else if self.state == State::DcsPassthrough {
                    self.finish_dcs(p);
                }
                self.clear();
                self.state = State::Escape;
                return;
            }
            _ => {}
        }

        match self.state {
            State::Ground => self.ground(p, byte),
            State::Escape => self.escape(p, byte),
            State::EscapeIntermediate => self.escape_intermediate(p, byte),
            State::CsiEntry => self.csi_entry(p, byte),
            State::CsiParam => self.csi_param(p, byte),
            State::CsiIntermediate => self.csi_intermediate(p, byte),
            State::CsiIgnore => self.csi_ignore(byte),
            State::OscString => self.osc_string(p, byte),
            State::DcsEntry => self.dcs_entry(byte),
            State::DcsParam => self.dcs_param(byte),
            State::DcsIntermediate => self.dcs_intermediate(byte),
            State::DcsPassthrough => self.dcs_passthrough(byte),
            State::DcsIgnore => {}
            State::StringIgnore => {}
        }
    }

    // ---- ground / UTF-8 -----------------------------------------------------

    fn ground<P: Perform>(&mut self, p: &mut P, byte: u8) {
        match byte {
            // ESC (0x1b), CAN (0x18), SUB (0x1a) are handled in `advance` before
            // reaching here; listed so the match stays exhaustive.
            0x18 | 0x1a | 0x1b => {}
            0x00..=0x17 | 0x19 | 0x1c..=0x1f => p.execute(byte),
            0x20..=0x7e => p.print(char::from(byte)),
            0x7f => {}                          // DEL: ignored in ground
            0x80..=0xbf => p.print('\u{FFFD}'), // stray continuation byte
            0xc0..=0xdf => self.utf8_begin(u32::from(byte & 0x1f), 1, 0x80),
            0xe0..=0xef => self.utf8_begin(u32::from(byte & 0x0f), 2, 0x800),
            0xf0..=0xf7 => self.utf8_begin(u32::from(byte & 0x07), 3, 0x1_0000),
            0xf8..=0xff => p.print('\u{FFFD}'), // invalid lead byte
        }
    }

    fn utf8_begin(&mut self, init: u32, remaining: u8, min: u32) {
        self.utf8_char = init;
        self.utf8_remaining = remaining;
        self.utf8_min = min;
    }

    /// Turn a completed code point into a char, rejecting overlong encodings and
    /// surrogates/out-of-range values as U+FFFD.
    fn finish_utf8(&self) -> char {
        if self.utf8_char < self.utf8_min {
            return '\u{FFFD}';
        }
        char::from_u32(self.utf8_char).unwrap_or('\u{FFFD}')
    }

    // ---- escape -------------------------------------------------------------

    fn escape<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if is_c0(byte) {
            p.execute(byte);
            return;
        }
        match byte {
            0x20..=0x2f => {
                self.collect_intermediate(byte);
                self.state = State::EscapeIntermediate;
            }
            0x50 => {
                self.clear();
                self.dcs.clear();
                self.state = State::DcsEntry; // DCS
            }
            0x58 | 0x5e | 0x5f => self.state = State::StringIgnore, // SOS / PM / APC
            0x5b => {
                self.clear();
                self.state = State::CsiEntry;
            }
            0x5d => {
                self.osc.clear();
                self.state = State::OscString;
            }
            // ST (`ESC \`) is a string *terminator*, not an escape sequence. The string
            // it closes has already been dispatched by the ESC that preceded this byte;
            // passing it on as `esc_dispatch(b'\\')` would be a phantom action every
            // consumer then has to know to ignore.
            0x5c => self.state = State::Ground,
            0x30..=0x7e => {
                p.esc_dispatch(&self.intermediates[..self.num_intermediates], byte);
                self.state = State::Ground;
            }
            _ => {} // 0x7f and anything else: ignore
        }
    }

    fn escape_intermediate<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if is_c0(byte) {
            p.execute(byte);
            return;
        }
        match byte {
            0x20..=0x2f => self.collect_intermediate(byte),
            0x30..=0x7e => {
                p.esc_dispatch(&self.intermediates[..self.num_intermediates], byte);
                self.state = State::Ground;
            }
            _ => {}
        }
    }

    // ---- CSI ----------------------------------------------------------------

    fn csi_entry<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if is_c0(byte) {
            p.execute(byte);
            return;
        }
        match byte {
            0x20..=0x2f => {
                self.collect_intermediate(byte);
                self.state = State::CsiIntermediate;
            }
            0x30..=0x39 => {
                self.param_digit(byte);
                self.state = State::CsiParam;
            }
            0x3a => {
                self.subparam_next();
                self.state = State::CsiParam;
            }
            0x3b => {
                self.param_next();
                self.state = State::CsiParam;
            }
            0x3c..=0x3f => {
                self.private = byte;
                self.state = State::CsiParam;
            }
            0x40..=0x7e => self.csi_dispatch(p, byte),
            _ => {
                // 0x7f (DEL) has no meaning inside a sequence; drop it.
                self.ignore = true;
                self.state = State::CsiIgnore;
            }
        }
    }

    fn csi_param<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if is_c0(byte) {
            p.execute(byte);
            return;
        }
        match byte {
            0x30..=0x39 => self.param_digit(byte),
            0x3a => self.subparam_next(),
            0x3b => self.param_next(),
            0x20..=0x2f => {
                self.collect_intermediate(byte);
                self.state = State::CsiIntermediate;
            }
            0x40..=0x7e => self.csi_dispatch(p, byte),
            _ => {
                // A private marker is only legal as the first byte after `[`, and DEL
                // is never legal: the sequence is malformed, so drop it.
                self.ignore = true;
                self.state = State::CsiIgnore;
            }
        }
    }

    fn csi_intermediate<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if is_c0(byte) {
            p.execute(byte);
            return;
        }
        match byte {
            0x20..=0x2f => self.collect_intermediate(byte),
            0x40..=0x7e => self.csi_dispatch(p, byte),
            _ => {
                // A parameter byte after an intermediate is malformed.
                self.ignore = true;
                self.state = State::CsiIgnore;
            }
        }
    }

    fn csi_ignore(&mut self, byte: u8) {
        // Consume until a final byte ends the (dropped) sequence.
        if (0x40..=0x7e).contains(&byte) {
            self.state = State::Ground;
        }
    }

    fn csi_dispatch<P: Perform>(&mut self, p: &mut P, byte: u8) {
        if self.param_started {
            self.push_param();
        }
        if !self.ignore {
            p.csi_dispatch(
                &self.params,
                &self.intermediates[..self.num_intermediates],
                self.private,
                byte,
            );
        }
        self.clear();
        self.state = State::Ground;
    }

    // ---- OSC ----------------------------------------------------------------

    fn osc_string<P: Perform>(&mut self, p: &mut P, byte: u8) {
        match byte {
            0x07 => {
                // BEL terminator (xterm convention).
                self.finish_osc(p, true);
                self.state = State::Ground;
            }
            0x00..=0x06 | 0x08..=0x17 | 0x19 | 0x1c..=0x1f => {} // ignore other controls
            _ => {
                if self.osc.len() < OSC_MAX {
                    self.osc.push(byte);
                }
            }
        }
    }

    /// Hand the collected OSC string to the performer, unless it hit the cap.
    ///
    /// A buffer sitting at exactly [`OSC_MAX`] is one we stopped filling, so we cannot
    /// know whether more was coming — and a truncated OSC is not a short OSC, it is a
    /// *different* one. The payload may be a hyperlink target (OSC 8), where half a URL
    /// still parses as a perfectly good URL pointing somewhere the child never named.
    /// So we drop it whole. The length *is* the overflow flag, which is why the parser
    /// carries no extra state for this: a `bool` field here cost `parse_escape` 7% (it
    /// reshaped `Parser` for the hot `advance` loop), and the price of deriving it
    /// instead is that a legitimate OSC of exactly 4096 bytes is dropped too — a title
    /// or URL that long is already past what we would honour.
    fn finish_osc<P: Perform>(&mut self, p: &mut P, bel_terminated: bool) {
        if self.osc.len() < OSC_MAX {
            p.osc_dispatch(&self.osc, bel_terminated);
        }
    }

    // ---- DCS ----------------------------------------------------------------
    //
    // A DCS is a CSI with a payload: `ESC P <params> <intermediates> <final> <data> ST`.
    // The prologue is parsed exactly like a CSI's — same parameters, same intermediates —
    // and the final byte names the sequence rather than performing it, because what
    // follows is the argument. `DCS $ q m ST` is "what are the current SGR settings?".

    fn dcs_entry(&mut self, byte: u8) {
        match byte {
            0x20..=0x2f => {
                self.collect_intermediate(byte);
                self.state = State::DcsIntermediate;
            }
            0x30..=0x39 => {
                self.param_digit(byte);
                self.state = State::DcsParam;
            }
            0x3a => {
                self.subparam_next();
                self.state = State::DcsParam;
            }
            0x3b => {
                self.param_next();
                self.state = State::DcsParam;
            }
            0x3c..=0x3f => {
                self.private = byte;
                self.state = State::DcsParam;
            }
            0x40..=0x7e => self.dcs_hook(byte),
            _ => self.state = State::DcsIgnore,
        }
    }

    fn dcs_param(&mut self, byte: u8) {
        match byte {
            0x30..=0x39 => self.param_digit(byte),
            0x3a => self.subparam_next(),
            0x3b => self.param_next(),
            0x20..=0x2f => {
                self.collect_intermediate(byte);
                self.state = State::DcsIntermediate;
            }
            0x40..=0x7e => self.dcs_hook(byte),
            _ => self.state = State::DcsIgnore,
        }
    }

    fn dcs_intermediate(&mut self, byte: u8) {
        match byte {
            0x20..=0x2f => self.collect_intermediate(byte),
            0x40..=0x7e => self.dcs_hook(byte),
            _ => self.state = State::DcsIgnore,
        }
    }

    /// The final byte: the prologue is complete, so remember what this DCS *is* and start
    /// collecting its payload.
    fn dcs_hook(&mut self, byte: u8) {
        if self.param_started {
            self.push_param();
        }
        if self.ignore {
            self.state = State::DcsIgnore;
            return;
        }
        self.dcs_final = byte;
        self.state = State::DcsPassthrough;
    }

    /// The payload. Capped like every other buffer the child fills: a DCS with a
    /// megabyte of payload must cost us nothing but the time to skip it.
    fn dcs_passthrough(&mut self, byte: u8) {
        if self.dcs.len() < DCS_MAX {
            self.dcs.push(byte);
        } else {
            // Over the cap: the sequence is unusable, so stop buffering and drop it whole
            // rather than acting on a truncated payload.
            self.state = State::DcsIgnore;
        }
    }

    fn finish_dcs<P: Perform>(&mut self, p: &mut P) {
        p.dcs_dispatch(
            &self.params,
            &self.intermediates[..self.num_intermediates],
            self.dcs_final,
            &self.dcs,
        );
    }

    // ---- parameter / intermediate bookkeeping -------------------------------

    fn clear(&mut self) {
        self.params.clear();
        self.cur_param = 0;
        self.cur_len = 0;
        self.param_started = false;
        self.num_intermediates = 0;
        self.private = 0;
        self.ignore = false;
    }

    fn param_digit(&mut self, byte: u8) {
        let digit = u16::from(byte - b'0');
        self.cur_param = self.cur_param.saturating_mul(10).saturating_add(digit);
        self.param_started = true;
    }

    /// A `:`: finalize the sub-parameter and stay inside the same parameter.
    fn subparam_next(&mut self) {
        self.params.push_value(self.cur_param, &mut self.cur_len);
        self.cur_param = 0;
        self.param_started = true;
    }

    /// A `;`: finalize the parameter (and whatever sub-parameter was open) and start
    /// the next one.
    fn param_next(&mut self) {
        self.params.push_value(self.cur_param, &mut self.cur_len);
        self.params.push_param(&mut self.cur_len);
        self.cur_param = 0;
        self.param_started = true;
    }

    /// The final byte: close whatever is still open.
    fn push_param(&mut self) {
        self.params.push_value(self.cur_param, &mut self.cur_len);
        self.params.push_param(&mut self.cur_len);
        self.cur_param = 0;
    }

    fn collect_intermediate(&mut self, byte: u8) {
        if self.num_intermediates < MAX_INTERMEDIATES {
            self.intermediates[self.num_intermediates] = byte;
            self.num_intermediates += 1;
        } else {
            self.ignore = true;
        }
    }
}

/// C0 controls that "execute" mid-sequence without ending it. Excludes ESC, CAN,
/// and SUB, which the top-level dispatch handles for every state.
fn is_c0(byte: u8) -> bool {
    matches!(byte, 0x00..=0x17 | 0x19 | 0x1c..=0x1f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recording [`Perform`]: a real implementation that logs the callbacks,
    /// so a test can assert the exact action sequence a byte stream produces.
    /// Not a mock (it invents no behavior); it is the "test oracle" the plan
    /// calls for.
    #[derive(Default)]
    struct Recorder {
        actions: Vec<Action>,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Action {
        Print(char),
        Execute(u8),
        Dcs {
            params: Vec<Vec<u16>>,
            intermediates: Vec<u8>,
            action: u8,
            data: Vec<u8>,
        },
        Csi {
            /// One entry per parameter, each holding its sub-parameters.
            params: Vec<Vec<u16>>,
            intermediates: Vec<u8>,
            private: u8,
            action: u8,
        },
        Esc {
            intermediates: Vec<u8>,
            byte: u8,
        },
        Osc(Vec<u8>),
    }

    impl Perform for Recorder {
        fn print(&mut self, c: char) {
            self.actions.push(Action::Print(c));
        }
        fn execute(&mut self, byte: u8) {
            self.actions.push(Action::Execute(byte));
        }
        fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], private: u8, action: u8) {
            self.actions.push(Action::Csi {
                params: params.iter().map(|p| p.to_vec()).collect(),
                intermediates: intermediates.to_vec(),
                private,
                action,
            });
        }
        fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8) {
            self.actions.push(Action::Esc {
                intermediates: intermediates.to_vec(),
                byte,
            });
        }
        fn osc_dispatch(&mut self, data: &[u8], _bel_terminated: bool) {
            self.actions.push(Action::Osc(data.to_vec()));
        }
        fn dcs_dispatch(&mut self, params: &Params, intermediates: &[u8], action: u8, data: &[u8]) {
            self.actions.push(Action::Dcs {
                params: params.iter().map(|p| p.to_vec()).collect(),
                intermediates: intermediates.to_vec(),
                action,
                data: data.to_vec(),
            });
        }
    }

    fn run(bytes: &[u8]) -> Vec<Action> {
        let mut p = Parser::new();
        let mut r = Recorder::default();
        p.advance_bytes(&mut r, bytes);
        r.actions
    }

    /// The reference path: feed one byte at a time through `advance`, bypassing the
    /// chunked fast path in `advance_bytes`. Any divergence between this and `run`
    /// is a fast-path bug.
    fn run_per_byte(bytes: &[u8]) -> Vec<Action> {
        let mut p = Parser::new();
        let mut r = Recorder::default();
        for &b in bytes {
            p.advance(&mut r, b);
        }
        r.actions
    }

    #[test]
    fn plain_ascii_prints() {
        assert_eq!(run(b"hi"), vec![Action::Print('h'), Action::Print('i')]);
    }

    #[test]
    fn c0_controls_execute() {
        assert_eq!(
            run(b"a\r\n"),
            vec![
                Action::Print('a'),
                Action::Execute(b'\r'),
                Action::Execute(b'\n')
            ]
        );
    }

    #[test]
    fn csi_with_params() {
        assert_eq!(
            run(b"\x1b[1;31m"),
            vec![Action::Csi {
                params: vec![vec![1], vec![31]],
                intermediates: vec![],
                private: 0,
                action: b'm',
            }]
        );
    }

    #[test]
    fn csi_no_params_is_empty() {
        assert_eq!(
            run(b"\x1b[H"),
            vec![Action::Csi {
                params: vec![],
                intermediates: vec![],
                private: 0,
                action: b'H',
            }]
        );
    }

    #[test]
    fn csi_empty_and_trailing_params_default() {
        // `;` produces a default (0) parameter on each side.
        assert_eq!(
            run(b"\x1b[;5H"),
            vec![Action::Csi {
                params: vec![vec![0], vec![5]],
                intermediates: vec![],
                private: 0,
                action: b'H',
            }]
        );
        assert_eq!(
            run(b"\x1b[5;H"),
            vec![Action::Csi {
                params: vec![vec![5], vec![0]],
                intermediates: vec![],
                private: 0,
                action: b'H',
            }]
        );
    }

    #[test]
    fn csi_private_marker() {
        assert_eq!(
            run(b"\x1b[?25h"),
            vec![Action::Csi {
                params: vec![vec![25]],
                intermediates: vec![],
                private: b'?',
                action: b'h',
            }]
        );
    }

    #[test]
    fn csi_intermediate() {
        // DECSCUSR: CSI Ps SP q
        assert_eq!(
            run(b"\x1b[2 q"),
            vec![Action::Csi {
                params: vec![vec![2]],
                intermediates: vec![b' '],
                private: 0,
                action: b'q',
            }]
        );
    }

    #[test]
    fn a_dcs_is_a_csi_with_a_payload() {
        // `ESC P <params> <intermediates> <final> <data> ST`. The prologue parses exactly
        // like a CSI's; the final byte names the sequence rather than performing it,
        // because what follows it is the argument.
        assert_eq!(
            run(b"\x1bP$qm\x1b\\"),
            vec![Action::Dcs {
                params: vec![],
                intermediates: vec![b'$'],
                action: b'q',
                data: b"m".to_vec(),
            }]
        );
        assert_eq!(
            run(b"\x1bP1;2+qabc\x1b\\"),
            vec![Action::Dcs {
                params: vec![vec![1], vec![2]],
                intermediates: vec![b'+'],
                action: b'q',
                data: b"abc".to_vec(),
            }]
        );
    }

    #[test]
    fn an_unterminated_dcs_never_leaks_its_payload_as_text() {
        // The hostile shape: a DCS that is never closed. Its bytes must not fall out onto
        // the screen, and the parser must not be stuck in it forever — the ESC that starts
        // anything else ends it.
        assert_eq!(run(b"\x1bPqpayload"), vec![]);
        assert_eq!(
            run(b"\x1bPqpayload\x1b[1mX"),
            vec![
                Action::Dcs {
                    params: vec![],
                    intermediates: vec![],
                    action: b'q',
                    data: b"payload".to_vec(),
                },
                Action::Csi {
                    params: vec![vec![1]],
                    intermediates: vec![],
                    private: 0,
                    action: b'm',
                },
                Action::Print('X'),
            ]
        );
    }

    #[test]
    fn a_dcs_payload_is_capped_like_every_other_buffer() {
        // The child writes this. A megabyte of payload must cost nothing but the time to
        // skip it, and an over-long one is dropped whole rather than acted on truncated.
        let mut input = b"\x1bP$q".to_vec();
        input.extend(std::iter::repeat_n(b'x', DCS_MAX * 2));
        input.extend_from_slice(b"\x1b\\");
        input.push(b'Z');
        assert_eq!(
            run(&input),
            vec![Action::Print('Z')],
            "dropped, and the stream recovers"
        );
    }

    #[test]
    fn colon_groups_sub_parameters_into_one_parameter() {
        // The ITU form: one parameter carrying its arguments as sub-parameters. This
        // used to drop the whole sequence, which meant a program underlining an error
        // with `SGR 4:3;58:2::255:0:0` lost the *colours* too, not just the style.
        assert_eq!(
            run(b"\x1b[38:2:1:2:3m"),
            vec![Action::Csi {
                params: vec![vec![38, 2, 1, 2, 3]],
                intermediates: vec![],
                private: 0,
                action: b'm',
            }]
        );
    }

    #[test]
    fn colons_and_semicolons_nest_the_way_the_standard_says() {
        // Semicolons separate parameters; colons separate the sub-parameters *within*
        // one. An empty sub-parameter is a zero, which is what "default" means here
        // (the empty slot in `38:2::r:g:b` is the colour-space id nobody uses).
        assert_eq!(
            run(b"\x1b[4:3;38:2::255:0:0m"),
            vec![Action::Csi {
                params: vec![vec![4, 3], vec![38, 2, 0, 255, 0, 0]],
                intermediates: vec![],
                private: 0,
                action: b'm',
            }]
        );
    }

    #[test]
    fn a_lone_colon_yields_a_defaulted_sub_parameter() {
        // Degenerate but legal shapes must not derail the parser.
        assert_eq!(
            run(b"\x1b[:1m"),
            vec![Action::Csi {
                params: vec![vec![0, 1]],
                intermediates: vec![],
                private: 0,
                action: b'm',
            }]
        );
        assert_eq!(
            run(b"\x1b[1:m"),
            vec![Action::Csi {
                params: vec![vec![1, 0]],
                intermediates: vec![],
                private: 0,
                action: b'm',
            }]
        );
    }

    #[test]
    fn a_flood_of_sub_parameters_is_bounded() {
        // The child writes this. A sequence with more values than we hold must truncate
        // and stay well-formed, never grow state and never trap.
        let mut input = b"\x1b[4".to_vec();
        for _ in 0..500 {
            input.extend_from_slice(b":9");
        }
        input.push(b'm');
        let actions = run(&input);
        match actions.as_slice() {
            [Action::Csi { params, action, .. }] => {
                assert_eq!(*action, b'm');
                assert_eq!(params.len(), 1, "still one parameter");
                let values = params.first().map(Vec::len).unwrap_or(0);
                assert!(
                    values <= MAX_PARAMS,
                    "{values} values, capped at {MAX_PARAMS}"
                );
                assert_eq!(params.first().and_then(|p| p.first()), Some(&4));
            }
            other => panic!("expected one CSI, got {other:?}"),
        }
    }

    #[test]
    fn esc_dispatch_and_charset() {
        assert_eq!(
            run(b"\x1bM"),
            vec![Action::Esc {
                intermediates: vec![],
                byte: b'M',
            }]
        );
        assert_eq!(
            run(b"\x1b(0"),
            vec![Action::Esc {
                intermediates: vec![b'('],
                byte: b'0',
            }]
        );
    }

    #[test]
    fn an_osc_too_long_for_the_buffer_is_dropped_whole() {
        // Truncating an OSC 8 payload at OSC_MAX would leave a prefix that still parses
        // as a perfectly good URL — a *different* URL from the one the child named, and
        // one we would then happily hand to xdg-open. A sequence we could not receive
        // whole was not received.
        let mut bytes = b"\x1b]8;;https://example.com/".to_vec();
        bytes.resize(bytes.len() + OSC_MAX, b'a');
        bytes.extend_from_slice(b"\x1b\\");
        let actions = run(&bytes);
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Osc(_))),
            "an overlong OSC dispatches nothing, {actions:?}"
        );

        // One byte under the cap still arrives, so the guard is a cap and not a wall.
        let mut ok = b"\x1b]0;".to_vec();
        ok.resize(OSC_MAX, b'a');
        ok.extend_from_slice(b"\x07");
        assert!(run(&ok).iter().any(|a| matches!(a, Action::Osc(_))));
    }

    #[test]
    fn osc_title_bel_and_st_terminated() {
        assert_eq!(
            run(b"\x1b]0;title\x07"),
            vec![Action::Osc(b"0;title".to_vec())]
        );
        // ST (ESC \) terminates too, and is not itself an action.
        assert_eq!(run(b"\x1b]2;hi\x1b\\"), vec![Action::Osc(b"2;hi".to_vec())]);
    }

    #[test]
    fn control_mid_csi_executes_without_breaking_it() {
        // A CR arriving mid-parameter acts immediately, then the CSI completes.
        assert_eq!(
            run(b"\x1b[1\r2H"),
            vec![
                Action::Execute(b'\r'),
                Action::Csi {
                    params: vec![vec![12]],
                    intermediates: vec![],
                    private: 0,
                    action: b'H',
                },
            ]
        );
    }

    #[test]
    fn utf8_multibyte_decodes() {
        // é (U+00E9, 2 bytes), 世 (U+4E16, 3 bytes), 😀 (U+1F600, 4 bytes).
        assert_eq!(
            run("é世😀".as_bytes()),
            vec![Action::Print('é'), Action::Print('世'), Action::Print('😀'),]
        );
    }

    #[test]
    fn utf8_malformed_becomes_replacement() {
        // A lead byte followed by a non-continuation: replacement, then the
        // stray byte is handled fresh.
        assert_eq!(
            run(&[0xc3, b'a']),
            vec![Action::Print('\u{FFFD}'), Action::Print('a')]
        );
        // A stray continuation byte alone.
        assert_eq!(run(&[0x80]), vec![Action::Print('\u{FFFD}')]);
        // Overlong encoding of '/' (0x2f) is rejected.
        assert_eq!(run(&[0xc0, 0xaf]), vec![Action::Print('\u{FFFD}')]);
    }

    #[test]
    fn param_values_saturate() {
        // A huge parameter must not overflow; it saturates at u16::MAX.
        let actions = run(b"\x1b[999999999H");
        assert_eq!(
            actions,
            vec![Action::Csi {
                params: vec![vec![u16::MAX]],
                intermediates: vec![],
                private: 0,
                action: b'H',
            }]
        );
    }

    #[test]
    fn too_many_params_are_bounded() {
        // Feeding far more than MAX_PARAMS separators must not grow state or panic.
        let mut input = b"\x1b[".to_vec();
        for _ in 0..1000 {
            input.push(b'1');
            input.push(b';');
        }
        input.push(b'm');
        let actions = run(&input);
        match &actions[..] {
            [Action::Csi { params, .. }] => assert!(params.len() <= MAX_PARAMS),
            other => panic!("expected one CSI, got {other:?}"),
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        // The standing hostile-input guard: a deterministic pseudo-random stream
        // of megabytes must parse without panicking. (A seeded LCG, no crate.)
        let mut p = Parser::new();
        let mut r = Recorder::default();
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..2_000_000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let byte = (seed >> 33) as u8;
            p.advance(&mut r, byte);
            // Keep the recorder from growing without bound over the whole run.
            r.actions.clear();
        }
    }

    #[test]
    fn printable_run_len_finds_ascii_runs() {
        assert_eq!(printable_run_len(b""), 0);
        assert_eq!(printable_run_len(b"hello"), 5);
        assert_eq!(printable_run_len(b"\nabc"), 0); // first byte non-printable
        assert_eq!(printable_run_len(b"hi\nthere"), 2); // LF stops the run
        assert_eq!(printable_run_len(b"ab\x7fcd"), 2); // DEL stops it
        assert_eq!(printable_run_len(b"ab\x1fcd"), 2); // a C0 control stops it
        assert_eq!(printable_run_len(b"ab\x80"), 2); // a UTF-8 high byte stops it
                                                     // The inclusive boundaries space (0x20) and tilde (0x7e) are printable.
        assert_eq!(printable_run_len(&[0x20, 0x7e]), 2);
        // Exactly one SWAR word, all printable.
        assert_eq!(printable_run_len(b"abcdefgh"), 8);
        // A non-printable exactly at the word boundary: the SWAR loop must stop at 8.
        assert_eq!(printable_run_len(b"abcdefgh\nij"), 8);
        // A control in the second word: 10 printable, then LF.
        assert_eq!(printable_run_len(b"abcdefghij\nkl"), 10);
        // A run spanning several words.
        assert_eq!(printable_run_len(&[b'x'; 100]), 100);
    }

    #[test]
    fn fast_path_matches_byte_at_a_time() {
        // The ASCII fast path in advance_bytes must be byte-for-byte identical to
        // feeding advance() one byte at a time. Build a deterministic stream biased
        // toward printable ASCII (long runs) but salted with the interesting bytes:
        // ESC, C0 controls, DEL, and UTF-8 lead/continuation/invalid bytes, so run
        // boundaries, SWAR word edges, and every state transition get exercised.
        let mut seed: u64 = 0xdead_beef_0bad_f00d;
        let mut bytes = Vec::with_capacity(50_000);
        for _ in 0..50_000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (seed >> 33) as u32;
            let b = match r % 16 {
                0 => 0x1b,                           // ESC (starts a sequence)
                1 => ((r >> 8) as u8) | 0x80,        // UTF-8 lead / continuation / invalid
                2 => ((r >> 8) as u8) & 0x1f,        // a C0 control
                3 => 0x7f,                           // DEL
                _ => 0x20 + ((r >> 8) % 0x5f) as u8, // printable ASCII 0x20..=0x7e
            };
            bytes.push(b);
        }
        // Whole slice at once.
        assert_eq!(run(&bytes), run_per_byte(&bytes));
        // And across arbitrary chunk splits: the PTY delivers arbitrary chunks, and
        // a run / escape sequence / multibyte char can straddle any boundary.
        let reference = run_per_byte(&bytes);
        for split in [1usize, 7, 8, 9, 63, 64, 100, 4096] {
            let mut p = Parser::new();
            let mut rec = Recorder::default();
            for chunk in bytes.chunks(split) {
                p.advance_bytes(&mut rec, chunk);
            }
            assert_eq!(rec.actions, reference, "chunk split {split}");
        }
    }
}
