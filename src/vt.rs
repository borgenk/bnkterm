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

/// Whether the first UTF-8 lead is still capable of completing validly with the
/// bytes already present. A missing continuation is not invalid—the next PTY
/// chunk may provide it—but a present out-of-range continuation keeps the
/// malformed sequence on the scalar oracle instead of opening a batch.
fn utf8_prefix_may_complete(bytes: &[u8]) -> bool {
    let Some(&lead) = bytes.first() else {
        return false;
    };
    let (length, lo, hi) = match lead {
        0xc2..=0xdf => (2usize, 0x80, 0xbf),
        0xe0 => (3, 0xa0, 0xbf),
        0xe1..=0xec | 0xee | 0xef => (3, 0x80, 0xbf),
        0xed => (3, 0x80, 0x9f),
        0xf0 => (4, 0x90, 0xbf),
        0xf1..=0xf3 => (4, 0x80, 0xbf),
        0xf4 => (4, 0x80, 0x8f),
        _ => return false,
    };
    let Some(&second) = bytes.get(1) else {
        return true;
    };
    if !(lo..=hi).contains(&second) {
        return false;
    }
    for index in 2..length {
        let Some(&continuation) = bytes.get(index) else {
            return true;
        };
        if !(0x80..=0xbf).contains(&continuation) {
            return false;
        }
    }
    true
}

/// The actions the parser emits. A consumer implements this to interpret the
/// stream; `grid::Screen` does so to drive the terminal, and tests do so to
/// record the action sequence. Kept low-level (raw params, not pre-interpreted
/// commands) so no per-sequence allocation is needed and the parser stays purely
/// syntactic. The parser is generic over the implementor, so calls monomorphize.
pub trait Perform {
    /// A printable character (already UTF-8 decoded).
    fn print(&mut self, c: char);
    /// A bounded run of already-decoded printable scalars. The default preserves
    /// the scalar action contract exactly; consumers such as the grid may
    /// override it when one run is a more useful unit of work.
    fn print_run(&mut self, chars: &[char]) {
        for &c in chars {
            self.print(c);
        }
    }
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

/// Scalars buffered between the parser and a [`Perform`] consumer. A fixed
/// initialized array keeps the hot path allocation-free without adding an
/// `unsafe` uninitialized-storage boundary. The capacity is an experiment knob,
/// not a terminal semantic limit: a full run is flushed and immediately resumed.
const TEXT_RUN_CAP: usize = 128;
/// A substantial ASCII suffix returns to the established SWAR + byte-grid path.
/// Short spaces and punctuation stay in their surrounding Unicode run; source
/// identifiers and prose words keep the faster ASCII specialization.
const ASCII_HANDOFF: usize = 8;

struct TextRun {
    chars: [char; TEXT_RUN_CAP],
    len: usize,
}

impl TextRun {
    fn new() -> Self {
        TextRun {
            chars: ['\0'; TEXT_RUN_CAP],
            len: 0,
        }
    }

    fn push<P: Perform>(&mut self, performer: &mut P, c: char) {
        if self.len >= self.chars.len() {
            self.flush(performer);
        }
        if let Some(slot) = self.chars.get_mut(self.len) {
            *slot = c;
            self.len += 1;
        }
    }

    fn flush<P: Perform>(&mut self, performer: &mut P) {
        if self.len == 0 {
            return;
        }
        if let Some(chars) = self.chars.get(..self.len) {
            performer.print_run(chars);
        }
        self.len = 0;
    }
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
    /// Whether more parameters or values arrived than this could hold.
    ///
    /// Recorded rather than shrugged off, because a *truncated* sequence is not a
    /// shorter version of the sequence — it is a different one. `SGR 38;2;255;0;0` cut
    /// after `38;2` does not mean "no colour", it means red becomes
    /// `Rgb(0,0,0)`: black text, which on most themes is invisible. An introducer owns
    /// the parameters that follow it, so honouring a prefix of one is the same class of
    /// bug as failing to consume it (see `Screen::sgr`), reached from the other side.
    /// The dispatch refuses the whole sequence instead.
    overflowed: bool,
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
        self.overflowed = false;
    }

    /// Finish the value being accumulated and add it to the parameter under
    /// construction. Silently dropped once the buffer is full, so a hostile sequence
    /// truncates rather than growing.
    #[inline]
    fn push_value(&mut self, value: u16, cur_len: &mut u8) {
        let Some(slot) = self.values.get_mut(self.total) else {
            self.overflowed = true;
            return;
        };
        *slot = value;
        self.total += 1;
        *cur_len = cur_len.saturating_add(1);
    }

    /// Close the parameter under construction, recording where its values start.
    #[inline]
    fn push_param(&mut self, cur_len: &mut u8) {
        let len = *cur_len;
        *cur_len = 0;
        if len == 0 {
            return;
        }
        if self.num >= MAX_PARAMS {
            self.overflowed = true;
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
    /// The byte range the *next* continuation may fall in.
    ///
    /// Usually `0x80..=0xbf`, but four lead bytes narrow it (Unicode Table 3-7), and
    /// that narrowing is the whole of UTF-8's well-formedness: `E0` restricted to
    /// `A0..BF` is what makes an overlong three-byte encoding unrepresentable rather
    /// than merely detectable afterwards, and `ED` restricted to `80..9F` is what makes
    /// a surrogate unrepresentable. Checking the range up front, instead of the value
    /// after the fact, is also what makes the *count* of U+FFFDs right (see `ground`).
    utf8_next: (u8, u8),
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
            utf8_next: (0x80, 0xbf),
        }
    }

    /// Feed a chunk of bytes. Reading the PTY in large chunks and handing the
    /// parser a slice (not a byte at a time) is the ingestion fast path.
    ///
    /// The bulk of a terminal stream is plain text. Ground-state printable ASCII
    /// keeps its established SWAR scan and byte-run callback. A multibyte scalar
    /// starts a bounded decoded run that may include following printable ASCII and
    /// ends at a control byte or input boundary. That run is decoded exactly once:
    /// no `str` validation pass precedes it.
    ///
    /// Controls and every non-ground byte still go through [`advance`](Self::advance),
    /// keeping the escape state machine physically separate from the larger text
    /// decoder. The scalar callback remains the run callback's default oracle, and
    /// the golden, split-invariance, and fuzz suites require both roads to agree.
    pub fn advance_bytes<P: Perform>(&mut self, performer: &mut P, bytes: &[u8]) {
        let mut rest = bytes;
        let mut text_run = None;
        while let Some(&first) = rest.first() {
            if self.state == State::Ground {
                if self.utf8_remaining == 0 && first.wrapping_sub(0x20) <= 0x5e {
                    // `first` is printable, so the run is at least one byte; `rest`
                    // strictly shrinks and the loop terminates.
                    let n = printable_run_len(rest);
                    if let Some(run) = rest.get(..n) {
                        performer.print_ascii(run);
                    }
                    rest = rest.get(n..).unwrap_or_default();
                    continue;
                }
                let continuation = self.utf8_remaining > 0
                    && (self.utf8_next.0..=self.utf8_next.1).contains(&first);
                if continuation || utf8_prefix_may_complete(rest) {
                    let run = text_run.get_or_insert_with(TextRun::new);
                    let consumed = self.advance_text_run(performer, rest, run);
                    if consumed > 0 {
                        rest = rest.get(consumed..).unwrap_or_default();
                    }
                    // A zero-byte result can still have emitted U+FFFD for a
                    // partial sequence. Retrying the same byte lets the ordinary
                    // ASCII/control branch own it with the corrected UTF-8 state.
                    continue;
                }
            }
            self.advance(performer, first);
            rest = rest.get(1..).unwrap_or_default();
        }
    }

    /// Flush whatever end-of-input makes final. Call once, after the last byte the child
    /// will ever write (a PTY EOF or a read error), and never at an ordinary read
    /// boundary.
    ///
    /// The only such state is a UTF-8 sequence split across a read:
    /// [`advance_bytes`](Self::advance_bytes) holds it for the continuation a live stream
    /// delivers in the next read (see `a_sequence_cut_by_a_chunk_boundary_waits_for_the_rest`).
    /// Once the child has exited there is no next read, so the held bytes are the stream's
    /// final maximal subpart and become a single U+FFFD (Unicode 16 §3.9), rather than
    /// vanishing. A half-built escape sequence names no character and so leaves nothing to
    /// show.
    ///
    /// Returns whether it printed anything, so a caller can tell the grid changed.
    pub fn finish<P: Perform>(&mut self, p: &mut P) -> bool {
        if self.utf8_remaining == 0 {
            return false;
        }
        self.utf8_remaining = 0;
        self.utf8_next = (0x80, 0xbf);
        p.print('\u{FFFD}');
        true
    }

    /// Feed one byte.
    pub fn advance<P: Perform>(&mut self, p: &mut P, byte: u8) {
        // Mid-UTF-8 (only possible in Ground): consume a continuation byte, or,
        // if this is not one, emit the replacement char and reprocess `byte`.
        if self.utf8_remaining > 0 {
            let (lo, hi) = self.utf8_next;
            if (lo..=hi).contains(&byte) {
                self.utf8_char = (self.utf8_char << 6) | u32::from(byte & 0x3f);
                self.utf8_remaining -= 1;
                // Only the second byte is ever restricted; the rest are plain
                // continuations.
                self.utf8_next = (0x80, 0xbf);
                if self.utf8_remaining == 0 {
                    p.print(self.finish_utf8());
                }
                return;
            }
            // The sequence ends here, at its maximal well-formed subpart, and `byte`
            // was never part of it: it is reprocessed below rather than swallowed.
            // That is what puts the `(` back in `e1 28 a1` -> U+FFFD `(` U+FFFD.
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

    /// Decode one ground-state text run into `run`, stopping before a C0
    /// control. A malformed continuation emits the replacement for the partial
    /// sequence, then reprocesses the offending byte exactly as [`advance`] does.
    ///
    /// `#[inline(never)]` keeps this larger loop out of the escape dispatcher:
    /// the escape-heavy benchmark is sensitive to code layout even when a text
    /// branch never executes.
    #[inline(never)]
    fn advance_text_run<P: Perform>(
        &mut self,
        performer: &mut P,
        bytes: &[u8],
        run: &mut TextRun,
    ) -> usize {
        let mut consumed = 0usize;
        while let Some(&byte) = bytes.get(consumed) {
            if self.utf8_remaining > 0 {
                let (lo, hi) = self.utf8_next;
                if (lo..=hi).contains(&byte) {
                    self.utf8_char = (self.utf8_char << 6) | u32::from(byte & 0x3f);
                    self.utf8_remaining -= 1;
                    self.utf8_next = (0x80, 0xbf);
                    consumed += 1;
                    if self.utf8_remaining == 0 {
                        let decoded = self.finish_utf8();
                        run.push(performer, decoded);
                    }
                    continue;
                }
                self.utf8_remaining = 0;
                run.push(performer, '\u{fffd}');
                // Stop before `byte`: it was not part of the malformed sequence,
                // and the scalar oracle should own the ill-formed boundary.
                break;
            }

            if byte.wrapping_sub(0x20) <= 0x5e {
                let tail = bytes.get(consumed..).unwrap_or_default();
                let ascii = printable_run_len(tail);
                if ascii >= ASCII_HANDOFF {
                    break;
                }
                if let Some(short) = tail.get(..ascii) {
                    for &value in short {
                        run.push(performer, char::from(value));
                    }
                }
                consumed = consumed.saturating_add(ascii);
                continue;
            }

            match byte {
                0x00..=0x1f => break,
                // Caught by the ASCII handoff above; retained for exhaustiveness.
                0x20..=0x7e => run.push(performer, char::from(byte)),
                0x7f => {} // DEL is ignored without breaking adjacent text.
                0x80..=0xbf | 0xc0 | 0xc1 | 0xf5..=0xff => run.push(performer, '\u{fffd}'),
                0xc2..=0xdf => self.utf8_begin(u32::from(byte & 0x1f), 1, (0x80, 0xbf)),
                0xe0 => self.utf8_begin(u32::from(byte & 0x0f), 2, (0xa0, 0xbf)),
                0xe1..=0xec => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0xbf)),
                0xed => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0x9f)),
                0xee | 0xef => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0xbf)),
                0xf0 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x90, 0xbf)),
                0xf1..=0xf3 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x80, 0xbf)),
                0xf4 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x80, 0x8f)),
            }
            consumed += 1;
        }
        run.flush(performer);
        consumed
    }

    fn ground<P: Perform>(&mut self, p: &mut P, byte: u8) {
        match byte {
            // ESC (0x1b), CAN (0x18), SUB (0x1a) are handled in `advance` before
            // reaching here; listed so the match stays exhaustive.
            0x18 | 0x1a | 0x1b => {}
            0x00..=0x17 | 0x19 | 0x1c..=0x1f => p.execute(byte),
            0x20..=0x7e => p.print(char::from(byte)),
            0x7f => {}                          // DEL: ignored in ground
            0x80..=0xbf => p.print('\u{FFFD}'), // stray continuation byte
            // Unicode Table 3-7, transcribed. The four narrowed ranges are the
            // interesting rows: `E0` and `F0` exclude the overlong encodings, `ED`
            // excludes the surrogates, and `F4` stops at U+10FFFF.
            0xc0 | 0xc1 => p.print('\u{FFFD}'), // no non-overlong two-byte form exists
            0xc2..=0xdf => self.utf8_begin(u32::from(byte & 0x1f), 1, (0x80, 0xbf)),
            0xe0 => self.utf8_begin(u32::from(byte & 0x0f), 2, (0xa0, 0xbf)),
            0xe1..=0xec => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0xbf)),
            0xed => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0x9f)),
            0xee | 0xef => self.utf8_begin(u32::from(byte & 0x0f), 2, (0x80, 0xbf)),
            0xf0 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x90, 0xbf)),
            0xf1..=0xf3 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x80, 0xbf)),
            0xf4 => self.utf8_begin(u32::from(byte & 0x07), 3, (0x80, 0x8f)),
            0xf5..=0xff => p.print('\u{FFFD}'), // past U+10FFFF, or not a lead at all
        }
    }

    fn utf8_begin(&mut self, init: u32, remaining: u8, next: (u8, u8)) {
        self.utf8_char = init;
        self.utf8_remaining = remaining;
        self.utf8_next = next;
    }

    /// Turn a completed code point into a char.
    ///
    /// No overlong or surrogate check is needed here, and its absence is the point: the
    /// per-lead ranges in `ground` mean an ill-formed value can never be accumulated in
    /// the first place. The `unwrap_or` is unreachable and kept only because the
    /// no-panic rule does not make exceptions for reasoning.
    fn finish_utf8(&self) -> char {
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
        // An overflowed sequence is refused entire, not honoured up to the cap. See
        // `Params::overflowed`: a prefix of an extended-colour introducer is not a
        // smaller request, it is a wrong one.
        if !self.ignore && !self.params.overflowed {
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
        // Refused entire when the prologue overflowed, exactly as a CSI is: a DCS's
        // parameters are laid out like a CSI's and carry the same hazard, and nothing
        // reads them today only because the DCS sequences bnkterm answers happen to be
        // identified by their intermediates and final byte alone. Leaving the asymmetry
        // in would hand a truncated `Params` to whoever first writes one that does.
        if self.params.overflowed {
            return;
        }
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
        decoded_runs: Vec<Vec<char>>,
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
        /// The payload, and whether BEL (rather than ST) ended it. The terminator is
        /// part of the sequence, so a recorder that drops it cannot round-trip.
        Osc(Vec<u8>, bool),
    }

    impl Perform for Recorder {
        fn print(&mut self, c: char) {
            self.actions.push(Action::Print(c));
        }
        fn print_run(&mut self, chars: &[char]) {
            self.decoded_runs.push(chars.to_vec());
            for &c in chars {
                self.print(c);
            }
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
        fn osc_dispatch(&mut self, data: &[u8], bel_terminated: bool) {
            self.actions
                .push(Action::Osc(data.to_vec(), bel_terminated));
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

    impl Action {
        /// Write this action back out as the bytes that would produce it.
        ///
        /// The inverse of parsing, which is what makes it an oracle: see
        /// [`round_trip`].
        fn encode(&self, out: &mut Vec<u8>) {
            match self {
                Action::Print(c) => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                Action::Execute(b) => out.push(*b),
                Action::Csi {
                    params,
                    intermediates,
                    private,
                    action,
                } => {
                    out.extend_from_slice(b"\x1b[");
                    if *private != 0 {
                        out.push(*private);
                    }
                    encode_params(params, out);
                    out.extend_from_slice(intermediates);
                    out.push(*action);
                }
                Action::Esc {
                    intermediates,
                    byte,
                } => {
                    out.push(0x1b);
                    out.extend_from_slice(intermediates);
                    out.push(*byte);
                }
                Action::Osc(data, bel) => {
                    out.extend_from_slice(b"\x1b]");
                    out.extend_from_slice(data);
                    if *bel {
                        out.push(0x07);
                    } else {
                        out.extend_from_slice(b"\x1b\\");
                    }
                }
                Action::Dcs {
                    params,
                    intermediates,
                    action,
                    data,
                } => {
                    out.extend_from_slice(b"\x1bP");
                    encode_params(params, out);
                    out.extend_from_slice(intermediates);
                    out.push(*action);
                    out.extend_from_slice(data);
                    out.extend_from_slice(b"\x1b\\");
                }
            }
        }
    }

    /// Parameters as they were written: sub-parameters joined by `:`, parameters by `;`.
    fn encode_params(params: &[Vec<u16>], out: &mut Vec<u8>) {
        for (i, p) in params.iter().enumerate() {
            if i > 0 {
                out.push(b';');
            }
            for (j, v) in p.iter().enumerate() {
                if j > 0 {
                    out.push(b':');
                }
                out.extend_from_slice(v.to_string().as_bytes());
            }
        }
    }

    /// Parse `src`, write the actions back out, and assert the bytes come back
    /// identical.
    ///
    /// This is a **structurally different oracle** from the golden suite, and the
    /// difference is the whole point. A golden tests `bytes → grid`, so it can only see
    /// what the parser *kept*: parse `CSI 4:3 m` as a plain underline, drop the `3` that
    /// says "curly", and every golden stays green because the grid has no way to show
    /// what was thrown away. Round-tripping asks the parser to write down what it
    /// understood, and a dropped parameter cannot survive that — the bytes come back
    /// shorter than they went in.
    ///
    /// It needs no expected value at all, which is the other half of the point: the
    /// property is internal, so there is no answer key to bless a bug into.
    fn round_trip(src: &[u8]) -> Vec<Action> {
        let actions = run(src);
        let mut out = Vec::new();
        for a in &actions {
            a.encode(&mut out);
        }
        assert!(
            out == src,
            "round trip changed the bytes\n  in:  {:?}\n  out: {:?}\n  actions: {actions:?}",
            String::from_utf8_lossy(src),
            String::from_utf8_lossy(&out),
        );
        actions
    }

    /// A round trip that lands on `canonical` rather than back on `src`.
    ///
    /// Not an escape hatch for bugs: it names the places where two spellings mean the
    /// same thing and the parser deliberately keeps only one of them. Each use is a
    /// claim that the difference carries no meaning, and it has to be argued at the call
    /// site — an omitted parameter *is* zero, so `CSI ;4m` and `CSI 0;4m` are the same
    /// sequence, and only one of them can come back.
    fn round_trip_as(src: &[u8], canonical: &[u8]) -> Vec<Action> {
        let actions = run(src);
        let mut out = Vec::new();
        for a in &actions {
            a.encode(&mut out);
        }
        assert!(
            out == canonical,
            "round trip did not canonicalize as expected\n  in:   {:?}\n  want: {:?}\n  got:  {:?}",
            String::from_utf8_lossy(src),
            String::from_utf8_lossy(canonical),
            String::from_utf8_lossy(&out),
        );
        actions
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
        // The child writes this. A sequence with more values than we hold is refused
        // whole: it must never grow state, never trap, and never arrive truncated —
        // a prefix of `38:2:...` is a *wrong* colour, not a smaller request.
        let mut input = b"\x1b[4".to_vec();
        for _ in 0..500 {
            input.extend_from_slice(b":9");
        }
        input.push(b'm');
        assert_eq!(
            run(&input),
            vec![],
            "the overflowed sequence dispatched nothing"
        );

        // Right up to the cap it is still a sequence, and still says what it said.
        let mut input = b"\x1b[4".to_vec();
        for _ in 0..MAX_PARAMS - 1 {
            input.extend_from_slice(b":9");
        }
        input.push(b'm');
        match run(&input).as_slice() {
            [Action::Csi { params, action, .. }] => {
                assert_eq!(*action, b'm');
                assert_eq!(params.len(), 1, "still one parameter");
                assert_eq!(params.first().map(Vec::len), Some(MAX_PARAMS));
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
            !actions.iter().any(|a| matches!(a, Action::Osc(..))),
            "an overlong OSC dispatches nothing, {actions:?}"
        );

        // One byte under the cap still arrives, so the guard is a cap and not a wall.
        let mut ok = b"\x1b]0;".to_vec();
        ok.resize(OSC_MAX, b'a');
        ok.extend_from_slice(b"\x07");
        assert!(run(&ok).iter().any(|a| matches!(a, Action::Osc(..))));
    }

    #[test]
    fn osc_title_bel_and_st_terminated() {
        assert_eq!(
            run(b"\x1b]0;title\x07"),
            vec![Action::Osc(b"0;title".to_vec(), true)]
        );
        // ST (ESC \) terminates too, and is not itself an action.
        assert_eq!(
            run(b"\x1b]2;hi\x1b\\"),
            vec![Action::Osc(b"2;hi".to_vec(), false)]
        );
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
    }

    /// How many U+FFFDs an ill-formed sequence becomes, transcribed from the tables that
    /// answer it. Not a detail: the count is a *column count*, so a stream of broken
    /// UTF-8 shifts everything after it by however many the terminal got wrong.
    ///
    /// Unicode 16 §3.9 defines a maximal subpart as the longest subsequence at an
    /// unconvertible offset that is either the initial subsequence of some well-formed
    /// sequence, or one byte. `C0` begins nothing well-formed (Table 3-7 starts at `C2`),
    /// so it is one byte, and the `AF` after it is another — two replacements, not one
    /// for the pair. The tables below say so in as many words.
    #[test]
    fn ill_formed_utf8_yields_one_replacement_per_maximal_subpart() {
        let printed = |bytes: &[u8]| -> String {
            run(bytes)
                .iter()
                .filter_map(|a| match a {
                    Action::Print(c) => Some(*c),
                    _ => None,
                })
                .collect()
        };

        // Table 3-8, non-shortest (overlong) forms.
        assert_eq!(printed(&[0xc0, 0xaf]), "\u{FFFD}\u{FFFD}");
        assert_eq!(printed(&[0xe0, 0x80, 0xbf]), "\u{FFFD}\u{FFFD}\u{FFFD}");
        assert_eq!(
            printed(&[0xf0, 0x81, 0x82, 0x41]),
            "\u{FFFD}\u{FFFD}\u{FFFD}A"
        );

        // Table 3-9, surrogates and values past U+10FFFF.
        assert_eq!(printed(&[0xed, 0xa0, 0x80]), "\u{FFFD}\u{FFFD}\u{FFFD}");
        assert_eq!(printed(&[0xed, 0xbf, 0xbf]), "\u{FFFD}\u{FFFD}\u{FFFD}");
        assert_eq!(
            printed(&[0xf4, 0x91, 0x92, 0x93, 0xff]),
            "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}"
        );
        assert_eq!(printed(&[0x41, 0x80, 0xbf, 0x42]), "A\u{FFFD}\u{FFFD}B");

        // Table 3-10, truncated sequences. Here a maximal subpart is *two* bytes: `E1 80`
        // is the start of something well-formed, so it collapses to one replacement
        // rather than two.
        assert_eq!(
            printed(&[0xe1, 0x80, 0xe2, 0xf0, 0x91, 0x92, 0xf1, 0xbf, 0x41]),
            "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}A"
        );

        // And the edges of Table 3-7 itself, which is what makes the above fall out.
        assert_eq!(
            printed(&[0xc2, 0x80]),
            "\u{80}",
            "the smallest two-byte form"
        );
        assert_eq!(
            printed(&[0xe0, 0xa0, 0x80]),
            "\u{800}",
            "the smallest three-byte"
        );
        assert_eq!(
            printed(&[0xed, 0x9f, 0xbf]),
            "\u{d7ff}",
            "just below the surrogates"
        );
        assert_eq!(printed(&[0xee, 0x80, 0x80]), "\u{e000}", "just above them");
        assert_eq!(
            printed(&[0xf4, 0x8f, 0xbf, 0xbf]),
            "\u{10ffff}",
            "the last code point"
        );
    }

    /// A sequence cut by a *chunk* boundary is held, not replaced.
    ///
    /// This is where bnkterm parts company with §3.9's letter, which calls a truncated
    /// sequence at the end of the stream ill-formed and replaces it. The distinction is
    /// end of *chunk* versus end of *stream*: a PTY read boundary lands wherever the
    /// kernel put it, and the rest of the crab is in the next read, so replacing at the
    /// chunk edge would corrupt every wide character unlucky enough to straddle one — and
    /// would break `every_scenario_is_invariant_to_chunking`, the property that says so.
    /// The real end of the stream is honored instead by [`Parser::finish`], which the
    /// next test covers.
    #[test]
    fn a_sequence_cut_by_a_chunk_boundary_waits_for_the_rest() {
        let mut p = Parser::new();
        let mut r = Recorder::default();
        p.advance_bytes(&mut r, &[0xf0, 0x9f]); // half a crab
        assert_eq!(
            r.actions,
            vec![],
            "nothing is emitted for a partial sequence"
        );
        p.advance_bytes(&mut r, &[0xa6, 0x80]); // the other half
        assert_eq!(r.actions, vec![Action::Print('\u{1f980}')]);
    }

    /// The mirror of the chunk-boundary case: at genuine end of input the held bytes can
    /// never be completed, so `finish` makes them final (one U+FFFD for the whole held
    /// subpart, not one per byte), and finishing a clean stream emits nothing.
    #[test]
    fn a_sequence_cut_by_end_of_stream_is_replaced_by_finish() {
        let mut p = Parser::new();
        let mut r = Recorder::default();
        p.advance_bytes(&mut r, &[0xf0, 0x9f]); // half a crab, and no more is coming
        assert_eq!(r.actions, vec![], "still held while the stream is open");
        assert!(p.finish(&mut r), "the held bytes flush at end of stream");
        assert_eq!(r.actions, vec![Action::Print('\u{FFFD}')]);
        // Idempotent, and a no-op once nothing is pending.
        assert!(!p.finish(&mut r), "nothing pending, nothing emitted");
        assert_eq!(r.actions, vec![Action::Print('\u{FFFD}')]);

        // A stream that ends cleanly (or with a half-built escape, which names no
        // character) has nothing to flush.
        let mut p = Parser::new();
        let mut r = Recorder::default();
        p.advance_bytes(&mut r, b"hi\x1b["); // trailing CSI introducer, no final byte
        assert!(!p.finish(&mut r));
        assert_eq!(r.actions, vec![Action::Print('h'), Action::Print('i')]);
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
        // Feeding far more than MAX_PARAMS separators must not grow state or panic, and
        // must not dispatch a truncated sequence either (see `Params::overflowed`).
        let mut input = b"\x1b[".to_vec();
        for _ in 0..1000 {
            input.push(b'1');
            input.push(b';');
        }
        input.push(b'm');
        assert_eq!(
            run(&input),
            vec![],
            "the overflowed sequence dispatched nothing"
        );

        // The text after it is still text: the sequence was consumed, not leaked.
        let mut input = input.clone();
        input.push(b'X');
        assert_eq!(run(&input), vec![Action::Print('X')]);

        // The parser is not left poisoned: the next sequence dispatches normally.
        let mut p = Parser::new();
        let mut r = Recorder::default();
        p.advance_bytes(&mut r, &input);
        p.advance_bytes(&mut r, b"\x1b[31m");
        assert!(
            matches!(r.actions.last(), Some(Action::Csi { action: b'm', .. })),
            "recovered: {:?}",
            r.actions
        );
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
    fn multibyte_text_is_one_bounded_decoded_run() {
        let mut parser = Parser::new();
        let mut recorder = Recorder::default();
        parser.advance_bytes(&mut recorder, "é = 日本\x1b[31mafter".as_bytes());
        assert_eq!(
            recorder.decoded_runs,
            [vec!['é', ' ', '=', ' ', '日', '本']]
        );
        assert_eq!(
            recorder.actions,
            [
                Action::Print('é'),
                Action::Print(' '),
                Action::Print('='),
                Action::Print(' '),
                Action::Print('日'),
                Action::Print('本'),
                Action::Csi {
                    params: vec![vec![31]],
                    intermediates: Vec::new(),
                    private: 0,
                    action: b'm',
                },
                Action::Print('a'),
                Action::Print('f'),
                Action::Print('t'),
                Action::Print('e'),
                Action::Print('r'),
            ]
        );
    }

    #[test]
    fn decoded_run_resumes_a_scalar_split_across_input_chunks() {
        let crab = "🦀!";
        let mut parser = Parser::new();
        let mut recorder = Recorder::default();
        parser.advance_bytes(&mut recorder, &crab.as_bytes()[..2]);
        assert!(recorder.decoded_runs.is_empty());
        parser.advance_bytes(&mut recorder, &crab.as_bytes()[2..]);
        assert_eq!(recorder.decoded_runs, [vec!['🦀', '!']]);
    }

    #[test]
    fn decoded_run_capacity_is_a_flush_not_a_semantic_limit() {
        let text = "é".repeat(TEXT_RUN_CAP + 17);
        let mut parser = Parser::new();
        let mut recorder = Recorder::default();
        parser.advance_bytes(&mut recorder, text.as_bytes());
        assert_eq!(recorder.decoded_runs.len(), 2);
        assert_eq!(
            recorder.decoded_runs.first().map(Vec::len),
            Some(TEXT_RUN_CAP)
        );
        assert_eq!(recorder.decoded_runs.get(1).map(Vec::len), Some(17));
        assert_eq!(recorder.actions.len(), TEXT_RUN_CAP + 17);
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

    /// The parser writes back exactly what it read, for every sequence bnkterm claims to
    /// understand.
    ///
    /// This is the oracle the golden suite structurally cannot be. A golden watches
    /// `bytes -> grid`, so it only ever sees what the parser *kept*: drop the `3` from
    /// `CSI 4:3 m` and the cell is still underlined, every golden is still green, and the
    /// curly underline an LSP asked for is silently straight forever. Here the bytes come
    /// back short and it fails immediately.
    #[test]
    fn every_sequence_round_trips() {
        // Text and C0.
        round_trip(b"hello");
        round_trip(b"a\r\nb\tc\x08d");
        round_trip("wide \u{4e00} mark e\u{301} crab \u{1f980}".as_bytes());

        // CSI: no parameters, one, several, and a private marker.
        round_trip(b"\x1b[m");
        round_trip(b"\x1b[0m");
        round_trip(b"\x1b[1;31m");
        round_trip(b"\x1b[H");
        round_trip(b"\x1b[1;2H");
        round_trip(b"\x1b[?25h");
        round_trip(b"\x1b[?1049l");
        round_trip(b"\x1b[3J");
        round_trip(b"\x1b[2K");
        round_trip(b"\x1b[10X");
        round_trip(b"\x1b[5P");
        round_trip(b"\x1b[1;10r");

        // Sub-parameters: the whole reason a parameter is a list and not a number.
        round_trip(b"\x1b[4:3m");
        round_trip(b"\x1b[4:0m");
        round_trip(b"\x1b[58:2:0:255:0:255m");
        round_trip(b"\x1b[38;2;1;2;3m");
        round_trip(b"\x1b[38;5;196m");
        round_trip_as(b"\x1b[0;32;58:2::1:2:3;3m", b"\x1b[0;32;58:2:0:1:2:3;3m");

        // Intermediates.
        round_trip(b"\x1b[1 q");
        round_trip(b"\x1b[?2026$p");

        // ESC, including the charset designators.
        round_trip(b"\x1b7");
        round_trip(b"\x1b8");
        round_trip(b"\x1bM");
        round_trip(b"\x1bD");
        round_trip(b"\x1bE");
        round_trip(b"\x1bc");
        round_trip(b"\x1b(0");
        round_trip(b"\x1b(B");
        round_trip(b"\x1b)0");
        round_trip(b"\x1b#8");

        // OSC, both terminators. The terminator is part of the sequence: a client that
        // sent BEL may only be listening for BEL.
        round_trip(b"\x1b]0;a title\x07");
        round_trip(b"\x1b]2;a title\x1b\\");
        round_trip(b"\x1b]8;id=x;http://example.com/\x1b\\");
        round_trip(b"\x1b]11;?\x07");

        // DCS.
        round_trip(b"\x1bP$qm\x1b\\");
        round_trip(b"\x1bP$qr\x1b\\");
        round_trip(b"\x1bP+q544e\x1b\\");

        // A realistic mixed stream.
        round_trip(b"\x1b[?1049h\x1b[H\x1b[2J\x1b[1;32mok\x1b[0m\r\n\x1b[?25l");
    }

    /// The three places where two spellings mean one thing, and the parser keeps one.
    ///
    /// Each is a claim that the difference carries no meaning, and each is argued rather
    /// than accepted: this is the escape hatch that would quietly swallow a real bug if
    /// it were used to make a failure go away.
    #[test]
    fn round_trips_that_canonicalize_say_why() {
        // An omitted parameter *is* zero (ECMA-48: an empty parameter takes its default,
        // and SGR's default is 0). `CSI ;4m` and `CSI 0;4m` are the same sequence, so
        // only one of them can come back.
        round_trip_as(b"\x1b[;4m", b"\x1b[0;4m");
        round_trip_as(b"\x1b[1;;2m", b"\x1b[1;0;2m");

        // The same rule one level down: the empty colour-space slot in the colon form of
        // truecolour is an omitted sub-parameter, which is zero.
        round_trip_as(b"\x1b[38:2::1:2:3m", b"\x1b[38:2:0:1:2:3m");

        // A parameter is a number, and numbers do not remember how they were written.
        round_trip_as(b"\x1b[00003;2H", b"\x1b[3;2H");
        round_trip_as(b"\x1b[0007m", b"\x1b[7m");
    }

    /// The oracle's own guard: a parser that drops a sub-parameter must fail this, and
    /// the assertion below is what proves the test can tell.
    ///
    /// Without this, `every_sequence_round_trips` could be vacuous — passing because the
    /// encoder faithfully re-encodes whatever the parser produced, however wrong.
    #[test]
    fn a_dropped_sub_parameter_cannot_survive_a_round_trip() {
        // What the parser actually does: keeps both, so the bytes return intact.
        let actions = round_trip(b"\x1b[4:3m");
        let Some(Action::Csi { params, .. }) = actions.first() else {
            panic!("expected a CSI, got {actions:?}");
        };
        assert_eq!(
            params,
            &vec![vec![4, 3]],
            "the sub-parameter survived parsing"
        );

        // What a parser that dropped the `3` would produce, and what it would encode back
        // to: `CSI 4m`, which is four bytes where five went in.
        let dropped = Action::Csi {
            params: vec![vec![4]],
            intermediates: vec![],
            private: 0,
            action: b'm',
        };
        let mut out = Vec::new();
        dropped.encode(&mut out);
        assert_eq!(out, b"\x1b[4m", "the loss is visible in the bytes");
        assert_ne!(
            out, b"\x1b[4:3m",
            "which is exactly what the round trip catches"
        );
    }
}
