//! The property lane's shared machinery: a seeded generator and a shrinker.
//!
//! Compiled only under `cfg(test)`. Two jobs, both about making a random find into
//! something a human can act on.
//!
//! **[`Rng`]** is the seeded LCG the fuzz loops draw from. It lives here because it was
//! previously four identical copies of the same two magic constants, and because the
//! seed is the reproduction: "reproducible, or it does not count" only holds if every
//! loop draws the same way.
//!
//! **[`shrink`]** is the half the fuzz lane was missing. A loop that asserts "it did not
//! panic" and throws the bytes away turns a real find into a sentence: the invariant
//! broke somewhere inside 1.2 MB of noise, 200 chunks deep, and the seed only reproduces
//! it until someone touches the generator — at which point the bug is loose again and
//! nobody knows. Shrinking closes that: a find comes back as the shortest byte string
//! that still breaks, which is small enough to read, paste into `tests/scenarios.rs` as
//! a permanent scenario, and fix. The find stops being an event and becomes a test.
//!
//! ```text
//!   1.2 MB of noise ─▶ invariant breaks ─▶ shrink ─▶ 6 bytes ─▶ paste into a scenario
//! ```

/// The seeded linear congruential generator the fuzz loops draw from.
///
/// The constants are Knuth's MMIX ones and the stream is taken from the *high* bits: an
/// LCG's low bits have famously short periods, so `seed as u8` would cycle through a
/// handful of values and fuzz nothing. Not cryptographic and not meant to be — the only
/// requirements are that it is cheap, needs no crate, and replays exactly.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// The next draw, from the high bits.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    pub(crate) fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    /// A draw in `0..n`. Panics on `n == 0`, which is a test-authoring bug rather than
    /// an input this has to survive.
    pub(crate) fn below(&mut self, n: u32) -> u32 {
        (self.next() as u32) % n
    }

    /// Fill `buf` with fresh bytes.
    pub(crate) fn fill(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.byte();
        }
    }
}

/// The CSI final bytes the grid actually dispatches, plus a few it does not.
///
/// Drawn from the real dispatch table on purpose: a generator that emits finals nothing
/// handles is testing the ignore path over and over. Most of these are handled — `c` and
/// `n` among them (DA and DSR, which also drive the reply path) — while `q` (DECLL) and
/// `t` (XTWINOPS) are the deliberate minority bnkterm ignores, because "consume, then
/// ignore" is a rule with its own bugs: an introducer that fails to consume its
/// parameters corrupts everything after it. Every byte here is a true final; a space
/// (0x20) is a CSI *intermediate*, not a final, so it has no place in this alphabet.
const CSI_FINALS: &[u8] = b"ABCDEFGHIJKLMPSTXZ@`abdfhlmrsucnqt";

/// The two-byte `ESC x` sequences worth reaching, and the charset designators.
const ESC_FINALS: &[u8] = b"78MDEHc=>";

/// Bytes that mean something on their own, weighted into the stream because a terminal
/// stream is mostly these and text.
///
/// `NUL`, `CAN`, `SUB` and `DEL` are here because leaving them out is what hid a class of
/// bug: they are the bytes whose whole job is to interact with a sequence in flight, so a
/// corpus without them exercises the parser's most cross-cutting rules zero times.
const C0: &[u8] = b"\r\n\t\x08\x0b\x0c\x0e\x0f\x00\x18\x1a\x7f";

/// Private modes worth toggling, because each one switches a *different* path on. Drawn
/// by name rather than from the generic parameter range, which tops out well below them:
/// a uniform generator emits `?2027` never, so the cluster-promotion path it enables was
/// unreachable no matter how many bytes were thrown at it.
const PRIVATE_MODES: &[u16] = &[
    1, 6, 7, 12, 25, 47, 1000, 1002, 1003, 1004, 1006, 1047, 1048, 1049, 2004, 2026, 2027, 2048,
];

/// The ANSI (unprefixed) modes worth toggling. IRM, insert mode, is the one that earns
/// the list: it takes the printer off its bulk path, so a corpus that never sets it
/// leaves the shifting path almost untested.
const ANSI_MODES: &[u16] = &[4, 20];

/// Sequence prologues to interrupt mid-flight. Each stops somewhere a parser has to hold
/// state — after an introducer, mid-parameter, after an intermediate — so that whatever
/// follows lands *inside* a sequence rather than between two.
const PROLOGUES: &[&[u8]] = &[
    b"\x1b[",
    b"\x1b[1",
    b"\x1b[?",
    b"\x1b[1;",
    b"\x1b[1 ",
    b"\x1bP",
    b"\x1bP$",
    b"\x1bP1;2",
    b"\x1bP+",
    b"\x1b]0;ab",
    b"\x1b(",
];

/// Bytes that interrupt a sequence, and the reason [`PROLOGUES`] exists. CAN and SUB
/// abort, `DEL` is skipped mid-sequence by xterm, `NUL` is ignored, and `ESC` restarts —
/// four different rules, each of which a corpus can only test by landing one of these
/// between an introducer and its final byte.
const INTERRUPTERS: &[u8] = b"\x18\x1a\x7f\x00\x1b\r\n\x07";

/// A generator of plausible terminal output.
///
/// Uniform random bytes are a poor fuzzer for a terminal and this is measurable rather
/// than a matter of taste: 1.2 MB of them contains about 15 `ESC [` pairs, of which 5
/// dispatch, reaching 5 of ~63 CSI finals. A parser's escape paths — where the bugs
/// live, because that is where the logic is — are effectively never entered. The stream
/// tests the printable path at great length and calls it coverage.
///
/// So this draws structure instead: mostly text, with well-formed escape sequences from
/// the real dispatch alphabet, wide glyphs, combining marks, and a minority of raw
/// garbage so the hostile-input contract still gets its exercise. The point is not
/// realism for its own sake — it is that `CSI 400 G` has to be *reachable* before any
/// number of bytes can find the bug behind it.
pub(crate) struct Stream(Rng);

impl Stream {
    pub(crate) fn new(seed: u64) -> Self {
        Stream(Rng::new(seed))
    }

    /// Append one weighted element to `out`.
    fn step(&mut self, out: &mut Vec<u8>) {
        match self.0.below(100) {
            // Text: the bulk of any real stream, and the bulk of this one.
            0..=41 => {
                let n = 1 + self.0.below(12);
                for _ in 0..n {
                    out.push(0x20 + self.0.byte() % 0x5f);
                }
            }
            42..=53 => out.push(C0[self.0.below(C0.len() as u32) as usize]),
            // A named private mode on or off. Each switches a different path on, and
            // several of them (insert mode, charset shifts, `?2027`) are what take the
            // grid *off* its bulk paths, so the slow paths only get tested at all when a
            // mode has been set first.
            54..=57 => {
                out.extend_from_slice(b"\x1b[");
                let mode = if self.0.below(4) == 0 {
                    // The ANSI modes, unprefixed. IRM (insert) is the one that matters:
                    // it takes the printer off its bulk path, so a corpus that never sets
                    // it barely tests the shifting path at all.
                    ANSI_MODES[self.0.below(ANSI_MODES.len() as u32) as usize]
                } else {
                    out.push(b'?');
                    PRIVATE_MODES[self.0.below(PRIVATE_MODES.len() as u32) as usize]
                };
                out.extend_from_slice(mode.to_string().as_bytes());
                out.push(if self.0.below(2) == 0 { b'h' } else { b'l' });
            }
            // A sequence cut off partway through by a byte that means something there.
            // The whole point is to land the interrupter *inside* a sequence: every other
            // arm emits a complete one, so the abort, skip and restart rules — the most
            // cross-cutting in the parser — were otherwise reached only by accident.
            58..=60 => {
                let p = PROLOGUES[self.0.below(PROLOGUES.len() as u32) as usize];
                out.extend_from_slice(p);
                out.push(INTERRUPTERS[self.0.below(INTERRUPTERS.len() as u32) as usize]);
                // Sometimes let the tail run on, so the parser has to decide whether the
                // sequence survived its interruption.
                if self.0.below(2) == 0 {
                    out.push(CSI_FINALS[self.0.below(CSI_FINALS.len() as u32) as usize]);
                }
            }
            // DCS: five parser states (`DcsEntry`, `DcsParam`, `DcsIntermediate`,
            // `DcsPassthrough`, `DcsIgnore`) that no other arm can enter. Both real users
            // are here — DECRQSS asking what a setting is, XTGETTCAP asking what terminfo
            // says — plus payloads that are neither.
            61..=64 => {
                out.extend_from_slice(b"\x1bP");
                match self.0.below(4) {
                    0 => {
                        out.extend_from_slice(b"$q");
                        out.extend_from_slice(match self.0.below(4) {
                            0 => b"m".as_slice(),
                            1 => b"r",
                            2 => b" q",
                            _ => b"\"p",
                        });
                    }
                    1 => {
                        out.extend_from_slice(b"+q");
                        for _ in 0..2 + self.0.below(6) {
                            out.push(b"0123456789abcdef"[self.0.below(16) as usize]);
                        }
                    }
                    2 => {
                        // Parameters and an intermediate, so the prologue states are
                        // walked rather than jumped over.
                        out.extend_from_slice(self.0.below(4).to_string().as_bytes());
                        out.push(b';');
                        out.extend_from_slice(self.0.below(4).to_string().as_bytes());
                        out.push(b'$');
                        out.push(b'q');
                    }
                    _ => {
                        for _ in 0..self.0.below(6) {
                            out.push(0x20 + self.0.byte() % 0x5f);
                        }
                    }
                }
                // Terminated properly, with a bare BEL, or not at all: an unterminated DCS
                // must stay bounded and must never leak its payload as text.
                match self.0.below(4) {
                    0 | 1 => out.extend_from_slice(b"\x1b\\"),
                    2 => out.push(0x07),
                    _ => {}
                }
            }
            // A well-formed CSI with small parameters. Small on purpose: the interesting
            // arithmetic is at the edges of the grid, not at 60000.
            65..=84 => {
                out.extend_from_slice(b"\x1b[");
                if self.0.below(8) == 0 {
                    out.push(b'?');
                }
                let params = self.0.below(3);
                for i in 0..params {
                    if i > 0 {
                        out.push(if self.0.below(6) == 0 { b':' } else { b';' });
                    }
                    let v = match self.0.below(4) {
                        0 => 0,
                        1 => 1 + self.0.below(9),
                        2 => 1 + self.0.below(40),
                        _ => self.0.below(300),
                    };
                    out.extend_from_slice(v.to_string().as_bytes());
                }
                out.push(CSI_FINALS[self.0.below(CSI_FINALS.len() as u32) as usize]);
            }
            85..=89 => {
                out.push(0x1b);
                match self.0.below(6) {
                    0 => out.extend_from_slice(b"(0"),
                    1 => out.extend_from_slice(b"(B"),
                    2 => out.extend_from_slice(b")0"),
                    3 => out.extend_from_slice(b"#8"),
                    _ => out.push(ESC_FINALS[self.0.below(ESC_FINALS.len() as u32) as usize]),
                }
            }
            // OSC, terminated both ways, and sometimes not at all.
            90..=92 => {
                out.extend_from_slice(b"\x1b]");
                out.extend_from_slice(self.0.below(12).to_string().as_bytes());
                out.push(b';');
                for _ in 0..self.0.below(8) {
                    out.push(0x20 + self.0.byte() % 0x5f);
                }
                match self.0.below(3) {
                    0 => out.push(0x07),
                    1 => out.extend_from_slice(b"\x1b\\"),
                    _ => {}
                }
            }
            // Wide glyphs and combining marks: the cell-pair and side-table paths that a
            // printable-ASCII stream never touches.
            93..=96 => {
                if self.0.below(6) == 0 {
                    self.promotion_beside_a_wide_glyph(out);
                    return;
                }
                let s = match self.0.below(5) {
                    0 => "\u{4e00}",
                    1 => "\u{3000}",
                    2 => "\u{1f980}",
                    3 => "e\u{301}",
                    _ => "\u{fe0f}",
                };
                out.extend_from_slice(s.as_bytes());
            }
            // Raw garbage, including invalid UTF-8: the hostile-input contract.
            _ => {
                for _ in 0..1 + self.0.below(4) {
                    out.push(self.0.byte());
                }
            }
        }
    }

    /// A narrow cluster sitting immediately left of a wide glyph, then the selector that
    /// makes it two columns wide.
    ///
    /// Every other arm emits its glyphs where the cursor happens to be, which is almost
    /// never *one column left of a wide pair* — so the column a growing cluster expands
    /// into was, in practice, always blank. That made the promotion a write into empty
    /// space and left the case where it displaces a leader (stranding that leader's spacer
    /// one column further right) unreachable at any stream length. The composite is
    /// deliberate: the placement, not the glyphs, is what the corpus could not produce.
    ///
    /// ```text
    ///   col:    n     n+1   n+2
    ///         [ ☀ ] [ 世 ] [spc]      ... then U+FE0F grows ☀ into n+1
    /// ```
    fn promotion_beside_a_wide_glyph(&mut self, out: &mut Vec<u8>) {
        let col = 1 + self.0.below(6);
        out.extend_from_slice(b"\x1b[");
        out.extend_from_slice((col + 1).to_string().as_bytes());
        out.push(b'G');
        out.extend_from_slice(match self.0.below(3) {
            0 => "\u{4e00}".as_bytes(),
            1 => "\u{3000}".as_bytes(),
            _ => "\u{1f980}".as_bytes(),
        });
        out.extend_from_slice(b"\x1b[");
        out.extend_from_slice(col.to_string().as_bytes());
        out.push(b'G');
        // A base that is narrow alone and wide once its selector lands, and a regional
        // indicator pair, which is the same promotion reached by a different rule.
        match self.0.below(3) {
            0 => out.extend_from_slice("\u{2600}\u{fe0f}".as_bytes()),
            1 => out.extend_from_slice("\u{1f1f3}\u{1f1f4}".as_bytes()),
            _ => out.extend_from_slice("#\u{fe0f}\u{20e3}".as_bytes()),
        }
    }

    /// A stream of roughly `len` bytes.
    pub(crate) fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 16);
        while out.len() < len {
            self.step(&mut out);
        }
        out
    }
}

/// How many predicate calls a shrink may make before it settles for what it has.
///
/// A bound rather than a promise of minimality: shrinking runs only on the failure path,
/// where the alternative is a human bisecting by hand, so "good enough, quickly" beats
/// "optimal, eventually". A 1.2 MB input reaches a few dozen bytes long before this.
const SHRINK_BUDGET: usize = 4_000;

/// The shortest subsequence of `input` this can find that still satisfies `fails`.
///
/// Delta debugging: try deleting one chunk at a time, coarse chunks first, and keep any
/// deletion that leaves the failure intact. When no chunk at the current granularity can
/// go, halve the chunk size and try again, down to single bytes.
///
/// ```text
///   [────────────────────────────]  n=2: try dropping each half
///   [────────────]                  it still fails, so keep the shorter one
///   [──────]                        n=4: finer, try each quarter
///   [─]                             ... down to single bytes
/// ```
///
/// `fails` must be deterministic, and it is called on inputs that are *not* failures, so
/// it has to be total — it returns `false` rather than panicking on a healthy input.
/// That is why the fuzz predicates here return `Option<String>` from a check instead of
/// asserting: an assertion cannot be asked "would this have failed?".
pub(crate) fn shrink(input: &[u8], fails: impl Fn(&[u8]) -> bool) -> Vec<u8> {
    let mut best = input.to_vec();
    let mut budget = SHRINK_BUDGET;
    // Chunks per pass. Grows (finer chunks) only once nothing coarser can be removed.
    let mut parts = 2usize;

    while best.len() > 1 && budget > 0 {
        let chunk = best.len().div_ceil(parts).max(1);
        let mut removed = false;

        let mut at = 0;
        while at < best.len() && budget > 0 {
            let end = (at + chunk).min(best.len());
            let mut candidate = Vec::with_capacity(best.len() - (end - at));
            candidate.extend_from_slice(best.get(..at).unwrap_or_default());
            candidate.extend_from_slice(best.get(end..).unwrap_or_default());
            budget -= 1;

            if !candidate.is_empty() && fails(&candidate) {
                best = candidate;
                removed = true;
                // Stay at this granularity: the input just got shorter, so the same
                // chunk index now covers different bytes and is worth retrying.
            } else {
                at = end;
            }
        }

        if !removed {
            if chunk == 1 {
                break;
            }
            parts = (parts * 2).min(best.len().max(2));
        }
    }
    best
}

/// Render `bytes` as a Rust byte-string literal, so a shrunk reproducer can be pasted
/// into a test without anyone hand-escaping it.
///
/// Escapes anything that is not plainly printable ASCII, which for a terminal stream is
/// most of it.
pub(crate) fn as_byte_literal(bytes: &[u8]) -> String {
    let mut out = String::from("b\"");
    for &b in bytes {
        match b {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            0x20..=0x7e => out.push(char::from(b)),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use crate::fuzz::{as_byte_literal, shrink, Rng, Stream, INTERRUPTERS};

    /// The stream has to replay exactly, or a seed in a failure report is worthless.
    #[test]
    fn the_generator_replays_from_its_seed() {
        let mut a = Rng::new(0xDEAD_BEEF_CAFE_1234);
        let mut b = Rng::new(0xDEAD_BEEF_CAFE_1234);
        for _ in 0..1000 {
            assert_eq!(a.byte(), b.byte());
        }
        let mut c = Rng::new(1);
        let mut d = Rng::new(2);
        let (x, y): (Vec<u8>, Vec<u8>) = (
            (0..64).map(|_| c.byte()).collect(),
            (0..64).map(|_| d.byte()).collect(),
        );
        assert_ne!(x, y, "different seeds give different streams");
    }

    /// Drawing from the high bits is the whole reason this is not `seed as u8`: an LCG's
    /// low bit alternates, so a low-bits generator fuzzes almost nothing.
    #[test]
    fn the_generator_covers_the_byte_range() {
        let mut r = Rng::new(0x1234_5678_9abc_def0);
        let mut seen = [false; 256];
        for _ in 0..20_000 {
            seen[usize::from(r.byte())] = true;
        }
        assert!(seen.iter().all(|&s| s), "every byte value is reachable");
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            assert!(r.below(5) < 5);
            assert!(r.below(1) == 0);
        }
    }

    #[test]
    fn fill_matches_repeated_bytes() {
        let mut a = Rng::new(99);
        let mut b = Rng::new(99);
        let mut buf = [0u8; 32];
        a.fill(&mut buf);
        for want in buf {
            assert_eq!(b.byte(), want);
        }
    }

    /// The shrinker's own guard: a needle buried in a haystack comes back as the needle.
    /// This is the workflow the fuzz lane depends on, so it cannot be dormant code that
    /// has never been run.
    #[test]
    fn shrinking_finds_the_needle_in_a_haystack() {
        let mut input = vec![b'a'; 5000];
        input.splice(2500..2500, *b"\x1b[9;9H");
        let minimal = shrink(&input, |b| b.windows(6).any(|w| w == b"\x1b[9;9H"));
        assert_eq!(minimal, b"\x1b[9;9H", "shrank to exactly what fails");
    }

    /// Two independent causes both have to survive: a shrinker that deletes one of them
    /// would report a reproducer that does not reproduce.
    #[test]
    fn shrinking_keeps_every_byte_the_failure_needs() {
        let input = b"xxxAxxxxBxxx".to_vec();
        let minimal = shrink(&input, |b| b.contains(&b'A') && b.contains(&b'B'));
        assert_eq!(minimal, b"AB");
    }

    /// A predicate nothing can satisfy any smaller leaves the input alone, and a
    /// shrinker must not return an empty "reproducer" that reproduces nothing.
    #[test]
    fn shrinking_an_irreducible_input_returns_it() {
        let input = b"abcd".to_vec();
        let minimal = shrink(&input, |b| b == b"abcd");
        assert_eq!(minimal, b"abcd");
    }

    /// Shrinking terminates on a predicate that is true of everything, rather than
    /// walking to an empty input or spinning.
    #[test]
    fn shrinking_always_terminates() {
        let input = vec![b'z'; 1000];
        let minimal = shrink(&input, |_| true);
        assert_eq!(minimal.len(), 1, "down to the smallest non-empty input");
    }

    /// The generator's reach, pinned — because a fuzzer that cannot reach a bug will
    /// never find it, however many bytes it burns, and nothing else in the suite would
    /// notice it had stopped reaching.
    ///
    /// The measured baseline this replaced: 1.2 MB of uniform random bytes dispatches
    /// about 5 CSI sequences, over 5 distinct finals. Removing a cursor clamp from `CHA`
    /// did not make it fail, because it never emitted a single `CSI n G`.
    #[test]
    fn the_stream_reaches_the_escape_paths() {
        let bytes = Stream::new(0xDEAD_BEEF_CAFE_1234).bytes(200_000);

        let mut finals = std::collections::BTreeSet::new();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == 0x1b && bytes[i + 1] == b'[' {
                let mut j = i + 2;
                while j < bytes.len() && matches!(bytes[j], 0x20..=0x3f) {
                    j += 1;
                }
                if let Some(&f) = bytes.get(j) {
                    if (0x40..=0x7e).contains(&f) {
                        finals.insert(f);
                    }
                }
                i = j + 1;
            } else {
                i += 1;
            }
        }

        assert!(
            finals.len() >= 30,
            "the stream dispatched only {} distinct CSI finals: {:?}",
            finals.len(),
            finals.iter().map(|&f| char::from(f)).collect::<Vec<_>>()
        );
        // The paths that only a real terminal stream reaches.
        let has = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
        assert!(has(b"\x1b(0"), "the line-drawing charset is reachable");
        assert!(has(b"\x1b7") || has(b"\x1b8"), "DECSC/DECRC are reachable");
        assert!(has(b"\x1b]"), "OSC is reachable");
        assert!(has("\u{4e00}".as_bytes()), "wide glyphs are reachable");
        assert!(has("\u{301}".as_bytes()), "combining marks are reachable");
    }

    /// The states a corpus can miss entirely while still looking thorough.
    ///
    /// Every assertion here was once a zero. Measured over the exact stream the grid fuzz
    /// uses, the generator emitted no `ESC P` at all, so five DCS parser states were never
    /// entered; no `?2027`, so the cluster-promotion path was unreachable; and none of
    /// NUL, CAN, SUB or DEL, so the parser's most cross-cutting rules — abort, skip,
    /// restart — went untested. Three real bugs were sitting behind those gaps.
    ///
    /// A count, not a boolean, because "reached once in 1.2 MB" is not coverage: it is a
    /// state the shrinker will almost never re-find.
    #[test]
    fn the_stream_reaches_the_states_a_uniform_corpus_never_would() {
        let bytes = Stream::new(0xDEAD_BEEF_CAFE_1234).bytes(300 * 4096);
        let count = |needle: &[u8]| bytes.windows(needle.len()).filter(|w| *w == needle).count();
        let byte_count = |b: u8| bytes.iter().filter(|&&x| x == b).count();

        // DCS: `DcsEntry`, `DcsParam`, `DcsIntermediate`, `DcsPassthrough`, `DcsIgnore`.
        assert!(count(b"\x1bP") > 500, "DCS: {}", count(b"\x1bP"));
        assert!(count(b"$q") > 100, "DECRQSS: {}", count(b"$q"));
        assert!(count(b"+q") > 100, "XTGETTCAP: {}", count(b"+q"));

        // The modes that switch the grid off its bulk paths.
        assert!(count(b"2027") > 20, "?2027: {}", count(b"2027"));
        assert!(count(b"\x1b[4h") > 5, "insert mode: {}", count(b"\x1b[4h"));

        // A promotable cluster placed one column left of a wide glyph. Reachability here
        // is about *placement*, not alphabet: wide glyphs and selectors were always in
        // the stream, but never arranged so that a growing cluster had to displace one.
        let beside =
            count(b"G\xe4\xb8\x80") + count(b"G\xe3\x80\x80") + count(b"G\xf0\x9f\xa6\x80");
        assert!(
            beside > 200,
            "cluster promotion beside a wide glyph: {beside}"
        );

        // The bytes whose entire job is to interact with a sequence in flight.
        for (name, b) in [("NUL", 0x00), ("CAN", 0x18), ("SUB", 0x1a), ("DEL", 0x7f)] {
            assert!(byte_count(b) > 100, "{name}: {}", byte_count(b));
        }

        // And they have to land *inside* a sequence, not merely appear. Count the ones
        // that follow an introducer with no final byte between.
        let interrupted = bytes
            .windows(3)
            .filter(|w| {
                w[0] == 0x1b && (w[1] == b'[' || w[1] == b'P') && INTERRUPTERS.contains(&w[2])
            })
            .count();
        assert!(
            interrupted > 100,
            "sequences interrupted before their final byte: {interrupted}"
        );
    }

    #[test]
    fn a_reproducer_prints_as_a_pastable_literal() {
        assert_eq!(as_byte_literal(b"ab"), "b\"ab\"");
        assert_eq!(as_byte_literal(b"\x1b[9;9H"), "b\"\\x1b[9;9H\"");
        assert_eq!(as_byte_literal(b"a\nb"), "b\"a\\nb\"");
        assert_eq!(as_byte_literal(b"q\"\\"), "b\"q\\\"\\\\\"");
        assert_eq!(as_byte_literal(b"\xff"), "b\"\\xff\"");
    }
}
