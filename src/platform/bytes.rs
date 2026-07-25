//! Byte-scan primitives, hand-widened with SWAR (eight bytes per `u64`) rather
//! than left to the autovectoriser, and with no intrinsics, no `unsafe`, and no
//! dependency.
//!
//! Printable ASCII (`0x20..=0x7e`) is asked about at both ends of the pipeline, by
//! stages that never meet:
//!
//! ```text
//!   PTY bytes ─▶ vt::Parser ─▶ grid ─▶ display list ─▶ render::gpu ─▶ vertices
//!                    │                                      │
//!            printable_run_len                        all_printable
//!            how far the ground-state              may this fixed-pitch run
//!            fast lane may run                     skip grapheme segmentation
//! ```
//!
//! Each reaches that set on its own reasoning — the parser's is stated at
//! [`crate::vt::Parser::advance_bytes`], the batcher's at
//! [`crate::render::gpu`]'s fixed-pitch path — and each stays correct however the
//! other is written. What they share is a *definition* and therefore a kernel:
//! writing the SWAR sweep twice would be two copies of a bit trick to get right,
//! for one set neither stage disagrees about.
//!
//! The sweep is written out rather than left to the optimiser because an early-exit
//! search is the loop shape LLVM reliably declines to widen. The batcher's previous
//! `.bytes().all(…)` measured 2.39 us over a 120x80 frame's runs where this measures
//! 0.35 us, including what SIMD intrinsics would and would not add.

/// Whether a single byte is printable ASCII (`0x20..=0x7e`).
///
/// One unsigned compare after a wrapping subtract rejects `b < 0x20` (which wraps
/// high), `b == 0x7f` (DEL, landing on `0x5f`), and `b >= 0x80` (landing at `0x60`
/// or above) together. This is the scalar statement of what [`printable_run_len`]
/// tests eight at a time, and the definition both of them answer to.
#[inline]
pub fn is_printable(b: u8) -> bool {
    b.wrapping_sub(0x20) <= 0x5e
}

/// Length of the leading run of printable ASCII (`0x20..=0x7e`) in `bytes`.
///
/// We sweep eight bytes at a time with SWAR and fall to [`is_printable`] at the
/// first word that isn't all-printable (and for the < 8-byte tail). Within a word,
/// a byte is non-printable iff it is `>= 0x80` (`& HI`), or `< 0x20` (subtracting
/// `0x20` borrows a high bit — a valid existence test once the high bits are known
/// clear), or `== 0x7f` (adding one carries into the high bit, and cannot cross a
/// byte boundary while every byte is `< 0x80`). The three terms are OR-ed: the word
/// is all-printable iff the result is zero. Endianness does not matter, each byte is
/// tested independently, so a native-order load is fine.
#[inline]
pub fn printable_run_len(bytes: &[u8]) -> usize {
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
        if !is_printable(b) {
            break;
        }
        i += 1;
    }
    i
}

/// Whether every byte of `bytes` is printable ASCII (`0x20..=0x7e`). Vacuously true
/// for an empty slice.
///
/// The all-or-nothing question and [`printable_run_len`]'s how-far question are one
/// scan: a run that covers the whole slice is the answer to both. Nothing is lost by
/// sharing the kernel, because the caller that asks this expects `true` — an ordinary
/// line of terminal text — and the affirmative case reads every byte either way.
#[inline]
pub fn all_printable(bytes: &[u8]) -> bool {
    printable_run_len(bytes) == bytes.len()
}

/// A bounds-checked forward cursor over a borrowed byte slice: the shared
/// mechanics behind the wire and store readers. It only hands out sub-slices and
/// advances past them, never decoding integers itself, so each format layers its
/// own endianness and error messages on top.
///
/// That separation is load-bearing: `wire` reads native-endian and `store` reads
/// little-endian, so their decoders are only interchangeable on little-endian
/// targets. This program is exactly that (`ffi.rs` restricts it to Linux
/// x86_64/arm64), but keeping the integer decoding in each wrapper means this
/// cursor stays correct even if that ever changes.
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// The next `n` bytes, advancing the cursor past them, or `None` when fewer
    /// than `n` bytes remain (the cursor is left unmoved). The caller turns
    /// `None` into its own format-specific error.
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    /// How many bytes have been consumed so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// How many bytes remain unread.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The SWAR sweep must agree with the one-byte definition on *every* byte value,
    /// at every offset within a word. A word-parallel trick that is wrong for one
    /// value in one lane is exactly the bug this kernel could plausibly have, and it
    /// would surface as a stray cell rather than a crash.
    #[test]
    fn the_swar_sweep_agrees_with_the_scalar_definition_everywhere() {
        for value in 0u8..=255 {
            assert_eq!(
                is_printable(value),
                (0x20..=0x7e).contains(&value),
                "is_printable({value:#04x})"
            );
            // Place `value` at each lane of a word, padded either side with a byte
            // known printable, so a lane-shifted borrow or carry cannot hide.
            for lane in 0..16 {
                let mut buf = [b'x'; 24];
                buf[lane] = value;
                let want = if is_printable(value) { 24 } else { lane };
                assert_eq!(printable_run_len(&buf), want, "{value:#04x} at lane {lane}");
                assert_eq!(all_printable(&buf), is_printable(value));
            }
        }
    }

    #[test]
    fn all_printable_is_the_whole_slice_answering_the_run_scan() {
        assert!(all_printable(b"")); // vacuously
        assert!(all_printable(b"plain ascii line ~"));
        assert!(!all_printable(b"tab\there"));
        assert!(!all_printable("héllo".as_bytes()));
        // Length classes around the eight-byte word, since the tail is a separate loop.
        for len in 0..40usize {
            let good = vec![b'a'; len];
            assert!(all_printable(&good), "{len} printable bytes");
            for bad_at in 0..len {
                let mut buf = good.clone();
                buf[bad_at] = b'\n';
                assert!(!all_printable(&buf), "LF at {bad_at} of {len}");
                assert_eq!(printable_run_len(&buf), bad_at);
            }
        }
    }

    #[test]
    fn cursor_takes_in_bounds_and_tracks_position() {
        let mut c = Cursor::new(&[1, 2, 3, 4, 5]);
        assert_eq!(c.pos(), 0);
        assert_eq!(c.remaining(), 5);
        assert_eq!(c.take(2), Some(&[1, 2][..]));
        assert_eq!(c.pos(), 2);
        assert_eq!(c.remaining(), 3);
        assert_eq!(c.take(3), Some(&[3, 4, 5][..]));
        assert_eq!(c.remaining(), 0);
        // Zero-length reads are always fine and stay put.
        assert_eq!(c.take(0), Some(&[][..]));
        assert_eq!(c.pos(), 5);
    }

    #[test]
    fn cursor_rejects_overrun_without_advancing() {
        let mut c = Cursor::new(&[1, 2, 3]);
        assert_eq!(c.take(2), Some(&[1, 2][..]));
        // One byte left, asking for two leaves the cursor unmoved.
        assert_eq!(c.take(2), None);
        assert_eq!(c.pos(), 2);
        assert_eq!(c.remaining(), 1);
        // A length that would overflow the offset is rejected too.
        assert_eq!(c.take(usize::MAX), None);
        assert_eq!(c.pos(), 2);
    }
}
