//! Branchless byte-scan primitives, written so the compiler autovectorises them
//! (SSE2/AVX2 on x86-64, NEON on aarch64) with no intrinsics and no dependency.
//! They back the rope's newline counting and line finding, and newline
//! normalisation on load, the hottest whole-buffer scans in a large-file open.

/// Count the occurrences of `needle` in `hay`.
///
/// Summing `(b == needle) as usize` is a branchless reduction the compiler turns
/// into vector compares plus a horizontal add, rather than the data-dependent
/// conditional increment a `filter().count()` compiles to. On a multi-megabyte
/// buffer that is several times faster.
pub fn count_byte(hay: &[u8], needle: u8) -> usize {
    hay.iter().map(|&b| usize::from(b == needle)).sum()
}

/// The byte index of the first `needle` in `hay`, or `None`. A byte-equality
/// `position` vectorises to a memchr-style scan.
pub fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
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
    fn count_byte_matches_a_naive_count() {
        let cases: [&[u8]; 5] = [b"", b"aaa", b"a\nb\nc\n", b"no newlines", &[b'\n'; 1000]];
        for hay in cases {
            let want = hay.iter().filter(|&&b| b == b'\n').count();
            assert_eq!(count_byte(hay, b'\n'), want, "count in {hay:?}");
        }
        // A buffer long enough to span many vector widths, newlines scattered.
        let big: Vec<u8> = (0..10_000u32)
            .map(|i| if i % 7 == 0 { b'\n' } else { b'x' })
            .collect();
        assert_eq!(
            count_byte(&big, b'\n'),
            big.iter().filter(|&&b| b == b'\n').count()
        );
    }

    #[test]
    fn find_byte_matches_position() {
        assert_eq!(find_byte(b"abc\ndef", b'\n'), Some(3));
        assert_eq!(find_byte(b"\nfirst", b'\n'), Some(0));
        assert_eq!(find_byte(b"no newline", b'\n'), None);
        assert_eq!(find_byte(b"", b'\n'), None);
        // Match past a vector-width boundary is still found at the right index.
        let mut buf = vec![b'x'; 5000];
        buf[4097] = b'\n';
        assert_eq!(find_byte(&buf, b'\n'), Some(4097));
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
