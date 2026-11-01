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

/// Append `n` as decimal ASCII, allocation-free: the replies, the key encoder and the
/// mouse encoder all build their numbers straight into an output buffer.
pub fn push_decimal(out: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    // A u32 is at most ten digits, written back to front.
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&tmp[i..]);
}

/// Append one nibble as a lowercase hex digit. Lowercase because that is what the X11
/// colour syntax and the OSC replies built on it use.
pub fn push_hex_nibble(out: &mut Vec<u8>, nibble: u8) {
    out.push(match nibble & 0xf {
        n @ 0..=9 => b'0' + n,
        n => b'a' + (n - 10),
    });
}

/// The bytes `text` percent-decodes to, strictly: a `%` not followed by two hex digits
/// yields `None`, and the caller decides what a bad escape means. Decodes as it goes,
/// so a check over a link allocates nothing.
pub fn percent_decoded(text: &str) -> impl Iterator<Item = Option<u8>> + '_ {
    let mut bytes = text.bytes();
    std::iter::from_fn(move || {
        let byte = bytes.next()?;
        if byte != b'%' {
            return Some(Some(byte));
        }
        let high = bytes.next().and_then(hex_value);
        let low = bytes.next().and_then(hex_value);
        Some(high.zip(low).map(|(high, low)| high << 4 | low))
    })
}

/// The value of one hex digit, either case.
fn hex_value(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

/// Decode base64, the encoding `OSC 52` and `OSC 7501` wrap text in. `None` for anything
/// malformed, which the caller drops.
///
/// Strict about what it accepts: the child writes this. Whitespace is skipped (mail-era
/// encoders wrap lines), padding is optional, and anything else is a refusal rather than
/// a guess: a decoder that improvises on bad input is how you smuggle bytes past it.
pub fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![0; input.len() / 4 * 3 + 3];
    let len = base64_decode_into(input, &mut out)?;
    out.truncate(len);
    Some(out)
}

/// [`base64_decode`] into `out`, answering how many bytes it wrote. `None` also when the
/// text decodes to more than `out` holds.
pub fn base64_decode_into(input: &[u8], out: &mut [u8]) -> Option<usize> {
    let mut len = 0;
    let mut put = |byte: u8| -> Option<()> {
        *out.get_mut(len)? = byte;
        len += 1;
        Some(())
    };
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
            put((triple >> 16) as u8)?;
            if padding < 2 {
                put((triple >> 8) as u8)?;
            }
            if padding < 1 {
                put(triple as u8)?;
            }
            n = 0;
        }
    }
    match n {
        // A clean end, padded or exactly aligned.
        0 => Some(len),
        // Unpadded input, which the spec allows: finish the partial group. `=` only
        // completes a full group, and the loop counts it into `n`, so a short group that
        // carried one is refused here.
        2 | 3 if padding == 0 => {
            let triple =
                (u32::from(quad[0]) << 18) | (u32::from(quad[1]) << 12) | (u32::from(quad[2]) << 6);
            put((triple >> 16) as u8)?;
            if n == 3 {
                put((triple >> 8) as u8)?;
            }
            Some(len)
        }
        // One leftover character encodes six bits of nothing, and a padded partial group
        // is a length base64 cannot produce.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::platform::bytes::*;

    #[test]
    fn base64_decodes_what_a_terminal_actually_receives() {
        assert_eq!(base64_decode(b"aGVsbG8="), Some(b"hello".to_vec()));
        assert_eq!(base64_decode(b"aGVsbG8h"), Some(b"hello!".to_vec()));
        assert_eq!(base64_decode(b"eA=="), Some(b"x".to_vec()));
        assert_eq!(base64_decode(b""), Some(Vec::new()));
        // Padding is optional, and line breaks (mail-era encoders wrap) are skipped.
        assert_eq!(base64_decode(b"aGVsbG8"), Some(b"hello".to_vec()));
        assert_eq!(base64_decode(b"aGVs\nbG8="), Some(b"hello".to_vec()));
        // Every byte of the alphabet, including the two that are not alphanumeric.
        assert_eq!(base64_decode(b"+/8="), Some(vec![0xfb, 0xff]));
        // And the refusals.
        assert_eq!(base64_decode(b"a"), None);
        assert_eq!(base64_decode(b"a==="), None);
        assert_eq!(base64_decode(b"aa=a"), None);
        assert_eq!(base64_decode(b"////_"), None);

        // Padding only completes a four-character group, so a short group cannot carry it.
        assert_eq!(base64_decode(b"AA="), None, "a padded two-character group");
        assert_eq!(base64_decode(b"A="), None, "a padded one-character group");
        assert_eq!(
            base64_decode(b"AAA=="),
            None,
            "padding past the group boundary"
        );
        // The well-formed neighbours of each, so the refusals above cannot be over-broad.
        assert_eq!(base64_decode(b"AA"), Some(vec![0x00]));
        assert_eq!(base64_decode(b"AAA"), Some(vec![0x00, 0x00]));
        assert_eq!(base64_decode(b"AAA="), Some(vec![0x00, 0x00]));
        assert_eq!(base64_decode(b"AA=="), Some(vec![0x00]));
    }

    #[test]
    fn base64_decodes_into_a_fixed_buffer_and_refuses_what_overflows_it() {
        let mut out = [0; 6];
        assert_eq!(base64_decode_into(b"aGVsbG8h", &mut out), Some(6));
        assert_eq!(&out, b"hello!");
        // One byte short, in a whole group and in an unpadded tail.
        assert_eq!(base64_decode_into(b"aGVsbG8h", &mut out[..5]), None);
        assert_eq!(base64_decode_into(b"aGVsbG8", &mut out[..4]), None);
        assert_eq!(base64_decode_into(b"aGVsbG8", &mut out[..5]), Some(5));
    }

    #[test]
    fn percent_decoding_is_strict() {
        let decode = |text| percent_decoded(text).collect::<Option<Vec<u8>>>();
        assert_eq!(decode("a%20b%2fc%2F"), Some(b"a b/c/".to_vec()));
        assert_eq!(decode("%E2%80%8B"), Some(vec![0xE2, 0x80, 0x8B]));
        for bad in ["%", "%2", "%zz", "%g0", "a%2"] {
            assert_eq!(decode(bad), None, "{bad}");
        }
    }

    #[test]
    fn decimal_and_hex_write_what_they_say() {
        let mut out = Vec::new();
        push_decimal(&mut out, 0);
        push_decimal(&mut out, 7);
        push_decimal(&mut out, 4_294_967_295); // the widest u32, ten digits
        assert_eq!(out, b"074294967295");

        out.clear();
        for nibble in [0u8, 9, 10, 15] {
            push_hex_nibble(&mut out, nibble);
        }
        assert_eq!(out, b"09af");
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
}
