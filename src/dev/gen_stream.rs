//! `--gen-stream`: build the ASCII stream `make cat-bench` runs over.
//!
//! Drops the source text's licence header, then duplicates the remaining body to
//! exactly the target size. The Makefile owns the fetch and the checksum. A pure
//! function of its input, so the stream is byte-identical on every machine and the
//! throughput derived from it stays comparable.
//!
//! ```text
//!   source.txt ── strip the first "*** START OF " header line ──▶ body
//!   body       ── duplicate to exactly `size` bytes (last copy truncated) ──▶ stream
//! ```

use crate::error::{Error, Result};

/// The default stream size, in decimal bytes (see the Makefile's note on why the
/// MB/s figure is computed against decimal, not binary, megabytes).
const DEFAULT_SIZE: usize = 150_000_000;

/// The source text opens with a licence header ending at this marker. Only the first
/// occurrence is a header: the file concatenates several works, each carrying its own
/// marker, and the later ones are stream content.
const START_MARKER: &[u8] = b"*** START OF ";

/// `bnkterm --gen-stream --input <source> --output <stream> [--size N]`.
pub fn run(args: &[String]) -> Result<()> {
    let input =
        flag(args, "--input").ok_or_else(|| Error::msg("--gen-stream needs --input FILE"))?;
    let output =
        flag(args, "--output").ok_or_else(|| Error::msg("--gen-stream needs --output FILE"))?;
    let size = match flag(args, "--size") {
        Some(s) => s
            .parse::<usize>()
            .map_err(|_| Error::msg(format!("--size: not a byte count: {s}")))?,
        None => DEFAULT_SIZE,
    };
    if size == 0 {
        return Err(Error::msg("--size must be positive"));
    }

    let data = std::fs::read(input).map_err(|e| Error::msg(format!("read {input}: {e}")))?;
    let body = strip_header(&data);
    let stream = build(body, size)?;
    std::fs::write(output, &stream).map_err(|e| Error::msg(format!("write {output}: {e}")))?;
    println!(
        "wrote {output}: {} bytes ({} header bytes stripped, {}-byte body duplicated)",
        stream.len(),
        data.len() - body.len(),
        body.len()
    );
    Ok(())
}

/// The value following `name` in `args`, if present.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(String::as_str)
}

/// `data` with the leading licence header removed: everything up to and including
/// the newline that ends the first [`START_MARKER`] line. An already-stripped file
/// (no marker, or a marker with no following newline) is returned unchanged.
fn strip_header(data: &[u8]) -> &[u8] {
    let Some(marker) = find(data, START_MARKER) else {
        return data;
    };
    match data[marker..].iter().position(|&b| b == b'\n') {
        Some(newline) => &data[marker + newline + 1..],
        None => data, // the marker sits on the last line; no body follows
    }
}

/// `body` duplicated to exactly `size` bytes (the last copy truncated). Errors on an
/// empty body, which has nothing to duplicate.
fn build(body: &[u8], size: usize) -> Result<Vec<u8>> {
    if body.is_empty() {
        return Err(Error::msg("body is empty after stripping the header"));
    }
    let reps = size / body.len();
    let tail = size % body.len();
    let mut out = Vec::with_capacity(size);
    for _ in 0..reps {
        out.extend_from_slice(body);
    }
    out.extend_from_slice(&body[..tail]);
    Ok(out)
}

/// The first index of `needle` in `haystack`, or `None`. A one-shot scan over a
/// downloaded file, so a plain window search is more than fast enough.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_the_first_header_line() {
        // A leading header line, then a body carrying an inner marker (a concatenated
        // work) which must survive: only the first is a header.
        let data = b"licence preamble *** START OF THE BOOK ***\nreal *** START OF two ***\nmore\n";
        assert_eq!(strip_header(data), b"real *** START OF two ***\nmore\n");
    }

    #[test]
    fn an_already_stripped_file_is_unchanged() {
        let data = b"no marker here\njust body\n";
        assert_eq!(strip_header(data), data);
    }

    #[test]
    fn a_marker_on_the_last_line_leaves_no_body() {
        // Marker present but no newline after it: nothing follows, so return it whole.
        let data = b"prefix *** START OF END";
        assert_eq!(strip_header(data), data);
    }

    #[test]
    fn build_duplicates_to_exactly_the_target_size() {
        let body = b"abcde"; // 5 bytes
        assert_eq!(build(body, 12).unwrap(), b"abcdeabcdeab"); // 2 whole + "ab"
        assert_eq!(build(body, 5).unwrap(), b"abcde"); // exact fit
        assert_eq!(build(body, 3).unwrap(), b"abc"); // less than one body
    }

    #[test]
    fn build_rejects_an_empty_body() {
        assert!(build(b"", 10).is_err());
    }

    #[test]
    fn strip_then_build_end_to_end() {
        let data = b"hdr *** START OF X ***\nBODY\n"; // body is "BODY\n", 5 bytes
        let body = strip_header(data);
        assert_eq!(body, b"BODY\n");
        assert_eq!(build(body, 12).unwrap(), b"BODY\nBODY\nBO");
    }
}
