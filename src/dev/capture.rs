//! `--capture <name> <cols> <rows> -- <cmd> [args...]`: run a program under a fixed
//! -size PTY and save every byte it writes as a fixture (`tests/fixtures/<name>.in`).
//! A `<name>.chunks` sidecar records the sizes of the PTY reads that produced it, so a
//! replay can reproduce either the real delivery fragmentation or a normalized 64 KiB
//! stream. Both files' SHA-256 values go in `tests/fixtures/CHECKSUMS`.
//!
//! A real program's output has no oracle: no one computed the grid it should produce.
//! A capture is registered in `tests/captures.rs` (add its name and size to `CAPTURES`)
//! and asserted only on what such an input supports — the pipeline survives it, the
//! grid's invariants hold, and the result does not depend on how the bytes were
//! chunked. Never as a golden; the hand-authored fixtures in `tests/golden.rs` are the
//! golden path.
//!
//! Capturing runs through [`Pty::spawn_command`], the same fork/exec/drain the live
//! terminal uses, so the `unsafe` stays in `pty.rs` and a fixture sees the PTY setup a
//! real session does. The captured `.in` and its checksum are committed.
//!
//! Interactive programs need keystrokes to render anything; drive them with `-c`/`-e`
//! flags, or capture a non-interactive render instead.

use crate::dev::sha256;
use crate::error::{Error, Result};
use crate::pty::{PollSet, Pty, ReadOutcome};
use std::path::Path;

/// Where fixtures live, relative to the repo root (the CWD `make`/`cargo` run in).
const FIXTURES: &str = "tests/fixtures";

/// `bnkterm --capture <name> <cols> <rows> -- <command> [args...]`.
pub fn run(args: &[String]) -> Result<()> {
    let i = args
        .iter()
        .position(|a| a == "--capture")
        .ok_or_else(|| Error::msg("no --capture flag"))?;
    let name = args.get(i + 1).ok_or_else(usage)?;
    let cols: usize = args
        .get(i + 2)
        .and_then(|s| s.parse().ok())
        .ok_or_else(usage)?;
    let rows: usize = args
        .get(i + 3)
        .and_then(|s| s.parse().ok())
        .ok_or_else(usage)?;
    if args.get(i + 4).map(String::as_str) != Some("--") {
        return Err(usage());
    }
    let command: Vec<&str> = args[i + 5..].iter().map(String::as_str).collect();
    if command.is_empty() {
        return Err(Error::msg("--capture: no command given after --"));
    }

    let capture = capture(cols, rows, &command)?;

    let fixtures = Path::new(FIXTURES);
    std::fs::create_dir_all(fixtures).map_err(|e| Error::msg(format!("create {FIXTURES}: {e}")))?;
    let in_path = fixtures.join(format!("{name}.in"));
    std::fs::write(&in_path, &capture.bytes)
        .map_err(|e| Error::msg(format!("write {}: {e}", in_path.display())))?;
    let chunks_path = fixtures.join(format!("{name}.chunks"));
    let chunks = encode_chunks(&capture.chunks);
    std::fs::write(&chunks_path, &chunks)
        .map_err(|e| Error::msg(format!("write {}: {e}", chunks_path.display())))?;
    record_checksum(fixtures, &format!("{name}.in"), &capture.bytes)?;
    record_checksum(fixtures, &format!("{name}.chunks"), &chunks)?;

    eprintln!(
        "capture: wrote {} bytes in {} chunks to {} ({cols}x{rows}). Register it by adding \
         (\"{name}\", {cols}, {rows}) to CAPTURES in tests/captures.rs, then run \
         `cargo test --test captures`. A real program's output has no oracle, so it is \
         tested oracle-less (pipeline survives, grid invariants hold, chunking is \
         irrelevant), never committed as a golden.",
        capture.bytes.len(),
        capture.chunks.len(),
        in_path.display()
    );
    Ok(())
}

fn usage() -> Error {
    Error::msg("usage: --capture <name> <cols> <rows> -- <command> [args...]")
}

struct Capture {
    bytes: Vec<u8>,
    chunks: Vec<usize>,
}

/// Run `command` under a `cols` x `rows` PTY and return every byte it writes plus
/// the read boundaries that delivered it, draining the master until the child
/// exits (the kernel reports EOF/EIO once the slave is gone). Blocks until then,
/// exactly as the Python capture did: an interactive program that never exits
/// will hang, so drive it non-interactively.
fn capture(cols: usize, rows: usize, command: &[&str]) -> Result<Capture> {
    // A program lays out for the grid it is told about: the size is on the PTS via
    // the winsize ioctl (set by `spawn_command`), and `COLUMNS`/`LINES` are the env
    // fallback some programs read instead. Set them (and a known `TERM`) before the
    // fork so the child inherits them. Safe to mutate the environment here: the
    // capture process is single-threaded, unlike the live terminal with its gatherers.
    std::env::set_var("TERM", "xterm-256color");
    std::env::set_var("COLUMNS", cols.to_string());
    std::env::set_var("LINES", rows.to_string());

    let pty = Pty::spawn_command(cols, rows, command)?;
    let mut poll = PollSet::new();
    let mut out = Vec::new();
    let mut chunks = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        poll.clear();
        poll.add(pty.fd());
        poll.wait(None)?; // block until the child writes or the slave hangs up
        match pty.read(&mut buf)? {
            ReadOutcome::Data(n) => {
                let Some(read) = buf.get(..n) else {
                    return Err(Error::msg("PTY read exceeded its destination buffer"));
                };
                out.extend_from_slice(read);
                chunks.push(n);
            }
            ReadOutcome::WouldBlock => {}
            ReadOutcome::Eof => break,
        }
    }
    // `pty` drops here: it closes the master and reaps the (already-exited) child.
    Ok(Capture { bytes: out, chunks })
}

/// Newline-separated decimal chunk lengths. Text keeps the sidecar inspectable,
/// diffable, and independent of host integer layout.
fn encode_chunks(chunks: &[usize]) -> Vec<u8> {
    let mut text = String::new();
    for chunk in chunks {
        use std::fmt::Write as _;
        let _ = writeln!(text, "{chunk}");
    }
    text.into_bytes()
}

/// Insert or replace the `sha256  <filename>` line in `CHECKSUMS`, kept sorted
/// by filename, matching the format the fixtures were originally recorded in.
/// Nothing in the test suite reads this file; it is a committed integrity record.
fn record_checksum(fixtures: &Path, filename: &str, data: &[u8]) -> Result<()> {
    let path = fixtures.join("CHECKSUMS");
    let mut lines: Vec<String> = match std::fs::read_to_string(&path) {
        Ok(text) => text
            .lines()
            .filter(|line| filename_of(line) != filename)
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    };
    lines.push(format!("{}  {filename}", sha256::hex(data)));
    lines.sort_by(|a, b| filename_of(a).cmp(filename_of(b)));
    let mut text = lines.join("\n");
    text.push('\n');
    std::fs::write(&path, text).map_err(|e| Error::msg(format!("write {}: {e}", path.display())))
}

/// The filename a `CHECKSUMS` line records, i.e. the text after the two-space
/// separator (`<hex>  <name>.in`). A malformed line sorts by its whole self.
fn filename_of(line: &str) -> &str {
    line.split_once("  ").map(|(_, name)| name).unwrap_or(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_is_the_text_past_the_two_space_separator() {
        assert_eq!(filename_of("abc123  less_readme.in"), "less_readme.in");
        assert_eq!(filename_of("no separator"), "no separator");
    }

    #[test]
    fn chunk_sidecar_is_one_decimal_length_per_line() {
        assert_eq!(encode_chunks(&[7, 65_536, 1]), b"7\n65536\n1\n");
    }
}
