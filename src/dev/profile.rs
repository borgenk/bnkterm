//! What shapes a byte stream actually contains, so a throughput number can be
//! explained rather than only observed: prose run lengths, the short labels between
//! SGR changes, wide CJK, zero-width marks, malformed input, and the chunk boundaries
//! the stream arrived on.
//!
//! Runs the real VT parser into an observational [`Perform`] sink, outside any timed
//! loop:
//!
//! ```text
//! bytes + chunk lengths -> vt::Parser -> semantic runs/actions -> StreamProfile
//! ```

use crate::error::{Error, Result};
use crate::vt::{Params, Parser, Perform};
use crate::width::width;
use std::fmt;
use std::path::{Path, PathBuf};

const NORMALIZED_CHUNK: usize = 64 * 1024;

struct Recording {
    name: &'static str,
    cols: usize,
    rows: usize,
}

const RECORDINGS: &[Recording] = &[
    Recording {
        name: "tui_vim",
        cols: 80,
        rows: 24,
    },
    Recording {
        name: "tui_ls",
        cols: 80,
        rows: 24,
    },
    Recording {
        name: "unicode_source",
        cols: 80,
        rows: 24,
    },
];

/// Which parser-call boundaries a recorded corpus should use.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// Replay the PTY read lengths captured beside the bytes. Old recordings
    /// without a sidecar fall back to normalized delivery and say so.
    Recorded,
    /// Feed the same content in the live path's nominal 64 KiB chunks.
    Normalized,
}

/// One real-program recording expanded to a useful lab size.
pub struct LabRecording {
    pub bytes: Vec<u8>,
    pub chunks: Vec<usize>,
    pub cols: usize,
    pub rows: usize,
    pub delivery: &'static str,
}

/// Names accepted by the performance lab in addition to generated streams.
pub fn recording_names() -> impl Iterator<Item = &'static str> {
    RECORDINGS.iter().map(|recording| recording.name)
}

/// Load and repeat a vendored real-program recording to at least `target`
/// bytes. `Ok(None)` means `name` is not a recording.
pub fn load_recording(
    name: &str,
    target: usize,
    requested: Delivery,
) -> Result<Option<LabRecording>> {
    let Some(recording) = RECORDINGS.iter().find(|recording| recording.name == name) else {
        return Ok(None);
    };
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let input_path = fixtures.join(format!("{name}.in"));
    let fixture = std::fs::read(&input_path)
        .map_err(|e| Error::msg(format!("read {}: {e}", input_path.display())))?;
    if fixture.is_empty() {
        return Err(Error::msg(format!("{} is empty", input_path.display())));
    }

    let sidecar = if requested == Delivery::Recorded {
        read_chunks(&fixtures.join(format!("{name}.chunks")), fixture.len())?
    } else {
        None
    };
    let mut bytes = Vec::with_capacity(target.saturating_add(fixture.len()));
    while bytes.len() < target {
        bytes.extend_from_slice(&fixture);
    }
    let (chunks, delivery) = match sidecar {
        Some(source) => (repeat_chunks(&source, bytes.len()), "recorded"),
        None => (
            fixed_chunk_lengths(bytes.len(), NORMALIZED_CHUNK),
            "normalized-64k",
        ),
    };
    Ok(Some(LabRecording {
        bytes,
        chunks,
        cols: recording.cols,
        rows: recording.rows,
        delivery,
    }))
}

/// Fixed parser-call lengths spanning `bytes`.
pub fn fixed_chunk_lengths(bytes: usize, chunk: usize) -> Vec<usize> {
    if chunk == 0 {
        return vec![bytes];
    }
    let full = bytes / chunk;
    let tail = bytes % chunk;
    let mut lengths = vec![chunk; full];
    if tail > 0 {
        lengths.push(tail);
    }
    lengths
}

fn read_chunks(path: &Path, expected_bytes: usize) -> Result<Option<Vec<usize>>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::msg(format!("read {}: {e}", path.display()))),
    };
    let mut chunks = Vec::new();
    for line in text.lines() {
        let length = line
            .parse::<usize>()
            .ok()
            .filter(|length| *length > 0)
            .ok_or_else(|| Error::msg(format!("malformed chunk length {line:?}")))?;
        chunks.push(length);
    }
    let total = chunks
        .iter()
        .fold(0usize, |sum, length| sum.saturating_add(*length));
    if total != expected_bytes {
        return Err(Error::msg(format!(
            "{} describes {total} bytes, recording has {expected_bytes}",
            path.display()
        )));
    }
    Ok(Some(chunks))
}

fn repeat_chunks(source: &[usize], target: usize) -> Vec<usize> {
    if source.is_empty() {
        return fixed_chunk_lengths(target, NORMALIZED_CHUNK);
    }
    let mut chunks = Vec::new();
    let mut total = 0usize;
    while total < target {
        for &length in source {
            let remaining = target.saturating_sub(total);
            if remaining == 0 {
                break;
            }
            let take = length.min(remaining);
            chunks.push(take);
            total = total.saturating_add(take);
        }
    }
    chunks
}

/// Semantic shape of one byte corpus, measured through the real VT parser.
pub struct StreamProfile {
    bytes: usize,
    scalars: usize,
    ascii_scalars: usize,
    multibyte_scalars: usize,
    replacement_scalars: usize,
    widths: [usize; 3],
    execute_actions: usize,
    sequence_actions: usize,
    run_scalars: Vec<usize>,
    run_encoded_bytes: Vec<usize>,
    chunks: Vec<usize>,
}

impl StreamProfile {
    /// Parse `bytes` at the supplied delivery boundaries and characterize the
    /// semantic text emitted. Empty or incomplete `chunks` are completed with
    /// one final slice, so an analyser request can never silently omit input.
    pub fn analyze(bytes: &[u8], chunks: &[usize]) -> Self {
        let mut parser = Parser::new();
        let mut observer = Observer::default();
        let mut offset = 0usize;
        let mut delivered = Vec::new();

        for &length in chunks {
            if offset >= bytes.len() {
                break;
            }
            let end = offset.saturating_add(length).min(bytes.len());
            if let Some(chunk) = bytes.get(offset..end) {
                if !chunk.is_empty() {
                    parser.advance_bytes(&mut observer, chunk);
                    delivered.push(chunk.len());
                }
            }
            offset = end;
        }
        if let Some(tail) = bytes.get(offset..) {
            if !tail.is_empty() {
                parser.advance_bytes(&mut observer, tail);
                delivered.push(tail.len());
            }
        }
        parser.finish(&mut observer);
        observer.finish_run();

        StreamProfile {
            bytes: bytes.len(),
            scalars: observer.scalars,
            ascii_scalars: observer.ascii_scalars,
            multibyte_scalars: observer.multibyte_scalars,
            replacement_scalars: observer.replacement_scalars,
            widths: observer.widths,
            execute_actions: observer.execute_actions,
            sequence_actions: observer.sequence_actions,
            run_scalars: observer.run_scalars,
            run_encoded_bytes: observer.run_encoded_bytes,
            chunks: delivered,
        }
    }

    /// Number of semantic printable runs separated by controls or escape actions.
    pub fn runs(&self) -> usize {
        self.run_scalars.len()
    }
}

impl fmt::Display for StreamProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scalar_runs = Distribution::new(&self.run_scalars);
        let byte_runs = Distribution::new(&self.run_encoded_bytes);
        let chunks = Distribution::new(&self.chunks);
        let width0 = self.widths.first().copied().unwrap_or(0);
        let width1 = self.widths.get(1).copied().unwrap_or(0);
        let width2 = self.widths.get(2).copied().unwrap_or(0);
        write!(
            f,
            "{} bytes, {} scalars (ASCII {}%, multibyte {}%, replacement {})\n\
             widths 0/1/2: {}/{}/{} ({}%/{}%/{}%)\n\
             actions execute/sequence: {}/{}; text runs: {}\n\
             run scalars {}; encoded bytes {}; chunks {}",
            self.bytes,
            self.scalars,
            percent(self.ascii_scalars, self.scalars),
            percent(self.multibyte_scalars, self.scalars),
            self.replacement_scalars,
            width0,
            width1,
            width2,
            percent(width0, self.scalars),
            percent(width1, self.scalars),
            percent(width2, self.scalars),
            self.execute_actions,
            self.sequence_actions,
            self.runs(),
            scalar_runs,
            byte_runs,
            chunks,
        )
    }
}

#[derive(Default)]
struct Observer {
    scalars: usize,
    ascii_scalars: usize,
    multibyte_scalars: usize,
    replacement_scalars: usize,
    widths: [usize; 3],
    execute_actions: usize,
    sequence_actions: usize,
    current_run_scalars: usize,
    current_run_encoded_bytes: usize,
    run_scalars: Vec<usize>,
    run_encoded_bytes: Vec<usize>,
}

impl Observer {
    fn observe_char(&mut self, c: char) {
        self.scalars = self.scalars.saturating_add(1);
        if c.is_ascii() {
            self.ascii_scalars = self.ascii_scalars.saturating_add(1);
        } else {
            self.multibyte_scalars = self.multibyte_scalars.saturating_add(1);
        }
        if c == '\u{fffd}' {
            self.replacement_scalars = self.replacement_scalars.saturating_add(1);
        }
        if let Some(count) = self.widths.get_mut(usize::from(width(c))) {
            *count = count.saturating_add(1);
        }
        self.current_run_scalars = self.current_run_scalars.saturating_add(1);
        self.current_run_encoded_bytes =
            self.current_run_encoded_bytes.saturating_add(c.len_utf8());
    }

    fn finish_run(&mut self) {
        if self.current_run_scalars == 0 {
            return;
        }
        self.run_scalars.push(self.current_run_scalars);
        self.run_encoded_bytes.push(self.current_run_encoded_bytes);
        self.current_run_scalars = 0;
        self.current_run_encoded_bytes = 0;
    }

    fn observe_sequence(&mut self) {
        self.finish_run();
        self.sequence_actions = self.sequence_actions.saturating_add(1);
    }
}

impl Perform for Observer {
    fn print(&mut self, c: char) {
        self.observe_char(c);
    }

    fn print_ascii(&mut self, bytes: &[u8]) {
        self.scalars = self.scalars.saturating_add(bytes.len());
        self.ascii_scalars = self.ascii_scalars.saturating_add(bytes.len());
        if let Some(width1) = self.widths.get_mut(1) {
            *width1 = width1.saturating_add(bytes.len());
        }
        self.current_run_scalars = self.current_run_scalars.saturating_add(bytes.len());
        self.current_run_encoded_bytes = self.current_run_encoded_bytes.saturating_add(bytes.len());
    }

    fn execute(&mut self, _byte: u8) {
        self.finish_run();
        self.execute_actions = self.execute_actions.saturating_add(1);
    }

    fn csi_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _private: u8, _action: u8) {
        self.observe_sequence();
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _byte: u8) {
        self.observe_sequence();
    }

    fn dcs_dispatch(&mut self, _params: &Params, _intermediates: &[u8], _action: u8, _data: &[u8]) {
        self.observe_sequence();
    }

    fn osc_dispatch(&mut self, _data: &[u8], _bel_terminated: bool) {
        self.observe_sequence();
    }
}

struct Distribution {
    values: Vec<usize>,
}

impl Distribution {
    fn new(values: &[usize]) -> Self {
        let mut values = values.to_vec();
        values.sort_unstable();
        Distribution { values }
    }

    fn percentile(&self, percentile: usize) -> usize {
        let Some(last) = self.values.len().checked_sub(1) else {
            return 0;
        };
        let index = last.saturating_mul(percentile).saturating_add(50) / 100;
        self.values.get(index).copied().unwrap_or(0)
    }
}

impl fmt::Display for Distribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p50={} p90={} p99={} max={}",
            self.percentile(50),
            self.percentile(90),
            self.percentile(99),
            self.values.last().copied().unwrap_or(0)
        )
    }
}

struct Percent(usize);

impl fmt::Display for Percent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.0 / 10, self.0 % 10)
    }
}

fn percent(part: usize, whole: usize) -> Percent {
    let tenths = part.saturating_mul(1_000).checked_div(whole).unwrap_or(0);
    Percent(tenths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_uses_semantic_runs_not_escape_payload_bytes() {
        let bytes = "abc\x1b[31m日本\x1b[0m!\n".as_bytes();
        let profile = StreamProfile::analyze(bytes, &[4, 2, 1, 99]);
        assert_eq!(profile.scalars, 6);
        assert_eq!(profile.ascii_scalars, 4);
        assert_eq!(profile.multibyte_scalars, 2);
        assert_eq!(profile.widths, [0, 4, 2]);
        assert_eq!(profile.sequence_actions, 2);
        assert_eq!(profile.execute_actions, 1);
        assert_eq!(profile.run_scalars, [3, 2, 1]);
    }

    #[test]
    fn incomplete_chunk_list_never_omits_the_tail() {
        let profile = StreamProfile::analyze("a界b".as_bytes(), &[1]);
        assert_eq!(profile.scalars, 3);
        assert_eq!(profile.widths, [0, 2, 1]);
        assert_eq!(profile.chunks, [1, 4]);
    }

    #[test]
    fn malformed_input_counts_replacements() {
        let profile = StreamProfile::analyze(&[b'a', 0xc0, 0xaf, b'b'], &[2, 2]);
        assert_eq!(profile.scalars, 4);
        assert_eq!(profile.replacement_scalars, 2);
    }

    #[test]
    fn recorded_corpus_repeats_to_target_at_its_original_dimensions() {
        let recording = load_recording("tui_vim", 4_096, Delivery::Normalized)
            .unwrap()
            .unwrap();
        assert!(recording.bytes.len() >= 4_096);
        assert_eq!(recording.cols, 80);
        assert_eq!(recording.rows, 24);
        assert_eq!(
            recording.chunks.iter().sum::<usize>(),
            recording.bytes.len()
        );
    }

    #[test]
    fn repeated_chunk_boundaries_cover_the_target_exactly() {
        assert_eq!(repeat_chunks(&[3, 2], 12), [3, 2, 3, 2, 2]);
    }
}
