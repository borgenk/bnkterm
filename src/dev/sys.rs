//! `/proc/self/smaps_rollup` for resident and swapped bytes. Linux-only.
//!
//! The rollup is a page-table walk. Sample it outside timed regions.

/// Where this process's pages are right now, in bytes, summed across every mapping
/// by `/proc/self/smaps_rollup`. Unlike a lifetime high-water mark this falls when
/// memory is released, which is the only way a reading can answer "did that help".
///
/// No `Pss` field: for the private anonymous memory a grid is made of, nothing else
/// maps those pages, so Pss equals Rss and would only restate it. Watch for `rss`
/// falling while `swap` rises, which is memory leaving the working set rather than
/// being counted differently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    /// Resident: backed by physical pages this instant.
    pub rss: u64,
    /// Pushed out to swap and still owned.
    pub swap: u64,
}

/// Read [`Memory`] for this process. All zeroes when `/proc` is unreadable, which
/// reads as no measurement rather than as a perfect result.
pub fn memory() -> Memory {
    match std::fs::read_to_string("/proc/self/smaps_rollup") {
        Ok(text) => parse_smaps_rollup(&text),
        Err(_) => Memory::default(),
    }
}

/// Pull the two fields out of a rollup. Every line past the header is
/// `Name:<whitespace><n> kB`, so the name is what precedes the first colon; the
/// header line's own colons sit inside its device number and match nothing.
fn parse_smaps_rollup(text: &str) -> Memory {
    let mut m = Memory::default();
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        // Exact names, so `Swap` never picks up `SwapPss` and `Rss` never picks up
        // `Pss_Shmem`.
        let field = match name {
            "Rss" => &mut m.rss,
            "Swap" => &mut m.swap,
            _ => continue,
        };
        *field = parse_kb(value);
    }
    m
}

/// `"    5892 kB"` to bytes. Zero for anything that does not parse, so a kernel
/// that renames or drops a field costs that one reading and never the run.
fn parse_kb(value: &str) -> u64 {
    value
        .split_whitespace()
        .next()
        .and_then(|n| n.parse::<u64>().ok())
        .map_or(0, |kb| kb.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use crate::dev::sys::*;

    /// A verbatim rollup from the development kernel (7.1.5), kept whole rather
    /// than trimmed to the two interesting lines: the near-miss names
    /// (`Pss_Shmem`, `SwapPss`) are the entire reason the match is exact, and a
    /// fixture without them could not catch a prefix match sneaking back in.
    const ROLLUP: &str = "\
562795990000-7ffe97093000 ---p 00000000 00:00 0                          [rollup]
Rss:                5892 kB
Pss:                 761 kB
Pss_Dirty:           108 kB
Pss_Anon:            108 kB
Pss_File:            653 kB
Pss_Shmem:             0 kB
Shared_Clean:       5776 kB
Shared_Dirty:          0 kB
Private_Clean:         8 kB
Private_Dirty:       108 kB
Referenced:         5892 kB
Anonymous:           108 kB
KSM:                   0 kB
LazyFree:              0 kB
AnonHugePages:         0 kB
ShmemPmdMapped:        0 kB
FilePmdMapped:         0 kB
Shared_Hugetlb:        0 kB
Private_Hugetlb:       0 kB
Swap:                 44 kB
SwapPss:              12 kB
Locked:                0 kB
";

    #[test]
    fn rollup_reports_the_two_fields_in_bytes() {
        let m = parse_smaps_rollup(ROLLUP);
        assert_eq!(m.rss, 5892 * 1024);
        // Not SwapPss's 12, which is the prefix an inexact match would take.
        assert_eq!(m.swap, 44 * 1024);
    }

    #[test]
    fn a_rollup_missing_or_malformed_reads_as_zero_not_as_garbage() {
        assert_eq!(parse_smaps_rollup(""), Memory::default());
        assert_eq!(parse_smaps_rollup("Rss: kB\nSwap: -\n"), Memory::default());
        // A header alone: its `00:00` splits on a colon and must match nothing.
        assert_eq!(
            parse_smaps_rollup("562795990000-7ffe97093000 ---p 00000000 00:00 0 [rollup]"),
            Memory::default()
        );
    }

    #[test]
    fn the_live_process_is_resident_somewhere() {
        // The one check that the path and the format are still real on this kernel:
        // a running process always has resident pages.
        let m = memory();
        assert!(m.rss > 0, "smaps_rollup reported no resident memory");
    }
}
