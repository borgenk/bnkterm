//! `getrusage(2)` for CPU time, `/proc/self/smaps_rollup` for resident, swapped and
//! referenced bytes. Linux-only; std already links libc, so the symbols resolve without
//! a `#[link]`.
//!
//! The rollup is a page-table walk. Sample it outside timed regions.

use core::ffi::{c_int, c_long};
use std::time::Duration;

/// `getrusage` target: this process (self), not its children.
const RUSAGE_SELF: c_int = 0;

#[repr(C)]
struct Timeval {
    tv_sec: c_long,
    tv_usec: c_long,
}

/// `struct rusage`, Linux x86_64/arm64 ABI: two timevals then sixteen longs. Only
/// the times are read; the rest is padding kept so the struct has the size the
/// kernel writes.
#[repr(C)]
#[allow(dead_code)]
struct Rusage {
    ru_utime: Timeval,
    ru_stime: Timeval,
    ru_maxrss: c_long,
    rest: [c_long; 15],
}

extern "C" {
    fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
}

fn rusage() -> Option<Rusage> {
    // SAFETY: `getrusage` fills a fully owned, correctly sized `Rusage`; a zeroed
    // start is valid for every field. A nonzero return means it wrote nothing.
    let mut u = std::mem::MaybeUninit::<Rusage>::zeroed();
    let rc = unsafe { getrusage(RUSAGE_SELF, u.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    Some(unsafe { u.assume_init() })
}

/// Total CPU time (user + system) this process has consumed so far. Take a delta
/// across a workload for its CPU cost; monotonic, so a delta is always >= 0.
pub fn cpu_time() -> Duration {
    match rusage() {
        Some(u) => dur(&u.ru_utime) + dur(&u.ru_stime),
        None => Duration::ZERO,
    }
}

fn dur(t: &Timeval) -> Duration {
    // tv_usec is microseconds in [0, 1e6); *1000 stays under a second in nanos.
    Duration::new(
        t.tv_sec.max(0) as u64,
        (t.tv_usec.max(0) as u32).wrapping_mul(1000),
    )
}

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
    /// Pages the kernel has seen touched since the reference bits were last
    /// cleared. The one field that reports an *advice* rather than an outcome:
    /// `MADV_COLD` clears the referenced bits of the range it covers, so advice is
    /// measurable here before any memory pressure exists to act on it.
    pub referenced: u64,
}

/// Read [`Memory`] for this process. All zeroes when `/proc` is unreadable, which
/// reads as no measurement rather than as a perfect result.
pub fn memory() -> Memory {
    match std::fs::read_to_string("/proc/self/smaps_rollup") {
        Ok(text) => parse_smaps_rollup(&text),
        Err(_) => Memory::default(),
    }
}

/// Pull the three fields out of a rollup. Every line past the header is
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
            "Referenced" => &mut m.referenced,
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
    use super::*;

    /// A verbatim rollup from the development kernel (7.1.5), kept whole rather
    /// than trimmed to the three interesting lines: the near-miss names
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
    fn rollup_reports_the_three_fields_in_bytes() {
        let m = parse_smaps_rollup(ROLLUP);
        assert_eq!(m.rss, 5892 * 1024);
        assert_eq!(m.referenced, 5892 * 1024);
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
        // The one check that the path and the format are still real on this kernel;
        // a running process always has resident pages, and Referenced is a superset
        // of nothing but always reported beside it.
        let m = memory();
        assert!(m.rss > 0, "smaps_rollup reported no resident memory");
    }
}
