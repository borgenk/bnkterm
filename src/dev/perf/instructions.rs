//! The user-space instructions this thread retires, from the kernel's performance
//! counters. Unlike a time, the count does not move with machine load or code placement,
//! so a change in it is a change in the work done.
//!
//! On a hybrid CPU each core type has its own counter that sees only its own cores, so
//! one is opened per type and their counts are summed.

use crate::platform::ffi::instruction_counter;
use std::fs::File;
use std::io::Read;

/// A hybrid CPU's core types, as `/sys/bus/event_source/devices` names their PMUs.
const CORE_TYPES: [&str; 2] = ["cpu_core", "cpu_atom"];

/// Open counters for the calling thread.
pub(super) struct Counter {
    files: [Option<File>; 2],
}

impl Counter {
    /// `None` where the kernel refuses performance events.
    pub(super) fn open() -> Option<Counter> {
        let open = |pmu| instruction_counter(pmu).ok().map(File::from);
        let files = match CORE_TYPES.map(pmu_type) {
            [None, None] => [Some(open(None)?), None],
            types => {
                let mut files = [None, None];
                for (file, pmu) in files.iter_mut().zip(types) {
                    if let Some(pmu) = pmu {
                        *file = Some(open(Some(pmu))?);
                    }
                }
                files
            }
        };
        Some(Counter { files })
    }

    /// The instructions retired so far, or `None` if a read fails.
    pub(super) fn read(&mut self) -> Option<u64> {
        let mut total = 0u64;
        for file in self.files.iter_mut().flatten() {
            let mut bytes = [0; 8];
            file.read_exact(&mut bytes).ok()?;
            total = total.saturating_add(u64::from_ne_bytes(bytes));
        }
        Some(total)
    }
}

fn pmu_type(name: &str) -> Option<u32> {
    let path = format!("/sys/bus/event_source/devices/{name}/type");
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use crate::dev::perf::instructions::*;
    use std::hint::black_box;

    #[test]
    fn counts_the_instructions_this_thread_retires() {
        // Where the kernel refuses performance events there is nothing to count.
        let Some(mut counter) = Counter::open() else {
            return;
        };
        let mut count = |n: u64| {
            let start = counter.read().unwrap();
            let mut sum = 0u64;
            for i in 0..n {
                sum = black_box(sum.wrapping_add(i));
            }
            black_box(sum);
            counter.read().unwrap() - start
        };
        let small = count(1_000);
        let large = count(1_000_000);
        assert!(large > 1_000_000, "at least one per iteration: {large}");
        assert!(
            large > small * 50,
            "{small} for 1000, {large} for a million"
        );
    }
}
