//! A per-thread allocation counter wrapping the system allocator, so a measured
//! operation reports allocations beside its wall-clock time.
//!
//! Per thread rather than process-wide: the test harness runs scenarios on many
//! threads at once, and a global counter would fold a sibling's allocations into the
//! window between [`reset`] and [`snapshot`]. The single-threaded `--perf` and
//! `--perf-lab` runs read the same either way. The counter is const-initialized and
//! non-`Drop`, so touching it inside `alloc` is a thread-local read with no lazy setup
//! and no re-entry.
//!
//! Installed as the global allocator only under the `perf-alloc` feature, since a
//! global allocator taxes every allocation in the process. With the feature off the
//! counter still exists and nothing writes it, so [`snapshot`] reads zero and the
//! allocation columns print as unmeasured; [`ENABLED`] tells the two apart.
//!
//! FreeType and HarfBuzz allocate on the C side, invisible to Rust's global allocator,
//! so a scenario dominated by FFI rasterization shows few allocations.

use std::cell::Cell;

/// Whether the counting allocator is actually installed (the `perf-alloc`
/// feature is on). The harness uses this to tell a measured zero from an
/// unmeasured one.
#[cfg(feature = "perf-alloc")]
pub const ENABLED: bool = true;
#[cfg(not(feature = "perf-alloc"))]
pub const ENABLED: bool = false;

thread_local! {
    /// The calling thread's allocation count. Add-only within a scenario (reset
    /// zeroes it, alloc adds, dealloc is not tracked), so it never underflows and is
    /// exactly reproducible for a given code path and input, independent of the
    /// machine. Const-initialized so accessing it never allocates or lazily inits,
    /// which is what makes it safe to touch from inside `alloc`.
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

/// The counts read out after a scenario: the number of allocation calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub allocs: u64,
}

/// Zero the measuring thread's counter before a scenario.
pub fn reset() {
    ALLOCS.with(|c| c.set(0));
}

/// Read the measuring thread's counter after a scenario.
pub fn snapshot() -> Stats {
    Stats {
        allocs: ALLOCS.with(Cell::get),
    }
}

/// Add one to the current thread's counter, ignoring the narrow window during
/// thread teardown when the thread-local is already gone.
#[cfg(feature = "perf-alloc")]
fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

/// The allocator: delegate every call to the system allocator, counting on the
/// way through. Installed as the global allocator only under `perf-alloc`.
#[cfg(feature = "perf-alloc")]
pub struct Counting;

#[cfg(feature = "perf-alloc")]
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = std::alloc::System.alloc(layout);
        if !ptr.is_null() {
            bump();
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = std::alloc::System.alloc_zeroed(layout);
        if !ptr.is_null() {
            bump();
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let new_ptr = std::alloc::System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // A realloc that grows is a fresh allocation's worth of work; count it
            // so growing a Vec in a loop shows the churn it really is.
            bump();
        }
        new_ptr
    }
}

#[cfg(feature = "perf-alloc")]
#[global_allocator]
static GLOBAL: Counting = Counting;
