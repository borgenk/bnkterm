//! The PTY gather thread: a dedicated reader that drains the cooked PTY master
//! into a bounded pool of fixed buffers, so the main thread parses previously
//! filled batches instead of racing the kernel one 200-byte read at a time.
//!
//! ```text
//!                              fixed, bounded Box<[u8]> pool
//!   child ─▶ PTY master ─▶ gather thread ─▶ [64K][64K]… ─▶ main thread
//!             cooked          read/poll        FIFO           Gatherer::next_batch
//!             output          only             ready queue    → vt::Parser → grid
//!                                 │                                 ▲
//!                                 └─ ready eventfd wake ────────────┘
//!                            free queue ◀── returned buffers ────────┘
//! ```
//!
//! # Why this exists
//!
//! bnkterm runs one thread: it reads the PTY, parses, and submits the GPU frame in
//! a loop. During a paint (~2–8 ms) the loop is not calling `read`, the tty's small
//! output buffer fills, the child blocks on `write`, and drain throughput is lost.
//! The Stage 0 probe (`docs/probes/gather_probe.rs`, recorded in
//! measured that a gather thread recovers that loss:
//! it holds the ~145 MB/s drain ceiling through a simulated 8 ms render stall,
//! where a single-threaded reader collapses to ~72 MB/s. This module is the
//! production form of that mechanism, *baseline only* (publish every nonempty
//! buffer on the first `EAGAIN`); the adaptive bridge is a later, switchable step.
//!
//! # Ownership and the stage boundary
//!
//! The gather thread owns *only* PTY reads and buffer assembly. It receives a
//! `F_DUPFD_CLOEXEC` duplicate of the master fd (sharing the open file description,
//! so its `O_NONBLOCK` flag too); the main thread keeps the original [`crate::pty::Pty`]
//! for writes, `TIOCSWINSZ`, and child reaping. No parser, grid, or terminal-mode
//! state crosses the boundary: the gather thread produces `bytes`, the main thread
//! feeds them to `vt::Parser`, preserving `bytes → vt::Parser → grid::Screen`.
//!
//! # The buffer pool state machine
//!
//! ```text
//!   Free ──take_free──▶ Filling ──publish──▶ Ready(len) ──next_batch──▶ Parsing ──drop──▶ Free
//!    ▲ (gather thread)           (gather)                (main thread)            (Batch)  │
//!    └──────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each buffer is an owned `Box<[u8]>` moved between two queues under one mutex;
//! no writable slice is ever shared. Both queues reserve their full capacity at
//! construction, so a buffer changing hands in steady state never allocates.
//! Exhaustion of the free queue is intentional backpressure: the gather thread
//! blocks on a condition variable, the kernel PTY queue fills, and the child
//! blocks on `write` exactly as it does with a single-threaded reader.

use std::collections::VecDeque;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use core::ffi::{c_int, c_short, c_uint, c_ulong, c_void};

use crate::error::{Error, Result};

/// Capacity of each pool buffer, matching bnkterm's live PTY read chunk so the
/// parser sees the same batch boundaries a real shell produces.
pub const BUF_CAP: usize = 64 * 1024;

/// Default pool depth: 64 × 64 KiB = 4 MiB of runway. The Stage 0 probe showed
/// this covers the measured render-stall range; it fills to the brim only at the
/// heaviest (8 ms) stall, then backpressures.
pub const DEFAULT_POOL_BUFS: usize = 64;

/// Why the gather stream ended. Delivered only after every buffered batch has been
/// consumed, so the main thread never observes the end ahead of pending bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherEnd {
    /// The child closed the slave (it exited): `read` returned 0, or the master
    /// reported `EIO` after draining.
    Eof,
    /// An unexpected read/poll error, carrying its `errno` for the caller to
    /// report. Never a panic: the gather thread consumes attacker-controlled bytes.
    ReadError(i32),
}

// ---------------------------------------------------------------------------
// The bounded buffer pool (the correctness crux; unit-tested in isolation)
// ---------------------------------------------------------------------------

/// The queues plus the end marker, all under one mutex.
struct Inner {
    /// Idle buffers the gather thread may fill.
    free: Vec<Box<[u8]>>,
    /// Filled buffers in FIFO publication order, each with its initialized length.
    ready: VecDeque<(Box<[u8]>, usize)>,
    /// Set once, in the same critical section as the final publish, so no
    /// completion can overtake buffered bytes.
    end: Option<GatherEnd>,
    /// Set to release a gather thread blocked on backpressure at shutdown.
    shutdown: bool,
}

/// The pool: the queues, plus the condition variable the gather thread waits on
/// when the free queue is empty (backpressure).
struct BufPool {
    inner: Mutex<Inner>,
    free_ready: Condvar,
}

impl BufPool {
    /// Allocate every buffer and reserve both queues up front, so moving a buffer
    /// between them in steady state never touches the allocator.
    fn new(nbufs: usize) -> Arc<BufPool> {
        let mut free = Vec::with_capacity(nbufs);
        for _ in 0..nbufs {
            free.push(vec![0u8; BUF_CAP].into_boxed_slice());
        }
        Arc::new(BufPool {
            inner: Mutex::new(Inner {
                free,
                ready: VecDeque::with_capacity(nbufs),
                end: None,
                shutdown: false,
            }),
            free_ready: Condvar::new(),
        })
    }

    /// Lock the inner state, recovering the guard even if a thread poisoned the
    /// mutex by panicking (nothing here panics, but this keeps the path panic-free
    /// rather than `unwrap`).
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gather side: take an idle buffer, blocking while the free queue is empty
    /// (the intentional backpressure). Returns `None` only on shutdown.
    fn take_free(&self) -> Option<Box<[u8]>> {
        let mut g = self.lock();
        loop {
            if g.shutdown {
                return None;
            }
            if let Some(b) = g.free.pop() {
                return Some(b);
            }
            g = self.free_ready.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Gather side: publish a filled buffer. Returns whether the ready queue went
    /// empty→nonempty, the only transition the caller signals the wake eventfd on
    /// (see the coalescing-correctness note below).
    fn publish(&self, buf: Box<[u8]>, len: usize) -> bool {
        let mut g = self.lock();
        let was_empty = g.ready.is_empty();
        g.ready.push_back((buf, len));
        was_empty
    }

    /// Gather side: publish the last (possibly empty) buffer and set the end marker
    /// atomically. An empty final buffer goes straight back to `free`.
    fn publish_final(&self, buf: Box<[u8]>, len: usize, end: GatherEnd) -> bool {
        let mut g = self.lock();
        let was_empty = g.ready.is_empty();
        if len > 0 {
            g.ready.push_back((buf, len));
        } else {
            g.free.push(buf);
        }
        g.end = Some(end);
        was_empty
    }

    /// Gather side: set the end marker with no buffer in hand (e.g. a poll error
    /// before any read).
    fn finish(&self, end: GatherEnd) -> bool {
        let mut g = self.lock();
        let was_empty = g.ready.is_empty();
        g.end = Some(end);
        was_empty
    }

    /// Main side: take the oldest ready buffer, if any.
    fn pop_ready(&self) -> Option<(Box<[u8]>, usize)> {
        self.lock().ready.pop_front()
    }

    /// Main side: return a parsed buffer to `free` and wake the gather thread if it
    /// is blocked on backpressure.
    fn return_free(&self, buf: Box<[u8]>) {
        {
            let mut g = self.lock();
            g.free.push(buf);
        }
        self.free_ready.notify_one();
    }

    /// Main side: the end marker, but only once the ready queue is drained,
    /// enforcing "all data batches → end marker → close".
    fn end_if_drained(&self) -> Option<GatherEnd> {
        let g = self.lock();
        if g.ready.is_empty() {
            g.end
        } else {
            None
        }
    }

    /// Main side: whether any ready batch is still undrained.
    fn has_ready(&self) -> bool {
        !self.lock().ready.is_empty()
    }

    /// Shutdown side: release a `take_free` blocked on an empty free queue.
    fn request_shutdown(&self) {
        self.lock().shutdown = true;
        self.free_ready.notify_all();
    }
}

// Correctness of the empty→nonempty wake coalescing, given exactly one producer
// (the gather thread), one consumer (the main thread), and one mutex serialising
// every queue operation:
//
//   * The consumer sleeps only after observing `ready` empty under the lock.
//   * The producer skips the wake only when it observed `ready` *non-empty* under
//     the lock. A non-empty ready queue means an undrained buffer exists, so the
//     consumer is not asleep (it drains to empty before it sleeps) and will pop
//     this buffer in the same pass — no wake needed.
//   * When the producer *does* observe empty, it signals; the eventfd counter
//     persists until the consumer reads it, so a signal racing the consumer's
//     pre-sleep window still makes the next poll return at once.
//
// The worst case is a harmless spurious wakeup, never a lost one.

// ---------------------------------------------------------------------------
// The gather thread
// ---------------------------------------------------------------------------

/// Outcome of one nonblocking read; `EIO` on a PTY master is EOF (the slave is
/// gone), not an error.
enum Rd {
    Data(usize),
    WouldBlock,
    Eof,
    Err(c_int),
}

fn do_read(fd: RawFd, buf: &mut [u8]) -> Rd {
    loop {
        // SAFETY: buf is a valid writable slice; read writes at most its len.
        let n = unsafe { read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n > 0 {
            return Rd::Data(n as usize);
        }
        if n == 0 {
            return Rd::Eof;
        }
        match errno() {
            EINTR => continue,
            EAGAIN => return Rd::WouldBlock,
            EIO => return Rd::Eof,
            e => return Rd::Err(e),
        }
    }
}

/// Which of the two watched fds became readable in one wait.
struct PollBits {
    pty: bool,
    stop: bool,
}

/// Wait for the PTY read fd or the stop fd, collapsing `EINTR`. A `timeout` cap is
/// pure defence: no wake should be lost, but a bug must not hang the thread.
fn poll_pty_or_stop(
    read_fd: RawFd,
    stop_fd: RawFd,
    timeout: Duration,
) -> core::result::Result<PollBits, c_int> {
    let ts = KernelTimespec {
        tv_sec: timeout.as_secs() as i64,
        tv_nsec: timeout.subsec_nanos() as i64,
    };
    loop {
        let mut fds = [
            Pollfd {
                fd: read_fd,
                events: POLLIN,
                revents: 0,
            },
            Pollfd {
                fd: stop_fd,
                events: POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: fds points at two valid pollfd entries; ts is a live timespec; a
        // null sigmask leaves the signal mask unchanged.
        let r = unsafe {
            ppoll(
                fds.as_mut_ptr(),
                2,
                &ts as *const KernelTimespec,
                core::ptr::null(),
            )
        };
        if r < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            return Err(e);
        }
        let hit = |p: &Pollfd| p.revents & (POLLIN | POLLHUP | POLLERR) != 0;
        return Ok(PollBits {
            pty: hit(&fds[0]),
            stop: hit(&fds[1]),
        });
    }
}

/// The gather loop: wait for the PTY to be readable (or a stop), then drain it into
/// pool buffers, publishing full buffers at once and any nonempty partial on the
/// first `EAGAIN` (baseline policy). The pool mutex is never held across a read,
/// poll, or eventfd write.
fn gather_loop(read_fd: RawFd, ready_efd: RawFd, stop_efd: RawFd, pool: Arc<BufPool>) {
    let wake = |woke: bool| {
        if woke {
            efd_signal(ready_efd);
        }
    };

    'outer: loop {
        match poll_pty_or_stop(read_fd, stop_efd, Duration::from_millis(100)) {
            Ok(bits) => {
                if bits.stop {
                    break; // clean shutdown
                }
                if !bits.pty {
                    continue; // timeout / spurious wake
                }
            }
            Err(e) => {
                wake(pool.finish(GatherEnd::ReadError(e)));
                break;
            }
        }

        let mut buf = match pool.take_free() {
            Some(b) => b,
            None => break, // shutdown while waiting for a free buffer
        };
        let mut len = 0usize;

        loop {
            if len == buf.len() {
                wake(pool.publish(buf, len));
                buf = match pool.take_free() {
                    Some(b) => b,
                    None => break 'outer,
                };
                len = 0;
            }
            match do_read(read_fd, &mut buf[len..]) {
                Rd::Data(n) => len += n,
                Rd::WouldBlock => {
                    // Baseline: publish any nonempty batch now, back to the wait.
                    if len > 0 {
                        wake(pool.publish(buf, len));
                    } else {
                        pool.return_free(buf);
                    }
                    continue 'outer;
                }
                Rd::Eof => {
                    wake(pool.publish_final(buf, len, GatherEnd::Eof));
                    break 'outer;
                }
                Rd::Err(e) => {
                    wake(pool.publish_final(buf, len, GatherEnd::ReadError(e)));
                    break 'outer;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The public handle
// ---------------------------------------------------------------------------

/// One ready batch, borrowed from the pool. Its bytes are valid until it drops,
/// at which point the buffer returns to the free queue for reuse. Parse
/// [`Batch::bytes`] and let it drop before taking the next batch.
pub struct Batch<'a> {
    buf: Option<Box<[u8]>>,
    len: usize,
    pool: &'a BufPool,
}

impl Batch<'_> {
    /// The initialized bytes the gather thread read into this buffer.
    pub fn bytes(&self) -> &[u8] {
        match &self.buf {
            Some(b) => &b[..self.len],
            None => &[],
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.return_free(b);
        }
    }
}

/// A running gather thread and the channel back to the main loop. Poll
/// [`Gatherer::ready_fd`] beside the Wayland socket; when it signals, call
/// [`Gatherer::clear_wakeup`] then drain [`Gatherer::next_batch`] until it yields
/// `None`, feeding each batch to the parser; [`Gatherer::completion`] reports the
/// end once every batch is drained. Dropping the `Gatherer` stops and joins the
/// thread.
pub struct Gatherer {
    pool: Arc<BufPool>,
    handle: Option<JoinHandle<()>>,
    /// The wake eventfd the event loop polls; the gather thread writes it.
    ready_efd: OwnedFd,
    /// The stop eventfd; dropping the `Gatherer` writes it to wake a polling thread.
    stop_efd: OwnedFd,
    /// The duplicated master fd the gather thread reads through. Held here so it
    /// outlives the thread: `Drop` joins the thread *before* this closes.
    _read_fd: OwnedFd,
}

impl Gatherer {
    /// Start gathering from `master_fd` (a PTY master). The fd is duplicated with
    /// `F_DUPFD_CLOEXEC` for the gather thread, so the caller keeps sole use of the
    /// original for writes and control. Uses the default 4 MiB pool.
    pub fn start(master_fd: RawFd) -> Result<Gatherer> {
        Gatherer::start_with_pool(master_fd, DEFAULT_POOL_BUFS)
    }

    /// As [`Gatherer::start`], with an explicit pool depth (buffers of [`BUF_CAP`]).
    pub fn start_with_pool(master_fd: RawFd, nbufs: usize) -> Result<Gatherer> {
        let read_fd = dup_cloexec(master_fd)?;
        let ready_efd = make_eventfd()?;
        let stop_efd = make_eventfd()?;
        let pool = BufPool::new(nbufs);

        let tpool = pool.clone();
        let (rfd, refd, sefd) = (
            read_fd.as_raw_fd(),
            ready_efd.as_raw_fd(),
            stop_efd.as_raw_fd(),
        );
        // The thread reads through `rfd` and signals/waits on `refd`/`sefd`, all
        // owned by this struct. Drop joins the thread before those fds close, so
        // the raw descriptors the thread holds never dangle.
        let handle = thread::Builder::new()
            .name("pty-gather".to_string())
            .spawn(move || gather_loop(rfd, refd, sefd, tpool))
            .map_err(|e| Error::msg(format!("spawn gather thread: {e}")))?;

        Ok(Gatherer {
            pool,
            handle: Some(handle),
            ready_efd,
            stop_efd,
            _read_fd: read_fd,
        })
    }

    /// The eventfd to poll beside the Wayland socket. Readable means the gather
    /// thread published at least one batch since the last [`Gatherer::clear_wakeup`].
    pub fn ready_fd(&self) -> RawFd {
        self.ready_efd.as_raw_fd()
    }

    /// Drain the wake eventfd's counter. Call once after `poll` reports
    /// [`Gatherer::ready_fd`] readable, before pulling batches, so the next `poll`
    /// blocks until the next publish rather than spinning.
    pub fn clear_wakeup(&self) {
        efd_clear(self.ready_efd.as_raw_fd());
    }

    /// Take the next ready batch, or `None` if the ready queue is momentarily
    /// empty. The returned [`Batch`] returns its buffer to the pool when dropped.
    pub fn next_batch(&self) -> Option<Batch<'_>> {
        self.pool.pop_ready().map(|(buf, len)| Batch {
            buf: Some(buf),
            len,
            pool: &self.pool,
        })
    }

    /// The stream-end marker, reported only once every ready batch has been taken,
    /// so a caller draining to `None` then checking this never sees the end ahead
    /// of buffered bytes.
    pub fn completion(&self) -> Option<GatherEnd> {
        self.pool.end_if_drained()
    }

    /// Whether ready batches remain undrained (i.e. the caller stopped on its
    /// fairness budget). The event loop bypasses its next blocking wait while this
    /// is true, so a continuous producer is drained across turns without stalling.
    pub fn has_pending(&self) -> bool {
        self.pool.has_ready()
    }
}

impl Drop for Gatherer {
    fn drop(&mut self) {
        // Wake the thread whether it is polling (stop eventfd) or blocked waiting
        // for a free buffer (shutdown flag + condvar), then join it before the
        // owned fds close.
        self.pool.request_shutdown();
        efd_signal(self.stop_efd.as_raw_fd());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// FFI. eventfd, F_DUPFD_CLOEXEC, and a two-fd ppoll, wrapped so the rest of the
// module deals only in safe types. Isolated here; the fd-owning
// wrappers hand back `OwnedFd` so descriptors are never leaked.
// ---------------------------------------------------------------------------

// fcntl(2) command for a close-on-exec duplicate (F_LINUX_SPECIFIC_BASE + 6).
const F_DUPFD_CLOEXEC: c_int = 1030;
// eventfd2(2) flags.
const EFD_CLOEXEC: c_int = 0o2000000;
const EFD_NONBLOCK: c_int = 0o4000;
// poll events.
const POLLIN: c_short = 0x001;
const POLLERR: c_short = 0x008;
const POLLHUP: c_short = 0x010;
// errno values we branch on.
const EINTR: c_int = 4;
const EIO: c_int = 5;
const EAGAIN: c_int = 11; // == EWOULDBLOCK on Linux

/// `struct pollfd` (`poll.h`); its size is pinned against the C ABI in the tests.
#[repr(C)]
struct Pollfd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

/// `struct timespec` (`time.h`), the `ppoll` timeout; size pinned in the tests.
#[repr(C)]
struct KernelTimespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn eventfd(initval: c_uint, flags: c_int) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn ppoll(
        fds: *mut Pollfd,
        nfds: c_ulong,
        timeout: *const KernelTimespec,
        sigmask: *const c_void,
    ) -> c_int;
    fn __errno_location() -> *mut c_int;
}

fn errno() -> c_int {
    // SAFETY: glibc exposes a valid thread-local errno here.
    unsafe { *__errno_location() }
}

fn errno_error(what: &str) -> Error {
    Error::msg(format!("{what} failed: errno {}", errno()))
}

/// Duplicate `fd` with `F_DUPFD_CLOEXEC`: the new fd shares the same open file
/// description (and its `O_NONBLOCK` flag), which is what the gather thread reads
/// through while the original stays the caller's control handle.
fn dup_cloexec(fd: RawFd) -> Result<OwnedFd> {
    // SAFETY: F_DUPFD_CLOEXEC takes an int minimum-fd arg; fd is a valid descriptor.
    let d = unsafe { fcntl(fd, F_DUPFD_CLOEXEC, 0) };
    if d < 0 {
        return Err(errno_error("fcntl(F_DUPFD_CLOEXEC)"));
    }
    // SAFETY: d is a fresh, owned descriptor returned by fcntl.
    Ok(unsafe { OwnedFd::from_raw_fd(d) })
}

/// A close-on-exec, nonblocking counter eventfd used purely as a wakeup.
fn make_eventfd() -> Result<OwnedFd> {
    // SAFETY: eventfd with valid flags returns a fresh fd or -1.
    let fd = unsafe { eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK) };
    if fd < 0 {
        return Err(errno_error("eventfd"));
    }
    // SAFETY: fd is a fresh, owned descriptor returned by eventfd.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Add 1 to the eventfd's counter to wake a poller. Best-effort: a failed wake is
/// covered by the poll timeout safety net, so it is never fatal.
fn efd_signal(fd: RawFd) {
    let one: u64 = 1;
    // SAFETY: writing 8 bytes of a u64 is the eventfd contract; overflow at
    // u64::MAX is unreachable at these rates.
    unsafe { write(fd, &one as *const u64 as *const c_void, 8) };
}

/// Drain the eventfd's counter to zero. Nonblocking, so an already-clear fd just
/// returns `EAGAIN`, which is fine.
fn efd_clear(fd: RawFd) {
    let mut v: u64 = 0;
    loop {
        // SAFETY: reading 8 bytes into a u64 is the eventfd contract.
        let n = unsafe { read(fd, &mut v as *mut u64 as *mut c_void, 8) };
        if n < 0 && errno() == EINTR {
            continue;
        }
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    // -- ABI pins -----------------------------------------------------------

    #[test]
    fn pollfd_and_timespec_match_the_c_abi() {
        assert_eq!(std::mem::size_of::<Pollfd>(), 8);
        assert_eq!(std::mem::size_of::<KernelTimespec>(), 16);
    }

    // -- Pure pool / state-machine invariants -------------------------------

    /// Total buffers are conserved across arbitrary take/publish/pop/return.
    #[test]
    fn pool_conserves_buffers() {
        let pool = BufPool::new(4);
        let count = |p: &BufPool| {
            let g = p.lock();
            g.free.len() + g.ready.len()
        };
        assert_eq!(count(&pool), 4);
        let a = pool.take_free().unwrap();
        let b = pool.take_free().unwrap();
        assert_eq!(count(&pool) + 2, 4); // two in flight
        pool.publish(a, 10);
        let (buf, len) = pool.pop_ready().unwrap();
        assert_eq!(len, 10);
        pool.return_free(buf);
        pool.return_free(b);
        assert_eq!(count(&pool), 4);
    }

    /// Ready batches come back in publication order.
    #[test]
    fn ready_is_fifo() {
        let pool = BufPool::new(4);
        for tag in 0..3u8 {
            let mut buf = pool.take_free().unwrap();
            buf[0] = tag;
            pool.publish(buf, 1);
        }
        for tag in 0..3u8 {
            let (buf, _) = pool.pop_ready().unwrap();
            assert_eq!(buf[0], tag);
            pool.return_free(buf);
        }
    }

    /// `publish` reports the empty→nonempty edge exactly once per idle→busy cycle.
    #[test]
    fn publish_signals_only_on_empty_to_nonempty_edge() {
        let pool = BufPool::new(4);
        let b0 = pool.take_free().unwrap();
        let b1 = pool.take_free().unwrap();
        assert!(pool.publish(b0, 1), "first publish is empty→nonempty");
        assert!(!pool.publish(b1, 1), "second publish sees a nonempty queue");
        let (buf, _) = pool.pop_ready().unwrap();
        pool.return_free(buf);
        let (buf, _) = pool.pop_ready().unwrap();
        pool.return_free(buf);
        let b2 = pool.take_free().unwrap();
        assert!(pool.publish(b2, 1), "publish after drain is a fresh edge");
    }

    /// The end marker is withheld until the ready queue drains.
    #[test]
    fn end_marker_is_ordered_after_data() {
        let pool = BufPool::new(4);
        let buf = pool.take_free().unwrap();
        pool.publish_final(buf, 5, GatherEnd::Eof);
        assert_eq!(
            pool.end_if_drained(),
            None,
            "end hidden while a batch waits"
        );
        let (buf, len) = pool.pop_ready().unwrap();
        assert_eq!(len, 5);
        pool.return_free(buf);
        assert_eq!(pool.end_if_drained(), Some(GatherEnd::Eof));
    }

    /// An empty final buffer still records the end and recycles the buffer.
    #[test]
    fn empty_final_records_end_and_recycles() {
        let pool = BufPool::new(2);
        let buf = pool.take_free().unwrap();
        pool.publish_final(buf, 0, GatherEnd::ReadError(5));
        assert_eq!(pool.end_if_drained(), Some(GatherEnd::ReadError(5)));
        assert_eq!(
            pool.lock().free.len(),
            2,
            "empty final buffer returned to free"
        );
    }

    /// Buffers cycle by identity; no new allocation happens after construction.
    #[test]
    fn buffers_are_recycled_not_reallocated() {
        let pool = BufPool::new(3);
        let original: std::collections::HashSet<*const u8> = {
            let g = pool.lock();
            g.free.iter().map(|b| b.as_ptr()).collect()
        };
        for _ in 0..50 {
            let buf = pool.take_free().unwrap();
            let was = pool.publish(buf, 1);
            assert!(was || pool.lock().ready.len() > 1);
            let (buf, _) = pool.pop_ready().unwrap();
            assert!(
                original.contains(&buf.as_ptr()),
                "a buffer pointer appeared that was not allocated at construction"
            );
            pool.return_free(buf);
        }
    }

    /// A `take_free` blocked on an empty free queue wakes when a buffer returns.
    #[test]
    fn take_free_blocks_then_wakes_on_return() {
        let pool = BufPool::new(1);
        let held = pool.take_free().unwrap(); // free is now empty
        let p2 = pool.clone();
        let waiter = thread::spawn(move || {
            let b = p2.take_free().expect("woken with a buffer, not shutdown");
            p2.return_free(b);
        });
        thread::sleep(Duration::from_millis(20)); // let the waiter block
        pool.return_free(held); // wakes it
        waiter.join().unwrap();
        assert_eq!(pool.lock().free.len(), 1);
    }

    /// Shutdown releases a blocked `take_free` with `None`.
    #[test]
    fn take_free_returns_none_on_shutdown() {
        let pool = BufPool::new(1);
        let _held = pool.take_free().unwrap(); // free empty
        let p2 = pool.clone();
        let waiter = thread::spawn(move || p2.take_free().is_none());
        thread::sleep(Duration::from_millis(20));
        pool.request_shutdown();
        assert!(
            waiter.join().unwrap(),
            "shutdown must release the waiter as None"
        );
    }

    // -- Real-PTY integration ----------------------------------------------

    const O_RDWR: c_int = 0o2;
    const O_NOCTTY: c_int = 0o400;
    const O_NONBLOCK: c_int = 0o4000;
    const F_GETFL: c_int = 3;
    const F_SETFL: c_int = 4;
    const TIOCSCTTY: c_ulong = 0x540E;

    extern "C" {
        fn posix_openpt(flags: c_int) -> c_int;
        fn grantpt(fd: c_int) -> c_int;
        fn unlockpt(fd: c_int) -> c_int;
        fn ptsname_r(fd: c_int, buf: *mut core::ffi::c_char, buflen: usize) -> c_int;
        fn fork() -> c_int;
        fn setsid() -> c_int;
        fn execvp(file: *const core::ffi::c_char, argv: *const *const core::ffi::c_char) -> c_int;
        fn dup2(oldfd: c_int, newfd: c_int) -> c_int;
        fn open(path: *const core::ffi::c_char, flags: c_int) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
        fn _exit(code: c_int) -> !;
    }

    /// A spawned PTY child, closing the master and reaping on drop. `None` when
    /// fork/exec is unavailable (a locked-down sandbox), so tests can skip cleanly.
    struct PtyChild {
        master: c_int,
        pid: c_int,
    }

    impl Drop for PtyChild {
        fn drop(&mut self) {
            unsafe {
                close(self.master);
                let mut status = 0;
                waitpid(self.pid, &mut status, 0);
            }
        }
    }

    /// Fork `argv` on a fresh nonblocking cooked PTY (OPOST/ONLCR untouched).
    fn spawn_on_pty(argv: &[&str]) -> Option<PtyChild> {
        unsafe {
            let master = posix_openpt(O_RDWR | O_NOCTTY);
            if master < 0 {
                return None;
            }
            if grantpt(master) != 0 || unlockpt(master) != 0 {
                close(master);
                return None;
            }
            let mut namebuf = [0u8; 128];
            if ptsname_r(
                master,
                namebuf.as_mut_ptr() as *mut core::ffi::c_char,
                namebuf.len(),
            ) != 0
            {
                close(master);
                return None;
            }
            let flags = fcntl(master, F_GETFL);
            fcntl(master, F_SETFL, flags | O_NONBLOCK);

            let cargv: Vec<CString> = argv.iter().map(|a| CString::new(*a).unwrap()).collect();
            let mut ptrs: Vec<*const core::ffi::c_char> =
                cargv.iter().map(|c| c.as_ptr()).collect();
            ptrs.push(core::ptr::null());

            let pid = fork();
            if pid < 0 {
                close(master);
                return None;
            }
            if pid == 0 {
                setsid();
                let slave = open(namebuf.as_ptr() as *const core::ffi::c_char, O_RDWR);
                if slave < 0 {
                    _exit(127);
                }
                ioctl(slave, TIOCSCTTY, 0 as c_ulong);
                dup2(slave, 0);
                dup2(slave, 1);
                dup2(slave, 2);
                if slave > 2 {
                    close(slave);
                }
                close(master);
                execvp(ptrs[0], ptrs.as_ptr());
                _exit(127);
            }
            Some(PtyChild { master, pid })
        }
    }

    /// Drive a gatherer to completion, feeding every batch to `sink`, with a wall
    /// deadline so a hang fails the test instead of blocking forever.
    fn drain_to_end(g: &Gatherer, mut sink: impl FnMut(&[u8]), deadline: Duration) -> GatherEnd {
        let start = Instant::now();
        loop {
            while let Some(b) = g.next_batch() {
                sink(b.bytes());
            }
            if let Some(end) = g.completion() {
                return end;
            }
            let mut fds = [Pollfd {
                fd: g.ready_fd(),
                events: POLLIN,
                revents: 0,
            }];
            let ts = KernelTimespec {
                tv_sec: 0,
                tv_nsec: 20_000_000,
            };
            unsafe { ppoll(fds.as_mut_ptr(), 1, &ts, core::ptr::null()) };
            g.clear_wakeup();
            assert!(
                start.elapsed() < deadline,
                "gather did not finish before deadline"
            );
        }
    }

    /// The ONLCR-expanded FNV of a payload: what the cooked PTY emits for it.
    fn expected_fnv(payload: &[u8]) -> (u64, u64) {
        let mut hash = 0xcbf29ce484222325u64;
        let mut total = 0u64;
        let mut push = |b: u8| {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
            total += 1;
        };
        for &b in payload {
            if b == b'\n' {
                push(b'\r');
                push(b'\n');
            } else {
                push(b);
            }
        }
        (hash, total)
    }

    fn fnv_of(hash: &AtomicU64, bytes: &[u8]) {
        let mut h = hash.load(Ordering::Relaxed);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        hash.store(h, Ordering::Relaxed);
    }

    /// Write a temp file of `size` pseudo-random printable bytes and return its path.
    fn temp_payload(name: &str, size: usize) -> (std::path::PathBuf, Vec<u8>) {
        let mut path = std::env::temp_dir();
        path.push(format!("bnkterm-gather-{name}-{}", std::process::id()));
        let mut data = Vec::with_capacity(size);
        let mut x = 0x243f6a8885a308d3u64;
        for _ in 0..size {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Printable ASCII plus the occasional newline (exercises ONLCR).
            let b = if x & 0x3f == 0 {
                b'\n'
            } else {
                32 + (x % 94) as u8
            };
            data.push(b);
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        (path, data)
    }

    /// Exact byte stream over many buffer turnovers (a multi-MB cat).
    #[test]
    fn gather_delivers_exact_stream_over_many_turnovers() {
        let (path, data) = temp_payload("turnovers", 3 * 1024 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            eprintln!("pty unavailable; skipping");
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start(child.master).unwrap();
        let (want_hash, want_len) = expected_fnv(&data);
        let hash = AtomicU64::new(0xcbf29ce484222325u64);
        let got = AtomicU64::new(0);
        let end = drain_to_end(
            &g,
            |b| {
                fnv_of(&hash, b);
                got.fetch_add(b.len() as u64, Ordering::Relaxed);
            },
            Duration::from_secs(20),
        );
        assert_eq!(end, GatherEnd::Eof);
        assert_eq!(got.load(Ordering::Relaxed), want_len, "byte count");
        assert_eq!(hash.load(Ordering::Relaxed), want_hash, "stream checksum");
        let _ = std::fs::remove_file(&path);
    }

    /// A payload smaller than one batch: a single partial buffer, then EOF.
    #[test]
    fn gather_publishes_final_partial_then_eof() {
        let (path, data) = temp_payload("small", 200);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start(child.master).unwrap();
        let (want_hash, want_len) = expected_fnv(&data);
        let hash = AtomicU64::new(0xcbf29ce484222325u64);
        let got = AtomicU64::new(0);
        let end = drain_to_end(
            &g,
            |b| {
                fnv_of(&hash, b);
                got.fetch_add(b.len() as u64, Ordering::Relaxed);
            },
            Duration::from_secs(5),
        );
        assert_eq!(end, GatherEnd::Eof);
        assert_eq!(got.load(Ordering::Relaxed), want_len);
        assert_eq!(hash.load(Ordering::Relaxed), want_hash);
        let _ = std::fs::remove_file(&path);
    }

    /// Pool exhaustion and recovery: a two-buffer pool with a slow consumer still
    /// delivers the exact stream (backpressure turns the buffers over correctly).
    #[test]
    fn pool_exhaustion_and_recovery() {
        let (path, data) = temp_payload("exhaust", 1024 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start_with_pool(child.master, 2).unwrap();
        let (want_hash, want_len) = expected_fnv(&data);
        let hash = AtomicU64::new(0xcbf29ce484222325u64);
        let got = AtomicU64::new(0);
        let mut n = 0u32;
        let end = drain_to_end(
            &g,
            |b| {
                // Occasionally dawdle so the two-buffer pool must backpressure.
                n += 1;
                if n.is_multiple_of(4) {
                    thread::sleep(Duration::from_micros(500));
                }
                fnv_of(&hash, b);
                got.fetch_add(b.len() as u64, Ordering::Relaxed);
            },
            Duration::from_secs(30),
        );
        assert_eq!(end, GatherEnd::Eof);
        assert_eq!(got.load(Ordering::Relaxed), want_len);
        assert_eq!(hash.load(Ordering::Relaxed), want_hash);
        let _ = std::fs::remove_file(&path);
    }

    /// A consumer stall longer than the pool runway: the gather thread backpressures
    /// (child blocks), and no bytes are lost when the consumer resumes.
    #[test]
    fn consumer_stall_longer_than_runway() {
        let (path, data) = temp_payload("stall", 2 * 1024 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start_with_pool(child.master, 4).unwrap();
        let (want_hash, want_len) = expected_fnv(&data);
        let hash = AtomicU64::new(0xcbf29ce484222325u64);
        let got = AtomicU64::new(0);
        let mut stalled = false;
        let end = drain_to_end(
            &g,
            |b| {
                if !stalled {
                    stalled = true;
                    thread::sleep(Duration::from_millis(50)); // > 4×64 KiB of runway
                }
                fnv_of(&hash, b);
                got.fetch_add(b.len() as u64, Ordering::Relaxed);
            },
            Duration::from_secs(30),
        );
        assert_eq!(end, GatherEnd::Eof);
        assert_eq!(got.load(Ordering::Relaxed), want_len);
        assert_eq!(hash.load(Ordering::Relaxed), want_hash);
        let _ = std::fs::remove_file(&path);
    }

    /// Stop while the gather thread is polling an idle child: `Drop` joins promptly.
    #[test]
    fn stop_while_polling() {
        // `sleep` produces nothing, so the gather thread parks in poll.
        let Some(child) = spawn_on_pty(&["/bin/sleep", "30"]) else {
            return;
        };
        let g = Gatherer::start(child.master).unwrap();
        thread::sleep(Duration::from_millis(30)); // ensure it is parked in poll
        let start = Instant::now();
        drop(g); // must wake via the stop eventfd and join
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "join hung on shutdown"
        );
    }

    /// Stop while the gather thread is blocked waiting for a free buffer: a tiny
    /// pool plus a consumer that never drains parks the thread in `take_free`;
    /// `Drop` must release it.
    #[test]
    fn stop_while_waiting_for_free() {
        let (path, _data) = temp_payload("waitfree", 4 * 1024 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start_with_pool(child.master, 1).unwrap();
        // Never call next_batch: the one buffer fills, publishes, and the thread
        // blocks in take_free waiting for the free queue.
        thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        drop(g);
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "join hung waiting for free"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The child exits while filled buffers are still queued: draining yields every
    /// byte, then the EOF marker.
    #[test]
    fn child_exit_while_buffers_queued() {
        let (path, data) = temp_payload("queued", 512 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start(child.master).unwrap();
        // Let the child finish and exit before we start draining, so batches and
        // the EOF marker are all queued when we begin.
        thread::sleep(Duration::from_millis(100));
        let (want_hash, want_len) = expected_fnv(&data);
        let hash = AtomicU64::new(0xcbf29ce484222325u64);
        let got = AtomicU64::new(0);
        let end = drain_to_end(
            &g,
            |b| {
                fnv_of(&hash, b);
                got.fetch_add(b.len() as u64, Ordering::Relaxed);
            },
            Duration::from_secs(10),
        );
        assert_eq!(end, GatherEnd::Eof);
        assert_eq!(got.load(Ordering::Relaxed), want_len);
        assert_eq!(hash.load(Ordering::Relaxed), want_hash);
        let _ = std::fs::remove_file(&path);
    }

    /// A read error propagates through the pool as `ReadError`, ordered after any
    /// buffered data, exactly like EOF. (A genuine non-EIO PTY read error is not
    /// inducible without a mock, so the propagation is exercised at the pool seam
    /// the gather loop feeds; the loop's error branch is straight-line code that
    /// calls `publish_final` identically to the EOF branch.)
    #[test]
    fn read_error_propagates_after_data() {
        let pool = BufPool::new(2);
        let buf = pool.take_free().unwrap();
        pool.publish(buf, 7); // a data batch already queued
        let buf = pool.take_free().unwrap();
        pool.publish_final(buf, 3, GatherEnd::ReadError(9)); // EBADF, say
        assert_eq!(pool.end_if_drained(), None, "error hidden behind data");
        let (buf, _) = pool.pop_ready().unwrap();
        pool.return_free(buf);
        assert_eq!(pool.end_if_drained(), None, "still one batch to go");
        let (buf, len) = pool.pop_ready().unwrap();
        assert_eq!(len, 3);
        pool.return_free(buf);
        assert_eq!(pool.end_if_drained(), Some(GatherEnd::ReadError(9)));
    }

    /// No buffer allocation in steady state: pointer identity is preserved across a
    /// full real-PTY cat, so every delivered buffer is one of the originals.
    #[test]
    fn no_steady_state_buffer_allocation() {
        let (path, _data) = temp_payload("noalloc", 1024 * 1024);
        let Some(child) = spawn_on_pty(&["/bin/cat", path.to_str().unwrap()]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let g = Gatherer::start_with_pool(child.master, 8).unwrap();
        let original: std::collections::HashSet<*const u8> = {
            let inner = g.pool.lock();
            inner.free.iter().map(|b| b.as_ptr()).collect()
        };
        let deadline = Duration::from_secs(20);
        let start = Instant::now();
        let end = loop {
            while let Some(b) = g.next_batch() {
                assert!(
                    original.contains(&b.bytes().as_ptr()),
                    "a batch used a buffer not allocated at construction"
                );
            }
            if let Some(end) = g.completion() {
                break end;
            }
            let mut fds = [Pollfd {
                fd: g.ready_fd(),
                events: POLLIN,
                revents: 0,
            }];
            let ts = KernelTimespec {
                tv_sec: 0,
                tv_nsec: 20_000_000,
            };
            unsafe { ppoll(fds.as_mut_ptr(), 1, &ts, core::ptr::null()) };
            g.clear_wakeup();
            assert!(start.elapsed() < deadline, "no-alloc drain hung");
        };
        assert_eq!(end, GatherEnd::Eof);
        let _ = std::fs::remove_file(&path);
    }
}
