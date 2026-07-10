//! The pseudoterminal: spawn `$SHELL` on the slave side of a PTY and talk to it
//! over the master fd. This is the terminal's other mouth. `vt.rs` reads what the
//! child says (its output bytes) and `input.rs` writes what the user says (key
//! bytes); this module is the pipe between them and the child process.
//!
//! ```text
//!   Pty::spawn ─▶ posix_openpt ─▶ fork ─┬─ child: setsid, TIOCSCTTY, dup2, exec $SHELL
//!                                        └─ parent: master fd (non-blocking) ── read/write
//! ```
//!
//! # Why the FFI lives here
//!
//! The PTY needs a cluster of libc calls (`posix_openpt`, `fork`, `execvp`, the
//! tty ioctls) that the portable `platform` layer deliberately does not carry:
//! not every app built on that layer has a PTY, so putting this in the vendored
//! leaf would muddy it. The raw ABI stays isolated in one module; this
//! file *is* that module for the PTY concern. The `unsafe` is confined to the FFI
//! section at the bottom and wrapped so [`Pty`]'s callers only ever deal in safe
//! types and `Result`.
//!
//! # The fork/exec dance
//!
//! By the time we spawn, the Vulkan driver may have started threads, so between
//! `fork` and `exec` the child may touch only async-signal-safe calls: no heap,
//! no `Result`, no panic. Every buffer the child needs (the argv pointers, the
//! slave path) is therefore built in the parent *before* the fork, and the child
//! branch calls nothing but raw syscalls and `_exit`.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::time::Duration;

use core::ffi::{c_char, c_int, c_short, c_ulong, c_void};

use crate::error::{Error, Result};

/// The outcome of a non-blocking read from the master fd.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadOutcome {
    /// `n` bytes were read into the caller's buffer.
    Data(usize),
    /// No data was ready (the child has produced nothing since the last read).
    WouldBlock,
    /// The child closed the slave (it exited); the terminal should shut down.
    Eof,
}

/// A spawned child on the far side of a pseudoterminal, owning the master fd and
/// the child's pid. Dropping it closes the master (which sends the child SIGHUP)
/// and reaps the child if it has exited.
pub struct Pty {
    master: OwnedFd,
    pid: i32,
}

impl Pty {
    /// Open a PTY, size it to `cols` x `rows`, and fork `$SHELL` (or `/bin/sh`) on
    /// the slave. The master is left non-blocking so the event loop can drain it
    /// without stalling. `TERM`/`COLORTERM` are exported so the child advertises
    /// the right capabilities.
    pub fn spawn(cols: usize, rows: usize) -> Result<Pty> {
        // SAFETY: posix_openpt with O_RDWR|O_NOCTTY returns a fresh master fd or
        // -1; we take ownership of a valid fd or map the error.
        let master_raw = unsafe { posix_openpt(O_RDWR | O_NOCTTY) };
        if master_raw < 0 {
            return Err(errno_error("posix_openpt"));
        }
        // SAFETY: master_raw is a fresh, owned fd from posix_openpt.
        let master = unsafe { OwnedFd::from_raw_fd(master_raw) };

        // SAFETY: master is a valid PTY master; grantpt/unlockpt only read it.
        if unsafe { grantpt(master.as_raw_fd()) } != 0 {
            return Err(errno_error("grantpt"));
        }
        if unsafe { unlockpt(master.as_raw_fd()) } != 0 {
            return Err(errno_error("unlockpt"));
        }

        let slave_path = ptsname(master.as_raw_fd())?;
        set_winsize(master.as_raw_fd(), cols, rows)?;
        set_nonblocking(master.as_raw_fd())?;
        // Mark tty input as UTF-8 (the child inherits it before the fork). The
        // output flags are left at the cooked default, so the pty still works if
        // this fails; it is not fatal.
        let _ = enable_iutf8(master.as_raw_fd());

        // Exported before the fork so the child inherits them. The process is
        // still single-purpose here; set_var is safe on this edition.
        std::env::set_var("TERM", "xterm-256color");
        std::env::set_var("COLORTERM", "truecolor");

        // Everything the child touches is built here, in the parent, where the
        // allocator is safe to use.
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let shell_c =
            CString::new(shell).map_err(|_| Error::msg("$SHELL contains an interior NUL"))?;
        // argv is NUL-terminated; argv[0] is the program itself.
        let argv: [*const c_char; 2] = [shell_c.as_ptr(), core::ptr::null()];

        // SAFETY: all pointers outlive the call; slave_path and shell_c are
        // NUL-terminated and stay alive through the fork. The child branch runs
        // only async-signal-safe syscalls before exec (see the module header).
        let pid = unsafe { fork_child_in_pty(master.as_raw_fd(), slave_path.as_ptr(), &argv) };
        if pid < 0 {
            return Err(errno_error("fork"));
        }
        Ok(Pty { master, pid })
    }

    /// The master fd, for the event loop to `poll` alongside the Wayland socket.
    pub fn fd(&self) -> RawFd {
        self.master.as_raw_fd()
    }

    /// Read whatever the child has produced into `buf`, non-blocking. `Eof` means
    /// the child exited; `WouldBlock` means nothing is ready yet.
    pub fn read(&self, buf: &mut [u8]) -> Result<ReadOutcome> {
        loop {
            // SAFETY: buf is a valid writable slice; read writes at most its len.
            let n = unsafe {
                read(
                    self.master.as_raw_fd(),
                    buf.as_mut_ptr() as *mut c_void,
                    buf.len(),
                )
            };
            if n > 0 {
                return Ok(ReadOutcome::Data(n as usize));
            }
            if n == 0 {
                return Ok(ReadOutcome::Eof);
            }
            match errno() {
                EINTR => continue,
                EAGAIN => return Ok(ReadOutcome::WouldBlock),
                // The kernel reports EIO on the master once the slave is gone and
                // its output is drained: the child has exited.
                EIO => return Ok(ReadOutcome::Eof),
                e => return Err(Error::msg(format!("pty read failed: errno {e}"))),
            }
        }
    }

    /// Write every byte to the child, retrying short writes and `EINTR`, and
    /// waiting for room on `EAGAIN` (the child is momentarily not reading). This
    /// carries the encoded key bytes to the shell.
    pub fn write_all(&self, mut bytes: &[u8]) -> Result<()> {
        while !bytes.is_empty() {
            // SAFETY: bytes is a valid slice; write reads at most its len.
            let n = unsafe {
                write(
                    self.master.as_raw_fd(),
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                )
            };
            if n > 0 {
                bytes = &bytes[n as usize..];
                continue;
            }
            match errno() {
                EINTR => continue,
                EAGAIN => {
                    // The PTY input buffer is full; wait until it drains.
                    poll_writable(self.master.as_raw_fd())?;
                }
                e => return Err(Error::msg(format!("pty write failed: errno {e}"))),
            }
        }
        Ok(())
    }

    /// Tell the child the window is now `cols` x `rows` cells; the kernel raises
    /// SIGWINCH on the child so full-screen programs repaint at the new size.
    pub fn resize(&self, cols: usize, rows: usize) -> Result<()> {
        set_winsize(self.master.as_raw_fd(), cols, rows)
    }

    /// Reap the child if it has exited, returning its exit status, else `None`
    /// (still running). Non-blocking.
    pub fn reap(&self) -> Option<c_int> {
        let mut status: c_int = 0;
        // SAFETY: status is a live local; WNOHANG makes this non-blocking.
        let r = unsafe { waitpid(self.pid, &mut status, WNOHANG) };
        (r == self.pid).then_some(status)
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Closing the master hangs up the slave (SIGHUP to the child); then reap
        // so it does not linger as a zombie. The OwnedFd closes itself after.
        let _ = self.reap();
    }
}

/// Which fds became readable in a single wait: the Wayland socket (`wayland`) and
/// optionally the PTY master (`pty`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ready {
    pub wayland: bool,
    pub pty: bool,
}

/// Block until the Wayland fd or the (optional) PTY fd is readable, or `timeout`
/// elapses. Stack-allocated pollfds, so the idle wait never touches the heap.
/// `timeout` of `None` blocks indefinitely.
pub fn wait_readable(
    wayland: RawFd,
    pty: Option<RawFd>,
    timeout: Option<Duration>,
) -> Result<Ready> {
    let mut fds = [
        Pollfd {
            fd: wayland,
            events: POLLIN,
            revents: 0,
        },
        Pollfd {
            fd: pty.unwrap_or(-1),
            events: POLLIN,
            revents: 0,
        },
    ];
    let nfds: c_ulong = if pty.is_some() { 2 } else { 1 };
    let millis = match timeout {
        None => -1,
        Some(d) => d.as_millis().min(c_int::MAX as u128) as c_int,
    };
    loop {
        // SAFETY: fds points at `nfds` valid pollfd entries for the call.
        let r = unsafe { poll(fds.as_mut_ptr(), nfds, millis) };
        if r < 0 {
            if errno() == EINTR {
                continue;
            }
            return Err(errno_error("poll"));
        }
        return Ok(Ready {
            wayland: fds[0].revents & (POLLIN | POLLHUP | POLLERR) != 0,
            pty: pty.is_some() && fds[1].revents & (POLLIN | POLLHUP | POLLERR) != 0,
        });
    }
}

/// Block until `fd` is writable (used to wait out a full PTY input buffer).
fn poll_writable(fd: RawFd) -> Result<()> {
    let mut pfd = Pollfd {
        fd,
        events: POLLOUT,
        revents: 0,
    };
    loop {
        // SAFETY: one valid pollfd; -1 timeout blocks until writable.
        let r = unsafe { poll(&mut pfd, 1, -1) };
        if r < 0 && errno() == EINTR {
            continue;
        }
        if r < 0 {
            return Err(errno_error("poll"));
        }
        return Ok(());
    }
}

// ---------------------------------------------------------------------------
// FFI. Raw libc surface for the PTY, process, and poll syscalls, wrapped so the
// rest of the module deals only in safe types, all confined to this one place.
// ---------------------------------------------------------------------------

// open(2) / posix_openpt(3) flags (Linux generic ABI).
const O_RDWR: c_int = 0o2;
const O_NOCTTY: c_int = 0o400;
const O_NONBLOCK: c_int = 0o4000;
// fcntl(2) commands.
const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
// tty ioctls (Linux).
const TIOCSCTTY: c_ulong = 0x540E;
const TIOCSWINSZ: c_ulong = 0x5414;

/// `IUTF8` (`termios.h` `c_iflag`): marks tty input as UTF-8 so a cooked-mode ERASE
/// deletes a whole multibyte character rather than one byte (see [`enable_iutf8`]).
/// `TCSANOW` applies a `tcsetattr` immediately. `NCCS` is the control-char array
/// length in `struct termios`.
const IUTF8: u32 = 0o40000;
const TCSANOW: c_int = 0;
const NCCS: usize = 32;
// waitpid options.
const WNOHANG: c_int = 1;
// poll events.
const POLLIN: c_short = 0x001;
const POLLOUT: c_short = 0x004;
const POLLERR: c_short = 0x008;
const POLLHUP: c_short = 0x010;
// errno values we branch on.
const EINTR: c_int = 4;
const EIO: c_int = 5;
const EAGAIN: c_int = 11; // == EWOULDBLOCK on Linux

/// `struct winsize` (`sys/ioctl.h`): the cell dimensions a `TIOCSWINSZ` carries.
/// The pixel fields are left zero; programs that care read the cell counts.
#[repr(C)]
struct Winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

/// `struct pollfd` (`poll.h`).
#[repr(C)]
struct Pollfd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

/// `struct termios` (`termios.h`), Linux generic ABI: four flag words, the line
/// discipline byte, the control-char array, and the two speeds. Mirrored field for
/// field only to read the current settings, set `IUTF8` in `c_iflag`, and write
/// them back; the size is pinned in the tests against the C ABI.
#[repr(C)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; NCCS],
    c_ispeed: u32,
    c_ospeed: u32,
}

extern "C" {
    fn posix_openpt(flags: c_int) -> c_int;
    fn grantpt(fd: c_int) -> c_int;
    fn unlockpt(fd: c_int) -> c_int;
    fn ptsname_r(fd: c_int, buf: *mut c_char, buflen: usize) -> c_int;
    fn fork() -> c_int;
    fn setsid() -> c_int;
    fn execvp(file: *const c_char, argv: *const *const c_char) -> c_int;
    fn dup2(oldfd: c_int, newfd: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn open(path: *const c_char, flags: c_int) -> c_int;
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn poll(fds: *mut Pollfd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn tcgetattr(fd: c_int, termios: *mut Termios) -> c_int;
    fn tcsetattr(fd: c_int, actions: c_int, termios: *const Termios) -> c_int;
    fn _exit(code: c_int) -> !;
    fn __errno_location() -> *mut c_int;
}

fn errno() -> c_int {
    // SAFETY: glibc/musl expose a valid thread-local errno here.
    unsafe { *__errno_location() }
}

fn errno_error(what: &str) -> Error {
    Error::msg(format!("{what} failed: errno {}", errno()))
}

/// The slave device path for `master`, resolved with the reentrant `ptsname_r`.
fn ptsname(master: RawFd) -> Result<CString> {
    let mut buf = [0u8; 128];
    // SAFETY: buf is a valid writable buffer of the given length.
    let rc = unsafe { ptsname_r(master, buf.as_mut_ptr() as *mut c_char, buf.len()) };
    if rc != 0 {
        return Err(Error::msg(format!("ptsname_r failed: errno {rc}")));
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    CString::new(&buf[..end]).map_err(|_| Error::msg("ptsname returned an embedded NUL"))
}

/// Set `master` non-blocking so the event loop can drain it without stalling.
fn set_nonblocking(master: RawFd) -> Result<()> {
    // SAFETY: F_GETFL takes no argument; the fd is valid.
    let flags = unsafe { fcntl(master, F_GETFL) };
    if flags < 0 {
        return Err(errno_error("fcntl(F_GETFL)"));
    }
    // SAFETY: F_SETFL takes the new flag word; the fd is valid.
    if unsafe { fcntl(master, F_SETFL, flags | O_NONBLOCK) } < 0 {
        return Err(errno_error("fcntl(F_SETFL)"));
    }
    Ok(())
}

/// Set `IUTF8` on the tty so cooked-mode line editing treats input as UTF-8: an
/// ERASE (backspace) deletes a whole multibyte character rather than a single byte.
/// This is the one termios tweak mainstream terminals make;
/// the output flags (`OPOST`/`ONLCR`) are left at the kernel default, so programs
/// that rely on the tty mapping `\n` to `\r\n` still render correctly. Setting it on
/// the master before the fork means the child inherits it. Best-effort: on failure
/// the pty just runs without it.
fn enable_iutf8(master: RawFd) -> Result<()> {
    // SAFETY: `t` is a live, correctly-typed local; tcgetattr fills it for `master`.
    let mut t: Termios = unsafe { core::mem::zeroed() };
    if unsafe { tcgetattr(master, &mut t) } != 0 {
        return Err(errno_error("tcgetattr"));
    }
    t.c_iflag |= IUTF8;
    // SAFETY: tcsetattr reads the live `t` and applies it to `master`.
    if unsafe { tcsetattr(master, TCSANOW, &t) } != 0 {
        return Err(errno_error("tcsetattr"));
    }
    Ok(())
}

/// Push the current cell dimensions to the tty via `TIOCSWINSZ`.
fn set_winsize(master: RawFd, cols: usize, rows: usize) -> Result<()> {
    let ws = Winsize {
        ws_row: rows.min(u16::MAX as usize) as u16,
        ws_col: cols.min(u16::MAX as usize) as u16,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: TIOCSWINSZ reads a `struct winsize` through the pointer; ws is a
    // live, correctly-typed local for the duration of the call.
    if unsafe { ioctl(master, TIOCSWINSZ, &ws as *const Winsize) } < 0 {
        return Err(errno_error("ioctl(TIOCSWINSZ)"));
    }
    Ok(())
}

/// Fork, and in the child set up the slave as the controlling terminal and exec
/// the shell. Returns the child pid to the parent, or -1 on a fork failure.
///
/// # Safety
///
/// The child branch runs between `fork` and `exec` in a possibly-threaded
/// process, so it calls only async-signal-safe syscalls and never allocates,
/// returns a `Result`, or panics; `slave_path` and `argv` must be valid,
/// NUL-terminated, and outlive the call (the caller builds them before forking).
unsafe fn fork_child_in_pty(
    master: RawFd,
    slave_path: *const c_char,
    argv: &[*const c_char; 2],
) -> c_int {
    let pid = fork();
    if pid != 0 {
        return pid; // parent (or -1); nothing else to do here
    }
    // --- child: async-signal-safe only, _exit on any failure ---
    setsid();
    let slave = open(slave_path, O_RDWR);
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
    execvp(argv[0], argv.as_ptr());
    // Only reached if exec failed.
    _exit(127);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winsize_and_pollfd_match_the_c_abi() {
        // These mirror C structs the kernel writes/reads; pin their sizes.
        assert_eq!(std::mem::size_of::<Winsize>(), 8);
        assert_eq!(std::mem::size_of::<Pollfd>(), 8);
    }

    #[test]
    fn termios_matches_the_c_abi() {
        // tcgetattr/tcsetattr read and write the whole struct, so its size must
        // match `struct termios` exactly: 4 flag words (16) + c_line (1) +
        // c_cc[32] (32), padded to align the two 4-byte speeds = 60 on Linux.
        assert_eq!(std::mem::size_of::<Termios>(), 60);
        assert_eq!(std::mem::align_of::<Termios>(), 4);
    }

    #[test]
    fn spawns_a_child_and_round_trips_bytes() {
        // End-to-end against a real child (no mocks): run `cat`, which echoes its
        // input back, and confirm the bytes make the round trip through the PTY.
        // Skipped where fork/exec is unavailable (a locked-down sandbox).
        std::env::set_var("SHELL", "/bin/cat");
        let Ok(pty) = Pty::spawn(80, 24) else {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        };
        pty.write_all(b"hello pty\n").expect("write to the child");
        // Drain until the echo arrives (cat echoes line-buffered by the tty).
        let mut got = Vec::new();
        let mut buf = [0u8; 4096];
        for _ in 0..200 {
            match pty.read(&mut buf).expect("read from the child") {
                ReadOutcome::Data(n) => {
                    got.extend_from_slice(&buf[..n]);
                    if got.windows(9).any(|w| w == b"hello pty") {
                        return; // success
                    }
                }
                ReadOutcome::WouldBlock => {
                    let _ =
                        wait_readable(pty.fd(), Some(pty.fd()), Some(Duration::from_millis(50)));
                }
                ReadOutcome::Eof => break,
            }
        }
        panic!("the child never echoed the input back: got {got:?}");
    }

    #[test]
    fn resize_after_spawn_is_accepted() {
        std::env::set_var("SHELL", "/bin/cat");
        let Ok(pty) = Pty::spawn(80, 24) else {
            return; // sandbox without fork/exec
        };
        pty.resize(120, 40).expect("TIOCSWINSZ on a live pty");
    }
}
