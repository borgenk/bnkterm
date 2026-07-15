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
//!
//! # Process environment
//!
//! `TERM` and `COLORTERM` are process-wide settings inherited by every spawned
//! child. The app exports them once at startup, before a gather thread exists;
//! [`Pty::spawn`] deliberately never mutates the environment because later calls
//! happen while the process is multi-threaded.

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

/// What the tty's line discipline is currently doing, decoded from the two
/// `c_lflag` bits that decide it. The child owns this state (it is the child that
/// calls `tcsetattr`); we only ever read it.
///
/// ```text
///   ICANON  ECHO   mode             who does this
///   ------  ----   --------------   ----------------------------------
///     1      1     Cooked           an ordinary shell prompt
///     1      0     PasswordPrompt   sudo / ssh / passwd reading a secret
///     0      *     Raw              vim / htop driving the screen itself
/// ```
///
/// `PasswordPrompt` is the interesting one, and it is why the pair is needed rather
/// than `ECHO` alone: a full-screen program *also* turns echo off, so echo-off on its
/// own cannot tell "hide what I type" from "I am painting the screen myself". Keeping
/// the line editor on (`ICANON`) while blinding it is the distinctive thing a password
/// prompt does, and it is the same test wezterm and ghostty make.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtyMode {
    /// Canonical with echo: the kernel line-edits and prints what is typed.
    Cooked,
    /// Canonical without echo: the kernel line-edits but prints nothing, so the
    /// user is typing something they do not want on the screen.
    PasswordPrompt,
    /// Non-canonical: the program reads keys itself and owns the display.
    Raw,
}

/// A spawned child on the far side of a pseudoterminal, owning the master fd and
/// the child's pid. Dropping it closes the master (which sends the child SIGHUP)
/// and reaps the child if it has exited.
pub struct Pty {
    master: OwnedFd,
    pid: i32,
}

/// A child whose PTY master has been closed but whose process may not have exited
/// yet. The app keeps these lightweight pid handles and retries nonblocking reaps
/// between event-loop turns so a tab closed mid-session cannot remain a zombie.
pub struct ZombieChild {
    pid: i32,
}

impl Pty {
    /// Open a PTY, size it to `cols` x `rows`, and fork `$SHELL` (or `/bin/sh`) on
    /// the slave. The master is non-blocking (so the event loop can drain it
    /// without stalling) and close-on-exec (so it is never inherited by another
    /// tab's shell). The caller sets process-wide terminal capability
    /// variables once, before any gather threads start, so every child inherits
    /// them without mutating the environment from a multi-threaded process.
    pub fn spawn(cols: usize, rows: usize) -> Result<Pty> {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        Self::spawn_command(cols, rows, &[&shell])
    }

    /// The general form of [`spawn`](Self::spawn): fork `argv` (with `argv[0]` the
    /// program) on a fresh PTY slave instead of the shell. The live terminal wants
    /// the shell; the golden-fixture capture tool wants an arbitrary command, and
    /// both share this one fork/exec dance so the `unsafe` never leaves this module.
    /// Errors on an empty `argv` or an argument carrying an interior NUL.
    pub fn spawn_command(cols: usize, rows: usize, argv: &[&str]) -> Result<Pty> {
        if argv.is_empty() {
            return Err(Error::msg("spawn_command needs a program to run"));
        }
        // O_CLOEXEC is set atomically at open so a *later* tab's fork/exec cannot
        // leak this master into its shell. A leaked master keeps its slave's
        // hangup from firing, so closing this tab would never reap its child (see
        // the multi-tab teardown in `super::tabs`). Doing it here, not with a
        // follow-up fcntl, closes the window where a concurrent fork could inherit
        // the fd before the flag is set.
        // SAFETY: posix_openpt with O_RDWR|O_NOCTTY|O_CLOEXEC returns a fresh master
        // fd or -1; we take ownership of a valid fd or map the error.
        let master_raw = unsafe { posix_openpt(O_RDWR | O_NOCTTY | O_CLOEXEC) };
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

        // Everything the child touches is built here, in the parent, where the
        // allocator is safe to use. The `CString`s own the bytes; `ptrs` is the
        // NUL-terminated argv (a trailing null past the arguments) the child execs.
        let cstrings: Vec<CString> = argv
            .iter()
            .map(|arg| {
                CString::new(*arg)
                    .map_err(|_| Error::msg("a command argument contains an interior NUL"))
            })
            .collect::<Result<_>>()?;
        let mut ptrs: Vec<*const c_char> = cstrings.iter().map(|c| c.as_ptr()).collect();
        ptrs.push(core::ptr::null());

        // SAFETY: all pointers outlive the call; slave_path and the argv strings are
        // NUL-terminated and stay alive through the fork. The child branch runs
        // only async-signal-safe syscalls before exec (see the module header).
        let pid = unsafe { fork_child_in_pty(master.as_raw_fd(), slave_path.as_ptr(), &ptrs) };
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

    /// The child's working directory, resolved from its `/proc/<pid>/cwd` symlink.
    /// `None` if the child has exited or the link cannot be read (no `/proc`, a
    /// permission edge). This is how a tab labels itself with its shell's directory
    /// without depending on the shell emitting OSC 7; it is read only when a tab's
    /// output settles, never on the byte path, so a plain `read_link` is fine.
    pub fn cwd(&self) -> Option<std::path::PathBuf> {
        std::fs::read_link(format!("/proc/{}/cwd", self.pid)).ok()
    }

    /// The name (`comm`) of the program in the foreground of this PTY, e.g. `zsh` at
    /// a prompt or `claude` while that runs. Resolved from `/proc/<pid>/stat`'s
    /// `tpgid` field (the tty's foreground process group), then that leader's
    /// `/proc/<tpgid>/comm`. `None` if `/proc` is unavailable or the fields cannot be
    /// read. Like [`cwd`](Self::cwd) it is read only when a tab settles, never on the
    /// byte path, so plain file reads are fine. Lets the tab bar prefix a directory
    /// only for configured programs without depending on any shell integration.
    ///
    /// `/proc/<pid>/stat` layout: field 2 (`comm`) is parenthesized and may itself
    /// contain spaces or `)`, so the fixed fields are parsed from *after the last*
    /// `)`. Counting from there: state, ppid, pgrp, session, tty_nr, tpgid — so
    /// `tpgid` is the sixth whitespace token.
    pub fn foreground_program(&self) -> Option<String> {
        let stat = std::fs::read_to_string(format!("/proc/{}/stat", self.pid)).ok()?;
        let tpgid = parse_tpgid(&stat)?;
        if tpgid <= 0 {
            return None;
        }
        let comm = std::fs::read_to_string(format!("/proc/{tpgid}/comm")).ok()?;
        let name = comm.trim_end_matches('\n');
        (!name.is_empty()).then(|| name.to_string())
    }

    /// What the line discipline is doing right now (see [`TtyMode`]), read straight
    /// from the kernel. `None` if the tty is gone, in which case the caller keeps
    /// whatever it last saw rather than inventing a mode.
    ///
    /// The master and the slave name the same line discipline, so a `tcsetattr` the
    /// *child* makes on its own tty is visible here without the child telling us
    /// anything: this is how a password prompt is noticed at all, since `sudo` prints
    /// no escape sequence to announce itself. It is one `TCGETS`, and like
    /// [`cwd`](Self::cwd) and [`foreground_program`](Self::foreground_program) it is
    /// called only when a tab's output settles, never per read (the drain does ~950k
    /// reads over a 150 MB cat; a syscall on that path is exactly the tax `PERF_LOG`
    /// exists to prevent).
    pub fn tty_mode(&self) -> Option<TtyMode> {
        // SAFETY: `t` is a live, correctly-typed local; tcgetattr either fills it for
        // a valid fd or returns nonzero, and we read it only on success.
        let mut t: Termios = unsafe { core::mem::zeroed() };
        if unsafe { tcgetattr(self.master.as_raw_fd(), &mut t) } != 0 {
            return None;
        }
        Some(match (t.c_lflag & ICANON != 0, t.c_lflag & ECHO != 0) {
            (true, true) => TtyMode::Cooked,
            (true, false) => TtyMode::PasswordPrompt,
            (false, _) => TtyMode::Raw,
        })
    }

    /// Do to this tty exactly what a password prompt does to its own: clear (or set)
    /// `ECHO` on the live line discipline. Test-only, and not a mock: it drives the
    /// same kernel object `sudo` drives, from the other end of the same pty (proved
    /// equivalent by `a_password_prompt_on_the_slave_side_is_visible_on_the_master`).
    /// It exists so the terminal's own tests can raise a real password prompt without
    /// an interactive shell, whose line editor moves the termios out from under them.
    #[cfg(test)]
    pub(crate) fn set_echo(&self, on: bool) {
        // SAFETY: `t` is a live, correctly-typed local, and the fd is this pty's own
        // master; tcgetattr fills `t` and tcsetattr applies it unchanged but for ECHO.
        let mut t: Termios = unsafe { core::mem::zeroed() };
        assert_eq!(unsafe { tcgetattr(self.master.as_raw_fd(), &mut t) }, 0);
        if on {
            t.c_lflag |= ECHO;
        } else {
            t.c_lflag &= !ECHO;
        }
        assert_eq!(
            unsafe { tcsetattr(self.master.as_raw_fd(), TCSANOW, &t) },
            0
        );
    }

    /// Reap the child if it has exited, returning its exit status, else `None`
    /// (still running). Non-blocking.
    pub fn reap(&self) -> Option<c_int> {
        let mut status: c_int = 0;
        // SAFETY: status is a live local; WNOHANG makes this non-blocking.
        let r = unsafe { waitpid(self.pid, &mut status, WNOHANG) };
        (r == self.pid).then_some(status)
    }

    /// Close the master (hanging up the child's terminal session) and retain only
    /// the pid for later nonblocking reaping.
    pub fn into_zombie(self) -> ZombieChild {
        // `Pty` implements Drop, so its fields cannot be moved out directly. Keep
        // the allocation inert, explicitly drop the master exactly once, and copy
        // the plain pid into the lightweight handoff object.
        let mut this = std::mem::ManuallyDrop::new(self);
        let pid = this.pid;
        // SAFETY: `ManuallyDrop` suppresses `Pty::drop`; `master` is initialized
        // and is dropped exactly here. The only remaining field is the Copy pid.
        unsafe { std::ptr::drop_in_place(&mut this.master) };
        ZombieChild { pid }
    }
}

impl ZombieChild {
    /// Try `waitpid(WNOHANG)`. Returns true once the child has been claimed (or is
    /// no longer waitable), at which point the caller should discard this handle.
    pub fn try_reap(&self) -> bool {
        let mut status: c_int = 0;
        // SAFETY: status is a live local and WNOHANG makes the wait nonblocking.
        let r = unsafe { waitpid(self.pid, &mut status, WNOHANG) };
        r == self.pid || (r < 0 && errno() == ECHILD)
    }
}

/// Extract the `tpgid` (the tty's foreground process group) from a `/proc/<pid>/stat`
/// line. Field 2, `comm`, is wrapped in parentheses and may itself contain spaces or
/// `)`, so the fixed numeric fields are read from *after the last* `)`. Counting from
/// there: state, ppid, pgrp, session, tty_nr, tpgid, so `tpgid` is the sixth token.
/// Returns `None` if the line is malformed.
fn parse_tpgid(stat: &str) -> Option<i32> {
    let after_comm = stat.rsplit_once(')')?.1;
    after_comm.split_whitespace().nth(5)?.parse().ok()
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Closing the master hangs up the slave (SIGHUP to the child); then reap
        // so it does not linger as a zombie. The OwnedFd closes itself after.
        let _ = self.reap();
    }
}

/// A reusable set of file descriptors waited on with `poll(2)`.
///
/// Callers clear and refill the set each loop turn. The backing vector keeps the
/// capacity grown so far, so the steady-state idle wait does not allocate even as
/// the registered descriptors change.
pub struct PollSet {
    fds: Vec<Pollfd>,
}

impl PollSet {
    /// Make an empty poll set. Capacity grows only when [`Self::add`] takes the
    /// set above its previous high-water mark.
    pub fn new() -> Self {
        Self { fds: Vec::new() }
    }

    /// Remove every registered descriptor while retaining the backing capacity.
    pub fn clear(&mut self) {
        self.fds.clear();
    }

    /// Register `fd` for readable, hangup, and error notification, returning the
    /// stable slot used to inspect this wait's result with [`Self::readable`].
    pub fn add(&mut self, fd: RawFd) -> usize {
        let slot = self.fds.len();
        self.fds.push(Pollfd {
            fd,
            events: POLLIN,
            revents: 0,
        });
        slot
    }

    /// Block until a registered descriptor is ready or `timeout` elapses.
    /// `None` waits indefinitely.
    pub fn wait(&mut self, timeout: Option<Duration>) -> Result<()> {
        let millis = match timeout {
            None => -1,
            Some(d) => d.as_millis().min(c_int::MAX as u128) as c_int,
        };
        loop {
            // SAFETY: `fds` owns `len` initialized `Pollfd` entries and remains
            // exclusively borrowed for the duration of the call.
            let r = unsafe { poll(self.fds.as_mut_ptr(), self.fds.len() as c_ulong, millis) };
            if r < 0 {
                if errno() == EINTR {
                    continue;
                }
                return Err(errno_error("poll"));
            }
            return Ok(());
        }
    }

    /// Whether `idx` was reported readable, hung up, or errored by the last wait.
    /// An out-of-range slot is never ready.
    pub fn readable(&self, idx: usize) -> bool {
        self.fds
            .get(idx)
            .is_some_and(|fd| fd.revents & (POLLIN | POLLHUP | POLLERR) != 0)
    }
}

impl Default for PollSet {
    fn default() -> Self {
        Self::new()
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
const O_CLOEXEC: c_int = 0o2000000;
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
/// `ICANON` and `ECHO` (`termios.h` `c_lflag`): the line editor and the echo of what
/// is typed. Read together they classify the tty (see [`TtyMode`]); their values are
/// pinned in the tests, since decoding a password prompt from the wrong bits would
/// silently show a lock at the wrong times.
const ICANON: u32 = 0o2;
const ECHO: u32 = 0o10;
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
const ECHILD: c_int = 10;

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
    #[cfg(test)]
    fn pipe(pipefd: *mut c_int) -> c_int;
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
/// `argv` must be non-empty (`argv[0]` is the program), which the callers ensure.
unsafe fn fork_child_in_pty(
    master: RawFd,
    slave_path: *const c_char,
    argv: &[*const c_char],
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
    fn tpgid_is_parsed_past_a_paren_or_space_in_comm() {
        // A normal line: comm "zsh", tpgid is the sixth field after ')'.
        // pid (comm) state ppid pgrp session tty_nr tpgid ...
        let stat = "1234 (zsh) S 1200 1234 1234 34816 5678 4194304 ...";
        assert_eq!(parse_tpgid(stat), Some(5678));

        // A hostile comm containing spaces and its own ')': the parse must key off
        // the *last* ')', so the fields still align and tpgid reads correctly.
        let nasty = "42 (weird ) name) R 1 42 42 0 -1 4194560 ...";
        assert_eq!(parse_tpgid(nasty), Some(-1));

        // At a prompt with no distinct foreground the caller treats tpgid == pgrp
        // as "the shell"; a parse of it still succeeds (the filtering is elsewhere).
        let prompt = "9 (bash) S 1 9 9 34816 9 4194304";
        assert_eq!(parse_tpgid(prompt), Some(9));

        // Malformed input yields None, never a panic on attacker-adjacent data.
        assert_eq!(parse_tpgid("garbage with no paren"), None);
        assert_eq!(parse_tpgid("7 (sh) S 1 7"), None);
    }

    #[test]
    fn termios_matches_the_c_abi() {
        // tcgetattr/tcsetattr read and write the whole struct, so its size must
        // match `struct termios` exactly: 4 flag words (16) + c_line (1) +
        // c_cc[32] (32), padded to align the two 4-byte speeds = 60 on Linux.
        assert_eq!(std::mem::size_of::<Termios>(), 60);
        assert_eq!(std::mem::align_of::<Termios>(), 4);
        // The c_lflag bits `tty_mode` decodes (asm-generic/termbits.h). Reading the
        // wrong bit would not fail loudly, it would just show a lock at the wrong
        // moments, so the values are pinned rather than trusted.
        assert_eq!(ICANON, 0x2);
        assert_eq!(ECHO, 0x8);
    }

    /// Flip one `c_lflag` bit on the live line discipline behind `fd`.
    fn set_lflag(fd: RawFd, bits: u32, on: bool) {
        let mut t: Termios = unsafe { core::mem::zeroed() };
        assert_eq!(unsafe { tcgetattr(fd, &mut t) }, 0, "tcgetattr");
        if on {
            t.c_lflag |= bits;
        } else {
            t.c_lflag &= !bits;
        }
        assert_eq!(unsafe { tcsetattr(fd, TCSANOW, &t) }, 0, "tcsetattr");
    }

    #[test]
    fn tty_mode_decodes_every_line_discipline_state() {
        // Against a real pty and the kernel's real line discipline (no mocks): drive
        // the two bits through all four combinations and check each classification.
        std::env::set_var("SHELL", "/bin/cat");
        let Ok(pty) = Pty::spawn(80, 24) else {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        };
        let fd = pty.fd();

        // A tty comes up cooked: the kernel line-edits and echoes.
        assert_eq!(pty.tty_mode(), Some(TtyMode::Cooked));

        // Echo off, line editor on: sudo/ssh/passwd reading a secret.
        set_lflag(fd, ECHO, false);
        assert_eq!(pty.tty_mode(), Some(TtyMode::PasswordPrompt));

        // Restoring echo (what sudo does once it has the password) ends the prompt.
        set_lflag(fd, ECHO, true);
        assert_eq!(pty.tty_mode(), Some(TtyMode::Cooked));

        // Line editor off: a full-screen program reading keys itself. Echo is off here
        // too, which is exactly why ECHO alone cannot stand in for a password prompt.
        set_lflag(fd, ICANON, false);
        set_lflag(fd, ECHO, false);
        assert_eq!(pty.tty_mode(), Some(TtyMode::Raw));

        // Non-canonical but echoing is a rare, deliberate state; it is still the
        // program driving the tty, so it must not read as a prompt.
        set_lflag(fd, ECHO, true);
        assert_eq!(pty.tty_mode(), Some(TtyMode::Raw));
    }

    #[test]
    fn a_password_prompt_on_the_slave_side_is_visible_on_the_master() {
        // The premise the whole feature rests on: `sudo` announces nothing: it calls
        // tcsetattr on *its* end of the pty (the slave, which is its stdin). If that
        // were not the same line discipline the master reads, no amount of polling
        // would ever see a password prompt. Prove it by doing precisely what sudo does
        // -- open the slave and clear ECHO on it -- and reading the master.
        std::env::set_var("SHELL", "/bin/cat");
        let Ok(pty) = Pty::spawn(80, 24) else {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        };
        assert_eq!(pty.tty_mode(), Some(TtyMode::Cooked));

        let slave_path = ptsname(pty.fd()).expect("the slave's path");
        // SAFETY: slave_path is a NUL-terminated path from ptsname_r. O_NOCTTY keeps
        // the test process from adopting the pty as its controlling terminal.
        let slave = unsafe { open(slave_path.as_ptr(), O_RDWR | O_NOCTTY) };
        assert!(slave >= 0, "open the slave: errno {}", errno());
        set_lflag(slave, ECHO, false);

        assert_eq!(pty.tty_mode(), Some(TtyMode::PasswordPrompt));

        // SAFETY: `slave` is the fd just opened above and is not used again.
        unsafe { close(slave) };
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
        let mut poll_set = PollSet::new();
        poll_set.add(pty.fd());
        for _ in 0..200 {
            match pty.read(&mut buf).expect("read from the child") {
                ReadOutcome::Data(n) => {
                    got.extend_from_slice(&buf[..n]);
                    if got.windows(9).any(|w| w == b"hello pty") {
                        return; // success
                    }
                }
                ReadOutcome::WouldBlock => {
                    let _ = poll_set.wait(Some(Duration::from_millis(50)));
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

    #[test]
    fn into_zombie_eventually_reaps_after_hangup() {
        std::env::set_var("SHELL", "/bin/cat");
        let Ok(pty) = Pty::spawn(80, 24) else {
            return; // sandbox without fork/exec
        };
        let child = pty.into_zombie();
        let pid = child.pid;
        let mut reaped = false;
        for _ in 0..200 {
            if child.try_reap() {
                reaped = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(reaped, "hung-up child did not become waitable");

        let mut status = 0;
        // SAFETY: `pid` belonged to this test and was reaped above; this verifies
        // a second wait cannot claim it again.
        assert_eq!(unsafe { waitpid(pid, &mut status, WNOHANG) }, -1);
        assert_eq!(errno(), ECHILD);
    }

    fn test_pipe() -> (OwnedFd, OwnedFd) {
        let mut fds = [-1; 2];
        // SAFETY: `fds` has room for the two descriptors written by pipe(2). On
        // success both are fresh and transferred immediately into `OwnedFd`s.
        assert_eq!(unsafe { pipe(fds.as_mut_ptr()) }, 0);
        unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) }
    }

    fn write_byte(fd: RawFd) {
        let byte = b'x';
        // SAFETY: `fd` is the live write side of a test pipe and `byte` is a
        // readable one-byte buffer.
        assert_eq!(
            unsafe { write(fd, &byte as *const u8 as *const c_void, 1) },
            1
        );
    }

    #[test]
    fn poll_set_maps_readiness_by_slot() {
        let (read_a, write_a) = test_pipe();
        let (read_b, write_b) = test_pipe();
        let mut set = PollSet::new();
        let a = set.add(read_a.as_raw_fd());
        let b = set.add(read_b.as_raw_fd());

        write_byte(write_b.as_raw_fd());
        set.wait(Some(Duration::from_millis(50)))
            .expect("poll the test pipes");

        assert!(!set.readable(a));
        assert!(set.readable(b));

        // Keep the write ends live until after poll so an idle read side is not
        // reported as a hangup.
        drop((write_a, write_b));
    }

    #[test]
    fn poll_set_timeout_reports_nothing_readable() {
        let (read, write) = test_pipe();
        let mut set = PollSet::new();
        let slot = set.add(read.as_raw_fd());

        set.wait(Some(Duration::from_millis(1)))
            .expect("poll timeout");

        assert!(!set.readable(slot));
        drop(write);
    }

    #[test]
    fn poll_set_clear_reuses_capacity() {
        let (read_a, write_a) = test_pipe();
        let (read_b, write_b) = test_pipe();
        let mut set = PollSet::new();
        set.add(read_a.as_raw_fd());
        set.add(read_b.as_raw_fd());
        let capacity = set.fds.capacity();

        set.clear();
        assert_eq!(set.add(read_b.as_raw_fd()), 0);
        assert_eq!(set.add(read_a.as_raw_fd()), 1);

        assert_eq!(set.fds.capacity(), capacity);
        drop((write_a, write_b));
    }
}
