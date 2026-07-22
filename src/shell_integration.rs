//! Auto-injected zsh shell integration.
//!
//! A terminal that reflows text on resize and a shell that repaints its prompt on
//! `SIGWINCH` disagree unless the shell tells the terminal where its prompt is: without
//! that, reflowing a full-width prompt bar to a different row count than the shell redraws
//! leaves the extra row stranded as visible garbage (see the freeze in
//! [`crate::grid::Screen::resize`]). The signal is OSC 133 semantic-prompt marks. Plain
//! zsh emits none, so the terminal has to arrange for them itself — which is exactly what
//! kitty and ghostty do, and the only reason they "just work" where alacritty (which
//! reflows but ships no integration) still mangles the prompt.
//!
//! ## The mechanism: `ZDOTDIR` redirection
//!
//! zsh reads its startup files from `$ZDOTDIR` (default `$HOME`). Point it at a generated
//! directory whose files source the user's real startup and then add the marks, and the
//! user's config runs untouched with the integration layered on top. The subtlety is that
//! `$ZDOTDIR/.zshenv` is read *before* `.zshrc`, and the user's `.zshenv` may itself set
//! `ZDOTDIR`, so each of our stages restores the user's `ZDOTDIR` to source their file,
//! records where it ended up, then points `ZDOTDIR` back at us so zsh finds our next
//! stage. A one-shot `precmd` restores it for good once startup is done.
//!
//! ```text
//!   zsh startup            our file (in $BNKTERM_INT_ZDOTDIR)     effect
//!   ─────────────────────  ────────────────────────────────────  ────────────────────────
//!   $ZDOTDIR/.zshenv   ──▶ restore user ZDOTDIR, source theirs,  user env as normal
//!   [.zprofile if login]   record it, point ZDOTDIR back at us
//!   $ZDOTDIR/.zshrc    ──▶ …source theirs… + add OSC 133 hooks    prompts now marked
//!   [.zlogin if login]
//! ```
//!
//! Best-effort throughout: a missing `$SHELL`, a non-zsh shell, an opt-out, or any I/O
//! failure leaves the environment exactly as it was. The child inherits the process
//! environment at `fork`, so [`install`] mutates it and therefore **must** run before any
//! thread starts (as [`crate::app::run`] does, beside the other capability exports).

use std::ffi::OsStr;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::PathBuf;

/// Set to `0` to disable auto-injection, for a user whose exotic startup it disturbs.
const OPT_OUT_VAR: &str = "BNKTERM_SHELL_INTEGRATION";

/// Holds the generated integration directory alive for the process and removes it on drop.
/// zsh sources the files once at a child's startup and never reads them again, so the
/// directory only has to outlive the session, which this does by living in
/// [`crate::app::run`]'s stack frame.
pub struct Session {
    dir: PathBuf,
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Point `$ZDOTDIR` at a generated integration directory for the zsh children this process
/// will spawn, so their prompts carry OSC 133 marks. Returns a [`Session`] the caller keeps
/// alive (dropping it removes the directory), or `None` when nothing was installed: not
/// zsh, opted out, or an I/O failure. Never errors — integration is a nicety, never a
/// reason the terminal fails to open.
///
/// Must be called before the first thread starts: it mutates the process environment, which
/// is unsound once other threads run.
pub fn install() -> Option<Session> {
    if std::env::var_os(OPT_OUT_VAR).as_deref() == Some(OsStr::new("0")) {
        return None;
    }
    if !shell_is_zsh(std::env::var_os("SHELL").as_deref()) {
        return None;
    }
    let dir = create_integration_dir().ok()?;
    if write_zsh_files(&dir).is_err() {
        let _ = std::fs::remove_dir_all(&dir);
        return None;
    }

    // Hand the child the user's current ZDOTDIR (so our .zshenv can restore it) and point
    // ZDOTDIR at our directory. `BNKTERM_INT_ZDOTDIR` is our directory, which the stages
    // return to after each hand-off.
    match std::env::var_os("ZDOTDIR") {
        Some(orig) => std::env::set_var("BNKTERM_ORIG_ZDOTDIR", orig),
        // No sentinel means "the user had none, fall back to $HOME"; make sure a stale one
        // inherited from a parent bnkterm cannot masquerade as the user's.
        None => std::env::remove_var("BNKTERM_ORIG_ZDOTDIR"),
    }
    std::env::set_var("BNKTERM_INT_ZDOTDIR", &dir);
    std::env::set_var("ZDOTDIR", &dir);
    Some(Session { dir })
}

/// Whether `$SHELL` names zsh, by the final path component so `/usr/bin/zsh`, `/bin/zsh`,
/// and a bare `zsh` all match while `/bin/bash` does not.
fn shell_is_zsh(shell: Option<&OsStr>) -> bool {
    shell
        .map(std::path::Path::new)
        .and_then(std::path::Path::file_name)
        .is_some_and(|name| name == OsStr::new("zsh"))
}

/// Create a fresh, private directory for the generated startup files and return its path.
/// Under `$XDG_RUNTIME_DIR` when the session has one (already a per-user 0700 directory),
/// otherwise the temp dir. Either way the name carries 128 bits from the kernel CSPRNG so it
/// is unpredictable, and it is made with `create_dir` (not `_all`) at mode 0700, so a name an
/// attacker raced to pre-create is rejected rather than silently reused. On a shared `/tmp`
/// that is what stops another user from planting the files bnkterm's zsh then sources.
fn create_integration_dir() -> std::io::Result<PathBuf> {
    let name = format!("bnkterm-shell-{}-{}", std::process::id(), random_token()?);
    // Prefer $XDG_RUNTIME_DIR (already a per-user 0700 dir); fall back to the temp dir when it
    // is unset or unusable. Security does not rest on the base: the unpredictable name, the
    // 0700 `create_dir` that fails on a pre-created name, and the O_EXCL file writes hold on a
    // shared /tmp too.
    let xdg = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from);
    let bases = xdg.into_iter().chain(std::iter::once(std::env::temp_dir()));
    let mut last_err: Option<std::io::Error> = None;
    for base in bases {
        let dir = base.join(&name);
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("no usable base directory")))
}

/// 128 bits from the kernel CSPRNG, hex-encoded, for an unpredictable directory name. Read
/// straight from `/dev/urandom` (bnkterm is Linux-only) rather than pulling in a crate.
fn random_token() -> std::io::Result<String> {
    use std::io::Read;
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    let mut token = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        for nibble in [b >> 4, b & 0x0f] {
            if let Some(c) = char::from_digit(u32::from(nibble), 16) {
                token.push(c);
            }
        }
    }
    Ok(token)
}

/// Write the four zsh startup shims into the already-created private `dir`. The
/// `.zprofile`/`.zlogin` pair matters only for a login shell (which bnkterm does not spawn
/// today, but a nested one or a future launch flag might), so they are written for
/// completeness; the load-bearing pair is `.zshenv` and `.zshrc`.
fn write_zsh_files(dir: &std::path::Path) -> std::io::Result<()> {
    write_new(dir.join(".zshenv"), ZSHENV.as_bytes())?;
    write_new(dir.join(".zprofile"), stage(".zprofile").as_bytes())?;
    write_new(dir.join(".zshrc"), zshrc().as_bytes())?;
    write_new(dir.join(".zlogin"), stage(".zlogin").as_bytes())?;
    Ok(())
}

/// Write `contents` to a brand-new file, failing if anything is already at `path`.
/// `create_new` is `O_CREAT | O_EXCL`, so a symlink or file pre-placed at the path is never
/// followed or truncated — the write fails instead. Mode 0600: the shims are ours alone.
fn write_new(path: PathBuf, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?
        .write_all(contents)
}

/// `.zshenv`: the bootstrap. It is the one stage that reads `BNKTERM_ORIG_ZDOTDIR` (the
/// user's ZDOTDIR as bnkterm found it) to restore it, sources the user's `.zshenv`, records
/// where that left ZDOTDIR (their `.zshenv` may have changed it) in `BNKTERM_USER_ZDOTDIR`
/// for the later stages, then points ZDOTDIR back at us.
const ZSHENV: &str = "\
# bnkterm shell integration (auto-injected). Source the user's zsh startup untouched, then
# add OSC 133 prompt marks. See crate::shell_integration.
if [[ -n \"${BNKTERM_ORIG_ZDOTDIR}\" ]]; then
  ZDOTDIR=\"${BNKTERM_ORIG_ZDOTDIR}\"
else
  unset ZDOTDIR
fi
unset BNKTERM_ORIG_ZDOTDIR
[[ -f \"${ZDOTDIR:-$HOME}/.zshenv\" ]] && source \"${ZDOTDIR:-$HOME}/.zshenv\"
export BNKTERM_USER_ZDOTDIR=\"${ZDOTDIR:-$HOME}\"
ZDOTDIR=\"${BNKTERM_INT_ZDOTDIR}\"
";

/// A plain hand-off stage (`.zprofile`, `.zlogin`): restore the user's ZDOTDIR, source their
/// file of the same `name`, re-record where ZDOTDIR ended up, and point it back at us.
fn stage(name: &str) -> String {
    format!(
        "\
# bnkterm shell integration (auto-injected). See crate::shell_integration.
ZDOTDIR=\"${{BNKTERM_USER_ZDOTDIR:-$HOME}}\"
[[ -f \"${{ZDOTDIR}}/{name}\" ]] && source \"${{ZDOTDIR}}/{name}\"
export BNKTERM_USER_ZDOTDIR=\"${{ZDOTDIR:-$HOME}}\"
ZDOTDIR=\"${{BNKTERM_INT_ZDOTDIR}}\"
"
    )
}

/// `.zshrc`: the hand-off stage, then the actual integration — OSC 133 `precmd`/`preexec`
/// hooks, and a one-shot `precmd` that restores the user's ZDOTDIR once startup is done (it
/// runs after `.zlogin`, so a login shell's last stage is still found first).
fn zshrc() -> String {
    format!(
        "{handoff}
# --- OSC 133 semantic prompt marks: tell the terminal where the prompt is, so a resize
# --- reflow does not fight the shell's own prompt redraw. See crate::grid::Screen::resize.
if [[ -o interactive ]] && autoload -Uz add-zsh-hook 2>/dev/null && (( $+functions[add-zsh-hook] )); then
  _bnkterm_precmd()  {{ local r=$?; print -rn -- $'\\e]133;D;'\"$r\"$'\\a\\e]133;A\\a' }}
  _bnkterm_preexec() {{ print -rn -- $'\\e]133;C\\a' }}
  add-zsh-hook precmd  _bnkterm_precmd
  add-zsh-hook preexec _bnkterm_preexec
  _bnkterm_restore_zdotdir() {{
    export ZDOTDIR=\"${{BNKTERM_USER_ZDOTDIR:-$HOME}}\"
    unset BNKTERM_USER_ZDOTDIR BNKTERM_INT_ZDOTDIR
    add-zsh-hook -d precmd _bnkterm_restore_zdotdir
    unset -f _bnkterm_restore_zdotdir
  }}
  add-zsh-hook precmd _bnkterm_restore_zdotdir
fi
",
        handoff = stage(".zshrc")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_zsh_by_the_final_path_component() {
        for yes in ["/usr/bin/zsh", "/bin/zsh", "zsh", "/opt/homebrew/bin/zsh"] {
            assert!(shell_is_zsh(Some(OsStr::new(yes))), "{yes}");
        }
        for no in [
            "/bin/bash",
            "/bin/sh",
            "/usr/bin/fish",
            "zsh-completions",
            "",
        ] {
            assert!(!shell_is_zsh(Some(OsStr::new(no))), "{no}");
        }
        assert!(!shell_is_zsh(None));
    }

    #[test]
    fn the_zshrc_installs_the_marks_and_sources_the_user_config() {
        let rc = zshrc();
        // OSC 133 A (prompt) and C (command output) marks, via precmd/preexec.
        assert!(rc.contains("133;A"), "prompt-start mark");
        assert!(rc.contains("133;C"), "command-start mark");
        assert!(rc.contains("add-zsh-hook precmd  _bnkterm_precmd"));
        assert!(rc.contains("add-zsh-hook preexec _bnkterm_preexec"));
        // It sources the user's real .zshrc, and hands ZDOTDIR back afterwards.
        assert!(rc.contains("source \"${ZDOTDIR}/.zshrc\""));
        assert!(rc.contains("ZDOTDIR=\"${BNKTERM_INT_ZDOTDIR}\""));
    }

    #[test]
    fn the_zshenv_restores_the_users_zdotdir_before_sourcing_theirs() {
        // The bootstrap must consult the sentinel bnkterm sets, source the user's .zshenv,
        // and hand control back — or the user's environment silently vanishes.
        assert!(ZSHENV.contains("BNKTERM_ORIG_ZDOTDIR"));
        assert!(ZSHENV.contains("source \"${ZDOTDIR:-$HOME}/.zshenv\""));
        assert!(ZSHENV.contains("export BNKTERM_USER_ZDOTDIR="));
        assert!(ZSHENV.ends_with("ZDOTDIR=\"${BNKTERM_INT_ZDOTDIR}\"\n"));
    }

    #[test]
    #[ignore = "spawns a real zsh under a PTY; run manually: cargo test --lib \
                injected_integration_marks_a_real_zsh_prompt -- --ignored --test-threads=1"]
    fn injected_integration_marks_a_real_zsh_prompt() {
        use crate::pty::{Pty, ReadOutcome};
        use std::time::{Duration, Instant};

        let shell = std::env::var("SHELL").unwrap_or_default();
        let zsh = if shell_is_zsh(Some(OsStr::new(&shell))) {
            shell
        } else if std::path::Path::new("/usr/bin/zsh").exists() {
            "/usr/bin/zsh".to_string()
        } else {
            eprintln!("no zsh available; skipping");
            return;
        };

        // Lay down the shims and point the child at them, exactly as `install` does. The
        // child inherits this environment across the fork.
        let dir = create_integration_dir().expect("create dir");
        write_zsh_files(&dir).expect("write shims");
        match std::env::var_os("ZDOTDIR") {
            Some(orig) => std::env::set_var("BNKTERM_ORIG_ZDOTDIR", orig),
            None => std::env::remove_var("BNKTERM_ORIG_ZDOTDIR"),
        }
        std::env::set_var("BNKTERM_INT_ZDOTDIR", &dir);
        std::env::set_var("ZDOTDIR", &dir);

        const MARK: &[u8] = b"\x1b]133;A"; // OSC 133 prompt-start
        let has_mark = |v: &[u8]| v.windows(MARK.len()).any(|w| w == MARK);

        let pty = Pty::spawn_command(80, 24, &[&zsh, "-i"]).expect("spawn zsh");
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && !has_mark(&out) {
            match pty.read(&mut buf) {
                Ok(ReadOutcome::Data(n)) => out.extend_from_slice(&buf[..n]),
                Ok(ReadOutcome::WouldBlock) => std::thread::sleep(Duration::from_millis(20)),
                Ok(ReadOutcome::Eof) | Err(_) => break,
            }
        }
        std::fs::remove_dir_all(&dir).ok();

        assert!(
            has_mark(&out),
            "zsh did not emit the injected OSC 133 prompt mark; got {:?}",
            String::from_utf8_lossy(&out)
        );
    }

    #[test]
    fn writes_the_four_startup_shims() {
        let dir = create_integration_dir().expect("create dir");
        write_zsh_files(&dir).expect("write shims");
        for f in [".zshenv", ".zprofile", ".zshrc", ".zlogin"] {
            assert!(dir.join(f).is_file(), "missing {f}");
        }
        // Every stage keeps the hand-off invariant: it points ZDOTDIR back at our dir.
        for f in [".zshenv", ".zprofile", ".zshrc", ".zlogin"] {
            let body = std::fs::read_to_string(dir.join(f)).unwrap();
            assert!(
                body.contains("BNKTERM_INT_ZDOTDIR"),
                "{f} drops the hand-off"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_integration_dir_is_private_and_unpredictable() {
        use std::os::unix::fs::PermissionsExt;
        let a = create_integration_dir().expect("dir a");
        let b = create_integration_dir().expect("dir b");
        // Two directories never collide, so the name cannot be guessed from the pid alone.
        assert_ne!(a, b);
        let mode = std::fs::metadata(&a).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "the directory must be private to the user");
        std::fs::remove_dir_all(&a).ok();
        std::fs::remove_dir_all(&b).ok();
    }

    #[test]
    fn write_new_refuses_to_clobber_a_pre_placed_path() {
        // The clobbering guard: a file already at the target makes the write fail rather than
        // truncate it (and, being O_EXCL, a symlink there is not followed either).
        let dir = create_integration_dir().expect("create dir");
        let target = dir.join(".zshenv");
        std::fs::write(&target, b"pre-existing").expect("plant a file");
        assert!(
            write_new(target.clone(), b"overwrite").is_err(),
            "an existing path must not be clobbered"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"pre-existing");
        std::fs::remove_dir_all(&dir).ok();
    }
}
