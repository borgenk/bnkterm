//! Auto-injected shell integration for zsh, bash and fish.
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
//! The same hooks answer a second question the terminal cannot answer alone: *whose* title
//! is on the tab. A window title (OSC 0/2) and a reported directory (OSC 7) are set by
//! whatever is running, and nothing takes them back when it exits — `ssh` is where that
//! shows, because the remote shell names the tab after the remote host and the tab keeps
//! that name long after `exit` returned you to your own machine. No sequence says "that was
//! the last command's, not mine"; the shell simply has to say what is true again at every
//! prompt, which is what each shell's `_bnkterm_report` does.
//!
//! ## Three shells, three mechanisms
//!
//! No shell agreed with any other on how a terminal gets code into it, and the differences
//! are not cosmetic: they decide how much of the user's own startup we have to re-create by
//! hand, which is the part that can go wrong.
//!
//! ```text
//!   zsh    ZDOTDIR -> our directory    displaces all four startup files; we source them
//!   bash   --rcfile <our file>         displaces ~/.bashrc alone; we source that
//!   fish   XDG_DATA_DIRS += our dir    displaces nothing: a vendor_conf.d drop-in
//! ```
//!
//! bash is the one that needs an argument rather than an environment variable, which is why
//! [`Session::shell_args`] exists and why the spawn path threads it: there is no `ZDOTDIR`
//! for bash, and the alternatives (`BASH_ENV`, `ENV` with `--posix`) are either
//! non-interactive-only or destroy the startup order outright.
//!
//! Their hooks differ as much as their loading. zsh has `precmd`/`preexec` built in; fish
//! has the `fish_prompt`/`fish_preexec`/`fish_postexec` events; bash has neither, so
//! `PROMPT_COMMAND` stands in for the first and `PS0` (expanded after a command is read,
//! before it runs) for the second. `PS0` is why this needs no `DEBUG` trap, and so cannot
//! fight a debugger hook the user installed.
//!
//! ## zsh: `ZDOTDIR` redirection
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
//! ## bash: `--rcfile`
//!
//! `bash --rcfile <ours>` is read *instead of* `~/.bashrc`, so ours sources theirs first
//! thing. `/etc/bash.bashrc` is untouched either way: bash reads it before the rcfile, so
//! it is not ours to load, and a distro that sets the tab title there (Arch does) still
//! wins over our own report because it runs later in `PROMPT_COMMAND`.
//!
//! ## fish: a `vendor_conf.d` drop-in
//!
//! fish scans `$XDG_DATA_DIRS/fish/vendor_conf.d/` at startup, which is a hook designed for
//! exactly this, so nothing of the user's is displaced and there is nothing to source back.
//! Our directory is prepended to the list (never replacing it, or fish would lose its own
//! vendor files), and the drop-in takes it back out so a child sees the session as it was.
//!
//! Best-effort throughout: a missing `$SHELL`, a shell none of the three, an opt-out, or
//! any I/O failure installs nothing. [`install`] returns the variables and arguments a shell
//! needs rather than setting them: [`crate::app::run`] applies them to its own environment
//! for a local shell, before any thread starts, and passes them to `flatpak-spawn` for one
//! on the host.

use std::ffi::{OsStr, OsString};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::PathBuf;

use crate::pty::EnvVar;

/// Set to `0` to disable auto-injection, for a user whose exotic startup it disturbs.
const OPT_OUT_VAR: &str = "BNKTERM_SHELL_INTEGRATION";

/// A shell the integration can reach. One variant per *startup protocol*, which is the
/// only axis on which these differ from the terminal's point of view: what the child is
/// told, and whether it is told through the environment or through an argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shell {
    Zsh,
    Bash,
    Fish,
}

/// What a shell has to be started with to find its shims.
struct Shims {
    args: Vec<String>,
    env: Vec<EnvVar>,
}

impl Shell {
    /// Which shell `$SHELL` names, by the final path component so `/usr/bin/zsh`,
    /// `/bin/zsh` and a bare `zsh` all match. `None` for a shell with no integration
    /// (ksh, nushell, a login shell that is not a shell at all), which is a no-op and not
    /// an error.
    fn named(shell: Option<&OsStr>) -> Option<Self> {
        match shell
            .map(std::path::Path::new)
            .and_then(std::path::Path::file_name)
            .and_then(OsStr::to_str)?
        {
            "zsh" => Some(Self::Zsh),
            "bash" => Some(Self::Bash),
            "fish" => Some(Self::Fish),
            _ => None,
        }
    }

    /// Write this shell's shims into the already-created private `dir`, answering with what
    /// the shell must be started with to find them. `env` answers for the environment it
    /// starts in. Bash is the one that has to be told in argv (`--rcfile <path>`); zsh and
    /// fish are reached through variables.
    fn install_into(
        self,
        dir: &std::path::Path,
        env: &dyn Fn(&str) -> Option<OsString>,
    ) -> std::io::Result<Shims> {
        match self {
            Self::Zsh => {
                write_zsh_files(dir)?;
                Ok(Shims {
                    args: Vec::new(),
                    env: zsh_env(dir, env("ZDOTDIR"))?,
                })
            }
            Self::Bash => {
                let rc = dir.join("bnkterm.bash");
                let arg = utf8(&rc)?;
                write_new(rc, BASHRC.as_bytes())?;
                Ok(Shims {
                    args: vec!["--rcfile".to_string(), arg],
                    env: Vec::new(),
                })
            }
            Self::Fish => {
                write_fish_files(dir)?;
                Ok(Shims {
                    args: Vec::new(),
                    env: fish_env(dir, env("XDG_DATA_DIRS"))?,
                })
            }
        }
    }
}

/// Holds the generated integration directory alive for the process and removes it on drop,
/// and carries what the child shell needs to find it ([`Self::shell_args`], [`Self::env`]).
///
/// A shell sources these files once at startup and never reads them again, so the directory
/// only has to outlive the session, which this does by living in [`crate::app::run`]'s
/// stack frame.
pub struct Session {
    dir: PathBuf,
    args: Vec<String>,
    env: Vec<EnvVar>,
}

impl Session {
    /// Extra arguments every shell this process spawns must be given for the integration to
    /// load. Empty for zsh and fish, which are reached through the environment; `--rcfile
    /// <path>` for bash, which has no environment variable that works for an interactive
    /// shell.
    ///
    /// These are a property of the *session*, not of a tab, so every tab passes the same
    /// ones and a tab opened an hour in loads the same shims as the first.
    pub fn shell_args(&self) -> &[String] {
        &self.args
    }

    /// Variables every shell this session spawns must start with, or without, for the
    /// integration to load.
    pub fn env(&self) -> &[EnvVar] {
        &self.env
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Generate the integration for `shell` and return what every shell must be started with to
/// load it: a [`Session`] the caller keeps alive (dropping it removes the directory), or
/// `None` when nothing was installed (a shell we do not speak, opted out, or an I/O
/// failure). Never errors: integration is a nicety, never a reason the terminal fails to
/// open.
///
/// `env` answers for the environment the shell will start in, which is this process's own
/// for a local shell and the host session's under Flatpak. The files go under the first of
/// `bases` a directory can be made in. Nothing here touches the process environment.
pub fn install(
    shell: &str,
    env: &dyn Fn(&str) -> Option<OsString>,
    bases: &[PathBuf],
) -> Option<Session> {
    if std::env::var_os(OPT_OUT_VAR).as_deref() == Some(OsStr::new("0")) {
        return None;
    }
    generate(Shell::named(Some(OsStr::new(shell)))?, env, bases)
}

/// [`install`] past the opt-out, which reads this process's own environment.
fn generate(
    shell: Shell,
    env: &dyn Fn(&str) -> Option<OsString>,
    bases: &[PathBuf],
) -> Option<Session> {
    let dir = create_integration_dir(bases).ok()?;
    match shell.install_into(&dir, env) {
        Ok(shims) => Some(Session {
            dir,
            args: shims.args,
            env: shims.env,
        }),
        Err(_) => {
            let _ = std::fs::remove_dir_all(&dir);
            None
        }
    }
}

/// Where a local shell's integration directory goes: `$XDG_RUNTIME_DIR` when the session has
/// one (already a per-user 0700 directory), else the temp dir.
pub fn local_bases() -> Vec<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .into_iter()
        .chain(std::iter::once(std::env::temp_dir()))
        .collect()
}

/// Create a fresh, private directory for the generated startup files under the first of
/// `bases` that takes one, and return its path. The name carries 128 bits from the kernel
/// CSPRNG so it is unpredictable, and it is made with `create_dir` (not `_all`) at mode
/// 0700, so a name an attacker raced to pre-create is rejected rather than silently reused.
/// On a shared `/tmp` that is what stops another user from planting the files bnkterm's zsh
/// then sources.
fn create_integration_dir(bases: &[PathBuf]) -> std::io::Result<PathBuf> {
    let name = format!("bnkterm-shell-{}-{}", std::process::id(), random_token()?);
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

/// Point the shell's `ZDOTDIR` at `dir`, handing it the user's own (`user`, from the
/// environment it starts in) in a sentinel so the first stage can restore it before sourcing
/// their startup.
fn zsh_env(dir: &std::path::Path, user: Option<OsString>) -> std::io::Result<Vec<EnvVar>> {
    let dir = utf8(dir)?;
    let user = user
        .map(|value| {
            value
                .into_string()
                .map_err(|_| std::io::Error::other("non-UTF-8 ZDOTDIR"))
        })
        .transpose()?;
    Ok(vec![
        // Removed when the user had none, meaning "fall back to $HOME": a stale sentinel
        // inherited from a parent bnkterm must not masquerade as the user's.
        EnvVar {
            name: "BNKTERM_ORIG_ZDOTDIR",
            value: user,
        },
        // Our directory, which the stages return to after each hand-off.
        EnvVar {
            name: "BNKTERM_INT_ZDOTDIR",
            value: Some(dir.clone()),
        },
        EnvVar {
            name: "ZDOTDIR",
            value: Some(dir),
        },
    ])
}

/// `path` as a `str`, since it ends up in a variable or an argv entry. A non-UTF-8 path is
/// refused rather than lossily mangled: a shell pointed at a half-right path loads nothing,
/// not even the user's own startup.
fn utf8(path: &std::path::Path) -> std::io::Result<String> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| std::io::Error::other("non-UTF-8 integration path"))
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
/// hooks, the per-prompt report of the shell's own title and directory, and a one-shot
/// `precmd` that restores the user's ZDOTDIR once startup is done (it runs after `.zlogin`,
/// so a login shell's last stage is still found first).
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
  # --- The prompt reports its own title (OSC 0, cleared) and directory (OSC 7), because
  # --- what the last command left behind belongs to that command and not to this prompt:
  # --- ssh names the tab after the remote host, and exiting never took the name back.
  # --- Prepended, so a user's own title-setting precmd runs after this one and still wins;
  # --- `return $ret` keeps the exit status intact for _bnkterm_precmd, which runs later
  # --- and reads it from $? for the D mark.
  _bnkterm_report() {{
    local ret=$?
    # Percent-encode $PWD bytewise: a directory name may hold any byte but `/` and NUL,
    # this sequence's own terminator included, and `nomultibyte` is what makes the encoding
    # the UTF-8 bytes the terminal decodes rather than the code points it does not.
    setopt localoptions extendedglob nomultibyte
    print -rn -- $'\\e]0;\\a\\e]7;file://'\"${{HOST}}${{PWD//(#m)[^A-Za-z0-9\\/._~-]/%${{(l:2::0:)$(([##16]#MATCH))}}}}\"$'\\a'
    return $ret
  }}
  (( ${{precmd_functions[(I)_bnkterm_report]}} )) || precmd_functions=(_bnkterm_report $precmd_functions)
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

/// The bash shim, reached by `bash --rcfile <this>`.
///
/// bash has neither of zsh's hooks, so both are stood in for:
///
/// ```text
///   precmd   PROMPT_COMMAND   runs before each prompt; a string, or an array from 5.1
///   preexec  PS0              expanded after a command is read, before it runs (4.4+)
/// ```
///
/// `PS0` is the part worth knowing about. The usual bash `preexec` is a `DEBUG` trap, which
/// fires before *every* simple command (so it needs a latch to find the first one) and is a
/// single global slot a user's debugger hook may already own. `PS0` is neither: it is
/// expanded exactly once per command line, and appending to it cannot take anything away
/// from anyone. On bash before 4.4 it is an unused variable and the `C` mark simply does not
/// appear, which costs nothing the terminal consumes today.
///
/// Written as a raw string so the shell reads what is written here: `\e` is bash's escape,
/// not Rust's.
const BASHRC: &str = r#"# bnkterm shell integration (auto-injected). Source the user's bash startup untouched,
# then add OSC 133 prompt marks. See crate::shell_integration.
#
# bash reads this *instead of* ~/.bashrc, which is what --rcfile means, so the user's file
# is sourced here by hand. /etc/bash.bashrc is not ours to load: bash reads it before this.
if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi

# Interactive shells only (a script has no prompt to mark), and once per shell: a nested
# bnkterm re-runs this file in a *new* bash, where the guard is unset again.
if [[ $- == *i* && -z "${_BNKTERM_INTEGRATED:-}" ]]; then
  _BNKTERM_INTEGRATED=1

  # Percent-encode $1 into _bnkterm_encoded. A directory name may hold any byte but `/` and
  # NUL, this sequence's own terminator included, so the path is encoded rather than trusted.
  # LC_ALL=C makes the indexing bytewise, which is what makes the escapes the UTF-8 bytes the
  # terminal decodes rather than code points it does not. It answers through a variable
  # because $(...) is a fork and this runs at every prompt.
  _bnkterm_encode() {
    local LC_ALL=C str=$1 out= i char
    for (( i = 0; i < ${#str}; i++ )); do
      char=${str:i:1}
      case $char in
        [-A-Za-z0-9._~/]) out+=$char ;;
        *) printf -v char '%%%02X' "'$char"; out+=$char ;;
      esac
    done
    _bnkterm_encoded=$out
  }

  # The prompt's own title (cleared) and directory, because what the last command left
  # behind belongs to that command: ssh names the tab after the remote host and exiting
  # never took the name back. First in PROMPT_COMMAND, so a config that sets its own title
  # runs after this and still wins, and it returns the status it was handed so the D mark
  # below still reports the command's rather than its own.
  _bnkterm_report() {
    local ret=$?
    _bnkterm_encode "${PWD}"
    printf '\e]0;\a\e]7;file://%s%s\a' "${HOSTNAME}" "${_bnkterm_encoded}"
    return $ret
  }

  # D (how the command ended) and A (a prompt starts here), last so the A mark sits closest
  # to the prompt itself.
  _bnkterm_marks() {
    local ret=$?
    printf '\e]133;D;%s\a\e]133;A\a' "$ret"
    return $ret
  }

  # PROMPT_COMMAND is a string, and from bash 5.1 may be an array (Arch's /etc/bash.bashrc
  # makes it one). Splice ours around whatever is there rather than replacing it.
  if [[ "$(declare -p PROMPT_COMMAND 2>/dev/null)" == 'declare -a'* ]]; then
    PROMPT_COMMAND=(_bnkterm_report "${PROMPT_COMMAND[@]}" _bnkterm_marks)
  elif [[ -n "${PROMPT_COMMAND}" ]]; then
    PROMPT_COMMAND=$'_bnkterm_report\n'"${PROMPT_COMMAND}"$'\n_bnkterm_marks'
  else
    PROMPT_COMMAND=$'_bnkterm_report\n_bnkterm_marks'
  fi

  # C: the command's output starts here. Appended, so a PS0 the user set still prints.
  PS0=${PS0}'\e]133;C\a'
fi
"#;

/// Write the fish drop-in under `dir`, in the `fish/vendor_conf.d/` layout fish looks for
/// inside each `$XDG_DATA_DIRS` entry. The subdirectories are ours alone (0700), created
/// inside a directory that is already unpredictable and private, so the guarantee
/// [`create_integration_dir`] establishes still holds for what lands under them.
fn write_fish_files(dir: &std::path::Path) -> std::io::Result<()> {
    let conf_d = dir.join("fish").join("vendor_conf.d");
    std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(&conf_d)?;
    write_new(conf_d.join("bnkterm.fish"), FISH_CONF.as_bytes())
}

/// Prepend the integration directory to an existing XDG data path.
///
/// The list is *prepended to*, never replaced: it is where fish finds its own vendor
/// completions and functions, and a session that set none still means the two spec defaults
/// rather than nothing at all — an empty `XDG_DATA_DIRS` would hide them.
fn fish_data_dirs(
    dir: &std::path::Path,
    existing: Option<std::ffi::OsString>,
) -> std::ffi::OsString {
    /// What the XDG base directory specification says an unset `XDG_DATA_DIRS` means.
    const XDG_DATA_DIRS_DEFAULT: &str = "/usr/local/share:/usr/share";
    let mut value = dir.as_os_str().to_owned();
    value.push(":");
    match existing.filter(|dirs| !dirs.is_empty()) {
        Some(existing) => value.push(existing),
        None => value.push(XDG_DATA_DIRS_DEFAULT),
    }
    value
}

/// Prepend `dir` to the shell's `XDG_DATA_DIRS` (`existing`, from the environment it starts
/// in) so fish finds the drop-in, leaving a sentinel the drop-in uses to take it back out.
fn fish_env(dir: &std::path::Path, existing: Option<OsString>) -> std::io::Result<Vec<EnvVar>> {
    let dirs = fish_data_dirs(dir, existing)
        .into_string()
        .map_err(|_| std::io::Error::other("non-UTF-8 XDG_DATA_DIRS"))?;
    Ok(vec![
        EnvVar {
            name: "XDG_DATA_DIRS",
            value: Some(dirs),
        },
        EnvVar {
            name: "BNKTERM_INT_DATA_DIR",
            value: Some(utf8(dir)?),
        },
    ])
}

/// The fish drop-in: `$XDG_DATA_DIRS/fish/vendor_conf.d/bnkterm.fish`.
///
/// The easiest of the three. fish designed a hook for exactly this, so nothing of the
/// user's is displaced and there is nothing to source back, and its `fish_prompt` /
/// `fish_preexec` / `fish_postexec` events map onto the marks one for one. `postexec`
/// carries the status, which is why `D` is emitted there rather than at the prompt.
const FISH_CONF: &str = r#"# bnkterm shell integration (auto-injected). See crate::shell_integration.

# Take our directory back out of the inherited list, so a child of this shell sees the
# session as it was. The sentinel names the one entry to drop, and goes with it.
if set -q BNKTERM_INT_DATA_DIR
    if set -q XDG_DATA_DIRS
        set -l kept
        for dir in (string split : -- $XDG_DATA_DIRS)
            if test "$dir" != "$BNKTERM_INT_DATA_DIR"
                set -a kept $dir
            end
        end
        set -gx XDG_DATA_DIRS (string join : -- $kept)
    end
    set -e BNKTERM_INT_DATA_DIR
end

# conf.d files are read by every fish, script or not; only a shell with a prompt has
# anything to mark.
if status is-interactive
    # The prompt's own title (cleared) and directory, because what the last command left
    # behind belongs to that command: ssh names the tab after the remote host and exiting
    # never took the name back. Then A, marking where this prompt starts.
    function _bnkterm_report --on-event fish_prompt -d "bnkterm: the prompt's own title and directory"
        # Encoded per component and rejoined, rather than in one pass over the whole path:
        # the terminal splits the host from the path at the first separator, so those have
        # to survive whether or not `--style=url` treats them as reserved. The leading one
        # is written here because splitting an absolute path yields an empty first field.
        set -l parts (string split / -- $PWD)
        set -e parts[1]
        printf '\e]0;\a\e]7;file://%s/%s\a\e]133;A\a' $hostname (string join / -- (string escape --style=url -- $parts))
    end

    # C: the command's output starts here.
    function _bnkterm_preexec --on-event fish_preexec -d "bnkterm: where a command's output starts"
        printf '\e]133;C\a'
    end

    # D: how it ended. $status is read first, before anything here can overwrite it.
    function _bnkterm_postexec --on-event fish_postexec -d "bnkterm: how a command ended"
        printf '\e]133;D;%s\a' $status
    end
end
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_each_shell_by_the_final_path_component() {
        // Whatever prefix a distro installs a shell under, the name is the last component.
        for (path, shell) in [
            ("/usr/bin/zsh", Shell::Zsh),
            ("/bin/zsh", Shell::Zsh),
            ("zsh", Shell::Zsh),
            ("/opt/homebrew/bin/zsh", Shell::Zsh),
            ("/bin/bash", Shell::Bash),
            ("/usr/local/bin/bash", Shell::Bash),
            ("/usr/bin/fish", Shell::Fish),
            ("fish", Shell::Fish),
        ] {
            assert_eq!(Shell::named(Some(OsStr::new(path))), Some(shell), "{path}");
        }
        // A shell with no integration is not an error, it is nothing at all. `/bin/sh` is
        // in this list on purpose: it is usually dash or a bash in POSIX mode, and neither
        // reads what the bash shim would be handed.
        for no in [
            "/bin/sh",
            "/usr/bin/ksh",
            "/usr/bin/nu",
            "zsh-completions",
            "",
        ] {
            assert_eq!(Shell::named(Some(OsStr::new(no))), None, "{no}");
        }
        assert_eq!(Shell::named(None), None);
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
    fn the_prompt_reports_its_own_title_and_directory() {
        let rc = zshrc();
        // The title a command left behind is cleared, and the directory re-reported, so a
        // tab named by a remote shell goes back to naming this machine after `exit`.
        assert!(
            rc.contains(r"$'\e]0;\a\e]7;file://'"),
            "title clear + cwd report"
        );
        // Prepended, not appended: a user's own precmd that sets a title runs afterwards
        // and keeps it, and `_bnkterm_precmd` still runs last (its `A` mark belongs next
        // to the prompt). Guarded on the hook not already being there, so sourcing this
        // twice in one shell leaves one hook, as `add-zsh-hook` does for the marks.
        assert!(rc.contains("precmd_functions=(_bnkterm_report $precmd_functions)"));
        assert!(rc.contains("${precmd_functions[(I)_bnkterm_report]}"));
        // The `D` mark reports the command's exit status, and it is read from `$?` in a
        // later hook — so this one has to hand the status on rather than report its own.
        assert!(rc.contains("local ret=$?"), "captures the status");
        assert!(rc.contains("return $ret"), "and hands it on");
        // The directory is percent-encoded bytewise; without `nomultibyte` the encoding
        // would be of code points, which is not what the terminal decodes.
        assert!(rc.contains("setopt localoptions extendedglob nomultibyte"));
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
    fn the_bash_shim_sources_the_user_and_hooks_both_ends_of_a_command() {
        // --rcfile displaces ~/.bashrc, so the shim owes the user their own file back.
        assert!(BASHRC.contains(r#"source "${HOME}/.bashrc""#));
        // The marks, and the prompt's own report of what a command may have overwritten.
        assert!(BASHRC.contains(r"\e]133;D;%s\a\e]133;A\a"), "D then A");
        assert!(
            BASHRC.contains(r"PS0=${PS0}'\e]133;C\a'"),
            "C, appended to PS0"
        );
        assert!(
            BASHRC.contains(r"\e]0;\a\e]7;file://%s%s\a"),
            "title clear + cwd"
        );
        // Spliced around whatever is already there, in both forms PROMPT_COMMAND takes:
        // the report first (so a config that sets its own title still wins) and the marks
        // last (so the A mark sits closest to the prompt).
        assert!(BASHRC
            .contains(r#"PROMPT_COMMAND=(_bnkterm_report "${PROMPT_COMMAND[@]}" _bnkterm_marks)"#));
        assert!(BASHRC.contains(
            "PROMPT_COMMAND=$'_bnkterm_report\\n'\"${PROMPT_COMMAND}\"$'\\n_bnkterm_marks'"
        ));
        // The D mark reports the command's status, and reads it from $? in a later hook.
        assert!(BASHRC.contains("return $ret"));
        // Interactive shells only, once per shell.
        assert!(BASHRC.contains(r#"if [[ $- == *i* && -z "${_BNKTERM_INTEGRATED:-}" ]]"#));
    }

    #[test]
    fn the_fish_shim_hooks_the_events_and_gives_the_data_dirs_back() {
        // One event per mark. D rides fish_postexec because that is where the status is.
        assert!(FISH_CONF.contains("--on-event fish_prompt"));
        assert!(FISH_CONF.contains("--on-event fish_preexec"));
        assert!(FISH_CONF.contains("--on-event fish_postexec"));
        assert!(FISH_CONF.contains(r"printf '\e]133;D;%s\a' $status"));
        assert!(FISH_CONF.contains(r"printf '\e]0;\a\e]7;file://%s/%s\a\e]133;A\a'"));
        // The directory is encoded per component, so the separators the terminal splits
        // host from path on survive whatever `--style=url` considers reserved.
        assert!(FISH_CONF.contains("string join / -- (string escape --style=url -- $parts)"));
        // Our entry comes back out of the inherited list, and the sentinel with it.
        assert!(FISH_CONF.contains("set -gx XDG_DATA_DIRS (string join : -- $kept)"));
        assert!(FISH_CONF.contains("set -e BNKTERM_INT_DATA_DIR"));
        // conf.d is read by every fish; only one with a prompt has anything to mark.
        assert!(FISH_CONF.contains("if status is-interactive"));
    }

    #[test]
    fn each_shell_lays_down_what_it_alone_needs() {
        // The files and the argv are the whole of what differs between the three, so this
        // pins each mechanism against the others rather than each in isolation.
        let dir = create_integration_dir(&local_bases()).expect("create dir");

        let none = |_: &str| None::<OsString>;
        let zsh = Shell::Zsh.install_into(&dir, &none).expect("zsh");
        assert!(zsh.args.is_empty(), "zsh is reached through ZDOTDIR");
        assert!(zsh.env.contains(&EnvVar {
            name: "ZDOTDIR",
            value: dir.to_str().map(String::from),
        }));
        assert!(dir.join(".zshenv").is_file() && dir.join(".zshrc").is_file());

        let bash = Shell::Bash.install_into(&dir, &none).expect("bash");
        assert!(
            bash.env.is_empty(),
            "bash is told in argv, so needs no variables"
        );
        let rc = dir.join("bnkterm.bash");
        assert!(rc.is_file(), "the rcfile is written");
        assert_eq!(
            bash.args,
            vec!["--rcfile".to_string(), rc.to_string_lossy().into_owned()],
            "bash has no environment variable that works, so it is told in argv"
        );

        let fish = Shell::Fish.install_into(&dir, &none).expect("fish");
        assert!(
            fish.args.is_empty(),
            "fish is reached through XDG_DATA_DIRS"
        );
        assert!(dir.join("fish/vendor_conf.d/bnkterm.fish").is_file());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_fish_data_dirs_keep_what_was_already_there() {
        // Replacing the list rather than prepending to it would hide fish's own vendor
        // files, and an unset one still means the two spec defaults, never nothing.
        let dir = std::path::Path::new("/run/user/1000/bnkterm-abc123");

        assert_eq!(
            fish_data_dirs(dir, Some("/opt/share:/usr/share".into())),
            std::ffi::OsString::from("/run/user/1000/bnkterm-abc123:/opt/share:/usr/share"),
            "ours leads, theirs survives"
        );
        for empty in [None, Some(std::ffi::OsString::new())] {
            assert_eq!(
                fish_data_dirs(dir, empty),
                std::ffi::OsString::from(
                    "/run/user/1000/bnkterm-abc123:/usr/local/share:/usr/share"
                ),
                "an unset or empty list means the spec's defaults, not an empty one"
            );
        }
    }

    #[test]
    fn zsh_starts_in_the_shims_with_the_users_zdotdir_set_aside() {
        let dir = std::path::Path::new("/run/user/1000/bnkterm-abc123");
        let var = |name: &'static str, value: Option<&str>| EnvVar {
            name,
            value: value.map(String::from),
        };

        assert_eq!(
            zsh_env(dir, Some("/home/u/.config/zsh".into())).expect("utf-8"),
            vec![
                var("BNKTERM_ORIG_ZDOTDIR", Some("/home/u/.config/zsh")),
                var("BNKTERM_INT_ZDOTDIR", Some("/run/user/1000/bnkterm-abc123")),
                var("ZDOTDIR", Some("/run/user/1000/bnkterm-abc123")),
            ]
        );
        assert_eq!(
            zsh_env(dir, None).expect("utf-8")[0],
            var("BNKTERM_ORIG_ZDOTDIR", None),
            "a user with no ZDOTDIR gets the sentinel removed, never a stale one"
        );
    }

    #[test]
    fn the_shell_is_set_up_from_the_environment_it_starts_in() {
        // Under Flatpak that is the host's, which this process cannot see in its own, so
        // every answer has to come from the lookup it is handed.
        let host = |name: &str| match name {
            "XDG_DATA_DIRS" => Some(OsString::from("/host/share")),
            _ => None,
        };
        let session = generate(Shell::Fish, &host, &[std::env::temp_dir()]).expect("installed");
        let dir = session.dir.to_str().expect("utf-8").to_string();
        assert_eq!(
            session.env(),
            [
                EnvVar {
                    name: "XDG_DATA_DIRS",
                    value: Some(format!("{dir}:/host/share")),
                },
                EnvVar {
                    name: "BNKTERM_INT_DATA_DIR",
                    value: Some(dir.clone()),
                },
            ]
        );
    }

    /// Whether `haystack` holds `needle` anywhere.
    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    /// The first interpreter of `names` that exists, for a live-shell test to drive.
    fn find_shell(names: &[&str]) -> Option<String> {
        let shell = std::env::var("SHELL").unwrap_or_default();
        if names.contains(&std::path::Path::new(&shell).file_name()?.to_str()?) {
            return Some(shell);
        }
        names
            .iter()
            .flat_map(|name| [format!("/usr/bin/{name}"), format!("/bin/{name}")])
            .find(|path| std::path::Path::new(path).exists())
    }

    /// Spawn `argv` on a PTY and read until it emits the prompt-start mark or three seconds
    /// pass, answering everything it printed.
    ///
    /// The mark is the right thing to wait for: every shim emits its report *before* the
    /// `A` of the same prompt, so seeing `A` means the whole prompt-time contract has
    /// already been exercised and the buffer holds all of it.
    fn drive_until_prompt(argv: &[&str]) -> Vec<u8> {
        use crate::pty::{Pty, ReadOutcome};
        use std::time::{Duration, Instant};

        const MARK: &[u8] = b"\x1b]133;A"; // OSC 133 prompt-start
        let Ok(pty) = Pty::spawn_command(80, 24, argv) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && !contains(&out, MARK) {
            match pty.read(&mut buf) {
                Ok(ReadOutcome::Data(n)) => out.extend_from_slice(&buf[..n]),
                Ok(ReadOutcome::WouldBlock) => std::thread::sleep(Duration::from_millis(20)),
                Ok(ReadOutcome::Eof) | Err(_) => break,
            }
        }
        out
    }

    /// What this process's working directory must look like in an `OSC 7`, percent-encoded
    /// here independently of every shim so the encoding is checked rather than echoed:
    /// everything outside the unreserved set (plus the separator) is an escaped byte, which
    /// is what makes a directory named with a BEL in it report as text rather than as the
    /// end of the sequence. Answers the `(scheme, path)` halves, because the host each
    /// shell reports sits between them.
    fn expected_osc7() -> (String, String) {
        let cwd = std::env::current_dir().expect("cwd");
        let mut path = String::new();
        for byte in cwd.as_os_str().as_encoded_bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'.' | b'_' | b'~' | b'-' => {
                    path.push(char::from(*byte));
                }
                _ => path.push_str(&format!("%{byte:02X}")),
            }
        }
        path.push('\x07');
        ("\x1b]7;file://".to_string(), path)
    }

    /// The prompt-time contract every shell's shim owes, asserted against what a live one
    /// printed: the marks, the cleared title, and this directory reported as an `OSC 7`.
    fn assert_prompt_contract(shell: &str, out: &[u8]) {
        let seen = String::from_utf8_lossy(out);
        assert!(
            contains(out, b"\x1b]133;A"),
            "{shell} did not emit the injected OSC 133 prompt mark; got {seen:?}"
        );
        assert!(
            contains(out, b"\x1b]0;\x07"),
            "{shell} did not retire the last command's title; got {seen:?}"
        );
        let (scheme, path) = expected_osc7();
        assert!(
            contains(out, scheme.as_bytes()) && contains(out, path.as_bytes()),
            "{shell} did not report this directory; got {seen:?}"
        );
    }

    #[test]
    #[ignore = "spawns a real zsh under a PTY and mutates the environment; run manually: \
                cargo test --lib injected_integration_marks_a_real -- --ignored \
                --test-threads=1"]
    fn injected_integration_marks_a_real_zsh_prompt() {
        let Some(zsh) = find_shell(&["zsh"]) else {
            eprintln!("no zsh available; skipping");
            return;
        };

        // Lay down the shims and point the child at them, as a local launch does: the child
        // inherits this environment across the fork.
        let dir = create_integration_dir(&local_bases()).expect("create dir");
        let shims = Shell::Zsh
            .install_into(&dir, &|name| std::env::var_os(name))
            .expect("install");
        shims.env.iter().for_each(EnvVar::apply);

        let out = drive_until_prompt(&[&zsh]);
        std::fs::remove_dir_all(&dir).ok();
        assert_prompt_contract("zsh", &out);
    }

    #[test]
    #[ignore = "spawns a real bash under a PTY and sources the user's ~/.bashrc; run \
                manually: cargo test --lib injected_integration_marks_a_real -- --ignored \
                --test-threads=1"]
    fn injected_integration_marks_a_real_bash_prompt() {
        let Some(bash) = find_shell(&["bash"]) else {
            eprintln!("no bash available; skipping");
            return;
        };

        // bash takes its instructions in argv rather than the environment, so this is the
        // whole of what `install` hands the child.
        let dir = create_integration_dir(&local_bases()).expect("create dir");
        let shims = Shell::Bash
            .install_into(&dir, &|name| std::env::var_os(name))
            .expect("install");
        let argv: Vec<&str> = std::iter::once(bash.as_str())
            .chain(shims.args.iter().map(String::as_str))
            .collect();

        let out = drive_until_prompt(&argv);
        std::fs::remove_dir_all(&dir).ok();
        assert_prompt_contract("bash", &out);
        // PS0 is expanded when a command is read, so the mark for it cannot appear before
        // one is: what this pins is that the shim reached the prompt with PS0 armed.
        assert!(
            contains(&out, b"\x1b]133;D;"),
            "bash did not report how the last command ended; got {:?}",
            String::from_utf8_lossy(&out)
        );
    }

    #[test]
    #[ignore = "spawns a real fish under a PTY and mutates the environment; run manually: \
                cargo test --lib injected_integration_marks_a_real -- --ignored \
                --test-threads=1"]
    fn injected_integration_marks_a_real_fish_prompt() {
        let Some(fish) = find_shell(&["fish"]) else {
            eprintln!("no fish available; skipping");
            return;
        };

        let dir = create_integration_dir(&local_bases()).expect("create dir");
        let shims = Shell::Fish
            .install_into(&dir, &|name| std::env::var_os(name))
            .expect("install");
        shims.env.iter().for_each(EnvVar::apply);

        let out = drive_until_prompt(&[&fish]);
        std::fs::remove_dir_all(&dir).ok();
        assert_prompt_contract("fish", &out);
    }

    #[test]
    fn writes_the_four_startup_shims() {
        let dir = create_integration_dir(&local_bases()).expect("create dir");
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
        let a = create_integration_dir(&local_bases()).expect("dir a");
        let b = create_integration_dir(&local_bases()).expect("dir b");
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
        let dir = create_integration_dir(&local_bases()).expect("create dir");
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
