//! Running inside a Flatpak sandbox. The sandbox's shell, `$PATH` and environment belong to
//! the runtime, so the shell a terminal opens has to start on the host, through
//! `flatpak-spawn --host`, with the host session's environment plus the variables the
//! terminal adds.
//!
//! ```text
//!   bnkterm (sandbox) ── PTY master
//!        │ fork, setsid, slave on stdio, left unclaimed
//!        ▼
//!   flatpak-spawn --host --env=… -- $SHELL      (sandbox)
//!        │ the session helper starts the command over D-Bus
//!        ▼
//!   $SHELL (host) ── claims the same PTY as its controlling terminal
//! ```
//!
//! The PTY lives in the sandbox's own `/dev/pts`, so the host cannot name it: `tty` fails
//! there, while job control, `Ctrl+C` and resizes all work.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Error, Result};
use crate::pty::EnvVar;

/// Whether this process runs inside a Flatpak sandbox, which always carries `/.flatpak-info`.
pub fn sandboxed() -> bool {
    Path::new("/.flatpak-info").exists()
}

/// The environment a command started on the host begins with.
pub struct Host {
    env: Vec<(OsString, OsString)>,
}

impl Host {
    /// Ask the host for its session environment: one round trip, at startup.
    pub fn query() -> Result<Self> {
        let output = Command::new("flatpak-spawn")
            .args(["--host", "env", "-0"])
            .output()
            .map_err(|e| Error::msg(format!("cannot run flatpak-spawn: {e}")))?;
        if !output.status.success() {
            return Err(Error::msg(format!(
                "flatpak-spawn --host failed: {}",
                output.status
            )));
        }
        Ok(Self {
            env: parse_env0(&output.stdout),
        })
    }

    /// The host's value for `name`.
    pub fn var(&self, name: &str) -> Option<OsString> {
        self.env
            .iter()
            .find(|(key, _)| key.as_os_str() == OsStr::new(name))
            .map(|(_, value)| value.clone())
    }
}

/// `env -0` output as pairs, split at each entry's first `=`. An entry with no `=` or an
/// empty name is skipped.
fn parse_env0(bytes: &[u8]) -> Vec<(OsString, OsString)> {
    bytes
        .split(|&b| b == 0)
        .filter_map(|entry| {
            let eq = entry.iter().position(|&b| b == b'=')?;
            let (key, rest) = entry.split_at(eq);
            let value = rest.get(1..).unwrap_or_default();
            (!key.is_empty()).then(|| {
                (
                    OsStr::from_bytes(key).to_os_string(),
                    OsStr::from_bytes(value).to_os_string(),
                )
            })
        })
        .collect()
}

/// Where this sandbox can write files a host process reads at the same path:
/// `$XDG_RUNTIME_DIR/app/$FLATPAK_ID`, which Flatpak shares with the host.
pub fn shared_dir() -> Option<PathBuf> {
    let runtime = PathBuf::from(std::env::var_os("XDG_RUNTIME_DIR")?);
    let id = std::env::var_os("FLATPAK_ID")?;
    runtime.is_absolute().then(|| runtime.join("app").join(id))
}

/// The argv that starts `command` on the host with `env` applied over the host session's
/// environment. `--watch-bus` takes the host process down with its `flatpak-spawn`, and
/// `--` keeps the command's own flags from being read as `flatpak-spawn`'s.
pub fn host_argv(command: &[String], env: &[EnvVar]) -> Vec<String> {
    let mut argv = vec![
        "flatpak-spawn".to_string(),
        "--host".to_string(),
        "--watch-bus".to_string(),
    ];
    argv.extend(env.iter().map(|var| match &var.value {
        Some(value) => format!("--env={}={value}", var.name),
        None => format!("--unset-env={}", var.name),
    }));
    argv.push("--".to_string());
    argv.extend(command.iter().cloned());
    argv
}

#[cfg(test)]
mod tests {
    use crate::flatpak::{host_argv, parse_env0, Host};
    use crate::pty::EnvVar;
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStrExt;

    #[test]
    fn env0_output_splits_into_pairs_at_the_first_equals() {
        let pairs = parse_env0(b"SHELL=/usr/bin/zsh\0EMPTY=\0EQ=a=b\0\0NOEQUALS\0=nameless\0");
        let expected = [("SHELL", "/usr/bin/zsh"), ("EMPTY", ""), ("EQ", "a=b")]
            .map(|(key, value)| (OsString::from(key), OsString::from(value)));
        assert_eq!(pairs, expected.to_vec());
    }

    #[test]
    fn a_value_that_is_not_utf8_survives_the_round_trip() {
        let pairs = parse_env0(b"DATA=/opt/\xff\0");
        assert_eq!(pairs[0].1.as_bytes(), b"/opt/\xff");
    }

    #[test]
    fn the_host_answers_only_for_what_it_reported() {
        let host = Host {
            env: parse_env0(b"SHELL=/usr/bin/fish\0HOME=/home/u\0"),
        };
        assert_eq!(host.var("SHELL"), Some(OsString::from("/usr/bin/fish")));
        assert_eq!(host.var("ZDOTDIR"), None);
    }

    #[test]
    fn the_command_follows_the_options_so_its_flags_stay_its_own() {
        let env = [
            EnvVar {
                name: "TERM",
                value: Some("xterm-256color".into()),
            },
            EnvVar {
                name: "BNKTERM_ORIG_ZDOTDIR",
                value: None,
            },
        ];
        let command = [
            "/usr/bin/bash".to_string(),
            "--rcfile".into(),
            "/run/user/1000/app/id/bnkterm.bash".into(),
        ];
        assert_eq!(
            host_argv(&command, &env),
            vec![
                "flatpak-spawn",
                "--host",
                "--watch-bus",
                "--env=TERM=xterm-256color",
                "--unset-env=BNKTERM_ORIG_ZDOTDIR",
                "--",
                "/usr/bin/bash",
                "--rcfile",
                "/run/user/1000/app/id/bnkterm.bash",
            ]
        );
    }
}
