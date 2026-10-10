//! Opening a clicked link in the user's default browser.
//!
//! The app resolves a click to a destination ([`crate::platform::link`] finds the
//! ones the text carries no markup for) and hands it here. We shell out to
//! `xdg-open`, the freedesktop way to reach whatever the user set as their default
//! handler, rather than hardcoding a browser. Which `xdg-open` follows the shell: the
//! host's when a Flatpak's shell runs there (see [`Opener`]).

use std::process::{Command, Stdio};

use crate::platform::error::{Error, Result};

/// The schemes we are willing to open, and so also exactly the schemes
/// [`crate::platform::link`] detects: a link the app decorates and then refuses to
/// follow is a worse bug than one it never decorated. Kept deliberately small: the
/// web and mail schemes text links to, plus `file:` for local references.
///
/// **This is a scheme allowlist, not a guarantee that nothing executable is reachable.**
/// The URL never reaches a shell — `javascript:` and `data:` are refused outright, and a
/// leading dash cannot become an `xdg-open` option — but `xdg-open` dispatches by
/// *handler*, so `file:///path/to/x.desktop` is launched by `gio open` rather than
/// displayed. Genuinely low: an attacker must already have written that file, and the
/// click is a deliberate Ctrl+click on a link whose target the user can read. Low is not
/// the same as impossible, and the comment here used to claim impossible.
pub const OPENABLE_SCHEMES: [&str; 4] = ["https://", "http://", "file://", "mailto:"];

/// Which `xdg-open` a link goes to: the one beside the shell that printed it, so a
/// `file:` path names the file the shell meant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opener {
    /// This process's own.
    Local,
    /// The host's, through `flatpak-spawn --host`. The runtime's own `xdg-open` is a portal
    /// client that drops `XDG_ACTIVATION_TOKEN` and resolves a path inside the sandbox.
    FlatpakHost,
}

/// Open `url` in the default browser via `opener`, detached from the app, handing the
/// handler `activation_token` when there is one so it can take focus.
///
/// `xdg-open` is a thin shell wrapper, so the URL is vetted first: a leading dash
/// (which it could read as an option) and any scheme not in [`OPENABLE_SCHEMES`]
/// are refused, so a clicked link can only ever name a web/file/mail target and never
/// a `javascript:`/`data:`-style payload. What the *handler* then does with a `file:`
/// URL is the desktop's business, not ours — see the note on [`OPENABLE_SCHEMES`].
/// `xdg-open` may exit at once or wait on the handler it starts (without a desktop it
/// recognises, it runs the handler in the foreground), so it is reaped on a detached
/// thread: it never lingers as a zombie and the launch never blocks the event loop.
pub fn open_url(url: &str, activation_token: Option<&str>, opener: Opener) -> Result<()> {
    if !can_open(url) {
        return Err(Error::msg(format!(
            "refusing to open URL with no allowed scheme: {url}"
        )));
    }
    let mut child = xdg_open(url, activation_token, opener).spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// The `xdg-open` invocation for a vetted `url`. The token rides in the opener's
/// environment, which `xdg-open` passes on to the handler it starts. On the host it
/// travels as `--env`, since the host command inherits nothing from this process, and
/// without `--watch-bus`, so what the link starts outlives the window.
fn xdg_open(url: &str, activation_token: Option<&str>, opener: Opener) -> Command {
    let mut command = match opener {
        Opener::Local => {
            let mut command = Command::new("xdg-open");
            if let Some(token) = activation_token {
                command.env("XDG_ACTIVATION_TOKEN", token);
            }
            command
        }
        Opener::FlatpakHost => {
            let mut command = Command::new("flatpak-spawn");
            command.arg("--host");
            if let Some(token) = activation_token {
                command.arg(format!("--env=XDG_ACTIVATION_TOKEN={token}"));
            }
            command.args(["--", "xdg-open"]);
            command
        }
    };
    command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

/// Whether [`open_url`] would launch `url`: it carries an [`OPENABLE_SCHEMES`]
/// scheme and does not start with a dash `xdg-open` could read as an option. The app
/// checks this before offering a link to the pointer, so a target we would refuse
/// never presents itself as clickable in the first place.
pub fn can_open(url: &str) -> bool {
    !url.starts_with('-')
        && OPENABLE_SCHEMES
            .iter()
            .any(|scheme| url.starts_with(scheme))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use crate::platform::browser::{can_open, open_url, xdg_open, Opener};

    #[test]
    fn allowed_schemes_pass() {
        assert!(can_open("https://example.com"));
        assert!(can_open("http://example.com"));
        assert!(can_open("file:///etc/hosts"));
        assert!(can_open("mailto:a@b.com"));
    }

    #[test]
    fn disallowed_schemes_rejected() {
        assert!(!can_open("javascript:alert(1)"));
        assert!(!can_open("data:text/html,x"));
        assert!(!can_open("ftp://example.com"));
        assert!(!can_open("example.com"));
        assert!(!can_open("./relative/page.md"));
    }

    #[test]
    fn open_url_refuses_unvetted_targets_without_spawning() {
        // A disallowed scheme or a leading dash never reaches xdg-open.
        assert!(open_url("javascript:alert(1)", None, Opener::Local).is_err());
        assert!(open_url("-flag", None, Opener::FlatpakHost).is_err());
        assert!(open_url("page.md", Some("token"), Opener::Local).is_err());
    }

    #[test]
    fn a_link_hands_its_activation_token_to_the_handler() {
        let with = xdg_open("https://example.com", Some("token-1"), Opener::Local);
        assert_eq!(with.get_program(), "xdg-open");
        assert_eq!(with.get_args().collect::<Vec<_>>(), ["https://example.com"]);
        let envs: Vec<_> = with.get_envs().collect();
        assert_eq!(
            envs,
            [(
                OsStr::new("XDG_ACTIVATION_TOKEN"),
                Some(OsStr::new("token-1"))
            )]
        );
        let without = xdg_open("https://example.com", None, Opener::Local);
        assert_eq!(without.get_envs().count(), 0);
    }

    #[test]
    fn a_link_from_a_host_shell_opens_on_the_host_with_its_token() {
        let with = xdg_open(
            "file:///home/u/notes.md",
            Some("token-1"),
            Opener::FlatpakHost,
        );
        assert_eq!(with.get_program(), "flatpak-spawn");
        assert_eq!(
            with.get_args().collect::<Vec<_>>(),
            [
                "--host",
                "--env=XDG_ACTIVATION_TOKEN=token-1",
                "--",
                "xdg-open",
                "file:///home/u/notes.md",
            ]
        );
        // The host command inherits nothing from here, so a local variable would be lost.
        assert_eq!(with.get_envs().count(), 0);

        let without = xdg_open("https://example.com", None, Opener::FlatpakHost);
        assert_eq!(
            without.get_args().collect::<Vec<_>>(),
            ["--host", "--", "xdg-open", "https://example.com"]
        );
    }
}
