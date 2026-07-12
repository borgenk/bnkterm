//! Opening a clicked link in the user's default browser.
//!
//! The app resolves a click to a destination ([`crate::platform::link`] finds the
//! ones the text carries no markup for) and hands it here. We shell out to
//! `xdg-open`, the freedesktop way to reach whatever the user set as their default
//! handler, rather than hardcoding a browser.

use std::process::{Command, Stdio};

use crate::platform::error::{Error, Result};

/// The schemes we are willing to open, and so also exactly the schemes
/// [`crate::platform::link`] detects: a link the app decorates and then refuses to
/// follow is a worse bug than one it never decorated. Kept deliberately small: the
/// web and mail schemes text links to, plus `file:` for local references. Nothing
/// executable (`javascript:`, `data:`) is reachable.
pub const OPENABLE_SCHEMES: [&str; 4] = ["https://", "http://", "file://", "mailto:"];

/// Open `url` in the default browser via `xdg-open`, detached from the app.
///
/// `xdg-open` is a thin shell wrapper, so the URL is vetted first: a leading dash
/// (which it could read as an option) and any scheme not in [`has_allowed_scheme`]
/// are refused, so a clicked link can only ever open a real web/file/mail target,
/// never run a `javascript:`/`data:`-style payload. `xdg-open` launches the handler
/// and exits almost immediately; it is reaped on a short-lived detached thread so
/// it never lingers as a zombie and the launch never blocks the event loop.
pub fn open_url(url: &str) -> Result<()> {
    if !can_open(url) {
        return Err(Error::msg(format!(
            "refusing to open URL with no allowed scheme: {url}"
        )));
    }
    let mut child = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// Whether [`open_url`] would launch `url`: it carries an [`OPENABLE_SCHEMES`]
/// scheme and does not start with a dash `xdg-open` could read as an option. The app
/// checks this before offering a link to the pointer, so a target we would refuse
/// never presents itself as clickable in the first place.
pub fn can_open(url: &str) -> bool {
    !url.starts_with('-') && has_allowed_scheme(url)
}

/// Whether `url` carries a scheme we are willing to open.
fn has_allowed_scheme(url: &str) -> bool {
    OPENABLE_SCHEMES
        .iter()
        .any(|scheme| url.starts_with(scheme))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_schemes_pass() {
        assert!(has_allowed_scheme("https://example.com"));
        assert!(has_allowed_scheme("http://example.com"));
        assert!(has_allowed_scheme("file:///etc/hosts"));
        assert!(has_allowed_scheme("mailto:a@b.com"));
    }

    #[test]
    fn disallowed_schemes_rejected() {
        assert!(!has_allowed_scheme("javascript:alert(1)"));
        assert!(!has_allowed_scheme("data:text/html,x"));
        assert!(!has_allowed_scheme("ftp://example.com"));
        assert!(!has_allowed_scheme("example.com"));
        assert!(!has_allowed_scheme("./relative/page.md"));
    }

    #[test]
    fn open_url_refuses_unvetted_targets_without_spawning() {
        // A disallowed scheme or a leading dash never reaches xdg-open.
        assert!(open_url("javascript:alert(1)").is_err());
        assert!(open_url("-flag").is_err());
        assert!(open_url("page.md").is_err());
    }
}
