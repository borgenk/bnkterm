//! Whether a link may open: the rules a link's target must pass before the terminal
//! offers it to the pointer or hands it to an opener.
//!
//! Every rule is a pure function of the target, so a verdict is worked out where a link
//! enters the terminal and kept with it: an OSC 8 anchor is judged once, when the grid
//! interns it. A link either opens or is refused with a [`Refusal`]; there is no third
//! outcome to confirm.

use crate::platform::host::{self, Host, Hostname};
use crate::platform::{browser, file_url, link, mailto};

/// What the rules decided about a link.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Open,
    Refused(Refusal),
}

/// Why a link is refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// The target does not start with a scheme in [`browser::OPENABLE_SCHEMES`].
    Scheme,
    /// The target carries a character that draws nothing, a control character or one
    /// [`link::is_hidden`] names, so the text the user reads is not the target that opens.
    Hidden,
    /// The target's host is one a browser would rewrite before resolving it
    /// ([`host::read`]), so the host the user reads is not the one reached.
    Host,
    /// A `file:` target names another machine. The opener would open the file at the
    /// same path on this one.
    ForeignHost,
    /// A `file:` target's path could be read more than one way
    /// ([`file_url::path_is_unambiguous`]), so the file checked may not be the file
    /// opened.
    FilePath,
    /// A `mailto:` target sets a header field beyond `to`, `cc`, `bcc`, `subject` and
    /// `body` ([`mailto::fields_are_allowed`]), such as `attach=`.
    MailField,
    /// An anchor's text starts like a link but names a host its target does not, so
    /// what the user reads is not where the click goes. Judged at lookup, since the text
    /// arrives after the anchor and can be overwritten.
    LabelHost,
    /// A `file:` target resolves to a regular file with an exec bit, which an opener may
    /// run rather than show. Judged at the click ([`judge_file`]).
    Executable,
    /// A `file:` target resolves to a `.desktop` file, which an opener launches. Judged
    /// at the click.
    Launcher,
    /// A `file:` target could not be checked: it is missing, or not readable. Judged at
    /// the click, and refused because a check that cannot run must not open.
    Unreachable,
}

/// The one rule that needs the filesystem, judged when a `file:` link is clicked: the URL
/// to hand the opener, or why it is refused.
///
/// The path is resolved through symlinks and `..`, so a link is judged by what it reaches
/// (a `notes.txt` pointing at a launcher is a launcher). The resolved path must pass
/// [`file_url::plain_path`] again, since a symlink can bring in any byte its target's name
/// holds, and must not be an executable file or a `.desktop` launcher. The opener is
/// handed a URL encoded from exactly that path, so the path checked is the path opened.
pub fn judge_file(url: &str) -> Result<String, Refusal> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;

    let path = file_url::decoded_path(url).ok_or(Refusal::FilePath)?;
    let canonical = std::fs::canonicalize(path).map_err(|_| Refusal::Unreachable)?;
    let resolved =
        file_url::plain_path(canonical.as_os_str().as_bytes()).ok_or(Refusal::FilePath)?;
    let meta = std::fs::metadata(resolved).map_err(|_| Refusal::Unreachable)?;
    let name = resolved.rsplit('/').next().unwrap_or_default();
    let launcher = name
        .get(name.len().saturating_sub(".desktop".len())..)
        .is_some_and(|ext| ext.eq_ignore_ascii_case(".desktop"));
    if launcher {
        Err(Refusal::Launcher)
    } else if meta.is_file() && meta.permissions().mode() & 0o111 != 0 {
        Err(Refusal::Executable)
    } else {
        Ok(file_url::encode(resolved))
    }
}

/// Judge `url` against every rule, on the machine named `hostname`.
pub fn judge(url: &str, hostname: &Hostname) -> Verdict {
    if !browser::can_open(url) {
        return Verdict::Refused(Refusal::Scheme);
    }
    if hides_a_character(url) {
        return Verdict::Refused(Refusal::Hidden);
    }
    if url.starts_with("mailto:") && !mailto::fields_are_allowed(url) {
        return Verdict::Refused(Refusal::MailField);
    }
    let file = url.starts_with("file:");
    match host::read(url) {
        Host::Refused => Verdict::Refused(Refusal::Host),
        Host::Name(name) if file && !hostname.is_local(name) => {
            Verdict::Refused(Refusal::ForeignHost)
        }
        _ if file && !file_url::path_is_unambiguous(url) => Verdict::Refused(Refusal::FilePath),
        Host::Name(_) | Host::Absent => Verdict::Open,
    }
}

/// Whether `url` holds a character that draws nothing: a control character, or one
/// [`link::is_hidden`] names. Visible non-ASCII is not one, since `rg --hyperlink-format`
/// prints paths unencoded. A control character reaches a target only through OSC 8, as
/// DEL or a C1 control.
///
/// Out of line: inlined into [`judge`], it moved code that never runs it, and the tab
/// bar frame measured 13% slower.
#[inline(never)]
fn hides_a_character(url: &str) -> bool {
    if url.is_ascii() {
        url.bytes().any(|b| b.is_ascii_control())
    } else {
        url.chars().any(|c| c.is_control() || link::is_hidden(c))
    }
}

#[cfg(test)]
mod tests {
    use crate::platform::verdict::*;

    #[test]
    fn a_target_hiding_a_character_is_refused() {
        for hidden in [
            '\u{200B}',
            '\u{00AD}',
            '\u{034F}',
            '\u{FE0F}',
            '\u{E0100}',
            '\u{202E}',
        ] {
            let url = format!("https://example.com/a{hidden}b");
            assert_eq!(
                judge(&url, &Hostname::default()),
                Verdict::Refused(Refusal::Hidden),
                "{url:?}"
            );
        }
        // Percent-encoded, the same character is visible in the target and opens.
        assert_eq!(
            judge("https://example.com/a%E2%80%8Bb", &Hostname::default()),
            Verdict::Open
        );
        // So does a character that draws something.
        assert_eq!(
            judge("https://de.wikipedia.org/wiki/Grüße", &Hostname::default()),
            Verdict::Open
        );
    }

    #[test]
    fn a_target_holding_a_control_character_is_refused() {
        for url in [
            "https://example.com/a\u{7F}b",
            "https://example.com/a\u{85}b",
            "https://example.com/\u{9B}31m",
            "https://example.com/a\tb",
        ] {
            assert_eq!(
                judge(url, &Hostname::default()),
                Verdict::Refused(Refusal::Hidden),
                "{url:?}"
            );
        }
        for url in [
            "https://example.com/a%7Fb",
            "https://example.com/a%C2%85b",
            "file:///home/u/файл.txt",
        ] {
            assert_eq!(judge(url, &Hostname::default()), Verdict::Open, "{url}");
        }
    }

    #[test]
    fn a_target_whose_host_a_browser_would_rewrite_is_refused() {
        for url in [
            "https://github.com@evil.com/",
            "https://p%D0%B0ypal.com/",
            "http://2130706433/",
        ] {
            assert_eq!(
                judge(url, &Hostname::default()),
                Verdict::Refused(Refusal::Host),
                "{url}"
            );
        }
        for url in [
            "https://xn--pypal-4ve.com/",
            "http://[::1]/",
            "mailto:a@b.com",
        ] {
            assert_eq!(judge(url, &Hostname::default()), Verdict::Open, "{url}");
        }
    }

    #[test]
    fn a_file_link_naming_another_machine_is_refused() {
        let here = Hostname::new("bnk-desktop");
        assert_eq!(
            judge("file://buildbox/home/u/notes.md", &here),
            Verdict::Refused(Refusal::ForeignHost)
        );
        for url in [
            "file:///home/u/notes.md",
            "file://localhost/home/u/notes.md",
            "file://BNK-desktop/home/u/notes.md",
            "https://buildbox/home/u/notes.md",
        ] {
            assert_eq!(judge(url, &here), Verdict::Open, "{url}");
        }
    }

    #[test]
    fn a_local_ls_hyperlink_opens() {
        // GNU `ls --hyperlink` names this machine the way the kernel does, so its
        // links must judge as local under the hostname read at startup.
        let fixture = fixture_dir("ls");
        std::fs::write(fixture.0.join("notes.txt"), b"x").unwrap();
        let out = std::process::Command::new("ls")
            .arg("--hyperlink=always")
            .arg(&fixture.0)
            .output();
        let Some(out) = out.ok().filter(|out| out.status.success()) else {
            eprintln!("no GNU ls with --hyperlink; skipping");
            return;
        };
        let text = String::from_utf8(out.stdout).unwrap();
        let uri = text
            .strip_prefix("\x1b]8;;")
            .and_then(|rest| rest.split_once('\x1b'))
            .map(|(uri, _)| uri)
            .unwrap();
        assert!(uri.starts_with("file://"), "{uri}");
        assert_eq!(judge(uri, &Hostname::read()), Verdict::Open, "{uri}");
    }

    #[test]
    fn a_file_link_whose_path_decodes_more_than_one_way_is_refused() {
        let here = Hostname::default();
        for url in [
            "file:///tmp/payload.desktop%0A",
            "file:///tmp/a%zz",
            "file:///tmp/a?b",
            "file:///tmp/a#b",
        ] {
            assert_eq!(
                judge(url, &here),
                Verdict::Refused(Refusal::FilePath),
                "{url}"
            );
        }
        assert!(matches!(
            judge("file:///tmp/a\\nb", &here),
            Verdict::Refused(_)
        ));
        assert_eq!(judge("file:///tmp/a%20b", &here), Verdict::Open);
    }

    #[test]
    fn a_mailto_link_with_a_field_beyond_the_allowlist_is_refused() {
        let here = Hostname::default();
        for url in [
            "mailto:a@b.com?attach=/home/u/.ssh/id_ed25519",
            "mailto:a@b.com?%61ttach=x",
            "mailto:a@b.com?in-reply-to=x",
        ] {
            assert_eq!(
                judge(url, &here),
                Verdict::Refused(Refusal::MailField),
                "{url}"
            );
        }
        assert_eq!(judge("mailto:a@b.com?subject=hi", &here), Verdict::Open);
    }

    /// A fresh directory for one test's fixtures, removed when it drops, pass or fail.
    /// Named by the clock as well as the pid, since a sandbox's pids repeat across runs
    /// sharing one temp directory.
    struct FixtureDir(std::path::PathBuf);

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture_dir(name: &str) -> FixtureDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("bnkterm-{name}-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        FixtureDir(std::fs::canonicalize(dir).unwrap())
    }

    /// Write `name` in `dir` with permission bits `mode`.
    fn write_file(dir: &std::path::Path, name: &str, mode: u32) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, b"x").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    fn url_of(path: &std::path::Path) -> String {
        file_url::encode(path.to_str().unwrap())
    }

    #[test]
    fn a_file_link_is_judged_at_the_click_by_what_it_reaches() {
        use std::os::unix::fs::symlink;
        let fixture = fixture_dir("judge-file");
        let dir = fixture.0.as_path();
        let notes = write_file(dir, "notes.txt", 0o644);
        let script = write_file(dir, "build.sh", 0o755);
        let launcher = write_file(dir, "payload.desktop", 0o644);
        let alias = dir.join("readme.txt");
        symlink(&launcher, &alias).unwrap();
        let sub = dir.join("docs");
        std::fs::create_dir(&sub).unwrap();

        let opened = judge_file(&url_of(&notes)).unwrap();
        assert_eq!(file_url::decoded_path(&opened).as_deref(), notes.to_str());
        assert!(judge_file(&url_of(&sub)).is_ok(), "a directory opens");
        assert_eq!(judge_file(&url_of(&script)), Err(Refusal::Executable));
        assert_eq!(judge_file(&url_of(&launcher)), Err(Refusal::Launcher));
        assert_eq!(
            judge_file(&url_of(&alias)),
            Err(Refusal::Launcher),
            "a symlink is judged by its target"
        );
        assert_eq!(
            judge_file(&url_of(&dir.join("missing.txt"))),
            Err(Refusal::Unreachable)
        );
    }

    #[test]
    fn a_symlink_to_a_name_with_a_newline_is_refused_by_its_resolved_path() {
        // The symlink form of `payload.desktop%0A`: `xdg-open` would drop the newline and
        // open the real launcher beside it, so the resolved path itself must be plain.
        use std::os::unix::fs::symlink;
        let fixture = fixture_dir("judge-newline");
        let dir = fixture.0.as_path();
        write_file(dir, "payload.desktop", 0o644);
        let trailing = write_file(dir, "payload.desktop\n", 0o600);
        let notes = dir.join("notes.txt");
        symlink(&trailing, &notes).unwrap();
        assert_eq!(judge_file(&url_of(&notes)), Err(Refusal::FilePath));
    }

    #[test]
    fn the_url_handed_to_the_opener_names_the_checked_path() {
        let fixture = fixture_dir("judge-encode");
        let dir = fixture.0.as_path();
        for name in [
            "a b.txt",
            "report #3.md",
            "what?.txt",
            "100%.txt",
            "файл.txt",
        ] {
            let path = write_file(dir, name, 0o644);
            let opened = judge_file(&url_of(&path)).unwrap();
            assert!(opened.is_ascii(), "{opened}");
            assert_eq!(file_url::decoded_path(&opened).as_deref(), path.to_str());
        }
    }
}
