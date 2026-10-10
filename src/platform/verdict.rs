//! Whether a link may open: the rules a link's target must pass before the terminal
//! offers it to the pointer or hands it to an opener.
//!
//! Every rule is a pure function of the target, so a verdict is worked out where a link
//! enters the terminal and kept with it: an OSC 8 anchor is judged once, when the grid
//! interns it. A link either opens or is refused with a [`Refusal`]; there is no third
//! outcome to confirm.

use crate::platform::host::{self, Host, Hostname};
use crate::platform::{browser, link};

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
    /// The target carries a character that draws nothing ([`link::is_hidden`]), so the
    /// text the user reads is not the target that opens.
    Hidden,
    /// The target's host is one a browser would rewrite before resolving it
    /// ([`host::read`]), so the host the user reads is not the one reached.
    Host,
    /// A `file:` target names another machine. The opener would open the file at the
    /// same path on this one.
    ForeignHost,
}

/// Judge `url` against every rule, on the machine named `hostname`.
pub fn judge(url: &str, hostname: &Hostname) -> Verdict {
    if !browser::can_open(url) {
        return Verdict::Refused(Refusal::Scheme);
    }
    if !url.is_ascii() && url.chars().any(link::is_hidden) {
        return Verdict::Refused(Refusal::Hidden);
    }
    match host::read(url) {
        Host::Refused => Verdict::Refused(Refusal::Host),
        Host::Name(name) if url.starts_with("file:") && !hostname.is_local(name) => {
            Verdict::Refused(Refusal::ForeignHost)
        }
        Host::Name(_) | Host::Absent => Verdict::Open,
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
        let dir = std::env::temp_dir().join(format!("bnkterm-ls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let out = std::process::Command::new("ls")
            .arg("--hyperlink=always")
            .arg(&dir)
            .output();
        let _ = std::fs::remove_dir_all(&dir);
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
}
