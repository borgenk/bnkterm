//! The line shown beside a hovered link while Ctrl is held: where a click on it goes, or
//! why it goes nowhere.
//!
//! An open link shows its host first, then the whole target, so the part that decides
//! where a click lands is the part a narrow window keeps; a host too long for the line
//! keeps its end, the labels that decide where it resolves. A refused link shows the
//! reason first, then the target. Nothing in the line is drawn raw that the eye cannot
//! read for what it is: a control, hidden or blank-drawing character is written as an
//! escape (`⟨U+200B⟩`), and so is any non-ASCII character in the part that names where
//! the link goes (a host, or a `mailto:` address), where a lookalike letter would pass
//! for the one it imitates.
//!
//! ```text
//!   open      github.com  https://github.com/borgenk/bnkterm
//!   long      …aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbb.evil.example
//!   refused   refused: a browser would rewrite its host  https://p⟨U+0430⟩ypal.com/
//! ```
//!
//! `xn--` labels are shown as written: that ASCII form is the host that is resolved, and
//! a host that would need converting is refused before it gets here.

use std::fmt::Write as _;
use std::ops::Range;

use crate::platform::host::{self, Host};
use crate::platform::link;
use crate::platform::verdict::Verdict;

/// Between the leading part (a host or a reason) and the target.
const GAP: &str = "  ";

/// U+2800 BRAILLE PATTERN BLANK, which draws as a space but is neither whitespace nor
/// hidden.
const BRAILLE_BLANK: char = '\u{2800}';

/// Write the preview line for `url`, judged `verdict`, into `out`, replacing what it held,
/// for a plate `room` cells wide. The painter cuts what does not fit at the end, which
/// is right for the target and the reason; an open link's host that would not survive
/// that cut is written as its end alone instead.
pub(crate) fn write(out: &mut String, url: &str, verdict: Verdict, room: usize) {
    out.clear();
    match verdict {
        Verdict::Open => {
            match host::read(url) {
                // An open link's host is ASCII, one cell a byte. A cut keeps at most
                // `room - 1` cells before its `…`.
                Host::Name(host) if !host.is_empty() && host.len() >= room => {
                    let keep = room.saturating_sub(1);
                    out.push('…');
                    out.push_str(host.get(host.len() - keep..).unwrap_or_default());
                    return;
                }
                Host::Name(host) if !host.is_empty() => {
                    out.push_str(host);
                    out.push_str(GAP);
                }
                _ => {}
            }
        }
        Verdict::Refused(refusal) => {
            out.push_str("refused: ");
            out.push_str(refusal.reason());
            out.push_str(GAP);
        }
    }
    let destination = destination(url);
    for (at, c) in url.char_indices() {
        // Every character these name is past ASCII, where a control or a plain space
        // is already what it looks like; the hidden table is searched only there.
        let unseen = !c.is_ascii()
            && (destination.contains(&at)
                || c.is_whitespace()
                || c == BRAILLE_BLANK
                || link::is_hidden(c));
        if c.is_control() || unseen {
            let _ = write!(out, "⟨U+{:04X}⟩", u32::from(c));
        } else {
            out.push(c);
        }
    }
}

/// The byte range of `url` that names where it goes: the authority, after the scheme's
/// `//` up to the first `/`, `?` or `#`; or for `mailto:`, the addresses before any `?`.
fn destination(url: &str) -> Range<usize> {
    if let Some(addresses) = url.strip_prefix("mailto:") {
        let start = url.len() - addresses.len();
        let end = addresses.find('?').map_or(url.len(), |len| start + len);
        return start..end;
    }
    let Some(start) = url
        .find(':')
        .map(|colon| colon + 1)
        .filter(|&after| url.get(after..).is_some_and(|rest| rest.starts_with("//")))
        .map(|after| after + 2)
    else {
        return 0..0;
    };
    let end = url
        .get(start..)
        .and_then(|rest| rest.find(['/', '?', '#']))
        .map_or(url.len(), |len| start + len);
    start..end
}

#[cfg(test)]
mod tests {
    use crate::link_preview::*;
    use crate::platform::verdict::Refusal;

    /// The line for a plate wide enough to take any of these whole.
    fn preview(url: &str, verdict: Verdict) -> String {
        let mut out = String::new();
        write(&mut out, url, verdict, 200);
        out
    }

    #[test]
    fn an_open_link_shows_its_host_first_then_the_target() {
        for (url, line) in [
            (
                "https://github.com/borgenk/bnkterm",
                "github.com  https://github.com/borgenk/bnkterm",
            ),
            (
                "https://xn--pypal-4ve.com/login",
                "xn--pypal-4ve.com  https://xn--pypal-4ve.com/login",
            ),
            ("http://[::1]:8080/a?b#c", "[::1]  http://[::1]:8080/a?b#c"),
            (
                "https://de.wikipedia.org/wiki/Grüße",
                "de.wikipedia.org  https://de.wikipedia.org/wiki/Grüße",
            ),
            ("mailto:a@b.com?subject=hi", "mailto:a@b.com?subject=hi"),
            ("file:///home/u/notes.txt", "file:///home/u/notes.txt"),
        ] {
            assert_eq!(preview(url, Verdict::Open), line, "{url}");
        }
    }

    #[test]
    fn a_refused_link_shows_why_first_and_escapes_what_cannot_be_seen() {
        let refused = |refusal| Verdict::Refused(refusal);
        for (url, refusal, line) in [
            (
                "https://example.com/a\u{200B}b",
                Refusal::Hidden,
                "refused: it hides a character  https://example.com/a⟨U+200B⟩b",
            ),
            (
                "https://example.com/\u{85}x",
                Refusal::Hidden,
                "refused: it hides a character  https://example.com/⟨U+0085⟩x",
            ),
            (
                "https://p\u{430}ypal.com/login",
                Refusal::Host,
                "refused: a browser would rewrite its host  https://p⟨U+0430⟩ypal.com/login",
            ),
            (
                "https://evil.example/x",
                Refusal::LabelHost,
                "refused: its text names another host  https://evil.example/x",
            ),
            (
                "javascript:alert(1)",
                Refusal::Scheme,
                "refused: bnkterm does not open this kind of link  javascript:alert(1)",
            ),
        ] {
            assert_eq!(preview(url, refused(refusal)), line, "{url:?}");
        }
        // A mail domain names where the link goes as a host does.
        assert_eq!(
            preview("mailto:support@p\u{430}ypal.com?subject=hi", Verdict::Open),
            "mailto:support@p⟨U+0430⟩ypal.com?subject=hi"
        );
        // Characters that draw as blanks are escaped wherever they sit.
        assert_eq!(
            preview("file:///tmp/a\u{A0}b\u{2028}c\u{2800}d", Verdict::Open),
            "file:///tmp/a⟨U+00A0⟩b⟨U+2028⟩c⟨U+2800⟩d"
        );
        // Non-ASCII outside the host is ordinary text, refused or not.
        assert_eq!(
            preview(
                "file://other/home/u/файл.txt",
                refused(Refusal::ForeignHost)
            ),
            "refused: it names a file on another machine  file://other/home/u/файл.txt"
        );
    }

    #[test]
    fn a_host_too_long_for_the_line_keeps_its_end() {
        // The labels at a host's end decide where it resolves, and a cut at the line's
        // end would keep `github.com.aaaa…` and lose `evil.example`.
        let host = format!(
            "github.com.{}.{}.evil.example",
            "a".repeat(60),
            "b".repeat(60)
        );
        let url = format!("https://{host}/login");
        let line = {
            let mut out = String::new();
            write(&mut out, &url, Verdict::Open, 79);
            out
        };
        assert!(line.starts_with('…'), "{line}");
        assert!(line.ends_with(".evil.example"), "{line}");
        assert_eq!(crate::width::text_cells(&line), 79);
        assert!(host.ends_with(line.trim_start_matches('…')));
        // A host that fits keeps the line host first, whatever the target's length.
        let mut out = String::new();
        write(&mut out, "https://github.com/x", Verdict::Open, 11);
        assert_eq!(out, "github.com  https://github.com/x");
        write(&mut out, "https://github.com/x", Verdict::Open, 10);
        assert_eq!(out, "…ithub.com");
    }
}
