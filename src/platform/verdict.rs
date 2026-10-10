//! Whether a link may open: the rules a link's target must pass before the terminal
//! offers it to the pointer or hands it to an opener.
//!
//! Every rule is a pure function of the target, so a verdict is worked out where a link
//! enters the terminal and kept with it: an OSC 8 anchor is judged once, when the grid
//! interns it. A link either opens or is refused with a [`Refusal`]; there is no third
//! outcome to confirm.

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
}

/// Judge `url` against every rule.
pub fn judge(url: &str) -> Verdict {
    if !browser::can_open(url) {
        Verdict::Refused(Refusal::Scheme)
    } else if !url.is_ascii() && url.chars().any(link::is_hidden) {
        Verdict::Refused(Refusal::Hidden)
    } else {
        Verdict::Open
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
            assert_eq!(judge(&url), Verdict::Refused(Refusal::Hidden), "{url:?}");
        }
        // Percent-encoded, the same character is visible in the target and opens.
        assert_eq!(judge("https://example.com/a%E2%80%8Bb"), Verdict::Open);
        // So does a character that draws something.
        assert_eq!(judge("https://de.wikipedia.org/wiki/Grüße"), Verdict::Open);
    }
}
