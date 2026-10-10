//! Whether a link may open: the rules a link's target must pass before the terminal
//! offers it to the pointer or hands it to an opener.
//!
//! Every rule is a pure function of the target, so a verdict is worked out where a link
//! enters the terminal and kept with it: an OSC 8 anchor is judged once, when the grid
//! interns it. A link either opens or is refused with a [`Refusal`]; there is no third
//! outcome to confirm.

use crate::platform::browser;

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
}

/// Judge `url` against every rule.
pub fn judge(url: &str) -> Verdict {
    if browser::can_open(url) {
        Verdict::Open
    } else {
        Verdict::Refused(Refusal::Scheme)
    }
}
