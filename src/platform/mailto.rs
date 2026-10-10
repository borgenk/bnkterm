//! Which `mailto:` links are handed to a mail client.
//!
//! RFC 6068 lets a `mailto:` URL set any header field, and mail clients have honoured
//! non-standard ones: an `attach=` field turns a click into a draft with a local file
//! (an SSH key) already attached. A link may set only the fields a message needs, and
//! one that sets anything else is refused rather than stripped, since a stripped link
//! opens something other than what was printed.

use crate::platform::bytes::percent_decoded;

/// The header fields a `mailto:` link may set.
const FIELDS: [&str; 5] = ["to", "cc", "bcc", "subject", "body"];

/// Whether every header field `url` sets is one of [`FIELDS`]. A field name is compared
/// after percent-decoding and ASCII case folding, so `ATTACH` and `%61ttach` are both
/// `attach`. A second `?`, which RFC 6068 does not allow in a field and which clients
/// split differently, refuses the link too.
pub fn fields_are_allowed(url: &str) -> bool {
    let Some((_, query)) = url.split_once('?') else {
        return true;
    };
    !query.contains('?')
        && query.split('&').all(|field| {
            let name = field.split_once('=').map_or(field, |(name, _)| name);
            FIELDS.iter().any(|allowed| decodes_to(name, allowed))
        })
}

/// Whether the percent-encoded field `name` decodes to `allowed`, without regard to
/// ASCII case.
fn decodes_to(name: &str, allowed: &str) -> bool {
    let mut decoded = percent_decoded(name);
    let mut wanted = allowed.bytes();
    loop {
        match (decoded.next(), wanted.next()) {
            (None, None) => return true,
            (Some(Some(got)), Some(want)) if got.eq_ignore_ascii_case(&want) => {}
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::platform::mailto::*;

    #[test]
    fn a_field_beyond_the_allowlist_is_refused() {
        for url in [
            "mailto:a@b.com?attach=/home/u/.ssh/id_ed25519",
            "mailto:a@b.com?ATTACH=/home/u/.ssh/id_ed25519",
            "mailto:a@b.com?%61ttach=/home/u/.ssh/id_ed25519",
            "mailto:a@b.com?subject=hi&in-reply-to=%3Cx%40y%3E",
            "mailto:a@b.com?subject=hi?attach=x",
            "mailto:a@b.com?subject%zz=hi",
            "mailto:a@b.com?",
        ] {
            assert!(!fields_are_allowed(url), "{url}");
        }
    }

    #[test]
    fn the_fields_a_message_needs_are_kept() {
        for url in [
            "mailto:a@b.com",
            "mailto:a@b.com?subject=Hi%20there&body=x",
            "mailto:a@b.com?CC=c@d.com&Bcc=e@f.com",
            "mailto:?to=a@b.com&%73ubject=x",
            "mailto:a@b.com?body",
        ] {
            assert!(fields_are_allowed(url), "{url}");
        }
    }
}
