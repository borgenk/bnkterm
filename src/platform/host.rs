//! Reading a link's host, one way, for every rule that compares or shows one.
//!
//! A browser does not resolve the host it is handed. It percent-decodes it, maps it
//! through IDNA, reads a numeric last label as an IPv4 address, and treats `\` as `/`
//! ([URL Standard, host parsing][host]). bnkterm implements none of that, so it accepts
//! only the forms those steps leave alone and refuses the rest. Whatever host a rule
//! compares, or a preview shows, is then the host that is reached.
//!
//! ```text
//!   https://user@ex%61mple.com:8080/path?q
//!           ╰────────┬─────────────╯
//!                authority: from `//` to the first `/`, `?` or `#`
//!   accepted  example.com   xn--pypal-4ve.com   127.0.0.1   [::1]
//!   refused   userinfo (`@`), `%`, non-ASCII, `\`, a numeric last label
//!             that is not canonical dotted-decimal IPv4
//! ```
//!
//! [host]: https://url.spec.whatwg.org/#host-parsing

use std::net::Ipv6Addr;
use std::rc::Rc;

/// Where the kernel keeps this machine's hostname. A Flatpak sandbox shares the UTS
/// namespace, so it reads the same name there.
const HOSTNAME_PATH: &str = "/proc/sys/kernel/hostname";

/// This machine's hostname, as `file:` links name it (`ls --hyperlink` prints
/// `file://HOSTNAME/path`). Empty when it could not be read, which leaves only an empty
/// host and `localhost` naming this machine.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Hostname(Rc<str>);

impl Hostname {
    /// Read the hostname once, at startup.
    pub fn read() -> Hostname {
        std::fs::read_to_string(HOSTNAME_PATH)
            .map(|name| Hostname(name.trim().into()))
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub fn new(name: &str) -> Hostname {
        Hostname(name.into())
    }

    /// Whether a `file:` URL's `host` names this machine: empty, `localhost`, or this
    /// hostname, without regard to case.
    pub fn is_local(&self, host: &str) -> bool {
        host.is_empty()
            || host.eq_ignore_ascii_case("localhost")
            || (!self.0.is_empty() && host.eq_ignore_ascii_case(&self.0))
    }
}

/// What [`read`] found where a link's host would be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Host<'a> {
    /// The scheme names no host (`mailto:`).
    Absent,
    /// The host as written: ASCII labels of letters, digits and hyphens, canonical
    /// dotted-decimal IPv4, or bracketed IPv6. Compared without regard to case. Empty
    /// only for `file:`, where it means this machine.
    Name(&'a str),
    /// A host a browser would rewrite before resolving it, or one it cannot read.
    Refused,
}

/// The host of `url`, read the one way every rule reads it. Only a scheme spelled with
/// `//` (`https://`, `http://`, `file://`) has one.
///
/// It runs on every bare-URL lookup, so it scans bytes rather than chars: everything
/// it looks for is ASCII.
pub fn read(url: &str) -> Host<'_> {
    let Some(colon) = url.bytes().position(|b| b == b':') else {
        return Host::Absent;
    };
    let scheme = url.get(..colon).unwrap_or_default();
    let Some(rest) = url
        .get(colon + 1..)
        .and_then(|rest| rest.strip_prefix("//"))
    else {
        return Host::Absent;
    };
    // A browser reads `\` as `/` in these schemes, so `https://evil.com\@github.com`
    // goes to `evil.com`, not to the `github.com` a plain split would find.
    if url.as_bytes().contains(&b'\\') {
        return Host::Refused;
    }
    let end = rest
        .bytes()
        .position(|b| matches!(b, b'/' | b'?' | b'#'))
        .unwrap_or(rest.len());
    let authority = rest.get(..end).unwrap_or_default();
    // Userinfo (`https://github.com@evil.com/` goes to `evil.com`), escapes, and
    // anything IDNA would map are all a host other than the one written.
    if authority
        .bytes()
        .any(|b| matches!(b, b'@' | b'%') || !b.is_ascii())
    {
        return Host::Refused;
    }
    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => match bracketed.split_once(']') {
            Some((v6, port)) if v6.parse::<Ipv6Addr>().is_ok() => {
                (authority.get(..v6.len() + 2).unwrap_or_default(), port)
            }
            _ => return Host::Refused,
        },
        None => authority
            .split_at_checked(
                authority
                    .bytes()
                    .position(|b| b == b':')
                    .unwrap_or(authority.len()),
            )
            .unwrap_or((authority, "")),
    };
    // A `file:` URL has no port, and a browser reads `file://c:/x` as the drive letter
    // of an empty host, not as a host `c`.
    let file = scheme.eq_ignore_ascii_case("file");
    if !port_is_plain(port) || (file && !port.is_empty()) {
        return Host::Refused;
    }
    if host.is_empty() {
        return if file {
            Host::Name(host)
        } else {
            Host::Refused
        };
    }
    if host.starts_with('[') || is_domain(host) {
        Host::Name(host)
    } else {
        Host::Refused
    }
}

/// Whether what follows the host is no port, or `:` and a port number.
fn port_is_plain(port: &str) -> bool {
    match port.strip_prefix(':') {
        None => port.is_empty(),
        Some(digits) => {
            digits.is_empty()
                || (digits.bytes().all(|b| b.is_ascii_digit()) && digits.parse::<u16>().is_ok())
        }
    }
}

/// Whether `host` is dot-separated labels of ASCII letters, digits and hyphens whose
/// last label does not read as a number, or is canonical dotted-decimal IPv4. A
/// browser reads a numeric last label as an address: `2130706433` and `0x7f.1` are
/// both `127.0.0.1`.
fn is_domain(host: &str) -> bool {
    let ldh = host.split('.').all(|label| {
        !label.is_empty()
            && label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    });
    let last = host.rsplit('.').next().unwrap_or_default();
    ldh && (!reads_as_number(last) || is_canonical_ipv4(host))
}

/// Whether a browser reads `label` as a number: decimal digits, or `0x` and hex digits.
fn reads_as_number(label: &str) -> bool {
    match label
        .strip_prefix("0x")
        .or_else(|| label.strip_prefix("0X"))
    {
        Some(hex) => hex.bytes().all(|b| b.is_ascii_hexdigit()),
        None => !label.is_empty() && label.bytes().all(|b| b.is_ascii_digit()),
    }
}

/// Whether `host` is four decimal octets, each 0–255 and written without a leading zero,
/// which a browser would read as octal.
fn is_canonical_ipv4(host: &str) -> bool {
    let mut octets = 0;
    host.split('.').all(|octet| {
        octets += 1;
        octet.parse::<u8>().is_ok() && (octet.len() == 1 || !octet.starts_with('0'))
    }) && octets == 4
}

#[cfg(test)]
mod tests {
    use crate::platform::host::*;

    #[test]
    fn a_host_a_browser_would_rewrite_is_refused() {
        for url in [
            "https://p%D0%B0ypal.com/",
            "https://ｅxample.com/",
            "https://pаypal.com/",
            "https://github.com@evil.com/",
            "https://evil.com\\@github.com/",
            "http://2130706433/",
            "http://0x7f.1/",
            "http://127.0.0.01/",
            "http://1.2.3/",
            "http://example.com./",
            "http://exa_mple.com/",
            "https:///x",
            "http://[::1/",
            "http://[zz]/",
            "http://example.com:http/",
            "http://example.com:99999/",
            "file://c:/x",
            "file://localhost:/etc/hosts",
            "file://host:80/x",
        ] {
            assert_eq!(read(url), Host::Refused, "{url}");
        }
    }

    #[test]
    fn a_host_read_as_written_is_kept() {
        for (url, host) in [
            ("https://xn--pypal-4ve.com/a", "xn--pypal-4ve.com"),
            ("http://127.0.0.1:8080/", "127.0.0.1"),
            ("http://[::1]/", "[::1]"),
            ("http://[::1]:3000", "[::1]"),
            ("https://Example.COM?q#f", "Example.COM"),
            ("http://localhost:/", "localhost"),
            ("http://0x7f.example.com/", "0x7f.example.com"),
            ("file:///etc/hosts", ""),
            ("file://localhost/etc/hosts", "localhost"),
            ("https://de.wikipedia.org/wiki/Grüße", "de.wikipedia.org"),
        ] {
            assert_eq!(read(url), Host::Name(host), "{url}");
        }
        assert_eq!(read("mailto:a@b.com"), Host::Absent);
    }

    #[test]
    fn a_file_host_is_local_when_it_names_this_machine() {
        let here = Hostname::new("bnk-desktop");
        for host in ["", "localhost", "LocalHost", "bnk-desktop", "BNK-Desktop"] {
            assert!(here.is_local(host), "{host:?}");
        }
        for host in ["other", "bnk-desktop.lan", "bnk"] {
            assert!(!here.is_local(host), "{host:?}");
        }
        // An unread hostname names nothing, not everything.
        assert!(!Hostname::default().is_local("bnk-desktop"));
        assert!(Hostname::default().is_local("localhost"));
    }
}
