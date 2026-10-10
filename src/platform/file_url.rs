//! Reading a `file:` link's path one way.
//!
//! Openers decode a `file:` URL differently. `xdg-open`'s generic path decodes through
//! `printf` inside command substitution, which drops a decoded trailing newline and turns
//! a literal `\n` into one, so it reads `file:///tmp/payload.desktop%0A` as
//! `/tmp/payload.desktop` while a strict decoder reads a different file with a newline
//! on the end. A path is therefore decoded once, strictly, and refused unless every
//! reading of it agrees.

use std::fmt::Write as _;

use crate::platform::bytes::percent_decoded;

/// The longest path the kernel opens, and so the most a decoded path may hold.
const PATH_MAX: usize = 4096;

/// Whether `url`, a `file:` URL, has a path every opener reads the same way: no query
/// or fragment, every `%` the start of a two-hex-digit escape, and a decoded path that
/// [`plain_path`] accepts.
///
/// It runs on every lookup of a `file:` link, so it decodes into a buffer on the stack
/// rather than allocating. A path too long for it is too long to open.
pub fn path_is_unambiguous(url: &str) -> bool {
    let Some(bytes) = path_bytes(url) else {
        return false;
    };
    let mut decoded = [0u8; PATH_MAX];
    let mut len = 0;
    for byte in bytes {
        let (Some(byte), Some(slot)) = (byte, decoded.get_mut(len)) else {
            return false;
        };
        *slot = byte;
        len += 1;
    }
    decoded.get(..len).and_then(plain_path).is_some()
}

/// The path `url` names, decoded the one way [`path_is_unambiguous`] reads it, when every
/// reader agrees on it. For the click, where the path is checked against the filesystem.
pub fn decoded_path(url: &str) -> Option<String> {
    let bytes = path_bytes(url)?.collect::<Option<Vec<u8>>>()?;
    plain_path(&bytes).map(str::to_string)
}

/// A `file://` URL for `path` with every byte outside the unreserved set and `/`
/// percent-encoded, so an opener decodes it back to exactly `path`. Safe for `xdg-open`'s
/// `printf` decoder only because a [`plain_path`] holds no control character or `\` for
/// it to drop or reinterpret.
pub fn encode(path: &str) -> String {
    let mut url = String::from("file://");
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            url.push(char::from(byte));
        } else {
            let _ = write!(url, "%{byte:02X}");
        }
    }
    url
}

/// The bytes of `url`'s path, percent-decoded strictly as they are read (`None` for a bad
/// escape). `None` outright when the URL has a query or fragment, or no path.
fn path_bytes(url: &str) -> Option<impl Iterator<Item = Option<u8>> + '_> {
    if url.bytes().any(|b| matches!(b, b'?' | b'#')) {
        return None;
    }
    let rest = url.strip_prefix("file://")?;
    let path = rest.get(rest.bytes().position(|b| b == b'/')?..)?;
    Some(percent_decoded(path))
}

/// `path` as text when every reader agrees on it: valid UTF-8, absolute, with no
/// control character (U+0000–U+001F, U+007F) and no `\`. Run on a link's decoded path,
/// and again on the path it resolves to before it is opened.
pub fn plain_path(path: &[u8]) -> Option<&str> {
    let text = std::str::from_utf8(path).ok()?;
    let plain = !path.iter().any(|&b| b < 0x20 || b == 0x7F || b == b'\\');
    (text.starts_with('/') && plain).then_some(text)
}

#[cfg(test)]
mod tests {
    use crate::platform::file_url::*;

    #[test]
    fn a_path_that_decodes_more_than_one_way_is_refused() {
        for url in [
            "file:///tmp/payload.desktop%0A",
            "file:///tmp/a%zz",
            "file:///tmp/a%2",
            "file:///tmp/a\\nb",
            "file:///tmp/a%5Cnb",
            "file:///tmp/a?b",
            "file:///tmp/a#b",
            "file:///tmp/a%00b",
            "file:///tmp/a%7Fb",
            "file:///tmp/%FF",
            "file://host",
        ] {
            assert!(!path_is_unambiguous(url), "{url}");
        }
    }

    #[test]
    fn a_path_read_one_way_is_kept() {
        for url in [
            "file:///etc/hosts",
            "file://localhost/tmp/a%20b.txt",
            "file:///home/u/%D1%84%D0%B0%D0%B9%D0%BB.txt",
            "file:///home/u/файл.txt",
            "file:///tmp/a%3Fb%23c",
        ] {
            assert!(path_is_unambiguous(url), "{url}");
        }
        let longest = format!("file://{}", "/a".repeat(PATH_MAX / 2));
        assert!(decoded_path(&longest).is_some());
        assert!(path_is_unambiguous(&longest));
        assert!(
            !path_is_unambiguous(&format!("{longest}a")),
            "past PATH_MAX"
        );
    }

    #[test]
    fn an_encoded_path_decodes_back_to_itself() {
        for path in [
            "/tmp/a b.txt",
            "/tmp/report #3.md",
            "/tmp/what?.txt",
            "/tmp/100%.txt",
            "/home/u/файл.txt",
            "/home/u/日本語/ノート.md",
        ] {
            let url = encode(path);
            assert!(url.is_ascii(), "{url}");
            assert_eq!(decoded_path(&url).as_deref(), Some(path), "{url}");
            assert!(path_is_unambiguous(&url), "{url}");
        }
    }
}
