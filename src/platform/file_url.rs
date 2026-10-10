//! Reading a `file:` link's path one way.
//!
//! Openers decode a `file:` URL differently. `xdg-open`'s generic path decodes through
//! `printf` inside command substitution, which drops a decoded trailing newline and turns
//! a literal `\n` into one, so it reads `file:///tmp/payload.desktop%0A` as
//! `/tmp/payload.desktop` while a strict decoder reads a different file with a newline
//! on the end. A path is therefore decoded once, strictly, and refused unless every
//! reading of it agrees.

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
    if url.bytes().any(|b| matches!(b, b'?' | b'#')) {
        return false;
    }
    let Some(rest) = url.strip_prefix("file://") else {
        return false;
    };
    let Some(path) = rest
        .bytes()
        .position(|b| b == b'/')
        .and_then(|at| rest.get(at..))
    else {
        return false;
    };
    let mut decoded = [0u8; PATH_MAX];
    let mut len = 0;
    for byte in percent_decoded(path) {
        let (Some(byte), Some(slot)) = (byte, decoded.get_mut(len)) else {
            return false;
        };
        *slot = byte;
        len += 1;
    }
    decoded.get(..len).and_then(plain_path).is_some()
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
        assert!(path_is_unambiguous(&longest));
        assert!(
            !path_is_unambiguous(&format!("{longest}a")),
            "past PATH_MAX"
        );
    }
}
