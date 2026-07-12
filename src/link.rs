//! Finding a URL inside a line of terminal text.
//!
//! Terminal output is unstructured prose: `git push` prints a pull-request URL,
//! `cargo` prints a docs link, a dev server prints `http://localhost:3000`. None of
//! them mark those as links (OSC 8, the escape sequence that would, is emitted by
//! almost nothing), so a terminal that wants them clickable has to *find* them in
//! the text. That is this module, and nothing else: a pure `&str -> Option<Range>`
//! scan holding no grid and no parser state, so it is exhaustively testable without
//! a window, a PTY, or a GPU.
//!
//! ```text
//!   "see (https://en.wikipedia.org/wiki/Cure_(album)) for more."
//!         ╰─┬──╯
//!           └─ initiation: a scheme we can open, at a token boundary
//!              termination then walks right, carrying
//!                last_safe  the last offset known to be *inside* the link
//!                stack      the brackets the URL itself opened
//!
//!   result:      https://en.wikipedia.org/wiki/Cure_(album)
//!                  the inner `(album)` is kept: its `)` closes a `(` the URL opened
//!                  the outer `)` is dropped: it closes nothing the URL opened
//!                  the trailing `.` is dropped: only punctuation follows it
//! ```
//!
//! Those two cases are the entire reason this is not a one-line regex, and both are
//! solved the way [UTS #58][uts58] specifies: one left-to-right pass with a backup
//! pointer and a bracket stack.
//!
//! - **Trailing punctuation.** A *hard* character (whitespace, a control, anything
//!   RFC 3986 forbids) always ends the link. A *soft* one (`.`, `,`, `!`, ...) is
//!   legal inside a URL yet is also how prose punctuates, so it may not *end* one.
//!   `last_safe` gets this for free: soft characters never advance it, so a run of
//!   them at the end is simply never reached. `example.com/a.b` keeps its dot;
//!   `see example.com/a.` gives it back to the sentence.
//! - **Balanced brackets.** A closing bracket belongs to the URL only when the URL
//!   opened it, which is why `last_safe` advances only at nesting depth zero: while
//!   a bracket is open the link cannot end, and an unmatched closer ends it at once.
//!   `en.wikipedia.org/wiki/Cure_(album)` keeps its parentheses; `(see example.com/a)`
//!   gives its `)` back to the prose.
//!
//! We detect exactly the schemes [`crate::platform::browser`] is willing to open and
//! no others. A link the terminal underlines and then refuses to follow is a worse
//! bug than one it never underlined, and the text being scanned is hostile by
//! construction (any program can print anything), so `javascript:` and `data:` are
//! never even candidates.
//!
//! [uts58]: https://www.unicode.org/reports/tr58/

use std::ops::Range;

/// The schemes we detect, deliberately the same set [`crate::platform::browser`]
/// will launch. `https` precedes `http` only for readability; they cannot collide,
/// since neither is a prefix of the other.
const SCHEMES: [&str; 4] = ["https://", "http://", "file://", "mailto:"];

/// How deep a URL may nest brackets before we stop believing it is one. UTS #58
/// allows 125; a real URL never passes two or three, and a fixed array keeps the
/// scan allocation-free, which matters because it runs on pointer motion.
const MAX_NESTING: usize = 16;

/// The byte range of the URL covering `at` in `text`, or `None` when that offset is
/// not inside one. `at` is a byte offset of the character being asked about (the
/// cell under the pointer, in the caller's case).
///
/// Candidates are examined left to right rather than searched outward from `at`,
/// because a URL's extent is a function of where it *starts*: only the scheme says
/// which characters are still part of it, so scanning backwards from a cell in the
/// middle could not tell `?` in a query string from `?` at the end of a question.
pub fn find_at(text: &str, at: usize) -> Option<Range<usize>> {
    for (start, scheme_len) in scheme_starts(text) {
        let Some(range) = extent(text, start, scheme_len) else {
            continue;
        };
        if range.end <= at {
            // This link ends before the offset asked about; a later one may cover it.
            continue;
        }
        // Candidates are ascending, so the first one reaching past `at` either covers
        // it or begins after it, and in the latter case nothing else can cover it.
        return (range.start <= at).then_some(range);
    }
    None
}

/// Every offset where a [`SCHEMES`] entry begins a token, with that scheme's length.
/// Ascending, which is what lets [`find_at`] stop at its first candidate past `at`.
fn scheme_starts(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..text.len()).filter_map(|at| Some((at, scheme_at(text, at)?)))
}

/// The length of the scheme starting exactly at `at`, or `None` when no scheme does.
///
/// The token-boundary check is what stops `xhttps://x` and `see-mailto:x` from
/// reading as links: a character that could *continue* a scheme (RFC 3986 §3.1's
/// `ALPHA / DIGIT / "+" / "-" / "."`) sitting immediately before means the match is
/// the tail of a longer word, not a scheme.
fn scheme_at(text: &str, at: usize) -> Option<usize> {
    let rest = text.get(at..)?;
    let scheme = SCHEMES.iter().find(|scheme| rest.starts_with(**scheme))?;
    let preceded = text
        .get(..at)
        .and_then(|before| before.chars().next_back())
        .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    (!preceded).then_some(scheme.len())
}

/// Walk right from the scheme at `start` to the end of the URL (the UTS #58
/// termination pass), or `None` when nothing survives past the scheme itself: a bare
/// `https://` has no host, so it is a word, not a link.
fn extent(text: &str, start: usize, scheme_len: usize) -> Option<Range<usize>> {
    let body = start + scheme_len;
    // The last offset known to be inside the link. It advances only at nesting depth
    // zero and only past a character that can legally *end* a URL, so a trailing run
    // of prose punctuation, or an unclosed bracket and everything after it, is simply
    // never reached and falls out of the range.
    let mut last_safe = body;
    let mut stack = ['\0'; MAX_NESTING];
    let mut depth = 0usize;
    for (offset, c) in text[body..].char_indices() {
        let next = body + offset + c.len_utf8();
        match classify(c) {
            Class::Hard => break,
            Class::Soft => {}
            Class::Open => {
                if depth == MAX_NESTING {
                    break;
                }
                stack[depth] = c;
                depth += 1;
            }
            Class::Close(opener) => {
                // A closer the URL never opened belongs to the text around it.
                if depth == 0 || stack[depth - 1] != opener {
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    last_safe = next;
                }
            }
            Class::Url => {
                if depth == 0 {
                    last_safe = next;
                }
            }
        }
    }
    (last_safe > body).then_some(start..last_safe)
}

/// What a character does to the URL being scanned.
enum Class {
    /// Ends the URL unconditionally: whitespace, a control, or an ASCII character
    /// RFC 3986 forbids in one.
    Hard,
    /// Legal inside a URL, but also how prose punctuates, so it may not end one.
    Soft,
    /// Opens a bracket pair the URL may go on to close.
    Open,
    /// Closes the pair opened by the character it carries.
    Close(char),
    /// An ordinary URL character, which may end one.
    Url,
}

/// Classify one character. The bracket and soft arms come first, so the RFC 3986
/// membership test below only has to speak for what is left.
fn classify(c: char) -> Class {
    match c {
        // `[` and `]` are not decoration: they fence an IPv6 literal host
        // (`http://[::1]:8080/`), so the same stack that balances `(album)` balances
        // those too.
        '(' | '[' => Class::Open,
        ')' => Class::Close('('),
        ']' => Class::Close('['),
        // Sentence punctuation that is legal in a URL yet usually ends the sentence
        // rather than the path: the classic `see http://x.com/a.`
        '.' | ',' | ':' | ';' | '!' | '?' | '\'' => Class::Soft,
        c if c.is_whitespace() || c.is_control() => Class::Hard,
        c if c.is_ascii_alphanumeric() => Class::Url,
        // The rest of RFC 3986's ASCII: unreserved, gen-delims, sub-delims, and the
        // `%` of a percent-escape.
        '-' | '_' | '~' | '/' | '#' | '@' | '$' | '&' | '*' | '+' | '=' | '%' => Class::Url,
        // Every other ASCII character (`"`, `<`, `>`, `\`, `^`, backtick, `{`, `|`,
        // `}`) is illegal in a URL, so it ends one. This is what makes the common
        // ways of quoting a link — `<http://x>`, `"http://x"` — come out clean with
        // no special case for them.
        c if c.is_ascii() => Class::Hard,
        // Non-ASCII: an IRI may carry it (an accented Wikipedia title), and it is
        // never prose punctuation we would have to guess about.
        _ => Class::Url,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The URL covering the first byte of `needle` in `text`, as [`find_at`] sees it.
    /// Panics when `needle` is absent, so a typo in a case reads as a test bug.
    fn at_needle(text: &str, needle: &str) -> Option<String> {
        let at = text.find(needle).expect("needle not in text");
        find_at(text, at).map(|r| text[r].to_string())
    }

    /// The URL covering the offset of the first `|` in `marked`, which is removed
    /// before scanning. Lets a case point at the exact cell being hovered.
    fn at_cursor(marked: &str) -> Option<String> {
        let at = marked.find('|').expect("no cursor in case");
        let text = marked.replace('|', "");
        find_at(&text, at).map(|r| text[r].to_string())
    }

    #[test]
    fn a_bare_url_is_found_whole() {
        assert_eq!(
            at_needle("https://example.com/a/b", "https"),
            Some("https://example.com/a/b".into())
        );
    }

    #[test]
    fn a_url_is_found_from_any_cell_inside_it() {
        let text = "see https://example.com/path now";
        // Every offset from the scheme's `h` to the path's last character resolves to
        // the same link, so hovering anywhere along it underlines the whole thing.
        // The word after it resolves to nothing.
        for needle in ["https", "//", "example", "com", "/path", "now"] {
            let at = text.find(needle).unwrap();
            let found = find_at(text, at).map(|r| &text[r]);
            let expected = (needle != "now").then_some("https://example.com/path");
            assert_eq!(found, expected, "at {needle:?}");
        }
    }

    #[test]
    fn an_offset_outside_every_url_finds_nothing() {
        let text = "see https://example.com now";
        assert_eq!(at_cursor("s|ee https://example.com now"), None);
        assert_eq!(find_at(text, text.len()), None);
        assert_eq!(find_at("no links here at all", 3), None);
        assert_eq!(find_at("", 0), None);
    }

    #[test]
    fn trailing_sentence_punctuation_is_left_to_the_sentence() {
        // Only punctuation follows, so none of it is part of the link.
        assert_eq!(
            at_needle("go to https://example.com/a.", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("https://example.com/a, and then", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("really? https://example.com/a?!", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("https://example.com/a...", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn punctuation_inside_a_url_is_kept() {
        // The same characters, with something other than punctuation after them, are
        // ordinary URL bytes: a dotted path, a query string, a port, a fragment.
        assert_eq!(
            at_needle("https://example.com/a.b.c", "https"),
            Some("https://example.com/a.b.c".into())
        );
        assert_eq!(
            at_needle("https://example.com/s?q=a,b&n=1#top", "https"),
            Some("https://example.com/s?q=a,b&n=1#top".into())
        );
        // A dev server's URL has no dot in the host at all, which is why we never
        // require one.
        assert_eq!(
            at_needle("serving on http://localhost:3000/ ok", "http"),
            Some("http://localhost:3000/".into())
        );
    }

    #[test]
    fn a_url_closes_the_brackets_it_opened() {
        assert_eq!(
            at_needle("https://en.wikipedia.org/wiki/Cure_(album) rocks", "https"),
            Some("https://en.wikipedia.org/wiki/Cure_(album)".into())
        );
        // Nested, and with the sentence's own period after it.
        assert_eq!(
            at_needle("https://example.com/a(b(c)d)e.", "https"),
            Some("https://example.com/a(b(c)d)e".into())
        );
        // An IPv6 literal host is bracket-balanced by the same stack.
        assert_eq!(
            at_needle("http://[::1]:8080/x", "http"),
            Some("http://[::1]:8080/x".into())
        );
    }

    #[test]
    fn a_bracket_the_url_did_not_open_belongs_to_the_prose() {
        assert_eq!(
            at_needle("(see https://example.com/a)", "https"),
            Some("https://example.com/a".into())
        );
        // The parenthetical closes *and* the sentence ends.
        assert_eq!(
            at_needle("(https://example.com/a).", "https"),
            Some("https://example.com/a".into())
        );
        // A bracket the URL opened but never closed takes everything after it out of
        // the link: half a bracketed path is not a path.
        assert_eq!(
            at_needle("https://example.com/a(b c", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn the_usual_ways_of_quoting_a_link_come_out_clean() {
        // `<`, `>`, and `"` cannot appear in a URL, so they end one with no special
        // case: this is RFC 3986 doing the work.
        assert_eq!(
            at_needle("<https://example.com/a>", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("\"https://example.com/a\"", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("'https://example.com/a'", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn a_scheme_must_start_a_token() {
        // The tail of a longer word only looks like a scheme.
        assert_eq!(at_needle("xhttps://example.com", "https"), None);
        assert_eq!(at_needle("not-a-mailto:x@y.com", "mailto"), None);
        assert_eq!(at_needle("1.https://example.com", "https"), None);
        // A delimiter before it is a boundary, so this one is real.
        assert_eq!(
            at_needle("url=https://example.com/a", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn only_schemes_we_can_open_are_detected() {
        // The detect set is the open set: never underline what we would then refuse
        // to follow, and never make a hostile scheme reachable in the first place.
        assert_eq!(at_needle("javascript:alert(1)", "javascript"), None);
        assert_eq!(at_needle("data:text/html,<h1>x", "data"), None);
        assert_eq!(at_needle("ftp://example.com/a", "ftp"), None);
        assert_eq!(at_needle("see example.com/a", "example"), None);
        assert_eq!(
            at_needle("mail me at mailto:a@b.com ok", "mailto"),
            Some("mailto:a@b.com".into())
        );
        assert_eq!(
            at_needle("file:///etc/hosts", "file"),
            Some("file:///etc/hosts".into())
        );
    }

    #[test]
    fn a_scheme_with_no_host_is_not_a_link() {
        // `extent` collapses to the scheme itself, which covers no offset past it.
        assert_eq!(find_at("https://", 0), None);
        assert_eq!(find_at("say https:// then", 4), None);
        assert_eq!(find_at("mailto:", 0), None);
    }

    #[test]
    fn the_right_url_is_returned_when_a_line_holds_several() {
        let text = "a https://one.com/x b http://two.com/y c";
        assert_eq!(
            at_needle(text, "https"),
            Some("https://one.com/x".into()),
            "first"
        );
        assert_eq!(
            at_needle(text, "http://two"),
            Some("http://two.com/y".into()),
            "second"
        );
        // The gap between them belongs to neither.
        assert_eq!(at_cursor("a https://one.com/x |b http://two.com/y c"), None);
        // A scheme nested in a query string is swallowed by the outer URL, not
        // reported as a second link.
        assert_eq!(
            at_needle("https://a.com/r?to=http://b.com/x", "http://b"),
            Some("https://a.com/r?to=http://b.com/x".into())
        );
    }

    #[test]
    fn an_iri_keeps_its_non_ascii() {
        assert_eq!(
            at_needle("https://de.wikipedia.org/wiki/Grüße ok", "https"),
            Some("https://de.wikipedia.org/wiki/Grüße".into())
        );
    }

    #[test]
    fn a_url_at_either_edge_of_the_line_is_whole() {
        assert_eq!(
            at_needle("https://example.com/a and text", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("text and https://example.com/a", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn control_characters_end_a_url() {
        // A program can print anything, including a URL run up against a control
        // byte; it must never be swallowed into the link.
        assert_eq!(
            at_needle("https://example.com/a\u{7}b", "https"),
            Some("https://example.com/a".into())
        );
        assert_eq!(
            at_needle("https://example.com/a\tb", "https"),
            Some("https://example.com/a".into())
        );
    }

    #[test]
    fn absurd_bracket_nesting_is_refused_rather_than_trusted() {
        // Past MAX_NESTING we stop believing this is a URL and keep only what was
        // safe before the nesting ran away. Bounded work on hostile input.
        let deep = format!("https://example.com/{}x", "(".repeat(MAX_NESTING + 4));
        assert_eq!(
            at_needle(&deep, "https"),
            Some("https://example.com/".into())
        );
    }
}
