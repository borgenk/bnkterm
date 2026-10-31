//! Hyperlinks: URLs found in the text, OSC 8 anchors, the verdict each is given, and the
//! bounded link table.

use crate::grid::tests::anchor;
use crate::grid::*;
use crate::platform::verdict::Refusal;
use crate::testutil::feed;

//
// The anchor form throughout: `ESC ] 8 ; <params> ; <URI> ESC \` opens, the cells
// printed after it belong to the link, and `ESC ] 8 ; ; ESC \` closes.

#[test]
fn link_at_finds_a_url_and_spans_exactly_its_cells() {
    let mut s = Screen::new(40, 2);
    feed(&mut s, b"see https://example.com/a now");
    let mut probe = LinkProbe::default();
    // Cols 4..=24 hold the URL ("see " is 4 cells, the URL 21).
    let hit = s.link_at(0, 10, &mut probe).unwrap();
    assert_eq!(hit, ((0, 4), (0, 24)));
    assert_eq!(probe.url(), "https://example.com/a");
    assert_eq!(probe.verdict(), Some(Verdict::Open));
    // Every cell of the URL resolves to the same span, so the underline does not
    // flicker as the pointer crosses it.
    for col in 4..=24 {
        assert_eq!(s.link_at(0, col, &mut probe), Some(hit), "col {col}");
    }
    // The prose on either side is not a link.
    assert_eq!(s.link_at(0, 3, &mut probe), None, "the space before");
    assert_eq!(s.link_at(0, 25, &mut probe), None, "the space after");
    assert_eq!(probe.url(), "", "a miss leaves no stale URL behind");
    assert_eq!(probe.verdict(), None, "nor a stale verdict");
}

#[test]
fn link_at_joins_a_soft_wrapped_url() {
    // A URL that runs off the right margin continues on the next row; half of it
    // is not a link, so the scan must read the logical line, not the display row.
    let mut s = Screen::new(16, 3);
    feed(&mut s, b"go https://example.com/xy");
    assert!(s.row_wraps(0));
    let mut probe = LinkProbe::default();
    // Hovering either half yields the whole link, spanning the wrap.
    let hit = ((0, 3), (1, 8));
    assert_eq!(s.link_at(0, 5, &mut probe), Some(hit), "the first row");
    assert_eq!(probe.url(), "https://example.com/xy");
    assert_eq!(s.link_at(1, 2, &mut probe), Some(hit), "past the wrap");
    assert_eq!(probe.url(), "https://example.com/xy");
}

#[test]
fn link_at_reads_the_scrollback_when_scrolled() {
    // The probe is a view concern: scrolled up, row 0 is history, and a link there
    // must be found at the cell it is actually *shown* in.
    let mut s = Screen::new(30, 2);
    feed(&mut s, b"https://example.com/old\r\nb\r\nc");
    s.scroll_view_up(2);
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 5, &mut probe), Some(((0, 0), (0, 22))));
    assert_eq!(probe.url(), "https://example.com/old");
}

#[test]
fn link_at_covers_both_halves_of_a_wide_glyph() {
    // A CJK character in an IRI occupies two cells. Probing the spacer must find
    // the link (the right half of a character is the character), and the span must
    // cover the spacer, or the underline would stop half a glyph short.
    let mut s = Screen::new(20, 2);
    feed(&mut s, "https://x.com/世".as_bytes());
    assert!(s.cell(0, 14).is_wide_leader());
    assert!(s.cell(0, 15).is_wide_spacer());
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 15));
    assert_eq!(s.link_at(0, 14, &mut probe), Some(hit), "the leader");
    assert_eq!(s.link_at(0, 15, &mut probe), Some(hit), "the spacer");
    assert_eq!(probe.url(), "https://x.com/世");
}

#[test]
fn link_at_ignores_a_column_past_the_grid() {
    let mut s = Screen::new(30, 2);
    feed(&mut s, b"https://example.com/a");
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 99, &mut probe), None);
}

#[test]
fn an_anchor_makes_text_that_is_not_a_url_a_link() {
    // The whole point of OSC 8: the label need not look like a URL. No text scan
    // could ever find this one, which is exactly why the child marked it up.
    let mut s = Screen::new(20, 2);
    let mut bytes = anchor("https://example.com/a", "click me");
    bytes.extend_from_slice(b" nope");
    feed(&mut s, &bytes);

    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 7)); // "click me" is 8 cells
    for col in 0..=7 {
        assert_eq!(s.link_at(0, col, &mut probe), Some(hit), "col {col}");
        assert_eq!(probe.url(), "https://example.com/a");
        assert_eq!(probe.verdict(), Some(Verdict::Open));
    }
    assert_eq!(
        s.link_at(0, 8, &mut probe),
        None,
        "the space past the anchor"
    );
    assert_eq!(s.link_at(0, 10, &mut probe), None, "and the text after it");
    assert_eq!(probe.url(), "", "a miss leaves no stale URL behind");
}

#[test]
fn an_anchor_outranks_the_text_underneath_it() {
    // The label happens to be a URL, and a *different* one from the target. The
    // child said where these cells go; the scanner does not get a vote.
    let mut s = Screen::new(40, 2);
    feed(
        &mut s,
        &anchor("https://real.example/x", "https://decoy.example"),
    );
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 3, &mut probe), Some(((0, 0), (0, 20))));
    assert_eq!(probe.url(), "https://real.example/x");
}

/// The verdict stored for the anchor shown at display `(row, col)`, `None` off any anchor.
fn anchor_verdict(s: &Screen, row: usize, col: usize) -> Option<Verdict> {
    s.links.get(s.view_cell(row, col).link).map(|t| t.verdict)
}

#[test]
fn a_refused_anchor_is_offered_as_a_link_with_its_reason() {
    for uri in [
        "javascript:alert(1)",
        "data:text/html,<script>x</script>",
        "vscode://file/etc/passwd",
        "ftp://host/x",
        "-flag-that-xdg-open-would-eat",
    ] {
        let mut s = Screen::new(20, 2);
        feed(&mut s, &anchor(uri, "click"));
        let mut probe = LinkProbe::default();
        assert_eq!(s.link_at(0, 2, &mut probe), Some(((0, 0), (0, 4))), "{uri}");
        assert_eq!(probe.url(), uri);
        assert_eq!(
            probe.verdict(),
            Some(Verdict::Refused(Refusal::Scheme)),
            "{uri}"
        );
    }
}

#[test]
fn a_refused_anchor_labelled_with_a_url_is_not_offered_the_label() {
    // The label is text the child chose. Scanning it would open a URL the anchor never
    // named, and hide that the anchor itself was refused.
    let mut s = Screen::new(40, 2);
    feed(
        &mut s,
        &anchor("javascript:alert(1)", "https://example.com/a"),
    );
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 5, &mut probe), Some(((0, 0), (0, 20))));
    assert_eq!(probe.url(), "javascript:alert(1)");
    assert_eq!(probe.verdict(), Some(Verdict::Refused(Refusal::Scheme)));
}

#[test]
fn a_url_hiding_a_character_is_one_refused_link() {
    for hidden in [
        '\u{200B}',
        '\u{00AD}',
        '\u{034F}',
        '\u{FE0F}',
        '\u{E0100}',
        '\u{202E}',
    ] {
        let url = format!("https://example.com/a{hidden}b");
        let mut s = Screen::new(60, 2);
        feed(&mut s, format!("see {url} ok").as_bytes());
        let mut probe = LinkProbe::default();
        let (start, end) = s.link_at(0, 6, &mut probe).unwrap();
        assert_eq!(probe.url(), url, "{hidden:?}");
        assert_eq!(start, (0, 4));
        assert_eq!(
            s.cell(end.0, end.1).rune,
            'b',
            "{hidden:?}: the link runs on past it"
        );
        assert_eq!(probe.verdict(), Some(Verdict::Refused(Refusal::Hidden)));

        let mut s = Screen::new(60, 2);
        feed(&mut s, &anchor(&url, "docs"));
        assert_eq!(
            anchor_verdict(&s, 0, 0),
            Some(Verdict::Refused(Refusal::Hidden)),
            "{hidden:?} in an anchor"
        );
    }
}

#[test]
fn an_anchor_whose_uri_holds_a_control_character_is_refused() {
    // The parser drops C0 controls inside an OSC, but DEL and a C1 control in UTF-8
    // reach the URI.
    for (stream, url) in [
        (
            &b"\x1b]8;;https://example.com/a\x7fb\x1b\\docs\x1b]8;;\x1b\\"[..],
            "https://example.com/a\u{7F}b",
        ),
        (
            &b"\x1b]8;;https://example.com/a\xc2\x85b\x07docs\x1b]8;;\x07"[..],
            "https://example.com/a\u{85}b",
        ),
    ] {
        let mut s = Screen::new(40, 2);
        feed(&mut s, stream);
        let mut probe = LinkProbe::default();
        assert_eq!(s.link_at(0, 1, &mut probe), Some(((0, 0), (0, 3))));
        assert_eq!(probe.url(), url);
        assert_eq!(probe.verdict(), Some(Verdict::Refused(Refusal::Hidden)));
    }
    // Visible non-ASCII is not a control: `rg --hyperlink-format` prints it unencoded.
    let mut s = Screen::new(40, 2);
    feed(&mut s, &anchor("file:///home/u/файл.txt", "файл.txt"));
    assert_eq!(anchor_verdict(&s, 0, 0), Some(Verdict::Open));
}

#[test]
fn a_link_whose_host_a_browser_would_rewrite_is_refused() {
    let url = "https://github.com@evil.com/x";
    let mut s = Screen::new(60, 2);
    feed(&mut s, format!("see {url} ok").as_bytes());
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 8, &mut probe), Some(((0, 4), (0, 32))));
    assert_eq!(probe.url(), url);
    assert_eq!(probe.verdict(), Some(Verdict::Refused(Refusal::Host)));

    let mut s = Screen::new(60, 2);
    feed(&mut s, &anchor(url, "github.com"));
    assert_eq!(
        anchor_verdict(&s, 0, 0),
        Some(Verdict::Refused(Refusal::Host))
    );
}

#[test]
fn a_file_link_naming_another_machine_is_refused() {
    // `ls --hyperlink` names the machine it ran on, so over ssh its links name the
    // remote host, and opening one here would open a different file.
    let mut s = Screen::new(60, 2);
    s.set_hostname(Hostname::new("here"));
    let mut bytes = anchor("file://there/home/u/a.txt", "a.txt");
    bytes.push(b' ');
    bytes.extend(anchor("file://here/home/u/a.txt", "b.txt"));
    bytes.extend_from_slice(b" file://there/home/u/c.txt");
    feed(&mut s, &bytes);
    assert_eq!(
        anchor_verdict(&s, 0, 0),
        Some(Verdict::Refused(Refusal::ForeignHost))
    );
    assert_eq!(anchor_verdict(&s, 0, 6), Some(Verdict::Open));
    let mut probe = LinkProbe::default();
    assert!(s.link_at(0, 15, &mut probe).is_some());
    assert_eq!(
        probe.verdict(),
        Some(Verdict::Refused(Refusal::ForeignHost)),
        "a bare file: URL too"
    );
}

#[test]
fn a_file_link_whose_path_decodes_more_than_one_way_is_refused() {
    let mut s = Screen::new(60, 2);
    feed(&mut s, &anchor("file:///tmp/payload.desktop%0A", "notes"));
    assert_eq!(
        anchor_verdict(&s, 0, 0),
        Some(Verdict::Refused(Refusal::FilePath))
    );
}

#[test]
fn a_mailto_link_that_attaches_a_file_is_refused() {
    let url = "mailto:a@b.com?subject=hi&attach=/home/u/.ssh/id_ed25519";
    let mut s = Screen::new(80, 2);
    feed(&mut s, format!("mail {url} now").as_bytes());
    let mut probe = LinkProbe::default();
    assert!(s.link_at(0, 8, &mut probe).is_some());
    assert_eq!(probe.url(), url);
    assert_eq!(probe.verdict(), Some(Verdict::Refused(Refusal::MailField)));
}

/// The verdict a lookup at display `(row, col)` reports.
fn lookup_verdict(s: &Screen, row: usize, col: usize) -> Option<Verdict> {
    let mut probe = LinkProbe::default();
    s.link_at(row, col, &mut probe)?;
    probe.verdict()
}

#[test]
fn an_anchor_whose_text_names_another_host_is_refused() {
    let refused = Some(Verdict::Refused(Refusal::LabelHost));
    for (target, text, verdict) in [
        ("https://evil.example/x", "https://github.com/x", refused),
        (
            "https://github.com/y",
            "https://github.com/x",
            Some(Verdict::Open),
        ),
        (
            "https://GitHub.com/y",
            "HTTPS://github.COM/x",
            Some(Verdict::Open),
        ),
        ("https://evil.example/x", "click here", Some(Verdict::Open)),
        (
            "https://evil.example/x",
            "https://github.com@evil.example/",
            refused,
        ),
        ("https://evil.example/x", "  https://github.com/x", refused),
        ("mailto:a@evil.example", "https://github.com/x", refused),
    ] {
        let mut s = Screen::new(60, 2);
        feed(&mut s, &anchor(target, text));
        assert_eq!(lookup_verdict(&s, 0, 4), verdict, "{text:?} over {target}");
    }
}

#[test]
fn an_anchor_text_disguised_from_the_check_is_still_read_as_the_eye_reads_it() {
    let refused = Some(Verdict::Refused(Refusal::LabelHost));
    let evil = "https://evil.example/x";
    for text in [
        // An invisible mark inside the scheme.
        "h\u{200B}ttps://github.com/x",
        "h\u{034F}ttps://github.com/x",
        "https:/\u{2060}/github.com/x",
        "https:\u{200D}//github.com/x",
        // A prefix that draws as a blank.
        "\u{2800}https://github.com/x",
        "\u{FFA0}https://github.com/x",
        // A scheme a browser repairs, or one in fullwidth letters.
        "https:\\github.com/x",
        "https:/github.com/x",
        "ｈｔｔｐｓ://github.com/x",
    ] {
        let mut s = Screen::new(60, 2);
        feed(&mut s, &anchor(evil, text));
        assert_eq!(lookup_verdict(&s, 0, 6), refused, "{text:?}");
    }

    // A concealed cell before the scheme draws as a space.
    let mut s = Screen::new(60, 2);
    feed(
        &mut s,
        &anchor(evil, "\x1b[8mX\x1b[28mhttps://github.com/x"),
    );
    assert_eq!(lookup_verdict(&s, 0, 6), refused, "concealed prefix");
}

#[test]
fn an_anchor_text_is_read_as_one_word_with_the_cells_beside_it() {
    let refused = Some(Verdict::Refused(Refusal::LabelHost));
    let evil = "https://evil.example/x";

    let mut s = Screen::new(60, 2);
    let mut bytes = b"https://".to_vec();
    bytes.extend(anchor(evil, "github.com/x"));
    feed(&mut s, &bytes);
    assert_eq!(
        lookup_verdict(&s, 0, 10),
        refused,
        "a scheme printed before it"
    );

    let mut s = Screen::new(60, 2);
    let mut bytes = anchor("https://github.com/", "https://");
    bytes.extend(anchor(evil, "github.com/x"));
    feed(&mut s, &bytes);
    assert_eq!(
        lookup_verdict(&s, 0, 10),
        refused,
        "a scheme in the anchor before"
    );

    let mut s = Screen::new(60, 2);
    let mut bytes = anchor("https://github.co/", "https://github.co");
    bytes.extend_from_slice(b"m/login");
    feed(&mut s, &bytes);
    assert_eq!(
        lookup_verdict(&s, 0, 10),
        refused,
        "a host finished after it"
    );
}

#[test]
fn an_anchor_text_naming_its_own_target_opens() {
    for (target, text) in [
        ("file:///etc/hosts", "FILE:///etc/hosts"),
        ("file:///home/u/src/main.rs", "file:12:fn main"),
        ("https://github.com/x", "see https://github.com/x"),
    ] {
        let mut s = Screen::new(60, 2);
        feed(&mut s, &anchor(target, text));
        assert_eq!(lookup_verdict(&s, 0, 1), Some(Verdict::Open), "{text:?}");
    }
}

#[test]
fn an_anchor_text_is_judged_where_it_wraps_and_after_it_is_rewritten() {
    let mut s = Screen::new(12, 3);
    feed(
        &mut s,
        &anchor("https://evil.example/x", "https://github.com/x"),
    );
    let refused = Some(Verdict::Refused(Refusal::LabelHost));
    assert_eq!(lookup_verdict(&s, 1, 2), refused, "across the wrap");

    // The same anchor reopened prints over its own text, which now reads as prose.
    let mut s = Screen::new(40, 2);
    feed(
        &mut s,
        &anchor("https://evil.example/x", "https://github.com/x"),
    );
    assert_eq!(lookup_verdict(&s, 0, 2), refused);
    feed(&mut s, b"\r");
    feed(
        &mut s,
        &anchor("https://evil.example/x", "documentation here  "),
    );
    assert_eq!(lookup_verdict(&s, 0, 2), Some(Verdict::Open));
}

#[test]
fn setting_the_hostname_judges_stored_links_again() {
    let mut s = Screen::new(40, 2);
    feed(&mut s, &anchor("file://here/a", "a"));
    assert_eq!(
        anchor_verdict(&s, 0, 0),
        Some(Verdict::Refused(Refusal::ForeignHost)),
        "an unread hostname names nothing"
    );
    s.set_hostname(Hostname::new("here"));
    assert_eq!(anchor_verdict(&s, 0, 0), Some(Verdict::Open));
}

#[test]
fn a_bare_url_never_runs_into_an_anchor() {
    // Probed from plain text beside an anchor, the scan must not read the anchor's label
    // as part of a bare URL: that opens a URL built from text the anchor owns, and shows
    // a refused anchor's cells as openable.
    let mut probe = LinkProbe::default();
    let mut s = Screen::new(40, 2);
    let mut bytes = anchor("javascript:alert(1)", "https://evil");
    bytes.extend_from_slice(b".example.com/x");
    feed(&mut s, &bytes);
    assert_eq!(s.link_at(0, 15, &mut probe), None, "a refused label's tail");

    let mut s = Screen::new(40, 2);
    let mut bytes = b"https://".to_vec();
    bytes.extend(anchor("javascript:alert(1)", "evil.com"));
    feed(&mut s, &bytes);
    assert_eq!(s.link_at(0, 2, &mut probe), None, "a scheme before a label");

    let mut s = Screen::new(40, 2);
    let mut bytes = b"https://a.com/".to_vec();
    bytes.extend(anchor("https://b.com/", "docs"));
    feed(&mut s, &bytes);
    assert_eq!(s.link_at(0, 3, &mut probe), Some(((0, 0), (0, 13))));
    assert_eq!(
        probe.url(),
        "https://a.com/",
        "an open anchor bounds it too"
    );
}

#[test]
fn an_anchor_with_no_readable_target_leaves_its_text_plain() {
    // Bad UTF-8 or an empty URI has nothing to keep, so it closes the open anchor.
    let mut s = Screen::new(20, 2);
    feed(&mut s, b"\x1b]8;;https://e.com/a\x1b\\ab");
    feed(&mut s, b"\x1b]8;;https://e.com/\xff\x1b\\cd");
    feed(&mut s, b"\x1b]8;;https://e.com/a\x1b\\ef\x1b]8;;\x1b\\gh");
    assert_eq!(anchor_verdict(&s, 0, 0), Some(Verdict::Open));
    assert_eq!(s.cell(0, 2).link, LinkId::NONE, "bad UTF-8");
    assert_eq!(anchor_verdict(&s, 0, 4), Some(Verdict::Open));
    assert_eq!(s.cell(0, 6).link, LinkId::NONE, "an empty URI");
    assert_eq!(s.links.targets.len(), 1);
}

#[test]
fn refused_anchors_count_against_the_link_budgets() {
    // A refused anchor is stored like any other, so a stream of distinct long refused
    // URIs meets the same byte budget and the same fruitless-sweep guard.
    let mut s = Screen::new(20, 2);
    let filler = "b".repeat(LINK_URL_MAX - 32);
    for i in 0..4000 {
        feed(&mut s, &anchor(&format!("javascript:{i}/{filler}"), "z"));
        feed(&mut s, b"\r\n");
    }
    assert!(
        s.links.bytes <= LINK_BYTES_MAX,
        "the table held {} bytes, past its {LINK_BYTES_MAX} budget",
        s.links.bytes
    );
    assert!(s.links.targets.len() < 4000, "the budget stopped it");
    assert!(s.links.swept_dry_at.is_some());

    // An anchor that opens finds no room either, and its text prints plain.
    feed(
        &mut s,
        &anchor(&format!("https://e.com/late/{filler}"), "late"),
    );
    assert_eq!(s.cell(1, 0).link, LinkId::NONE);
    assert_eq!(s.link_at(1, 0, &mut LinkProbe::default()), None);
}

#[test]
fn a_sweep_keeps_each_surviving_link_its_verdict() {
    // Burn the id space on alternately refused and open anchors, most of which scroll
    // out of history, then force a sweep. The survivors are renumbered, and each must
    // keep the verdict of its own URL rather than one shifted from a neighbour.
    let mut s = Screen::new(20, 2);
    let mut bytes = Vec::new();
    for i in 0..LINK_LIMIT {
        let uri = if i % 2 == 0 {
            format!("javascript:{i}")
        } else {
            format!("https://e.com/{i}")
        };
        bytes.extend(anchor(&uri, "x"));
        bytes.extend_from_slice(b"\r\n");
    }
    feed(&mut s, &bytes);
    assert_eq!(
        s.links.targets.len(),
        LINK_LIMIT,
        "refused anchors spent ids too"
    );

    feed(&mut s, &anchor("https://e.com/fresh", "NEW"));
    assert!(s.links.targets.len() < LINK_LIMIT, "the sweep ran");
    for target in &s.links.targets {
        assert_eq!(
            target.verdict,
            verdict::judge(&target.url, &Hostname::default()),
            "{}",
            target.url
        );
    }
    // Row 0 shows the last anchor printed before the fresh one, which was open; the
    // one before it, a line up in history, was refused.
    assert_eq!(anchor_verdict(&s, 0, 0), Some(Verdict::Open));
    s.scroll_view_up(1);
    assert_eq!(
        anchor_verdict(&s, 0, 0),
        Some(Verdict::Refused(Refusal::Scheme))
    );
}

#[test]
fn an_sgr_reset_does_not_close_an_anchor() {
    // `ls --hyperlink` with colors sets a color inside the anchor and emits SGR 0
    // after the name while *still* inside it. If a rendition reset closed the link,
    // the most widespread OSC 8 producer there is would half-work. The link rides
    // on the pen but is not an SGR attribute; only `OSC 8 ; ;` closes it.
    let mut s = Screen::new(20, 2);
    feed(
        &mut s,
        b"\x1b]8;;https://example.com/d\x1b\\\x1b[34mdir\x1b[0m/\x1b]8;;\x1b\\",
    );
    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 3, &mut probe),
        Some(((0, 0), (0, 3))),
        "the `/` printed after the reset is still inside the anchor"
    );
    assert_eq!(probe.url(), "https://example.com/d");
    assert_eq!(s.cell(0, 0).fg, Color::Ansi(4), "the color still applied");
    assert_eq!(
        s.cell(0, 3).fg,
        Color::Default,
        "and the reset still reset it"
    );
}

#[test]
fn an_anchor_joins_across_a_soft_wrap() {
    // Ten label cells on an eight-column grid: the anchor is one link across the
    // margin, not two, exactly as a wrapped bare URL is.
    let mut s = Screen::new(8, 3);
    feed(&mut s, &anchor("https://example.com/w", "abcdefghij"));
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (1, 1));
    assert_eq!(s.link_at(0, 4, &mut probe), Some(hit), "before the wrap");
    assert_eq!(s.link_at(1, 1, &mut probe), Some(hit), "after it");
    assert_eq!(probe.url(), "https://example.com/w");
}

#[test]
fn two_anchors_sharing_one_url_stay_two_links() {
    // Interning by URL hands both runs the same id. The *run* is what bounds a
    // link, so the plain cell between them still splits it into two hover targets.
    let mut s = Screen::new(20, 2);
    let mut bytes = anchor("https://e.com/a", "ab");
    bytes.extend_from_slice(b" ");
    bytes.extend(anchor("https://e.com/a", "cd"));
    feed(&mut s, &bytes);

    assert!(s.cell(0, 0).link.is_set());
    assert_eq!(s.cell(0, 0).link, s.cell(0, 3).link, "interned once");
    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 0, &mut probe),
        Some(((0, 0), (0, 1))),
        "the first"
    );
    assert_eq!(
        s.link_at(0, 4, &mut probe),
        Some(((0, 3), (0, 4))),
        "the second"
    );
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the gap between them");
}

#[test]
fn an_anchor_covers_both_halves_of_a_wide_glyph() {
    // The spacer carries the link too, so the run never breaks mid-character and
    // probing the right half of a wide glyph finds the glyph's link.
    let mut s = Screen::new(20, 2);
    feed(&mut s, &anchor("https://e.com/w", "世a"));
    assert!(s.cell(0, 0).is_wide_leader());
    assert!(s.cell(0, 1).is_wide_spacer());
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 2));
    assert_eq!(s.link_at(0, 0, &mut probe), Some(hit), "the leader");
    assert_eq!(s.link_at(0, 1, &mut probe), Some(hit), "the spacer");
    assert_eq!(probe.url(), "https://e.com/w");
}

#[test]
fn erasing_a_cell_takes_it_out_of_its_anchor() {
    // Erased ground is inside no link (`blank_cell` carries `LinkId::NONE`), so the
    // run breaks there and the halves become separate links — the honest reading
    // once the label is no longer contiguous.
    let mut s = Screen::new(20, 2);
    feed(&mut s, &anchor("https://e.com/a", "abcde"));
    feed(&mut s, b"\x1b[1;3H\x1b[X"); // ECH one cell at col 2
    assert_eq!(s.cell(0, 2).link, LinkId::NONE);

    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 0, &mut probe),
        Some(((0, 0), (0, 1))),
        "the left half"
    );
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the erased cell");
    assert_eq!(
        s.link_at(0, 4, &mut probe),
        Some(((0, 3), (0, 4))),
        "the right half"
    );
}

#[test]
fn the_id_param_is_ignored_and_an_empty_uri_closes() {
    // `id=` is a hint that two runs are one link. We intern by URL, which already
    // gives identical URLs one identity, so the hint buys nothing — and honouring a
    // child-chosen id would let it fuse unrelated runs into one hover target.
    let mut s = Screen::new(20, 2);
    feed(
        &mut s,
        b"\x1b]8;id=xyz;https://e.com/a\x1b\\ab\x1b]8;;\x1b\\cd",
    );
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 1, &mut probe), Some(((0, 0), (0, 1))));
    assert_eq!(probe.url(), "https://e.com/a");
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the empty URI closed it");
}

#[test]
fn a_full_link_table_collects_dead_ids_and_keeps_linking() {
    // Burn the whole u16 id space on anchors that then scroll out of history. The
    // next one must still work: without the collector the table would stay full for
    // the life of the process and OSC 8 would quietly stop linking anything.
    let mut s = Screen::new(20, 2);
    let mut bytes = Vec::new();
    for i in 0..LINK_LIMIT {
        bytes.extend(anchor(&format!("https://e.com/{i}"), "x"));
        bytes.extend_from_slice(b"\r\n");
    }
    feed(&mut s, &bytes);
    assert_eq!(s.links.targets.len(), LINK_LIMIT, "the id space is spent");

    // Row 0 shows the last anchor printed; row 1 is where the cursor now sits.
    let last = format!("https://e.com/{}", LINK_LIMIT - 1);
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 0, &mut probe), Some(((0, 0), (0, 0))));
    assert_eq!(probe.url(), last);

    // One more distinct URL: intern finds the table full, collects, and succeeds.
    feed(&mut s, &anchor("https://e.com/fresh", "NEW"));
    assert!(
        s.links.targets.len() < LINK_LIMIT,
        "the ids of the scrolled-out anchors came back"
    );
    assert_eq!(s.link_at(1, 1, &mut probe), Some(((1, 0), (1, 2))));
    assert_eq!(probe.url(), "https://e.com/fresh");

    // And the survivors were *renumbered*, not orphaned: a cell that outlived the
    // collection still resolves to the URL it always had.
    assert_eq!(s.link_at(0, 0, &mut probe), Some(((0, 0), (0, 0))));
    assert_eq!(
        probe.url(),
        last,
        "a survivor kept its URL across the collection"
    );
}

#[test]
fn the_link_table_is_bounded_in_bytes_and_not_only_in_count() {
    // Capping the *count* is not a bound on memory, because the child picks the length
    // too. A URL's only other limit is `OSC_MAX` (4096), and each is stored twice, so
    // 65534 distinct ~4 KB URIs — one anchored cell each, all live, nothing for
    // `collect_links` to reclaim — pin roughly 537 MB. That is forty times what the
    // scrollback's own cells cost, from a child that merely prints.
    let mut s = Screen::new(20, 2);

    // A URL past the per-URL cap is not clickable, and does not grow the table at all.
    let huge = format!("https://e.com/{}", "a".repeat(LINK_URL_MAX));
    feed(&mut s, &anchor(&huge, "x"));
    assert!(s.links.targets.is_empty(), "nothing was interned");
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 0, &mut probe), None, "and the cell is plain");
    assert_eq!(s.cell(0, 0).rune, 'x', "the text still reads, as always");

    // One byte under it still works, so the cap is where it says it is.
    let big = format!("https://e.com/{}", "a".repeat(LINK_URL_MAX - 15));
    assert_eq!(big.len(), LINK_URL_MAX - 1);
    feed(&mut s, &anchor(&big, "y"));
    assert_eq!(s.links.targets.len(), 1);
    assert_eq!(s.link_at(0, 1, &mut probe), Some(((0, 1), (0, 1))));
    assert_eq!(probe.url(), big);

    // And the table's own byte budget holds against a stream of distinct long URLs,
    // whatever the id space says. Every one of these is live (each has a cell, and the
    // 2-row screen keeps scrolling them into a 10k-line history), so a sweep frees
    // nothing and the budget is the only thing standing between the child and memory.
    let mut s = Screen::new(20, 2);
    let filler = "b".repeat(LINK_URL_MAX - 32);
    for i in 0..4000 {
        feed(&mut s, &anchor(&format!("https://e.com/{i}/{filler}"), "z"));
        feed(&mut s, b"\r\n");
    }
    assert!(
        s.links.bytes <= LINK_BYTES_MAX,
        "the table held {} bytes, past its {LINK_BYTES_MAX} budget",
        s.links.bytes
    );
    assert!(
        s.links.targets.len() < 4000,
        "the budget, not the id space, is what stopped it"
    );

    // And a full table does not buy a mark-and-sweep per anchor. Every intern past the
    // budget fails, and each failure used to ask for a walk over every cell in both
    // buffers — 800k cells, per 2 KB of child output. The guard records that a sweep
    // came back empty and declines to repeat it until a row has aged out or the epoch
    // has broken.
    assert!(
        s.links.swept_dry_at.is_some(),
        "the table filled without the fruitless-sweep guard ever engaging"
    );
}
