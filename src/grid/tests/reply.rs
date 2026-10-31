//! Replies: what the terminal answers (DA, DSR, DECRQM, DECRQSS, XTGETTCAP, colour
//! queries, focus and size reports), and the bound on how much it will owe.

use crate::color::Rgb;
use crate::grid::tests::print_str;
use crate::grid::*;
use crate::testutil::feed;

/// A colour the terminal was told to use is the colour it reports back.
///
/// The same self-oracle shape as the DECRQSS one below, over the other query path: a
/// theme-aware program (every modern editor) sets a palette entry, asks what it is,
/// and paints to match. No expected string is written here — the property is that
/// `set` and `query` agree, and it holds for every colour, every index, and both
/// terminators.
#[test]
fn a_colour_query_answers_with_the_colour_that_was_set() {
    let mut s = Screen::new(4, 1);
    for index in [0u16, 7, 8, 15, 16, 128, 196, 255] {
        for rgb in [
            Rgb::new(0, 0, 0),
            Rgb::new(255, 255, 255),
            Rgb::new(0x12, 0x34, 0x56),
            Rgb::new(1, 2, 3),
        ] {
            let mut set = Vec::new();
            set.extend_from_slice(format!("\x1b]4;{index};").as_bytes());
            crate::color::write_x11_color(rgb, &mut set);
            set.push(0x07);
            feed(&mut s, &set);
            s.take_responses();

            feed(&mut s, format!("\x1b]4;{index};?\x07").as_bytes());
            let reply = s.take_responses();
            let text = String::from_utf8_lossy(&reply).to_string();
            let body = text
                .strip_prefix(&format!("\u{1b}]4;{index};"))
                .and_then(|t| t.strip_suffix('\u{7}'))
                .unwrap_or_else(|| panic!("index {index}: bad reply {text:?}"));
            assert_eq!(
                crate::color::parse_x11_color(body.as_bytes()),
                Some(rgb),
                "index {index} was set to {rgb:?} and reported {body:?}"
            );
        }
    }

    // The reply mirrors the terminator it was asked with: a client that sent BEL may
    // only be listening for BEL, and one that sent ST for ST.
    feed(&mut s, b"\x1b]4;9;?\x1b\\");
    let reply = s.take_responses();
    assert!(reply.ends_with(b"\x1b\\"), "ST asked, ST answered");
    assert!(!reply.contains(&0x07));
    feed(&mut s, b"\x1b]4;9;?\x07");
    let reply = s.take_responses();
    assert!(reply.ends_with(&[0x07]), "BEL asked, BEL answered");
}

/// DECRQSS answers with a sequence that *reproduces* the setting, so the answer can
/// be checked against itself: ask, replay the reply into a fresh screen, and the two
/// must end up in the same state.
///
/// A self-oracle, and worth more than a table of expected strings would be. Nobody
/// writes down what the reply "should" say, so nobody can write down something wrong
/// and bless it; the property is that a program which saves a setting, changes it,
/// and puts it back gets what it had — which is the entire reason DECRQSS exists.
#[test]
fn a_decrqss_answer_reproduces_the_state_it_reports() {
    for sgr in [
        &b""[..],
        b"\x1b[1m",
        b"\x1b[1;4;31m",
        b"\x1b[3;7;9m",
        b"\x1b[38;2;255;0;128m",
        b"\x1b[48;5;196m",
        b"\x1b[4:3m",
        b"\x1b[4:2;38;5;9;48;2;1;2;3m",
        b"\x1b[0m",
    ] {
        // Set the pen and print through it, so the cell carries what the pen was.
        let mut asked = Screen::new(4, 1);
        feed(&mut asked, sgr);
        feed(&mut asked, b"\x1bP$qm\x1b\\");
        let reply = asked.take_responses();
        print_str(&mut asked, "x");

        // The reply is `DCS 1 $ r <sgr> ST`; take the <sgr> body out of it.
        let text = String::from_utf8_lossy(&reply).to_string();
        let body = text
            .strip_prefix("\u{1b}P1$r")
            .and_then(|t| t.strip_suffix("\u{1b}\\"))
            .unwrap_or_else(|| panic!("{sgr:?}: not a valid DECRQSS answer: {text:?}"));

        // Replay it into a screen that has never seen the original sequence.
        let mut replayed = Screen::new(4, 1);
        feed(&mut replayed, format!("\u{1b}[{body}").as_bytes());
        print_str(&mut replayed, "x");

        assert_eq!(
            replayed.cell(0, 0),
            asked.cell(0, 0),
            "{sgr:?} was reported as {body:?}, which does not reproduce it"
        );
    }
}

#[test]
fn device_attributes_and_status_queries_get_answers() {
    let mut s = Screen::new(80, 24);
    // Primary DA.
    feed(&mut s, b"\x1b[c");
    assert_eq!(s.take_responses(), b"\x1b[?1;2c");
    assert!(s.take_responses().is_empty(), "the queue drains");
    // Secondary DA.
    feed(&mut s, b"\x1b[>c");
    assert_eq!(s.take_responses(), b"\x1b[>0;0;0c");
    // DSR status: OK.
    feed(&mut s, b"\x1b[5n");
    assert_eq!(s.take_responses(), b"\x1b[0n");
}

#[test]
fn xtversion_says_who_we_are() {
    // It gets us onto nobody's allowlist — that is not what it is for. It is what a
    // terminal is supposed to say when asked, and silence is how you stay unknown.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[>0q");
    let reply = s.take_responses();
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(text.starts_with("\x1bP>|bnkterm "), "{text:?}");
    assert!(text.ends_with("\x1b\\"), "{text:?}");
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn decrqss_answers_with_the_sequence_that_would_restore_the_setting() {
    // A program saves a setting, changes it, and puts it back without ever knowing
    // what it was, so the reply is a sequence to replay, not a description.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"\x1bP$qm\x1b\\"); // "what is SGR?"
    assert_eq!(s.take_responses(), b"\x1bP1$r0m\x1b\\");

    feed(&mut s, b"\x1b[1;4;38;5;200m"); // bold, underlined, colour 200
    feed(&mut s, b"\x1bP$qm\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP1$r0;1;4;38;5;200m\x1b\\");

    // The scroll region, 1-based like the sequence that sets it.
    feed(&mut s, b"\x1b[2;3r");
    feed(&mut s, b"\x1bP$qr\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP1$r2;3r\x1b\\");

    // A setting we do not report is answered "not supported" — a real answer, so the
    // program stops waiting. Claiming support and inventing a value would be worse
    // than silence: it would be restored verbatim.
    feed(&mut s, b"\x1bP$q\"p\x1b\\"); // DECSCL
    assert_eq!(s.take_responses(), b"\x1bP0$r\x1b\\");
}

#[test]
fn xtgettcap_answers_what_our_terminfo_would_have_said() {
    // The query that lets a program learn what we can do *without* a terminfo entry
    // installed for us — which matters, because we ship none and claim to be xterm.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1bP+q544e\x1b\\"); // "TN" — terminal name
    assert_eq!(s.take_responses(), b"\x1bP1+r544e=626e6b7465726d\x1b\\");

    feed(&mut s, b"\x1bP+q524742\x1b\\"); // "RGB" — do you mean it about truecolor?
    assert_eq!(s.take_responses(), b"\x1bP1+r524742=382f382f38\x1b\\");

    // Several at once, and an unknown one answered 0 rather than ignored.
    feed(&mut s, b"\x1bP+q436f;7a7a7a\x1b\\"); // "Co" (colors), then "zzz"
    assert_eq!(
        s.take_responses(),
        b"\x1bP1+r436f=323536\x1b\\\x1bP0+r7a7a7a\x1b\\"
    );

    // Not hex at all: still an answer, still not a guess.
    feed(&mut s, b"\x1bP+qnothex\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP0+r\x1b\\");

    // This reply is the *only* one in the terminal that puts child-supplied bytes back on
    // the pty (the name returns verbatim, because it is already hex on the wire). What
    // makes that safe is `hex_decode` refusing anything but hex digits *before* the echo,
    // so a name carrying a control byte is answered without being repeated. Asserted here
    // rather than argued in prose, because it is the property that would
    // silently stop holding if that validator were ever loosened.
    for hostile in [
        &b"\x1bP+q54\x074e\x1b\\"[..],     // BEL inside the name
        &b"\x1bP+q54\x0d\x0a4e\x1b\\"[..], // CR LF inside the name
        &b"\x1bP+q544\x1b\\"[..],          // odd length, so half a byte
    ] {
        feed(&mut s, hostile);
        let reply = s.take_responses();
        assert_eq!(
            reply, b"\x1bP0+r\x1b\\",
            "refused without echo: {hostile:?}"
        );
        assert!(
            !reply.iter().any(|b| matches!(b, 0x00..=0x06 | 0x08..=0x1a)),
            "no control byte from the request reached the reply: {reply:?}"
        );
    }
}

#[test]
fn a_dcs_we_do_not_answer_is_swallowed_whole() {
    // Sixel arrives here (`DCS q …`), and so does anything else we have no use for.
    // The payload must never leak onto the screen as text.
    let mut s = Screen::new(20, 2);
    feed(&mut s, b"ab\x1bPq#0;2;0;0;0#0~~@@vv@@~~@@~~$\x1b\\cd");
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    assert!(s.take_responses().is_empty());
}

#[test]
fn in_band_resize_reports_only_to_a_program_that_asked() {
    let mut s = Screen::new(80, 24);
    s.set_pixel_size(640, 384);

    // Nobody asked: a resize says nothing. A program that never enabled `?2048` would
    // read the report as garbage on its input.
    s.resize(100, 30);
    assert!(s.take_responses().is_empty());

    // Enabling it reports immediately, which is the point: a program handed a fresh
    // pty has no other way to ask how big it is.
    feed(&mut s, b"\x1b[?2048h");
    assert_eq!(s.take_responses(), b"\x1b[48;30;100;384;640t");

    // And every resize after it. This is the news SIGWINCH cannot carry across an ssh
    // hop or a tmux, because the signal stops at this side of the pty.
    s.set_pixel_size(800, 400);
    s.resize(120, 40);
    s.report_size();
    assert_eq!(s.take_responses(), b"\x1b[48;40;120;400;800t");

    feed(&mut s, b"\x1b[?2048l");
    s.report_size();
    assert!(s.take_responses().is_empty(), "and it can be turned off");
}

#[test]
fn focus_events_are_reported_only_to_a_program_that_asked() {
    // A program that never enabled `?1004` would read `CSI I` as a stray key press,
    // so silence is not politeness here, it is correctness.
    let mut s = Screen::new(10, 2);
    s.report_focus(true);
    s.report_focus(false);
    assert!(s.take_responses().is_empty(), "nobody asked");

    feed(&mut s, b"\x1b[?1004h");
    s.report_focus(true);
    assert_eq!(s.take_responses(), b"\x1b[I");
    s.report_focus(false);
    assert_eq!(s.take_responses(), b"\x1b[O");

    // And it can be turned back off.
    feed(&mut s, b"\x1b[?1004l");
    s.report_focus(true);
    assert!(s.take_responses().is_empty());
}

#[test]
fn every_reply_near_the_cap_is_whole_or_absent() {
    // The reply budget is checked once per answer: one that starts under the cap is
    // written whole, and past it nothing is written. Several answers are built in pieces,
    // and the cap used to be checked per piece, so `CSI ? 7 $ p` near the cap queued
    // `\x1b[?7;1` without its `$y`, and an OSC colour query past it queued its body with
    // no header. Each expected answer is what a fresh screen says to the same query.
    let queries: &[&[u8]] = &[
        b"\x1b[?7$p",         // DECRQM
        b"\x1b[6n",           // CPR
        b"\x1b[?6n",          // DECXCPR
        b"\x1b[>0q",          // XTVERSION
        b"\x1bP$qm\x1b\\",    // DECRQSS, SGR
        b"\x1bP$qr\x1b\\",    // DECRQSS, scroll region
        b"\x1bP+q544e\x1b\\", // XTGETTCAP, TN
        b"\x1b]11;?\x07",     // OSC 11
        b"\x1b]4;1;?\x1b\\",  // OSC 4
        b"\x1b[?u",           // kitty keyboard query
    ];
    for &query in queries {
        let query_text = String::from_utf8_lossy(query);
        let query = &query_text;
        let mut fresh = Screen::new(10, 3);
        feed(&mut fresh, query.as_bytes());
        let whole = fresh.take_responses();
        assert!(!whole.is_empty(), "{query:?} has an answer");

        let mut s = Screen::new(10, 3);
        s.responses = vec![b'x'; RESPONSE_MAX - 1];
        feed(&mut s, query.as_bytes());
        assert_eq!(
            String::from_utf8_lossy(&s.responses()[RESPONSE_MAX - 1..]),
            String::from_utf8_lossy(&whole),
            "{query:?} just under the cap is answered whole"
        );

        let mut s = Screen::new(10, 3);
        s.responses = vec![b'x'; RESPONSE_MAX];
        feed(&mut s, query.as_bytes());
        assert_eq!(
            String::from_utf8_lossy(&s.responses()[RESPONSE_MAX..]),
            "",
            "{query:?} at the cap is not answered at all"
        );
    }

    // Several XTGETTCAP names are several answers: the first starts under the cap and goes
    // whole, and the second, starting past it, is dropped whole.
    let mut fresh = Screen::new(10, 3);
    feed(&mut fresh, b"\x1bP+q544e\x1b\\");
    let first = fresh.take_responses();
    let mut s = Screen::new(10, 3);
    s.responses = vec![b'x'; RESPONSE_MAX - 1];
    feed(&mut s, b"\x1bP+q544e;436f\x1b\\");
    assert_eq!(&s.responses()[RESPONSE_MAX - 1..], &first[..]);

    // A kitty push answers nothing, so a spent budget must not drop it.
    let mut s = Screen::new(10, 3);
    s.responses = vec![b'x'; RESPONSE_MAX];
    feed(&mut s, b"\x1b[>1u");
    assert_ne!(s.kitty_flags(), crate::input::KittyFlags::NONE);
}

#[test]
fn a_flood_of_queries_cannot_mint_unbounded_replies() {
    // The child decides how many questions it asks, so reply generation has to be
    // bounded here rather than trusting the far side to drain. A file full of `\x1b[c`
    // is ~21,800 device-attributes queries per 64 KiB, each answered — and the `cat`
    // printing it never reads its stdin, so nothing consumes the answers.
    let mut s = Screen::new(80, 24);
    let flood = b"\x1b[c".repeat(40_000);
    feed(&mut s, &flood);
    // A soft cap by design: the budget is checked before an answer is built, never
    // during, so the last one runs past the line rather than being cut in half. The
    // overshoot is one reply, and the flood it replaces is ~280 KB of answers.
    let len = s.responses().len();
    assert!(len >= RESPONSE_MAX, "the cap is where the flood stopped");
    assert!(
        len < RESPONSE_MAX + 1024,
        "and it overshoots by at most the one answer already under way, got {len}"
    );

    // Capped, not truncated: what is there is whole answers. A half-written reply is a
    // malformed escape sequence in the child's input, which is worse than no reply.
    let one = b"\x1b[?1;2c";
    assert!(
        !s.responses().is_empty(),
        "and the early ones were answered"
    );
    assert_eq!(
        s.responses().len() % one.len(),
        0,
        "the cap fell on an answer boundary"
    );
    assert!(s.responses().chunks(one.len()).all(|c| c == one));

    // Draining resets the budget: it is a per-parse bound, not a lifetime one, so a
    // long-lived shell asking a normal question an hour later still gets an answer.
    s.clear_responses();
    feed(&mut s, b"\x1b[c");
    assert_eq!(s.responses(), one);
}

#[test]
fn decrqm_reports_the_modes_we_have_and_admits_the_ones_we_do_not() {
    // Of the five modes nvim asks about before drawing anything, we now implement all
    // but one. The honest answer for that one is 0 ("I do not know this mode") — never
    // 2, which would mean "I know it and it is off" and invite nvim to turn it on and
    // then rely on it.
    let mut s = Screen::new(10, 2);
    // `?2031` (colour-scheme change notification) is the one left, and we say so.
    feed(&mut s, b"\x1b[?2031$p");
    assert_eq!(s.take_responses(), b"\x1b[?2031;0$y", "not ours to claim");

    // The ones we do implement report their real state, and follow it as it changes.
    feed(&mut s, b"\x1b[?7$p"); // DECAWM, on by default
    assert_eq!(s.take_responses(), b"\x1b[?7;1$y");
    feed(&mut s, b"\x1b[?7l\x1b[?7$p"); // turn it off, ask again
    assert_eq!(s.take_responses(), b"\x1b[?7;2$y");

    feed(&mut s, b"\x1b[?2004$p"); // bracketed paste, off by default
    assert_eq!(s.take_responses(), b"\x1b[?2004;2$y");
    feed(&mut s, b"\x1b[?2004h\x1b[?2004$p");
    assert_eq!(s.take_responses(), b"\x1b[?2004;1$y");

    // Synchronized output: we *do* implement it, so it reports honestly rather than
    // 0 — which is the whole point of answering, since a mode nobody can discover is
    // a mode nobody will ever use.
    feed(&mut s, b"\x1b[?2026$p");
    assert_eq!(s.take_responses(), b"\x1b[?2026;2$y");
    feed(&mut s, b"\x1b[?2026h\x1b[?2026$p");
    assert_eq!(s.take_responses(), b"\x1b[?2026;1$y");

    // Same for the rest of what this round added, which is the whole point of keeping
    // the two tables in step: a mode we act on is a mode we report.
    feed(&mut s, b"\x1b[?1004$p");
    assert_eq!(s.take_responses(), b"\x1b[?1004;2$y");
    feed(&mut s, b"\x1b[?2048$p");
    assert_eq!(s.take_responses(), b"\x1b[?2048;2$y");
    // Enabling `?2048` also reports the size, so drain that before asking again.
    feed(&mut s, b"\x1b[?2048h");
    let _ = s.take_responses();
    feed(&mut s, b"\x1b[?2048$p");
    assert_eq!(s.take_responses(), b"\x1b[?2048;1$y");

    // The ANSI space answers on its own form, without the `?`.
    feed(&mut s, b"\x1b[4$p"); // IRM
    assert_eq!(s.take_responses(), b"\x1b[4;2$y");
    feed(&mut s, b"\x1b[4h\x1b[4$p");
    assert_eq!(s.take_responses(), b"\x1b[4;1$y");
    // And an ANSI mode we do not know is 0, like any other.
    feed(&mut s, b"\x1b[20$p"); // LNM
    assert_eq!(s.take_responses(), b"\x1b[20;0$y");
}

#[test]
fn every_mode_reports_what_setting_and_resetting_it_did() {
    // DECRQM must agree with `h` and `l` for every mode the terminal acts on: a mode it
    // sets but does not report is never discovered, and one it reports but ignores is a
    // lie a program will build on. `?1048` is left out: its reset is a cursor restore,
    // which keeps the save, so it stays reported as set (its own test covers it).
    let modes: &[(&str, u16)] = &[
        ("", 4),
        ("?", 1),
        ("?", 6),
        ("?", 7),
        ("?", 12),
        ("?", 25),
        ("?", 47),
        ("?", 1000),
        ("?", 1002),
        ("?", 1003),
        ("?", 1004),
        ("?", 1006),
        ("?", 1047),
        ("?", 1049),
        ("?", 2004),
        ("?", 2026),
        ("?", 2027),
        ("?", 2048),
    ];
    for &(prefix, mode) in modes {
        let mut s = Screen::new(10, 4);
        for (action, state) in [("h", 1), ("l", 2)] {
            feed(&mut s, format!("\x1b[{prefix}{mode}{action}").as_bytes());
            // `?2048h` answers with a size report; only the DECRPM is under test.
            s.take_responses();
            feed(&mut s, format!("\x1b[{prefix}{mode}$p").as_bytes());
            assert_eq!(
                String::from_utf8(s.take_responses()).unwrap(),
                format!("\x1b[{prefix}{mode};{state}$y"),
                "{prefix}{mode}{action}"
            );
        }
    }
}

#[test]
fn osc_11_answers_what_the_background_actually_is() {
    // The reply nvim and helix wait for. They ask whether they are sitting on a dark
    // or a light background and choose an entire colour scheme from the answer; a
    // terminal that says nothing does not get a default, it gets a guess.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;?\x07");
    // The default background is #282c34, in the 16-bit-per-channel form xterm uses.
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x07");

    // Foreground and cursor answer on their own numbers.
    feed(&mut s, b"\x1b]10;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]10;rgb:ffff/ffff/ffff\x07");
    feed(&mut s, b"\x1b]12;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]12;rgb:ffff/0000/7878\x07");
}

#[test]
fn a_color_reply_uses_the_terminator_it_was_asked_with() {
    // A client that asked with ST may not be listening for BEL, and vice versa.
    // Mirroring the request is the one answer that is right for both.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;?\x1b\\"); // asked with ST
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x1b\\");
    feed(&mut s, b"\x1b]11;?\x07"); // asked with BEL
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x07");
}

#[test]
fn the_kitty_query_answers_with_the_flags_really_in_force() {
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?0u");
    feed(&mut s, b"\x1b[>1u\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?1u");
    // An application asking for everything (0b11111) is told exactly what it will
    // get: disambiguate and event types, which we implement, and not alternate keys,
    // all-keys-as-escape-codes or associated text, which we do not. Reporting those as
    // set would leave it waiting for reports that never come.
    feed(&mut s, b"\x1b[>31u\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?3u");
    assert_eq!(
        s.kitty_flags(),
        KittyFlags::DISAMBIGUATE | KittyFlags::REPORT_EVENT_TYPES
    );
}

#[test]
fn cursor_position_report_is_one_based() {
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[3;10H"); // move to row 3, col 10 (1-based)
    feed(&mut s, b"\x1b[6n"); // request CPR
    assert_eq!(s.take_responses(), b"\x1b[3;10R");
    // The extended (?6n) form carries a trailing page number.
    feed(&mut s, b"\x1b[?6n");
    assert_eq!(s.take_responses(), b"\x1b[?3;10;1R");
}
