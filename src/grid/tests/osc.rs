//! What the child sets out of band: the title, the working directory, prompt marks, the
//! palette, the clipboard and program status.

use crate::color::{Ground, Rgb};
use crate::grid::*;
use crate::program_status::State;
use crate::testutil::feed;

#[test]
fn osc_7_takes_the_shells_word_for_the_directory() {
    let mut s = Screen::new(10, 2);
    assert_eq!(s.reported_cwd(), None);
    feed(&mut s, b"\x1b]7;file://host/home/testuser/projects\x07");
    assert_eq!(s.reported_cwd(), Some("/home/testuser/projects"));

    // A filename may hold any byte a filename may hold, which is why the path is
    // encoded at all: this one has a space and a semicolon in it, and the semicolon
    // would otherwise have looked like the end of the field.
    feed(&mut s, b"\x1b]7;file://host/tmp/a%20b%3Bc\x07");
    assert_eq!(s.reported_cwd(), Some("/tmp/a b;c"));

    // A bare path (some shells send one) is accepted; a stray `%` is a literal `%`,
    // because a directory that is merely unusual must not be mangled.
    feed(&mut s, b"\x1b]7;/plain/path\x07");
    assert_eq!(s.reported_cwd(), Some("/plain/path"));
    feed(&mut s, b"\x1b]7;file://h/100%pure\x07");
    assert_eq!(s.reported_cwd(), Some("/100%pure"));
}

#[test]
fn osc_133_marks_where_each_command_began_and_how_it_ended() {
    // The shell tells the terminal what it could never work out for itself: where one
    // command ended and the next began. Everything good downstream falls out of this.
    let mut s = Screen::new(20, 6);
    feed(&mut s, b"\x1b]133;A\x07$ ls\r\n"); // a prompt on row 0
    feed(&mut s, b"\x1b]133;C\x07"); // output starts on row 1
    feed(&mut s, b"a.txt\r\n");
    feed(&mut s, b"\x1b]133;D;0\x07"); // and it succeeded

    feed(&mut s, b"\x1b]133;A\x07$ false\r\n"); // a second prompt on row 2
    feed(&mut s, b"\x1b]133;C\x07\x1b]133;D;1\x07"); // which failed

    let prompts = s.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].row, s.abs_row(0));
    assert_eq!(prompts[0].output, Some(s.abs_row(1)));
    assert_eq!(prompts[0].exit, Some(0), "ls worked");
    assert_eq!(prompts[1].row, s.abs_row(2));
    assert_eq!(prompts[1].exit, Some(1), "false did not");
}

#[test]
fn the_exit_code_belongs_to_the_command_that_just_ended() {
    // These are the bytes a real bash emits, in the order it emits them. The ordering
    // is the subtlety: `D` reports the status of the command that *just finished*, and
    // it arrives immediately before the `A` that starts the next prompt — so the code
    // attaches to the prompt behind it, never the one in front.
    let mut s = Screen::new(20, 6);
    feed(&mut s, b"\x1b]133;A\x07\x1b]7;file://h/tmp\x07$ false\r\n");
    feed(&mut s, b"\x1b]133;D;1\x07"); // `false` failed...
    feed(&mut s, b"\x1b]133;A\x07$ "); // ...and only now does the next prompt open

    let prompts = s.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].exit, Some(1), "the failure landed on `false`");
    assert_eq!(prompts[1].exit, None, "and not on the prompt after it");
    assert_eq!(s.reported_cwd(), Some("/tmp"));
}

#[test]
fn a_prompt_redrawn_in_place_is_still_the_same_prompt() {
    // zsh with syntax highlighting redraws its prompt on every keystroke, re-marking
    // the same row each time. Counting those as separate prompts would fill the list
    // with hundreds of copies of the one you are typing at.
    let mut s = Screen::new(20, 4);
    for _ in 0..50 {
        feed(&mut s, b"\x1b]133;A\x07");
    }
    assert_eq!(s.prompts().len(), 1);
}

#[test]
fn a_flood_of_prompts_stays_bounded_and_keeps_the_newest() {
    // A shell in a loop can mark a prompt on every line forever. The list must stop
    // growing, and what it forgets must be the oldest marks.
    let mut s = Screen::new(20, 4);
    for _ in 0..PROMPT_LIMIT * 2 {
        feed(&mut s, b"\x1b]133;A\x07\r\n");
    }
    feed(&mut s, b"\x1b]133;A\x07");

    let rows: Vec<u64> = s.prompts().iter().map(|p| p.row.0).collect();
    assert!(rows.len() <= PROMPT_LIMIT, "{} marks kept", rows.len());
    assert!(rows.len() > PROMPT_LIMIT / 2, "{} marks kept", rows.len());
    assert_eq!(
        rows.last(),
        Some(&s.cursor_abs_row().0),
        "the newest mark is kept"
    );
    assert!(
        rows.windows(2).all(|w| w[1] == w[0] + 1),
        "the kept marks are the newest run, with none missing"
    );
}

#[test]
fn a_prompt_is_marked_on_the_cursor_row_even_while_the_view_is_scrolled_back() {
    // The shell marks a prompt when it prints one, and that is wherever the *cursor*
    // is — which has nothing to do with what the user happens to be looking at. Run a
    // slow command, scroll back to read while it works, and the prompt it prints on
    // finishing must still be marked on the row it was printed on. Reading the mark
    // off the display instead lands it `view_offset` rows up in history, on a line the
    // command printed, which then misdirects both the prompt jump and the reflow's
    // freeze point.
    let mut s = Screen::with_scrollback(20, 4, 100);
    feed(&mut s, b"\x1b]133;A\x07$ slow\r\n");
    feed(&mut s, b"\x1b]133;C\x07");
    for i in 0..10 {
        feed(&mut s, format!("out {i}\r\n").as_bytes());
    }
    s.scroll_view_up(5);
    assert_eq!(s.view_offset(), 5, "the user is reading history");

    feed(&mut s, b"\x1b]133;D;0\x07\x1b]133;A\x07$ ");
    let live = s
        .primary
        .abs_of(s.primary.scrollback.len() + s.primary.cursor.row);
    assert_eq!(
        s.prompts().last().map(|p| p.row),
        Some(live),
        "the mark names the row the prompt was printed on"
    );
}

#[test]
fn jumping_to_a_prompt_scrolls_past_everything_the_command_printed() {
    // The payoff, and the thing a scrollbar can never do: a scrollbar knows how far you
    // have come, this knows *what* you have come past. Ten screens of build log go by
    // in one keystroke, because the terminal knows where the command that printed them
    // started.
    let mut s = Screen::new(20, 4);
    feed(&mut s, b"\x1b]133;A\x07$ build\r\n");
    for i in 0..30 {
        feed(&mut s, format!("log line {i}\r\n").as_bytes());
    }
    feed(&mut s, b"\x1b]133;A\x07$ ");

    let first = s.prompts()[0].row;
    assert!(s.view_offset() == 0, "we are at the live bottom");
    assert!(s.scroll_to_prompt(true), "there is a prompt behind us");
    assert_eq!(s.abs_row(0), first, "and it is now the top line on screen");
    // Read through the *view*, not the live screen: the whole point is that we are
    // looking at history, and `row_string` would show what is live under it.
    let seen: String = (0..20).map(|c| s.view_cell(0, c).rune).collect();
    assert_eq!(seen.trim_end(), "$ build");

    // Nothing further back to find.
    assert!(!s.scroll_to_prompt(true));
    // And forward again, to the prompt we left. It sits near the live bottom, so it
    // cannot be lifted to the top row — there is not enough text under it to fill the
    // screen — and it comes to rest as high as the history allows. What matters is that
    // it is on screen.
    let last = s.prompts()[1].row;
    assert!(s.scroll_to_prompt(false));
    assert!(
        s.display_row(last).is_some(),
        "the prompt we jumped to is visible"
    );
}

#[test]
fn marks_are_dropped_when_the_rows_they_name_stop_meaning_anything() {
    // They hold row ids, and a reset renumbers the world. Offering to jump to a prompt
    // that is no longer there is worse than offering nothing — it is the same rule the
    // selection follows, for the same reason.
    let mut s = Screen::new(20, 4);
    feed(&mut s, b"\x1b]133;A\x07$ ls\r\n");
    assert_eq!(s.prompts().len(), 1);
    feed(&mut s, b"\x1b[2J"); // a full erase ends the epoch
    assert!(s.prompts().is_empty());
}

#[test]
fn osc_sets_title_through_the_parser() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b]0;my title\x07");
    assert_eq!(s.title(), "my title");
    feed(&mut s, b"\x1b]2;other\x1b\\");
    assert_eq!(s.title(), "other");
}

#[test]
fn running_a_full_screen_program_does_not_forget_the_prompts_behind_it() {
    // Entering the alt screen shows a different set of cells, so a *selection* made over
    // the old ones has to go — but the primary buffer's rows are untouched by the switch,
    // and the prompt marks name lines in it, so those ids still resolve to exactly the
    // lines they always did. Breaking the whole row-identity regime cleared them anyway,
    // so running and quitting `vim` forgot every prompt older than the one the shell
    // redraws on the way out: the entire history prompt-jump exists to walk.
    let mut s = Screen::with_scrollback(20, 6, 100);
    for i in 0..4 {
        feed(&mut s, b"\x1b]133;A\x07$ cmd\r\n\x1b]133;C\x07");
        feed(&mut s, format!("out {i}\r\n").as_bytes());
        feed(&mut s, b"\x1b]133;D;0\x07");
    }
    let before: Vec<_> = s.prompts().iter().map(|p| p.row).collect();
    assert_eq!(before.len(), 4, "four commands run at four prompts");

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[?1049h"); // vim starts
    assert_ne!(
        s.row_epoch(),
        epoch,
        "a display-anchored selection still dies: those cells are gone"
    );
    feed(&mut s, b"\x1b[2Jediting\r\n");
    feed(&mut s, b"\x1b[?1049l"); // and quits

    let after: Vec<_> = s.prompts().iter().map(|p| p.row).collect();
    assert_eq!(
        after, before,
        "every prompt survived, naming the same lines"
    );
    // And they are still usable, which is the whole point: prompt-jump walks back
    // through them rather than finding one lone entry the shell redrew on the way out.
    assert!(s.scroll_to_prompt(true), "jump back to the previous prompt");
    assert!(s.scroll_to_prompt(true), "and the one before that");
}

#[test]
fn osc_52_puts_the_childs_text_on_the_clipboard() {
    // How nvim over SSH, tmux's copy mode and Claude Code's `/copy` reach the system
    // clipboard: they cannot see the compositor, and the terminal can.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]52;c;aGVsbG8=\x07"); // "hello"
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Clipboard, b"hello".to_vec())]
    );
    assert!(s.take_clipboard_writes().is_empty(), "the queue drains");

    // `p` is the primary selection, and a program may name several at once.
    feed(&mut s, b"\x1b]52;p;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Primary, b"hi".to_vec())]
    );
    feed(&mut s, b"\x1b]52;pc;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![
            (ClipboardTarget::Clipboard, b"hi".to_vec()),
            (ClipboardTarget::Primary, b"hi".to_vec()),
        ]
    );
    // An empty selection field means the clipboard.
    feed(&mut s, b"\x1b]52;;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Clipboard, b"hi".to_vec())]
    );
}

#[test]
fn osc_52_never_reads_the_clipboard_back() {
    // The protocol defines a read, and we refuse it on purpose.
    //
    // A terminal cannot tell which program printed a byte. `cat` a hostile file and
    // it can send this; the reply goes straight back down the pty to whatever is
    // reading. That turns "display some text" into "exfiltrate the clipboard", which
    // routinely holds passwords. The answer is silence, not a reply.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]52;c;?\x07");
    assert!(s.take_responses().is_empty(), "no answer, ever");
    assert!(s.take_clipboard_writes().is_empty(), "and nothing written");
}

#[test]
fn a_malformed_osc_52_is_dropped_rather_than_half_decoded() {
    // The child writes this, so it is hostile input. A decoder that improvises on bad
    // bytes is how you smuggle bytes past it.
    let mut s = Screen::new(10, 2);
    for junk in [
        &b"\x1b]52;c;not base64!!\x07"[..],
        &b"\x1b]52;c;aGVsbG8\x3d\x3d\x3d\x07"[..], // three padding characters
        &b"\x1b]52;c;a\x07"[..],                   // a lone leftover character
        &b"\x1b]52;c;aG=sbG8=\x07"[..],            // padding in the middle
    ] {
        feed(&mut s, junk);
        assert!(
            s.take_clipboard_writes().is_empty(),
            "refused: {:?}",
            std::str::from_utf8(junk)
        );
    }
}

#[test]
fn a_program_reset_returns_to_the_configured_colours() {
    let mut s = Screen::new(10, 2);
    let mut configured = Theme::default();
    configured.fg = Rgb::new(0x11, 0x22, 0x33);
    configured.bg = Rgb::new(0x44, 0x55, 0x66);
    configured.cursor = Rgb::new(0x77, 0x88, 0x99);
    configured.set_indexed(1, Rgb::new(0xaa, 0xbb, 0xcc));
    s.set_theme(std::rc::Rc::new(configured));

    feed(
        &mut s,
        b"\x1b]10;#ff0000\x07\x1b]11;#00ff00\x07\x1b]12;#0000ff\x07",
    );
    feed(&mut s, b"\x1b]4;1;#010203\x07");
    assert_eq!(s.theme().fg, Rgb::new(0xff, 0, 0));
    assert_eq!(s.theme().indexed(1), Rgb::new(1, 2, 3));

    feed(&mut s, b"\x1b]110\x07\x1b]111\x07\x1b]112\x07");
    assert_eq!(s.theme().fg, Rgb::new(0x11, 0x22, 0x33), "configured fg");
    assert_eq!(s.theme().bg, Rgb::new(0x44, 0x55, 0x66), "configured bg");
    assert_eq!(
        s.theme().cursor,
        Rgb::new(0x77, 0x88, 0x99),
        "configured cursor"
    );

    // Set the foreground again, so the palette reset below has a *live* named colour
    // to leave alone rather than one that already matches the configuration.
    feed(&mut s, b"\x1b]10;#ff0000\x07\x1b]4;1;#010203\x07");
    feed(&mut s, b"\x1b]104\x07");
    assert_eq!(
        s.theme().indexed(1),
        Rgb::new(0xaa, 0xbb, 0xcc),
        "a whole-palette reset lands on the configured palette too"
    );
    assert_eq!(
        s.theme().fg,
        Rgb::new(0xff, 0, 0),
        "and leaves the named colours alone, as OSC 104 must"
    );
}

#[test]
fn screens_share_one_baseline_but_never_a_live_palette() {
    let mut base = Theme::default();
    base.fg = Rgb::new(0x11, 0x22, 0x33);
    let base = std::rc::Rc::new(base);

    let mut first = Screen::new(4, 2);
    let mut second = Screen::new(4, 2);
    first.set_theme(std::rc::Rc::clone(&base));
    second.set_theme(std::rc::Rc::clone(&base));
    assert_eq!(
        std::rc::Rc::strong_count(&base),
        3,
        "one allocation, held here and by both screens"
    );

    feed(&mut first, b"\x1b]10;#ff0000\x07");
    assert_eq!(first.theme().fg, Rgb::new(0xff, 0, 0));
    assert_eq!(
        second.theme().fg,
        Rgb::new(0x11, 0x22, 0x33),
        "the neighbouring tab is untouched"
    );

    feed(&mut first, b"\x1b]110\x07");
    assert_eq!(
        first.theme().fg,
        Rgb::new(0x11, 0x22, 0x33),
        "and the reset lands on the shared baseline"
    );
}

#[test]
fn osc_10_11_12_set_the_colours_and_the_query_follows() {
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;#ff0000\x07"); // a red background
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0));
    feed(&mut s, b"\x1b]11;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:ffff/0000/0000\x07");
    // A default cell now paints on the new background: the change is not cosmetic,
    // it is the palette the whole grid resolves against.
    assert_eq!(
        Color::Default.resolve(s.theme(), Ground::Background),
        Rgb::new(0xff, 0, 0)
    );
    // `OSC 111` puts it back.
    feed(&mut s, b"\x1b]111\x07");
    assert_eq!(s.theme().bg, Theme::default().bg);
}

#[test]
fn osc_4_sets_and_queries_any_palette_entry() {
    let mut s = Screen::new(10, 2);
    // A base16 theme script repaints far past the ANSI 16, which is exactly why the
    // palette is stored rather than computed.
    feed(&mut s, b"\x1b]4;1;rgb:aa/bb/cc;17;#010203\x07");
    assert_eq!(s.theme().indexed(1), Rgb::new(0xaa, 0xbb, 0xcc));
    assert_eq!(s.theme().indexed(17), Rgb::new(1, 2, 3));
    // And an SGR colour resolves through it.
    assert_eq!(
        Color::Ansi(1).resolve(s.theme(), Ground::Foreground),
        Rgb::new(0xaa, 0xbb, 0xcc)
    );
    feed(&mut s, b"\x1b]4;1;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]4;1;rgb:aaaa/bbbb/cccc\x07");
    // `OSC 104` puts the named entries back without touching fg/bg/cursor.
    feed(&mut s, b"\x1b]11;#ff0000\x07");
    feed(&mut s, b"\x1b]104;1\x07");
    assert_eq!(s.theme().indexed(1), Theme::default().indexed(1));
    assert_eq!(s.theme().indexed(17), Rgb::new(1, 2, 3), "only 1 was reset");
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0), "and the bg survived");
    // With no index, the whole indexed palette resets — still not fg/bg/cursor.
    feed(&mut s, b"\x1b]104\x07");
    assert_eq!(s.theme().indexed(17), Theme::default().indexed(17));
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0));
}

#[test]
fn a_malformed_colour_is_ignored_not_guessed_at() {
    // The child writes these, so they are hostile input like everything else.
    let mut s = Screen::new(10, 2);
    let before = *s.theme();
    for junk in [
        &b"\x1b]11;lightgoldenrodyellow\x07"[..], // an X11 name: we do not ship rgb.txt
        &b"\x1b]11;rgb:zz/zz/zz\x07"[..],
        &b"\x1b]11;#12345\x07"[..], // not divisible by three
        &b"\x1b]11;\x07"[..],
        &b"\x1b]4;999;#ffffff\x07"[..], // index out of range
        &b"\x1b]4;notanumber;#ffffff\x07"[..],
    ] {
        feed(&mut s, junk);
    }
    assert_eq!(*s.theme(), before, "nothing moved");
    assert!(s.take_responses().is_empty(), "and nothing was answered");
}

#[test]
fn osc_7501_answers_the_query_in_kind_and_never_echoes_a_report() {
    let mut s = Screen::new(20, 4);
    feed(&mut s, b"\x1b]7501;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]7501;?\x07");
    feed(&mut s, b"\x1b]7501;?\x1b\\");
    assert_eq!(s.take_responses(), b"\x1b]7501;?\x1b\\");

    // A report is never answered, and the answer to a later query carries nothing of it.
    feed(&mut s, b"\x1b]7501;state=blocked:id=x:msg=SGk\x07");
    assert!(s.take_responses().is_empty());
    assert_eq!(s.program_status(), Some(State::Blocked));
    feed(&mut s, b"\x1b]7501;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]7501;?\x07");
}

#[test]
fn program_status_belongs_to_the_terminal_and_ends_at_a_new_prompt() {
    let mut s = Screen::new(20, 6);
    feed(&mut s, b"\x1b]133;A\x07$ deploy\r\n");
    feed(
        &mut s,
        b"\x1b]7501;state=working\x07\x1b]7501;state=done:id=eu\x07",
    );
    assert_eq!(s.program_status(), Some(State::Done));

    // Neither the alt screen nor a soft reset touches the records.
    feed(&mut s, b"\x1b[?1049h\x1b[!p\x1b[?1049l");
    feed(&mut s, b"\x1b]7501;state=clear:id=eu\x07");
    assert_eq!(s.program_status(), Some(State::Working));

    // The next prompt ends what was running and keeps what finished.
    feed(
        &mut s,
        b"\x1b]7501;state=done:id=eu\x07\x1b]7501;state=blocked\x07",
    );
    feed(&mut s, b"\r\n\x1b]133;A\x07$ ");
    assert_eq!(s.program_status(), Some(State::Done));

    // So does a prompt redrawn in place.
    feed(&mut s, b"\x1b]7501;state=blocked\x07\r\x1b]133;A\x07$ ");
    assert_eq!(s.program_status(), Some(State::Done));

    feed(&mut s, b"\x1b]7501;state=clear\x07");
    assert_eq!(s.program_status(), None);

    feed(&mut s, b"\x1b]7501;state=error\x07\x1bc");
    assert_eq!(s.program_status(), None, "RIS removes every record");
}

#[test]
fn a_prompt_on_the_same_row_after_a_command_is_a_new_prompt() {
    // A command whose program sent the cursor home, then the shell's next prompt on the
    // row of the last one. The mark belongs to the new prompt, with nothing recorded yet.
    // Either half of a command is enough: `D` without `C` (bash before 4.4 has no `PS0`
    // to send it), with or without its optional code, and `C` alone.
    for command in [
        &b"\x1b]133;C\x07\x1b]133;D;130\x07"[..],
        b"\x1b]133;D;0\x07",
        b"\x1b]133;D\x07",
        b"\x1b]133;C\x07",
    ] {
        let mut s = Screen::new(20, 6);
        feed(&mut s, b"\x1b]133;A\x07$ agent\r\n");
        feed(&mut s, command);
        feed(&mut s, b"\x1b[H\x1b]133;A\x07$ ");
        let fresh = Prompt {
            row: s.abs_row(0),
            output: None,
            exit: None,
            ended: false,
        };
        assert_eq!(s.prompts(), [fresh], "{command:?}");
    }
}

#[test]
fn osc_7501_caps_the_sequence_as_sent_counting_what_the_parser_strips() {
    // `len` bytes as sent, padded with an unknown key. The last `nuls` before the
    // terminator are NULs, which the parser strips.
    let report = |len: usize, nuls: usize, bel: bool| {
        let end: &[u8] = if bel { b"\x07" } else { b"\x1b\\" };
        let mut seq = b"\x1b]7501;state=done:pad=".to_vec();
        seq.resize(len - end.len() - nuls, b'x');
        seq.resize(len - end.len(), 0);
        seq.extend_from_slice(end);
        seq
    };
    let applied = |seq: Vec<u8>| {
        let mut s = Screen::new(20, 4);
        feed(&mut s, &seq);
        s.program_status() == Some(State::Done)
    };
    assert!(applied(report(64, 8, false)), "NULs alone discard nothing");
    for bel in [false, true] {
        assert!(applied(report(4096, 0, bel)), "at the cap, bel={bel}");
        assert!(!applied(report(4097, 0, bel)), "past it, bel={bel}");
        assert!(
            !applied(report(4097, 1000, bel)),
            "past it in NULs, bel={bel}"
        );
    }
}
