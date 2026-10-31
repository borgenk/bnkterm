//! Modes and resets: the alt screen, the saved cursor, soft and hard reset, the cursor
//! style, the mouse modes and the keyboard protocols.

use crate::color::Rgb;
use crate::grid::tests::print_str;
use crate::grid::*;
use crate::testutil::feed;

#[test]
fn mode_12_blinks_the_cursor_without_reshaping_it() {
    // DECSCUSR binds shape and blink together in one parameter, so a program that
    // wants to change only the blink has no way to say it. `?12` is that way.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[4 q"); // a steady underline
    assert_eq!(s.cursor_style(), CursorStyle::Underline);
    assert!(!s.cursor_blinks());

    feed(&mut s, b"\x1b[?12h");
    assert!(s.cursor_blinks(), "blinking now");
    assert_eq!(
        s.cursor_style(),
        CursorStyle::Underline,
        "still an underline"
    );

    feed(&mut s, b"\x1b[?12l");
    assert!(!s.cursor_blinks());
    // And it reports honestly, like every other mode we act on.
    feed(&mut s, b"\x1b[?12$p");
    assert_eq!(s.take_responses(), b"\x1b[?12;2$y");
}

#[test]
fn decstr_resets_the_settings_and_keeps_the_screen() {
    // The point of a soft reset: put the *settings* back without throwing away what
    // is on the screen. A program that has wedged the terminal sends this to hand a
    // sane state back to the shell, which is why RIS would be too big a hammer.
    let mut s = Screen::new(10, 4);
    print_str(&mut s, "keep me");
    feed(&mut s, b"\x1b[1;31m"); // bold red
    feed(&mut s, b"\x1b[?1h"); // DECCKM: application cursor keys
    feed(&mut s, b"\x1b[?6h"); // DECOM: origin mode
    feed(&mut s, b"\x1b[?25l"); // hide the cursor
    feed(&mut s, b"\x1b[4h"); // IRM: insert mode
    feed(&mut s, b"\x1b[2;3r"); // a scroll region
    feed(&mut s, b"\x1b="); // DECKPAM: application keypad
    feed(&mut s, b"\x1b[3;2H"); // park the cursor somewhere
    feed(&mut s, b"\x1b[!p"); // DECSTR

    assert_eq!(s.row_string(0).trim_end(), "keep me", "the screen survived");
    assert_eq!(s.cursor(), (2, 1), "and so did the cursor position");
    assert!(s.pen.attrs.is_empty(), "rendition reset");
    assert_eq!(s.pen.fg, Color::Default);
    assert!(s.cursor_visible());
    assert!(!s.app_cursor_keys());
    assert!(!s.keypad_app());
    assert!(!s.origin_mode);
    assert!(!s.insert_mode);
    let region = (s.active().scroll_top, s.active().scroll_bottom);
    assert_eq!(region, (0, 3), "margins back to the full screen");

    // The DEC table says autowrap is *reset* by DECSTR. xterm leaves it on, and a
    // terminal that stops wrapping the moment a program soft-resets would mangle
    // every long line the shell prints afterwards. We follow xterm.
    assert!(s.autowrap, "autowrap stays on: xterm, not the DEC table");
    feed(&mut s, b"\x1b[1;1Habcdefghijkl"); // 12 chars into a 10-column row
    assert!(s.row_wraps(0), "so a long line still wraps");
    assert_eq!(s.row_string(1).trim_end(), "kl");
}

#[test]
fn decstr_drops_the_saved_cursor() {
    // DECSC state resets to "home position", so a DECRC after a soft reset must not
    // resurrect a cursor from before it.
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"\x1b[3;5H\x1b7"); // park and save
    feed(&mut s, b"\x1b[!p"); // soft reset
    feed(&mut s, b"\x1b[1;1H\x1b8"); // home, then restore
    assert_eq!(s.cursor(), (0, 0), "there was nothing to restore");
}

#[test]
fn a_private_marker_does_not_execute_its_ansi_namesake() {
    // Three sequences that used to be executed as entirely different commands,
    // because the final byte was matched without looking at the marker in front of
    // it. Each must now be a no-op: dropping an unknown sequence is always safe,
    // guessing at it is not.
    let mut s = Screen::new(10, 5);

    // XTRESTORE (`CSI ? Ps r`) restores saved private modes. It used to land on
    // DECSTBM and reset the scroll region, homing the cursor.
    feed(&mut s, b"\x1b[2;4r"); // a real DECSTBM: rows 2..4
    feed(&mut s, b"\x1b[5;3H"); // park the cursor inside it
    feed(&mut s, b"\x1b[?1r"); // XTRESTORE (restore DECCKM) — must do nothing
    assert_eq!(s.cursor(), (4, 2), "the cursor did not move");
    let region = (s.active().scroll_top, s.active().scroll_bottom);
    assert_eq!(region, (1, 3), "the scroll region survived");

    // XTSMGRAPHICS (`CSI ? Pi;Pa;Pv S`) is a sixel geometry query. It used to land
    // on SU and scroll the screen.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"top\r\nmid\r\nlow");
    feed(&mut s, b"\x1b[?1;1;0S");
    assert_eq!(s.row_string(0).trim_end(), "top", "nothing scrolled");

    // `CSI > 4 h` is not a mode set; it used to turn on insert mode (IRM).
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[>4h");
    feed(&mut s, b"ab");
    feed(&mut s, b"\x1b[1;1H");
    feed(&mut s, b"X");
    assert_eq!(
        s.row_string(0).trim_end(),
        "Xb",
        "X overwrote, not inserted"
    );
}

#[test]
fn alt_screen_is_separate_and_restores() {
    let mut s = Screen::new(6, 2);
    print_str(&mut s, "main");
    s.set_mode(1049, true, true); // enter alt, save cursor
    assert!(s.is_alt());
    assert_eq!(s.dump().trim(), ""); // alt starts cleared
    assert_eq!(s.cursor(), (0, 4), "the cursor carries onto the alt screen");
    s.move_to(0, 0);
    print_str(&mut s, "alt");
    assert_eq!(s.row_string(0).trim_end(), "alt");
    s.set_mode(1049, true, false); // leave alt, restore cursor
    assert!(!s.is_alt());
    assert_eq!(s.row_string(0).trim_end(), "main"); // primary intact
    assert_eq!(s.cursor(), (0, 4)); // cursor restored to after "main"
}

/// The alt screen is a different set of cells, not a different cursor: switching
/// carries the cursor across rather than homing it. `?1049h` saves the cursor and
/// clears the alt, and a program that restores expects to land where it started.
#[test]
fn the_cursor_carries_onto_the_alt_screen() {
    let mut s = Screen::new(10, 3);
    s.move_to(2, 5);
    feed(&mut s, b"\x1b[?1049h");
    assert_eq!(s.cursor(), (2, 5), "not homed to the origin");
    feed(&mut s, b"x");
    assert_eq!(s.cell(2, 5).rune, 'x');
}

/// The cursor is one cursor, and it carries *both* ways.
///
/// Carrying it only on the way in would be a cursor that is shared entering the alt
/// screen and per-buffer leaving it. `?1049` cannot tell, because its restore lands
/// on top; `?47`/`?1047` — what a program using the alt screen without the
/// save/restore pair sends — can.
#[test]
fn the_cursor_carries_off_the_alt_screen_as_well_as_onto_it() {
    let mut s = Screen::new(10, 3);
    print_str(&mut s, "main");
    assert_eq!(s.cursor(), (0, 4));

    feed(&mut s, b"\x1b[?1047h");
    assert_eq!(s.cursor(), (0, 4), "carried onto the alt screen");
    feed(&mut s, b"\x1b[3;1H");
    assert_eq!(s.cursor(), (2, 0));

    feed(&mut s, b"\x1b[?1047l");
    assert_eq!(
        s.cursor(),
        (2, 0),
        "and back off it: one cursor, not one per buffer"
    );
    // The primary's cells are of course untouched by any of it.
    assert_eq!(s.row_string(0).trim_end(), "main");
}

/// Each screen owns its DECSC slot, and it survives a trip to the other screen and
/// back: a program may save on the alt screen, leave, return, and restore.
#[test]
fn each_screen_keeps_its_own_saved_cursor_across_a_switch() {
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"\x1b[?1049h"); // to the alt screen
    s.move_to(3, 7);
    feed(&mut s, b"\x1b7"); // DECSC on the alt screen
    feed(&mut s, b"\x1b[?1049l"); // back to the primary
    feed(&mut s, b"\x1b[?1049h"); // and to the alt screen again
    feed(&mut s, b"\x1b[1;1H"); // somewhere else entirely
    feed(&mut s, b"\x1b8"); // DECRC finds what the alt screen saved
    assert_eq!(s.cursor(), (3, 7));
}

#[test]
fn save_and_restore_cursor() {
    let mut s = Screen::new(10, 5);
    s.move_to(2, 4);
    feed(&mut s, b"\x1b[1m");
    s.save_cursor();
    s.move_to(0, 0);
    feed(&mut s, b"\x1b[0m");
    s.restore_cursor();
    assert_eq!(s.cursor(), (2, 4));
    assert!(s.pen.attrs.contains(Attrs::BOLD));
}

#[test]
fn reset_returns_to_power_on_state() {
    let mut s = Screen::new(6, 3);
    feed(&mut s, b"\x1b[31m");
    print_str(&mut s, "junk");
    s.set_scroll_region(1, 2);
    s.reset();
    assert_eq!(s.dump().trim(), "");
    assert_eq!(s.cursor(), (0, 0));
    assert_eq!(s.pen.fg, Color::Default);
    assert_eq!(s.dimensions(), (6, 3));
}

#[test]
fn alt_screen_via_parser() {
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"main");
    feed(&mut s, b"\x1b[?1049h\x1b[H"); // the cursor carries over, so home it
    assert!(s.is_alt());
    feed(&mut s, b"alt");
    assert_eq!(s.row_string(0).trim_end(), "alt");
    feed(&mut s, b"\x1b[?1049l");
    assert_eq!(s.row_string(0).trim_end(), "main");
}

#[test]
fn mode_1048_saves_and_restores_the_cursor() {
    // `?1049` is defined as `?1047` plus `?1048` plus a clear, so a program emitting the
    // pre-1049 pair got the screen switch and silently no cursor save.
    let mut s = Screen::new(10, 5);
    feed(&mut s, b"\x1b[3;4H\x1b[?1048h"); // park, then save
    assert_eq!(s.cursor(), (2, 3));
    feed(&mut s, b"\x1b[1;1H");
    assert_eq!(s.cursor(), (0, 0));
    feed(&mut s, b"\x1b[?1048l"); // restore
    assert_eq!(s.cursor(), (2, 3), "the saved cursor came back");

    // And it reports honestly, because a mode that answers "unrecognised" is a mode
    // nobody will use. xterm reports whether a cursor is currently saved.
    let mut s = Screen::new(10, 5);
    feed(&mut s, b"\x1b[?1048$p");
    assert_eq!(s.take_responses(), b"\x1b[?1048;2$y", "nothing saved yet");
    feed(&mut s, b"\x1b[?1048h\x1b[?1048$p");
    assert_eq!(s.take_responses(), b"\x1b[?1048;1$y", "now there is");
}

#[test]
fn each_alt_screen_mode_clears_when_xterm_does() {
    // xterm's three alt-screen modes differ in when they clear the alt screen: `?47`
    // never, `?1047` on the way out, `?1049` on the way in.
    let reenter = |modes: &[u8]| {
        let mut s = Screen::new(12, 3);
        feed(&mut s, modes);
        s.row_string(0).trim_end().to_string()
    };

    // `?47` re-enters the screen it left, rather than a blank one.
    assert_eq!(
        reenter(b"\x1b[?47h\x1b[1;1HALTCONTENT\x1b[?47l\x1b[?47h"),
        "ALTCONTENT"
    );

    // `?1047` clears on the way out, so re-entering finds it blank whichever mode
    // re-enters, including `?47`, which clears nothing itself.
    assert_eq!(
        reenter(b"\x1b[?1047h\x1b[1;1HALT\x1b[?1047l\x1b[?1047h"),
        ""
    );
    assert_eq!(reenter(b"\x1b[?1047h\x1b[1;1HALT\x1b[?1047l\x1b[?47h"), "");

    // `?1049h` clears on the way in.
    assert_eq!(
        reenter(b"\x1b[?1049h\x1b[1;1HALT\x1b[?1049l\x1b[?1049h"),
        ""
    );
}

#[test]
fn the_kitty_keyboard_stack_pushes_and_pops() {
    let mut s = Screen::new(80, 24);
    // Nothing pushed: legacy.
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    // `CSI > 1 u`: the push every application starts with.
    feed(&mut s, b"\x1b[>1u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // A nested program pushes its own, then pops back to its parent's.
    feed(&mut s, b"\x1b[>0u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    feed(&mut s, b"\x1b[<u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // Popping past the bottom empties the stack rather than underflowing.
    feed(&mut s, b"\x1b[<99u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
}

#[test]
fn the_kitty_set_form_replaces_sets_and_clears() {
    let mut s = Screen::new(80, 24);
    // `CSI = <flags> ; <mode> u`. Mode 1 (the default) replaces.
    feed(&mut s, b"\x1b[=1;1u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // Mode 3 clears the named bits.
    feed(&mut s, b"\x1b[=1;3u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    // Mode 2 sets them.
    feed(&mut s, b"\x1b[=1;2u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // An unknown mode changes nothing.
    feed(&mut s, b"\x1b[=1;9u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
}

#[test]
fn a_kitty_push_loop_cannot_grow_the_stack_without_bound() {
    // The child controls the depth, so it is hostile input like any other. The cap
    // holds and the most recent intent (the top) survives.
    let mut s = Screen::new(80, 24);
    for _ in 0..(KITTY_STACK_LIMIT * 4) {
        feed(&mut s, b"\x1b[>0u");
    }
    feed(&mut s, b"\x1b[>1u");
    assert_eq!(s.kitty_stack().len(), KITTY_STACK_LIMIT);
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
}

#[test]
fn the_kitty_stack_is_per_screen_so_a_dead_program_cannot_strand_the_shell() {
    // The scenario, and it is the ordinary one: `nvim` sends `?1049h` and then
    // `CSI > 1 u`, gets `SIGKILL`ed, and never sends the matching `CSI < 1 u`. The
    // shell's `?1049l` returns to the main screen — which must be speaking exactly the
    // protocol it was speaking before `nvim` started. With one shared stack it is not:
    // `Esc` becomes `CSI 27u` and `Ctrl+C` becomes `CSI 99;5u`, so zsh vi-mode never
    // leaves insert, fzf bindings misfire, and Ctrl+C prints garbage.
    //
    // This is why kitty's protocol specifies a stack per screen, and it is the whole
    // point of having a stack at all: a program cannot strand the terminal in a mode the
    // shell underneath it does not understand.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[>2u"); // the shell's own flags, on the main screen
    assert_eq!(s.kitty_flags(), KittyFlags::REPORT_EVENT_TYPES);

    feed(&mut s, b"\x1b[?1049h");
    assert_eq!(
        s.kitty_flags(),
        KittyFlags::NONE,
        "the alt screen starts on its own empty stack, not the main screen's"
    );
    feed(&mut s, b"\x1b[>1u\x1b[>1u"); // and the program pushes, twice, and dies

    feed(&mut s, b"\x1b[?1049l");
    assert_eq!(
        s.kitty_flags(),
        KittyFlags::REPORT_EVENT_TYPES,
        "the shell got its own protocol back, not the dead program's"
    );

    // And the alt screen kept its own, so re-entering resumes where it left off rather
    // than inheriting whatever the shell has been doing meanwhile.
    feed(&mut s, b"\x1b[?1049h");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
}

#[test]
fn a_bare_csi_u_still_restores_the_cursor() {
    // SCORC and the kitty protocol share the final byte `u`, and only the private
    // marker tells them apart. A parser that got this wrong would break either the
    // cursor save/restore or the keyboard protocol.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[5;3H\x1b[s"); // save at row 5, col 3
    feed(&mut s, b"\x1b[1;1H\x1b[u"); // home, then restore
    assert_eq!(s.cursor(), (4, 2));
    assert_eq!(s.kitty_flags(), KittyFlags::NONE, "SCORC pushed no flags");
    assert!(s.take_responses().is_empty(), "and answered no query");
}

#[test]
fn xtmodkeys_sets_and_resets_the_modify_other_keys_level() {
    let mut s = Screen::new(80, 24);
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
    feed(&mut s, b"\x1b[>4;2m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level2);
    feed(&mut s, b"\x1b[>4;1m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level1);
    // No level given resets, which is how a program turns it back off on exit.
    feed(&mut s, b"\x1b[>4m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
    // A resource other than 4 (modifyCursorKeys, say) is not ours to act on.
    feed(&mut s, b"\x1b[>4;2m\x1b[>1;2m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level2);
}

#[test]
fn a_reset_takes_the_keyboard_protocols_with_it() {
    // RIS is the escape hatch for a program that died mid-mode. If the flags
    // survived it, the shell underneath would keep receiving CSI u for its keys.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[>1u\x1b[>4;2m");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    feed(&mut s, b"\x1bc"); // RIS
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
}

#[test]
fn a_reset_does_not_take_what_was_never_the_childs_to_reset() {
    // RIS wipes the terminal, and the terminal is the part the child owns. Four things
    // are not that, and rebuilding the whole `Screen` took all four.
    let mut s = Screen::new(80, 24);
    s.set_pixel_size(640, 384); // the app's, from the window
    feed(&mut s, b"\x1b]0;a title\x07");

    // A reply queued and a reset, in one write — which is one `read` from the pty, so
    // the parser sees them back to back. The answer to a question already asked cannot
    // be un-asked by what comes after it; xterm delivers it.
    feed(&mut s, b"\x1b[6n\x1bc");
    assert_eq!(
        s.take_responses(),
        b"\x1b[1;1R",
        "the cursor report queued before the reset still arrives"
    );

    assert_eq!(
        s.title(),
        "a title",
        "xterm's RIS does not reset the window title"
    );

    // And the pixel size is a physical fact about a window the child cannot see and did
    // not set. Cleared, `?2048` answers `0x0` until the user happens to resize — the
    // child asks a question the terminal knows the answer to and is told zero.
    feed(&mut s, b"\x1b[?2048h");
    assert_eq!(s.take_responses(), b"\x1b[48;24;80;384;640t");

    // The configured colours are the app's as well, set once when the tab opens. RIS
    // undoes the child's OSC 11 back to the configured background, not the built-in one,
    // and an OSC reset afterwards still has the configured value to return to.
    let mut s = Screen::new(80, 24);
    let mut configured = Theme::default();
    configured.bg = Rgb::new(1, 2, 3);
    s.set_theme(std::rc::Rc::new(configured));
    feed(&mut s, b"\x1b]11;#ff0000\x07\x1bc");
    assert_eq!(
        s.theme().bg,
        Rgb::new(1, 2, 3),
        "RIS keeps the configured colours"
    );
    feed(&mut s, b"\x1b]11;#ff0000\x07\x1b]111\x07");
    assert_eq!(
        s.theme().bg,
        Rgb::new(1, 2, 3),
        "and OSC 111 still finds them"
    );

    // What the child *did* set is gone, which is the whole point of RIS.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[?7l\x1b[4h\x1b[?2027h\x1bc");
    feed(&mut s, b"\x1b[?7$p");
    assert_eq!(s.take_responses(), b"\x1b[?7;1$y", "autowrap back on");
    feed(&mut s, b"\x1b[?2027$p");
    assert_eq!(s.take_responses(), b"\x1b[?2027;2$y", "clustering back off");
}

#[test]
fn decscusr_sets_the_cursor_shape_and_blink() {
    let mut s = Screen::new(10, 2);
    // Power-on default is a steady block (see `CursorAppearance`).
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(!s.cursor_blinks());
    feed(&mut s, b"\x1b[4 q"); // steady underline
    assert_eq!(s.cursor_style(), CursorStyle::Underline);
    assert!(!s.cursor_blinks());
    feed(&mut s, b"\x1b[5 q"); // blinking bar
    assert_eq!(s.cursor_style(), CursorStyle::Bar);
    assert!(s.cursor_blinks());
    feed(&mut s, b"\x1b[1 q"); // blinking block, asked for explicitly
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(s.cursor_blinks());
    feed(&mut s, b"\x1b[7 q"); // out of range: ignored, the last style stands
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(s.cursor_blinks());
}

/// `DECSCUSR 0` means "the terminal's default", not xterm's literal blinking
/// block. TUIs (codex, anything on crossterm's `SetCursorStyle::DefaultUserShape`)
/// emit it while restoring the terminal on exit, so reading it as a blink request
/// leaves a blinking cursor behind in the shell long after the program is gone.
#[test]
fn decscusr_zero_restores_the_power_on_cursor() {
    let mut s = Screen::new(10, 2);
    for style in [&b"\x1b[5 q"[..], b"\x1b[1 q", b"\x1b[3 q"] {
        feed(&mut s, style);
        assert!(s.cursor_blinks(), "the program asked for a blink");
        feed(&mut s, b"\x1b[0 q");
        assert_eq!(s.cursor_style(), CursorStyle::Block);
        assert!(!s.cursor_blinks(), "DECSCUSR 0 restores the steady default");
    }
}

#[test]
fn mouse_modes_track_the_dec_private_toggles() {
    use crate::mouse::MouseProtocol;
    let mut s = Screen::new(10, 4);
    assert!(!s.mouse_mode().reports(), "off by default");
    feed(&mut s, b"\x1b[?1000h");
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::Press);
    feed(&mut s, b"\x1b[?1002h"); // a higher level supersedes
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
    feed(&mut s, b"\x1b[?1006h"); // SGR is an orthogonal encoding flag
    assert!(s.mouse_mode().sgr);
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
    feed(&mut s, b"\x1b[?1002l"); // disabling turns reporting off
    assert!(!s.mouse_mode().reports());
    assert!(s.mouse_mode().sgr, "the encoding flag is independent");
}
