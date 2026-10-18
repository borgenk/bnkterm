//! The keys the window keeps from the child: the modal "leader" tables layered over the
//! terminal, wezterm-style, and the direct chords beside them.
//!
//! A terminal normally hands every key straight to the child. bnkterm adds an
//! opt-in modal layer on top: pressing the leader (Ctrl+A) swallows that key and
//! arms a transient mode in which the next key drives the tab bar instead of the
//! shell. A centered overlay names the armed mode and its keys, so the modality is
//! never invisible (the classic modal-editor complaint).
//!
//! ```text
//!   Normal ──Ctrl+A──▶ Leader ──w──▶ Tabs ⟲  (↑/↓ switch, shift+↑/↓ reorder)
//!     ▲                  │  c new         │
//!     │                  │  x close       │
//!     └── esc / other ───┴── esc / enter ─┘
//!
//!   Ctrl+A Ctrl+A ─▶ send a literal Ctrl+A to the child (the escape hatch, so
//!                    readline's beginning-of-line is still reachable).
//! ```
//!
//! Holding Ctrl through a command key is the same as releasing it first: `Ctrl+A
//! Ctrl+C` opens a tab exactly as `Ctrl+A c` does. Typing the sequence fast is how
//! it is typed in practice, and a lagging Ctrl release must not turn a command into
//! a silent cancel.
//!
//! Normal mode also binds chords that need no leader: Ctrl+Tab and Ctrl+PgUp/PgDn switch
//! tabs, Ctrl+Shift+T/W open and close one, and Ctrl+Shift+C/V copy and paste.
//!
//! The whole thing is two pure functions with no window or PTY: [`advance`] is the
//! `(mode, key, mods) -> (mode, disposition)` transition, unit-tested as a table;
//! [`paint_overlay`] appends the indicator to a display list, diffed and presented
//! like any other frame content. The app owns the current [`KeyMode`], feeds every
//! resolved key through [`advance`], and acts on the [`Disposition`].

use crate::color::Theme;
use crate::input::{Key, Mods};
use crate::panel::Panel;
use crate::platform::geom::Rect;
use crate::render::display::DisplayList;
use crate::term_render::CellMetrics;
use crate::width;

/// The modal key-table state. `Normal` is the transparent default in which every
/// key passes through; the others are the armed leader tables that intercept keys
/// for tab control until they return to `Normal`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum KeyMode {
    #[default]
    Normal,
    /// Ctrl+A was pressed; the next key is a leader command (one-shot).
    Leader,
    /// Ctrl+A then `w`: a sticky table where arrows switch and reorder tabs until
    /// Escape/Enter (or an unrecognised key) leaves it.
    Tabs,
}

/// A window operation a key resolves to. The app applies these to the tab manager and
/// the clipboard; keeping them a plain enum lets the state machine stay free of both
/// and stay testable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WindowAction {
    NewTab,
    CloseTab,
    /// Make the tab one slot along the active one. Selection wraps at both edges.
    SelectTab(Dir),
    /// Reorder the active tab one slot along, keeping it active. Reordering does not
    /// wrap: flinging a tab past the edge to the far side would be surprising.
    MoveTab(Dir),
    /// Put the active tab's selection on the clipboard.
    Copy,
    /// Paste the clipboard into the active tab.
    Paste,
}

/// Which way along the strip an action travels: toward index 0, or toward the end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Dir {
    Prev,
    Next,
}

/// What the app does with a key after the mode machine has seen it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Disposition {
    /// Not bound here: process it normally (the scrollback keys, then the child).
    Passthrough,
    /// The key drove the machine; do not send it onward. Run `action` if present.
    Consumed(Option<WindowAction>),
    /// Leave the mode and send this exact key to the child. The leader escape
    /// hatch: Ctrl+A Ctrl+A types a literal Ctrl+A rather than re-arming.
    SendLiteral(Key, Mods),
}

/// The leader key: Ctrl and only Ctrl over an `a` (either case, so Caps Lock does
/// not defeat it). Requiring an exact chord keeps Ctrl+Shift+A and Alt+Ctrl+A free
/// for other bindings and for the child.
fn is_leader(key: Key, mods: Mods) -> bool {
    matches!(key, Key::Char { typed, .. } if typed.eq_ignore_ascii_case(&'a')) && mods == Mods::CTRL
}

/// The lowercase letter a leader command key carries, or `None` when the key is not
/// a character. Shift is folded away so `W` and `w` are the same command, and so is
/// Ctrl: typing the sequence quickly means the leader's Ctrl is still down when the
/// command key lands, so `Ctrl+A Ctrl+C` has to mean what `Ctrl+A c` means. GNU
/// screen binds both spellings of every command for this reason. Alt is not folded —
/// it is nobody's stuck modifier, and leaving it out keeps Alt chords unclaimed.
///
/// The leader itself is matched before this runs, so folding Ctrl here does not
/// swallow the `Ctrl+A Ctrl+A` escape hatch.
fn command_letter(key: Key, mods: Mods) -> Option<char> {
    if mods.contains(Mods::ALT) {
        return None;
    }
    match key {
        Key::Char { typed, .. } => Some(typed.to_ascii_lowercase()),
        _ => None,
    }
}

/// The chords bound without the leader. Each takes a Ctrl key the child would otherwise
/// receive. The letters also need Shift, because plain Ctrl+T/W/C/V are the child's
/// (Ctrl+C above all). The letter may be the key's `base` or what it typed: a dead key
/// that composes with it changes only `typed` (`´` then `Ctrl+Shift+V` types `Ǘ`), and
/// a layout with Latin on the Shift level changes only `base` (the Hebrew V key is `ה`).
/// The page keys take Ctrl alone, because Shift+Page is the scrollback's.
fn window_chord(key: Key, mods: Mods) -> Option<WindowAction> {
    if !mods.contains(Mods::CTRL) {
        return None;
    }
    match key {
        Key::Tab if !mods.contains(Mods::ALT) => {
            let dir = if mods.contains(Mods::SHIFT) {
                Dir::Prev
            } else {
                Dir::Next
            };
            Some(WindowAction::SelectTab(dir))
        }
        Key::PageUp if mods == Mods::CTRL => Some(WindowAction::SelectTab(Dir::Prev)),
        Key::PageDown if mods == Mods::CTRL => Some(WindowAction::SelectTab(Dir::Next)),
        Key::Char { typed, base } if mods.contains(Mods::SHIFT) => {
            let letter = |c: char| match c.to_ascii_lowercase() {
                't' => Some(WindowAction::NewTab),
                'w' => Some(WindowAction::CloseTab),
                'c' => Some(WindowAction::Copy),
                'v' => Some(WindowAction::Paste),
                _ => None,
            };
            letter(base).or_else(|| letter(typed))
        }
        _ => None,
    }
}

/// The transition function. Given the current mode and a resolved key press,
/// return the next mode and what the app should do with the key. Pure: no I/O, no
/// manager, so the entire table is pinned by unit tests.
pub(crate) fn advance(mode: KeyMode, key: Key, mods: Mods) -> (KeyMode, Disposition) {
    match mode {
        KeyMode::Normal => {
            if is_leader(key, mods) {
                return (KeyMode::Leader, Disposition::Consumed(None));
            }
            match window_chord(key, mods) {
                Some(action) => (KeyMode::Normal, Disposition::Consumed(Some(action))),
                None => (KeyMode::Normal, Disposition::Passthrough),
            }
        }
        KeyMode::Leader => {
            // Ctrl+A again is the escape hatch: emit a real Ctrl+A and disarm.
            if is_leader(key, mods) {
                return (
                    KeyMode::Normal,
                    Disposition::SendLiteral(Key::plain('a'), Mods::CTRL),
                );
            }
            match command_letter(key, mods) {
                Some('w') => (KeyMode::Tabs, Disposition::Consumed(None)),
                // `c` is wezterm's new-tab leader key; `t` is kept as an alias.
                Some('c') | Some('t') => (
                    KeyMode::Normal,
                    Disposition::Consumed(Some(WindowAction::NewTab)),
                ),
                Some('x') => (
                    KeyMode::Normal,
                    Disposition::Consumed(Some(WindowAction::CloseTab)),
                ),
                // Escape, an arrow, or any unbound key cancels the one-shot leader
                // and is swallowed (the overlay vanishing is the signal it did).
                _ => (KeyMode::Normal, Disposition::Consumed(None)),
            }
        }
        KeyMode::Tabs => {
            // Ctrl+A reopens the leader menu from within the sticky tab table.
            if is_leader(key, mods) {
                return (KeyMode::Leader, Disposition::Consumed(None));
            }
            let shift = mods.contains(Mods::SHIFT);
            match key {
                Key::Up | Key::Left | Key::Down | Key::Right => {
                    let dir = match key {
                        Key::Up | Key::Left => Dir::Prev,
                        _ => Dir::Next,
                    };
                    // Shift reorders the tab instead of selecting one.
                    let action = if shift {
                        WindowAction::MoveTab(dir)
                    } else {
                        WindowAction::SelectTab(dir)
                    };
                    (KeyMode::Tabs, Disposition::Consumed(Some(action)))
                }
                Key::Escape | Key::Enter => (KeyMode::Normal, Disposition::Consumed(None)),
                // Any other key leaves the sticky table (and is swallowed), so a
                // stray keystroke can never leak into the shell from tab mode.
                _ => (KeyMode::Normal, Disposition::Consumed(None)),
            }
        }
    }
}

/// The overlay hint per armed mode: the keys the mode binds. Only single-width
/// clusters (ASCII plus the arrows) so the fixed-pitch
/// [`crate::term_render::push_cell_text`] runs never break.
const LEADER_LINES: &[&str] = &["w tabs   c new   x close   esc cancel"];
const TABS_LINES: &[&str] = &["↑↓ switch   shift+↑↓ reorder   esc done"];

/// Append the mode indicator to `out`: a darkened rounded panel, then the mode's
/// hint line centered on the surface. Nothing is drawn in `Normal` mode, so a
/// passthrough frame is byte-identical to one built without the overlay. Strings are
/// drawn from the shared `strings` pool like every other run, so a steady overlaid
/// frame stays allocation-free.
pub(crate) fn paint_overlay(
    mode: KeyMode,
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    metrics: CellMetrics,
    theme: &Theme,
    surface_w: i32,
    surface_h: i32,
) {
    let lines: &[&str] = match mode {
        KeyMode::Normal => return,
        KeyMode::Leader => LEADER_LINES,
        KeyMode::Tabs => TABS_LINES,
    };
    if metrics.w <= 0 || metrics.h <= 0 {
        return;
    }

    // The content box in cells, padded by a cell horizontally and half a cell
    // vertically, centered on the surface.
    let content_cols = lines
        .iter()
        .map(|line| width::text_cells(line))
        .max()
        .unwrap_or(0) as i32;
    let panel = Panel::roomy(metrics);
    let panel_w = panel.width(content_cols);
    let panel_h = panel.height(lines.len() as i32);
    let rect = Rect {
        x: ((surface_w - panel_w) / 2).max(0),
        y: ((surface_h - panel_h) / 2).max(0),
        w: panel_w,
        h: panel_h,
    };

    // No accent frame: the hint text alone names the mode.
    let mut pen = panel.open(out, strings, rect, theme.fg, Panel::plate(theme));
    for (index, line) in lines.iter().enumerate() {
        // Center each line within the content box.
        let line_cols = width::text_cells(line) as i32;
        pen.write(index as i32, (content_cols - line_cols) / 2, line);
    }
}

#[cfg(test)]
mod tests {
    use crate::keymode::*;
    use crate::platform::grapheme;
    use crate::render::display::DrawCmd;

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    #[test]
    fn normal_arms_the_leader_on_ctrl_a_alone() {
        assert_eq!(
            advance(KeyMode::Normal, Key::plain('a'), Mods::NONE),
            (KeyMode::Normal, Disposition::Passthrough)
        );
        // Ctrl+A alone is the leader; Ctrl+Shift+A and Alt+Ctrl+A are not.
        assert_eq!(
            advance(KeyMode::Normal, Key::plain('a'), Mods::CTRL),
            (KeyMode::Leader, Disposition::Consumed(None))
        );
        assert_eq!(
            advance(KeyMode::Normal, Key::plain('a'), Mods::CTRL | Mods::SHIFT),
            (KeyMode::Normal, Disposition::Passthrough)
        );
        assert_eq!(
            advance(KeyMode::Normal, Key::plain('a'), Mods::CTRL | Mods::ALT),
            (KeyMode::Normal, Disposition::Passthrough)
        );
    }

    /// The letter key as Shift types it: `typed` upper, `base` lower.
    fn shifted(letter: char) -> Key {
        Key::Char {
            typed: letter.to_ascii_uppercase(),
            base: letter,
        }
    }

    #[test]
    fn normal_resolves_the_window_chords_and_stays_normal() {
        let ctrl_shift = Mods::CTRL | Mods::SHIFT;
        for (key, mods, action) in [
            (Key::Tab, Mods::CTRL, WindowAction::SelectTab(Dir::Next)),
            (Key::Tab, ctrl_shift, WindowAction::SelectTab(Dir::Prev)),
            (Key::PageUp, Mods::CTRL, WindowAction::SelectTab(Dir::Prev)),
            (
                Key::PageDown,
                Mods::CTRL,
                WindowAction::SelectTab(Dir::Next),
            ),
            (shifted('t'), ctrl_shift, WindowAction::NewTab),
            (shifted('w'), ctrl_shift, WindowAction::CloseTab),
            (shifted('c'), ctrl_shift, WindowAction::Copy),
            (shifted('v'), ctrl_shift, WindowAction::Paste),
            // A waiting dead key composed with the letter; the key is still V.
            (
                Key::Char {
                    typed: 'Ǘ',
                    base: 'v',
                },
                ctrl_shift,
                WindowAction::Paste,
            ),
            // Hebrew puts Latin on the Shift level; the key itself is ה.
            (
                Key::Char {
                    typed: 'V',
                    base: 'ה',
                },
                ctrl_shift,
                WindowAction::Paste,
            ),
        ] {
            assert_eq!(
                advance(KeyMode::Normal, key, mods),
                (KeyMode::Normal, Disposition::Consumed(Some(action))),
                "{key:?} with {mods:?}"
            );
        }
    }

    #[test]
    fn near_misses_of_the_window_chords_reach_the_child() {
        for (key, mods) in [
            // Ctrl+C is the interrupt and Ctrl+V a literal-next; neither may copy or paste.
            (Key::plain('c'), Mods::CTRL),
            (Key::plain('v'), Mods::CTRL),
            (Key::plain('t'), Mods::CTRL),
            (Key::plain('w'), Mods::CTRL),
            (shifted('x'), Mods::CTRL | Mods::SHIFT),
            (Key::Tab, Mods::NONE),
            (Key::Tab, Mods::CTRL | Mods::ALT),
            // Shift+Page belongs to the scrollback, with or without Ctrl.
            (Key::PageUp, Mods::SHIFT),
            (Key::PageUp, Mods::CTRL | Mods::SHIFT),
            (Key::PageDown, Mods::CTRL | Mods::SHIFT),
            (Key::PageUp, Mods::CTRL | Mods::ALT),
            (Key::PageDown, Mods::CTRL | Mods::ALT),
        ] {
            assert_eq!(
                advance(KeyMode::Normal, key, mods),
                (KeyMode::Normal, Disposition::Passthrough),
                "{key:?} with {mods:?}"
            );
        }
    }

    #[test]
    fn leader_double_tap_sends_a_literal_ctrl_a() {
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('a'), Mods::CTRL),
            (
                KeyMode::Normal,
                Disposition::SendLiteral(Key::plain('a'), Mods::CTRL)
            )
        );
    }

    #[test]
    fn leader_commands_map_and_disarm() {
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('w'), Mods::NONE),
            (KeyMode::Tabs, Disposition::Consumed(None))
        );
        // Shift is folded away, so W is w.
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('W'), Mods::SHIFT),
            (KeyMode::Tabs, Disposition::Consumed(None))
        );
        // `c` (wezterm's binding) and its `t` alias both open a new tab and disarm.
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('c'), Mods::NONE),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(WindowAction::NewTab))
            )
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('t'), Mods::NONE),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(WindowAction::NewTab))
            )
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('x'), Mods::NONE),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(WindowAction::CloseTab))
            )
        );
        // Ctrl is folded away too: a command typed before the leader's Ctrl comes
        // back up is the same command.
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('c'), Mods::CTRL),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(WindowAction::NewTab))
            )
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('x'), Mods::CTRL),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(WindowAction::CloseTab))
            )
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('w'), Mods::CTRL),
            (KeyMode::Tabs, Disposition::Consumed(None))
        );
        // Alt is not folded, so an Alt chord still cancels rather than commanding.
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('c'), Mods::ALT),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
        // Escape and unbound keys cancel the one-shot leader, swallowing the key.
        assert_eq!(
            advance(KeyMode::Leader, Key::Escape, Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('z'), Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
    }

    /// Typed fast, `Ctrl+A c` twice is really `Ctrl` down, `a`, `c`, `a`, `c`, `Ctrl`
    /// up: the hand never lets the modifier go between the rounds. Both rounds must
    /// still open a tab, and the leader must stay reachable from Leader mode itself
    /// so the second `a` re-arms rather than cancelling.
    #[test]
    fn a_held_ctrl_across_the_whole_sequence_opens_both_tabs() {
        let mut mode = KeyMode::Normal;
        let mut opened = 0;
        for key in ['a', 'c', 'a', 'c'] {
            let (next, disposition) = advance(mode, Key::plain(key), Mods::CTRL);
            mode = next;
            if let Disposition::Consumed(Some(WindowAction::NewTab)) = disposition {
                opened += 1;
            }
            assert!(
                !matches!(disposition, Disposition::Passthrough),
                "{key} leaked to the child"
            );
        }
        assert_eq!(opened, 2, "both rounds open a tab");
        assert_eq!(mode, KeyMode::Normal);
    }

    #[test]
    fn tabs_arrows_switch_and_reorder_and_stay() {
        for (key, plain, moved) in [
            (
                Key::Up,
                WindowAction::SelectTab(Dir::Prev),
                WindowAction::MoveTab(Dir::Prev),
            ),
            (
                Key::Left,
                WindowAction::SelectTab(Dir::Prev),
                WindowAction::MoveTab(Dir::Prev),
            ),
            (
                Key::Down,
                WindowAction::SelectTab(Dir::Next),
                WindowAction::MoveTab(Dir::Next),
            ),
            (
                Key::Right,
                WindowAction::SelectTab(Dir::Next),
                WindowAction::MoveTab(Dir::Next),
            ),
        ] {
            assert_eq!(
                advance(KeyMode::Tabs, key, Mods::NONE),
                (KeyMode::Tabs, Disposition::Consumed(Some(plain)))
            );
            assert_eq!(
                advance(KeyMode::Tabs, key, Mods::SHIFT),
                (KeyMode::Tabs, Disposition::Consumed(Some(moved)))
            );
        }
    }

    #[test]
    fn tabs_exits_on_escape_enter_or_a_stray_key() {
        assert_eq!(
            advance(KeyMode::Tabs, Key::Escape, Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
        assert_eq!(
            advance(KeyMode::Tabs, Key::Enter, Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
        // A stray letter leaves the mode and is swallowed, never reaching the shell.
        assert_eq!(
            advance(KeyMode::Tabs, Key::plain('l'), Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(None))
        );
        // Ctrl+A from tab mode reopens the leader menu.
        assert_eq!(
            advance(KeyMode::Tabs, Key::plain('a'), Mods::CTRL),
            (KeyMode::Leader, Disposition::Consumed(None))
        );
    }

    #[test]
    fn normal_mode_draws_no_overlay() {
        let mut out = Vec::new();
        let mut strings = Vec::new();
        paint_overlay(
            KeyMode::Normal,
            &mut out,
            &mut strings,
            METRICS,
            &Theme::default(),
            800,
            600,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn overlay_is_a_centered_borderless_panel_of_plain_single_width_text() {
        let mut out = Vec::new();
        let mut strings = Vec::new();
        let theme = Theme::default();
        paint_overlay(
            KeyMode::Tabs,
            &mut out,
            &mut strings,
            METRICS,
            &theme,
            800,
            600,
        );

        // Exactly one rounded panel (no magenta accent frame), then the hint run(s).
        assert!(matches!(out[0], DrawCmd::RoundRect { .. }));
        assert_eq!(
            out.iter()
                .filter(|cmd| matches!(cmd, DrawCmd::RoundRect { .. }))
                .count(),
            1,
            "the accent border is gone"
        );
        assert!(out.len() >= 2, "panel and at least one text run");

        // The panel is centered on the surface.
        let panel = match &out[0] {
            DrawCmd::RoundRect { rect, .. } => *rect,
            _ => unreachable!(),
        };
        let panel_cx = panel.x + panel.w / 2;
        let panel_cy = panel.y + panel.h / 2;
        assert!((panel_cx - 400).abs() <= METRICS.w, "horizontally centered");
        assert!((panel_cy - 300).abs() <= METRICS.h, "vertically centered");

        // Every text run is the plain foreground color (no accent) and single-width.
        for cmd in &out {
            if let DrawCmd::Cells { color, text, .. } = cmd {
                assert_eq!(*color, theme.fg.to_u32(), "overlay text is plain fg");
                assert!(grapheme::graphemes(text)
                    .all(|(_, cluster)| width::cluster_width(cluster) == 1));
            }
        }
    }
}
