//! The modal "leader" key tables layered over the terminal, wezterm-style.
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
//! The whole thing is two pure functions with no window or PTY: [`advance`] is the
//! `(mode, key, mods) -> (mode, disposition)` transition, unit-tested as a table;
//! [`paint_overlay`] appends the indicator to a display list, diffed and presented
//! like any other frame content. The app owns the current [`KeyMode`], feeds every
//! resolved key through [`advance`], and acts on the [`Disposition`].

use crate::color::{Rgb, Theme};
use crate::input::{Key, Mods};
use crate::platform::freetype::{FaceKey, FontStyle};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd, RoundedCorners};
use crate::term_render::{self, CellMetrics};

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

/// A tab-bar operation a leader key resolves to. The app maps these onto the
/// `app::tabs::Tabs` manager; keeping them a plain enum lets the state
/// machine stay free of the manager and stay testable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TabAction {
    New,
    Close,
    Prev,
    Next,
    /// Reorder the active tab one slot toward index 0.
    MovePrev,
    /// Reorder the active tab one slot toward the end.
    MoveNext,
}

/// What the app does with a key after the mode machine has seen it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Disposition {
    /// Not a mode key: process it normally (the existing chords, then the child).
    Passthrough,
    /// The key drove the machine; do not send it onward. Run `action` if present.
    Consumed(Option<TabAction>),
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

/// The transition function. Given the current mode and a resolved key press,
/// return the next mode and what the app should do with the key. Pure: no I/O, no
/// manager, so the entire table is pinned by unit tests.
pub(crate) fn advance(mode: KeyMode, key: Key, mods: Mods) -> (KeyMode, Disposition) {
    match mode {
        KeyMode::Normal => {
            if is_leader(key, mods) {
                (KeyMode::Leader, Disposition::Consumed(None))
            } else {
                (KeyMode::Normal, Disposition::Passthrough)
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
                Some('c') | Some('t') => {
                    (KeyMode::Normal, Disposition::Consumed(Some(TabAction::New)))
                }
                Some('x') => (
                    KeyMode::Normal,
                    Disposition::Consumed(Some(TabAction::Close)),
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
                Key::Up | Key::Left => {
                    let action = if shift {
                        TabAction::MovePrev
                    } else {
                        TabAction::Prev
                    };
                    (KeyMode::Tabs, Disposition::Consumed(Some(action)))
                }
                Key::Down | Key::Right => {
                    let action = if shift {
                        TabAction::MoveNext
                    } else {
                        TabAction::Next
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
/// [`term_render::push_cell_text`] runs never break.
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
    let content_cols = lines.iter().map(|line| cells_wide(line)).max().unwrap_or(0) as i32;
    let pad_x = metrics.w;
    let pad_y = (metrics.h / 2).max(1);
    let panel_w = content_cols * metrics.w + 2 * pad_x;
    let panel_h = lines.len() as i32 * metrics.h + 2 * pad_y;
    let x0 = ((surface_w - panel_w) / 2).max(0);
    let y0 = ((surface_h - panel_h) / 2).max(0);

    let panel_bg = theme.bg.mix(Rgb::new(0, 0, 0), 1, 2);
    let radius = (metrics.h / 3).max(2);

    // A darkened rounded panel, no accent frame: the hint text alone names the mode.
    out.push(DrawCmd::RoundRect {
        rect: Rect {
            x: x0,
            y: y0,
            w: panel_w,
            h: panel_h,
        },
        color: panel_bg.to_u32(),
        radius,
        corners: RoundedCorners::Both,
    });

    let face = FaceKey::Prose {
        size: metrics.size,
        style: FontStyle::Regular,
    };
    for (index, line) in lines.iter().enumerate() {
        // Center each line within the content box.
        let line_cols = cells_wide(line) as i32;
        let offset = (content_cols - line_cols) / 2 * metrics.w;
        let x = x0 + pad_x + offset;
        let baseline = y0 + pad_y + index as i32 * metrics.h + metrics.baseline;
        term_render::push_cell_text(
            out,
            strings,
            line,
            x,
            baseline,
            metrics,
            face,
            theme.fg.to_u32(),
            panel_bg.to_u32(),
        );
    }
}

/// Display columns a line occupies, cluster-aware (every hint line here is
/// single-width, but the measurement matches the painter's).
fn cells_wide(text: &str) -> usize {
    grapheme::graphemes(text)
        .map(|(_, cluster)| term_render::display_cluster_width(cluster).max(1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn normal_passes_everything_but_the_leader() {
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
            (KeyMode::Normal, Disposition::Consumed(Some(TabAction::New)))
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('t'), Mods::NONE),
            (KeyMode::Normal, Disposition::Consumed(Some(TabAction::New)))
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('x'), Mods::NONE),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(TabAction::Close))
            )
        );
        // Ctrl is folded away too: a command typed before the leader's Ctrl comes
        // back up is the same command.
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('c'), Mods::CTRL),
            (KeyMode::Normal, Disposition::Consumed(Some(TabAction::New)))
        );
        assert_eq!(
            advance(KeyMode::Leader, Key::plain('x'), Mods::CTRL),
            (
                KeyMode::Normal,
                Disposition::Consumed(Some(TabAction::Close))
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
            if let Disposition::Consumed(Some(TabAction::New)) = disposition {
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
            (Key::Up, TabAction::Prev, TabAction::MovePrev),
            (Key::Left, TabAction::Prev, TabAction::MovePrev),
            (Key::Down, TabAction::Next, TabAction::MoveNext),
            (Key::Right, TabAction::Next, TabAction::MoveNext),
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
                    .all(|(_, cluster)| term_render::display_cluster_width(cluster) == 1));
            }
        }
    }
}
