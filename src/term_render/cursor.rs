//! The cursor, painted after the rows so it stacks over the cell it sits on: a block that
//! inverts the glyph under it, a hollow block while the window is unfocused, a bar, an
//! underline, or the padlock that stands in for it while the tty takes a password.

use crate::grid::Cell;
use crate::platform::freetype::{FaceKey, FontStyle};
use crate::platform::geom::Rect;
use crate::render::display::DrawCmd;
use crate::term_render::{
    push_owned_cells, style_of, take_string, CursorRender, CursorShape, Painter, LOCK_GLYPH,
};

/// The width of a bar (`DECSCUSR 5/6`) cursor, in pixels.
const BAR_CURSOR_WIDTH: i32 = 2;

impl Painter<'_> {
    /// The cursor, drawn last so it stacks over the cell it sits on. Nothing is
    /// drawn when the cursor is hidden or its position is off the grid. A block
    /// cursor on a wide glyph covers both halves and, when focused, re-stamps the
    /// glyph in the cell background so it reads as inverted.
    pub(super) fn cursor(&mut self, cursor: CursorRender, cols: usize, rows: usize) {
        if !cursor.visible {
            return;
        }
        let (cr, mut cc) = self.screen.cursor();
        // The live cursor row maps to `cr + view_offset` on screen; when scrolled
        // far enough into history it falls below the window and is not drawn.
        let Some(dr) = cr.checked_add(self.screen.view_offset()) else {
            return;
        };
        if dr >= rows || cc >= cols {
            return;
        }
        // A cursor parked on a wide spacer belongs to the leader on its left.
        if self.cell(dr, cc).is_wide_spacer() && cc > 0 {
            cc -= 1;
        }
        let cell = self.cell(dr, cc);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let m = self.metrics;
        let x = self.cell_x(cc);
        let y = self.cell_y(dr);
        let color = self.theme.cursor.to_u32();
        let w = width_cells * m.w;
        match cursor.shape {
            CursorShape::Block => self.block_cursor(dr, cc, cursor.focused),
            // The padlock stands in for the cursor, in the cursor's colour and on the cell's
            // own background: no block behind it, and it looks the same focused or not. It
            // is a status light rather than a caret style, and "this window is waiting for a
            // password" is worth reading from across the desk.
            //
            // Without a symbols font there is no padlock to draw, and a tofu box is worse
            // than no lock at all, so the cursor quietly stays whatever the child asked for.
            CursorShape::Lock if m.lock_glyph => {
                // The lock is the whole message, so it gets a clean cell: paint out whatever
                // is under the cursor (blank at a real password prompt, but nothing promises
                // where the cursor is parked) and stand the glyph on that background.
                let behind = self.cell_bg_of(cell);
                self.list.push(DrawCmd::Fill {
                    rect: Rect { x, y, w, h: m.h },
                    color: behind.to_u32(),
                });
                let baseline = self.baseline(dr);
                let mut text = take_string(self.strings);
                text.push(LOCK_GLYPH);
                let ink = self.theme.cursor;
                self.push_glyph_text(x, baseline, 1, FontStyle::Regular, (ink, behind), text);
            }
            CursorShape::Lock => self.block_cursor(dr, cc, cursor.focused),
            CursorShape::Bar => self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y,
                    w: BAR_CURSOR_WIDTH,
                    h: m.h,
                },
                color,
            }),
            CursorShape::Underline => {
                let thickness = (m.size as i32 / 8).max(2);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y: y + m.h - thickness,
                        w: width_cells * m.w,
                        h: thickness,
                    },
                    color,
                });
            }
        }
    }

    /// Re-draw the glyph beneath a focused block cursor in the cell's background
    /// colour, so the character shows as a cutout in the cursor block.
    ///
    /// Drawn through the same command as the row, so a box, block, or braille glyph
    /// stays procedural under the cursor and a narrow emoji keeps its cell size.
    ///
    /// A concealed cell has no glyph to cut out: [`Shown`](crate::term_render::Shown)
    /// returns a space with no ink, and this draws nothing. That is the point of asking
    /// rather than reading the cell directly: the cursor is the one place a masked field
    /// is guaranteed to have a freshly typed character sitting under it.
    fn stamp_inverted_glyph(&mut self, row: usize, col: usize, cell: Cell, width_cells: i32) {
        let shown = self.shown(row, col, cell);
        if !shown.inked {
            return;
        }
        let m = self.metrics;
        let x = self.cell_x(col);
        let baseline = self.baseline(row);
        let mut text = take_string(self.strings);
        text.push(shown.rune);
        if shown.marks {
            text.extend(self.marks(row, col));
        }
        // The inverted glyph takes the cell's own background (reverse honoured),
        // ignoring any selection so the cursor stays legible over a selection. It is
        // stamped over the cursor block, so that colour is the background the glyph
        // anti-aliasing is weighted against.
        let ink = self.cell_bg_of(cell);
        let style = style_of(cell.attrs);
        let cursor = self.theme.cursor;
        if cell.is_wide_leader() {
            self.push_glyph_text(x, baseline, width_cells, style, (ink, cursor), text);
        } else {
            let face = FaceKey::Prose {
                size: m.size,
                style,
            };
            let (color, bg) = (ink.to_u32(), cursor.to_u32());
            push_owned_cells(self.list, text, x, baseline, 1, m, face, color, bg);
        }
    }

    /// The block cursor: filled with the glyph inverted out of it when the window has
    /// focus, a hollow outline when it does not (the xterm convention). Shared with the
    /// lock, which falls back to it on a cell too small to draw a padlock in.
    fn block_cursor(&mut self, row: usize, col: usize, focused: bool) {
        let m = self.metrics;
        let cell = self.cell(row, col);
        let width_cells = if cell.is_wide_leader() { 2 } else { 1 };
        let (x, y) = (self.cell_x(col), self.cell_y(row));
        if !focused {
            self.hollow_block(x, y, width_cells * m.w);
            return;
        }
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x,
                y,
                w: width_cells * m.w,
                h: m.h,
            },
            color: self.theme.cursor.to_u32(),
        });
        self.stamp_inverted_glyph(row, col, cell, width_cells);
    }

    /// A hollow block outline (unfocused window) as four thin edge fills.
    fn hollow_block(&mut self, x: i32, y: i32, w: i32) {
        let m = self.metrics;
        let color = self.theme.cursor.to_u32();
        let t = 1;
        let edges = [
            Rect { x, y, w, h: t }, // top
            Rect {
                x,
                y: y + m.h - t,
                w,
                h: t,
            }, // bottom
            Rect { x, y, w: t, h: m.h }, // left
            Rect {
                x: x + w - t,
                y,
                w: t,
                h: m.h,
            }, // right
        ];
        for rect in edges {
            self.list.push(DrawCmd::Fill { rect, color });
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::term_render::tests::*;
    use crate::term_render::*;
    use crate::testutil::feed;

    #[test]
    fn block_cursor_fills_its_cell_and_inverts_the_glyph() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X"); // cursor now rests at column 1 (after X)
        s.move_to(0, 0); // park it on the glyph
        let t = Theme::default();
        let list = build_display_list(&FrameInputs {
            cursor: CursorRender::default(),
            ..inputs(&s, &t)
        });
        // The last commands are the cursor block then the inverted glyph.
        let cursor_fill = fills(&list)
            .into_iter()
            .find(|(r, c)| r.x == 0 && r.w == M.w && *c == t.cursor.to_u32())
            .expect("a filled cursor block");
        assert_eq!(
            cursor_fill.0,
            Rect {
                x: 0,
                y: 0,
                w: M.w,
                h: M.h
            }
        );
        // The inverted 'X' is drawn last, in the background colour.
        match list.last().unwrap() {
            DrawCmd::Cells { text, color, .. } => {
                assert_eq!(text, "X");
                assert_eq!(*color, t.bg.to_u32(), "the cursor glyph is inverted");
            }
            other => panic!("expected the inverted glyph last, got {other:?}"),
        }
    }

    #[test]
    fn bar_and_underline_cursors_do_not_invert() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let t = Theme::default();
        for shape in [CursorShape::Bar, CursorShape::Underline] {
            let list = build_display_list(&FrameInputs {
                cursor: CursorRender {
                    shape,
                    ..CursorRender::default()
                },
                ..inputs(&s, &t)
            });
            // No inverted glyph: the only Text/Cells for 'X' is the normal one.
            let glyphs = list
                .iter()
                .filter(|c| matches!(c, DrawCmd::Cells { .. } | DrawCmd::Text { .. }))
                .count();
            assert_eq!(glyphs, 1, "{shape:?} leaves the glyph alone");
        }
    }

    #[test]
    fn a_concealed_cell_is_still_concealed_under_the_cursor() {
        // `SGR 8` is what a TUI masking a password field sends when it will not or cannot
        // use tty echo-off, and the cursor is the one cell of such a field that is
        // *guaranteed* to hold a character the user just typed. A cursor path reading the
        // cell's rune directly therefore leaks the password back onto the screen at one
        // character per keypress — the conceal holding everywhere except the place it
        // matters most.
        //
        // Stated as an indistinguishability: two concealed cells differing only in what
        // they conceal must paint identically. That catches a leak through any command,
        // not just the inverted stamp this was found in.
        let t = Theme::default();
        let frame = |rune: &[u8]| {
            let mut s = Screen::new(4, 1);
            feed(&mut s, b"\x1b[8m");
            feed(&mut s, rune);
            s.move_to(0, 0);
            build_display_list(&FrameInputs {
                cursor: CursorRender::default(), // visible, focused: the block that inverts
                ..inputs(&s, &t)
            })
        };

        let secret = frame(b"s");
        assert_eq!(
            secret,
            frame(b" "),
            "a concealed 's' painted differently from a concealed space"
        );
        assert!(
            !secret.iter().any(|c| matches!(
                c,
                DrawCmd::Text { text, .. } | DrawCmd::Cells { text, .. } if text.contains('s')
            )),
            "the concealed character reached the display list: {secret:?}"
        );

        // And the conceal is not a blanket "draw nothing under the cursor": an ordinary
        // cell still gets its glyph inverted out of the block.
        let mut plain = Screen::new(4, 1);
        feed(&mut plain, b"s");
        plain.move_to(0, 0);
        let list = build_display_list(&FrameInputs {
            cursor: CursorRender::default(),
            ..inputs(&plain, &t)
        });
        assert!(
            matches!(list.last(), Some(DrawCmd::Cells { text, .. }) if text == "s"),
            "an unconcealed cell still stamps: {list:?}"
        );
    }

    /// The cursor stamp is the row's own command for the cell, with inverted colours.
    #[test]
    fn the_cursor_stamps_a_cell_through_the_command_its_row_draws_it_with() {
        let t = Theme::default();
        for glyph in ["X", "⠂", "─", "▛", "⚠", "漢", "\u{1D400}"] {
            let mut s = Screen::new(4, 1);
            feed(&mut s, glyph.as_bytes());
            s.move_to(0, 0);
            let row = list_of(&s);
            let mut want = row
                .iter()
                .find(|c| matches!(c, DrawCmd::Cells { .. } | DrawCmd::Text { .. }))
                .unwrap_or_else(|| panic!("{glyph:?} draws a glyph: {row:?}"))
                .clone();
            match &mut want {
                DrawCmd::Cells { color, bg, .. } | DrawCmd::Text { color, bg, .. } => {
                    *color = t.bg.to_u32();
                    *bg = t.cursor.to_u32();
                }
                other => panic!("not a glyph command: {other:?}"),
            }
            let stamped = build_display_list(&FrameInputs {
                cursor: CursorRender::default(),
                ..inputs(&s, &t)
            });
            assert_eq!(
                stamped.last(),
                Some(&want),
                "{glyph:?}: the stamp must be the row's own command, inverted"
            );
        }
    }

    /// A `Lock` cursor over a cell holding 'X', at `metrics`. Returns the display list.
    fn lock_frame(metrics: CellMetrics, t: &Theme) -> DisplayList {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0); // park the cursor on the glyph
        build_display_list(&FrameInputs {
            metrics,
            cursor: CursorRender {
                shape: CursorShape::Lock,
                ..CursorRender::default()
            },
            ..inputs(&s, t)
        })
    }

    #[test]
    fn the_lock_typesets_the_padlock_glyph_on_a_cleared_cell() {
        // bnkterm's lock is *their* lock: the same Nerd Fonts codepoint wezterm and ghostty
        // typeset, not a lookalike drawn from rectangles. It stands in the cursor's colour
        // on the cell's own background, with no block behind it to swallow it, and the cell
        // is cleared first so whatever the cursor is parked on cannot show through.
        let t = Theme::default();
        let list = lock_frame(M, &t);

        let cell = Rect {
            x: 0,
            y: 0,
            w: M.w,
            h: M.h,
        };
        assert_eq!(
            fills(&list).last(),
            Some(&(cell, t.bg.to_u32())),
            "the cell is cleared to its own background before the glyph lands"
        );
        match list.last() {
            Some(DrawCmd::Text {
                text,
                color,
                bg,
                x,
                face,
                ..
            }) => {
                assert_eq!(
                    text,
                    &LOCK_GLYPH.to_string(),
                    "the padlock, not a drawn shape"
                );
                assert_eq!(*color, t.cursor.to_u32(), "in the cursor's colour");
                assert_eq!(*bg, t.bg.to_u32());
                assert_eq!(*x, 0, "in the cursor's cell");
                assert_eq!(
                    *face,
                    FaceKey::Prose {
                        size: M.size,
                        style: FontStyle::Regular
                    }
                );
            }
            other => panic!("expected the padlock glyph last, got {other:?}"),
        }
    }

    #[test]
    fn a_font_stack_without_the_padlock_falls_back_to_an_ordinary_cursor() {
        // The padlock is private-use, so a machine with no symbols font simply has no such
        // glyph. Typesetting it anyway would draw a tofu box: a cursor that is not merely
        // absent but actively confusing, on exactly the machines least equipped to explain
        // it. So the lock is silently declined and the caret stays what the child asked for.
        let t = Theme::default();
        let bare = CellMetrics {
            lock_glyph: false,
            ..M
        };
        let list = lock_frame(bare, &t);

        assert!(
            !list.iter().any(|c| matches!(
                c,
                DrawCmd::Text { text, .. } if text.contains(LOCK_GLYPH)
            )),
            "no padlock is typeset when the font stack has none"
        );
        // An ordinary focused block: filled cell, glyph inverted out of it.
        assert!(
            fills(&list).contains(&(
                Rect {
                    x: 0,
                    y: 0,
                    w: M.w,
                    h: M.h
                },
                t.cursor.to_u32()
            )),
            "the cursor falls back to a block, not to nothing"
        );
        match list.last() {
            Some(DrawCmd::Cells { text, color, .. }) => {
                assert_eq!(text, "X", "the inverted glyph of a normal block cursor");
                assert_eq!(*color, t.bg.to_u32());
            }
            other => panic!("expected the inverted glyph last, got {other:?}"),
        }
    }

    #[test]
    fn the_lock_looks_the_same_focused_or_not() {
        // The lock is a status light, not a caret style: "this window is waiting for a
        // password" is worth reading from across the desk, focused or not.
        let t = Theme::default();
        let focused = lock_frame(M, &t);
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let unfocused = build_display_list(&FrameInputs {
            cursor: CursorRender {
                shape: CursorShape::Lock,
                focused: false,
                ..CursorRender::default()
            },
            ..inputs(&s, &t)
        });
        assert_eq!(
            focused.last(),
            unfocused.last(),
            "the padlock does not hollow out when the window loses focus"
        );
    }

    #[test]
    fn hidden_cursor_draws_nothing_extra() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0);
        let hidden = list_of(&s);
        // Same as a plain render: base fill + the one glyph run.
        assert_eq!(hidden.len(), 2);
    }

    #[test]
    fn the_cursor_is_hidden_when_scrolled_out_of_view() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour"); // cursor on the live bottom row
        s.scroll_view_up(2); // live rows scrolled below the window
        let t = Theme::default();
        let list = build_display_list(&FrameInputs {
            cursor: CursorRender::default(), // visible + focused
            ..inputs(&s, &t)
        });
        // No inverted glyph and no cursor block colour: the cursor sits off-view.
        let cursor_color = Theme::default().cursor.to_u32();
        assert!(
            !fills(&list).iter().any(|(_, c)| *c == cursor_color),
            "the cursor is not drawn while viewing history"
        );
    }
}
