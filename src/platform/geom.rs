//! Window-pixel geometry primitives shared by every layer: the app lays lines
//! out in them, the painter clips to them, and the GPU backend scissors to them.

/// An axis-aligned rectangle in window pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    /// Whether the window point `(px, py)` falls inside this rectangle, left/top
    /// inclusive and right/bottom exclusive, as the pointer hit-tests want.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        let (x, y) = (px as i32, py as i32);
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}
