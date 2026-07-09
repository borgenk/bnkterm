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
