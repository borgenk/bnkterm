//! bnkterm: a native Wayland + Vulkan terminal emulator in Rust, with almost no
//! dependencies.
//!
//! The crate is split lib + thin bin on purpose. The terminal core (`vt`,
//! `grid`, `color`, ...) is greenfield we build stage by stage, and a stage's
//! types routinely exist a step before their first caller does. A library keeps
//! that honest: its public surface is reachable API, so a not-yet-wired type is
//! not "dead code" to be silenced with an `#[allow]`, and the core stays
//! testable without opening a window (the binary is just the shell around it).
//!
//! The stage boundaries the modules fall along:
//!
//! ```text
//!   bytes ─▶ vt::Parser ─▶ grid::Screen ─▶ display list ─▶ GPU
//!           (state machine) (cells, cursor,   (frame as
//!                            scrollback)        data)
//! ```
//!
//! `color` is the shared vocabulary the grid and the eventual renderer both
//! speak; `error` is the terminal-agnostic foundation.

pub mod app;
pub mod color;
pub mod error;
pub mod grid;
pub mod input;
pub mod mouse;
pub mod pty;
pub mod term_render;
pub mod vt;
pub mod width;

// The platform (Wayland, input, font, foundation) and render (display list, GPU
// batcher, Vulkan backend) layers are self-contained leaves, each with a
// boundary test that proves it in isolation. They carry more surface than a
// terminal currently drives (URL opening, the cursor-shape protocol, grapheme
// word boundaries, ...), so some fns, constants, and re-exports read as unused
// without being dead: they are latent capability the leaves keep intact. The
// allow says so once here instead of scattered through the leaf files.
#[allow(dead_code, unused_imports)]
mod platform;
#[allow(dead_code, unused_imports)]
mod render;
