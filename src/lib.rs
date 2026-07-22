//! The modules fall along these stage boundaries:
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
pub mod config;
pub mod error;
/// The property lane's shared machinery (a seeded generator and a shrinker), compiled
/// only under test so none of it can reach the shipping binary.
#[cfg(test)]
mod fuzz;
pub mod gather;
pub mod grid;
pub mod input;
mod keymode;
pub mod mouse;
pub mod pty;
pub mod shell_integration;
mod tab_bar;
pub mod term_render;
pub mod vt;
pub mod width;

// The platform (Wayland, input, font, foundation) and render (display list, GPU
// batcher, Vulkan backend) layers are self-contained leaves, each with a
// boundary test that proves it in isolation. They carry more surface than a
// terminal currently drives (grapheme word boundaries, parts of the Wayland
// protocol, ...), so some fns, constants, and re-exports read as unused without
// being dead: they are latent capability the leaves keep intact. The allow says
// so once here instead of scattered through the leaf files.
#[allow(dead_code, unused_imports)]
mod platform;
#[allow(dead_code, unused_imports)]
mod render;
