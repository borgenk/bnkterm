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

#![allow(rustdoc::private_intra_doc_links)]

pub mod app;
pub mod color;
pub mod config;
#[cfg(test)]
mod dev;
pub mod error;
pub mod flatpak;
pub mod gather;
pub mod grid;
pub mod input;
mod keymode;
pub mod mouse;
mod notice;
mod platform;
pub mod pty;
mod render;
pub mod shell_integration;
mod tab_bar;
pub mod term_render;
pub mod vt;
pub mod width;
