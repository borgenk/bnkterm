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
