//! Tooling for working on the terminal, compiled only under `cfg(test)` and reached
//! through `make` targets, so none of it can reach the shipping binary.
//!
//! The property lane's seeded generator and shrinker (`fuzz`), the PNG encoder that
//! writes a captured frame out (`png`), and the harness that drives a window and
//! saves a screenshot of it (`screenshot`).

pub mod fuzz;
pub mod png;
pub mod screenshot;
