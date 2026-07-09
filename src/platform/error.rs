//! A minimal error type. bnkterm has no dependencies, so there is no
//! anyhow/thiserror; a string-backed error with the few `From` conversions we
//! need is enough, and it keeps every fallible path returning `Result`.

use std::fmt;

#[derive(Debug)]
pub struct Error {
    msg: String,
}

impl Error {
    pub fn msg(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::msg(format!("io error: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
