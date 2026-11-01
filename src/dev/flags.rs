//! The dev tools' arguments. A tool reads the ones after its mode flag in one pass and
//! refuses any it does not take.

use crate::error::{Error, Result};

/// The argument after `flag`.
pub fn value<S>(flag: &str, args: &mut impl Iterator<Item = S>) -> Result<S> {
    args.next()
        .ok_or_else(|| Error::msg(format!("{flag} needs a value")))
}

/// The argument after `flag`, parsed.
pub fn parsed<T: std::str::FromStr>(
    flag: &str,
    args: &mut impl Iterator<Item = impl AsRef<str>>,
) -> Result<T> {
    let text = value(flag, args)?;
    let text = text.as_ref();
    text.parse()
        .map_err(|_| Error::msg(format!("{flag}: not a valid value: {text}")))
}

pub fn unknown(arg: &str) -> Error {
    Error::msg(format!("unknown argument {arg:?}"))
}
