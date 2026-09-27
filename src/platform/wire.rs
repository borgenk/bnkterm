//! Wayland wire-format encoding and decoding.
//!
//! A message is: `u32 object_id`, then `u32 (size << 16 | opcode)` where size
//! is the whole message length in bytes including this 8-byte header, then the
//! arguments. Everything is host byte order (little-endian on our targets), and
//! every argument is padded to a 4-byte boundary. File descriptors are not in
//! the byte stream; they travel as SCM_RIGHTS ancillary data (see `ffi`).

use crate::platform::error::{Error, Result};

/// One request argument. The `Bind` variant is the "generic" new-id used only
/// by `wl_registry.bind`, where the interface is not known statically and is
/// therefore sent inline as interface-string + version + id.
pub enum Arg<'a> {
    Int(i32),
    Uint(u32),
    Object(u32),
    NewId(u32),
    Str(&'a str),
    Bind {
        interface: &'a str,
        version: u32,
        new_id: u32,
    },
}

/// A fully received event: its target object, opcode, and the argument bytes
/// after the header. Owned so the connection buffer can advance independently.
/// [`Default`] gives an empty one to reuse as decode scratch across the event loop,
/// so a warmed frame decodes into the same body buffer without allocating.
#[derive(Default)]
pub struct Message {
    pub object: u32,
    pub opcode: u16,
    pub body: Vec<u8>,
}

/// The largest request a compositor accepts. libwayland reads each client through a
/// 4096-byte connection buffer unless the compositor raises it, and a request that does
/// not fit is a protocol error that closes the connection.
const MAX_REQUEST: usize = 4096;

/// The longest string a request can carry as its only argument: [`MAX_REQUEST`] less the
/// 8-byte header, the string's 4-byte length and its NUL. That total is already 4-byte
/// aligned, so no padding is left over.
const MAX_SOLE_STRING: usize = MAX_REQUEST - 8 - 4 - 1;

/// `s` cut at a character boundary so that a request carrying it as its only argument
/// fits in [`MAX_REQUEST`]. For strings the child controls, such as the window title.
pub fn fit_sole_string(s: &str) -> &str {
    let end = s.floor_char_boundary(MAX_SOLE_STRING);
    s.get(..end).unwrap_or_default()
}

/// Append an encoded request to `buf`.
///
/// A request larger than [`MAX_REQUEST`] is **dropped**, not sent: the compositor would
/// answer it with a protocol error and close the connection, taking every tab with it.
/// Callers with variable-length input fit it first ([`fit_sole_string`]); this check is
/// the backstop that keeps a caller that did not from being fatal. Dropped rather than
/// truncated, because a cut frame would put the next request's bytes where this one's
/// arguments should be. The same bound keeps the length inside the header's 16-bit size
/// field.
pub fn encode(buf: &mut Vec<u8>, object: u32, opcode: u16, args: &[Arg]) {
    let start = buf.len();
    buf.extend_from_slice(&object.to_ne_bytes());
    buf.extend_from_slice(&[0u8; 4]); // size|opcode, patched once length is known
    for arg in args {
        encode_arg(buf, arg);
    }
    let size = match u16::try_from(buf.len() - start) {
        Ok(size) if usize::from(size) <= MAX_REQUEST => size,
        _ => {
            buf.truncate(start);
            return;
        }
    };
    let word = (u32::from(size) << 16) | u32::from(opcode);
    if let Some(header) = buf.get_mut(start + 4..start + 8) {
        header.copy_from_slice(&word.to_ne_bytes());
    }
}

fn encode_arg(buf: &mut Vec<u8>, arg: &Arg) {
    match arg {
        Arg::Int(v) => buf.extend_from_slice(&v.to_ne_bytes()),
        Arg::Uint(v) | Arg::Object(v) | Arg::NewId(v) => buf.extend_from_slice(&v.to_ne_bytes()),
        Arg::Str(s) => encode_str(buf, s),
        Arg::Bind {
            interface,
            version,
            new_id,
        } => {
            encode_str(buf, interface);
            buf.extend_from_slice(&version.to_ne_bytes());
            buf.extend_from_slice(&new_id.to_ne_bytes());
        }
    }
}

/// Encode a string: length (including the trailing NUL) then the bytes, the
/// NUL, and zero padding up to a 4-byte boundary.
fn encode_str(buf: &mut Vec<u8>, s: &str) {
    let len = s.len() as u32 + 1;
    buf.extend_from_slice(&len.to_ne_bytes());
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

/// Sequential reader over an event's argument bytes, bounds-checked, decoding each
/// argument in native byte order (the wire format is host-endian).
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// The next `n` bytes, advancing past them. A short message is an error and leaves
    /// the reader where it was, so a caller that recovers reads the same bytes again
    /// rather than a shifted view of them.
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::msg("truncated wayland message"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| Error::msg("truncated wayland message"))?;
        self.pos = end;
        Ok(slice)
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a `wl_fixed`: a signed 24.8 fixed-point number, as pixels. Used for
    /// pointer coordinates.
    pub fn fixed(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        let raw = i32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
        Ok(raw as f32 / 256.0)
    }

    /// Read an array argument: a length-prefixed blob of bytes, padded to a
    /// 4-byte boundary. Returned as raw bytes; callers that only need to skip it
    /// (like the `states` array in xdg_toplevel.configure) ignore the result.
    pub fn array(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        let pad = (4 - (len % 4)) % 4;
        let _ = self.take(pad)?;
        Ok(bytes)
    }

    /// Read a string argument, dropping its trailing NUL and its padding.
    pub fn string(&mut self) -> Result<&'a str> {
        let len = self.u32()? as usize;
        if len == 0 {
            return Ok("");
        }
        let bytes = self.take(len)?;
        let pad = (4 - (len % 4)) % 4;
        let _ = self.take(pad)?;
        core::str::from_utf8(&bytes[..len - 1])
            .map_err(|_| Error::msg("invalid utf-8 in wayland string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_carries_size_and_opcode() {
        let mut buf = Vec::new();
        encode(&mut buf, 5, 2, &[Arg::Uint(0xdead_beef), Arg::Int(-1)]);
        assert_eq!(buf.len(), 8 + 4 + 4);
        let object = u32::from_ne_bytes(buf[0..4].try_into().unwrap());
        let word = u32::from_ne_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(object, 5);
        assert_eq!(word >> 16, 16, "size in bytes");
        assert_eq!(word & 0xffff, 2, "opcode");
    }

    #[test]
    fn string_then_uint_roundtrips_with_padding() {
        let mut buf = Vec::new();
        encode(&mut buf, 1, 0, &[Arg::Str("wl_shm"), Arg::Uint(7)]);
        // String: 4 (len) + "wl_shm\0" (7) padded to 8 = 12 bytes. Plus the
        // uint (4) and the 8-byte header.
        assert_eq!(buf.len(), 8 + 12 + 4);
        let mut r = Reader::new(&buf[8..]);
        assert_eq!(r.string().unwrap(), "wl_shm");
        assert_eq!(r.u32().unwrap(), 7);
    }

    #[test]
    fn bind_encodes_interface_version_and_id() {
        let mut buf = Vec::new();
        encode(
            &mut buf,
            2,
            0,
            &[
                Arg::Uint(3),
                Arg::Bind {
                    interface: "xdg_wm_base",
                    version: 1,
                    new_id: 9,
                },
            ],
        );
        let mut r = Reader::new(&buf[8..]);
        assert_eq!(r.u32().unwrap(), 3, "global name");
        assert_eq!(r.string().unwrap(), "xdg_wm_base");
        assert_eq!(r.u32().unwrap(), 1, "version");
        assert_eq!(r.u32().unwrap(), 9, "new id");
    }

    #[test]
    fn a_request_past_the_compositor_limit_is_dropped() {
        // The limit a libwayland compositor enforces, measured against weston on
        // libwayland 1.26: a 4096-byte set_title is accepted, and 4100 bytes draws a
        // protocol error and a closed connection.
        let mut buf = Vec::new();
        encode(&mut buf, 1, 0, &[Arg::Uint(7)]);
        let queued = buf.clone();

        // One word over is dropped whole, as is one too long for the header's 16-bit
        // size field, and what was already queued stays intact.
        encode(&mut buf, 8, 2, &[Arg::Str(&"x".repeat(4084))]);
        encode(&mut buf, 8, 2, &[Arg::Str(&"x".repeat(0x1_0000))]);
        assert_eq!(buf.len(), queued.len(), "an oversized request was queued");
        assert_eq!(buf, queued);

        // The stream carries on: the next request encodes at the offset the dropped
        // ones would have taken.
        encode(&mut buf, 4, 5, &[Arg::Uint(9)]);
        let word = u32::from_ne_bytes(buf[queued.len() + 4..queued.len() + 8].try_into().unwrap());
        assert_eq!((word >> 16, word & 0xffff), (12, 5));

        // The refusal is at the limit and not short of it: 4083 bytes of string, its
        // 4-byte length and the 8-byte header fill 4096 exactly.
        let mut buf = Vec::new();
        encode(&mut buf, 8, 2, &[Arg::Str(&"x".repeat(4083))]);
        assert_eq!(buf.len(), 4096);
    }

    #[test]
    fn a_fitted_string_makes_a_request_that_is_sent() {
        // A child's title can be ~12 KiB. Fitting cuts it on a character boundary (the
        // two-byte "é" lands one byte short of the 4083 limit) to a request the encoder
        // keeps, so a long title still reaches the window.
        let long = "é".repeat(3000);
        let fitted = fit_sole_string(&long);
        assert_eq!(fitted.len(), 4082);
        assert!(long.starts_with(fitted));
        let mut buf = Vec::new();
        encode(&mut buf, 8, 2, &[Arg::Str(fitted)]);
        assert_eq!(buf.len(), 4096);

        assert_eq!(fit_sole_string("bnkterm"), "bnkterm");
    }

    #[test]
    fn reader_rejects_truncation() {
        let mut r = Reader::new(&[0u8, 0, 0]);
        assert!(r.u32().is_err());

        // A refused read leaves the position alone, so the three bytes are still
        // there to be read another way. A length that would overflow the offset is
        // refused too, rather than wrapping into an in-bounds slice.
        let mut r = Reader::new(&[1u8, 2, 3, 4, 5, 6, 7]);
        assert_eq!(r.u32().unwrap(), u32::from_ne_bytes([1, 2, 3, 4]));
        assert!(r.u32().is_err(), "three bytes left, four wanted");
        assert!(r.take(usize::MAX).is_err());
        assert_eq!(r.take(3).unwrap(), &[5, 6, 7]);
    }

    #[test]
    fn array_reads_length_prefixed_bytes_with_padding() {
        // A 5-byte array: u32 length, 5 bytes, then 3 padding bytes (to 8). A
        // trailing uint proves the reader resumed at the right offset.
        let mut buf = Vec::new();
        buf.extend_from_slice(&5u32.to_ne_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 0, 0, 0]);
        buf.extend_from_slice(&99u32.to_ne_bytes());
        let mut r = Reader::new(&buf);
        assert_eq!(r.array().unwrap(), &[1, 2, 3, 4, 5]);
        assert_eq!(r.u32().unwrap(), 99);
    }

    #[test]
    fn fixed_decodes_signed_24_8() {
        // 256 (0x100) is 1.0; -256 is -1.0; 128 is 0.5.
        let mut buf = Vec::new();
        for raw in [256i32, -256, 128] {
            buf.extend_from_slice(&raw.to_ne_bytes());
        }
        let mut r = Reader::new(&buf);
        assert_eq!(r.fixed().unwrap(), 1.0);
        assert_eq!(r.fixed().unwrap(), -1.0);
        assert_eq!(r.fixed().unwrap(), 0.5);
    }
}
