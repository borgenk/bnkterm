//! The clipboard: owning the selection to serve a copy, and receiving another
//! app's selection to paste. This is the `wl_data_device` dance, trimmed to text
//! (a terminal copies text, not images). Copy puts
//! the grid selection on the clipboard; paste writes the received text to the
//! child, wrapped in the bracketed-paste markers when the program asked for them
//! (`?2004`), so an editor can tell a paste from typing.
//!
//! ```text
//!   copy:  selection text ─▶ wl_data_source (offer text mimes) ─▶ set_selection
//!   paste: wl_data_offer.receive(mime, pipe) ─▶ read to EOF ─▶ (bracket?) ─▶ PTY
//! ```

use std::os::fd::AsRawFd;

use super::message::ToTerminal;
use super::State;
use crate::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::protocol::{
    wl_data_device, wl_data_device_manager, wl_data_offer, wl_data_source,
};
use crate::platform::wire::{Arg, Reader};

/// The MIME types we advertise on copy and accept on paste, preferred first.
pub(super) const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

/// Clipboard bookkeeping. When we own the selection, `source` is our live
/// `wl_data_source` and `data` the bytes it serves. Incoming offers are tracked
/// so a paste can ask for a text MIME.
pub(super) struct ClipboardState {
    pub(super) source: u32,
    pub(super) data: Vec<u8>,
    pub(super) incoming_offer: u32,
    pub(super) incoming_text_mime: Option<String>,
    pub(super) selection_offer: u32,
    pub(super) selection_text_mime: Option<String>,
}

impl ClipboardState {
    pub(super) fn new() -> Self {
        Self {
            source: 0,
            data: Vec::new(),
            incoming_offer: 0,
            incoming_text_mime: None,
            selection_offer: 0,
            selection_text_mime: None,
        }
    }
}

impl State {
    /// A `wl_data_device` event: a new offer being introduced, or the selection
    /// changing to one (or to null).
    pub(super) fn on_data_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_device::EV_DATA_OFFER => {
                self.clipboard.incoming_offer = r.u32()?;
                self.clipboard.incoming_text_mime = None;
            }
            wl_data_device::EV_SELECTION => {
                let offer = r.u32()?;
                if self.clipboard.selection_offer != 0 && self.clipboard.selection_offer != offer {
                    self.conn
                        .request(self.clipboard.selection_offer, wl_data_offer::DESTROY, &[]);
                }
                self.clipboard.selection_offer = offer;
                self.clipboard.selection_text_mime =
                    if offer != 0 && offer == self.clipboard.incoming_offer {
                        self.clipboard.incoming_text_mime.clone()
                    } else {
                        None
                    };
            }
            _ => {}
        }
        Ok(())
    }

    /// A `wl_data_source` event on the source we own: serve our bytes to a paster,
    /// or tear the source down when another selection replaces ours.
    pub(super) fn on_data_source(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_source::EV_SEND => {
                // Every MIME we advertise serves the same bytes.
                let _mime = r.string()?;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("data_source.send arrived without its fd"))?;
                // A paster that closes its read end early makes this EPIPE; serving
                // the clipboard must never take the terminal down, so report and go on.
                if let Err(e) = ffi::write_all(fd.as_raw_fd(), &self.clipboard.data) {
                    eprintln!("bnkterm: clipboard serve failed: {e}");
                }
            }
            wl_data_source::EV_CANCELLED => {
                if self.clipboard.source != 0 {
                    self.conn
                        .request(self.clipboard.source, wl_data_source::DESTROY, &[]);
                }
                self.clipboard.source = 0;
                self.clipboard.data.clear();
            }
            _ => {}
        }
        Ok(())
    }

    /// One MIME type of the incoming offer; remember the first (preferred) text one
    /// so a paste knows what to request.
    pub(super) fn on_offer_mime(&mut self, r: &mut Reader) -> Result<()> {
        let mime = r.string()?;
        if is_text_mime(mime)
            && (self.clipboard.incoming_text_mime.is_none() || mime == CLIPBOARD_MIMES[0])
        {
            self.clipboard.incoming_text_mime = Some(mime.to_string());
        }
        Ok(())
    }

    /// Become the clipboard owner serving `data` as text, replacing any source we
    /// held. Needs the data device (skips silently without it). Called from the
    /// outbox drain when the core reports a fresh selection to own.
    pub(super) fn set_clipboard(&mut self, data: Vec<u8>) {
        let (Some(manager), true) = (self.data_device_manager, self.data_device != 0) else {
            return;
        };
        if self.clipboard.source != 0 {
            self.conn
                .request(self.clipboard.source, wl_data_source::DESTROY, &[]);
        }
        let source = self.alloc_id();
        self.conn.request(
            manager,
            wl_data_device_manager::CREATE_DATA_SOURCE,
            &[Arg::NewId(source)],
        );
        for mime in CLIPBOARD_MIMES {
            self.conn
                .request(source, wl_data_source::OFFER, &[Arg::Str(mime)]);
        }
        self.conn.request(
            self.data_device,
            wl_data_device::SET_SELECTION,
            &[Arg::Object(source), Arg::Uint(self.last_serial)],
        );
        self.clipboard.source = source;
        self.clipboard.data = data;
    }

    /// Paste the clipboard's text to the child: newlines become `CR` (as if the
    /// lines were typed and entered), and the whole run is wrapped in the
    /// bracketed-paste markers when the program enabled them (`?2004`). Snaps the
    /// view to the bottom, like any input.
    pub(super) fn paste(&mut self) -> Result<()> {
        // The window fetches the clipboard text (the data-device dance below); the
        // terminal normalizes, brackets, and writes it (see `TerminalCore::apply`).
        if let Some(text) = self.clipboard_text()? {
            self.core.apply(ToTerminal::Paste(text.into_bytes()))?;
        }
        Ok(())
    }

    /// The clipboard's text, or `None`. When we own the clipboard our own bytes are
    /// returned directly: routing through the compositor would deadlock (we would
    /// block reading the pipe while also owing our source a `send`). Otherwise the
    /// bytes come over a pipe (give the compositor the write end, read to EOF).
    fn clipboard_text(&mut self) -> Result<Option<String>> {
        if self.clipboard.source != 0 {
            return Ok(Some(
                String::from_utf8_lossy(&self.clipboard.data).into_owned(),
            ));
        }
        let (Some(mime), true) = (
            self.clipboard.selection_text_mime.clone(),
            self.clipboard.selection_offer != 0,
        ) else {
            return Ok(None);
        };
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            self.clipboard.selection_offer,
            wl_data_offer::RECEIVE,
            &[Arg::Str(&mime)],
            write_end.as_raw_fd(),
        )?;
        drop(write_end);
        let bytes = ffi::read_to_end(read_end.as_raw_fd())?;
        Ok(String::from_utf8(bytes).ok())
    }
}

/// Whether a clipboard MIME type carries text we can paste.
pub(super) fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/") || mime == "UTF8_STRING"
}
