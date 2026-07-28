//! The two text selections: the clipboard (Ctrl+Shift+C/V) and the primary
//! selection (select-to-copy, middle-click paste). Both are the same
//! `wl_data_device`-style dance, trimmed to text (a terminal copies text, not
//! images): own a source to serve a copy, track an incoming offer to paste. The
//! primary selection is a near-twin of the clipboard without drag-and-drop, so one
//! code path drives both, keyed by a [`SelectionOps`] table of the wire opcodes
//! that differ; only the object families and those opcodes change.
//!
//! ```text
//!   copy:  selection text ─▶ source (offer text mimes) ─▶ set_selection
//!   paste: offer.receive(mime, pipe) ─▶ read to EOF ─▶ (bracket?) ─▶ PTY
//! ```

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::time::Duration;

use super::message::ToTerminal;
use super::State;
use crate::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::protocol::{
    wl_data_device, wl_data_device_manager, wl_data_offer, wl_data_source,
    zwp_primary_selection_device_manager_v1, zwp_primary_selection_device_v1,
    zwp_primary_selection_offer_v1, zwp_primary_selection_source_v1,
};
use crate::platform::wire::{Arg, Reader};

/// The MIME types we advertise on copy and accept on paste, preferred first.
pub(super) const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

/// How long a paste waits on the clipboard's owner before giving up, and how much it
/// will accept. The owner is another application, and the wait is on the thread that
/// renders and handles keys, so neither can be unbounded. Half a second is far beyond
/// any real transfer (they finish in microseconds, in one `read`) and short enough that
/// a broken peer costs a dropped paste rather than a hung terminal. 16 MiB is likewise
/// far past anything a person pastes into a shell.
const PASTE_BUDGET: Duration = Duration::from_millis(500);
const PASTE_MAX: usize = 16 * 1024 * 1024;

/// How much of a selection can still be owed to a paster before we stop tracking it.
/// Serving is asynchronous (see [`State::pump_selection_sends`]) and a receiver that
/// never reads would otherwise pin our copy of the data forever; four concurrent
/// transfers is already generous, since each is one application asking for one paste.
const MAX_PENDING_SENDS: usize = 4;

/// A selection transfer in flight: the pipe the compositor handed us, our copy of the
/// bytes, and how far into them the receiver has taken. The data is owned rather than
/// borrowed because the selection can be replaced (or the source cancelled, which clears
/// it) while a paster is still reading the old one.
pub(super) struct PendingSend {
    fd: OwnedFd,
    data: Vec<u8>,
    head: usize,
}

impl PendingSend {
    /// The pipe to watch for writability while this transfer is still owed bytes.
    pub(super) fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
}

/// Which of the two selection transports a request or event belongs to. They share
/// one state machine over two object families, so the code is written once and the
/// caller picks the transport.
#[derive(Clone, Copy)]
enum Transport {
    Clipboard,
    Primary,
}

/// The wire opcodes that differ between the clipboard and the primary selection;
/// the message flow over them is identical. One table per transport, selected by
/// [`State::ops`].
struct SelectionOps {
    create_source: u16,
    source_offer: u16,
    source_destroy: u16,
    ev_send: u16,
    ev_cancelled: u16,
    set_selection: u16,
    ev_data_offer: u16,
    ev_selection: u16,
    offer_receive: u16,
    offer_destroy: u16,
}

const CLIPBOARD_OPS: SelectionOps = SelectionOps {
    create_source: wl_data_device_manager::CREATE_DATA_SOURCE,
    source_offer: wl_data_source::OFFER,
    source_destroy: wl_data_source::DESTROY,
    ev_send: wl_data_source::EV_SEND,
    ev_cancelled: wl_data_source::EV_CANCELLED,
    set_selection: wl_data_device::SET_SELECTION,
    ev_data_offer: wl_data_device::EV_DATA_OFFER,
    ev_selection: wl_data_device::EV_SELECTION,
    offer_receive: wl_data_offer::RECEIVE,
    offer_destroy: wl_data_offer::DESTROY,
};

const PRIMARY_OPS: SelectionOps = SelectionOps {
    create_source: zwp_primary_selection_device_manager_v1::CREATE_SOURCE,
    source_offer: zwp_primary_selection_source_v1::OFFER,
    source_destroy: zwp_primary_selection_source_v1::DESTROY,
    ev_send: zwp_primary_selection_source_v1::EV_SEND,
    ev_cancelled: zwp_primary_selection_source_v1::EV_CANCELLED,
    set_selection: zwp_primary_selection_device_v1::SET_SELECTION,
    ev_data_offer: zwp_primary_selection_device_v1::EV_DATA_OFFER,
    ev_selection: zwp_primary_selection_device_v1::EV_SELECTION,
    offer_receive: zwp_primary_selection_offer_v1::RECEIVE,
    offer_destroy: zwp_primary_selection_offer_v1::DESTROY,
};

/// One selection transport's bookkeeping. When we own the selection, `source` is
/// our live source object and `data` the bytes it serves. Incoming offers are
/// tracked so a paste can ask for a text MIME.
pub(super) struct SelectionState {
    pub(super) source: u32,
    pub(super) data: Vec<u8>,
    pub(super) incoming_offer: u32,
    pub(super) incoming_text_mime: Option<String>,
    pub(super) selection_offer: u32,
    pub(super) selection_text_mime: Option<String>,
}

impl SelectionState {
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

    /// Record an offer the compositor has just introduced. Returns the offer this one
    /// supersedes, which the caller must destroy.
    ///
    /// Something has to be returned, because an offer nobody destroys is a leak — a
    /// client id (`alloc_id` never reuses them) plus a compositor resource. And offers
    /// that are introduced and never named are not an edge case: `wl_data_device` carries
    /// drag-and-drop offers on the same object, bnkterm handles no DnD events at all, so
    /// every drag over the surface introduces one that no `selection` will ever claim.
    ///
    /// Never the selection's own offer. The compositor introduces an offer and then names
    /// it, so `incoming_offer` and `selection_offer` are routinely the same id, and that
    /// one is still ours to paste from.
    fn introduce(&mut self, offer: u32) -> Option<u32> {
        let stale = self.incoming_offer;
        self.incoming_offer = offer;
        self.incoming_text_mime = None;
        (stale != 0 && stale != offer && stale != self.selection_offer).then_some(stale)
    }

    /// Offer `mime` as a candidate paste type, keeping the best one seen.
    ///
    /// **Best, not first.** The old rule took the first `text/*` and could only be
    /// displaced by an exact `text/plain;charset=utf-8` — but `text/html`,
    /// `text/uri-list` and `text/plain;charset=utf-16` are all `text/*` too, and sources
    /// advertise the richest type first (LibreOffice and several GTK apps lead with
    /// `text/html`). So copying a word from a document and pasting it at a shell prompt
    /// put `<meta http-equiv=...><p>ls -la</p>` on the command line. A `utf-16` win is
    /// worse still: the bytes are not UTF-8, so the paste arrives as a row of replacement
    /// characters — visible garbage rather than an invisible no-op, but still wrong.
    ///
    /// The rank is recomputed from whatever is stored rather than kept beside it, so the
    /// two cannot drift.
    fn offer_mime(&mut self, mime: &str) {
        let Some(rank) = mime_rank(mime) else { return };
        let best = self.incoming_text_mime.as_deref().and_then(mime_rank);
        if best.is_none_or(|best| rank < best) {
            self.incoming_text_mime = Some(mime.to_string());
        }
    }

    /// Record the selection becoming `offer` (`0` clears it). Returns the offer the
    /// previous selection held, for the caller to destroy.
    ///
    /// Clearing `incoming_offer` here is what keeps its invariant — zero, or an offer
    /// that is alive. Without it a `selection(0)` that destroys the old offer would leave
    /// `incoming_offer` naming a dead object, and the next introduction would destroy it
    /// a second time.
    fn take_selection(&mut self, offer: u32) -> Option<u32> {
        let old = self.selection_offer;
        // Only an offer we were introduced to carries a MIME we learned; anything else
        // (including a clear) leaves the selection with no text type.
        let becomes_ours = offer != 0 && offer == self.incoming_offer;
        let mime = self.incoming_text_mime.take();
        self.selection_text_mime = if becomes_ours { mime } else { None };
        self.selection_offer = offer;
        self.incoming_offer = 0;
        (old != 0 && old != offer).then_some(old)
    }
}

impl State {
    // Clipboard entry points (Ctrl+Shift+C/V), thin wrappers over the shared path.

    /// A `wl_data_device` event on the clipboard device.
    pub(super) fn on_data_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        self.on_selection_device(Transport::Clipboard, opcode, r)
    }

    /// A `wl_data_source` event on the clipboard source we own.
    pub(super) fn on_data_source(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        self.on_selection_source(Transport::Clipboard, opcode, r)
    }

    /// One MIME type of the incoming clipboard offer.
    pub(super) fn on_offer_mime(&mut self, r: &mut Reader) -> Result<()> {
        self.on_selection_offer_mime(Transport::Clipboard, r)
    }

    /// Become the clipboard owner serving `data` as text.
    pub(super) fn set_clipboard(&mut self, data: Vec<u8>) {
        self.set_selection(Transport::Clipboard, data);
    }

    /// Paste the clipboard's text to the child (Ctrl+Shift+V).
    pub(super) fn paste(&mut self) -> Result<()> {
        self.paste_from(Transport::Clipboard)
    }

    // Primary selection entry points (select-to-copy, middle-click paste).

    /// A `zwp_primary_selection_device_v1` event on the primary device.
    pub(super) fn on_primary_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        self.on_selection_device(Transport::Primary, opcode, r)
    }

    /// A `zwp_primary_selection_source_v1` event on the primary source we own.
    pub(super) fn on_primary_source(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        self.on_selection_source(Transport::Primary, opcode, r)
    }

    /// One MIME type of the incoming primary offer.
    pub(super) fn on_primary_offer_mime(&mut self, r: &mut Reader) -> Result<()> {
        self.on_selection_offer_mime(Transport::Primary, r)
    }

    /// Become the primary-selection owner serving `data` as text (copy-on-select).
    pub(super) fn set_primary(&mut self, data: Vec<u8>) {
        self.set_selection(Transport::Primary, data);
    }

    /// Paste the primary selection's text to the child (middle-click).
    pub(super) fn paste_primary(&mut self) -> Result<()> {
        self.paste_from(Transport::Primary)
    }

    // The shared implementation, keyed by transport.

    fn ops(t: Transport) -> SelectionOps {
        match t {
            Transport::Clipboard => CLIPBOARD_OPS,
            Transport::Primary => PRIMARY_OPS,
        }
    }

    fn selection(&self, t: Transport) -> &SelectionState {
        match t {
            Transport::Clipboard => &self.clipboard,
            Transport::Primary => &self.primary,
        }
    }

    fn selection_mut(&mut self, t: Transport) -> &mut SelectionState {
        match t {
            Transport::Clipboard => &mut self.clipboard,
            Transport::Primary => &mut self.primary,
        }
    }

    /// The device object (`0` when the compositor lacks the manager).
    fn selection_device(&self, t: Transport) -> u32 {
        match t {
            Transport::Clipboard => self.data_device,
            Transport::Primary => self.primary_device,
        }
    }

    /// The manager global, or `None` when the compositor did not advertise it.
    fn selection_manager(&self, t: Transport) -> Option<u32> {
        match t {
            Transport::Clipboard => self.data_device_manager,
            Transport::Primary => self.primary_manager,
        }
    }

    /// A device event: a new offer being introduced, or the selection changing to
    /// one (or to null).
    fn on_selection_device(&mut self, t: Transport, opcode: u16, r: &mut Reader) -> Result<()> {
        let ops = Self::ops(t);
        // Wire decoding here, the offer lifecycle in [`SelectionState`], which is what
        // makes "who owns this offer now, and who destroys it" testable without a
        // compositor.
        let doomed = if opcode == ops.ev_data_offer {
            self.selection_mut(t).introduce(r.u32()?)
        } else if opcode == ops.ev_selection {
            self.selection_mut(t).take_selection(r.u32()?)
        } else {
            None
        };
        if let Some(offer) = doomed {
            self.conn.request(offer, ops.offer_destroy, &[]);
        }
        Ok(())
    }

    /// A source event on the source we own: serve our bytes to a paster, or tear the
    /// source down when another selection replaces ours.
    fn on_selection_source(&mut self, t: Transport, opcode: u16, r: &mut Reader) -> Result<()> {
        let ops = Self::ops(t);
        if opcode == ops.ev_send {
            // Every MIME we advertise serves the same bytes.
            let _mime = r.string()?;
            let fd = self
                .conn
                .take_fd()
                .ok_or_else(|| Error::msg("selection source.send arrived without its fd"))?;
            self.begin_selection_send(t, fd);
        } else if opcode == ops.ev_cancelled {
            let source = self.selection(t).source;
            if source != 0 {
                self.conn.request(source, ops.source_destroy, &[]);
            }
            let st = self.selection_mut(t);
            st.source = 0;
            st.data.clear();
        }
        Ok(())
    }

    /// One MIME type of an incoming offer; keep the best text one seen so a paste knows
    /// what to request. See [`SelectionState::offer_mime`].
    fn on_selection_offer_mime(&mut self, t: Transport, r: &mut Reader) -> Result<()> {
        let mime = r.string()?;
        self.selection_mut(t).offer_mime(mime);
        Ok(())
    }

    /// Become the transport's owner serving `data` as text, replacing any source we
    /// held. Needs the manager and device (skips silently without them). Called from
    /// the outbox drain when the core reports a fresh selection to own.
    fn set_selection(&mut self, t: Transport, data: Vec<u8>) {
        let Some(manager) = self.selection_manager(t) else {
            return;
        };
        let device = self.selection_device(t);
        if device == 0 {
            return;
        }
        let ops = Self::ops(t);
        let old_source = self.selection(t).source;
        if old_source != 0 {
            self.conn.request(old_source, ops.source_destroy, &[]);
        }
        let source = self.alloc_id();
        self.conn
            .request(manager, ops.create_source, &[Arg::NewId(source)]);
        for mime in CLIPBOARD_MIMES {
            self.conn
                .request(source, ops.source_offer, &[Arg::Str(mime)]);
        }
        self.conn.request(
            device,
            ops.set_selection,
            &[Arg::Object(source), Arg::Uint(self.last_serial)],
        );
        let st = self.selection_mut(t);
        st.source = source;
        st.data = data;
    }

    /// Start serving our selection down `fd`, writing what fits now and queueing the
    /// rest for [`Self::pump_selection_sends`].
    ///
    /// The write cannot be finished here, for the same reason the PTY's cannot: a pipe
    /// holds 64 KiB by default and the receiver decides when to read. Select a megabyte
    /// of scrollback and paste it into an application that reads lazily (or a clipboard
    /// manager that defers reading entirely) and a blocking write stops the event loop
    /// mid-transfer — the window stops repainting while another program's scheduling
    /// decides when it resumes.
    ///
    /// A deadline is not an option here the way it is for a paste. Giving up halfway
    /// through delivers a *truncated selection* to whatever asked for it, silently and
    /// plausibly, so the only correct outcomes are all of it or a closed pipe.
    fn begin_selection_send(&mut self, t: Transport, fd: OwnedFd) {
        if let Err(e) = ffi::set_nonblocking(fd.as_raw_fd()) {
            eprintln!("bnkterm: selection serve failed: {e}");
            return;
        }
        // Oldest first: a receiver that never reads is the one that piles up, and a fresh
        // request is likelier to be a live paster than a stalled one.
        if self.pending_sends.len() >= MAX_PENDING_SENDS {
            self.pending_sends.remove(0);
        }
        self.pending_sends.push(PendingSend {
            fd,
            data: self.selection(t).data.clone(),
            head: 0,
        });
        self.pump_selection_sends();
    }

    /// Push every in-flight selection transfer as far as its receiver will take it,
    /// dropping the ones that finish or fail. A closed read end reports `EPIPE` rather
    /// than raising `SIGPIPE`, because the Rust runtime ignores it process-wide (only the
    /// PTY child restores the default), so a paster that goes away is an ordinary error.
    pub(super) fn pump_selection_sends(&mut self) {
        self.pending_sends.retain_mut(|send| {
            while send.head < send.data.len() {
                match ffi::write_some(send.fd.as_raw_fd(), &send.data[send.head..]) {
                    Ok(0) => return true, // the pipe is full; come back on POLLOUT
                    Ok(n) => send.head += n,
                    Err(e) => {
                        // Almost always EPIPE: the paster took what it wanted and closed.
                        // Serving a selection must never take the terminal down.
                        eprintln!("bnkterm: selection serve failed: {e}");
                        return false;
                    }
                }
            }
            false // delivered in full; closing the fd is what signals EOF
        });
    }

    /// Paste a transport's text to the child: the core normalizes, brackets, and
    /// writes it (see `TerminalCore::apply`). A no-op when the transport is empty.
    fn paste_from(&mut self, t: Transport) -> Result<()> {
        if let Some(text) = self.selection_text(t)? {
            self.tabs
                .active_mut()
                .apply(ToTerminal::Paste(text.into_bytes()))?;
        }
        Ok(())
    }

    /// A transport's text, or `None`. When we own it our own bytes are returned
    /// directly: routing through the compositor would deadlock (we would block
    /// reading the pipe while also owing our source a `send`). Otherwise the bytes
    /// come over a pipe (give the compositor the write end, read to EOF).
    fn selection_text(&mut self, t: Transport) -> Result<Option<String>> {
        if self.selection(t).source != 0 {
            return Ok(Some(
                String::from_utf8_lossy(&self.selection(t).data).into_owned(),
            ));
        }
        let st = self.selection(t);
        let (Some(mime), offer) = (st.selection_text_mime.clone(), st.selection_offer) else {
            return Ok(None);
        };
        if offer == 0 {
            return Ok(None);
        }
        let ops = Self::ops(t);
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            offer,
            ops.offer_receive,
            &[Arg::Str(&mime)],
            write_end.as_raw_fd(),
        )?;
        // Our copy goes now, so the only writer left is the owner: without this the read
        // below never sees EOF, because the pipe still has a write end open here.
        drop(write_end);
        // Bounded, because the sender is another application and this is the thread that
        // also renders, dispatches Wayland and handles keys. A clipboard owner that
        // accepts the transfer and then never writes and never closes would otherwise
        // hang every tab until SIGKILL, on an ordinary Ctrl+Shift+V. A real transfer
        // completes in microseconds, so the budget is only ever spent on a broken peer.
        let bytes = ffi::read_to_end_bounded(read_end.as_raw_fd(), PASTE_BUDGET, PASTE_MAX)?;
        Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
    }
}

/// Whether a clipboard MIME type carries text we can paste.
pub(super) fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/") || mime == "UTF8_STRING"
}

/// How good a text MIME is for pasting into a terminal, **lower is better**; `None` for
/// one that carries no text at all.
///
/// A total order rather than a "first `text/*` wins" rule, because a source advertises
/// its *richest* type first and the richest type is the one a terminal wants least. The
/// tiers:
///
/// 0. `text/plain;charset=utf-8` — exactly what was asked for, and what we advertise
///    when copying ([`CLIPBOARD_MIMES`]).
/// 1. `UTF8_STRING` — the X11 atom naming the same bytes, still offered by Xwayland
///    clients.
/// 2. `text/plain` — text with no charset stated, which is UTF-8 in every locale a
///    terminal runs in.
/// 3. Everything else under `text/`: `text/html`, `text/uri-list`,
///    `text/plain;charset=utf-16`. Taken only when a source offers nothing better, on
///    the grounds that some text beats no paste — a source offering *only* markup is
///    rare, where one offering markup *first* is the common case.
fn mime_rank(mime: &str) -> Option<u8> {
    match mime {
        "text/plain;charset=utf-8" => Some(0),
        "UTF8_STRING" => Some(1),
        "text/plain" => Some(2),
        _ if is_text_mime(mime) => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::app::clipboard::SelectionState;

    #[test]
    fn an_offer_that_never_becomes_the_selection_is_still_destroyed() {
        // `wl_data_device` carries drag-and-drop offers on the same object as clipboard
        // ones, and bnkterm handles no DnD events at all — so every drag over the surface
        // introduces an offer that no `selection` will ever claim. Only offers that
        // *became* the selection were destroyed, so each of those drags leaked a client
        // id (never reused) and a compositor resource.
        let mut st = SelectionState::new();

        assert_eq!(st.introduce(10), None, "nothing to supersede yet");
        assert_eq!(
            st.introduce(11),
            Some(10),
            "the drag's offer is superseded and must be destroyed"
        );
        assert_eq!(st.introduce(12), Some(11));
        assert_eq!(st.introduce(12), None, "the same offer twice is not a leak");
    }

    #[test]
    fn the_selections_own_offer_survives_the_next_introduction() {
        // The compositor introduces an offer and then names it, so `incoming_offer` and
        // `selection_offer` are routinely the same id. Destroying it on the next
        // introduction would take down the offer a paste still has to read from.
        let mut st = SelectionState::new();
        st.introduce(10);
        st.incoming_text_mime = Some("text/plain;charset=utf-8".to_string());
        assert_eq!(st.take_selection(10), None, "no previous selection");
        assert_eq!(st.selection_offer, 10);
        assert_eq!(
            st.selection_text_mime.as_deref(),
            Some("text/plain;charset=utf-8"),
            "an offer that became the selection carries its MIME across"
        );

        assert_eq!(st.introduce(11), None, "the selection's offer is not stale");
        assert_eq!(st.selection_offer, 10, "and it is still the selection");
        assert_eq!(st.introduce(12), Some(11), "but the drag's offer goes");
        assert_eq!(st.selection_offer, 10);
    }

    #[test]
    fn a_cleared_selection_does_not_leave_a_destroyed_offer_named() {
        // The double-free trap. `selection(0)` destroys the old offer, and
        // `incoming_offer` was pointing at that very id — so without clearing it, the
        // next introduction would ask the compositor to destroy a dead object.
        let mut st = SelectionState::new();
        st.introduce(10);
        st.take_selection(10);

        assert_eq!(st.take_selection(0), Some(10), "the selection was cleared");
        assert_eq!(st.selection_offer, 0);
        assert_eq!(st.incoming_offer, 0, "and nothing still names the dead id");
        assert_eq!(
            st.introduce(11),
            None,
            "so the next introduction destroys nothing"
        );

        // A selection replaced by another's is the ordinary case, and the old one goes.
        let mut st = SelectionState::new();
        st.introduce(20);
        st.take_selection(20);
        st.introduce(21);
        assert_eq!(st.take_selection(21), Some(20));
    }

    #[test]
    fn the_best_offered_text_type_wins_whatever_order_it_arrives_in() {
        // A source advertises its richest type first, and the richest type is the one a
        // terminal wants least. Taking the first `text/*` therefore lost to whatever the
        // source led with: LibreOffice and several GTK apps lead with `text/html`, so
        // copying a word from a document and pasting it at a shell prompt put
        // `<meta http-equiv=...><p>ls -la</p>` on the command line.
        let best = |offers: &[&str]| {
            let mut st = SelectionState::new();
            for m in offers {
                st.offer_mime(m);
            }
            st.incoming_text_mime
        };

        assert_eq!(
            best(&["text/html", "text/plain", "text/plain;charset=utf-8"]).as_deref(),
            Some("text/plain;charset=utf-8"),
            "the richest type arriving first does not win"
        );
        assert_eq!(
            best(&["text/plain;charset=utf-8", "text/plain", "text/html"]).as_deref(),
            Some("text/plain;charset=utf-8"),
            "and the best arriving first is not displaced by the rest"
        );

        // utf-16 is `text/*` and is not UTF-8, so winning meant a paste of replacement
        // characters — visible garbage where it used to be an invisible no-op.
        assert_eq!(
            best(&["text/plain;charset=utf-16", "text/plain"]).as_deref(),
            Some("text/plain")
        );
        // The X11 atom names the same bytes as our first choice, so it outranks bare
        // `text/plain` but not an explicit utf-8.
        assert_eq!(
            best(&["text/plain", "UTF8_STRING"]).as_deref(),
            Some("UTF8_STRING")
        );
        assert_eq!(
            best(&["UTF8_STRING", "text/plain;charset=utf-8"]).as_deref(),
            Some("text/plain;charset=utf-8")
        );

        // Some text beats no paste: a source offering nothing better still gets used.
        assert_eq!(best(&["text/html"]).as_deref(), Some("text/html"));
        // And a non-text offer is not a candidate at all.
        assert_eq!(best(&["image/png", "application/pdf"]), None);
        assert_eq!(
            best(&["image/png", "text/plain"]).as_deref(),
            Some("text/plain")
        );
    }

    #[test]
    fn a_selection_we_were_not_introduced_to_carries_no_mime() {
        // A MIME learned from one offer must not be attributed to a different one: the
        // paste would then ask for a type the new source never advertised.
        let mut st = SelectionState::new();
        st.introduce(10);
        st.incoming_text_mime = Some("text/plain".to_string());

        st.take_selection(99);
        assert_eq!(st.selection_offer, 99);
        assert_eq!(st.selection_text_mime, None);
    }
}
