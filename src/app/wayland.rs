//! The window's side of the Wayland protocol that the clipboard and presentation modules do
//! not own: object ids, binding the registry's globals, and routing each incoming event to
//! the code that handles it.

use crate::app::clipboard::Transport;
use crate::app::{State, MAX_DIMENSION};
use crate::error::{Error, Result};
use crate::platform::protocol::{
    self, wl_buffer, wl_callback, wl_display, wl_keyboard, wl_output, wl_registry, wl_seat,
    wl_surface, wp_cursor_shape_manager_v1, wp_fractional_scale_v1, xdg_surface, xdg_toplevel,
    xdg_wm_base,
};
use crate::platform::wire::{self, Arg, Message, Reader};

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

impl State {
    /// Allocate an object id and ask `parent` to create it: the "`NewId` alone" shape
    /// every plain object-creating request takes. The caller keeps the id.
    pub(super) fn create(&mut self, parent: u32, opcode: u16) -> u32 {
        let id = self.alloc_id();
        self.conn.request(parent, opcode, &[Arg::NewId(id)]);
        id
    }

    /// The same, for the requests that also name the object the new one attaches to
    /// (a surface, a seat, a pointer).
    pub(super) fn create_for(&mut self, parent: u32, opcode: u16, object: u32) -> u32 {
        let id = self.alloc_id();
        self.conn
            .request(parent, opcode, &[Arg::NewId(id), Arg::Object(object)]);
        id
    }

    pub(super) fn alloc_id(&mut self) -> u32 {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        })
    }

    pub(super) fn roundtrip(&mut self) -> Result<()> {
        let callback = self.create(protocol::WL_DISPLAY, wl_display::SYNC);
        self.pending_sync = Some(callback);
        self.run_until(|s| s.pending_sync.is_none())
    }

    pub(super) fn handle(&mut self, msg: &Message) -> Result<()> {
        // Closing the last tab can happen while more Wayland messages are already
        // queued. The process is leaving; ignore those messages so none can route
        // input through an intentionally empty tab list.
        if self.closed {
            return Ok(());
        }
        let mut r = Reader::new(&msg.body);

        if msg.object == protocol::WL_DISPLAY {
            match msg.opcode {
                wl_display::EV_ERROR => {
                    let object = r.u32()?;
                    let code = r.u32()?;
                    let message = r.string()?;
                    return Err(Error::msg(format!(
                        "wayland protocol error: object {object} code {code}: {message}"
                    )));
                }
                wl_display::EV_DELETE_ID => {
                    let id = r.u32()?;
                    if id < SERVER_ID_BASE {
                        self.forget_destroyed_source(id);
                        self.released_keyboards.retain(|&keyboard| keyboard != id);
                        self.free_ids.push(id);
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        if msg.object == self.objects.registry && msg.opcode == wl_registry::EV_GLOBAL {
            return self.on_global(&mut r);
        }

        if Some(msg.object) == self.pending_sync && msg.opcode == wl_callback::EV_DONE {
            self.pending_sync = None;
            return Ok(());
        }

        // The frame callback fired: the compositor is ready for the next frame.
        if self
            .frame_callback
            .is_some_and(|callback| callback.id == msg.object)
            && msg.opcode == wl_callback::EV_DONE
        {
            self.frame_callback = None;
            // Keep a fade going: a scrollbar mid-transition wants the frame the
            // compositor has just offered. A step whose quantised colours did not move
            // presents nothing and arms no new callback, which is what `retry_at` and the
            // wake deadline are for; this is the fast path while it is visibly changing.
            if self.tabs.active().scrollbar_animating() {
                self.tabs.mark_dirty();
            }
            return Ok(());
        }

        if msg.object == self.objects.xdg_surface && msg.opcode == xdg_surface::EV_CONFIGURE {
            self.pending_configure = Some(r.u32()?);
            if let Some((lw, lh)) = self.pending_size.take() {
                // The configure size is logical (surface-local); the buffer and
                // grid are device pixels. Record the logical size (for the viewport
                // destination) and resize the grid to the scaled device size.
                self.scale.logical = (lw, lh);
                self.scale.synced = false;
                let (dw, dh) = self.device_size(lw, lh);
                self.resize_to(dw, dh);
            }
            self.configured = true;
            self.tabs.mark_dirty();
            return Ok(());
        }

        // wp_fractional_scale_v1.preferred_scale: the compositor's scale as 120ths.
        // This is the primary scale source; adopting it reopens the fonts and
        // resizes the buffer to exact device resolution.
        if self.scale.fractional_scale != 0
            && msg.object == self.scale.fractional_scale
            && msg.opcode == wp_fractional_scale_v1::EV_PREFERRED_SCALE
        {
            let scale_120 = r.u32()?;
            self.set_scale_120(scale_120);
            return Ok(());
        }

        // wl_surface.enter/leave and wl_output.scale drive the integer fallback
        // (used only when the fractional path is absent).
        if msg.object == self.objects.surface
            && (msg.opcode == wl_surface::EV_ENTER || msg.opcode == wl_surface::EV_LEAVE)
        {
            let output = r.u32()?;
            if msg.opcode == wl_surface::EV_ENTER {
                if !self.scale.entered.contains(&output) {
                    self.scale.entered.push(output);
                }
            } else {
                self.scale.entered.retain(|&o| o != output);
            }
            self.refresh_integer_scale();
            return Ok(());
        }
        if msg.opcode == wl_output::EV_SCALE
            && self.scale.outputs.iter().any(|(id, _)| *id == msg.object)
        {
            let factor = r.u32()? as i32;
            for out in &mut self.scale.outputs {
                if out.0 == msg.object {
                    out.1 = factor.max(1);
                }
            }
            self.refresh_integer_scale();
            return Ok(());
        }

        if msg.object == self.objects.toplevel {
            match msg.opcode {
                xdg_toplevel::EV_CONFIGURE => {
                    let w = r.u32()?;
                    let h = r.u32()?;
                    let _states = r.array()?;
                    if w > 0 && h > 0 {
                        self.pending_size = Some((w.min(MAX_DIMENSION), h.min(MAX_DIMENSION)));
                    }
                }
                xdg_toplevel::EV_CLOSE => self.closed = true,
                _ => {}
            }
            return Ok(());
        }

        if Some(msg.object) == self.objects.wm_base && msg.opcode == xdg_wm_base::EV_PING {
            let serial = r.u32()?;
            self.conn
                .request(msg.object, xdg_wm_base::PONG, &[Arg::Uint(serial)]);
            return Ok(());
        }

        if Some(msg.object) == self.objects.seat && msg.opcode == wl_seat::EV_CAPABILITIES {
            // A capability can go and come back: a keyboard unplugged, a KVM switch, a
            // Bluetooth device asleep. The compositor makes the old object inert when it
            // goes, so it is released then, and a fresh one is created when it returns.
            let caps = r.u32()?;
            let has_keyboard = caps & wl_seat::CAP_KEYBOARD != 0;
            let has_pointer = caps & wl_seat::CAP_POINTER != 0;
            if !has_keyboard && self.objects.keyboard != 0 {
                self.release_keyboard()?;
            }
            if !has_pointer && self.objects.pointer != 0 {
                self.release_pointer()?;
            }
            if has_keyboard && self.objects.keyboard == 0 {
                self.objects.keyboard = self.create(msg.object, wl_seat::GET_KEYBOARD);
            }
            if has_pointer && self.objects.pointer == 0 {
                let pointer = self.create(msg.object, wl_seat::GET_POINTER);
                self.objects.pointer = pointer;
                // Pair the pointer with a cursor-shape device so we can ask for the
                // I-beam over the grid. The manager is bound in the same global burst
                // as the seat, so it is known by the time capabilities arrive; without
                // it the pointer keeps the compositor default shape.
                if let Some(manager) = self.objects.cursor_shape_manager {
                    self.objects.cursor_shape_device =
                        self.create_for(manager, wp_cursor_shape_manager_v1::GET_POINTER, pointer);
                }
            }
            return Ok(());
        }

        if self.released_keyboards.contains(&msg.object) {
            if msg.opcode == wl_keyboard::EV_KEYMAP {
                drop(self.conn.take_fd());
            }
            return Ok(());
        }

        if self.objects.keyboard != 0 && msg.object == self.objects.keyboard {
            return self.on_keyboard(msg.opcode, &mut r);
        }

        if self.objects.pointer != 0 && msg.object == self.objects.pointer {
            return self.on_pointer(msg.opcode, &mut r);
        }

        if let Some(result) = self.on_selection_message(msg, &mut r) {
            return result;
        }

        if let Some(result) = self.on_activation_message(msg, &mut r) {
            return result;
        }

        if self.presentation.feedback_id != 0 && msg.object == self.presentation.feedback_id {
            return self.on_dmabuf_feedback(msg.opcode, &mut r);
        }

        if msg.opcode == wl_buffer::EV_RELEASE {
            if msg.object == self.presentation.buffers[0] {
                self.presentation.busy[0] = false;
            } else if msg.object == self.presentation.buffers[1] {
                self.presentation.busy[1] = false;
            } else {
                // A release for a buffer a resize retired: the compositor is done reading it,
                // so its image can finally be freed under it safely.
                self.release_retired(msg.object);
            }
        }
        Ok(())
    }

    fn on_global(&mut self, r: &mut Reader) -> Result<()> {
        let name = r.u32()?;
        let interface = r.string()?;
        let version = r.u32()?;
        match interface {
            protocol::IFACE_COMPOSITOR => {
                // Version 4 is a hard floor, not a preference: the frame is presented with
                // `wl_surface.damage_buffer`, which does not exist before it, and there is
                // no correct fallback worth carrying (`damage` would need every rect
                // divided by the scale, throwing away precision the buffer has). Leaving
                // the global unbound turns this into the clear startup error in
                // `bring_up` rather than a protocol error on the first frame.
                if version < protocol::VERSION_COMPOSITOR {
                    return Ok(());
                }
                let id = self.bind_capped(name, interface, version, protocol::VERSION_COMPOSITOR);
                self.objects.compositor = Some(id);
            }
            protocol::IFACE_WM_BASE => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_WM_BASE);
                self.objects.wm_base = Some(id);
            }
            // The first seat only. The keyboard and pointer belong to the seat that created
            // them, so a second seat whose capabilities lack a keyboard must not be able to
            // release the first seat's.
            protocol::IFACE_SEAT if self.objects.seat.is_none() => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_SEAT);
                self.objects.seat = Some(id);
                self.seat_version = version.min(protocol::VERSION_SEAT);
            }
            protocol::IFACE_CURSOR_SHAPE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_CURSOR_SHAPE_MANAGER,
                );
                self.objects.cursor_shape_manager = Some(id);
            }
            protocol::IFACE_DATA_DEVICE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_DATA_DEVICE_MANAGER,
                );
                self.selections[Transport::Clipboard as usize].manager = Some(id);
            }
            protocol::IFACE_PRIMARY_SELECTION => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_PRIMARY_SELECTION,
                );
                self.selections[Transport::Primary as usize].manager = Some(id);
            }
            protocol::IFACE_DMABUF if version >= protocol::VERSION_DMABUF => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DMABUF);
                self.presentation.dmabuf = Some(id);
                self.presentation.dmabuf_version = version.min(protocol::VERSION_DMABUF);
            }
            protocol::IFACE_DRM_SYNCOBJ => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DRM_SYNCOBJ);
                self.presentation.syncobj_manager = Some(id);
            }
            protocol::IFACE_OUTPUT => {
                // One global per monitor; bind each and default its scale to 1
                // until a `scale` event refines it (integer fallback only).
                let id = self.bind_capped(name, interface, version, protocol::VERSION_OUTPUT);
                self.scale.outputs.push((id, 1));
            }
            protocol::IFACE_FRACTIONAL_SCALE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_FRACTIONAL_SCALE_MANAGER,
                );
                self.scale.fractional_manager = Some(id);
            }
            protocol::IFACE_VIEWPORTER => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_VIEWPORTER);
                self.scale.viewporter = Some(id);
            }
            protocol::IFACE_ACTIVATION => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_ACTIVATION);
                self.activation.global = Some(id);
            }
            _ => {}
        }
        Ok(())
    }

    /// Bind the just-advertised global at the version we negotiate: the
    /// compositor's advertised `version` capped at `max`.
    fn bind_capped(&mut self, name: u32, interface: &str, version: u32, max: u32) -> u32 {
        let id = self.alloc_id();
        self.conn.request(
            self.objects.registry,
            wl_registry::BIND,
            &[
                Arg::Uint(name),
                Arg::Bind {
                    interface,
                    version: version.min(max),
                    new_id: id,
                },
            ],
        );
        id
    }

    /// Set the toplevel title, fitted to what one request can carry: the child's OSC
    /// title can run to ~12 KiB (4 KiB of bytes, each invalid one widened to U+FFFD). The
    /// core deduplicates on the sending side (it only emits a `Title` when the child's
    /// title actually changes), so this just makes the request; bring-up calls it once
    /// for the initial app name.
    pub(super) fn set_toplevel_title(&mut self, title: &str) {
        self.conn.request(
            self.objects.toplevel,
            xdg_toplevel::SET_TITLE,
            &[Arg::Str(wire::fit_sole_string(title))],
        );
    }
}
