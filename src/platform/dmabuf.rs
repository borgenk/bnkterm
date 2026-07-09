//! `zwp_linux_dmabuf_v1` feedback: the compositor's advertisement of which DRM
//! formats and modifiers it accepts in a dmabuf-backed `wl_buffer`, and which
//! GPU it wants those buffers allocated on. This is the negotiation half of the
//! GPU presentation path; the Vulkan side consumes the
//! committed [`Feedback`] to pick a device and an image modifier.
//!
//! The protocol delivers feedback as a burst of events on a
//! `zwp_linux_dmabuf_feedback_v1` object:
//!
//! ```text
//!   format_table (an fd the client mmaps: N 16-byte format+modifier entries)
//!   main_device  (a dev_t)
//!   tranche_target_device / tranche_formats / tranche_flags / tranche_done  (per tranche)
//!   done         (commit everything since the last done)
//! ```
//!
//! [`FeedbackState`] accumulates that burst and exposes the committed snapshot;
//! the compositor may resend the whole set at any time (a GPU hotplug, a
//! scanout change), so `done` replaces the snapshot rather than extending it.
//! This module is pure parsing and state: the fd mmap and the wire routing live
//! with the caller (`app.rs`), so everything here is testable on plain bytes.

use crate::platform::error::{Error, Result};

/// DRM fourcc for 32-bit xRGB, little-endian (`XR24`), the dmabuf twin of the
/// `wl_shm` XRGB8888 the software path renders today.
pub const DRM_FORMAT_XRGB8888: u32 = fourcc(b"XR24");
/// The linear (no tiling) layout modifier, `DRM_FORMAT_MOD_LINEAR`.
pub const MOD_LINEAR: u64 = 0;

/// A DRM fourcc: four ASCII bytes packed little-endian.
const fn fourcc(code: &[u8; 4]) -> u32 {
    (code[0] as u32) | (code[1] as u32) << 8 | (code[2] as u32) << 16 | (code[3] as u32) << 24
}

/// One (format, modifier) pair from the compositor's format table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FormatModifier {
    /// DRM fourcc (not a `wl_shm` enum value).
    pub format: u32,
    pub modifier: u64,
}

/// One preference tranche: a target device and the format+modifier pairs usable
/// on it. Tranches arrive in decreasing order of compositor preference (an
/// earlier tranche may be scanout-capable, a later one composited).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tranche {
    /// The dev_t of the DRM device this tranche's buffers should be allocated on.
    pub device: u64,
    /// Bit 0x1 means the tranche is for direct scanout.
    pub flags: u32,
    pub formats: Vec<FormatModifier>,
}

/// A committed feedback snapshot: everything the compositor announced between
/// binding (or the previous `done`) and the last `done`.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Feedback {
    /// The dev_t of the device the compositor composites with; the GPU backend
    /// must allocate from a matching device or the import will fail or copy.
    pub main_device: u64,
    pub tranches: Vec<Tranche>,
}

impl Feedback {
    /// Deduplicated modifiers for `format` across every tranche, in tranche
    /// (preference) order. This is the list to intersect with what the render
    /// device supports when choosing an image layout.
    pub fn modifiers_for(&self, format: u32) -> Vec<u64> {
        let mut out = Vec::new();
        for t in &self.tranches {
            for f in &t.formats {
                if f.format == format && !out.contains(&f.modifier) {
                    out.push(f.modifier);
                }
            }
        }
        out
    }
}

/// Accumulates one feedback burst and holds the last committed snapshot.
#[derive(Default)]
pub struct FeedbackState {
    /// The parsed format table. Kept across `done`: a re-announcement reuses the
    /// previous table unless the compositor sends a new one.
    table: Vec<FormatModifier>,
    pending: Feedback,
    /// The tranche being accumulated. Its device defaults to the pending main
    /// device when the compositor omits `tranche_target_device`.
    tranche_device: Option<u64>,
    tranche_flags: u32,
    tranche_formats: Vec<FormatModifier>,
    ready: Option<Feedback>,
}

impl FeedbackState {
    /// The last committed snapshot, once a `done` has arrived.
    pub fn ready(&self) -> Option<&Feedback> {
        self.ready.as_ref()
    }

    /// Install the format table from the mmapped `format_table` fd's bytes.
    pub fn set_table(&mut self, bytes: &[u8]) {
        self.table = parse_format_table(bytes);
    }

    /// How many entries the current format table holds (for diagnostics).
    pub fn table_len(&self) -> usize {
        self.table.len()
    }

    /// The `main_device` event: a dev_t as a wire array.
    pub fn main_device(&mut self, bytes: &[u8]) -> Result<()> {
        self.pending.main_device = parse_dev(bytes)?;
        Ok(())
    }

    /// The `tranche_target_device` event: a dev_t as a wire array.
    pub fn tranche_device(&mut self, bytes: &[u8]) -> Result<()> {
        self.tranche_device = Some(parse_dev(bytes)?);
        Ok(())
    }

    /// The `tranche_formats` event: native-endian u16 indices into the format
    /// table. An index past the table (a compositor bug) is skipped rather than
    /// poisoning the whole tranche.
    pub fn tranche_formats(&mut self, bytes: &[u8]) {
        for pair in bytes.chunks_exact(2) {
            let idx = u16::from_ne_bytes([pair[0], pair[1]]) as usize;
            if let Some(&fm) = self.table.get(idx) {
                self.tranche_formats.push(fm);
            }
        }
    }

    pub fn tranche_flags(&mut self, flags: u32) {
        self.tranche_flags = flags;
    }

    /// The `tranche_done` event: seal the accumulated tranche.
    pub fn tranche_done(&mut self) {
        let device = self
            .tranche_device
            .take()
            .unwrap_or(self.pending.main_device);
        self.pending.tranches.push(Tranche {
            device,
            flags: self.tranche_flags,
            formats: std::mem::take(&mut self.tranche_formats),
        });
        self.tranche_flags = 0;
    }

    /// The `done` event: commit the pending burst as the current snapshot. The
    /// format table survives for the next burst; everything else resets.
    pub fn done(&mut self) {
        self.ready = Some(std::mem::take(&mut self.pending));
        self.tranche_device = None;
        self.tranche_flags = 0;
        self.tranche_formats.clear();
    }
}

/// Parse the format table: 16-byte entries of `u32 format`, 4 bytes padding,
/// `u64 modifier`, all native-endian. A trailing partial entry is ignored.
pub fn parse_format_table(bytes: &[u8]) -> Vec<FormatModifier> {
    bytes
        .chunks_exact(16)
        .map(|e| FormatModifier {
            format: u32::from_ne_bytes([e[0], e[1], e[2], e[3]]),
            modifier: u64::from_ne_bytes([e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15]]),
        })
        .collect()
}

/// Parse a dev_t sent as a wire array. The kernel ABI dev_t is 64 bits, and
/// that is what compositors send; anything else is malformed.
fn parse_dev(bytes: &[u8]) -> Result<u64> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        Error::msg(format!(
            "dmabuf device array is {} bytes, want 8",
            bytes.len()
        ))
    })?;
    Ok(u64::from_ne_bytes(arr))
}

/// The major number of a dev_t (glibc encoding).
pub fn dev_major(dev: u64) -> u32 {
    (((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff)) as u32
}

/// The minor number of a dev_t (glibc encoding).
pub fn dev_minor(dev: u64) -> u32 {
    (((dev >> 12) & 0xffff_ff00) | (dev & 0xff)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one 16-byte format-table entry.
    fn entry(format: u32, modifier: u64) -> Vec<u8> {
        let mut e = Vec::new();
        e.extend_from_slice(&format.to_ne_bytes());
        e.extend_from_slice(&[0u8; 4]);
        e.extend_from_slice(&modifier.to_ne_bytes());
        e
    }

    /// Encode a dev_t the way glibc's makedev does.
    fn makedev(major: u64, minor: u64) -> u64 {
        ((major & 0xffff_f000) << 32)
            | ((major & 0xfff) << 8)
            | ((minor & 0xffff_ff00) << 12)
            | (minor & 0xff)
    }

    #[test]
    fn fourcc_matches_drm_xrgb8888() {
        // 'X' 'R' '2' '4' packed little-endian, the value from drm_fourcc.h.
        assert_eq!(DRM_FORMAT_XRGB8888, 0x3432_5258);
    }

    #[test]
    fn format_table_parses_entries_and_ignores_a_partial_tail() {
        let mut bytes = entry(DRM_FORMAT_XRGB8888, MOD_LINEAR);
        bytes.extend(entry(0x3432_5241, 0x00ff_ffff_ffff_ffff));
        bytes.extend_from_slice(&[0xAA; 7]); // truncated third entry
        let table = parse_format_table(&bytes);
        assert_eq!(
            table,
            vec![
                FormatModifier {
                    format: DRM_FORMAT_XRGB8888,
                    modifier: MOD_LINEAR
                },
                FormatModifier {
                    format: 0x3432_5241,
                    modifier: 0x00ff_ffff_ffff_ffff
                },
            ]
        );
    }

    #[test]
    fn dev_major_minor_roundtrip_glibc_encoding() {
        let dev = makedev(226, 128);
        assert_eq!(dev_major(dev), 226);
        assert_eq!(dev_minor(dev), 128);
        // Large numbers exercise the split bit fields.
        let dev = makedev(0x12345, 0xabcdef);
        assert_eq!(dev_major(dev), 0x12345);
        assert_eq!(dev_minor(dev), 0xabcdef);
    }

    #[test]
    fn a_full_burst_commits_on_done() {
        let mut fb = FeedbackState::default();
        let mut table = entry(DRM_FORMAT_XRGB8888, MOD_LINEAR);
        table.extend(entry(DRM_FORMAT_XRGB8888, 0x0300_0000_0000_0001));
        fb.set_table(&table);
        fb.main_device(&makedev(226, 128).to_ne_bytes()).unwrap();
        fb.tranche_device(&makedev(226, 128).to_ne_bytes()).unwrap();
        // Indices 0 and 1 resolve; 9 is out of range and skipped.
        let mut indices = Vec::new();
        for i in [0u16, 1, 9] {
            indices.extend_from_slice(&i.to_ne_bytes());
        }
        fb.tranche_formats(&indices);
        fb.tranche_flags(0);
        assert!(fb.ready().is_none(), "nothing commits before done");
        fb.tranche_done();
        fb.done();

        let got = fb.ready().expect("committed");
        assert_eq!(dev_major(got.main_device), 226);
        assert_eq!(got.tranches.len(), 1);
        assert_eq!(got.tranches[0].formats.len(), 2);
        assert_eq!(
            got.modifiers_for(DRM_FORMAT_XRGB8888),
            vec![MOD_LINEAR, 0x0300_0000_0000_0001]
        );
    }

    #[test]
    fn a_tranche_without_target_device_uses_the_main_device() {
        let mut fb = FeedbackState::default();
        fb.set_table(&entry(DRM_FORMAT_XRGB8888, MOD_LINEAR));
        fb.main_device(&makedev(226, 128).to_ne_bytes()).unwrap();
        fb.tranche_formats(&0u16.to_ne_bytes());
        fb.tranche_done();
        fb.done();
        let got = fb.ready().expect("committed");
        assert_eq!(got.tranches[0].device, got.main_device);
    }

    #[test]
    fn a_reannouncement_replaces_the_snapshot_and_keeps_the_table() {
        let mut fb = FeedbackState::default();
        fb.set_table(&entry(DRM_FORMAT_XRGB8888, MOD_LINEAR));
        fb.main_device(&makedev(226, 128).to_ne_bytes()).unwrap();
        fb.tranche_formats(&0u16.to_ne_bytes());
        fb.tranche_done();
        fb.done();

        // Second burst: no new table, a different main device, one tranche.
        fb.main_device(&makedev(226, 129).to_ne_bytes()).unwrap();
        fb.tranche_formats(&0u16.to_ne_bytes());
        fb.tranche_done();
        fb.done();

        let got = fb.ready().expect("committed");
        assert_eq!(dev_minor(got.main_device), 129);
        assert_eq!(got.tranches.len(), 1, "old tranches do not accumulate");
        assert_eq!(got.tranches[0].formats.len(), 1, "table survived the done");
    }

    #[test]
    fn a_malformed_device_array_is_an_error() {
        let mut fb = FeedbackState::default();
        assert!(fb.main_device(&[1, 2, 3, 4]).is_err());
    }

    #[test]
    fn modifiers_for_deduplicates_across_tranches() {
        let mut fb = FeedbackState::default();
        let mut table = entry(DRM_FORMAT_XRGB8888, MOD_LINEAR);
        table.extend(entry(DRM_FORMAT_XRGB8888, 7));
        fb.set_table(&table);
        fb.main_device(&makedev(226, 128).to_ne_bytes()).unwrap();
        // Two tranches, both listing entry 0; the second adds entry 1.
        fb.tranche_formats(&0u16.to_ne_bytes());
        fb.tranche_done();
        let mut indices = Vec::new();
        for i in [0u16, 1] {
            indices.extend_from_slice(&i.to_ne_bytes());
        }
        fb.tranche_formats(&indices);
        fb.tranche_done();
        fb.done();
        let got = fb.ready().expect("committed");
        assert_eq!(got.modifiers_for(DRM_FORMAT_XRGB8888), vec![MOD_LINEAR, 7]);
    }
}
