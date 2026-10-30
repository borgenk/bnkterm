//! The two texture atlases glyph rasters are packed into. Rasters go onto shelves, and
//! each new one waits in an [`AtlasUpload`] until the frame takes it, so only new rasters
//! cross to the GPU.

use crate::render::gpu::{AtlasInfo, AtlasUpload, UploadRegion};

/// Atlases start here and double when full, up to [`ATLAS_MAX`]. Growing wipes the
/// atlas and the frame build restarts, so no stale coordinates survive. Full at the
/// cap, an atlas is wiped back to the start instead (see [`Atlas::spent`]). Either way
/// the live frame's glyphs are rasterized again, a burst bounded by the frame.
const ATLAS_START: u32 = 1024;

const ATLAS_MAX: u32 = 8192;

/// Where a raster landed in an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) w: u32,
    pub(super) h: u32,
}

/// A shelf-packing texture atlas. Rasters are packed onto horizontal shelves (rows
/// share a height class), and each new raster's bytes wait in an [`AtlasUpload`] until
/// the frame takes them. The texels live only on the GPU: a texture is recreated only
/// when the atlas is wiped, and the rasters packed after the wipe are its whole upload.
pub struct Atlas {
    width: u32,
    height: u32,
    /// The side the atlas grows to at most: [`ATLAS_MAX`], or less in tests that
    /// need to fill one.
    max: u32,
    /// Bytes per texel: 1 (R8 coverage) or 4 (BGRA emoji).
    bpp: u32,
    /// Shelves as (y, height, x_used).
    shelves: Vec<(u32, u32, u32)>,
    next_y: u32,
    pub(super) generation: u32,
    /// Rasters written since the last [`Atlas::take_upload`].
    pending: AtlasUpload,
    /// The atlas holds slots nothing will look up again: it filled at its cap, or
    /// the records pointing into it were dropped.
    /// [`build_frame_into`](crate::render::gpu::build_frame_into) wipes it and rebuilds
    /// the frame into the reclaimed space.
    pub(super) spent: bool,
    /// A slot map pointing into this atlas was dropped since the atlas was last
    /// wiped (see [`Atlas::records_dropped`]).
    dropped_since_wipe: bool,
}

impl Atlas {
    pub fn new(bpp: u32) -> Self {
        Self::with_max(bpp, ATLAS_MAX)
    }

    /// An atlas that grows to at most `max` texels a side, starting at
    /// [`ATLAS_START`] or at `max` when that is smaller.
    pub(super) fn with_max(bpp: u32, max: u32) -> Self {
        let side = ATLAS_START.min(max);
        Self {
            width: side,
            height: side,
            max,
            bpp,
            shelves: Vec::new(),
            next_y: 0,
            generation: 0,
            pending: AtlasUpload::default(),
            spent: false,
            dropped_since_wipe: false,
        }
    }

    pub fn info(&self) -> AtlasInfo {
        AtlasInfo {
            width: self.width,
            height: self.height,
            generation: self.generation,
        }
    }

    /// Reserve a `w` x `h` slot. `None` means the raster does not fit: it is
    /// larger than the atlas can grow, or the atlas is full at its cap, which marks
    /// it [`spent`](Self::spent) so the frame build wipes it and the glyph packs
    /// into the reclaimed space. Growing wipes all previous slots and bumps the
    /// generation, which the frame builder answers by restarting the build.
    fn reserve(&mut self, w: u32, h: u32) -> Option<Slot> {
        if w == 0 || h == 0 || w > self.max || h > self.max {
            return None;
        }
        loop {
            // A shelf fits if the raster is no taller than the shelf and the
            // remaining width holds it. Shelves are created at the raster's
            // height, so height classes cluster and waste stays bounded.
            if let Some(shelf) = self
                .shelves
                .iter_mut()
                .find(|(_, sh, used)| h <= *sh && used + w <= self.width)
            {
                let slot = Slot {
                    x: shelf.2,
                    y: shelf.0,
                    w,
                    h,
                };
                shelf.2 += w;
                return Some(slot);
            }
            if self.next_y + h <= self.height && w <= self.width {
                self.shelves.push((self.next_y, h, w));
                let slot = Slot {
                    x: 0,
                    y: self.next_y,
                    w,
                    h,
                };
                self.next_y += h;
                return Some(slot);
            }
            if !self.grow() {
                self.spent = true;
                return None;
            }
        }
    }

    /// Double the atlas (square), wiping every slot. Returns false at the cap.
    fn grow(&mut self) -> bool {
        let next = (self.width * 2).min(self.max);
        if next == self.width {
            return false;
        }
        self.reset(next);
        true
    }

    /// Empty the atlas back to its starting side.
    pub(super) fn wipe(&mut self) {
        self.reset(ATLAS_START.min(self.max));
        self.dropped_since_wipe = false;
    }

    /// A slot map pointing into this atlas was dropped, orphaning its slots. The
    /// first drop after a wipe marks the atlas spent. A second one before the next
    /// wipe means the screen alone holds more glyphs than the map, so wiping on
    /// every frame would reclaim nothing; the orphans are left until the atlas fills
    /// at its cap and is wiped there.
    pub(super) fn records_dropped(&mut self) {
        if !self.dropped_since_wipe {
            self.spent = true;
        }
        self.dropped_since_wipe = true;
    }

    /// Empty the atlas at `side` texels square and bump the generation, which kills
    /// every slot handed out before and has the renderer recreate the texture.
    fn reset(&mut self, side: u32) {
        self.width = side;
        self.height = side;
        self.shelves.clear();
        self.next_y = 0;
        self.generation = self.generation.wrapping_add(1);
        self.pending.clear();
    }

    /// Queue a raster written at `slot` for the next upload. `src` is `w * h * bpp`
    /// bytes, row-major, tight; [`Self::pack`] has checked it holds that many.
    fn write(&mut self, slot: Slot, src: &[u8]) {
        if let Some(src) = src.get(..self.raster_len(slot.w, slot.h)) {
            self.pending
                .push(slot, |bytes| bytes.extend_from_slice(src));
        }
    }

    /// Queue a `w`-word-per-row ARGB raster (`w * h` words, row-major, tight) written at
    /// `slot`, one pixel's `to_ne_bytes` at a time. Lets the color atlas (`bpp == 4`)
    /// pack a glyph straight from its words without first collecting them into a byte
    /// vector.
    fn write_words(&mut self, slot: Slot, src: &[u32]) {
        let words = (slot.w as usize).saturating_mul(slot.h as usize);
        if let Some(src) = src.get(..words) {
            self.pending.push(slot, |bytes| {
                bytes.extend(src.iter().flat_map(|word| word.to_ne_bytes()));
            });
        }
    }

    /// Reserve a slot for a `w` x `h` raster and write `src` into it, returning
    /// the slot; `None` for a zero-area raster or a full atlas (the caller
    /// records the empty placement so the pen still advances). `src` is
    /// `w * h * bpp` bytes, row-major, tight.
    pub(super) fn pack(&mut self, w: u32, h: u32, src: &[u8]) -> Option<Slot> {
        if w == 0 || h == 0 || src.len() < self.raster_len(w, h) {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write(slot, src);
        Some(slot)
    }

    /// Like [`pack`](Self::pack) but from ARGB words (the color-glyph path), queuing
    /// their bytes directly. `src` is `w * h` words, row-major, tight.
    fn pack_words(&mut self, w: u32, h: u32, src: &[u32]) -> Option<Slot> {
        if w == 0 || h == 0 || src.len() < (w as usize).saturating_mul(h as usize) {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write_words(slot, src);
        Some(slot)
    }

    /// How many bytes a tight `w`-by-`h` raster occupies at this atlas's depth: what
    /// [`Self::write`] will read, and therefore what [`Self::pack`] insists on having.
    ///
    /// The check exists because the dimensions and the buffer come from different places
    /// and are not guaranteed to agree. `copy_coverage` returns an empty `Vec` when
    /// FreeType hands back a null `bitmap.buffer`, while `Glyph::width`/`rows` keep the
    /// nonzero dimensions the metrics reported — and `packed_scalar` then packs
    /// `(g.width, g.rows, &g.coverage)`, so the slice range in `write` traps. Release
    /// builds are `panic = "abort"`, so that is the whole process for a font or driver
    /// anomaly. `pack_words` already had this shape; `pack` did not.
    fn raster_len(&self, w: u32, h: u32) -> usize {
        (w as usize)
            .saturating_mul(h as usize)
            .saturating_mul(self.bpp as usize)
    }

    /// Hand the rasters written since the last call to `out`, which takes them in
    /// place of what it held; the two buffers trade back and forth, so a frame of new
    /// glyphs allocates nothing once both have grown.
    pub fn take_upload(&mut self, out: &mut AtlasUpload) {
        out.clear();
        std::mem::swap(out, &mut self.pending);
    }
}

impl AtlasUpload {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    fn clear(&mut self) {
        self.regions.clear();
        self.bytes.clear();
    }

    /// Queue a raster written at `slot`, its bytes appended by `fill`.
    fn push(&mut self, slot: Slot, fill: impl FnOnce(&mut Vec<u8>)) {
        self.regions.push(UploadRegion {
            x: slot.x,
            y: slot.y,
            w: slot.w,
            h: slot.h,
            offset: self.bytes.len(),
        });
        fill(&mut self.bytes);
    }
}

/// Pack an emoji cluster's straight-`0xAARRGGBB` raster into the color atlas.
/// `None` for a zero-area or malformed glyph or a full atlas; the placement is
/// recorded as `None` so the pen advances without drawing.
pub(super) fn pack_argb(atlas: &mut Atlas, w: u32, h: u32, argb: &[u32]) -> Option<Slot> {
    // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim, queued straight from
    // the words rather than first collected into a byte-conversion vector.
    // `pack_words` owns the dimension-versus-buffer check, so no caller can skip it.
    atlas.pack_words(w, h, argb)
}

#[cfg(test)]
mod tests {
    use crate::render::gpu::atlas::*;
    use crate::render::gpu::tests::*;
    use crate::render::gpu::*;

    #[test]
    fn atlas_packs_onto_shelves_and_queues_what_it_writes() {
        let mut a = Atlas::new(1);
        let s1 = a.reserve(10, 8).expect("fits");
        let s2 = a.reserve(6, 8).expect("fits on the same shelf");
        assert_eq!((s1.x, s1.y), (0, 0));
        assert_eq!((s2.x, s2.y), (10, 0), "same shelf, packed rightward");
        let s3 = a.reserve(10, 20).expect("taller: a new shelf");
        assert_eq!(s3.y, 8, "new shelf below the first");
        a.write(s1, &[7u8; 80]);
        let mut up = AtlasUpload::default();
        a.take_upload(&mut up);
        assert_eq!(
            up.regions,
            [UploadRegion {
                x: 0,
                y: 0,
                w: 10,
                h: 8,
                offset: 0
            }]
        );
        assert!(up.bytes.iter().all(|&b| b == 7));
        a.take_upload(&mut up);
        assert!(up.is_empty(), "taken once");
    }

    /// Two glyphs far apart in the atlas upload their own texels. Their bounding box,
    /// what used to go up, is here most of a shelf's width by two shelves' height.
    #[test]
    fn an_upload_carries_only_the_rasters_written() {
        let mut a = Atlas::new(1);
        a.pack(900, 8, &[1; 900 * 8])
            .expect("a wide raster on the first shelf");
        a.pack(10, 300, &[2; 10 * 300])
            .expect("a tall one on a second shelf");
        let mut up = AtlasUpload::default();
        a.take_upload(&mut up);

        let near = a.pack(10, 8, &[3; 10 * 8]).expect("beside the wide one");
        let far = a
            .pack(10, 300, &[4; 10 * 300])
            .expect("beside the tall one");
        a.take_upload(&mut up);

        assert_eq!(
            up.bytes.len(),
            10 * 8 + 10 * 300,
            "only the two new rasters"
        );
        let placed: Vec<_> = up.regions.iter().map(|r| (r.x, r.y, r.w, r.h)).collect();
        assert_eq!(
            placed,
            [(near.x, near.y, 10, 8), (far.x, far.y, 10, 300)],
            "each where it was packed"
        );
        let second = up.regions[1].offset;
        assert!(up.bytes[..second].iter().all(|&b| b == 3));
        assert!(up.bytes[second..].iter().all(|&b| b == 4));
    }

    #[test]
    fn packing_a_raster_shorter_than_its_dimensions_refuses_instead_of_trapping() {
        // The dimensions and the buffer come from different places and are not guaranteed
        // to agree: `copy_coverage` returns an empty `Vec` when FreeType hands back a null
        // `bitmap.buffer`, while `Glyph::width`/`rows` keep the nonzero dimensions the
        // metrics reported. `packed_scalar` then packs `(width, rows, &coverage)` and the
        // slice range inside `write` traps — and release is `panic = "abort"`, so a font
        // or driver anomaly takes the whole process.
        let mut a = Atlas::new(1);
        assert_eq!(a.pack(4, 4, &[]), None, "the empty-coverage case exactly");
        assert_eq!(a.pack(4, 4, &[0u8; 15]), None, "one byte short is short");
        assert!(
            a.pack(4, 4, &[0u8; 16]).is_some(),
            "and exactly enough fits"
        );

        // The colour atlas counts four bytes to the pixel, so the same raster needs four
        // times the buffer. A check that forgot `bpp` would pass this and then trap.
        let mut a = Atlas::new(4);
        assert_eq!(a.pack(4, 4, &[0u8; 16]), None);
        assert!(a.pack(4, 4, &[0u8; 64]).is_some());

        // And the word path, which already refused, still does.
        let mut a = Atlas::new(4);
        assert_eq!(a.pack_words(2, 2, &[0u32; 3]), None);
        assert!(a.pack_words(2, 2, &[0u32; 4]).is_some());
    }

    #[test]
    fn pack_words_lays_down_the_same_bytes_as_a_conversion_would() {
        // The color-glyph path queues ARGB words straight into the 4-bpp upload; the
        // result must match converting each word to its native bytes first, so
        // dropping that conversion vector changed nothing on screen.
        let mut a = Atlas::new(4);
        let argb: [u32; 4] = [0x1122_3344, 0x5566_7788, 0x99AA_BBCC, 0xDDEE_FF00];
        a.pack_words(2, 2, &argb).expect("fits");
        let mut up = AtlasUpload::default();
        a.take_upload(&mut up);
        assert_eq!((up.regions[0].w, up.regions[0].h), (2, 2));
        let expect: Vec<u8> = argb.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(up.bytes, expect);
    }

    #[test]
    fn atlas_grows_by_wiping_and_bumping_generation() {
        let mut a = Atlas::new(1);
        let g0 = a.generation;
        // A raster wider than the atlas forces growth.
        let s = a.reserve(ATLAS_START + 1, 4).expect("grows to fit");
        assert!(a.generation > g0);
        assert_eq!(a.width, ATLAS_START * 2);
        assert_eq!((s.x, s.y), (0, 0), "the grown atlas starts empty");
        // Beyond the cap is a clean None.
        assert!(a.reserve(ATLAS_MAX + 1, 1).is_none());
    }

    /// The direct-mapped rows hold atlas coordinates exactly as the map does, so an
    /// atlas wipe has to drop both. The map has always been dropped on a generation
    /// bump; if the rows are not, every ASCII glyph silently renders from a dead slot
    /// — the one corruption splitting the cache in two could introduce, and one no
    /// rendering test would notice until it was on screen.
    #[test]
    fn an_atlas_wipe_invalidates_the_direct_mapped_ascii_rows() {
        let mut cache = GlyphCache::default();
        let face = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        cache.cache_scalar((face, 'A'), placed());
        cache.cache_scalar((face, '\u{2500}'), placed());
        assert!(cache.ascii_get(face, 'A').is_some());
        assert!(cache.scalar_slots.contains_key(&(face, '\u{2500}')));

        // A raster wider than the atlas forces it to grow, wiping every slot.
        let g0 = cache.glyphs.generation;
        let emoji_gen = cache.emoji.generation;
        cache
            .glyphs
            .reserve(ATLAS_START + 1, 4)
            .expect("grows to fit");
        assert!(
            cache.glyphs.generation > g0,
            "the wipe bumps the generation"
        );
        cache.reset_for(g0, emoji_gen);

        assert!(
            cache.ascii_get(face, 'A').is_none(),
            "a direct-mapped ASCII row survived an atlas wipe, so it now points at a dead slot"
        );
        assert!(
            cache.scalar_slots.is_empty(),
            "the map is dropped as before"
        );
    }

    /// Dropping a slot map orphans the atlas slots its records held. Unless the
    /// atlas is wiped with it, the glyphs re-pack into fresh space and each drop
    /// walks the atlas closer to its cap. Every fill of the map gets its wipe, not
    /// only the first.
    #[test]
    fn dropping_a_slot_map_wipes_the_atlas_it_pointed_into() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache {
            max_records: 16,
            ..GlyphCache::default()
        };
        let face = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        for round in 0..2 {
            cache.glyphs.reserve(4, 4).expect("fits");
            cache.emoji.reserve(4, 4).expect("fits");
            let (glyphs_gen, emoji_gen) = (cache.glyphs.generation, cache.emoji.generation);
            for i in 0..=cache.max_records {
                let ch = char::from_u32(0x1_0000 + i as u32).expect("valid scalar");
                cache.cache_scalar((face, ch), placed());
                cache.cache_cluster(EmojiFit::Em, face, &format!("c{i}"), Some(placed()));
            }

            build_frame(&fonts, &[], &mut cache, GAMMA);
            assert_ne!(
                cache.glyphs.generation, glyphs_gen,
                "round {round}: the glyph atlas kept its slots"
            );
            assert_ne!(
                cache.emoji.generation, emoji_gen,
                "round {round}: the emoji atlas kept its slots"
            );
            assert_eq!((cache.glyphs.next_y, cache.emoji.next_y), (0, 0));
        }
    }
}
