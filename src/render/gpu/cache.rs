//! The glyph cache's slot maps: where each scalar, box glyph and emoji cluster was packed,
//! with its pen metrics, so a steady frame emits its quads without rasterizing anything.
//! Every map is bounded, and dropping one tells the atlas its slots are orphaned.

use std::collections::HashMap;

use crate::platform::freetype::FaceKey;
use crate::render::gpu::atlas::{Atlas, Slot};
use crate::render::gpu::{CacheStats, GlyphCache};

/// The most slot records a glyph cache keeps before it is dropped whole. The atlas
/// is bounded by `ATLAS_MAX`, but the slot maps are not: a `None` record is kept
/// for every distinct non-emoji cluster (the negative cache) and for every glyph
/// that would not fit a full atlas, so a child printing a stream of distinct
/// clusters or glyphs would grow them without limit. Far above any real workload's
/// distinct-glyph count, so ordinary use never trips it; when it does, the map is
/// dropped and the atlas it pointed into is wiped, so the glyphs re-cache into
/// reclaimed space (a bounded burst) rather than accumulating forever.
const MAX_GLYPH_CACHE: usize = 1 << 16;

/// The first printable ASCII scalar, and how many there are (`0x20..=0x7E`): the
/// span [`GlyphCache`] direct-maps.
const ASCII_FIRST: u32 = 0x20;

const ASCII_SLOTS: usize = 95;

/// The most faces that hold a direct-mapped ASCII row at once. Four styles across a
/// couple of sizes is the real ceiling (a `FaceKey`'s size changes only when the
/// *user* zooms, never at a child's request), so this is slack, not a budget; past
/// it, ASCII simply falls back to the map and stays correct.
const MAX_ASCII_FACES: usize = 16;

/// A glyph's atlas placement plus its pen metrics, cached beside the slot so a
/// steady-state frame emits the quad without re-rasterising the glyph. `slot` is
/// `None` for an inkless glyph (a space, or an emoji that shaped to no ink) or
/// one the atlas could not fit; the pen still advances by `advance`, so the
/// caret and the glyphs after it land where layout measured.
#[derive(Clone, Copy)]
pub(super) struct PackedGlyph {
    pub(super) slot: Option<Slot>,
    pub(super) left: i32,
    pub(super) top: i32,
    pub(super) advance: f32,
}

/// One face's placements for printable ASCII, indexed by `ch - 0x20`.
pub(super) struct AsciiRow {
    face: FaceKey,
    slots: Box<[Option<PackedGlyph>; ASCII_SLOTS]>,
}

/// Emoji placements per face, by cluster. Nested so lookups borrow the cluster as
/// `&str` (no per-frame `String`). A `None` value records a cluster that is not
/// color emoji, so a steady-state frame takes the per-character path without
/// re-consulting the `Face`.
pub(super) type ClusterSlots = HashMap<FaceKey, HashMap<String, Option<PackedGlyph>>>;

/// The size an emoji cluster is rasterized at, which also picks the map that caches
/// it. The face key fixes both sizes, so `(face_key, cluster)` is a complete key
/// within either map.
#[derive(Clone, Copy)]
pub(super) enum EmojiFit {
    /// The em, for a wide cluster on the
    /// [`Batcher::text`](crate::render::gpu::Batcher::text) path: the ~square glyph
    /// fills both cells and its advance carries the pen.
    Em,
    /// One cell's width, for a single-width cluster on the
    /// [`Batcher::cells`](crate::render::gpu::Batcher::cells) path, so it does not spill
    /// past its column.
    Cell(u32),
}

impl EmojiFit {
    pub(super) fn target(self, face_key: FaceKey) -> u32 {
        match self {
            EmojiFit::Em => face_key.size(),
            EmojiFit::Cell(width) => width,
        }
    }
}

impl CacheStats {
    /// Total lookups (hits + misses).
    pub fn lookups(self) -> u64 {
        self.hits + self.misses
    }

    /// Fraction of lookups served from the cache, 0.0 when nothing was looked up.
    pub fn hit_rate(self) -> f32 {
        match self.lookups() {
            0 => 0.0,
            total => self.hits as f32 / total as f32,
        }
    }
}

impl Default for GlyphCache {
    /// Two empty atlases: one byte per pixel for the mono glyphs, four for the colour
    /// emoji. Both grow on demand.
    fn default() -> Self {
        Self {
            glyphs: Atlas::new(1),
            emoji: Atlas::new(4),
            ascii_slots: Vec::new(),
            scalar_slots: HashMap::new(),
            box_slots: HashMap::new(),
            cluster_slots: HashMap::new(),
            cell_cluster_slots: HashMap::new(),
            max_records: MAX_GLYPH_CACHE,
            stats: CacheStats::default(),
        }
    }
}

impl GlyphCache {
    /// Hits and misses since the cache was made.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// The direct-mapped index for a printable-ASCII scalar, or `None` for anything
    /// else (which belongs in [`Self::scalar_slots`]). The partition is total: a
    /// character is direct-mapped or hashed, never both, so the two can never
    /// disagree about a placement.
    fn ascii_index(ch: char) -> Option<usize> {
        let i = (ch as u32).wrapping_sub(ASCII_FIRST);
        (i < ASCII_SLOTS as u32).then_some(i as usize)
    }

    /// A printable-ASCII glyph's cached placement: find the face, index the row.
    pub(super) fn ascii_get(&self, face: FaceKey, ch: char) -> Option<PackedGlyph> {
        let i = Self::ascii_index(ch)?;
        self.ascii_slots.iter().find(|r| r.face == face)?.slots[i]
    }

    /// Drop every slot record that points into an atlas wiped since `glyphs_gen` or
    /// `emoji_gen`.
    pub(super) fn reset_for(&mut self, glyphs_gen: u32, emoji_gen: u32) {
        if self.glyphs.generation != glyphs_gen {
            self.scalar_slots.clear();
            // Box rasters live in the same atlas, so a wipe takes them with it.
            self.box_slots.clear();
            // The direct-mapped rows hold atlas coordinates too; a wipe invalidates
            // them exactly as it does the map, and forgetting them here would leave
            // every ASCII glyph pointing into a dead slot.
            self.ascii_slots.clear();
        }
        if self.emoji.generation != emoji_gen {
            self.cluster_slots.clear();
            self.cell_cluster_slots.clear();
        }
    }

    /// The map that caches emoji at `fit`'s size.
    pub(super) fn clusters(&self, fit: EmojiFit) -> &ClusterSlots {
        match fit {
            EmojiFit::Em => &self.cluster_slots,
            EmojiFit::Cell(_) => &self.cell_cluster_slots,
        }
    }

    /// Record a scalar glyph's placement, dropping the whole scalar cache first if it
    /// has reached [`MAX_GLYPH_CACHE`], so distinct-glyph spam cannot grow it without
    /// bound. The dropped records held glyph-atlas slots that would otherwise be
    /// orphaned while the glyphs re-pack into fresh space, so the atlas is told (see
    /// [`Atlas::records_dropped`]).
    pub(super) fn cache_scalar(&mut self, key: (FaceKey, char), packed: PackedGlyph) {
        let (face, ch) = key;
        if let Some(i) = Self::ascii_index(ch) {
            if let Some(row) = self.ascii_slots.iter_mut().find(|r| r.face == face) {
                row.slots[i] = Some(packed);
                return;
            }
            if self.ascii_slots.len() < MAX_ASCII_FACES {
                let mut slots = Box::new([None; ASCII_SLOTS]);
                slots[i] = Some(packed);
                self.ascii_slots.push(AsciiRow { face, slots });
                return;
            }
            // Past the face ceiling ASCII keeps working, just through the map, which
            // is why `ascii_get` missing is never taken as "no such glyph".
        }
        if self.scalar_slots.len() >= self.max_records {
            self.scalar_slots.clear();
            self.glyphs.records_dropped();
        }
        self.scalar_slots.insert(key, packed);
    }

    /// Record a cluster's placement (or `None` for a non-emoji cluster: the negative
    /// cache that keeps a steady frame off the shaper), capping the per-face map the
    /// same way so a stream of distinct clusters cannot grow it without bound, and
    /// telling the emoji atlas its records pointed into.
    pub(super) fn cache_cluster(
        &mut self,
        fit: EmojiFit,
        face_key: FaceKey,
        cluster: &str,
        packed: Option<PackedGlyph>,
    ) {
        let map = match fit {
            EmojiFit::Em => &mut self.cluster_slots,
            EmojiFit::Cell(_) => &mut self.cell_cluster_slots,
        };
        let by = map.entry(face_key).or_default();
        if by.len() >= self.max_records {
            by.clear();
            self.emoji.records_dropped();
        }
        by.insert(cluster.to_string(), packed);
    }
}

#[cfg(test)]
mod tests {
    use crate::render::gpu::cache::*;
    use crate::render::gpu::tests::*;
    use crate::render::gpu::*;

    #[test]
    fn glyph_caches_stay_bounded_under_distinct_input() {
        // Past the cap the maps drop and re-fill rather than growing forever, so a
        // hostile stream of distinct glyphs or clusters cannot exhaust memory.
        let mut cache = GlyphCache::default();
        let face = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let none = PackedGlyph {
            slot: None,
            left: 0,
            top: 0,
            advance: 0.0,
        };
        for i in 0..MAX_GLYPH_CACHE + 5 {
            let ch = char::from_u32(0x1_0000 + i as u32).expect("valid scalar");
            cache.cache_scalar((face, ch), none);
            cache.cache_cluster(EmojiFit::Em, face, &format!("c{i}"), None);
            cache.cache_cluster(EmojiFit::Cell(8), face, &format!("c{i}"), None);
        }
        assert!(cache.scalar_slots.len() <= MAX_GLYPH_CACHE);
        for map in [&cache.cluster_slots, &cache.cell_cluster_slots] {
            let clusters: usize = map.values().map(|m| m.len()).sum();
            assert!(clusters <= MAX_GLYPH_CACHE);
        }
    }

    /// Text the emoji font never draws stays out of the emoji maps, so a screen of
    /// distinct letters cannot fill them, drop the emoji placed so far and wipe their
    /// atlas.
    #[test]
    fn plain_text_leaves_the_emoji_cache_alone() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache {
            max_records: 16,
            ..GlyphCache::default()
        };
        build_frame(&fonts, &prose("😀"), &mut cache, GAMMA);
        let emoji_gen = cache.emoji.generation;
        let greek: String = ('α'..='ω').collect();
        build_frame(&fonts, &prose(&greek), &mut cache, GAMMA);
        assert_eq!(
            cache.emoji.generation, emoji_gen,
            "letters wiped the emoji atlas"
        );
        let clusters: usize = cache.cluster_slots.values().map(|m| m.len()).sum();
        assert_eq!(clusters, 1, "only the emoji is in the map");
    }

    /// One glyph on each lookup path: an emoji, then an ASCII letter, a box-drawing
    /// glyph and an accented letter on the grid. The first frame packs all four and the
    /// same frame again finds all four.
    #[test]
    fn stats_count_each_lookup_path() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::default();
        let mut list = prose("😀");
        list.push(DrawCmd::Cells {
            bounds: Rect {
                x: 0,
                y: 20,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 36,
            cell_w: 10,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            bg: 0,
            text: "a─é".to_string(),
        });
        build_frame(&fonts, &list, &mut cache, GAMMA);
        let first = cache.stats();
        assert_eq!((first.hits, first.misses), (0, 4), "a cold frame misses");
        build_frame(&fonts, &list, &mut cache, GAMMA);
        let second = cache.stats();
        assert_eq!((second.hits, second.misses), (4, 4), "a warm frame hits");
    }

    /// The two stores must partition the characters between them: direct-mapped or
    /// hashed, never both, or the two could disagree about one glyph's placement.
    #[test]
    fn ascii_is_direct_mapped_and_every_other_scalar_is_hashed() {
        let mut cache = GlyphCache::default();
        let face = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        // The span and both its edges.
        for ch in [' ', 'A', '~'] {
            cache.cache_scalar((face, ch), placed());
            assert!(
                cache.ascii_get(face, ch).is_some(),
                "{ch:?} should be mapped"
            );
            assert!(
                !cache.scalar_slots.contains_key(&(face, ch)),
                "{ch:?} should not also be hashed"
            );
        }
        // Just outside it at either end, plus the exotica an attacker actually picks.
        for ch in ['\u{1f}', '\u{7f}', 'é', '\u{2500}', '\u{1F600}'] {
            cache.cache_scalar((face, ch), placed());
            assert!(
                cache.ascii_get(face, ch).is_none(),
                "{ch:?} is not printable ASCII and must not be direct-mapped"
            );
            assert!(
                cache.scalar_slots.contains_key(&(face, ch)),
                "{ch:?} should be hashed"
            );
        }
    }

    /// Past the face ceiling the rows stop being handed out, and ASCII has to keep
    /// working through the map — `ascii_get` missing a glyph means "not here", never
    /// "no such glyph".
    #[test]
    fn ascii_past_the_face_ceiling_falls_back_to_the_map() {
        let mut cache = GlyphCache::default();
        for i in 0..MAX_ASCII_FACES {
            cache.cache_scalar(
                (
                    FaceKey::Prose {
                        size: i as u32,
                        style: crate::platform::freetype::FontStyle::Regular,
                    },
                    'A',
                ),
                placed(),
            );
        }
        assert_eq!(cache.ascii_slots.len(), MAX_ASCII_FACES);
        let extra = FaceKey::Prose {
            size: 9999,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        cache.cache_scalar((extra, 'A'), placed());
        assert!(
            cache.ascii_get(extra, 'A').is_none(),
            "no row is left for this face"
        );
        assert!(
            cache.scalar_slots.contains_key(&(extra, 'A')),
            "so the glyph must still be reachable through the map"
        );
    }
}
