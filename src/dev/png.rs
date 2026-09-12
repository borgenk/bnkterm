//! PNG encoding, for the screenshot harness next door.
//!
//! A frame arrives as ARGB8888 words and goes out as 8-bit RGBA. The surface is
//! presented as `XRGB8888`, so the alpha the GPU hands back means nothing; the
//! harness sets it, opaque everywhere but the rounded corners it carves, and those
//! corners are the reason the file carries an alpha channel at all. The zlib stream
//! is one deflate block in the fixed Huffman codes over an LZ77 pass, which needs no
//! code-length table and still compresses a terminal frame heavily, since most of one
//! is long runs of the background colour. Nothing reads a PNG back; the reader in the
//! tests checks what this wrote.

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Encode pixels, row-major and `width` words to a row, as an 8-bit RGBA PNG. The
/// words carry straight alpha, which is what PNG stores.
pub fn encode(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
    let raw = scanlines(pixels, width, height);
    let mut out = Vec::from(SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    // 8 bits per sample, color type 6 (truecolor with alpha), no compression,
    // filter or interlace variation beyond the one PNG defines.
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// The raw PNG image data: every row prefixed with filter type 0 (none).
fn scanlines(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut raw = Vec::with_capacity(h * (1 + w * 4));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let px = pixels.get(y * w + x).copied().unwrap_or(0);
            raw.push((px >> 16) as u8);
            raw.push((px >> 8) as u8);
            raw.push(px as u8);
            raw.push((px >> 24) as u8);
        }
    }
    raw
}

/// Wrap `data` in a zlib stream: one fixed-Huffman deflate block over an LZ77
/// pass, then the Adler checksum.
fn zlib(data: &[u8]) -> Vec<u8> {
    // CMF 0x78 (deflate, 32K window) and FLG 0x01, whose check bits make the
    // pair a multiple of 31.
    let mut w = BitWriter::new(vec![0x78, 0x01]);
    w.bits(1, 1); // final block
    w.bits(1, 2); // fixed Huffman codes
    deflate_fixed(data, &mut w);
    let mut out = w.finish();
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Bits go out least significant first, except Huffman codes, which deflate
/// packs most significant first.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new(out: Vec<u8>) -> Self {
        BitWriter { out, acc: 0, n: 0 }
    }

    fn bits(&mut self, value: u32, count: u32) {
        self.acc |= (value & ((1 << count) - 1)) << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, whose bits are defined most significant first.
    fn code(&mut self, code: u32, count: u32) {
        for i in (0..count).rev() {
            self.bits((code >> i) & 1, 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// Emit one literal or length symbol in deflate's fixed code.
fn fixed_symbol(w: &mut BitWriter, sym: u16) {
    match sym {
        0..=143 => w.code(0x30 + u32::from(sym), 8),
        144..=255 => w.code(0x190 + u32::from(sym) - 144, 9),
        256..=279 => w.code(u32::from(sym) - 256, 7),
        _ => w.code(0xc0 + u32::from(sym) - 280, 8),
    }
}

/// Greedy LZ77 over a hash of the next three bytes, emitted in the fixed code.
fn deflate_fixed(data: &[u8], w: &mut BitWriter) {
    const WINDOW: usize = 32768;
    const MIN_MATCH: usize = 3;
    const MAX_MATCH: usize = 258;
    /// How far back along a hash chain to look, which bounds the worst case on
    /// a flat image where one hash covers a very long run.
    const MAX_CHAIN: usize = 32;

    let mut heads = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; data.len().max(1)];
    let hash = |d: &[u8], i: usize| -> usize {
        ((usize::from(d[i]) << 10) ^ (usize::from(d[i + 1]) << 5) ^ usize::from(d[i + 2]))
            & ((1 << 15) - 1)
    };

    let mut i = 0;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0usize, 0usize);
        if i + MIN_MATCH <= data.len() {
            let h = hash(data, i);
            let mut candidate = heads[h];
            let mut walked = 0;
            while candidate != usize::MAX && walked < MAX_CHAIN {
                let dist = i - candidate;
                if dist > WINDOW {
                    break;
                }
                let max = MAX_MATCH.min(data.len() - i);
                let mut len = 0;
                while len < max && data[candidate + len] == data[i + len] {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = dist;
                    if len == MAX_MATCH {
                        break;
                    }
                }
                candidate = prev[candidate];
                walked += 1;
            }
            prev[i] = heads[h];
            heads[h] = i;
        }

        if best_len >= MIN_MATCH {
            let li = LENGTH_BASE
                .iter()
                .rposition(|&b| usize::from(b) <= best_len)
                .unwrap_or(0);
            fixed_symbol(w, 257 + li as u16);
            w.bits(
                (best_len - usize::from(LENGTH_BASE[li])) as u32,
                u32::from(LENGTH_EXTRA[li]),
            );
            let di = DIST_BASE
                .iter()
                .rposition(|&b| usize::from(b) <= best_dist)
                .unwrap_or(0);
            w.code(di as u32, 5);
            w.bits(
                (best_dist - usize::from(DIST_BASE[di])) as u32,
                u32::from(DIST_EXTRA[di]),
            );
            // Index the bytes the match covered so later matches can find them.
            let mut k = i + 1;
            while k < i + best_len && k + MIN_MATCH <= data.len() {
                let h = hash(data, k);
                prev[k] = heads[h];
                heads[h] = k;
                k += 1;
            }
            i += best_len;
        } else {
            fixed_symbol(w, u16::from(data[i]));
            i += 1;
        }
    }
    fixed_symbol(w, 256);
}

/// The length and distance alphabets, as deflate fixes them.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Append a length-tagged, CRC-checked PNG chunk.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Crc::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// Running CRC-32, computed bitwise so there is no table to carry.
struct Crc(u32);

impl Crc {
    fn new() -> Self {
        Crc(0xffff_ffff)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u32::from(b);
            for _ in 0..8 {
                // The reflected CRC-32 polynomial.
                let mask = (self.0 & 1).wrapping_neg();
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & mask);
            }
        }
    }

    fn finish(self) -> u32 {
        !self.0
    }
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    // Chunked so the accumulators cannot overflow before the reduction.
    for block in data.chunks(5552) {
        for &byte in block {
            a += u32::from(byte);
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little-endian bit reader over the deflate stream.
    struct Bits<'a> {
        data: &'a [u8],
        byte: usize,
        bit: u32,
    }

    impl<'a> Bits<'a> {
        fn new(data: &'a [u8]) -> Self {
            Bits {
                data,
                byte: 0,
                bit: 0,
            }
        }

        fn next(&mut self) -> u32 {
            let byte = self.data[self.byte];
            let value = u32::from((byte >> self.bit) & 1);
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.byte += 1;
            }
            value
        }

        /// Header and extra bits, which deflate packs least significant first.
        fn bits(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, i| acc | self.next() << i)
        }

        /// A Huffman code, most significant bit first.
        fn code(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, _| acc << 1 | self.next())
        }
    }

    /// One symbol in the fixed code, the four ranges fixed_symbol writes.
    fn fixed_code(bits: &mut Bits) -> u16 {
        let seven = bits.code(7);
        if seven < 0x18 {
            return 256 + seven as u16;
        }
        let eight = seven << 1 | bits.code(1);
        if eight < 0xc0 {
            return (eight - 0x30) as u16;
        }
        if eight < 0xc8 {
            return 280 + (eight - 0xc0) as u16;
        }
        let nine = eight << 1 | bits.code(1);
        144 + (nine - 0x190) as u16
    }

    /// Decompress what zlib produced. The encoder emits one fixed-Huffman block
    /// and nothing else, so anything else here is a failure.
    fn inflate(stream: &[u8]) -> Vec<u8> {
        let mut bits = Bits::new(&stream[2..]);
        assert_eq!(bits.bits(1), 1, "the block is the last one");
        assert_eq!(bits.bits(2), 1, "the block uses the fixed codes");

        let mut out: Vec<u8> = Vec::new();
        loop {
            match fixed_code(&mut bits) {
                256 => return out,
                sym @ 0..=255 => out.push(sym as u8),
                sym => {
                    let i = sym as usize - 257;
                    let len = usize::from(LENGTH_BASE[i])
                        + bits.bits(u32::from(LENGTH_EXTRA[i])) as usize;
                    let d = bits.code(5) as usize;
                    let back =
                        usize::from(DIST_BASE[d]) + bits.bits(u32::from(DIST_EXTRA[d])) as usize;
                    let start = out.len() - back;
                    for k in 0..len {
                        out.push(out[start + k]);
                    }
                }
            }
        }
    }

    /// The IDAT payload, with every chunk's length and CRC checked on the way.
    fn idat(png: &[u8]) -> Vec<u8> {
        assert_eq!(&png[..8], &SIGNATURE, "the file starts as a PNG");
        let mut at = SIGNATURE.len();
        let mut kinds = Vec::new();
        let mut data = Vec::new();
        while at < png.len() {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().expect("length")) as usize;
            let kind = &png[at + 4..at + 8];
            let body = &png[at + 8..at + 8 + len];
            let want =
                u32::from_be_bytes(png[at + 8 + len..at + 12 + len].try_into().expect("crc"));
            let mut crc = Crc::new();
            crc.update(kind);
            crc.update(body);
            assert_eq!(crc.finish(), want, "CRC over {kind:?}");
            if kind == b"IDAT" {
                data.extend_from_slice(body);
            }
            kinds.push(String::from_utf8_lossy(kind).into_owned());
            at += 12 + len;
        }
        assert_eq!(at, png.len(), "chunks tile the file exactly");
        assert_eq!(kinds, ["IHDR", "IDAT", "IEND"]);
        data
    }

    /// What a frame should compress to: filter type 0, then RGBA.
    fn scanlines_of(pixels: &[u32], width: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for row in pixels.chunks(width) {
            raw.push(0);
            for px in row {
                raw.extend_from_slice(&[
                    (px >> 16) as u8,
                    (px >> 8) as u8,
                    *px as u8,
                    (px >> 24) as u8,
                ]);
            }
        }
        raw
    }

    /// Encode, then read it all back: container, deflate block, checksum.
    fn round_trip(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
        let png = encode(pixels, width, height);
        let stream = idat(&png);
        let raw = inflate(&stream);
        assert_eq!(
            &stream[stream.len() - 4..],
            &adler32(&raw).to_be_bytes(),
            "the checksum covers what came out"
        );
        assert_eq!(raw, scanlines_of(pixels, width as usize));
        png
    }

    /// Channels reach the file in PNG's order, alpha last.
    #[test]
    fn pixels_are_written_as_rgba_in_order() {
        let png = round_trip(&[0xff11_2233, 0x8044_5566], 2, 1);
        assert_eq!(&png[16..20], &2u32.to_be_bytes(), "width");
        assert_eq!(&png[20..24], &1u32.to_be_bytes(), "height");
    }

    /// Alpha is carried, not dropped: the rounded corners the harness carves are
    /// the whole reason this is RGBA, so two words differing only there must not
    /// encode the same.
    #[test]
    fn alpha_reaches_the_file() {
        assert_ne!(encode(&[0x0011_2233], 1, 1), encode(&[0xff11_2233], 1, 1));
    }

    /// A flat image is mostly repeats, which is what LZ77 is for. If this stops
    /// holding, the encoder stopped compressing.
    #[test]
    fn a_flat_image_compresses_far_below_its_raw_size() {
        let (w, h) = (256u32, 256u32);
        let png = round_trip(&vec![0xff20_3040; (w * h) as usize], w, h);
        let raw = (w * h * 4) as usize;
        assert!(
            png.len() < raw / 20,
            "expected heavy compression, got {} bytes for {raw} raw",
            png.len()
        );
    }

    /// Noise cannot compress much and must still come back byte for byte.
    #[test]
    fn noisy_data_round_trips_even_when_it_cannot_compress() {
        let pixels: Vec<u32> = (0..97u32 * 61)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        round_trip(&pixels, 97, 61);
    }

    /// A run longer than one match carries, so a copy reads from what the copy
    /// before it produced.
    #[test]
    fn a_run_longer_than_a_single_match_round_trips() {
        round_trip(&vec![0xff00_ff00; 4096], 64, 64);
    }

    /// A frame shaped like a terminal: a background with sparse glyph ink over it,
    /// wider than it is tall, and transparent corners.
    #[test]
    fn a_frame_shaped_like_a_terminal_round_trips() {
        let (w, h) = (320u32, 96u32);
        let mut pixels: Vec<u32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if y % 8 < 6 && x % 9 < 4 {
                    0xffd0_d0d0
                } else {
                    0xff10_1418
                }
            })
            .collect();
        pixels[0] = 0x0010_1418;
        round_trip(&pixels, w, h);
    }

    /// The two check values PNG relies on, against vectors from their specs.
    #[test]
    fn crc_and_adler_match_known_vectors() {
        let mut crc = Crc::new();
        crc.update(b"IEND");
        assert_eq!(crc.finish(), 0xae42_6082, "IEND's CRC is fixed by the spec");

        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn the_header_describes_the_image() {
        let png = encode(&[0xffff_0000; 6], 3, 2);
        assert_eq!(&png[..8], &SIGNATURE);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &3u32.to_be_bytes());
        assert_eq!(&png[20..24], &2u32.to_be_bytes());
        // 8-bit truecolor with alpha.
        assert_eq!(&png[24..26], &[8, 6]);
        assert!(png.ends_with(&[0xae, 0x42, 0x60, 0x82]), "IEND closes it");
    }
}
