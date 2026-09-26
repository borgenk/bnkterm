//! Seeded generators for the measurement streams. Each produces roughly a requested
//! number of bytes, shaped to exercise one ingestion path: plain ASCII throughput,
//! wide and multibyte UTF-8, emoji clusters, and the escape-dense traffic a redrawing
//! full-screen program emits. A fixed-seed SplitMix64 and no clock, so every run
//! produces the same bytes and a difference between runs is a difference in the code.
//!
//! Generators return bytes rather than `String`: the escape-heavy stream carries
//! control bytes that are not text. `ascii` and `unicode` hold no combining marks or
//! ZWJ, so they stress the print path at zero per-cell allocation; `emoji` and
//! `combining` drive the grid's combining side table, the one text path that still
//! allocates per marked cell.

/// A stream generator: builds a stream of roughly the requested byte size.
type Generator = fn(usize) -> Vec<u8>;

/// The streams the harness knows, paired with their generators. The single
/// registry [`NAMES`] and [`generate`] both derive from, so a name and its
/// dispatch can never drift apart.
const CORPORA: &[(&str, Generator)] = &[
    ("ascii", ascii),
    ("unicode", unicode),
    ("unicode_width1", unicode_width1),
    ("unicode_cjk", unicode_cjk),
    ("unicode_mixed", unicode_mixed),
    ("emoji", emoji),
    ("combining", combining),
    ("escape_heavy", escape_heavy),
    ("short_runs", short_runs),
    ("wrap_edges", wrap_edges),
    ("invalid_utf8", invalid_utf8),
];

/// The stream names, in run order. Derived from [`CORPORA`].
pub const NAMES: &[&str] = &stream_names();

const fn stream_names() -> [&'static str; CORPORA.len()] {
    let mut names = [""; CORPORA.len()];
    let mut i = 0;
    while i < CORPORA.len() {
        names[i] = CORPORA[i].0;
        i += 1;
    }
    names
}

/// Generate the named stream at approximately `target_bytes`, or `None` for an
/// unknown name. May overshoot by one unit (a line, a row); never undershoots.
pub fn generate(name: &str, target_bytes: usize) -> Option<Vec<u8>> {
    CORPORA
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, generator)| generator(target_bytes))
}

/// A minimal SplitMix64: enough spread to shape realistic streams, fully
/// deterministic from its seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Push `n` as decimal digits, no allocation. Used to build escape parameters.
fn push_int(out: &mut Vec<u8>, mut n: usize) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    let start = out.len();
    while n > 0 {
        out.push(b'0' + (n % 10) as u8);
        n /= 10;
    }
    out[start..].reverse();
}

const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "eiusmod",
    "tempor",
    "incididunt",
];

/// Ordinary ASCII prose in newline-terminated lines: the plain-throughput
/// baseline, the bulk of what a terminal ever prints.
fn ascii(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(1);
    let mut out = Vec::with_capacity(target + 64);
    while out.len() < target {
        let words = 8 + rng.below(10);
        for w in 0..words {
            if w > 0 {
                out.push(b' ');
            }
            out.extend_from_slice(WORDS[rng.below(WORDS.len())].as_bytes());
        }
        out.push(b'\n');
    }
    out
}

/// Combining-mark-free, ZWJ-free lines of mixed scripts, including wide (CJK,
/// kana, Hangul) runs: the UTF-8 decode path and the wide-cell placement, with
/// no per-cell allocation so it doubles as a zero-allocation steady-state check.
const SCRIPTS: &[&str] = &[
    "Съешь ещё этих мягких французских булок да выпей чаю",
    "Ελληνικά γράμματα και δοκιμή κειμένου ροής",
    "いろはにほへとちりぬるをわかよたれそ日本語文字",
    "동해물과 백두산이 마르고 닳도록 하느님이 보우하사",
    "中文字符测试文本包含常用的汉字组合与标点",
];

fn unicode(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(4);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        out.extend_from_slice(SCRIPTS[rng.below(SCRIPTS.len())].as_bytes());
        out.push(b'\n');
    }
    out
}

/// Non-ASCII prose whose scalars are all one cell wide. This separates UTF-8
/// decoding and Unicode property lookup from the grid's wide-pair bookkeeping.
const WIDTH1_PROSE: &[&str] = &[
    "På vår øy står blåbær ved siden av røde epler",
    "Zażółć gęślą jaźń, bientôt déjà vu après l'été",
    "Съешь ещё этих мягких французских булок",
    "Ελληνικά γράμματα σε μία γραμμή κειμένου",
];

fn unicode_width1(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(11);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        let line = WIDTH1_PROSE
            .get(rng.below(WIDTH1_PROSE.len()))
            .copied()
            .unwrap_or("På vår øy står blåbær");
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out
}

/// CJK, kana, and Hangul prose: nearly every decoded scalar needs a two-cell
/// leader/spacer pair, isolating the wide-cell half of a future batch writer.
const CJK_PROSE: &[&str] = &[
    "日本語文字列と端末表示の性能測定",
    "中文字符测试包含常用汉字与标点",
    "동해물과 백두산이 마르고 닳도록",
    "かなカナ漢字を含む文章の描画試験",
];

fn unicode_cjk(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(12);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        let line = CJK_PROSE
            .get(rng.below(CJK_PROSE.len()))
            .copied()
            .unwrap_or("日本語文字列");
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out
}

/// The shape source listings, compiler diagnostics, and shell prompts usually
/// have: long ASCII identifiers and punctuation interrupted by short Unicode
/// words. It prevents a fast path from winning only on uninterrupted non-ASCII.
const MIXED_LINES: &[&str] = &[
    "src/parser.rs:184: melding = \"ugyldig tegnfølge\"; status=avvist",
    "warning[E0308]: ожидается `Result<T>`, получено `Option<T>`",
    "render[row=17]: 日本語 text, width=2, fallback=false",
    "test unicode_width: Ελληνικά + ASCII_123 + 中文 passed",
];

fn unicode_mixed(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(13);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        let line = MIXED_LINES
            .get(rng.below(MIXED_LINES.len()))
            .copied()
            .unwrap_or("render: 日本語 text");
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out
}

/// Emoji-dense lines, including a ZWJ family and a flag: the wide-cluster path,
/// and (via the zero-width joiner/variation-selector) the combining side table.
/// Allocates per marked cell, unlike the fixed-shape gate scenarios.
const EMOJI: &[&str] = &["😀", "🚀", "🎉", "🌍", "🔥", "✨", "👍", "🧪", "👨‍👩‍👧‍👦", "🏳️‍🌈"];

fn emoji(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(5);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        let count = 8 + rng.below(16);
        for _ in 0..count {
            out.extend_from_slice(EMOJI[rng.below(EMOJI.len())].as_bytes());
        }
        out.push(b'\n');
    }
    out
}

/// Latin letters each carrying one to three combining marks (accents, diacritics):
/// the base-plus-mark shape a shell prints for accented text. Every marked cell touches
/// the grid's combining side table, so like `emoji` this allocates as it varies; the
/// gate's fixed-shape `grid_combining` scenario is what holds the warmed path at zero.
const MARKS: &[char] = &[
    '\u{0300}', // grave
    '\u{0301}', // acute
    '\u{0302}', // circumflex
    '\u{0303}', // tilde
    '\u{0304}', // macron
    '\u{0308}', // diaeresis
    '\u{0323}', // dot below
    '\u{0327}', // cedilla
];

fn combining(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(6);
    let mut out = Vec::with_capacity(target + 256);
    let mut buf = [0u8; 4];
    while out.len() < target {
        let letters = 8 + rng.below(16);
        for _ in 0..letters {
            out.push(b'a' + rng.below(26) as u8);
            for _ in 0..1 + rng.below(3) {
                let mark = MARKS[rng.below(MARKS.len())];
                out.extend_from_slice(mark.encode_utf8(&mut buf).as_bytes());
            }
        }
        out.push(b'\n');
    }
    out
}

/// Escape-dense output modelling `ls --color` and an htop-style redraw: short
/// colored runs (256-color SGR), resets, and periodic absolute cursor moves with
/// line erases. This is the traffic a terminal's parser exists to chew through
/// fast, and the case a naive per-byte parser chokes on.
fn escape_heavy(target: usize) -> Vec<u8> {
    let mut rng = Rng::new(10);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        let cells = 4 + rng.below(8);
        for _ in 0..cells {
            out.extend_from_slice(b"\x1b[38;5;");
            push_int(&mut out, rng.below(256));
            out.push(b'm');
            out.extend_from_slice(WORDS[rng.below(WORDS.len())].as_bytes());
            out.extend_from_slice(b"\x1b[0m ");
        }
        out.extend_from_slice(b"\r\n");
        // Every so often, an absolute reposition + line erase, as a live TUI does
        // when it repaints a changed row in place.
        if rng.below(4) == 0 {
            out.extend_from_slice(b"\x1b[");
            push_int(&mut out, 1 + rng.below(24));
            out.push(b';');
            push_int(&mut out, 1 + rng.below(80));
            out.push(b'H');
            out.extend_from_slice(b"\x1b[K");
        }
    }
    out
}

/// Short printable runs separated by SGR changes and cursor controls. Unlike
/// `escape_heavy`, the payload is mixed UTF-8, so this measures whether setup and
/// flush overhead erase the value of batching the small runs a coloured TUI emits.
fn short_runs(target: usize) -> Vec<u8> {
    const RUNS: &[&str] = &["ok", "feil", "日本", "λ", "данные", "42"];
    let mut rng = Rng::new(14);
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        for _ in 0..8 {
            out.extend_from_slice(b"\x1b[38;5;");
            push_int(&mut out, rng.below(256));
            out.push(b'm');
            let run = RUNS.get(rng.below(RUNS.len())).copied().unwrap_or("ok");
            out.extend_from_slice(run.as_bytes());
            out.extend_from_slice(b"\x1b[0m ");
        }
        out.extend_from_slice(b"\r\n\x1b[K");
    }
    out
}

/// Rows ending immediately before, on, and after a 120-column boundary, with a
/// two-cell glyph at each interesting edge. This keeps the wrap and wide-glyph
/// fallbacks visible instead of averaging them away in long prose.
fn wrap_edges(target: usize) -> Vec<u8> {
    const PREFIXES: &[usize] = &[117, 118, 119, 120, 121];
    let mut out = Vec::with_capacity(target + 256);
    let mut i = 0usize;
    while out.len() < target {
        let prefix = PREFIXES.get(i % PREFIXES.len()).copied().unwrap_or(119);
        out.resize(out.len() + prefix, b'x');
        out.extend_from_slice("界y\r\n".as_bytes());
        i = i.wrapping_add(1);
    }
    out
}

/// Valid text interspersed with each important malformed UTF-8 shape. The
/// parser must preserve Unicode maximal-subpart replacement semantics while a
/// future run decoder crosses and flushes its bounded buffers.
fn invalid_utf8(target: usize) -> Vec<u8> {
    const UNITS: &[&[u8]] = &[
        b"valid ",
        &[0xc0, 0xaf], // two invalid lead/continuation bytes
        b" tail\n",
        &[0xe0, 0x80, 0x80], // overlong three-byte form
        b" text ",
        &[0xed, 0xa0, 0x80], // surrogate
        b"\n",
        &[0xf4, 0x90, 0x80, 0x80], // above U+10FFFF
        b" done\n",
    ];
    let mut out = Vec::with_capacity(target + 256);
    while out.len() < target {
        for unit in UNITS {
            out.extend_from_slice(unit);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_stream_meets_its_target() {
        for name in NAMES {
            let doc = generate(name, 32 * 1024).unwrap_or_else(|| panic!("{name} missing"));
            assert!(
                doc.len() >= 32 * 1024,
                "{name} undershot: {} bytes",
                doc.len()
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        for name in NAMES {
            let a = generate(name, 16 * 1024).unwrap();
            let b = generate(name, 16 * 1024).unwrap();
            assert_eq!(a, b, "{name} is not deterministic");
        }
    }

    #[test]
    fn unknown_stream_is_none() {
        assert!(generate("does_not_exist", 1024).is_none());
    }

    #[test]
    fn text_streams_are_valid_utf8() {
        // The text streams are decodable text (the parser's UTF-8 path should see
        // no replacement chars from them).
        for name in ["ascii", "unicode", "emoji", "combining"] {
            let doc = generate(name, 8 * 1024).unwrap();
            assert!(
                std::str::from_utf8(&doc).is_ok(),
                "{name} is not valid UTF-8"
            );
        }
    }

    #[test]
    fn escape_stream_carries_escapes() {
        let doc = generate("escape_heavy", 8 * 1024).unwrap();
        assert!(doc.contains(&0x1b), "escape_heavy has no ESC bytes");
    }

    #[test]
    fn push_int_roundtrips() {
        for n in [0usize, 1, 9, 10, 255, 1234, 65535] {
            let mut v = Vec::new();
            push_int(&mut v, n);
            assert_eq!(std::str::from_utf8(&v).unwrap(), n.to_string());
        }
    }
}
