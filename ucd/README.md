# Vendored Unicode Character Database

The UCD data files the terminal generates its property tables from, pinned to one
Unicode version and committed so a build (and a table regeneration) never touches the
network. Everything here is verbatim from the Unicode Consortium's public files,
redistributed under the Unicode Terms of Use their headers cite. The repository's
MIT license covers the rest of the tree.

**Version: 18.0.0**

| File | Feeds | Property used |
| --- | --- | --- |
| `EastAsianWidth.txt` | `src/width_tables.rs` | East Asian Width (W, F → 2 columns) |
| `DerivedGeneralCategory.txt` | `src/width_tables.rs` | General Category (Mn, Me, Cf → 0 columns) |
| `GraphemeBreakProperty.txt` | `src/platform/grapheme_tables.rs` | Grapheme_Cluster_Break |
| `emoji-data.txt` | `src/platform/grapheme_tables.rs` | Extended_Pictographic (rule GB11) |
| `DerivedCoreProperties.txt` | `src/platform/grapheme_tables.rs` | Indic_Conjunct_Break (rule GB9c) |
| `GraphemeBreakTest.txt` | `grapheme.rs` test | the official UAX #29 conformance suite |

## Refreshing on a Unicode bump

These came from the Arch `unicode-character-database` package (`/usr/share/unicode/`),
which tracks the latest Unicode. To move to a new version:

1. `make deps` (installs `unicode-character-database`), or fetch the files from
   `https://www.unicode.org/Public/<version>/ucd/`.
2. Copy the six files above into this directory (from `/usr/share/unicode/`, note the
   `extracted/`, `auxiliary/`, and `emoji/` subpaths).
3. `make tables` — regenerates the committed tables. It is deterministic and offline.
4. `cargo test` — the grapheme conformance test over `GraphemeBreakTest.txt` and the
   width spot-checks must pass; bump the pinned version in `width.rs`'s test.
