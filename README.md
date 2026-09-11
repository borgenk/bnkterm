# BNK Term

[![CI](https://github.com/borgenk/bnkterm/actions/workflows/ci.yml/badge.svg)](https://github.com/borgenk/bnkterm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/borgenk/bnkterm)](https://github.com/borgenk/bnkterm/releases/latest)

A native Wayland + Vulkan terminal emulator written in Rust, with no crate dependencies.

_Disclaimer: learning project, non-standard Rust (nightly, unsafe FFI),
built mainly for my own use, AI-assisted._

## Requirements

- A Linux desktop running a **Wayland** session
- A compositor that supports **xdg_wm_base** and **zwp_linux_dmabuf_v1** version 4
- A working **Vulkan** device
- The freetype, harfbuzz, fontconfig and libxkbcommon libraries

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/borgenk/bnkterm/main/install.sh | sh
```

This drops the binary in `~/.local/bin` and installs the desktop entry and icon.

### Flatpak

Download the bundle from the
[releases page](https://github.com/borgenk/bnkterm/releases/latest):

```sh
flatpak install --user ./bnkterm-*.flatpak
```
