#!/usr/bin/env sh
set -eu

# Downloads a release from GitHub and installs it into ~/.local/bin, along with
# the desktop entry and icon so it shows up in your app menu.
# Pin a specific version with BNKTERM_VERSION=v0.1.0.

REPO="borgenk/bnkterm"
APP_NAME="bnkterm"

select_fetch() {
    if command -v curl > /dev/null 2>&1; then
        fetch() { curl -fsSL "$1" -o "$2"; }
        fetch_stdout() { curl -fsSL "$1"; }
    elif command -v wget > /dev/null 2>&1; then
        fetch() { wget -q "$1" -O "$2"; }
        fetch_stdout() { wget -q "$1" -O -; }
    else
        echo "Error: curl or wget is required" >&2
        return 1
    fi
}

# The file's sha256, or non-zero when the machine has nothing to compute it with.
sha256_of() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum > /dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    elif command -v openssl > /dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        return 1
    fi
}

# Verify the archive before anything is extracted or installed.
verify_checksum() {
    file="$1"
    sums_url="$2"

    if ! fetch "$sums_url" "$file.sha256"; then
        echo "Error: could not download the release checksum" >&2
        return 1
    fi

    if ! actual="$(sha256_of "$file")"; then
        echo "Error: sha256sum, shasum, or openssl is required" >&2
        return 1
    fi

    expected="$(cut -d' ' -f1 < "$file.sha256")"
    if [ "${#expected}" -ne 64 ]; then
        echo "Error: release checksum is malformed" >&2
        return 1
    fi
    case "$expected" in
        *[!0-9A-Fa-f]*)
            echo "Error: release checksum is malformed" >&2
            return 1 ;;
    esac
    expected="$(printf '%s' "$expected" | tr 'A-F' 'a-f')"
    actual="$(printf '%s' "$actual" | tr 'A-F' 'a-f')"

    if [ "$expected" != "$actual" ]; then
        echo "Error: checksum mismatch, refusing to install" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        return 1
    fi

    echo "Checksum verified"
}

desktop_exec_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/`/\\`/g; s/\$/\\$/g; s/%/%%/g'
}

install_desktop_entry() {
    source_file="$1"
    target_file="$2"
    executable="$3"
    escaped="$(desktop_exec_escape "$executable")"
    temporary="${target_file}.tmp.$$"
    found=0

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            Exec=*)
                printf 'Exec="%s"\n' "$escaped"
                found=1 ;;
            *) printf '%s\n' "$line" ;;
        esac
    done < "$source_file" > "$temporary"

    if [ "$found" -ne 1 ]; then
        rm -f "$temporary"
        return 1
    fi
    chmod 644 "$temporary"
    mv "$temporary" "$target_file"
}

main() {
    platform="$(uname -s)"
    arch="$(uname -m)"

    case "$platform" in
        Linux)
            case "$arch" in
                x86_64 | x86-64 | x64 | amd64)
                    TARGET="x86_64-unknown-linux-gnu" ;;
                *)
                    echo "Error: unsupported architecture: $arch"
                    exit 1 ;;
            esac
            ;;
        *)
            echo "Error: unsupported OS: $platform (bnkterm is Linux-only)"
            exit 1
            ;;
    esac

    select_fetch || exit 1

    if [ -n "${BNKTERM_VERSION:-}" ]; then
        VERSION="$BNKTERM_VERSION"
    else
        echo "Resolving latest version..."
        VERSION="$(fetch_stdout "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' \
            | head -n1 \
            | cut -d'"' -f4)"
        if [ -z "$VERSION" ]; then
            echo "Error: failed to resolve latest version"
            exit 1
        fi
    fi

    FILENAME="${APP_NAME}-${VERSION}-${TARGET}.tar.gz"
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${FILENAME}"

    if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ]; then
        TMP_DIR="$(mktemp -d "$TMPDIR/bnkterm-XXXXXX")"
    else
        TMP_DIR="$(mktemp -d "/tmp/bnkterm-XXXXXX")"
    fi
    trap 'rm -rf "$TMP_DIR"' EXIT

    echo "Downloading bnkterm ${VERSION} for ${TARGET}..."
    if ! fetch "$URL" "$TMP_DIR/$FILENAME"; then
        echo "Error: could not download ${VERSION}"
        echo "  $URL"
        exit 1
    fi

    verify_checksum "$TMP_DIR/$FILENAME" "${URL}.sha256"

    echo "Extracting..."
    tar -xzf "$TMP_DIR/$FILENAME" -C "$TMP_DIR"

    if [ ! -f "$TMP_DIR/$APP_NAME" ]; then
        echo "Error: the archive holds no bnkterm binary"
        exit 1
    fi

    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"
    mv "$TMP_DIR/$APP_NAME" "$INSTALL_DIR/$APP_NAME"
    chmod +x "$INSTALL_DIR/$APP_NAME"

    echo "Installed bnkterm ${VERSION} to ${INSTALL_DIR}/${APP_NAME}"

    # Report linked libraries that the loader cannot find.
    if command -v ldd > /dev/null 2>&1; then
        missing="$(ldd "$INSTALL_DIR/$APP_NAME" 2>/dev/null | grep 'not found' || true)"
        if [ -n "$missing" ]; then
            echo ""
            echo "Warning: bnkterm will not start, these libraries are missing:"
            echo "$missing" | awk '{print "  " $1}'
            echo ""
            echo "They come from freetype, harfbuzz, fontconfig and libxkbcommon;"
            echo "install your distribution's runtime packages for those."
        fi
    fi

    # ldd cannot see libvulkan because bnkterm loads it at runtime.
    if command -v ldconfig > /dev/null 2>&1; then
        if ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1'; then
            echo ""
            echo "Warning: libvulkan.so.1 is not on the library path, bnkterm needs it."
            echo "Install the Vulkan loader (vulkan-icd-loader on Arch, libvulkan1 on"
            echo "Debian and Ubuntu, vulkan-loader on Fedora) and your GPU's driver."
        fi
    fi

    # Missing Nerd Font symbols only disables the password-lock cursor.
    if command -v fc-list > /dev/null 2>&1; then
        if ! fc-list 2>/dev/null | grep -qi 'symbols nerd font'; then
            echo ""
            echo "Note: no Symbols Nerd Font found, so the password-lock cursor is off."
            echo "Install ttf-nerd-fonts-symbols (Arch) or the Nerd Fonts symbols cut"
            echo "to turn it on."
        fi
    fi

    # Desktop integration is optional.
    DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
    APPS_DIR="$DATA_HOME/applications"
    ICONS_DIR="$DATA_HOME/icons/hicolor"
    if [ -f "$TMP_DIR/${APP_NAME}.desktop" ]; then
        if mkdir -p "$APPS_DIR" \
            && install_desktop_entry \
                "$TMP_DIR/${APP_NAME}.desktop" \
                "$APPS_DIR/${APP_NAME}.desktop" \
                "$INSTALL_DIR/$APP_NAME"; then
            update-desktop-database "$APPS_DIR" > /dev/null 2>&1 || true
        else
            echo "Note: could not install the desktop entry into $APPS_DIR"
        fi
    fi
    if [ -d "$TMP_DIR/icons/hicolor" ]; then
        if mkdir -p "$ICONS_DIR" && cp -r "$TMP_DIR/icons/hicolor/." "$ICONS_DIR/"; then
            gtk-update-icon-cache -f -t "$ICONS_DIR" > /dev/null 2>&1 || true
        else
            echo "Note: could not install the icon into $ICONS_DIR"
        fi
    fi

    if [ "$(command -v "$APP_NAME")" = "$INSTALL_DIR/$APP_NAME" ]; then
        echo "Run with 'bnkterm'"
    else
        echo ""
        echo "To run bnkterm from your terminal, add ~/.local/bin to your PATH:"

        case "${SHELL:-}" in
            *zsh)
                echo "  echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.zshrc"
                echo "  source ~/.zshrc"
                ;;
            *fish)
                echo "  fish_add_path -U \$HOME/.local/bin"
                ;;
            *)
                echo "  echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc"
                echo "  source ~/.bashrc"
                ;;
        esac
    fi
}

# Sourced by the test, run otherwise.
if [ "${BNKTERM_INSTALL_LIB:-}" != "1" ]; then
    main "$@"
fi
