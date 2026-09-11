#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
BNKTERM_INSTALL_LIB=1
export BNKTERM_INSTALL_LIB
. "$ROOT/install.sh"

TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bnkterm-install-test-XXXXXX")"
trap 'rm -rf "$TEST_DIR"' EXIT

fail() {
    echo "install test: $1" >&2
    exit 1
}

expect_checksum_failure() {
    if (verify_checksum "$TEST_DIR/archive" "unused") > /dev/null 2>&1; then
        fail "$1"
    fi
}

printf 'archive payload\n' > "$TEST_DIR/archive"
checksum="$(sha256sum "$TEST_DIR/archive" | cut -d' ' -f1)"

fetch() {
    printf '%s  archive\n' "$checksum" > "$2"
}
verify_checksum "$TEST_DIR/archive" "unused" > /dev/null

fetch() {
    return 1
}
expect_checksum_failure "a missing checksum was accepted"

fetch() {
    printf 'not-a-digest\n' > "$2"
}
expect_checksum_failure "a malformed checksum was accepted"

fetch() {
    printf '%064d  archive\n' 0 > "$2"
}
expect_checksum_failure "a checksum mismatch was accepted"

fetch() {
    return 0
}
if (PATH=/nonexistent verify_checksum "$TEST_DIR/archive" "unused") > /dev/null 2>&1; then
    fail "verification without a digest tool was accepted"
fi

stage="$TEST_DIR/stage"
mkdir -p "$stage/icons/hicolor/scalable/apps" "$TEST_DIR/fake-bin"
printf '#!/usr/bin/env sh\nexit 0\n' > "$stage/bnkterm"
chmod +x "$stage/bnkterm"
cp "$ROOT/assets/bnkterm.desktop" "$stage/bnkterm.desktop"
cp "$ROOT/assets/bnkterm.svg" "$stage/icons/hicolor/scalable/apps/bnkterm.svg"
archive="$TEST_DIR/bnkterm-v9.9.9-x86_64-unknown-linux-gnu.tar.gz"
tar czf "$archive" -C "$stage" .
printf '%s  %s\n' "$(sha256sum "$archive" | cut -d' ' -f1)" "$(basename "$archive")" \
    > "$archive.sha256"

cat > "$TEST_DIR/fake-bin/curl" <<'EOF'
#!/usr/bin/env sh
set -eu
url=""
output=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            shift
            output="$1" ;;
        -*) ;;
        *) url="$1" ;;
    esac
    shift
done
case "$url" in
    *.sha256) cp "$BNKTERM_TEST_SUM" "$output" ;;
    *) cp "$BNKTERM_TEST_ARCHIVE" "$output" ;;
esac
EOF
chmod +x "$TEST_DIR/fake-bin/curl"

home="$TEST_DIR/Home space%quote\""
data_home="$TEST_DIR/data"
BNKTERM_TEST_ARCHIVE="$archive" \
BNKTERM_TEST_SUM="$archive.sha256" \
BNKTERM_INSTALL_LIB=0 \
BNKTERM_VERSION=v9.9.9 \
HOME="$home" \
XDG_DATA_HOME="$data_home" \
PATH="$TEST_DIR/fake-bin:/usr/bin:/bin" \
    sh "$ROOT/install.sh" > "$TEST_DIR/install.log"

[ -x "$home/.local/bin/bnkterm" ] || fail "the binary was not installed"
expected="Exec=\"$TEST_DIR/Home space%%quote\\\"/.local/bin/bnkterm\""
desktop="$data_home/applications/bnkterm.desktop"
grep -Fqx -- "$expected" "$desktop" \
    || fail "the installed desktop entry does not contain the escaped absolute path"

if command -v desktop-file-validate > /dev/null 2>&1; then
    desktop-file-validate "$desktop"
fi

echo "install tests passed"
