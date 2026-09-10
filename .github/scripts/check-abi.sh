#!/usr/bin/env sh
set -eu

# A binary takes the symbol versions of the machine that linked it, so building
# on a newer distribution quietly raises the glibc a download needs.
#
#   check-abi.sh FLOOR BINARY...

# A dotted version as one integer, so the shell can tell 2.9 from 2.34.
version_key() {
    rest="${1#*.}"
    case "$1" in
        *.*.*) patch="${1##*.}" ;;
        *) patch=0 ;;
    esac
    printf '%d%03d%03d' "${1%%.*}" "${rest%%.*}" "$patch"
}

# Every versioned glibc reference in a binary, as "GLIBC_x.y symbol". The wide
# form keeps readelf from eliding long names out of the list.
glibc_refs() {
    readelf -W --dyn-syms "$1" |
        sed -n 's/.*[[:space:]]\([A-Za-z_][A-Za-z0-9_]*\)@@*\(GLIBC_[0-9][0-9.]*\).*/\2 \1/p' |
        sort -u
}

# The references above the floor, and a non-zero status when there are any.
above_floor() {
    over="$(glibc_refs "$2" | while read -r version symbol; do
        if [ "$(version_key "${version#GLIBC_}")" -gt "$1" ]; then
            printf '  %s %s\n' "$version" "$symbol"
        fi
    done)"
    [ -z "$over" ] || { printf '%s\n' "$over"; return 1; }
}

main() {
    if [ "$#" -lt 2 ]; then
        echo "usage: check-abi.sh FLOOR BINARY..." >&2
        exit 2
    fi

    if ! command -v readelf > /dev/null 2>&1; then
        echo "Error: checking the ABI floor needs readelf, from binutils" >&2
        exit 2
    fi

    floor="$1"
    shift
    floor_key="$(version_key "$floor")"

    status=0
    for binary in "$@"; do
        if [ ! -f "$binary" ]; then
            echo "Error: no such binary: $binary" >&2
            exit 2
        fi
        if over="$(above_floor "$floor_key" "$binary")"; then
            echo "ok   $binary runs on glibc $floor"
        else
            echo "FAIL $binary needs a glibc newer than $floor"
            printf '%s\n' "$over"
            status=1
        fi
    done
    return "$status"
}

# Sourced by the test, run otherwise.
if [ "${BNKTERM_ABI_LIB:-}" != "1" ]; then
    main "$@"
fi
