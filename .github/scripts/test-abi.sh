#!/usr/bin/env sh
set -eu

# The ABI floor check, run against binaries this machine already has. The two
# quiet failures it guards: glibc versions compare as numbers, not as text, and
# readelf elides long symbol names unless asked for the wide form.

cd "$(dirname "$0")/../.."
CHECK=.github/scripts/check-abi.sh

BNKTERM_ABI_LIB=1 . "./$CHECK"
# Cleared again, so the runs below start the script rather than source it.
unset BNKTERM_ABI_LIB

failures=0
check() {
    if [ "$2" = "$3" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1"
        echo "     want: $2"
        echo "     got:  $3"
        failures=$((failures + 1))
    fi
}

lower() {
    if [ "$(version_key "$1")" -lt "$(version_key "$2")" ]; then echo yes; else echo no; fi
}

check "2.9 sorts below 2.34" "yes" "$(lower 2.9 2.34)"
check "2.2.5 sorts below 2.3" "yes" "$(lower 2.2.5 2.3)"
check "2.34 does not sort below itself" "no" "$(lower 2.34 2.34)"
check "2.43 does not sort below 2.36" "no" "$(lower 2.43 2.36)"

# A binary every machine that can run this has, and one that uses glibc.
SUBJECT="$(command -v sh)"
refs="$(sh "$CHECK" 2.0 "$SUBJECT" | grep -c '  GLIBC_' || true)"
check "the subject binary references glibc" "yes" \
    "$([ "$refs" -gt 0 ] && echo yes || echo no)"

status() {
    if "$@" > /dev/null 2>&1; then echo 0; else echo $?; fi
}

check "a floor below the binary fails" "1" "$(status sh "$CHECK" 2.0 "$SUBJECT")"
check "a floor above the binary passes" "0" "$(status sh "$CHECK" 99.0 "$SUBJECT")"
check "no binary is a usage error" "2" "$(status sh "$CHECK" 2.36)"
check "a missing binary is an error" "2" "$(status sh "$CHECK" 2.36 /no/such/binary)"

# readelf and objdump have to account for the same references: an elided name
# never reaches the comparison.
check "every reference is counted" \
    "$(objdump -T "$SUBJECT" | grep -c '(GLIBC_' || true)" \
    "$refs"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "abi checks passed"
