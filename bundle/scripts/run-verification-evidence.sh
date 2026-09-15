#!/usr/bin/env bash

# Check the compact H1--H8 numerical record in the installed bundle.
# This is read-only: it never creates or overwrites host results.

set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image=${VICMF6_IMAGE:-vic-mf6:local}

if [ "$#" -ne 0 ]; then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

exec docker run --rm --init "$image" verification-evidence
