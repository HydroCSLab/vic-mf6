#!/usr/bin/env bash

# verify that each source checkout matches the revision recorded by the bundle.

set -euo pipefail

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
project_dir=$(CDPATH= cd -- "$bundle_dir/.." && pwd)
repo_dir="$bundle_dir"
lock_file="$bundle_dir/components.lock"

while IFS='|' read -r component expected_revision component_path repository; do
    case "$component" in
        ''|'#'*) continue ;;
    esac

    checkout="$repo_dir/$component_path"
    if [ ! -e "$checkout/.git" ]; then
        printf '%s\n' "component is not initialized: $component_path" >&2
        printf '%s\n' "run: git submodule update --init --recursive" >&2
        exit 2
    fi

    actual_revision=$(git -C "$checkout" rev-parse HEAD)
    if [ "$actual_revision" != "$expected_revision" ]; then
        printf '%s\n' \
            "$component revision mismatch: expected=$expected_revision actual=$actual_revision" >&2
        exit 2
    fi

    configured_repository=$(
        git -C "$project_dir" config --file .gitmodules \
            --get "submodule.bundle/$component_path.url"
    )
    if [ "$configured_repository" != "$repository" ]; then
        printf '%s\n' \
            "$component repository mismatch: expected=$repository actual=$configured_repository" >&2
        exit 2
    fi
done < "$lock_file"

(
    cd "$bundle_dir/examples/stehekin/input"
    sha256sum --check checksums.sha256
)

printf '%s\n' "[OK] component revisions match components.lock"
