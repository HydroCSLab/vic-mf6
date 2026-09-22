#!/usr/bin/env bash

# build the complete scientific runtime from the recorded component revisions.

set -euo pipefail

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
project_dir=$(CDPATH= cd -- "$bundle_dir/.." && pwd)
image=${1:-${VICMF6_IMAGE:-vic-mf6:local}}
source_repository=${VICMF6_SOURCE_REPOSITORY:-https://github.com/mabdazzam/vic-mf6}
version=${VICMF6_VERSION:-local}
revision=$(git -C "$project_dir" rev-parse HEAD)

if [ "$#" -gt 1 ]; then
    printf '%s\n' "usage: $0 [IMAGE]" >&2
    exit 2
fi

"$bundle_dir/scripts/check-components.sh"

docker build \
    --network "${VICMF6_BUILD_NETWORK:-default}" \
    --build-arg "USER_ID=$(id -u)" \
    --build-arg "GROUP_ID=$(id -g)" \
    --build-arg "SOURCE_REPOSITORY=$source_repository" \
    --build-arg "VERSION=$version" \
    --build-arg "REVISION=$revision" \
    --tag "$image" \
    --file "$bundle_dir/Dockerfile" \
    "$project_dir"

printf '%s\n' "[OK] built $image"
