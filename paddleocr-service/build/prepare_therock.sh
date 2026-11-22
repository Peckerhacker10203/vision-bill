#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?Pass the initialized TheRock repository with origin configured}"
INVENTORY="${2:?Pass the source inventory output path}"
JOBS="${3:?Pass the build job count}"
PIN="${4:?Pass the pinned TheRock commit}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Fetch all files in the pinned root tree together so checkout needs no lazy blob requests.
git -C "$SOURCE" fetch --no-tags --depth=1 origin "$PIN"
git -C "$SOURCE" checkout --detach "$PIN"
git -C "$SOURCE" ls-files --stage > "$(dirname -- "$INVENTORY")/therock-original-gitlinks.txt"

# TheRock's fetch only initializes selected top-level sources. Populate all
# recorded pins first so an update cannot reset an upstream-patched checkout.
git -C "$SOURCE" submodule update --init --recursive --filter=blob:none --checkout --jobs "$JOBS"
(
  cd "$SOURCE"
  python build_tools/fetch_sources.py --no-remote --apply-patches
)
python "$RECIPE/source_inventory.py" "$SOURCE" "$INVENTORY"
