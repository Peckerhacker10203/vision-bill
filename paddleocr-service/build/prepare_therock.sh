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

# The pinned topology does not consume the root documentation repository.
# Initialize every other recursive gitlink before upstream patches its checkouts.
git -C "$SOURCE" submodule update --init --recursive --filter=blob:none --checkout --jobs "$JOBS" \
  -- . ':(top,exclude,literal)rocm-docs'
(
  cd "$SOURCE"
  python build_tools/fetch_sources.py --no-remote --apply-patches
)
python "$RECIPE/source_inventory.py" "$SOURCE" "$INVENTORY" --omit-root-gitlink rocm-docs
