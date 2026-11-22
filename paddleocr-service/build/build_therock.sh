#!/usr/bin/env bash
# Unexecuted native qualification candidate; this script never declares GPU support.
set -euo pipefail
BUILD_JOBS="${BUILD_JOBS:-8}"
SOURCE_ROOT="${SOURCE_ROOT:-/opt/native-src}"
SDK="${SDK:-/opt/rocm}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE=/opt/ocr/build-evidence
THEROCK_SRC="$SOURCE_ROOT/TheRock"
THEROCK_BUILD="$SOURCE_ROOT/therock-build"
WIN_HEADERS="$SOURCE_ROOT/win32metadata"
# Fresh sources/caches only: a previous native build is not a reproducible input.
for path in "$THEROCK_SRC" "$THEROCK_BUILD" "$WIN_HEADERS"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "Native build path already exists: $path" >&2
    exit 1
  fi
done
mkdir -p "$SOURCE_ROOT" "$EVIDENCE"
TARGETS="$(python "$RECIPE/../ocr_service/targets.py" --targets "${AMDGPU_TARGETS:-}" --output "$EVIDENCE/targets.json")"
export VISION_BILL_AMDGPU_TARGETS="$TARGETS"

git init --quiet "$THEROCK_SRC"
git -C "$THEROCK_SRC" remote add origin https://github.com/ROCm/TheRock.git
bash "$RECIPE/prepare_therock.sh" "$THEROCK_SRC" "$EVIDENCE/therock-sources.json" "$BUILD_JOBS" \
  b324869fb3116353ac7eec5e6926aa3d452ac987
# The WSL thunk needs Windows SDK headers at compile time and the AMD WKMI
# archive referenced by the rocm-systems gitlink. Neither is a host driver.
git init -q "$WIN_HEADERS"
git -C "$WIN_HEADERS" remote add origin https://github.com/microsoft/win32metadata.git
git -C "$WIN_HEADERS" sparse-checkout set generation/WinSDK/RecompiledIdlHeaders/shared
git -C "$WIN_HEADERS" fetch --depth=1 --filter=blob:none origin a40ed3186943620f304649a077b78f8271034736
git -C "$WIN_HEADERS" checkout --detach FETCH_HEAD
git -C "$WIN_HEADERS" rev-parse HEAD > "$EVIDENCE/windows-sdk-headers.commit"
WIN_SDK="$WIN_HEADERS/generation/WinSDK/RecompiledIdlHeaders/shared"
test -s "$WIN_SDK/d3dkmthk.h"
WKMI="$THEROCK_SRC/rocm-systems/shared/amdgpu-windows-interop/wkmi/lnx/lib/libwkmi.a"
grep -Fxq -- '- md5: 53126f6263d5ca1381b84e8a637e6a1e' "$WKMI.dvc"
curl --fail --location --retry 3 \
  https://therock-dvc.s3.us-east-2.amazonaws.com/rocm-systems/files/md5/53/126f6263d5ca1381b84e8a637e6a1e \
  --output "$WKMI"
test "$(stat -c %s "$WKMI")" = 333088
echo "53126f6263d5ca1381b84e8a637e6a1e  $WKMI" | md5sum -c -
sha256sum "$WKMI" > "$EVIDENCE/wkmi.sha256"
(
  cd "$THEROCK_SRC"
  INSTALL_PREFIX=/usr/local bash dockerfiles/install_pinned_patchelf.sh
)
# All SDK components retained, including host APIs for upstream-excluded Lt targets.
cmake -S "$THEROCK_SRC" -B "$THEROCK_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DTHEROCK_ENABLE_ALL=ON \
  -DTHEROCK_ENABLE_WSL=ON -DTHEROCK_ENABLE_WSL_ROCDXG=ON "-DTHEROCK_WSL_WIN_SDK=$WIN_SDK" \
  -DTHEROCK_AMDGPU_FAMILIES= "-DTHEROCK_AMDGPU_TARGETS=$TARGETS" \
  -DTHEROCK_DIST_AMDGPU_FAMILIES= "-DTHEROCK_DIST_AMDGPU_TARGETS=$TARGETS" \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=vision-bill-candidate-v1 -DBUILD_TESTING=OFF
# TheRock launches separate cmake --build processes for its subprojects.
CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" cmake --build "$THEROCK_BUILD" --target dist-rocm
cmake --install "$THEROCK_BUILD" --component rocm --prefix "$SDK"
sha256sum "$SDK/lib/librocdxg.so.1" > "$EVIDENCE/wsl-bridge.sha256"
cp "$THEROCK_BUILD/CMakeCache.txt" "$EVIDENCE/therock-CMakeCache.txt"
# Legacy FindHIP + Linux linker helper from the SAME patched source pin.
HIP_COMMON="$THEROCK_SRC/rocm-systems/projects/hip"
if [[ ! -f "$SDK/lib/cmake/hip/FindHIP.cmake" ]]; then
  mkdir -p "$SDK/lib/cmake/hip"
  cp "$HIP_COMMON/cmake/FindHIP.cmake" "$SDK/lib/cmake/hip/"
  cp -R "$HIP_COMMON/cmake/FindHIP" "$SDK/lib/cmake/hip/"
fi
if [[ ! -x "$SDK/bin/hipcc_cmake_linker_helper" ]]; then
  install -m 755 "$HIP_COMMON/bin/hipcc_cmake_linker_helper" "$SDK/bin/"
fi
for helper in "$SDK/bin/hipcc" "$SDK/bin/hipconfig" "$SDK/bin/hipcc_cmake_linker_helper" \
  "$SDK/lib/cmake/hip/FindHIP.cmake" "$SDK/lib/cmake/hip/FindHIP/run_hipcc.cmake" \
  "$SDK/lib/cmake/hip/FindHIP/run_make2cmake.cmake"; do
  test -f "$helper"
  sha256sum "$helper" >> "$EVIDENCE/hip-helpers.sha256"
done
