#!/usr/bin/env bash
# Unexecuted native qualification candidate; this script never declares GPU support.
set -euo pipefail
BUILD_JOBS="${BUILD_JOBS:-8}"
SOURCE_ROOT="${SOURCE_ROOT:-/opt/native-src}"
SDK="${SDK:-/opt/rocm}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE=/opt/ocr/build-evidence
mkdir -p "$SOURCE_ROOT" "$EVIDENCE" /opt/ocr/bin /opt/wheels
THEROCK_SRC="$SOURCE_ROOT/TheRock"
THEROCK_BUILD="$SOURCE_ROOT/therock-build"
PADDLE_SRC="$SOURCE_ROOT/Paddle"
# Fresh sources/caches only: a previous native build is not a reproducible input.
test ! -e "$THEROCK_SRC" && test ! -e "$PADDLE_SRC" && test ! -e "$THEROCK_BUILD"
TARGETS="$(python "$RECIPE/../ocr_service/targets.py" --targets "${AMDGPU_TARGETS:-}" --output "$EVIDENCE/targets.json")"
export VISION_BILL_AMDGPU_TARGETS="$TARGETS"

git clone --no-checkout https://github.com/ROCm/TheRock.git "$THEROCK_SRC"
git -C "$THEROCK_SRC" checkout --detach b324869fb3116353ac7eec5e6926aa3d452ac987
git -C "$THEROCK_SRC" ls-files --stage > "$EVIDENCE/therock-original-gitlinks.txt"
bash "$RECIPE/prepare_therock.sh" "$THEROCK_SRC" "$EVIDENCE/therock-sources.json" "$BUILD_JOBS"
(
  cd "$THEROCK_SRC"
  INSTALL_PREFIX=/usr/local bash dockerfiles/install_pinned_patchelf.sh
)
# All SDK components retained, including host APIs for upstream-excluded Lt targets.
cmake -S "$THEROCK_SRC" -B "$THEROCK_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DTHEROCK_ENABLE_ALL=ON \
  -DTHEROCK_AMDGPU_FAMILIES= "-DTHEROCK_AMDGPU_TARGETS=$TARGETS" \
  -DTHEROCK_DIST_AMDGPU_FAMILIES= "-DTHEROCK_DIST_AMDGPU_TARGETS=$TARGETS" \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=vision-bill-candidate-v1 -DBUILD_TESTING=OFF
cmake --build "$THEROCK_BUILD" --target dist-rocm --parallel "$BUILD_JOBS"
cmake --install "$THEROCK_BUILD" --component rocm --prefix "$SDK"
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

git clone --no-checkout https://github.com/PaddlePaddle/Paddle.git "$PADDLE_SRC"
git -C "$PADDLE_SRC" checkout --detach 7688495538f4d6c1893f084dd238a402e8f68ab6
git -C "$PADDLE_SRC" submodule update --init --recursive --jobs "$BUILD_JOBS"
(
  cd "$PADDLE_SRC"
  git apply --check "$RECIPE/hip-rdna2-v1.patch"
  git apply "$RECIPE/hip-rdna2-v1.patch"
)
sha256sum "$RECIPE/hip-rdna2-v1.patch" > "$EVIDENCE/compatibility-patch.sha256"
python "$RECIPE/source_inventory.py" "$PADDLE_SRC" "$EVIDENCE/paddle-sources.json"
export ROCM_PATH="$SDK" HIP_PATH="$SDK" HIP_CLANG_PATH="$SDK/lib/llvm/bin"
export PATH="$SDK/bin:$SDK/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib/llvm/lib:$SDK/lib/rocm_sysdeps/lib"
export CMAKE_PREFIX_PATH="$SDK" CMAKE_MODULE_PATH="$SDK/lib/cmake/hip"
export PADDLE_VERSION=3.3.1 WITH_ROCM=ON WITH_GPU=OFF WITH_PYTHON=ON
export WITH_TESTING=OFF WITH_DISTRIBUTE=OFF WITH_RCCL=ON
export WITH_CINN=OFF WITH_TENSORRT=OFF WITH_FLASHATTN=OFF
export WITH_MKL=OFF WITH_ONEDNN=OFF MAX_JOBS="$BUILD_JOBS"
export CMAKE_EXPORT_COMPILE_COMMANDS=ON CMAKE_VERBOSE_MAKEFILE=ON VERBOSE=1
(
  cd "$PADDLE_SRC"
  python setup.py bdist_wheel 2>&1 | tee "$EVIDENCE/paddle-build.log"
)
cp "$PADDLE_SRC"/dist/*.whl /opt/wheels/
sha256sum /opt/wheels/*.whl > "$EVIDENCE/native-wheels.sha256"
cp "$PADDLE_SRC/build/CMakeCache.txt" "$EVIDENCE/paddle-CMakeCache.txt"
if [[ -f "$PADDLE_SRC/build/compile_commands.json" ]]; then
  cp "$PADDLE_SRC/build/compile_commands.json" "$EVIDENCE/paddle-compile-commands.json"
fi
python "$RECIPE/source_inventory.py" "$PADDLE_SRC" "$EVIDENCE/paddle-built-sources.json"
"$SDK/bin/hipcc" "$RECIPE/hip-device-info.cpp" -o /opt/ocr/bin/hip-device-info
# ELF/linker resolution and real code-object inventories are separate gates.
ldd /opt/ocr/bin/hip-device-info > "$EVIDENCE/hip-probe.ldd"
if grep -q 'not found' "$EVIDENCE/hip-probe.ldd"; then exit 1; fi
# Runtime wheel install and kernel inventory run in Docker after the locked closure.
