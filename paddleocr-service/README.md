# Native OCR runtime release gate

These are Linux x86_64/Python 3.12 **qualification candidates**. Neither CUDA nor
any Radeon target has been hardware-qualified here. Do not advertise a target
until its real `qualification.json` has `qualified: true`. No weights or real
receipt fixtures belong in Git or the Docker build context.

## Build inputs

`runtime-lock.json` records the immutable Ubuntu base, Ubuntu package snapshot,
Paddle/PaddleOCR/PaddleX/TheRock source pins, compatibility patch hash, Rust
archive hash, exact global RDNA2 target list and upstream optional exclusions.
The common Python execution and native build locks contain real resolved hashes.
Source packages are built without dependency resolution, using the hashed build
bootstrap. The archived PaddleOCR/PaddleX sources use their pinned release
versions through `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*`.

CUDA deliberately has an unresolved **build gate** in `requirements.cuda.lock`:
the execution environment returned CONNECT 403 for the official index and wheel
host. On a connected machine run `make ocr-lock-cuda`. This downloads only the
exact official cu126 Paddle GPU 3.3.1 cp312 x86_64 wheel, inspects its real
METADATA, freezes the full closure, hashes the bytes, and emits the lock plus
`requirements.cuda.attestation.json`. Review both outputs before
`make ocr-build-cuda`. Missing, changed, substituted or unresolved locks fail the
Docker check; there is no PyPI/CPU/ROCm wheel fallback.

`make ocr-build-rocm` builds the full TheRock SDK and Paddle from the approved
source pins. Provision substantial disk/RAM/CPU capacity first; this is a large
native compilation, not an ordinary sidecar dependency install. `OCR_BUILD_JOBS`
controls parallelism. The versioned patch replaces both legacy HIP target lists,
uses the flat SDK and same-pin legacy FindHIP helpers, requires MIOpen, and stops
the forced download of FlashAttention. FP32 runs with FlashAttention off.

Build evidence is retained under `/opt/ocr/build-evidence`: original gitlinks,
recursive actual checkout HEADs and dirty patches, patch/`.smrev` hashes, native
wheel hashes, CMake caches, compiler commands/log, OS packages, and required
library/payload hashes. Global targets are exactly gfx1030/1031/1032/1034.
Optional upstream exclusions (including hipBLASLt's gfx1100 fallback) stay intact;
their host headers/APIs are retained, with no claim of RDNA2 fused/Lt kernels.

Missing required libraries, incorrect Paddle HIP target flags and parsed missing
required target payloads fail the build. Packed/JIT formats which LLVM cannot
prove remain `offline_status: unverified`, with all inspected bytes hashed.
Only a real selected-GFX FP32 GPU matmul/convolution probe with checked results,
followed by complete model inference and fresh replay, can close that gate.
Unknown compile/API/kernel/probe failures are failures, never downgrades.

## Local qualification inputs

Use ignored `server_data/ocr-runtime/` for the private corpus, model directories,
and outputs; `server_data/ocr-cache/` is the actual `PADDLE_PDX_CACHE_HOME` mount.
Download/provision the four approved models separately into `models/`:

- `PP-DocLayoutV3`
- `PP-LCNet_x1_0_doc_ori`
- `UVDoc`
- `PaddleOCR-VL-1.6-0.9B`

Create `candidate.runtime.json` matching `ocr_service.profile.RuntimeSettings`:
`accelerator`, `gpu_index` (inside the container), all four `model_dirs` using
absolute container paths, `cache_dir: /var/cache/paddlex`,
`qualification_manifest: /var/lib/ocr/static-manifest.json`,
`expected_packages: {}` for initial sealing,
`expected_paddle_commit: 7688495538f4d6c1893f084dd238a402e8f68ab6`, and
`image_identity` containing the actual immutable built image digest. Obtain the
local image content ID with `docker image inspect --format '{{.Id}}' IMAGE`.
The qualifier fills the exact installed active dependency versions/content hashes;
a nonempty supplied map must match exactly. The sealer includes `paddlex[ocr]`
and all active marker/extras-aware transitive dependencies, with exactly one
native Paddle distribution. No dependency identities are hand-entered.

`fixtures.toml` must have exactly five `[[cases]]` entries, one each with `kind`
`upright`, `rotated`, `warped`, `table`, `formula`; each entry specifies `path`
(relative to the TOML file or an absolute container path) and the actual original
file `sha256`. Use five distinct real inputs. For `table` and `formula`, also set
`required_labels = ["table"]` or `["formula"]`. Their native results must contain
that structure. Keep the upright input at `fixtures/upright.png` for the default
startup path, or explicitly change `OCR_QUALIFICATION_IMAGE_PATH`.

Use `make ocr-qualify-cuda` or `make ocr-qualify-rocm`. These commands run only the
sidecar, with the runtime input/output volume writable for this gate. For Radeon,
set `OCR_RENDER_NODE`, `OCR_RENDER_GID` and `OCR_VIDEO_GID` to the actual host node
and groups; only KFD and the selected render node are passed. For NVIDIA,
`OCR_CUDA_DEVICE_ID` selects the host device; the selected single device normally
has container index 0. No architecture spoofing or privileged container is used.
The direct CLI is:

```sh
python -m ocr_service.qualify --accelerator cuda \
  --runtime /var/lib/ocr/candidate.runtime.json \
  --cases /var/lib/ocr/fixtures.toml --output /var/lib/ocr/qualification.json
```

`--runtime` defaults to `OCR_RUNTIME_SETTINGS_PATH`. The CLI exits nonzero on any
failure and writes a sanitized failure stage/type; framework output and fixture
text never go to general logs. Each native child has a 600-second bound and is
killed/reaped on timeout.

## What qualifies

The acyclic sequence is installed dependency inspection → separate PIR/legacy
artifact inspector subprocesses → exact unsealed native construction and hooks →
seal → fresh ordinary `build_pipeline` → `adapter.qualify` → gated `predict`.
No fabricated coverage enters construction. Exported DenseTensor inventory is
CPU artifact evidence; actual compiled device/FP32 evidence comes from native
hooks. Static Layout/Orientation/UVDoc are audited separately and every floating
exported parameter must bind by its exact preserved name to an observed GPU
input. Shape matching, arbitrary zipping and many-to-one binding are forbidden.

Each corpus case executes twice during collection. Every individual forward must
have the same complete floating hook record set; a corpus union is never accepted.
Unproven CPU floats, optimized-away/renamed weight bindings, varying branch
coverage, missing callbacks or unstable fresh-process identifiers fail the gate.
Two independent fresh workers replay each case, first qualifying the sealed
upright smoke through the ordinary adapter. Reports distinguish startup, actual
cold smoke, first case **after smoke**, and warm case timings; they do not claim
cold case inference after warming with smoke. Dynamic VL parameters and all three
static models must pass FP32/device checks on each replay.

Only a full pass writes `qualification.runtime.json`; a new failed attempt removes
a stale file at that output path. `qualification.json` records exact profile,
model/image/runtime identities, actual device/GFX, source hashes, timing and audit
counts/digests. It contains no OCR text or images. Only the actual target is added
to `qualified_targets`; repeat the release gate on each physical target.

## Startup handoff

Use the generated `qualification.runtime.json`, the sealed manifest, exact model
bytes and exact upright image for deployment. Compose mounts runtime inputs
read-only and exposes no OCR host port. `/health` is liveness; `/ready` becomes
healthy only after fresh startup qualification. The base stack keeps OCR optional
under profile `ocr`; explicitly select a CUDA/ROCm overlay and configure the
application OCR enabled/required/fallback policy. Never silently activate OCR or
reprocess stored receipts as part of packaging.

CPU-only unit tests and `docker buildx build --check` validate contracts/recipes,
not native build success, device kernel support or GPU qualification.
