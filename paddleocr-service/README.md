# Native OCR runtime release gate

These are Linux x86_64/Python 3.12 **qualification candidates**. Neither CUDA nor
any Radeon target has been hardware-qualified here. Do not advertise a target
until its real `qualification.json` has `qualified: true`. No weights or real
receipt fixtures belong in Git or the Docker build context.

## Fixed extraction policy

PaddleOCR-VL-1.6 (`pipeline_version=v1.6`), PP-DocLayoutV3, native Paddle,
FP32; batch 1 throughout, internal PaddleX queues off, chart/seal off. Document
orientation classification and UVDoc unwarping are ON by user choice. EXIF
orientation normalization is independent and still applies. Region min 112896 /
max 2000000 pixels (including table regions), OCR max_new_tokens 4096, no extra
sampling parameters. The official VLM grid alignment applies **per region**;
2MP is not a whole-working-image downscale. No platform-specific precision,
backend, resolution or token reduction is permitted. Formatter 8192/context
capacity are separate app/model facts, and unknown context stays null.

Original bytes/hash/photo identity survive first-frame/EXIF conversion and
white RGB transparency composition (`white-document-rgb-v2`). Actual preprocessor
pixels become a separately hashed working PNG with transform metadata. Native v2
responses contain this PNG and recognized blocks in **working coordinates**.
No inverse UVDoc transform is fabricated (`inverse_mapping=null`). Raw diagnostic
responses can contain receipt text/pixels and belong only in private artifacts.

## Build inputs

`runtime-lock.json` records the immutable Ubuntu base, Ubuntu package snapshot,
Paddle/PaddleOCR/PaddleX/TheRock source pins, compatibility patch hash, Rust
archive hash, default RDNA2 target list and upstream optional exclusions. The
selected build candidate profile is recorded in build evidence/kernels.json.
The common Python execution and native build locks contain real resolved hashes.
`requirements.txt` is not the production build/install contract; use the selected
Dockerfile and hashed accelerator/runtime locks.
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
Docker check; there is no PyPI/CPU/ROCm wheel fallback. Validation parses actual
executable requirements, preserves every entry of the pinned common runtime lock,
and requires the complete attestation. Includes, install options, duplicate or
conditional native requirements and comment-only evidence are rejected. The Make
targets provide the pinned packaging parser in an isolated uv tool environment.

`make ocr-build-rocm` builds the full TheRock SDK and Paddle from the approved
source pins. Provision substantial disk/RAM/CPU capacity first; this is a large
native compilation, not an ordinary sidecar dependency install. `OCR_BUILD_JOBS`
controls parallelism. The versioned patch replaces both legacy HIP target lists,
uses the flat SDK and same-pin legacy FindHIP helpers, requires MIOpen, and stops
the forced download of FlashAttention. FP32 runs with FlashAttention off.

Build evidence is retained under `/opt/ocr/build-evidence`: original gitlinks,
recursive actual checkout HEADs and dirty patches, patch/`.smrev` hashes, native
wheel hashes, CMake caches, compiler commands/log, OS packages, and required
library/payload hashes. Default build candidates are gfx1030/1031/1032/1034.
`OCR_AMDGPU_TARGETS` selects a semicolon-separated **build-only** candidate list:

```sh
OCR_AMDGPU_TARGETS='gfx1030;gfx1100' make ocr-build-rocm
```

Selection changes both TheRock settings and Paddle HIP compiler inputs and is
sealed into `kernels.json`. It does not prove support: even syntactically accepted
architectures must compile, provide inspected libraries/kernels and pass the
physical gate on that exact board. The actual observed HIP architecture must
match the embedded selected profile. Rebuild/requalify after profile changes;
there is no runtime allowlist or architecture spoofing. Intended Linux consumer
scope starts at RX6000/RDNA2 and includes later build candidates requiring their
own physical qualification; no board is actually qualified here.
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
Provision the four exact approved model directories separately into private
`server_data/ocr-runtime/models/` before qualification (Compose exposes them under
`/var/lib/ocr/models/`). This repository has no automatic model-download Make target;
copy/provision the intended model bytes using the official model distribution on
the target host, then retain their identities in qualification. The mounted cache
is not a substitute for complete explicit model directories:

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
`image_identity` containing the actual image identity expected by the selected
qualification route. For prepared releases this is the immutable registry manifest
reference; the separate config blob hash must equal the actual local Docker `.Id`.
`docker image inspect --format '{{.Id}}' IMAGE` returns the config ID, not the
registry manifest digest. Never substitute it for a published image reference.
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

Use `make ocr-qualify-cuda` or `make ocr-qualify-rocm`. This is an explicit offline
hardware gate, never a build prerequisite or automatic comparative benchmark.
First stop/drain the application's analysis, benchmark and standalone workers and
stop its OCR sidecar. Release only Vision Bill's model IDs on its dedicated LLM
endpoint before starting the gate. Verify absence with Ollama's `/api/ps`, or
confirm each canonical llama.cpp router ID has `status.value=unloaded` in
`GET /models`; see [backend control](../OPENAI_COMPATIBLE.md). Stopping the app
alone does not unload its LLM. Leave unrelated GPU applications alone.
These commands run only the
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

Only a full pass publishes `qualification.runtime.json` and a successful report.
Publication errors keep qualification false and remove stale or partially written
serving settings; the CLI exits nonzero. A failure report is written when storage
permits. `qualification.json` records exact profile,
model/image/runtime identities, actual device/GFX, source hashes, timing and audit
counts/digests. It contains no OCR text or images. Only the actual target is added
to `qualified_targets`; repeat the release gate on each physical target.

## Independent service and client

The service, neutral `ocr_contract` and HTTP-only client build from the independent
`paddleocr-service/` context; they do not import the VisionBill application. Prefer
verified [prepared OCR bundles](../distribution/README.md) for installation once
actually released. No such native release was produced here.

OCR-only bundles default to `OCR_SERVICE_MODE=standalone`. One request owns the
whole serial native load/infer/park lifecycle and returns validated original/working
identity and geometry. Managed owner tokens/lifecycle routes are rejected in this
mode. For a VisionBill pair, explicitly set `OCR_SERVICE_MODE=managed` before start:
the app/client supplies generation-fenced lifecycle ownership. Combined bundles
always use managed mode. Never mix standalone HTTP ownership with a managed pair.

From a prepared Python environment in the source checkout, an explicit standalone
request is:

```bash
python paddleocr-service/client.py /private/synthetic.png \
  --url http://127.0.0.1:8082 --timeout-seconds 600
```

`--media-type` can override MIME inference. The client sends the exact original
once with request UUID, photo index and SHA256, validates the matching response,
and prints JSON that may contain receipt text/pixels; keep output private. It needs
HTTP/contract dependencies, no app or native framework import. The endpoint must
actually be exposed by your standalone deployment; source Compose publishes no
OCR host port. The app's optional endpoint selector is independent of service mode.

## First setup and serving handoff

Prepared bundles run `ocr_service.initialize` separately before serving. It checks
[pinned release assets](release-bundle.schema.json), image config/source identities,
licenses and model/fixture inventories, provisions their verified bytes, and runs
the full fixed native qualification. Successful complete state selects runtime and
smoke paths; an exact matching cached installation is reused. This initializer is
required first setup, never an unsolicited historical/Gemma model comparison or
part of a 600-second receipt attempt. See [bundle ordering](../distribution/README.md).

Serving starts parked with no native child/loaded models. `/health` is passive;
`/ready` stays false with `ocr_parked` until explicitly loaded and qualified. `/info`
can expose last qualified identity as history, not current GPU readiness. Polling
never cold-loads OCR merely to answer readiness. An already-explicitly-loaded idle
worker's death can schedule one bounded coordinated replacement, fenced against
active inference/park; failed qualification stays unready without an endless loop.
Parking disables recovery until a new explicit load; shutdown does not revive workers.

Model/package/image/source/target changes invalidate the seal and require fresh
qualification. The app distinguishes parked `can_load` from qualified readiness,
waits before claim on unavailable/incompatible managed pairs, and does not silently
opt out. A cold unmanaged native pipeline also waits before claim.

Advanced source deployments can use the existing manual `make ocr-qualify-cuda` or
`make ocr-qualify-rocm` path above, then run the selected Compose overlay. Those
commands preserve the native entrypoint supervisor, with explicit
`OCR_RUNTIME_SETTINGS_PATH` and `OCR_QUALIFICATION_IMAGE_PATH`; these variables select
the manual runtime route instead of requiring automatic selected-state discovery.
They do not create a prepared release. The direct `ocr_service.qualify` CLI is the
manual qualifier, while `ocr_service.initialize` consumes release bundle/config/ID
inputs and supervises setup as container PID1. Do not substitute one CLI's flags
for the other. The default native Dockerfile intentionally fails; base source
Compose is not a CPU runtime. Source GPU overlays enable legacy OCR; canonical
`OCR__ENDPOINT` wins if explicitly present. Ordinary OCR never permits image fallback,
regardless of deprecated required/allow/alternate fields. To change policy, follow
[worker drain and explicit adoption](../UPGRADE_AND_VERIFY.md#disable-pause-and-recover).

The initializer, OCR entrypoint and derived Ollama supervisor retain PID1 with
`init: false`; do not insert Docker `--init` or detach their owned native processes.
An exclusive installation lock excludes initialization from serving; per-role locks
exclude duplicate initializer/OCR/Ollama roles. Durable markers survive hard death.
`initialize --force` requests requalification only after serving stops and does not
clear crash fences. Ordinary startup never clears them by age or missing local PID.
After hard death, externally confirm **all** managed initializer/OCR/Ollama containers
using the volume are removed and all their descendants are gone. Only then archive/
remove `/coord/installation.active-*.json` markers; retain the installation and role
lockfiles. Never delete live lockfiles or run `compose down --volumes` on user data.
This installation fence is distinct from the app's model-handoff journal below.

## Internal model ownership

With `OCR__MANAGE_MODEL_LIFECYCLE=true` (default), Linux app processes, uploads,
benchmarks including direct vision, and supported standalone clients share one
advisory lock and durable journal. `OCR__COORDINATION_PATH` defaults to
`gpu-handoff.lock` beside `VISION_BILL_CONFIG_PATH`; Compose uses its existing
`/app/config` bind mount. Every process must use the **same underlying file**,
even when host/container paths differ, and the same spelling of the LLM endpoint.
Keep the directory private to Vision Bill and use local storage supporting flock.

One owner releases and verifies its exact owned LLM model IDs, loads OCR, processes
every receipt photo, then parks OCR and confirms native child death. Only then can
formatting, repairs and classification run; the owner unloads and verifies its LLM
before releasing ownership. `/lifecycle/claim`, `/load` and `/park` (all under the
`/lifecycle` prefix) and `/ocr` use generation-fenced owner tokens; stale requests
cannot park/reload a successor. These internal routes must stay on the private
application network. No device-exclusive mode, foreign process termination, or
unloading of every model returned by provider discovery is used.

Use a **dedicated Vision Bill LLM endpoint** for managed sharing; the backend cannot
identify which client owns a shared resident model. Unrelated GPU programs and
unrelated model IDs remain allowed; their memory use can still cause capacity
failures. OpenAI-compatible inference has no standard unload contract. Its default
`LLM__MODEL_LIFECYCLE=none` is rejected for managed sharing; explicit `llamacpp`
enables the dedicated router protocol and waits for exact-ID `unloaded` confirmation.
A successful unload POST, sleeping state or missing model is insufficient. See
[OpenAI-compatible backend setup](../OPENAI_COMPATIBLE.md). Pure LLM OpenAI operation
remains supported. `OCR__MANAGE_MODEL_LIFECYCLE=false` is used by explicit CPU/replay
software stubs;
it makes no GPU handoff claim. Do not recommend it for a native shared pair; a cold
unmanaged pipeline waits before claim and cannot activate the parked worker.

Managed forced-OCR benchmarks, standalone OCR and frozen OCR-enabled uploads
require current production OCR selected (canonical nonblank endpoint or legacy
`OCR__ENABLED=true`) and
`OCR__MANAGE_MODEL_LIFECYCLE=true` on every app worker. Otherwise ordinary direct
LLM work would not share their ownership. Forced qualification/execution fails
before any load; previously frozen managed uploads wait before claim without
spending attempts or rewriting their saved intent. Benchmark capabilities mark
managed OCR unavailable in this configuration. Pure LLM deployments and explicitly
unmanaged CPU/replay modes remain supported.

An explicitly requested OCR benchmark qualifies and parks OCR when freezing its
identity. Readiness polling does not. The frozen identity omits transient residency
but includes the actual qualified profile/runtime, checked again against execution.
Recreate old benchmark manifests after changing this lifecycle contract.

Failed teardown or unconfirmed unload blocks the next phase. A crashed/cancelled
or failed model request whose completion is unknown leaves `unsettled=true` in
the journal, even if a momentary residence probe looks idle. The attempt deadline remains
600 seconds including cleanup. Cleanup uses at most 30 seconds of the remaining
absolute attempt budget; when none remains, pending/unsettled state stays durable
for the next owner's confirmed claim/release barrier. Expiry never becomes a
successful result. A late response cannot clear a successor's journal.
For manual recovery, stop/drain all Vision Bill app/standalone workers and its
sidecar, then establish that the dedicated LLM endpoint has no outstanding
requests (restart only that dedicated endpoint if necessary). Unload/verify only
the journal's owned model IDs using `/api/ps` for Ollama or exact-ID
`GET /models` status `unloaded` for llama.cpp. Only after those checks, with all owners stopped,
archive/reset only the adjacent `.json` handoff journal and restart parked services;
retain the lockfile inode so all participants continue locking the same file.
Never reset ownership while workers are active, delete live lockfiles or restart a
shared foreign runtime. Installation crash markers require the stricter external
all-managed-containers-removed procedure above; clearing the app journal does not
clear that fence.

The historical CPU PP-OCR comparator has an independent hashed build target in
`scripts/ocr_harness/Dockerfile`, selected with
`python3 -m scripts.ocr_harness build --only pp-medium`; it is not a production
sidecar or native v2 qualification source. See the [harness](../scripts/ocr_harness/README.md).

CPU-only unit tests and `docker buildx build --check` validate contracts/recipes,
not native build success, device kernel support or GPU qualification.

Official CUDA lock resolution, native CUDA/ROCm builds, physical inference and
qualification, hardware quality/latency/VRAM and current CPU comparator installation/
inference remain UNEXECUTED here. No VRAM/performance blocker was established.
Historical numerical reports remain historical; see the
[accepted software evidence](../UPGRADE_AND_VERIFY.md#verification-evidence).
