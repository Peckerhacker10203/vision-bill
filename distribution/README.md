# Prepared installations and distribution maintenance

The intended user flow is a downloaded, verified prebuilt installation bundle,
then `docker compose up -d`. **Only the producer/consumer source and software gates
ship in this change. No actual public bundles, qualified native images, real-model
installation or physical GPU qualification were produced here.** Until maintainers
complete the gates below, there is no verified native download to recommend.

## Choose an installation

All generated bundles target Linux AMD64 and pin images by immutable registry
manifest digest. They contain `compose.yaml`, `release.json`, `SHA256SUMS` and the
selected evidence/assets under `release/`; there are no build keys or source mounts.

| Variant | Included services | Operator supplies |
| --- | --- | --- |
| `app` | config-init, PostgreSQL, app | Required LLM endpoint/model; optional managed OCR endpoint |
| `ocr-cuda` | native initializer and independent OCR | Qualified NVIDIA device and approved model/fixture assets |
| `ocr-rocm` | native initializer and independent OCR | Exact qualified selected AMD target and approved assets |
| `combined-cuda` | app, PostgreSQL, native OCR, derived Ollama, initializers | Qualified NVIDIA device |
| `combined-rocm` | app, PostgreSQL, native OCR, derived Ollama, initializers | Exact qualified selected AMD target |

Once an actual release is available, obtain the bundle from the maintainer's
verified release, validate its release/source/image identities and supplied checksums,
and extract it into a dedicated persistent directory. Do not use synthetic test
manifests or replace immutable references with guessed tags. From that directory:

```bash
sha256sum -c SHA256SUMS
# Configure the selected variant in a private .env first.
docker compose up -d
```

For `app`, set Compose variables `LLM_PROVIDER` (`ollama` or `openai`), `LLM_HOST`
and `LLM_MODEL`. They seed initial configuration; changing them later does not
replace existing persisted credentials/config. Use the real reachable local URL,
including `/v1` for the intended OpenAI API. A fresh credentialless OpenAI endpoint
gets the `none` API-key sentinel; configure real credentials privately if needed.
`OCR_ENDPOINT` is optional: blank selects direct vision, a URL selects strict OCR.
The app uses mandatory LLM inference in either mode. Generic OpenAI inference works
without managed OCR; shared native OCR requires verified Ollama or explicit dedicated
llama.cpp lifecycle support and stable identities. An incompatible pair waits before
claiming a receipt. Follow [backend setup](../OPENAI_COMPATIBLE.md).

App/combined UI binding defaults to `127.0.0.1:8080`; `APP_BIND_ADDRESS` and `APP_PORT`
change it. Independent OCR defaults to `127.0.0.1:8082` using `OCR_BIND_ADDRESS` and
`OCR_PORT`; keep native lifecycle routes on trusted private networks. Device selectors
are `OCR_CUDA_DEVICE_ID` for NVIDIA, or `OCR_RENDER_NODE`, `OCR_RENDER_GID` and
`OCR_VIDEO_GID` for Radeon. Supply actual host values. Intended AMD candidates start
at RX6000/RDNA2; a build target is not physical support for every board.

### First setup, credentials and persistence

`config-init` writes `/app/config/config.yaml` and `/app/config/initial-admin.yaml`
in `app-config`, and `/run/pg-secret/password` in the separate `pg-secret` volume.
Directories are private (0700); credential/config files are 0600. Random auth,
bootstrap admin and PostgreSQL secrets are persisted. PostgreSQL mounts only its
data and the read-only password-secret volume; it never receives the app config or
initial admin file. Repeated `up`/initialization preserves these bytes/credentials
and rejects conflicting or incomplete state instead of silently regenerating secrets.

Retrieve initial credentials only on your private terminal after app startup:

```bash
docker compose exec app cat /app/config/initial-admin.yaml
```

Sign in with that real bootstrap account. There is no universal password and no
first-visitor promotion. Registration, when enabled, creates ordinary users. Treat
the initial-admin file and persistent config as initial private credential material;
keep them in your backup and restrict access. The initial file is not a password
reset UI. Preserve auth secret/pepper to retain password/session compatibility.

Combined bundles run `ocr-init` to completion before OCR/Ollama serving. The separate
`model-init` waits for Ollama, pulls the bundled default model data when needed and
verifies its expected digest; app startup waits for `model-init` success as well as
config-init and healthy PostgreSQL. Pull/tags verification is data provisioning,
not LLM generation or an accuracy benchmark. `provision-model --bundled-model MODEL`
skips provisioning when persistent provider, endpoint or model deliberately differs;
it never overwrites custom selection or silently pulls a different model. For
custom selections, provision the chosen real model independently and follow the
same lifecycle/identity rules. App startup migrates schema to 0019; read
[upgrade and grants](../UPGRADE_AND_VERIFY.md) before reusing existing data.

Native first setup verifies the release bundle, licenses/hashes/inventory, image
config/source identities and model/fixture assets, then performs the **complete fixed
physical qualification** in a separate initializer. It publishes a selected complete
installation only on success. This is required setup work, not a historical Gemma
comparison or part of a receipt attempt. A matching complete cached installation can
be reused. Ordinary API start and passive `/health` do not cold-load models; explicit
pending work can resume under readiness/policy guards. Readiness can recover an
already-explicitly-loaded crashed worker with coordinated phase fences.

Independent OCR bundles default to `OCR_SERVICE_MODE=standalone`: each HTTP request
owns serial load/infer/park, without VisionBill or an LLM dependency. Set
`OCR_SERVICE_MODE=managed` **before starting an independent OCR bundle paired with
VisionBill**, and arrange the shared app/client handoff file and dedicated LLM
endpoint. Combined bundles fix OCR mode to managed. Do not disable lifecycle
coordination to make a shared native pair start. A cold unmanaged native pipeline
waits before claim. The independent HTTP client and API are documented in the
[native guide](../paddleocr-service/README.md#independent-service-and-client).

### Stop, cut over and recover

Back up the database **and** original/working files, config, credentials and native
state consistently. Stop/drain all app, analysis, benchmark and standalone owners
before changing pipeline/provider/endpoints, migrating or native requalification.
Old/new workers must not overlap. Explicit current-policy retry can adopt new routing
for eligible unfinished work with CAS/history; it cannot rewrite completed receipts
or human edits. See [retry semantics](../MULTI_IMAGE_RECEIPTS.md#attempts-and-explicit-recovery).

Native initializer, OCR and derived Ollama retain their supervisors as container
PID1 (`init: false`). Installation locks prevent initialization during serving;
per-role lifetime locks prevent duplicate servers. Durable active markers survive
hard death. A stale marker blocks work regardless of age; `--force` does not bypass
it. Do not add Docker `--init` or replace the supervisor with a detached server.

For hard-crash recovery, first externally confirm **all** managed initializer, OCR
and Ollama containers using that coordination volume have been removed and their
processes are gone. Only then may the operator archive/remove the corresponding
`installation.active-*.json` crash markers from `/coord`; keep installation/per-role
lockfiles intact. No age timeout, force/autoclear or missing local PID proves safety
across namespaces. Never delete live lockfiles. Do not run `compose down --volumes`
on persisted user data. Recover the separate app model-handoff journal only after
its owned requests/models have been reconciled, as in the
[native ownership guide](../paddleocr-service/README.md#internal-model-ownership).

## Maintainers: real artifact inputs and workflows

The distribution generator is [scripts/distribution.mjs](../scripts/distribution.mjs),
with [artifacts.schema.json](artifacts.schema.json). CPU-only assembly is:

```bash
node scripts/distribution.mjs --manifest /private/artifacts.json \
  --variant app --output /private/new-bundle
```

`--variant` accepts the five choices above. Output must not already exist. This
validates exact asset bytes/evidence and assembles a bundle; it does not build,
pull, qualify or publish. The manifest includes version, exact source repository/
commit, platform, image references **and separate config digests**, asset hashes/
sizes, and app-check evidence. Combined variants add a real model digest, license
text/redistribution permission and provenance. Native variants add their release
bundle, raw image config, full successful qualification, seal and CUDA attestation
where applicable. The [native bundle schema](../paddleocr-service/release-bundle.schema.json)
defines exact four-model archive trees, five real fixture cases, licenses, provenance,
source/profile identities and actual target proof. Do not invent hashes or fixtures.

The shipped manual producer and consumer are:

1. [Prepare approved distribution inputs](../.github/workflows/distribution-inputs.yml)
   runs `scripts/distribution-inputs.mjs --stage` on the runner labeled
   `vision-bill-release-inputs`, in environment `distribution-inputs`. A maintainer
   stages reviewed `release-inputs.json` and exactly the selected native/model/license
   assets under the fixed trusted root
   `/var/lib/vision-bill/distribution-inputs/<source SHA>/<version>/<variant>/`.
   It rejects symlinks/untrusted or writable ownership paths and missing approved files;
   it never takes an arbitrary host directory, downloads assets or publishes. Its
   `distribution-inputs` artifact carries a bound source/version/variant/actor/run
   receipt and the exact asset inventory.
2. [Verified prebuilt distribution](../.github/workflows/distribution-release.yml)
   consumes `inputs_run_id`. Guarding verifies successful same-fork producer workflow
   identity, current producer admin/maintain permission, source SHA, version, variant,
   actor/run receipt and asset hashes. `--verify DIR` checks the downloaded inventory.
   The consumer rejects extra/missing staged files. The selected exact source runs
   CI before build/qualification and publication.
3. The release driver `scripts/distribution-release.mjs` executes `--phase build
   --inputs DIR --output DIR`, then `--phase qualify --output DIR`, then only after
   success `--phase publish --output DIR`. These are workflow-bound phases requiring
   dispatch identity/environment, not standalone user installation commands.
   Build creates exact-source OCI candidates without publication; qualify runs the
   native initializer/full physical gate on the actual selected GPU. App-only choice
   validates the CPU bundle and has no native qualification claim.
4. Publish revalidates transported OCI/config/evidence identities, pushes immutable
   images, verifies **anonymous pullability**, then publishes release assets last.
   A failure before the last step must not create a successful release.

Both workflows require explicit dispatch on the configured fork:
`DISTRIBUTION_FORK_REPOSITORY` must equal the repository, `source_sha` must equal the
actual dispatch `github.sha`, and version/variant must match the staged plan. Only
admin/maintain actors qualify. These workflows must first be adopted on the fork's
**default branch** for GitHub to expose manual dispatch, with environments, package
permissions and operator runners configured. This source change does not adopt them,
merge/update any main branch, dispatch a run or publish a release.

Native runners need actual CUDA or selected Radeon hardware, Docker/buildx, skopeo,
uv and adequate build/archive capacity. Labels are `vision-bill-cuda` and
`vision-bill-rocm`. Combined builds select an immutable accelerator-specific upstream
Ollama package; build and real runtime probe evidence must prove the chosen
CUDA/ROCm backend and actual device/GFX. Generic or CPU-only Ollama evidence cannot
stand in for this. ROCm selected targets must agree with compile inputs, kernel
inventory and the actual qualified AMD device. CUDA requires real official wheel
resolution/attestation; there is no CPU wheel or alternate-precision fallback.

OCI registry **manifest** digest identifies transport/pull bytes; local Docker image
`.Id` identifies the config blob. Neither substitutes for the other. The producer
records both, verifies raw config hashes, source/revision labels, model digests,
package hashes and actual device results. Curated licensed real models, redistributable
fixtures, immutable base images and physical runners remain external prerequisites;
the producer code is shipped, those proofs are not. Software stubs/PID1 tests never
certify native model accuracy, speed, VRAM usage or universal AMD support.

The older app-only image release is documented in [RELEASES.md](../RELEASES.md);
its success does not imply any native bundle is qualified or available.
