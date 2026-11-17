# Prepared installations and distribution maintenance

Install a prepared bundle by downloading a verified release, configuring the
selected variant and running Docker Compose. This package provides the distribution
tools and local software verification. Public release assets and physical GPU
qualification require a separate release run.

## Choose an installation

The Linux AMD64 bundles pin container images by registry manifest digest. They
contain `compose.yaml`, `release.json`, `SHA256SUMS`, `check-compose.sh`, a runtime
environment template and the selected assets and evidence under `release/`.

| Variant | Included services | Operator supplies |
| --- | --- | --- |
| `app` | Config initializer, PostgreSQL, app | External LLM endpoint and explicit model ID. Optional OCR endpoint. |
| `ocr-cuda` | Cache gate, maintenance and independent OCR | Qualified NVIDIA device and approved OCR assets. |
| `ocr-rocm` | Cache gate, maintenance and independent OCR | Qualified selected AMD device and approved OCR assets. |
| `app-ocr-cuda` | App, PostgreSQL, initializers and managed OCR | Qualified NVIDIA device, approved OCR assets and dedicated compatible LLM endpoint/model. |
| `app-ocr-rocm` | App, PostgreSQL, initializers and managed OCR | Qualified selected AMD device, approved OCR assets and dedicated compatible LLM endpoint/model. |

Provision the chosen model on your own Ollama or OpenAI-compatible endpoint.
Installation starts the listed services. Benchmarking is a separate operator action.
Native read-only routes and lifecycle control belong on trusted private networks.

## Configure and start

Use Docker Compose 2.30 or later for literal environment file support. Verify the
release/source/image identities and supplied checksums before starting. Keep
`COMPOSE_PROJECT_NAME` in the shell or Compose's `.env`; keep the app settings in
a separate private `runtime.env`.

```bash
sha256sum -c SHA256SUMS
export COMPOSE_PROJECT_NAME=vision-bill
install -m 0600 runtime.env.example runtime.env
```

The copy command is for fresh setup. Keep the real runtime file on an existing
deployment. For a fresh app, populate `LLM__PROVIDER` (`ollama` or `openai`),
`LLM__HOST` and `LLM__MODEL_NAME` explicitly. An OpenAI-compatible example follows.
Use literal values without shell quoting, `export` or interpolation in `runtime.env`.

```dotenv
LLM__PROVIDER=openai
LLM__HOST=http://host.docker.internal:8083/v1
LLM__MODEL_NAME=your-installed-model-id
LLM__API_KEY=none
```

Use the endpoint's actual model ID and reachable URL. Supply the real key for an
authenticated endpoint. Fresh credentialless OpenAI installs can omit the key or
use `none`. Ollama ignores it. On restored installations, a private empty runtime
file takes its values from the saved configuration. In paired bundles retain the
required managed OCR configuration described below.

For the `app` variant, a nonblank `OCR__ENDPOINT` selects OCR and an explicit
blank value disables it. Omission preserves the applicable saved or legacy
configuration. The app-and-OCR variants require compatible saved and runtime
topology with `OCR__ENDPOINT=http://ocr:8082`, managed lifecycle and fallback
disabled. Resolve conflicting saved OCR settings explicitly before startup. Ordinary OCR
formatting, repair and classification use text and evidence. Direct vision needs
an image-capable selected model. Managed native OCR needs stable identities,
verified Ollama release or dedicated llama.cpp router lifecycle, and an explicit
setup binding. An incompatible pair waits before spending an inference attempt.
See [inference and handoff setup](../OPENAI_COMPATIBLE.md).

App port defaults to `127.0.0.1:8080`, with `APP_BIND_ADDRESS` and `APP_PORT`
available for Compose interpolation. Independent OCR defaults to
`127.0.0.1:8082`, with `OCR_BIND_ADDRESS` and `OCR_PORT`. Set
`OCR_CUDA_DEVICE_ID` for NVIDIA or `OCR_RENDER_NODE`, `OCR_RENDER_GID` and
`OCR_VIDEO_GID` for Radeon as needed. Linux AMD target selection starts with
RX 6000/RDNA2 candidates. Qualify the exact device before relying on it.

```bash
sh ./check-compose.sh
# For an OCR variant, complete explicit native maintenance first.
docker compose up -d
```

### Native cache check and maintenance

The ordinary `ocr-init` service checks the complete qualified cache.
On a cold or changed installation, it reports that explicit maintenance is
required. It cannot start qualification or load models. The profiled
`ocr-maintenance` service verifies the release and runs full qualification
before publishing complete selected state.

Stop participating app, OCR, benchmark and standalone owners. Reconcile
outstanding work and ensure the external LLM is quiesced or OCR has its own GPU.
Use one truthful invocation-local maintenance intent. The commands below use
the CUDA example and read its config digest from `release.json`. For ROCm use
`ocr-rocm` instead. `jq` is used only to read that metadata field.

```bash
image_id=$(jq -er '.images["ocr-cuda"].config_digest' release.json)
docker compose --profile ocr-maintenance run --rm --no-deps ocr-maintenance \
  maintain --bundle /release/bundle.json --state-dir /var/lib/ocr \
  --image-config /release/image-config.json --image-id "$image_id" \
  --maintenance-intent external-llm-quiesced
```

For `app-ocr-cuda` and `app-ocr-rocm`, add
`--handoff-lock /app/config/gpu-handoff.lock`, using the original shared lock
path if customized. With physically separate hardware use
`--maintenance-intent separate-ocr-device`. The initializer checks ownership
and app journal state. Only the operator can attest external GPU quiescence.
`--force` requests requalification and leaves all crash and ownership fences
in effect. The ordinary `up` can proceed after a successful maintenance run.

Independent OCR bundles default to `OCR_SERVICE_MODE=standalone`. Direct clients
use that service without an app or an LLM. Before pairing it with Vision Bill,
set `OCR_SERVICE_MODE=managed` and arrange the original shared coordinator
storage and dedicated endpoint. The app-and-OCR variants set managed mode and
seed `OCR__ENDPOINT=http://ocr:8082` in their runtime template. Adopt the
endpoint and selected model IDs deliberately before managed work.
See [native API and client](../paddleocr-service/README.md#independent-service-and-client)
and [endpoint adoption](../OPENAI_COMPATIBLE.md#adopt-the-managed-pair).

### Credentials and persistence

`config-init` creates private configuration and initial admin credentials in
`app-config`, and the PostgreSQL password in `pg-secret`. The directories use
0700 and private configuration and secret files use 0600. PostgreSQL mounts its
data and a read-only password volume. Restored compatible state reuses the
original credentials instead of generating replacements. After startup, read the
initial admin credentials only in your private terminal.

```bash
docker compose exec app cat /app/config/initial-admin.yaml
```

Sign in with that administrator. Self-registration, when enabled, creates ordinary
users. Keep the initial-admin file, configuration and auth secret/pepper private
and backed up. The app migrates the database to schema 0020. See
[upgrade and grants](../UPGRADE_AND_VERIFY.md).

### Stop, cut over and recover

Take a consistent backup of the database, uploads, working views, credentials,
config and native state. Stop the old owners before switching deployment or
running migration. Keep the original Compose project name and volume mounts even
if the new bundle is extracted to a renamed directory. The app and database
volume identities are separate from the retired bundled Ollama model volume.
Leave that old model volume in place. Move to a deliberately configured external
endpoint if the saved host named the removed Ollama service. Startup retains the
saved endpoint and queued jobs can wait until it is reachable or updated.
Handle unfinished routing changes through
[explicit current-policy retry](../MULTI_IMAGE_RECEIPTS.md#attempts-and-explicit-recovery).

Native initializer and OCR roles use container PID1 supervision. Keep their
installation and role lockfiles intact. A retained active marker blocks work
regardless of age. For hard-crash recovery, confirm that every participant using
the original coordination volume has stopped and that its processes are gone,
including retired managed Ollama containers if applicable. Only then archive
the relevant `installation.active-*.json` marker. Reconcile any separately
retained app journal with the old endpoint before changing identity. Preserve
persistent volumes during Compose teardown.
See [native ownership and recovery](../paddleocr-service/README.md#internal-model-ownership).

## Maintainers: real artifact inputs and workflows

The distribution generator is [scripts/distribution.mjs](../scripts/distribution.mjs),
with [artifacts.schema.json](artifacts.schema.json). CPU-only assembly is:

```bash
node scripts/distribution.mjs --manifest /private/artifacts.json \
  --variant app --output /private/new-bundle
```

`--variant` accepts the five choices above. Output must not already exist. This
validates exact asset bytes/evidence and assembles a bundle; it does not build,
pull, qualify or publish. Manifest schema version 2 records the version, exact source repository and
commit, platform, image references, separate config digests, asset hashes and
sizes, and app-check evidence. Native variants add their release
bundle, raw image config, full successful qualification, seal and CUDA attestation
where applicable. The [native bundle schema](../paddleocr-service/release-bundle.schema.json)
defines exact four-model archive trees, five real fixture cases, licenses, provenance,
source/profile identities and actual target proof. Use actual artifact bytes and
qualification evidence. Schema-v1 manifests and the retired `combined-*` variant
names are rejected. Version 2 ships no LLM server or LLM weights.

The shipped manual producer and consumer are:

1. [Prepare approved distribution inputs](../.github/workflows/distribution-inputs.yml)
   runs `scripts/distribution-inputs.mjs --stage` on the runner labeled
   `vision-bill-release-inputs`, in environment `distribution-inputs`. A maintainer
   stages a reviewed schema-v2 `release-inputs.json` and exactly the selected native
   assets, approved OCR model inputs and licenses under the fixed trusted root
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
`vision-bill-rocm`. Native release qualification requires explicit
`DISTRIBUTION_MAINTENANCE_INTENT` for the runner's actual LLM quiescence or separate
OCR device. ROCm selected targets must agree with compile inputs, kernel
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
