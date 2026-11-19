# vision-bill — Your Private Invoice Tracker

vision-bill extracts receipts with local models and provides a bundled web UI for
review, collections, search, spending statistics and explicit admin model comparisons.
Choose an external LLM endpoint and an explicit model. OCR is optional. Ollama and
[OpenAI-compatible backends](OPENAI_COMPATIBLE.md), including llama.cpp, are supported.
Keep endpoints local to retain local processing. Argon2 passwords, session cookies,
owner-scoped data and an optional admin-wide view support multiple users.

## Install a prepared bundle

The intended end-user installation is a **downloaded, verified prebuilt bundle**,
then `docker compose up -d` from its extracted directory. Choose app with an external
LLM, independent CUDA/ROCm OCR, or app with optional Dedicated OCR on CUDA/ROCm
and an external LLM. Every app variant uses your endpoint. Prepare its private
`runtime.env` and set a stable Compose project name before startup. In a combined
installation the OCR controller starts idle and only downloads the four pinned
PaddleOCR model archives when an admin enables **Dedicated OCR** or requests
setup in the configuration. Follow the [installation guide](distribution/README.md).

**This change ships source and release gates only. No qualified native images or
prepared public bundles were published or physically qualified here.** There is no
release download URL to substitute yet. Maintainers must complete the real artifact
and hardware gates before advertising an installable native bundle. Linux AMD
candidates start at RX6000/RDNA2; each selected board requires qualification.

Once a verified bundle exists, app variants expose the UI at
`http://127.0.0.1:8080` by default. First setup creates a real bootstrap administrator
and private persisted credentials. Registration creates ordinary users; the first
visitor is never promoted. Repeated startup preserves credentials and data.
Read the [upgrade guide](UPGRADE_AND_VERIFY.md) before using existing data.

## Processing and review

Every upload (one to ten ordered photos, at most 20 MB each) returns committed
`202 Accepted`, a durable ID and `Location` for authenticated status polling.
The worker waits for usable capabilities before claiming an attempt, extracts
printed observations, retains partial drafts, checks source/amount/overlap evidence,
and applies classification downstream. A complete saved observation can finish
after interruption without repeating inference once ownership is settled. Missing
printed observations stay null and unavailable categories or tags remain visible. Human corrections and original bytes survive review.
Verify changes database workflow state without moving files. See the
[multi-photo/API workflow](MULTI_IMAGE_RECEIPTS.md).

For external OCR, set the canonical `OCR__ENDPOINT` to an HTTP(S) URL. Set it
explicitly blank to keep direct vision. Leave it unset to use the combined
installation's Dedicated OCR action. When OCR is configured, ordinary formatting,
repairs and classification use **text/evidence only**; OCR failure never sends
receipt images to the LLM. Deprecated fallback fields remain for request replay
and history, not as new permission. Compare OCR and direct vision only through an
explicit admin benchmark. Changed routing can hold pending work until an explicit
current-policy retry; completed receipts are never automatically reprocessed.

## Configuration and comparisons

Process environment > dotenv > persistent YAML controls settings. Prepared bundles
store `/app/config/config.yaml` in the private `app-config` volume. Developer
Compose binds `server_data/config` there and requires the configuration directory
and files to be owned by the app container's effective user. The admin Settings
page reports sources. For writable fields, model, temperature and registration
changes apply immediately, while provider and host changes require a restart. Credentials and database settings are not exposed there. Formatter output
defaults to 8192 tokens; OCR's 4096-token cap and model context capacity are separate.
Unknown context stays unknown. Managed same-GPU OCR/LLM work requires verified
Ollama or the explicit dedicated llama.cpp router lifecycle, stable identities and
shared coordination storage. Explicitly adopt the dedicated endpoint and selected
model identities before running managed work. Incompatible pairs wait before claim. This coordinates
owned models and leaves the physical GPU available to independent processes.

Benchmark Gold requires human verification plus explicit qualification of the
**current** document and sources. Bypass or legacy `verified` alone is insufficient.
Admin runs freeze Gold, ordered source hashes and execution identity; v4 diagnostics
include failed cases and exact receipt/item/amount checks. See
[scripts and role setup](scripts/README.md) and the
[comparison harness](scripts/ocr_harness/README.md). Historical CPU comparisons and
[split experiments](scripts/receipt_split_experiment.md) remain research evidence;
blanket splitting and global product deduplication are not production behavior.
Benchmarks are explicitly requested. App installation and startup use the
configured endpoint and leave model provisioning to its operator. Native
startup checks the qualified installation cache and explicit maintenance performs
physical qualification. Persisted pending work can resume when its readiness and
policy checks pass.

## Advanced: develop from source

Python 3.12+, PostgreSQL, uv and Node/npm are used. `make help` lists commands;
`make setup`, `make migrate`, `make run`, `make fe-install` and `make fe-dev`
prepare a developer installation. For source Compose, copy `.env.example` to a
private `.env`, set a random auth secret, database credentials and explicit LLM
provider, reachable endpoint and installed model ID,
then use `docker compose up -d --build`. For a fresh source installation, set
`AUTH__BOOTSTRAP_USERNAME` and `AUTH__BOOTSTRAP_PASSWORD` before the first account
if you need admin access. Ordinary users can register from the login page.
Base Compose starts the app and PostgreSQL. For a native candidate and the
Dedicated OCR action, use the [source GPU profile instructions](paddleocr-service/README.md#dedicated-ocr-from-source).

`make lint`, `make test`, `make fe-verify`, `make release-test`,
`make distribution-test`, `make distribution-pid1-test`, `make compose-check`,
`make test-e2e` and `make fe-test-e2e` are software gates. E2E creates/removes unique
owned stacks. See the [architecture guide](AGENTS.md),
[frontend guide](frontend/README.md),
[accepted evidence and limits](UPGRADE_AND_VERIFY.md#verification-evidence), and
[release procedures](RELEASES.md). This package is Unreleased.
