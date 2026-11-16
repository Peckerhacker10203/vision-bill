# vision-bill — Your Private Invoice Tracker

vision-bill extracts receipts with local models and provides a bundled web UI for
review, collections, search, spending statistics and model comparisons. Ollama is
the default; [OpenAI-compatible backends](OPENAI_COMPATIBLE.md), including llama.cpp,
are also supported. Keep
that endpoint local to retain local processing. Argon2 passwords, session cookies,
owner-scoped data and an optional admin-wide view support multiple users.

## Start with direct vision

For a **fresh installation**, copy the template, set a random `AUTH__SECRET_KEY`,
your database values and an installed vision model:

```bash
cp .env.example .env
# Edit .env before starting; host Ollama is reachable as host.docker.internal.
docker compose up -d --build
```

Open `http://localhost:8080`. Registration is open by default; bootstrap credentials
create the first admin only on an empty users table. PostgreSQL has no published
host port. The entrypoint runs Alembic with the same `PG__*` identity as the app:
read the [upgrade and verification guide](UPGRADE_AND_VERIFY.md) before upgrading
an existing installation or separating owner/runtime credentials.

## Processing and review

Every upload (one to ten ordered photos, at most 20 MB each) returns committed
`202 Accepted`, a durable ID and `Location` for authenticated status polling.
The worker waits for usable capabilities, extracts printed observations, retains
partial drafts, checks source/amount/overlap evidence, and applies optional
classification downstream. Missing observations stay null; no amounts or currency
are invented. Human corrections and original bytes survive review. Verify changes
database workflow state without moving files. See [multi-photo/API workflow](MULTI_IMAGE_RECEIPTS.md).

OCR is **off in the base app**. Direct vision needs no sidecar. Explicit CUDA or
ROCm overlays enable OCR and require a physically qualified native sidecar; the
[OCR release gate](paddleocr-service/README.md) describes candidate builds,
per-board qualification and the fixed VL1.6/native/FP32 profile. CUDA and Radeon
candidates have not been physically qualified here. Fallback requires global
permission **and** an explicit per-upload choice; required OCR forbids fallback.

## Configuration and comparisons

Environment > dotenv > persistent YAML (`/app/config/config.yaml`, mounted from
`server_data/config`) controls settings. The admin Settings page reports sources;
model/temperature/registration changes apply immediately, provider/host changes
need restart. Credentials and database settings are not exposed there. Formatter
output defaults to 8192 tokens; OCR's 4096-token cap and model context capacity
are separate limits. Unknown context remains unknown. Managed OCR sharing with
llama.cpp requires its dedicated router and explicit `LLM__MODEL_LIFECYCLE=llamacpp`;
standard OpenAI inference alone has no model-unload API. See the
[backend setup and handoff guide](OPENAI_COMPATIBLE.md).

Benchmark Gold requires ordinary human verification plus explicit qualification of
the **current** document and sources. Bypass or legacy `verified` alone is
insufficient. Admin runs freeze selected Gold, ordered source hashes and execution
identity; v4 diagnostics include failed cases and exact receipt/item/amount checks.
See [scripts and role setup](scripts/README.md) and the
[comparison harness](scripts/ocr_harness/README.md). Historical CPU comparisons
and [split experiments](scripts/receipt_split_experiment.md) remain research evidence;
blanket splitting and global product deduplication are not production behavior.

## Development and releases

Python 3.12+, PostgreSQL, uv and Node/npm are used. `make help` lists commands;
`make setup`, `make migrate`, `make run`, `make fe-install` and `make fe-dev`
prepare a developer installation. `make lint`, `make test`, `make fe-verify`,
`make release-test`, `make test-e2e` and `make fe-test-e2e` are verification gates.
E2E commands create and remove unique owned stacks. See the
[agent architecture guide](AGENTS.md), [frontend guide](frontend/README.md),
[accepted evidence and upgrade limits](UPGRADE_AND_VERIFY.md#verification-evidence),
and [manual release procedure](RELEASES.md). This package is Unreleased.
