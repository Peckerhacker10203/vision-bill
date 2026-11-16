# Vision-Bill Agent Guide

## Scope and architecture

vision-bill is a local-first, multi-user receipt API and bundled SvelteKit SPA.
Ollama is the default; OpenAI-compatible providers are optional and user-configured.
Keep receipt processing on local endpoints unless the user explicitly chooses an
external endpoint. Never introduce external receipt processing or receipt contents
in general logs. Preserve original bytes, human corrections, exact money and owners.

Public operator contracts are in [UPGRADE_AND_VERIFY.md](UPGRADE_AND_VERIFY.md),
[MULTI_IMAGE_RECEIPTS.md](MULTI_IMAGE_RECEIPTS.md),
[paddleocr-service/README.md](paddleocr-service/README.md) and
[scripts/README.md](scripts/README.md). These describe current behavior; historical
experiment reports do not certify production hardware or accuracy.

## Layout and stack

- Python 3.12+, FastAPI/Uvicorn, Pydantic settings/models, asyncpg/raw SQL,
  PostgreSQL, Alembic hand-written migrations, Argon2id and HMAC session cookies.
- `src/vision_bill/main.py`: app lifespan, providers, services, schedulers, SPA.
- `src/vision_bill/api/`: authenticated routers and public auth/UI config routes.
- `src/vision_bill/model/`: Receipt, nullable extraction, OCR/source, workflow,
  review and experiment/benchmark contracts.
- `src/vision_bill/service/`: orchestration, checks, extraction and benchmarks.
- `src/vision_bill/provider/db/`: raw SQL and transactional ownership/CAS/leases;
  services delegate DB access here. `provider/llm/`: model discovery, extraction,
  repairs/classification; services do not call provider SDKs directly.
- `alembic/versions/`: migrations 0001–0018. No application DDL/autogenerate.
- `paddleocr-service/ocr_service/`: isolated native pipeline, qualification and
  disposable worker lifecycle. `build/` and locks pin runtime/build candidates.
- `frontend/`: SvelteKit static SPA, TypeScript, Tailwind, Vitest, Playwright,
  English/German dictionaries. Built `out/` and app `static/` are generated.
- `e2e/`: unique owned stack runner, strict LLM/OCR stubs and fixture preparation.
- `scripts/`: operational CLIs, CPU OCR comparison and separate split experiment.
- `image-sticher/image-stitcher.html`: optional standalone stitcher, served by
  `docker-compose.image-stichter.yml` on 8081 (historical directory/file spelling).
- `server_data/`, real fixtures, reports/traces, models and local dotenv files are
  private/gitignored. `docs/` is ignored local scratch: put public guides in tracked
  root/existing public paths. Never link public prose to ignored scratch artifacts.

## API and ownership

All `/api/v1` data routes require authentication; auth register/login/logout and
`GET /system/ui-config` are public. Benchmarks/settings require admin. Admin access
across owners additionally requires `auth.admin_can_see_all`; preserve scope on
all reads/mutations, sources, drafts, retries, collections and benchmark references.
Entity IDs are UUIDs. DB-unavailable paths return 503.

- `POST /images`: 1–10 repeated multipart `receipt` fields, at most 20 MB/photo;
  committed 202 + durable ID + Location for **all** uploads. Options `model_id`,
  `bypass_review`, `allow_ocr_fallback` and owner-scoped `Idempotency-Key` preserve
  explicit presence/intent, ordered hashes and first defaults.
- `GET /images`, `GET /images/{id}`, `/file?part=N`, `/thumb`,
  `/views/{view_id}`, `/extractions/{extraction_id}` expose scoped state/evidence.
  No successful synchronous upload shortcut remains.
- `POST /images/analyze` triggers a shared analysis cycle and returns only results
  visible to the caller. `/images/{id}/retry` requires expected_generation/request_key.
  Draft GET reads current state; PATCH requires expected_revision;
  `/draft/finalize` also takes request_key.
- Receipts list/get/update/delete/verify retain exact amounts and file paths.
  Verify uses current operation/review revision and exact finding acknowledgements;
  integrity failures block it. No Verify/bypass file moves.
- Deletion commits rows plus durable owned-file cleanup metadata, then unlinks
  validated originals/thumbnail/working derivatives. Same authorized DELETE resumes
  interrupted cleanup. Active/referenced resources conflict; no arbitrary sweeper.
- Search/statistics use verified receipts. Collections preserve owner-scoped active
  membership, even for see-all admin changes. Tags are normalized vocabulary.
- Benchmark create returns 202/Location; status/results/reevaluate are scoped.
  GET/POST `/benchmarks/references/{receipt_id}` previews and explicitly qualifies
  current human Gold. Verified/bypass/legacy flags alone never suffice.

## Workflow and contracts

`analysis_scheduler.py` coordinates readiness and queue draining;
`submission_service.py` executes fenced whole attempts; `workflow_db.py` owns durable
claims/generations/retry/backoff/drafts/extractions/views/deletion. No ad-hoc worker
loops. Default whole-attempt deadline 600s, max three attempts, 30/60s backoff,
cap 300s, lease 900s; readiness/model-start waits can remain pending at zero attempts.
Stale completions are fenced; a later-photo failure never commits a successful
prefix as a receipt or complete checkpoint. Legacy linked receipts are not re-OCRed.

`extraction_pipeline.py` selects direct vision or explicit OCR followed by text
formatting. Freeze requested/actual model digests and configured alternate identity.
OCR fallback requires optional OCR plus global allow AND explicit request consent;
required OCR prohibits it. All photos and their ordered hashes participate in
identity/replay. Model startup/readiness failure must not silently switch models.

Printed observations live in nullable `ExtractionDraft`; category/tags are optional
downstream enrichment. Missing currency/amounts stay missing. The completed Receipt
boundary and checks decide persistence/finalization, never invented defaults.
Only proposed adjacent suffix/prefix overlaps that pass shared duplicate verification
may be removed. Preserve repeated purchases; no global deduplication or blanket
splitting. `receipt_review.py` checks version 2 performs exact Decimal diagnostics
and policy-dependent financial checks without changing printed arithmetic. Review
exceptions require one exact current finding ID/nonblank comment each; integrity
findings cannot be waived. Local time supports valid seconds and microseconds.

`ImageService` preserves original bytes/hash, creates thumbnails nonfatally and
saves exact validated working PNG artifacts. Shared white RGB composition is
`white-document-rgb-v2`; EXIF correction and native orientation/unwarp are distinct.
Working geometry stays in working coordinates; inverse mapping remains null.

## Native OCR policy

Base app OCR is off. CUDA/ROCm overlays explicitly activate OCR, with sidecar
profile `ocr`; default Dockerfile fails and requires a selected accelerator.
Linux x86_64 candidates target CUDA and native ROCm, including intended consumer
RX6000+/RDNA2. RDNA2 default gfx1030/1031/1032/1034 is distinct from a selected
build-only candidate profile (`OCR_AMDGPU_TARGETS`) and actual per-board qualification.
Never claim universal AMD support or GPU qualification from CPU tests.

Fixed PaddleOCR-VL-1.6/v1.6, PP-DocLayoutV3, native Paddle FP32, batch 1 everywhere,
queues/chart/seal off, orientation/unwarp on, min 112896/max 2000000 **per VLM region**
including tables, OCR 4096. Official upstream grid alignment is not a whole-photo
2MP resize. No platform-dependent precision/backend/pixel/token substitutions or
extra sampling parameters. EXIF applies independently. Formatter output defaults
8192; discovered context capacity remains separate and may be unknown.

`/health` is liveness; `/ready` is qualified-worker readiness; `/info` carries
capability/profile/runtime identity. Serving uses target-sealed model/package/
image/source/device evidence; invalidate/requalify after identity changes.
Readiness polling detects idle death and schedules bounded coordinated replacement;
failed replacement qualification stays unready, no endless loop/shutdown revival.
Actual CUDA/ROCm builds, official CUDA lock resolution, physical inference/quality
and current CPU comparator installation/inference are unexecuted here.

## Benchmarks and scripts

`benchmark_service.py` / `benchmark_db.py` freeze current explicitly human-qualified
Gold and manifest identity, queue model×receipt jobs and fence leases/begin/retry/
finish. Data/reference changes are checked transactionally. V4 exact full-receipt,
occurrence-aware items, amounts/taxes and all-case denominators supplement the
separately named historical success-only averages/v3 scores. Do not redefine old
results or leak expected values into inference. Standalone benchmarks require
`--user-id` and read a consistent SELECT-only snapshot. See script help/current
parser before documenting flags. Historical CPU transcripts have their own explicit
loader contract; they do not become native OCR v2 by rerunning a CPU adapter.

## Configuration, lifecycle and migration credentials

`config.py` precedence: process environment > `.env`/`.env.local` > persistent YAML.
Nested names use `__`. `VISION_BILL_CONFIG_PATH` defaults `/app/config/config.yaml`;
admin writes are atomic/0600. Editable settings expose source/restart flags without
credentials. Provider/host need restart; model/temperature/registration apply live.
`AUTH__SECRET_KEY` is required, optional pepper otherwise falls back to it. Empty
users table permits bootstrap admin and legacy ownership backfill. Registration
default true, admin-wide view default false. Preserve auth/session boundaries.

Lifespan creates provider/services, initializes DB, bootstraps admin, starts analysis
and benchmark workers when ready; shutdown stops workers before closing DB.
SPA fallback is registered after API routes. DB initialization failures are sanitized.

Current migration head `0018`. Stop/drain old app/benchmark workers **before** 0011/0017
queue migrations. Alembic uses settings `PG__*`, including on the image entrypoint;
`MIGRATION_DATABASE_URL` is only the grant tool input. Bootstrap missing group roles
as admin/CREATEROLE, execute grants/default ACL as the actual migration owner,
migrate with owner PG credentials, refresh grants, then run app with runtime login.
An already-current schema is prerequisite for runtime-credential entrypoint startup.
Never imply automatic migration/runtime credential switching or destructive downgrade.
See the public upgrade runbook for backup, legacy mapping and compatible backout.

## Development and verification

`make help` is the command index. `make setup` synchronizes the Python environment;
use a private `UV_PROJECT_ENVIRONMENT` in shared worktrees and do not sync a shared
`.venv` symlink. `make migrate` applies Alembic; `make run` serves dev API on 8080.
`make fe-install` uses npm ci; `make fe-dev` proxies 5173→8080; `make fe-sync` builds
and copies static assets. Do not change locks/dependencies as a prose edit.

Canonical checks: `make lint` (Ruff/mypy on app/native/build/e2e/harness/scripts),
`make test` (non-E2E pytest), `make fe-verify` (Svelte/lint/format/Vitest/E2E types/build),
`make release-test`, `make compose-check`, `make test-e2e`, `make fe-test-e2e`.
E2E uses `e2e/run.py` to allocate unique projects, ports, private settings and fixture
copies and clean only its owned resources. `make e2e-up/down` requires explicit
`E2E_STATE`; no shared fixed stack may be adopted. Playwright can use
`VB_E2E_BASE_URL` for an intentionally supplied stack. Stubs validate known original
and normalized hashes; synthetic tests are not model-quality/hardware evidence.
[Fixture rules](tests/e2e/data/README.md) and [accepted counts/limits](UPGRADE_AND_VERIFY.md#verification-evidence)
state the evidence boundaries. CI verifies software; manual release is separate.

Pre-commit hooks run lint/unit/frontend checks. Documentation-only verification
should follow the assigned scope; never claim unrun broad gates. Prototype scripts
named in the scripts guide have no supported Make targets. Keep real fixtures and generated artifacts
private and out of Git/build contexts. Do not contact placeholder external APIs.

## Edit files
- The edit tool matches oldString byte-for-byte. Read the file again immediately before each edit, and again after every successful edit line content shifts.
- Copy oldString verbatim from the file you just read. Do not retype, reindent or normalise it. Preserve tabs, trailing spaces and blank lines.
- Keep oldString to the smallest span that is still unique — usually one or two lines, not a whole block.
- After two failed edits on the same file, stop and report. Do not switch to rewriting the file.
