# Operational scripts

Supported Make targets read the trusted shell-compatible ignored `.env.scripts`:

```bash
cp .env.scripts.example .env.scripts
```

Edit it for deliberately selected local services. `SCRIPTS_ENV=/private/tools.env`
selects another file. Make exports those values over application dotenv settings.
The base Compose DB has no host port: host-side scripts need an explicitly
reachable connection/network; template ports do not publish one. Keep credentials
in private files, not ARGS or shell history. Reports and traces may contain receipt
text and must remain private. See [upgrade/roles/recovery](../UPGRADE_AND_VERIFY.md).

## Context-budget diagnostic

```bash
make script-context-budget ARGS="--model installed-local-model"
make script-context-budget ARGS="--model installed-local-model --image /private/sample.png"
make script-context-budget ARGS="--model installed-local-model --db-sample 24"
```

`--host`/`OLLAMA_HOST` selects local Ollama; `--app-url` reads tag vocabulary,
`--tags-file` supplies a JSON tag list, and `--db-sample` reads existing image paths.
The command sends selected images to that endpoint. It requests a prompt count
with `num_predict=1`, bounding requested context by reported capacity and diagnostic
`--num-ctx-cap` (default 16384). Missing/zero/truncated counts are rejected; show metadata
can supply context if tags metadata does not. It counts image/reserved output once;
projected tag growth is an estimate. `--output-budget` defaults 2000 and `--tag-probe`
defaults 250. These diagnostic values do not set the app's formatter 8192, OCR 4096 or
model context window and do not automatically alter resolution/precision/limits.

## Create an API benchmark

The API must be running and the account admin. References must have ordinary human
verification and explicit qualification of the **current** document/sources through
the benchmark reference preview/confirm UI/API. Legacy or bypass `verified` is
insufficient. Put `VB_USERNAME`/`VB_PASSWORD` in the private env.

```bash
make script-create-benchmark ARGS="--model installed-local-model --limit 10"
make script-create-benchmark ARGS="--receipt-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1"
```

`--model` and `--receipt-id` can repeat; otherwise current eligible scoped Gold is
selected. Other flags include `--category`, `--max-source-confidence`, `--limit`,
`--request-timeout-seconds` (default 300, at most 600), `--council-policy` and explicit
council thresholds/flags. This CLI does not expose every experiment UI/API setting;
consult `--help` and `BenchmarkCreate` before assuming a pipeline flag exists.

## Standalone benchmark (no API)

Requires a real application owner UUID and current explicitly qualified human Gold.
Use a genuine SELECT-only reporting login and a connection able to read the
application tables. Original files must be available at their recorded paths.

```bash
make script-standalone-benchmark ARGS="--user-id 125b2a52-60ac-4e1f-b9bf-d223f15c8cd1 --model installed-local-model --limit 20 --output /private/benchmark-results.json"
```

`--user-id` is mandatory; omission never means all owners. `--category`, `--limit`,
`--db-host`, `--db-port`, `--db-user`, `--db-name`, `--ocr-url`, `--llm-host` are
actual selection/connection flags. Prefer private `PG__PASSWORD` over
`--db-password`. `LLM__MODEL_NAME`, `LLM__HOST`, `OCR__SERVICE_URL`, `OCR__TIMEOUT_SECONDS` and
`PG__*` provide defaults. The script explicitly enables OCR and always selects
the OCR pipeline; it does not adopt every application `OCR__*` policy setting.
It reads an owner-scoped, repeatable-read read-only snapshot, without row locks,
DB mutation, API startup or qualification writes. Every selected case, including
changed/missing sources and extraction failures, stays in the v4 denominator.

Output includes manifest hash and model/source/profile/prompt/schema/output/context
identity, observed-field mask, full-receipt correctness, ordered item precision/
recall, exact count/order and amounts. `avg_score` remains separately labeled
**Historical average score (successful cases)**. It is not the all-case score or
perfect-receipt rate. This CLI disables fallback and supplies no consent flag;
do not interpret OCR failure as permission to use vision. Unknown context remains null. No expected
reference values enter inference.

## Native OCR-only diagnostic

First build and **physically qualify** the selected sidecar according to the
[native gate](../paddleocr-service/README.md). Base Compose alone does not build a
usable backend; default native Dockerfile deliberately fails. To serve a qualified
ROCm candidate, for example:

```bash
docker compose --profile ocr -f docker-compose.yml -f docker-compose.rocm.yml up -d
# Run only from a client that can reach this deliberately configured local endpoint:
make script-test-ocr ARGS="--ocr-url http://127.0.0.1:8082 --data-dir /private/synthetic-images --timeout 600 --quiet --output /private/ocr-results.json"
```

Use the CUDA overlay instead for a qualified CUDA image. The GPU overlay explicitly
enables application OCR; it is not an OCR-only app-off switch. Compose publishes no
OCR host port. `127.0.0.1:8082` works only if a local deployment explicitly exposes
it; otherwise use a client on the sidecar network with its service URL. Do not run
this command against an unrelated endpoint or assume an unqualified image is ready.

`--timeout`/`OCR__TIMEOUT_SECONDS` defaults 300. Effective requests are bounded at
600 seconds; increasing it cannot extend the app whole-attempt deadline. This
explicit diagnostic activates its OCR provider, calls no LLM/API, and treats empty
transcription as failure even with HTTP200. Current native v2 JSON includes text,
working-coordinate blocks, profile/runtime identity and actual working PNG payload;
raw output can contain pixels and receipt text. `--quiet` suppresses raw console
output, but `--output` is still private. Rebuild the selected image after native
source changes and rerun qualification: a checkout does not update a running image.
No production CPU fallback or `--timeout 900` smoke recipe is supported.

## Thumbnails and optional stitcher

`scripts/generate_thumbnails.py` loads `scripts/.env` and application defaults.
Configure its genuine DB credentials and `IMAGES__SAVE_DIR`/`IMAGES__TMP_DIR` for the
chosen installation. It only selects rows with null thumbnail_path:

```bash
uv run python scripts/generate_thumbnails.py --dry-run
uv run python scripts/generate_thumbnails.py --limit 20
```

Dry-run reads the DB/filesystem, counts resolvable candidates and missing paths;
it **does not decode images**, validate image integrity, create directories, generate
thumbnails or update rows. A candidate count is not a successful-generation count.
Actual mode makes white-composited EXIF-normalized WebP thumbnails; new uploads
already do so nonfatally. It does not regenerate existing thumbnails automatically.

```bash
docker compose -f docker-compose.image-stichter.yml up -d
```

This optional standalone tool binds tracked `image-sticher/image-stitcher.html`
and serves it on 8081. It keeps selected photo order despite asynchronous reads,
invalidates export on new/reordered batches and blocks incomplete/failed exports.
It does not upload to the app automatically or promote blanket stitching/splitting
into the production extraction path.

## Database roles, backup and upgrade

See [the credential-specific procedure](../UPGRADE_AND_VERIFY.md#credentials-and-grants)
for owner migrations, runtime/readonly grants and missing-group bootstrap. Missing
groups require admin/CREATEROLE first; then `make db-configure-roles` must use the
actual Alembic owner for grants/default privileges on current/future tables and
sequences. `MIGRATION_DATABASE_URL` is only this Make tool's connection; entrypoint
Alembic uses `PG__*`. Runtime credentials cannot apply pending owner DDL. Stop/drain
old workers, back up/restore-test DB and files, migrate to 0018 with owner credentials,
refresh grants and start only compatible workers. No automatic destructive backout.
`bash scripts/pg-backup.sh --help` lists backup/restore options.

## Comparisons and historical prototypes

The [OCR harness](ocr_harness/README.md) is isolated from production native OCR.
The independent historical CPU Paddle target is built by
`python3 -m scripts.ocr_harness build --only pp-medium`; its lock does not certify
physical inference here. The [split experiment](receipt_split_experiment.md)
remains separate. [OCR historical results](ocr_harness/BENCHMARK_RESULTS.md) and
[split historical results](receipt_split_results.md) retain their original numerical
meaning and selected datasets; do not substitute them for current GPU qualification.

`run_functional_tests.py`, `llm_benchmark_runner.py`, `single_image.py`,
`test_external_api.py` are historical placeholders with hard-coded/mock behavior
and no supported Make targets. `test_gpc_tagging.py` is also not a supported
benchmark target. Do not contact prototype external endpoints with receipt data.
