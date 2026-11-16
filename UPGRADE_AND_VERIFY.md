# Unreleased package: upgrade and verification

This guide covers the stabilized package and its subsequent lazy OCR/LLM handoff
and optional OpenAI-compatible llama.cpp router lifecycle. The verification table
below identifies its historical checkpoints; it does not attest later runtime
changes. See [backend setup, ownership and changes](OPENAI_COMPATIBLE.md). This is
an operator procedure, not an executed deployment. The original PR14 optional OCR integration has been stabilized here;
PR15's splitting experiments and historical results remain separate evidence.
Those source imports are not described as remotely merged PRs. The colleague
attachment supplied settings only; no colleague project logic was imported. The
user separately enabled document orientation/unwarping and raised the per-region
maximum to 2,000,000 pixels.

## Resulting processing order

1. Authenticate, validate one to ten ordered originals, retain their bytes/hashes,
   resolve upload intent and commit the submission. Return 202 plus `Location`.
2. Wait for the frozen pipeline's usable model/OCR capabilities before spending an
   attempt. Resolve/freeze actual execution identities; claim with generation,
   token and lease fencing.
3. Normalize document pixels and extract all photos, using direct vision when OCR
   is off or the explicitly enabled native sidecar when OCR is selected. Persist
   exact working views and a complete all-photo checkpoint only when valid.
   Managed OCR first confirms release of owned LLMs and later confirms native OCR
   worker exit before any LLM formatting request.
4. Format nullable printed observations with the configured LLM. Reconcile only
   verified adjacent boundary overlaps; apply classification as enrichment after
   extraction. Preserve missing data, repeated purchases and original descriptions.
5. Save an immutable extraction and editable draft/receipt, evaluate source,
   overlap and financial checks, then require human correction/acknowledgement as
   necessary. Clean bypass can verify, but cannot qualify benchmark Gold.
6. Explicitly qualify current human-reviewed Gold before freezing a benchmark
   dataset and running live extraction or validated replay against it.

The accepted source reviews corrected failures that synchronous/verified-only
instructions concealed: upload acknowledgement now follows commit; readiness waits
do not exhaust attempts; later-photo failure cannot save a success prefix; fallback
binds actual model identity and explicit consent; nullable observations survive
classification; financial checks expose uncertainty without changing printed money;
review and retry use revision/generation fences; deletion can resume owned cleanup;
Gold freezes current human evidence rather than trusting a legacy verified bit.
The additional audits repaired idle native recovery, selected AMD target profiles,
transparent thumbnails, account/view lifetimes, localized findings, role/connection
identity seams, isolated build inputs and truthful script/comparator behavior.
These are correctness changes, not measured OCR accuracy or speed improvements.

## Backup and upgrade order

Keep the existing app image/source identity, configuration, auth secret/pepper,
PostgreSQL roles and ownership, original uploads (including legacy temporary
paths), working artifacts and thumbnails. A database-only backup is insufficient.
Use protected storage and test restoration to an isolated database/filesystem
before changing the installation. Freeze writes or take a consistent coordinated
snapshot so DB references and files agree. Preserve existing exact amounts; this
upgrade does not recalculate historical values or automatically re-OCR receipts.

**Stop/drain all old analysis and benchmark workers before migrating or starting
new queue logic.** Do not run old and new workers concurrently. Stop the app while
keeping PostgreSQL available for migration/backup. The entrypoint automatically
runs Alembic, so an ordinary `compose up` is already a migration operation.

The helper accepts a trusted shell-compatible private environment file. It resolves
the configured PostgreSQL image, uses custom-format `pg_dump`, refuses overwrite
and restores in one transaction. A private env must describe an actually reachable
host/network; the base Compose database has no published host port. For example,
with `PG__HOST=db`, `PG__PORT=5432` in that file and the base network available:

```bash
bash scripts/pg-backup.sh --backup /private/backup.dump \
  --env-file /private/backup.env --network postgres_net
# Restore to a prepared isolated target described by a different private env:
bash scripts/pg-backup.sh --restore /private/backup.dump \
  --env-file /private/restore.env --network isolated_restore_net
```

`--replace` explicitly drops objects present in the backup before restore; use it
only for an intentionally replaceable target. Backup/restore need genuine database
privileges and separately provisioned login/group roles. No operator backup,
restore, migration or service command was run while writing this guide.

### Credentials and grants

Use a genuine schema owner/migration login, a separate app login in
`vision_bill_runtime`, and a reporting login in `vision_bill_readonly`. The app
requires writes and cannot use the reporting login. Neither runtime nor reporting
login should be an owner or superuser. The initial Compose `POSTGRES_USER` is a
bootstrap database identity, not automatic provisioning of this role separation.

Creating missing NOLOGIN groups in `scripts/configure_db_roles.sql` requires an
administrator/CREATEROLE identity. An ordinary NOCREATEROLE migration owner cannot
bootstrap missing groups. Have the database administrator create them first:

```sql
CREATE ROLE vision_bill_runtime NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE ROLE vision_bill_readonly NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
```

Then run the unchanged grant procedure **as the actual Alembic object owner**, with
`MIGRATION_DATABASE_URL` and `PG__DB` set in a protected `.env.scripts`:

```bash
make db-configure-roles
```

The SQL grants CONNECT/schema USAGE, runtime SELECT/INSERT/UPDATE/DELETE and
readonly SELECT on all current tables, runtime USAGE/SELECT on sequences, and
matching default privileges for future tables/sequences created by that owner.
Run it before migration to establish defaults and again afterwards for new objects.
An administrator provisions separate login passwords and grants the appropriate
group. Readonly reporting logins may also set `default_transaction_read_only=on`
and a bounded `statement_timeout`.

**Alembic uses `PG__USER`, `PG__PASSWORD`, `PG__DB`, `PG__HOST`, `PG__PORT` from
application settings. `MIGRATION_DATABASE_URL` belongs only to the Make grant tool;
it does not switch entrypoint/Alembic credentials.** Set actual owner `PG__*`
credentials in a protected migration environment before running:

```bash
uv run alembic current
uv run alembic upgrade head
uv run alembic current
```

Run from a prepared checkout/environment on a network that reaches the intended
DB. Clear conflicting process/dotenv settings and use a private
`VISION_BILL_CONFIG_PATH` as appropriate: process env > dotenv > YAML precedence
still applies. No DSN/password needs to appear in public commands or shell history.
The resulting head is **0018**. Before starting uvicorn with runtime credentials,
complete owner migration and grants. The current image still executes Alembic at
startup using the runtime `PG__*` identity; it can only check an already-current
schema, not apply pending owner DDL. Future upgrades must repeat the owner step.
Do not temporarily leave the app serving as the owner to simplify migration.

### Fresh and legacy mapping

Fresh installations apply 0001 through 0018. Existing supported 0010 installations
apply additive workflow/review/reference/claim/deletion migrations 0011–0018.
Migration 0011 preserves linked receipts, verified flags, amounts and historical
paths and marks their submissions analyzed; they are not re-OCRed. Receipt-free
legacy failed/processing images become pending with attempt_count 0/max_attempts
3, retaining prior status/error/processing history in `legacy_events`. Historical
benchmark attempt counts that cannot be known are not invented by migration 0017.
0014 retains exact Decimal/NUMERIC values; 0018 supplies durable owned deletion
cleanup. Earlier existing migrations remain in the chain.

After head/grants, start only the compatible new application. Verify login and
owner/admin scope, polling of a new synthetic upload, a preserved old receipt and
its sources/amounts, draft review, and an explicitly qualified synthetic benchmark
reference. Linked old receipts require no extraction. Keep private receipts and
raw traces out of general logs/reports.

### Disable, pause and recover

Keep workers stopped during a migration failure; fix owner credentials/grants or
restore the coordinated backup to an isolated target before cutover. For a safe
pause, stop the app/workers and retain database/files. To disable OCR for **new**
uploads, remove the GPU overlay and set `OCR__ENABLED=false`,
`OCR__REQUIRED=false`; restart deliberately. Existing submissions retain frozen
intent and may keep waiting for their original capabilities. This is not a queue
rewrite. Never start an old worker on the new queue/schema and never automate a
destructive Alembic downgrade. Data-bearing downgrade guards intentionally refuse
loss. Backout needs a compatible image or explicit restore of the coordinated
pre-upgrade database/files, with new workers stopped and post-backup writes handled.

Deletion commits database removal plus an owner-scoped durable manifest before
unlinking files. If cleanup is interrupted, retry the same authorized image/receipt
DELETE; missing already-removed files are acceptable, changed identities/symlinks
are not. Cleanup metadata remains until all owned cleanup succeeds. This is not a
general background sweeper. Active or still-referenced resources return conflicts;
benchmark references can prevent receipt deletion. Do not manually remove arbitrary
paths or assume every failed HTTP response means the DB deletion rolled back.

## Review and financial contract

Checks version **2** separates `integrity` blockers from review exceptions.
Required missing observations, unavailable/changed originals and invalid source
references prevent finalization/verification. Unknown source mapping and unresolved
overlaps require review. Observations remain nullable; human correction fills
actual missing mandatory data before finalization. No automatic currency/amount
substitution or arithmetic correction is performed.

`AmountPolicy` states tax basis (`gross/net/unknown`), subtotal origin
(`printed/derived_net/unknown`), tax information (`present/explicit_zero/missing`),
discount treatment (`included_in_lines/subtract/unknown`) and tip treatment
(`included/add/outside_receipt/none/unknown`). Unsupported/unknown conventions
produce findings, not a forced net-plus-tax equation. With supported policy the
line sum checks subtotal (subtract tax for a gross derived-net subtotal); total
adds tax for net/derived-net, subtracts discount only for `subtract`, and adds tip
only for `add`. Gross derived-net with separate discount/tip can be uncertain and
is flagged rather than guessed. Explicit zero tax conflicts with a nonzero tax.

Decimal comparisons flag differences **greater than** 0.01 for quantity×unit price
versus line total, 0.05 for subtotal, 0.02 for tax sum and 0.05 for total. These are
review tolerances in recorded monetary units, not score tolerances or permission
to round stored values. V4 exact amount metrics forgive no monetary discrepancy.
Local receipt time accepts valid `HH:MM`, `HH:MM:SS` and up to six fractional second
digits; preserve seconds/microseconds. Invalid clock values are rejected.

Use `expected_operation_revision` and `expected_review_revision` from current
receipt review state. Every current review finding needs exactly one
`{finding_id, comment}` acknowledgement with a nonblank comment; stale, extra or
duplicate IDs are invalid. The operation locks/rechecks current state; stale CAS
returns conflict. Integrity findings cannot be acknowledged away. Changes can
invalidate findings/revisions and require fresh human review. Bypass can auto-verify
only with no current findings and identified originals; it is not human verification
or Gold qualification. Verify updates DB/history and preserves file paths.

## Benchmarks and reproducibility

Admin reference preview/qualification lives at
`/api/v1/benchmarks/references/<receipt_id>` (GET then POST with
`expected_review_revision` and `expected_document_sha256`). Gold requires current
ordinary human verification, exact acknowledgements and complete available ordered
originals with matching hashes. Legacy/bypass verified flags are insufficient.
The immutable reference records observed-field mask, expected data, review/schema/
scoring versions, owner/actor identity and source evidence. Edits require current
qualification; run creation checks the current reference again transactionally.

Admin runs explicitly select models/receipts and freeze source/model digests,
profile/runtime, prompt/schema/source/output/context metadata, policy and alternate
fallback identity. Direct vision needs vision capability; OCR needs text formatting;
normal upload follows frozen OCR configuration. Replay must match every ordered
photo/hash/profile/runtime and retained artifact; no expected/reference values enter
inference. Fallback is off unless explicitly allowed by both policies and is labeled
with actual model identity. Changed model digests wait rather than silently drifting.
Observed masks prevent absent reference observations from being treated as truth.

V4 reports full-receipt correctness, occurrence-aware ordered item precision/recall,
exact item count/order, exact amounts/taxes and failed-case denominators. Keep the
separately named historical success-only average and v3 weighted score meaning;
neither is a perfect-receipt rate. Frozen Gold is not a population accuracy claim.
The [standalone CLI](scripts/README.md) requires an explicit owner `--user-id` and
uses a consistent read-only snapshot; it cannot benchmark all owners by omission.
Historical CPU transcripts use a separate explicit adapter contract, never fake
native qualification/geometry. See the [harness guide](scripts/ocr_harness/README.md).

## Verification evidence

Accepted source `7d82847` follows all fourteen planned tasks, the accepted
whole-package review/repairs and four additional audits with twenty accepted repair
dispositions. The preceding whole-package checkpoint `f116fe7` had fresh complete
336 backend and 97 browser passes. Final additional-audit evidence is:

| Gate | Accepted evidence and boundary |
| --- | --- |
| Backend E2E | Complete 343 passed in 849.93s on `da5c3cc`; carried only across unchanged backend/runtime/dependency inputs to `7d82847` |
| Python unit/static | 1098 passing; Ruff clean and mypy 113 source files; carried only for unchanged components |
| Frontend components | Fresh 31 files / 213 tests; Svelte 0 errors / 0 warnings, ESLint/Prettier and E2E TypeScript clean, static build at `7d82847` |
| Browser | Fresh complete 97 passed in 7.3m on a new normally built current-source app image at `7d82847` |
| Release/config | 14 release tests and three Compose validations carried only for unchanged components |

The backend command was the unfiltered private-stack `e2e/run.py run --no-build`
execution against the attested normal app artifact, after two capacity-interrupted
127-test attempts. The final browser command was canonical `make fe-test-e2e` with
a private build override supplying only build-time CA trust for dependency fetches.
Final image `sha256:2fb2149d728efe7dad04842a6fb1e8f92c96e65002a97b0aee5f4af99453c1c3`
attested all 100 backend source identities and 83 frontend assets; unchanged base/
dependency/backend layers justified backend carry. Owned cleanup finished with no
remaining resources. A previous 96-pass/1-fail browser run remains failed history;
it is not combined with the final pass. Nonblocking npm/Vite timing notices remain.
These counts are historical package evidence, not suites rerun for documentation
or evidence for the later lifecycle extensions. Follow the
[current branch CI](https://github.com/Peckerhacker10203/vision-bill/actions?query=branch%3Acodex%2Fvision-bill-update-package)
for subsequent GPU-handoff/OpenAI changes and read each job's actual conclusion.

Actual official CUDA wheel-lock resolution, CUDA/ROCm native compilation, physical
GPU inference/qualification, current historical CPU image installation/inference,
quality/latency/VRAM measurements remain **UNEXECUTED** here. CPU stubs and source/
CMake checks certify software contracts only. No VRAM/performance blocker was
established, and universal AMD support is not claimed. Intended Linux consumer
scope is RX 6000+/RDNA2 and later selected candidates, each requiring its own real
qualification. See [native prerequisites and commands](paddleocr-service/README.md).

For future changes, canonical gates are `make lint`, `make test`, `make fe-verify`,
`make release-test`, `make compose-check`, `make test-e2e`, `make fe-test-e2e`.
Use isolated environments and unique owned E2E stacks. Safe documentation checks
can run help/parsers, local links and synthetic effective Compose without launching
services; they cannot replace real operator/hardware acceptance. Publishing remains
a separate [manual release](RELEASES.md) action; no tag/release/deployment is implied.
