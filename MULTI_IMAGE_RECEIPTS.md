# Uploads, multi-photo receipts and review

Use one to ten ordered photos (at most 20 MB each) of one receipt. Include header
and total and a few overlapping rows between adjacent photos; reorder before
upload. Repeated purchases elsewhere remain distinct.

## Durable acceptance and polling

`POST /api/v1/images` accepts repeated multipart `receipt` fields. **Single and
multi-photo uploads both return 202**, after committing originals and submission
identity. `Location: /api/v1/images/<id>` identifies the authenticated status
resource. The response ID is the submission/image UUID, not a completed receipt.
A disconnected client can resume by ID; `/upload?job=<id>` survives reload.
Queue shows pending/processing/errors; completed receipts appear in Receipts.

Optional query parameters are `model_id`, `bypass_review` and
`allow_ocr_fallback`. An `Idempotency-Key` is owner-scoped: replaying the same
ordered original hashes and explicitly supplied options returns the same ID;
changed content or intent returns 409. Omitting a flag differs from explicitly
sending false. The first resolved defaults are frozen, so changing server settings
does not reinterpret a replay. Generated paths/thumbnails are not request identity.

Read `status`, `terminal`, `generation`, `attempt_count`, `max_attempts`, `retry_at`,
`wait_reason`, `result_code`, `draft_available`, `receipt_id` and `source_views`.
Pending capability/provider/model-start waits can consume zero attempts. Terminal
states are `analyzed`, `needs_review`, `failed`, `timed_out`, `unreadable`.
EN/DE UI explanations distinguish waiting from failed attempts; raw codes remain
secondary diagnostics. Acceptance is not a claim that extraction succeeded.

## Attempts and explicit recovery

Each whole attempt has a 600-second maximum including OCR, discovery, formatter
and repair work; defaults allow three automatic attempts with 30/60-second
backoff, configurable cap 300 seconds and a 900-second claim lease. Lower configured
deadlines apply. Generation/claim tokens fence stale completions and expired leases.
Unusable readiness is checked before claiming. Retryable failures retry within the
bound; unreadable input is terminal. A later photo failure never saves a successful
prefix as a completed receipt or reusable all-photo OCR checkpoint.

`POST /api/v1/images/<id>/retry` takes `expected_generation` and a nonempty
`request_key`. Replaying that key returns its original outcome; a new retry must
use the current generation and a new key. Conflicts return 409. Retrying preserves
sources and frozen intent rather than silently adopting new server defaults.

## Extraction and source evidence

Direct vision receives all ordered photos. OCR processes each photo with the fixed
native profile, then formats all ordered transcripts. Requested and actual model
IDs/digests, OCR profile/runtime and fallback reason are recorded. Optional fallback
requires frozen global permission AND explicit `allow_ocr_fallback=true`; required
OCR forbids it. A text-only formatter needs an explicitly configured/discovered
vision alternate to fall back; that alternate's identity is frozen too.

Only model-proposed ordered suffix/prefix overlaps between adjacent photos may be
removed. `helper/duplicates.py::verify_duplicate` checks description, quantity and
line total (exact defaults, normalized case/whitespace in descriptions). Every pair
must pass. No global deduplication or blanket image splitting is applied.

Original bytes, hashes and photo/part identities are retained. Working PNGs have
their own IDs, dimensions, hashes and transform metadata. EXIF normalization is
independent of orientation classification/unwarping; transparency is composited on
white (`white-document-rgb-v2`). OCR boxes are in the actual working image's
coordinates. An inverse unwarp mapping is unknown (`null`), never invented.
Authenticated `/file?part=N` serves originals; `/views/<view_id>` serves validated
working PNGs with private/no-store caching. Review displays both.

## Drafts, checks and deletion

Printed extraction observations can be null, including currency and amounts;
classification/category/tags are downstream enrichment. A partial draft is saved
for correction through `GET/PATCH /api/v1/images/<id>/draft` with
`expected_revision`; finalization takes `expected_revision` and `request_key`.
Required missing values block finalization, rather than acquiring made-up defaults.

Receipt review uses checks version 2. See the
[amount policy and human acknowledgement contract](UPGRADE_AND_VERIFY.md#review-and-financial-contract).
Verify uses current operation/review revisions and exact finding IDs/comments;
integrity failures cannot be acknowledged away. Bypass verifies only clean current
checks with identified originals; it does not create human-qualified Gold.
Verification does not move original or working files.

Resources and mutations are owner-scoped; admins cross owners only with
`AUTH__ADMIN_CAN_SEE_ALL=true`. Account switches invalidate old UI requests,
queries, notices and view state. Deletion first commits authorized database removal
and a durable owned-file manifest, then removes validated originals, thumbnails
and working derivatives. Failed cleanup remains pending; repeat the same authorized
DELETE to finish it. Referenced receipts/active submissions conflict. See the
[upgrade and recovery runbook](UPGRADE_AND_VERIFY.md).
