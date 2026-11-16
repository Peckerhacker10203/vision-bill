# Vision Bill frontend

SvelteKit/TypeScript SPA with static adapter, Tailwind, TanStack Query and EN/DE
localization. FastAPI serves the built SPA with client-route fallback; `/api`
keeps API routing. Docker builds the frontend from source.

From the repository root:

```bash
make fe-install       # npm ci, including the required patch-package postinstall
make fe-dev           # :5173, /api proxy to :8080
make fe-verify        # Svelte/ESLint/Prettier, Vitest, E2E types, production build
make fe-sync          # build and copy frontend/out into src/vision_bill/static
make fe-test-e2e      # complete Playwright run in a unique owned app/stub/DB stack
```

`npm run preview` inside `frontend` previews the static build. It still needs an
API endpoint; preview does not start PostgreSQL or inference. `make fe` exposes
other package scripts, for example `make fe CMD=typecheck:e2e`.

Routes cover login, uploads, queue, receipt/draft review, collections, search,
statistics, benchmarks and settings. Uploads always use durable 202/status polling;
`/upload?job=<id>` resumes after reload. Review preserves exact decimal strings,
ordered row identities, original/current descriptions and local receipt times
including valid seconds/microseconds. Source views distinguish original and working
coordinates; unknown inverse transforms remain null. EN/DE findings explain checks2,
readiness waits and fallback outcomes. See [workflow](../MULTI_IMAGE_RECEIPTS.md).

Auth requests are serialized and guarded by session generation. Account changes
cancel/clear queries and remount authenticated views; delayed replies cannot attach
to a successor account/view. Collection, benchmark-result and tag-edit mutations
are bound to their originating owner/view. Backend ownership remains authoritative.
Benchmark model capability depends on pipeline: direct vision needs vision; OCR
uses text formatting (including eligible multimodal models); upload follows OCR
configuration. Gold selection requires explicit current human qualification.

Keep private receipts, traces, Playwright outputs and local dotenv files out of Git
and builds. `.dockerignore` excludes generated test reports and credentials;
[the shell patch](patches/README.md) remains a required build input.
