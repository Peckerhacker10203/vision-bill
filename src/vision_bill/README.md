# Application package

FastAPI entrypoint: `main.py`. Routers enforce authentication/ownership, services
coordinate durable submission/review/benchmark workflows, providers own database
and local inference I/O, and models define exact contracts. Schema changes live in
Alembic (current head `0019`), not application startup DDL.

See the [architecture guide](../../AGENTS.md),
[upload/review contract](../../MULTI_IMAGE_RECEIPTS.md) and
[upgrade/verification runbook](../../UPGRADE_AND_VERIFY.md).
