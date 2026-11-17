"""Explicit dedication of an operator's existing external inference endpoint."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic

from ..model.handoff import HandoffBinding, HandoffStatus, HandoffWait
from ..service.gpu_handoff import ModelHandoff, current_owner
from .bootstrap import write_private


async def adopt_binding(
    handoff: ModelHandoff,
    model_ids: Sequence[str],
    *,
    dedicated_coordinated_requests: bool,
    external_eager_loading_disabled: bool,
    deadline: float,
) -> HandoffBinding:
    """Record explicit operator attestations only while all owned state is idle."""
    if dedicated_coordinated_requests is not True or external_eager_loading_disabled is not True:
        raise ValueError("model_handoff_explicit_attestations_required")
    binding = HandoffBinding(
        coordination_path=str(handoff.path.resolve()), resources=handoff.resources,
        model_ids=list(model_ids), dedicated_coordinated_requests=True,
        external_eager_loading_disabled=True,
    )
    if not handoff.provider.supports_model_release:
        raise ValueError("model_release_unsupported")
    if current_owner() is not None:
        raise HandoffWait(HandoffStatus(state="busy", reason_code="model_handoff_busy"))
    async with asyncio.timeout(max(0, deadline - monotonic())):
        async with handoff.exclusion(deadline, wait=False):
            if binding.resources != handoff.resources or binding.coordination_path != str(handoff.path.resolve()):
                raise ValueError("model_handoff_identity_changed")
            saved = handoff.read_journal()
            if saved is not None and (saved.unsettled or saved.ocr_pending or saved.models):
                raise HandoffWait(HandoffStatus(
                    state="recovery_required", reason_code="model_handoff_journal_not_clean",
                ))
            # Existing bindings also fence endpoint/path changes and malformed state.
            if handoff.binding_path.exists():
                handoff.read_binding()
            await handoff.require_residency([], deadline)
            write_private(handoff.binding_path, binding.model_dump_json() + "\n", replace=True)
    return binding


async def _run(args: argparse.Namespace) -> int:
    from ..config import settings
    from ..provider.factory import get_llm_provider, get_ocr_provider

    provider = None
    try:
        provider = get_llm_provider(settings.llm)
        ocr = get_ocr_provider(settings.ocr)
        if ocr is None or not ocr.managed_lifecycle:
            status = HandoffStatus(state="configuration_required", reason_code="model_handoff_managed_ocr_required")
        else:
            handoff = ModelHandoff(args.handoff_lock or settings.ocr.coordination_path, provider, ocr)
            deadline = monotonic() + 30
            if args.command == "adopt":
                await adopt_binding(
                    handoff, args.model_id,
                    dedicated_coordinated_requests=args.dedicated_coordinated_requests,
                    external_eager_loading_disabled=args.external_eager_loading_disabled,
                    deadline=deadline,
                )
                status = HandoffStatus(state="ready")
            else:
                status = await handoff.inspect(model_id=args.model_id or settings.llm.model_name, deadline=deadline)
        print(status.model_dump_json())
        return 0 if status.state == "ready" else 1
    except HandoffWait as exc:
        print(exc.status.model_dump_json())
        if exc.status.reason_code == "model_handoff_foreign_residency":
            print("Quiesce the external endpoint using its own controls, then retry with zero resident models.", file=sys.stderr)
        return 1
    except (ValueError, OSError, TimeoutError):
        # Endpoint URLs and SDK exceptions can contain configured credentials.
        print(json.dumps({"state": "recovery_required", "reason_code": "model_handoff_setup_failed"}))
        return 1
    finally:
        client = getattr(provider, "_client", None)
        if client is not None:
            if hasattr(client, "aclose"):
                await client.aclose()
            elif hasattr(client, "close"):
                await client.close()
            elif hasattr(client, "_client"):
                await client._client.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "adopt"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--handoff-lock", type=Path)
        sub.add_argument("--model-id", action="append" if command == "adopt" else "store", required=command == "adopt")
        if command == "adopt":
            sub.add_argument("--dedicated-coordinated-requests", action="store_true", required=True)
            sub.add_argument("--external-eager-loading-disabled", action="store_true", required=True)
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
