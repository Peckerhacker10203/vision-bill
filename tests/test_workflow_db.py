"""Public contract and artifact boundary tests (no database needed)."""

import importlib.util
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vision_bill.model import workflow


def test_transaction_provider_exists():
    assert importlib.util.find_spec("vision_bill.provider.db.workflow_db") is not None


def test_artifact_boundary_requires_saved_path():
    assert hasattr(workflow, "PersistedWorkingView"), "typed saved artifact contract missing"
    with pytest.raises(ValidationError):
        workflow.PersistedWorkingView(  # type: ignore[attr-defined]
            part_id=uuid4(),
            photo_index=0,
            source_sha256="a" * 64,
            metadata={
                "view_id": str(uuid4()),
                "media_type": "image/png",
                "width": 1,
                "height": 1,
                "sha256": "b" * 64,
            },
        )


def test_outcome_cannot_complete_without_receipt_and_review():
    with pytest.raises(ValidationError):
        workflow.AttemptOutcome(kind="complete")


def test_raw_png_in_generic_submission_metadata_is_rejected():
    with pytest.raises(ValidationError):
        workflow.SubmissionCreate(
            id=uuid4(),
            parts=[{"image_path": "/tmp/original"}],
            source_manifest=[{"photo_index": 0, "source_sha256": "a" * 64}],
            requested_options={"png_base64": "secret"},
            effective_options={},
            request_fingerprint="fingerprint",
        )
