"""Conservative admission boundary; full financial review policy follows separately."""

from collections.abc import Sequence

from pydantic import ValidationError

from ..model.extraction import ExtractionDraft
from ..model.receipt import Receipt
from ..model.review import ReviewEvaluation, ReviewFinding


def receipt_from_draft(document: ExtractionDraft) -> Receipt:
    """Pass explicit nulls through validation; never substitute Receipt defaults."""
    try:
        receipt = Receipt.model_validate(document.model_dump())
    except ValidationError as exc:
        raise ValueError("draft_incomplete") from exc
    if not receipt.merchant_name.strip() or any(
        not item.description.strip() for item in receipt.line_items
    ):
        raise ValueError("draft_incomplete")
    return receipt


def admission_review(
    document: ExtractionDraft,
    *,
    originals_valid: bool,
    findings: Sequence[ReviewFinding] = (),
) -> ReviewEvaluation:
    complete = True
    try:
        receipt_from_draft(document)
    except ValueError:
        complete = False
    return ReviewEvaluation(
        findings=list(findings),
        can_finalize=complete
        and originals_valid
        and not any(f.severity == "integrity" for f in findings),
        can_auto_verify=False,
    )
