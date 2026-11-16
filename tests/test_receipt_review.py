"""Exact financial review: printed values and evidence are never rewritten."""

from decimal import Decimal
from uuid import UUID

import pytest

from vision_bill.model.extraction import ExtractionDraft
from vision_bill.model.ocr import SourceRef
from vision_bill.model.receipt import Receipt
from vision_bill.model.review import AmountPolicy, FindingAcknowledgement, OriginalAvailability
from vision_bill.service import receipt_review as review

PART = UUID(int=1)
ORIGINALS = [OriginalAvailability(part_id=PART, available=True, sha256_matches=True)]
POLICY = AmountPolicy(
    tax_basis="gross",
    discount_treatment="included_in_lines",
    tip_treatment="none",
    subtotal_origin="printed",
    tax_information="present",
)


def document(**changes):
    data = {
        "confidence": 90,
        "merchant_name": "Shop",
        "date": "2026-01-01",
        "currency": "EUR",
        "subtotal": "10",
        "total": "10",
        "tax_total": "1.60",
        "taxes": [{"name": "VAT", "amount": "1.60"}],
        "line_items": [
            {"description": "Item", "quantity": 2, "unit_price": "5", "total_price": "10"}
        ],
    }
    return Receipt.model_validate(dict(data, **changes))


def evaluate(doc=None, policy=POLICY, sources=None, overlaps=None, originals=ORIGINALS):
    assert hasattr(review, "evaluate_receipt"), "pure financial evaluator is missing"
    return review.evaluate_receipt(
        doc or document(), policy, sources or {}, overlaps or [], originals
    )


@pytest.mark.parametrize("basis,total", [("gross", "10"), ("net", "11.60")])
def test_tax_basis_does_not_double_add_included_tax(basis, total):
    result = evaluate(document(total=total), POLICY.model_copy(update={"tax_basis": basis}))
    assert result.can_auto_verify and result.findings == []


@pytest.mark.parametrize(
    "discount,tip_policy,tip,total",
    [
        ("included_in_lines", "included", "2", "10"),
        ("subtract", "add", "2", "11"),
        ("included_in_lines", "outside_receipt", "2", "10"),
    ],
)
def test_discount_and_tip_semantics(discount, tip_policy, tip, total):
    result = evaluate(
        document(discount_total="1", tip=tip, total=total),
        POLICY.model_copy(update={"discount_treatment": discount, "tip_treatment": tip_policy}),
    )
    assert result.can_auto_verify


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("total", "10.05", None),
        ("total", "10.0501", "amount_discrepancy"),
        ("tax_total", "1.62", None),
        ("tax_total", "1.6201", "tax_discrepancy"),
    ],
)
def test_exact_boundary_tolerances(field, value, code):
    result = evaluate(document(**{field: value}))
    assert [f.code for f in result.findings] == ([] if code is None else [code])


@pytest.mark.parametrize("value,expected", [("10.01", []), ("10.0101", ["line_discrepancy"])])
def test_line_tolerance(value, expected):
    doc = document(
        line_items=[{"description": "Item", "quantity": 2, "unit_price": "5", "total_price": value}]
    )
    assert [f.code for f in evaluate(doc).findings] == expected


def test_negative_refunds_and_repeated_rows_preserved():
    rows = [
        {
            "description": "Return",
            "quantity": 1,
            "unit_price": "-5.00001",
            "total_price": "-5.00001",
        }
    ] * 2
    doc = document(subtotal="-10.00002", total="-10.00002", line_items=rows)
    before = doc.model_dump_json()
    assert evaluate(doc).can_auto_verify
    assert doc.model_dump_json() == before
    assert len(doc.line_items) == 2


def test_nulls_are_integrity_and_do_not_gain_defaults():
    draft = ExtractionDraft.model_validate(document().model_dump())
    draft.currency = None
    result = evaluate(draft)
    assert not result.can_finalize and not result.can_auto_verify
    assert any(f.severity == "integrity" and f.field_path == "currency" for f in result.findings)
    assert draft.currency is None


@pytest.mark.parametrize(
    "originals",
    [
        [],
        [OriginalAvailability(part_id=PART, available=False, sha256_matches=True)],
        [OriginalAvailability(part_id=PART, available=True, sha256_matches=False)],
    ],
)
def test_missing_or_changed_original_cannot_finalize(originals):
    assert not evaluate(originals=originals).can_finalize


def test_unknown_mapping_overlap_policy_remain_reviewable():
    ref = SourceRef(part_id=PART, photo_index=0, source_sha256="a" * 64)
    result = evaluate(
        policy=AmountPolicy(),
        sources={"total": [ref]},
        overlaps=[{"first_index": 0, "second_index": 1, "verified": False, "removed_index": None}],
    )
    codes = {f.code for f in result.findings}
    assert {
        "unknown_tax_basis",
        "unknown_subtotal_origin",
        "missing_tax_information",
        "unknown_source_mapping",
        "ambiguous_overlap",
    } <= codes
    assert result.can_finalize and not result.can_auto_verify


def test_stable_ids_bind_whole_material_document_but_ignore_classification():
    first = evaluate(document(total="12"))
    assert first.findings[0].id == evaluate(document(total=Decimal("12.000"))).findings[0].id
    assert (
        first.findings[0].id != evaluate(document(total="12", merchant_name="Other")).findings[0].id
    )
    assert first.findings[0].id == evaluate(document(total="12", category="grocery")).findings[0].id


@pytest.mark.parametrize("first,equivalent,different", [
    ("14:30", "14:30:00.0", "14:30:01"),
    ("14:30:15.1", "14:30:15.100000", "14:30:15.100001"),
    ("14:30:00.000001", "14:30:00.000001", "14:30:00.000002"),
])
def test_time_canonicalization_keeps_material_and_finding_ids_stable(first, equivalent, different):
    left = document(total="12", time=first)
    right = document(total="12", time=equivalent)
    changed = document(total="12", time=different)
    assert review.material_document(left) == review.material_document(right)
    assert [f.id for f in evaluate(left).findings] == [f.id for f in evaluate(right).findings]
    assert review.material_document(left) != review.material_document(changed)
    assert evaluate(left).findings[0].id != evaluate(changed).findings[0].id


def test_acknowledgements_exact_and_integrity_not_acknowledgeable():
    assert hasattr(review, "validate_acknowledgements"), "acknowledgement gate missing"
    result = evaluate(document(total="12"))
    ack = FindingAcknowledgement(
        finding_id=result.findings[0].id, comment="Printed total confirmed"
    )
    review.validate_acknowledgements(result, [ack])
    for bad in (
        [],
        [ack, ack],
        [ack.model_copy(update={"finding_id": "stale"})],
        [ack.model_copy(update={"comment": "  "})],
    ):
        with pytest.raises(ValueError):
            review.validate_acknowledgements(result, bad)
    integrity = evaluate(originals=[])
    with pytest.raises(ValueError):
        review.validate_acknowledgements(
            integrity,
            [FindingAcknowledgement(finding_id=f.id, comment="allow") for f in integrity.findings],
        )


def test_automatic_submission_gate_uses_real_financial_checks():
    from vision_bill.model.extraction import (
        InferenceEnvelope,
        InferenceTelemetry,
        PipelineProvenance,
    )
    from vision_bill.service.submission_service import SubmissionService

    envelope = InferenceEnvelope(
        receipt=document(),
        amount_policy=POLICY,
        provenance=PipelineProvenance(
            requested_pipeline="direct_vision",
            used_pipeline="direct_vision",
            source_image_count=1,
            ordered_source_sha256=["a" * 64],
            model_id="test",
        ),
        telemetry=InferenceTelemetry(elapsed_ms=1, max_output_tokens=100),
    )
    result = SubmissionService._outcome(envelope, [], ORIGINALS)
    assert result.review.can_auto_verify
    envelope.receipt.total = Decimal(20)
    result = SubmissionService._outcome(envelope, [], ORIGINALS)
    assert not result.review.can_auto_verify
    assert [f.code for f in result.review.findings] == ["amount_discrepancy"]


def test_foreign_source_reference_is_integrity_even_with_known_coordinates():
    ref = SourceRef(
        part_id=UUID(int=999), photo_index=0, source_sha256="a" * 64, coordinate_space="original"
    )
    result = evaluate(sources={"total": [ref]})
    assert not result.can_finalize
    assert any(f.code == "source_integrity_failed" for f in result.findings)


def test_upstream_finding_identity_is_bound_to_current_material():
    from vision_bill.model.review import ReviewFinding

    finding = ReviewFinding(
        id="provider-id", code="ocr_uncertain", severity="review", field_path="merchant_name"
    )
    assert (
        "retained_findings" in __import__("inspect").signature(review.evaluate_receipt).parameters
    )
    first = review.evaluate_receipt(
        document(), POLICY, {}, [], ORIGINALS, retained_findings=[finding]
    )
    second = review.evaluate_receipt(
        document(merchant_name="Other"), POLICY, {}, [], ORIGINALS, retained_findings=[finding]
    )
    assert not first.can_auto_verify and first.can_finalize
    assert first.findings[0].id != "provider-id" and first.findings[0].id != second.findings[0].id


def test_upstream_integrity_failure_is_not_discarded_as_recalculated_code():
    from vision_bill.model.review import ReviewFinding

    finding = ReviewFinding(
        id="external", code="source_integrity_failed", severity="integrity", field_path="originals"
    )
    result = review.evaluate_receipt(
        document(), POLICY, {}, [], ORIGINALS, retained_findings=[finding]
    )
    assert not result.can_finalize and not result.can_auto_verify


def test_finding_ids_and_exposed_checks_version_change_together(monkeypatch):
    before = evaluate(document(total="12"))
    monkeypatch.setattr(review, "CHECKS_VERSION", "next")
    after = evaluate(document(total="12"))
    assert after.checks_version == "next"
    assert after.findings[0].id != before.findings[0].id


def test_approved_overlap_reviews_final_receipt_without_reinterpreting_raw_rows():
    from vision_bill.model.extraction import (
        InferenceEnvelope,
        InferenceTelemetry,
        PipelineProvenance,
    )
    from vision_bill.service.submission_service import SubmissionService

    raw = ExtractionDraft.model_validate(document().model_dump())
    raw.line_items.append(raw.line_items[0].model_copy(deep=True))
    envelope = InferenceEnvelope(
        receipt=document(),
        draft=raw,
        amount_policy=POLICY,
        overlap_decisions=[
            {"first_index": 0, "second_index": 1, "verified": True, "removed_index": 1}
        ],
        provenance=PipelineProvenance(
            requested_pipeline="direct_vision",
            used_pipeline="direct_vision",
            source_image_count=1,
            ordered_source_sha256=["a" * 64],
            model_id="test",
        ),
        telemetry=InferenceTelemetry(elapsed_ms=1, max_output_tokens=100),
    )
    result = SubmissionService._outcome(envelope, [], ORIGINALS)
    assert result.review.can_auto_verify
    assert len(envelope.draft.line_items) == 2 and len(envelope.receipt.line_items) == 1
    envelope.draft.currency = None
    incomplete = SubmissionService._outcome(envelope, [], ORIGINALS)
    assert not incomplete.review.can_finalize and incomplete.kind == "partial"


async def test_legacy_availability_does_not_fabricate_persisted_part_ids(tmp_path):
    from vision_bill.model.db.image import ImagePart
    from vision_bill.service.image_service import ImageService

    paths = [tmp_path / "first", tmp_path / "second"]
    for path in paths:
        path.write_bytes(b"legacy original")
    result = await ImageService.check_originals(
        [ImagePart(image_path=str(path), photo_index=i) for i, path in enumerate(paths)]
    )
    assert [entry.part_id for entry in result] == [None, None]
    assert [entry.photo_index for entry in result] == [0, 1]
    assert all(entry.available for entry in result)


def test_readable_legacy_original_is_human_only_not_automatic_bypass():
    legacy = OriginalAvailability(part_id=None, photo_index=0, available=True, sha256_matches=True)
    result = evaluate(originals=[legacy])
    assert result.can_finalize
    assert not result.can_auto_verify
    review.validate_acknowledgements(result, [])
