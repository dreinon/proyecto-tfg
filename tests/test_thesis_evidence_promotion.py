from __future__ import annotations

import csv
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from score_super_resolution.contracts import ContractValidationError, validate_instance

ROOT = Path(__file__).parents[1]
CLAIM_PATH = ROOT / "docs" / "claim-evidence.csv"
MATRIX_PATH = ROOT / "docs" / "literature" / "sota-matrix.csv"
CLAIM_COLUMNS = (
    "schema_version",
    "record_type",
    "claim_id",
    "chapter_section",
    "status",
    "evidence_ids",
    "limitations",
    "reviewer",
    "review_date",
)
PRIMARY_OR_OFFICIAL = {"primary_paper", "official_dataset_documentation"}
PROMOTED_CLAIM_IDS = {
    "SOTA-CNN-FOUNDATION",
    "SOTA-CURRENT-GENERATIVE-RISK",
    "SOTA-DEGRADATION-BOUNDARY",
    "SOTA-DIRECT-SCORE-EVIDENCE",
    "SOTA-DOCUMENT-SCOPE",
    "SOTA-INTERPOLATION-ROLE",
    "SOTA-PERCEPTION-DISTORTION",
    "SOTA-TRANSFORMER-PROGRESSION",
}


def _csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _typed_claim(row: dict[str, str]) -> dict[str, object]:
    return {
        **row,
        "schema_version": int(row["schema_version"]),
        "evidence_ids": row["evidence_ids"].split("|") if row["evidence_ids"] else [],
    }


def _evidence_registry() -> dict[str, dict[str, str]]:
    _, rows = _csv_rows(MATRIX_PATH)
    return {row["evidence_id"]: row for row in rows}


def _promotable_claims(
    rows: list[dict[str, str]], evidence: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    promoted = []
    for row in rows:
        typed = _typed_claim(row)
        validate_instance("claim-evidence", typed)
        if row["status"] != "reviewed":
            continue

        for evidence_id in typed["evidence_ids"]:
            assert isinstance(evidence_id, str)
            support = evidence.get(evidence_id)
            if support is None:
                raise ValueError(f"unresolved reviewed evidence ID: {evidence_id}")
            if support["screening_status"] != "included":
                raise ValueError(f"reviewed evidence is not included: {evidence_id}")
            if support["source_type"] not in PRIMARY_OR_OFFICIAL:
                raise ValueError(f"reviewed evidence is not primary or official: {evidence_id}")
        promoted.append(typed)
    return promoted


def test_claim_ledger_has_exact_header_and_every_row_passes_shared_contract() -> None:
    columns, rows = _csv_rows(CLAIM_PATH)

    assert columns == CLAIM_COLUMNS
    assert rows
    for row in rows:
        assert None not in row
        validate_instance("claim-evidence", _typed_claim(row))


def test_only_reviewed_resolved_primary_or_official_claims_are_promotable() -> None:
    _, rows = _csv_rows(CLAIM_PATH)
    promoted = _promotable_claims(rows, _evidence_registry())

    assert {claim["claim_id"] for claim in promoted} == PROMOTED_CLAIM_IDS
    for claim in promoted:
        assert claim["limitations"]
        assert claim["reviewer"]
        assert date.fromisoformat(str(claim["review_date"])) <= date(2026, 8, 18)


def test_pending_and_draft_rows_cannot_be_selected_for_promotion() -> None:
    _, rows = _csv_rows(CLAIM_PATH)
    promoted_ids = {claim["claim_id"] for claim in _promotable_claims(rows, _evidence_registry())}

    unreviewed_ids = {row["claim_id"] for row in rows if row["status"] in {"draft", "pending"}}
    assert unreviewed_ids
    assert promoted_ids.isdisjoint(unreviewed_ids)


@pytest.mark.parametrize("field", ("evidence_ids", "limitations", "reviewer", "review_date"))
def test_incomplete_reviewed_claims_fail_the_shared_contract(field: str) -> None:
    _, rows = _csv_rows(CLAIM_PATH)
    reviewed = next(row for row in rows if row["status"] == "reviewed")
    incomplete = _typed_claim(reviewed)
    incomplete[field] = [] if field == "evidence_ids" else ""

    with pytest.raises(ContractValidationError):
        validate_instance("claim-evidence", incomplete)


def test_unknown_claim_column_fails_the_shared_contract() -> None:
    _, rows = _csv_rows(CLAIM_PATH)
    unknown = _typed_claim(rows[0])
    unknown["unknown_column"] = "not allowed"

    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("claim-evidence", unknown)


def test_unresolved_reviewed_evidence_id_blocks_promotion() -> None:
    _, rows = _csv_rows(CLAIM_PATH)
    reviewed = deepcopy(next(row for row in rows if row["status"] == "reviewed"))
    reviewed["evidence_ids"] = "EVID-NOT-IN-REGISTRY"

    with pytest.raises(ValueError, match="unresolved reviewed evidence ID"):
        _promotable_claims([reviewed], _evidence_registry())
