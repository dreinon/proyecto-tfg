from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from score_super_resolution.review_evidence import (
    REVIEW_FIELDS,
    ReviewEvidenceError,
    canonical_review_csv,
    read_review,
    validate_human_cell,
)
from score_super_resolution.smb_review_ui import SMBReviewSession


TRACKED_REVIEW = Path(__file__).parents[1] / "data" / "audits" / "smb-review-v1.csv"
CONTROL_CHARACTERS = tuple(chr(codepoint) for codepoint in (*range(0x20), 0x7F, *range(0x80, 0xA0)))
FORMULA_PREFIXES = ("=2+2", "+2+2", "-2+2", "@SUM(1,1)", "  =2+2", "\v@SUM(1,1)")


def _safe_rows() -> list[dict[str, str]]:
    with TRACKED_REVIEW.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REVIEW_FIELDS
        return list(reader)


def _session(review_path: Path) -> SMBReviewSession:
    session = object.__new__(SMBReviewSession)
    session.review_path = review_path
    return session


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_bytes(canonical_review_csv(rows))


@pytest.mark.parametrize("payload", FORMULA_PREFIXES)
def test_formula_leading_human_cells_are_rejected_without_reflection(payload: str) -> None:
    with pytest.raises(ReviewEvidenceError) as caught:
        validate_human_cell(payload, field="rationale", review_key="smb-test-000000")

    message = str(caught.value)
    assert "rationale" in message
    assert "smb-test-000000" in message
    assert payload not in message


@pytest.mark.parametrize("control", CONTROL_CHARACTERS, ids=lambda value: f"U+{ord(value):04X}")
def test_complete_control_character_table_is_rejected(control: str) -> None:
    payload = f"texto{control}seguro"

    with pytest.raises(ReviewEvidenceError) as caught:
        validate_human_cell(payload, field="reviewer", review_key="smb-test-000000")

    message = str(caught.value)
    assert "reviewer" in message
    assert "smb-test-000000" in message
    assert payload not in message


def test_existing_accented_review_round_trips_to_identical_canonical_bytes() -> None:
    document = read_review(TRACKED_REVIEW)

    assert len(document.rows) == 699
    assert any("Daniel Reinón García" == row["reviewer"] for row in document.rows)
    assert canonical_review_csv(document.rows) == TRACKED_REVIEW.read_bytes()
    assert document.canonical_bytes == TRACKED_REVIEW.read_bytes()


@pytest.mark.parametrize(
    ("reviewer", "rationale"),
    (("=2+2", "Justificación segura"), ("Daniel Reinón García", "@SUM(1,1)")),
)
def test_direct_ui_save_rejects_formula_examples(
    tmp_path: Path, reviewer: str, rationale: str
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    session = _session(review_path)

    with pytest.raises(ReviewEvidenceError):
        session.save_item(
            item_id="smb-test-000000",
            reviewer=reviewer,
            quality_flags=(),
            suitability="suitable",
            rationale=rationale,
        )

    assert review_path.read_bytes() == canonical_review_csv(_safe_rows())


@pytest.mark.parametrize("field", REVIEW_FIELDS)
def test_ui_write_boundary_rejects_unsafe_content_in_every_cell(
    tmp_path: Path, field: str
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    rows = _safe_rows()
    _write_rows(review_path, rows)
    mutated = copy.deepcopy(rows)
    mutated[0][field] = "  =2+2"
    session = _session(review_path)

    with pytest.raises(ReviewEvidenceError, match=field):
        session.write_rows(mutated)

    assert review_path.read_bytes() == canonical_review_csv(rows)


def test_display_neutralization_is_not_valid_canonical_input() -> None:
    assert validate_human_cell(
        "'=2+2", field="rationale", review_key="smb-test-000000"
    ) == "'=2+2"
    with pytest.raises(ReviewEvidenceError):
        validate_human_cell("=2+2", field="rationale", review_key="smb-test-000000")
