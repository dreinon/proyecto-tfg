from __future__ import annotations

import csv
import re
from copy import deepcopy
from pathlib import Path

import pytest

from score_super_resolution.contracts import ContractValidationError, validate_instance

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
DECISION_PATH = DOCS / "decision-log.md"
DEVIATION_PATH = DOCS / "deviation-log.md"
CLAIM_PATH = DOCS / "claim-evidence.csv"
GATE_PATH = DOCS / "checkpoints" / "phase-01-human-gates.md"
EMAIL_PATH = DOCS / "tutor-email-draft.md"

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
MATERIAL_COLUMNS = (
    "Record ID",
    "Date",
    "Owner",
    "Status",
    "Subject",
    "Rationale",
    "Evidence reference",
    "Affected controls/runs",
)
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _markdown_table(path: Path, heading: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    section = text.split(heading, maxsplit=1)[1]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    header = tuple(cell.strip() for cell in lines[0].strip("|").split("|"))
    rows = []
    for line in lines[2:]:
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if len(cells) != len(header):
            break
        rows.append(dict(zip(header, cells, strict=True)))
    return header, rows


def _validate_material_records(rows: list[dict[str, str]]) -> None:
    assert rows
    for row in rows:
        assert tuple(row) == MATERIAL_COLUMNS
        assert all(row[column].strip() for column in MATERIAL_COLUMNS)
        assert row["Evidence reference"].startswith("external-ref:") or row[
            "Evidence reference"
        ].startswith("tracked-ref:")


@pytest.mark.parametrize(
    ("path", "heading"),
    (
        (DECISION_PATH, "## Decision register"),
        (DEVIATION_PATH, "## Pending deviation triggers"),
    ),
)
def test_material_records_are_complete(path: Path, heading: str) -> None:
    columns, rows = _markdown_table(path, heading)

    assert columns == MATERIAL_COLUMNS
    _validate_material_records(rows)
    assert len({row["Record ID"] for row in rows}) == len(rows)


def test_material_record_validation_rejects_missing_accountability() -> None:
    _, rows = _markdown_table(DECISION_PATH, "## Decision register")
    incomplete = deepcopy(rows)
    incomplete[0]["Rationale"] = ""

    with pytest.raises(AssertionError):
        _validate_material_records(incomplete)


def test_decisions_and_deviations_keep_human_outcomes_pending() -> None:
    _, decisions = _markdown_table(DECISION_PATH, "## Decision register")
    _, triggers = _markdown_table(DEVIATION_PATH, "## Pending deviation triggers")

    assert any(row["Record ID"] == "DEC-ACAD-01" for row in decisions)
    assert any(row["Record ID"] == "DEV-ACAD-01" for row in triggers)
    assert all("pending" in row["Status"] for row in decisions + triggers)
    assert "No enacted deviations are recorded" in DEVIATION_PATH.read_text(encoding="utf-8")


def _claim_rows() -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with CLAIM_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _typed_claim(row: dict[str, str]) -> dict[str, object]:
    return {
        **row,
        "schema_version": int(row["schema_version"]),
        "evidence_ids": row["evidence_ids"].split("|") if row["evidence_ids"] else [],
    }


def test_claim_ledger_uses_exact_schema_and_validates_every_row() -> None:
    columns, rows = _claim_rows()

    assert columns == CLAIM_COLUMNS
    assert rows
    for row in rows:
        validate_instance("claim-evidence", _typed_claim(row))
        for value in row.values():
            if value:
                assert not value.lstrip().startswith(FORMULA_PREFIXES), value


def test_unreviewed_claims_are_not_promoted_or_misrepresented() -> None:
    _, rows = _claim_rows()

    assert all(row["status"] in {"draft", "pending"} for row in rows)
    gated = [row for row in rows if "CTRL-05" in row["claim_id"] or "DELV-01" in row["claim_id"]]
    assert gated
    assert all(row["status"] == "pending" for row in gated)
    assert all("does not establish completion" in row["limitations"] for row in gated)


def test_claim_contract_rejects_incomplete_and_unsubstantiated_review() -> None:
    _, rows = _claim_rows()
    incomplete = _typed_claim(rows[0])
    del incomplete["claim_id"]
    with pytest.raises(ContractValidationError):
        validate_instance("claim-evidence", incomplete)

    unsupported = _typed_claim(rows[0])
    unsupported.update(status="reviewed", evidence_ids=[], limitations="", reviewer="", review_date="")
    with pytest.raises(ContractValidationError):
        validate_instance("claim-evidence", unsupported)


def test_human_gate_material_is_private_safe_and_routed_to_closeout() -> None:
    text = GATE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "## Public-rule checks" in text
    assert "## Private human-only checks" in text
    assert "academic-closeout" in lowered
    assert "acad-01" in lowered and "acad-02" in lowered and "acad-03" in lowered
    assert "private evidence stays outside git" in lowered
    assert "pending" in lowered
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    for forbidden in ("hf_token", "password", "credential", "dni", "completed", "approved"):
        assert forbidden not in lowered


def test_tutor_email_is_an_unsent_single_question_about_dataset_compatibility() -> None:
    text = EMAIL_PATH.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "status: draft — not sent" in lowered
    assert "academic-closeout" in lowered
    assert "smb" in lowered
    assert "conjuntos de datos" in lowered
    assert "compatib" in lowered
    assert text.count("¿") == 1 and text.count("?") == 1
    assert "checkpoint" not in lowered
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
