from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
LITERATURE_ROOT = ROOT / "docs" / "literature"
MATRIX_COLUMNS = [
    "evidence_id",
    "full_reference",
    "year",
    "source_type",
    "problem",
    "domain",
    "data",
    "degradation",
    "scale",
    "method",
    "metrics",
    "reported_results",
    "limitations",
    "code_url",
    "checkpoint_identity",
    "licence",
    "relevance_to_tfg",
    "doi_or_official_url",
    "verification_date",
    "screening_status",
    "notes",
]
CLAIM_COLUMNS = [
    "schema_version",
    "record_type",
    "claim_id",
    "chapter_section",
    "status",
    "evidence_ids",
    "limitations",
    "reviewer",
    "review_date",
]
SCREENING_COLUMNS = [
    "evidence_id",
    "discovered_via",
    "full_reference",
    "year",
    "source_type",
    "doi_or_official_url",
    "canonical_identity",
    "duplicate_of",
    "screening_status",
    "screening_reason",
    "verification_date",
    "notes",
]
REQUIRED_LAYERS = {
    "classical_interpolation",
    "fidelity_sr",
    "perceptual_sr",
    "cnn_foundation",
    "transformer_foundation",
    "document_or_score",
    "controlled_degradation",
    "real_degradation",
    "evaluation",
    "semantic_risk",
    "current_2026",
}
PRIMARY_OR_OFFICIAL = {"primary_paper", "official_dataset_documentation"}
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _rows(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (LITERATURE_ROOT / name).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _assert_safe_complete_csv(rows: list[dict[str, str]]) -> None:
    assert rows
    for row in rows:
        assert all(value.strip() for value in row.values()), row
        for value in row.values():
            assert not value.lstrip().startswith(FORMULA_PREFIXES), value


def test_search_protocol_records_a_reproducible_focused_review() -> None:
    protocol = (LITERATURE_ROOT / "search-protocol.md").read_text(encoding="utf-8")
    required_phrases = (
        "## Scope and review questions",
        "## Sources and exact queries",
        "## Search dates",
        "## Citation chasing",
        "## Deduplication",
        "## Inclusion criteria",
        "## Exclusion criteria",
        "## Coverage targets",
        "## Saturation and stop rule",
        "## Known limitations",
        "Phase 3 refresh",
        "Phase 5 synthesis",
    )
    for phrase in required_phrases:
        assert phrase in protocol

    assert "through 2026-08-18" in protocol
    assert "not exhaustive" in protocol.lower()
    assert "primary" in protocol.lower() and "official" in protocol.lower()


def test_sota_matrix_has_exact_d08_shape_and_complete_safe_rows() -> None:
    columns, rows = _rows("sota-matrix.csv")
    assert columns == MATRIX_COLUMNS
    _assert_safe_complete_csv(rows)

    evidence_ids = [row["evidence_id"] for row in rows]
    assert len(evidence_ids) == len(set(evidence_ids))
    for row in rows:
        assert re.fullmatch(r"EVID-[A-Z0-9-]+", row["evidence_id"])
        assert row["source_type"] in PRIMARY_OR_OFFICIAL
        assert row["doi_or_official_url"].startswith("https://")
        assert date.fromisoformat(row["verification_date"]) <= date(2026, 8, 18)
        assert row["screening_status"] == "included"
        assert row["licence"] not in {"unknown", "not_reported"}
        assert row["limitations"] not in {"none", "not_applicable"}
        assert row["relevance_to_tfg"] not in {"none", "not_applicable"}


def test_matrix_covers_every_predeclared_layer_and_landmark_to_current_years() -> None:
    _, rows = _rows("sota-matrix.csv")
    layers: set[str] = set()
    years: set[int] = set()
    for row in rows:
        years.add(int(row["year"]))
        match = re.search(r"(?:^|;)coverage_layers=([^;]+)", row["notes"])
        assert match, row["evidence_id"]
        layers.update(match.group(1).split("|"))

    assert REQUIRED_LAYERS <= layers
    assert min(years) <= 1981
    assert 2026 in years


def test_screening_log_preserves_inclusions_duplicates_and_exclusions() -> None:
    columns, rows = _rows("screening-log.csv")
    assert columns == SCREENING_COLUMNS
    _assert_safe_complete_csv(rows)

    statuses = {row["screening_status"] for row in rows}
    assert {"included", "excluded", "duplicate", "discovery_only"} <= statuses
    identities = {row["evidence_id"] for row in rows}
    for row in rows:
        assert row["canonical_identity"]
        if row["screening_status"] == "duplicate":
            assert row["duplicate_of"] in identities
        else:
            assert row["duplicate_of"] == "not_applicable"
        if row["screening_status"] == "excluded":
            assert row["screening_reason"] not in {"included", "not_applicable"}


def test_claim_candidates_resolve_to_primary_or_official_matrix_evidence() -> None:
    matrix_columns, matrix_rows = _rows("sota-matrix.csv")
    claim_columns, claims = _rows("claim-candidates.csv")
    assert matrix_columns == MATRIX_COLUMNS
    assert claim_columns == CLAIM_COLUMNS
    assert claims

    evidence = {row["evidence_id"]: row for row in matrix_rows}
    for claim in claims:
        assert claim["schema_version"] == "1"
        assert claim["record_type"] == "claim-evidence"
        assert claim["status"] in {"draft", "pending", "reviewed", "rejected"}
        assert claim["limitations"]
        ids = claim["evidence_ids"].split("|")
        assert ids and all(evidence_id in evidence for evidence_id in ids)
        assert all(
            evidence[evidence_id]["source_type"] in PRIMARY_OR_OFFICIAL for evidence_id in ids
        )
        if claim["status"] == "reviewed":
            assert claim["reviewer"] and claim["reviewer"] != "not_assigned"
            date.fromisoformat(claim["review_date"])


def test_review_does_not_select_models_or_claim_smb_outcomes() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            LITERATURE_ROOT / "search-protocol.md",
            LITERATURE_ROOT / "sota-matrix.csv",
            LITERATURE_ROOT / "claim-candidates.csv",
        )
    ).lower()
    prohibited = (
        "selected model",
        "chosen checkpoint",
        "recommended checkpoint",
        "smb achieved",
        "smb result",
        "smb psnr",
        "smb ssim",
    )
    for phrase in prohibited:
        assert phrase not in text


@pytest.mark.parametrize(
    "filename",
    ("screening-log.csv", "sota-matrix.csv", "claim-candidates.csv"),
)
def test_csv_files_have_no_embedded_nul_or_carriage_return(filename: str) -> None:
    payload = (LITERATURE_ROOT / filename).read_bytes()
    assert b"\x00" not in payload
    assert b"\r" not in payload
