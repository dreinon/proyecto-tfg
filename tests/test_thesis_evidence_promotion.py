from __future__ import annotations

import csv
import shutil
from collections import Counter
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
import yaml

from score_super_resolution import smb_audit
from score_super_resolution.contracts import ContractValidationError, validate_instance

ROOT = Path(__file__).parents[1]
CLAIM_PATH = ROOT / "docs" / "claim-evidence.csv"
MATRIX_PATH = ROOT / "docs" / "literature" / "sota-matrix.csv"
ACTIVE_MANIFEST_PATH = ROOT / "data" / "manifests" / "smb-evaluation-v1.yaml"
RECOVERY_DESCRIPTOR_PATH = ROOT / "data" / "manifests" / "smb-evaluation-v1-recovery.yaml"
RECOVERY_RECORDS_PATH = ROOT / "data" / "manifests" / "smb-evaluation-v1-recovery.jsonl.gz"
AUDIT_PATH = ROOT / "data" / "audits" / "smb-audit-v1.yaml"
SAMPLE_PATH = ROOT / "data" / "audits" / "smb-visual-sample-v1.csv"
REVIEW_PATH = ROOT / "data" / "audits" / "smb-review-v1.csv"
DATA_CHAPTER_PATH = ROOT.parent / "memoria" / "chapters" / "04-datos.tex"
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

FORBIDDEN_SMB_OVERCLAIMS = (
    "685 páginas revisadas visualmente",
    "621 páginas no muestreadas son idóneas",
    "procedencia individual confirmada",
    "redistribución por elemento permitida",
    "figuras de SMB permitidas",
    "resultados de superresolución sobre SMB",
    "EVALUATION_UNLOCKED",
)


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


def _resolved_tracked_smb_evidence(
    tmp_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    active_copy = tmp_path / "smb-evaluation-v1.yaml"
    shutil.copyfile(ACTIVE_MANIFEST_PATH, active_copy)
    generation_root = tmp_path / "recovered-generations"

    smb_audit.recover_active_manifest(
        active_path=active_copy,
        recovery_descriptor_path=RECOVERY_DESCRIPTOR_PATH,
        recovery_records_path=RECOVERY_RECORDS_PATH,
        generation_root=generation_root,
    )
    return smb_audit.resolve_active_manifest(
        active_path=active_copy,
        generation_root=generation_root,
    )


def _assert_bounded_smb_narrative(text: str) -> None:
    normalized = " ".join(text.split())
    required_fragments = (
        "96332e8c4ac81cbdb7f61093ec5a4bfff76a0adb",
        "685 filas",
        "681",
        "cuatro",
        "260 grupos",
        "64 páginas",
        "621 páginas",
        "14 pares",
        "ningún par exacto",
        "CC BY-NC 4.0",
        "procedencia por elemento no está disponible",
        "redistribución no queda establecida",
        "reproducción de figuras permanece prohibida",
        "881d576e604ec9faef73ae1a302222cd21575116505f2146bfe2b1f45ffcde38",
        "59c038b4105b0df81878b1feec30155b52602371ace48a9044a7f7c8edad6e30",
        "AUDITED\\_LOCKED",
        "ACAD-03",
    )
    for fragment in required_fragments:
        assert fragment in normalized
    for forbidden in FORBIDDEN_SMB_OVERCLAIMS:
        assert forbidden not in normalized


def test_data_chapter_matches_recoverable_smb_v2_evidence(tmp_path: Path) -> None:
    descriptor, rows = _resolved_tracked_smb_evidence(tmp_path)
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))
    recovery = yaml.safe_load(RECOVERY_DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    _, sample_rows = _csv_rows(SAMPLE_PATH)
    _, review_rows = _csv_rows(REVIEW_PATH)

    processing = Counter(row["processing_status"] for row in rows)
    visual = Counter(str(row["visual_review"]["status"]) for row in rows)
    paired_reasons = Counter(
        str(row["paired_ineligibility_reason"]) for row in rows if row["paired_eligible"] is False
    )
    exact_pairs = {
        relation["pair_id"]
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "exact"
    }
    perceptual_pairs = {
        relation["pair_id"]: relation["disposition"]
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "perceptual"
    }
    review_kinds = Counter(row["review_kind"] for row in review_rows)

    assert descriptor["generation_id"] == recovery["generation_id"]
    assert descriptor["records_sha256"] == recovery["records_sha256"]
    assert descriptor["source_revision"] == recovery["source_revision"]
    assert descriptor["source_provenance"] == recovery["source_provenance"]
    assert descriptor["benchmark_state"] == recovery["benchmark_state"] == "AUDITED_LOCKED"
    assert descriptor["row_count"] == recovery["row_count"] == audit["row_count"] == 685
    assert len(rows) == 685
    assert processing == {"processed": 685}
    assert sum(row["paired_eligible"] is True for row in rows) == 681
    assert paired_reasons == {"invalid_region_annotation": 4}
    assert len({row["source_group_id"] for row in rows}) == 260
    assert visual == {"sampled_human_reviewed": 64, "not_visually_reviewed": 621}
    assert sum(row["audit_sample_member"] is True for row in rows) == len(sample_rows) == 64
    assert exact_pairs == set()
    assert len(perceptual_pairs) == 14
    assert set(perceptual_pairs.values()) == {"distinct"}
    assert review_kinds == {"item_policy": 685, "visual_item": 64, "duplicate_pair": 14}
    assert all(
        row["rights"]["dataset_licence"]["identifier"] == "CC-BY-NC-4.0"
        and row["rights"]["access_status"] == "confirmed"
        and row["rights"]["item_provenance"]["status"] == "unavailable"
        and row["rights"]["redistribution"]["status"] == "not_established"
        and row["rights"]["figure_reproduction"]["status"] == "prohibited"
        for row in rows
    )

    _assert_bounded_smb_narrative(DATA_CHAPTER_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("overclaim", FORBIDDEN_SMB_OVERCLAIMS)
def test_data_chapter_boundary_rejects_smb_overclaims(overclaim: str) -> None:
    chapter = DATA_CHAPTER_PATH.read_text(encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_bounded_smb_narrative(f"{chapter}\n{overclaim}\n")


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
