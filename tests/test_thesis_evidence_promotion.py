from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from copy import deepcopy
from datetime import date
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from score_super_resolution import smb_audit
from score_super_resolution.contracts import ContractValidationError, validate_instance

ROOT = Path(__file__).parents[1]
CLAIM_PATH = ROOT / "docs" / "claim-evidence.csv"
MATRIX_PATH = ROOT / "docs" / "literature" / "sota-matrix.csv"
ACTIVE_MANIFEST_PATH = ROOT / "data" / "manifests" / "smb-evaluation-v1.yaml"
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
EXPECTED_GENERATION_ID = "75b98a56897a0fbb18ee5580943f6b88b5ef36f7bba408002ac3bd626918d752"
EXPECTED_RECORDS_SHA256 = "56cebb344ae512a3dea16ce4848bce197f971493c98c5c388fe9a094922b820b"
EXPECTED_RECOVERY_BUNDLE_ID = "5782207ba7784fd9fa27e7cd82e319d2fa5d70ae56a674f89d21199895005862"

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


def _active_pointer_and_recovery_paths() -> tuple[dict[str, object], Path, Path]:
    pointer = yaml.safe_load(ACTIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(pointer, dict)
    validate_instance("manifest-active", pointer, version=2)
    project_root = ROOT.resolve()
    selected: list[Path] = []
    for field in ("recovery_descriptor_path", "recovery_records_path"):
        relative = Path(str(pointer[field]))
        assert not relative.is_absolute() and ".." not in relative.parts
        resolved = (ROOT / relative).resolve()
        assert resolved.is_relative_to(project_root) and resolved.is_file()
        selected.append(resolved)
    return pointer, selected[0], selected[1]


def _resolved_tracked_smb_evidence(
    tmp_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _, recovery_descriptor_path, recovery_records_path = _active_pointer_and_recovery_paths()
    active_copy = tmp_path / "smb-evaluation-v1.yaml"
    shutil.copyfile(ACTIVE_MANIFEST_PATH, active_copy)
    generation_root = tmp_path / "recovered-generations"

    smb_audit.recover_active_manifest(
        active_path=active_copy,
        recovery_descriptor_path=recovery_descriptor_path,
        recovery_records_path=recovery_records_path,
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
        "618",
        "67",
        "260 grupos",
        "64 páginas",
        "621 páginas",
        "14 pares",
        "CC BY-NC 4.0",
        "procedencia por elemento no está disponible",
        "redistribución no queda establecida",
        "reproducción de figuras permanece prohibida",
        "AUDITED\\_LOCKED",
        "ACAD-03",
    )
    for fragment in required_fragments:
        assert fragment in normalized
    for forbidden in FORBIDDEN_SMB_OVERCLAIMS:
        assert forbidden not in normalized


def test_data_chapter_matches_recoverable_smb_v2_evidence(tmp_path: Path) -> None:
    descriptor, rows = _resolved_tracked_smb_evidence(tmp_path)
    pointer, recovery_descriptor_path, recovery_records_path = _active_pointer_and_recovery_paths()
    audit = yaml.safe_load(AUDIT_PATH.read_text(encoding="utf-8"))
    recovery = yaml.safe_load(recovery_descriptor_path.read_text(encoding="utf-8"))
    _, sample_rows = _csv_rows(SAMPLE_PATH)
    _, review_rows = _csv_rows(REVIEW_PATH)

    processing = Counter(row["processing_status"] for row in rows)
    visual = Counter(str(row["visual_review"]["status"]) for row in rows)
    paired_reasons = Counter(
        str(row["paired_ineligibility_reason"]) for row in rows if row["paired_eligible"] is False
    )
    represented_exact_pairs = {
        tuple(relation["item_ids"])
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "exact"
    }
    exact_relation_occurrences = Counter(
        relation["pair_id"]
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "exact"
    )
    exact_relations = {
        relation["pair_id"]: relation
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "exact"
    }
    by_framed_hash: dict[str, list[str]] = {}
    for row in rows:
        if row["processing_status"] != "processed":
            continue
        pixel_sha256 = row["image"]["pixel_sha256"]
        assert isinstance(pixel_sha256, str)
        by_framed_hash.setdefault(pixel_sha256, []).append(str(row["item_id"]))
    derived_exact_pairs = {
        pair for item_ids in by_framed_hash.values() for pair in combinations(sorted(item_ids), 2)
    }
    perceptual_pairs = {
        relation["pair_id"]: relation["disposition"]
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["candidate_type"] == "perceptual"
    }
    review_kinds = Counter(row["review_kind"] for row in review_rows)

    assert pointer["generation_id"] == descriptor["generation_id"] == recovery["generation_id"]
    assert pointer["records_sha256"] == descriptor["records_sha256"] == recovery["records_sha256"]
    assert descriptor["generation_id"] == EXPECTED_GENERATION_ID
    assert descriptor["records_sha256"] == EXPECTED_RECORDS_SHA256
    assert recovery["bundle_id"] == EXPECTED_RECOVERY_BUNDLE_ID
    assert (
        pointer["recovery_descriptor_sha256"]
        == hashlib.sha256(recovery_descriptor_path.read_bytes()).hexdigest()
    )
    assert (
        pointer["recovery_records_sha256"]
        == hashlib.sha256(recovery_records_path.read_bytes()).hexdigest()
    )
    assert descriptor["source_revision"] == recovery["source_revision"]
    assert descriptor["source_provenance"] == recovery["source_provenance"]
    assert descriptor["benchmark_state"] == recovery["benchmark_state"] == "AUDITED_LOCKED"
    assert descriptor["row_count"] == recovery["row_count"] == audit["row_count"] == 685
    assert len(rows) == 685
    assert processing == {"processed": 685}
    assert sum(row["paired_eligible"] is True for row in rows) == 618
    assert paired_reasons == {
        "missing_required_region_text": 66,
        "invalid_region_annotation": 1,
    }
    assert len({row["source_group_id"] for row in rows}) == 260
    assert visual == {"sampled_human_reviewed": 64, "not_visually_reviewed": 621}
    assert sum(row["audit_sample_member"] is True for row in rows) == len(sample_rows) == 64
    assert represented_exact_pairs == derived_exact_pairs
    assert set(exact_relation_occurrences.values()) <= {2}
    assert all(
        relation["reviewer"] is None and relation["reviewed_at"] is None
        for relation in exact_relations.values()
    )
    assert descriptor["review_inference"]["exact_pair_automated_count"] == len(derived_exact_pairs)
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
