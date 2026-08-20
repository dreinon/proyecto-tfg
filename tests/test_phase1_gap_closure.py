from __future__ import annotations

import copy
import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import yaml

from score_super_resolution import smb_audit
from score_super_resolution.contracts import validate_instance
from score_super_resolution.review_evidence import read_review

ROOT = Path(__file__).parents[1]
ACTIVE_PATH = ROOT / "data" / "manifests" / "smb-evaluation-v1.yaml"
SOURCE_PATH = ROOT / "data" / "sources" / "smb.yaml"
SAMPLE_PATH = ROOT / "data" / "audits" / "smb-visual-sample-v1.csv"
REVIEW_PATH = ROOT / "data" / "audits" / "smb-review-v1.csv"
ARCHIVE_SCRIPT = ROOT / "scripts" / "verify-phase1-clean-archive.sh"

EXPECTED_REVISION = "96332e8c4ac81cbdb7f61093ec5a4bfff76a0adb"
RECOVERY_PREFIX = "data/manifests/recovery/canonical-pixel-v2"
LITERAL_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_source_descriptor(descriptor: dict[str, Any]) -> None:
    contract = {**descriptor, "record_type": "source-descriptor"}
    for field in ("metadata_reviewed_at", "upstream_updated_at"):
        value = contract[field]
        if not isinstance(value, str):
            contract[field] = value.isoformat().replace("+00:00", "Z")
    validate_instance("source-descriptor", contract)
    assert descriptor["schema_version"] == 1
    assert descriptor["key"] == "smb"
    assert descriptor["provider"] == "hugging_face"
    assert descriptor["repository_id"] == "PRAIG/SMB"
    assert descriptor["revision"] == EXPECTED_REVISION
    assert descriptor["role"] == "evaluation_benchmark"
    assert descriptor["upstream_splits"] == {"test": {"examples": 685}}
    assert descriptor["access"] == {
        "gated": "manual",
        "authentication_environment_variable": "HF_TOKEN",
        "store_credentials_in_repository": False,
    }
    serialized = yaml.safe_dump(descriptor, sort_keys=True)
    assert all(pattern.search(serialized) is None for pattern in LITERAL_SECRET_PATTERNS)


def _selected_recovery_paths(pointer: dict[str, Any]) -> tuple[Path, Path, str]:
    selected: list[Path] = []
    bundle_ids: set[str] = set()
    for field, checksum_field, filename in (
        ("recovery_descriptor_path", "recovery_descriptor_sha256", "manifest-recovery.yaml"),
        ("recovery_records_path", "recovery_records_sha256", "manifest-records.jsonl.gz"),
    ):
        relative = Path(str(pointer[field]))
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.parts[:4] == tuple(RECOVERY_PREFIX.split("/"))
        assert len(relative.parts) == 6 and relative.name == filename
        bundle_id = relative.parts[-2]
        assert re.fullmatch(r"[0-9a-f]{64}", bundle_id)
        bundle_ids.add(bundle_id)
        resolved = (ROOT / relative).resolve()
        assert resolved.is_relative_to(ROOT.resolve()) and resolved.is_file()
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == pointer[checksum_field]
        selected.append(resolved)
    assert len(bundle_ids) == 1
    return selected[0], selected[1], bundle_ids.pop()


def _assert_descriptor_mutations_fail(descriptor: dict[str, Any]) -> None:
    mutations: list[Any] = [
        lambda value: value.pop("revision"),
        lambda value: value.__setitem__("revision", "main"),
        lambda value: value.__setitem__("repository_id", "someone/SMB"),
        lambda value: value.__setitem__("role", "training_source"),
        lambda value: value.__setitem__("upstream_splits", {"train": {"examples": 685}}),
        lambda value: value["access"].__setitem__("token", "literal-credential-value"),
        lambda value: value.__setitem__("secret", "credential-value"),
        lambda value: value["access"].__setitem__(
            "authentication_environment_variable", "literal-credential"
        ),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(descriptor)
        mutate(changed)
        with pytest.raises((AssertionError, ValueError)):
            _safe_source_descriptor(changed)


def test_active_phase1_gap_closure_reconciles(tmp_path: Path) -> None:
    source_descriptor = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(source_descriptor, dict)
    _safe_source_descriptor(source_descriptor)
    _assert_descriptor_mutations_fail(source_descriptor)

    pointer = yaml.safe_load(ACTIVE_PATH.read_text(encoding="utf-8"))
    assert isinstance(pointer, dict)
    validate_instance("manifest-active", pointer, version=2)
    recovery_descriptor_path, recovery_records_path, bundle_id = _selected_recovery_paths(pointer)

    active_copy = tmp_path / "data" / "manifests" / ACTIVE_PATH.name
    active_copy.parent.mkdir(parents=True)
    shutil.copyfile(ACTIVE_PATH, active_copy)
    generation_root = tmp_path / "empty-generations"
    assert not generation_root.exists()

    recovery_report = smb_audit.recover_active_manifest(
        active_path=active_copy,
        recovery_descriptor_path=recovery_descriptor_path,
        recovery_records_path=recovery_records_path,
        generation_root=generation_root,
    )
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_copy,
        generation_root=generation_root,
    )
    smb_audit.validate_v2_manifest_collection(descriptor, rows)
    for row in rows:
        validate_instance("manifest-row", row, version=2)
    report = smb_audit.reconcile_manifest(
        active_path=active_copy,
        generation_root=generation_root,
    )

    assert (
        pointer["generation_id"] == descriptor["generation_id"] == recovery_report["generation_id"]
    )
    assert (
        pointer["records_sha256"]
        == descriptor["records_sha256"]
        == recovery_report["records_sha256"]
    )
    assert descriptor["source_key"] == "smb"
    assert descriptor["source_revision"] == EXPECTED_REVISION
    assert descriptor["upstream_split"] == "test"
    assert descriptor["project_split"] == "evaluation"
    assert descriptor["benchmark_state"] == recovery_report["benchmark_state"] == "AUDITED_LOCKED"
    assert descriptor["source_provenance"]["dirty"] is False
    paired_eligible_count = sum(row["paired_eligible"] is True for row in rows)
    source_group_count = len({str(row["source_group_id"]) for row in rows})
    assert report == {
        "row_count": 685,
        "processed": 685,
        "failed": 0,
        "paired_eligible": paired_eligible_count,
        "generation_id": pointer["generation_id"],
        "records_sha256": pointer["records_sha256"],
        "benchmark_state": "AUDITED_LOCKED",
        "exclusion_count": len(descriptor["exclusions"]),
        "source_group_count": source_group_count,
    }
    assert bundle_id in pointer["recovery_descriptor_path"]
    assert bundle_id in pointer["recovery_records_path"]

    selected_ids = set(
        smb_audit.select_visual_sample(
            rows,
            seed=int(descriptor["sample_selection"]["seed"]),
            sample_size=int(descriptor["sample_selection"]["sample_size"]),
        )
    )
    manifest_sample_ids = {
        str(row["item_id"]) for row in rows if row["audit_sample_member"] is True
    }
    sample_rows = _csv_rows(SAMPLE_PATH)
    frozen_sample_ids = {row["item_id"] for row in sample_rows}
    review_document = read_review(REVIEW_PATH)
    assert review_document.canonical_bytes == REVIEW_PATH.read_bytes()
    review_rows = list(review_document.rows)
    review_kinds = Counter(row["review_kind"] for row in review_rows)
    visual_review_ids = {
        row["item_id"] for row in review_rows if row["review_kind"] == "visual_item"
    }
    assert selected_ids == manifest_sample_ids == frozen_sample_ids == visual_review_ids
    assert len(selected_ids) == 64
    assert Counter(str(row["visual_review"]["status"]) for row in rows) == {
        "sampled_human_reviewed": 64,
        "not_visually_reviewed": 621,
    }
    assert review_kinds == {"item_policy": 685, "visual_item": 64, "duplicate_pair": 14}
    assert len(review_rows) == 763

    rows_by_id = {str(row["item_id"]): row for row in rows}
    assert len(rows_by_id) == 685
    assert all(
        row["source_key"] == "smb"
        and row["source_revision"] == EXPECTED_REVISION
        and row["split"] == "test"
        for row in rows
    )
    policy_rows = {
        row["item_id"]: row for row in review_rows if row["review_kind"] == "item_policy"
    }
    assert set(policy_rows) == set(rows_by_id)
    assert all(
        policy_rows[item_id]["source_group_id"] == row["source_group_id"]
        for item_id, row in rows_by_id.items()
    )
    assert len({str(row["source_group_id"]) for row in rows}) == 260

    derived_exclusions = [
        {
            "upstream_index": row["upstream_index"],
            "item_id": row["item_id"],
            "reason": row["paired_ineligibility_reason"],
        }
        for row in rows
        if row["paired_eligible"] is False
    ]
    assert descriptor["exclusions"] == derived_exclusions
    assert paired_eligible_count == 618
    assert len(derived_exclusions) == 67
    assert Counter(item["reason"] for item in derived_exclusions) == {
        "missing_required_region_text": 66,
        "invalid_region_annotation": 1,
    }

    by_framed_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["processing_status"] == "processed":
            pixel_sha256 = row["image"]["pixel_sha256"]
            assert isinstance(pixel_sha256, str)
            by_framed_hash[pixel_sha256].append(str(row["item_id"]))
    derived_exact_pairs = {
        pair for members in by_framed_hash.values() for pair in combinations(sorted(members), 2)
    }
    exact_occurrences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    perceptual_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relations = row["duplicate_relations"]
        assert row["duplicate_summary"]["exact_relation_count"] == sum(
            relation["candidate_type"] == "exact" for relation in relations
        )
        assert row["duplicate_summary"]["perceptual_relation_count"] == sum(
            relation["candidate_type"] == "perceptual" for relation in relations
        )
        for relation in relations:
            if relation["candidate_type"] == "exact":
                exact_occurrences[tuple(relation["item_ids"])].append(relation)
            else:
                perceptual_occurrences[str(relation["pair_id"])].append(relation)
    assert set(exact_occurrences) == derived_exact_pairs
    assert descriptor["review_inference"]["exact_pair_automated_count"] == len(derived_exact_pairs)
    for pair, relations in exact_occurrences.items():
        assert len(relations) == 2
        assert {str(relation["counterpart_item_id"]) for relation in relations} == set(pair)
        assert all(
            relation["evidence_basis"] == "canonical_pixel_sha256"
            and relation["reviewer"] is None
            and relation["reviewed_at"] is None
            and relation["rationale"]
            == "Derived from matching canonical framed RGBA pixel SHA-256 values."
            for relation in relations
        )

    assert len(perceptual_occurrences) == 14
    assert all(len(relations) == 2 for relations in perceptual_occurrences.values())
    assert {
        relation["disposition"]
        for relations in perceptual_occurrences.values()
        for relation in relations
    } == {"distinct"}
    review_pair_ids = {
        row["review_key"] for row in review_rows if row["review_kind"] == "duplicate_pair"
    }
    assert review_pair_ids == set(perceptual_occurrences)

    assert all(
        row["rights"]["dataset_licence"]["identifier"] == "CC-BY-NC-4.0"
        and row["rights"]["access_status"] == "confirmed"
        and row["rights"]["item_provenance"]["status"] == "unavailable"
        and row["rights"]["redistribution"]
        == {
            "status": "not_established",
            "reviewed_basis_ref": None,
        }
        and row["rights"]["figure_reproduction"]
        == {
            "status": "prohibited",
            "reviewed_basis_ref": None,
        }
        for row in rows
    )
    assert descriptor["hash_provenance"]["pixels"] == {
        "algorithm": "sha256",
        "version": 2,
        "canonicalization": "canonical-rgba-frame-v2",
        "domain_separator": "smb-canonical-rgba-frame-v2",
        "decoder_library": "Pillow",
        "decoder_version": "12.3.0",
        "output_mode": "RGBA8",
        "alpha_policy": "retain-alpha-and-underlying-rgb",
        "orientation_policy": "stored-raster-ignore-exif",
        "metadata_policy": "ignore-non-raster-metadata",
        "max_encoded_bytes": 67_108_864,
        "max_pixels": 100_000_000,
        "failure_policy": "safe-explicit-failure-no-digest",
    }
    assert descriptor["duplicate_provenance"]["exact"] == {
        "algorithm": "canonical-pixel-sha256",
        "version": 2,
    }
    forbidden_outcome_fields = {
        "checkpoint",
        "lr_path",
        "metric",
        "model_id",
        "ranking",
        "sr_path",
    }
    assert forbidden_outcome_fields.isdisjoint(descriptor)
    assert all(forbidden_outcome_fields.isdisjoint(row) for row in rows)


def test_clean_archive_scans_head_and_tar_listing_with_the_same_forbidden_policy() -> None:
    script = ARCHIVE_SCRIPT.read_text(encoding="utf-8")

    assert "git ls-tree -r --name-only HEAD" in script
    assert 'tar -tf "$archive_path"' in script
    assert 'tar -tf "$archive_path" | rg -i "$forbidden_path_pattern"' in script
