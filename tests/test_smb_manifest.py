from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import score_super_resolution.smb_audit as smb_audit
from score_super_resolution.contracts import (
    ContractValidationError,
    load_schema,
    validate_instance,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "smb" / "records.json"
SCHEMA_IDS = ("manifest-active", "manifest-descriptor", "manifest-row")
V2_SCHEMA_IDS = ("manifest-active", "manifest-descriptor", "manifest-row")


def _fixtures() -> dict[str, dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("schema_id", SCHEMA_IDS)
def test_manifest_schemas_self_validate(schema_id: str) -> None:
    schema = load_schema(schema_id)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == f"urn:score-super-resolution:schema:v1:{schema_id}"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("fixture_name", "schema_id"),
    (
        ("active_pointer", "manifest-active"),
        ("manifest_descriptor", "manifest-descriptor"),
        ("normal_row", "manifest-row"),
        ("failure_row", "manifest-row"),
    ),
)
def test_valid_manifest_fixtures_pass(fixture_name: str, schema_id: str) -> None:
    validate_instance(schema_id, _fixtures()[fixture_name])


@pytest.mark.parametrize(
    ("fixture_name", "schema_id"),
    (
        ("active_pointer", "manifest-active"),
        ("manifest_descriptor", "manifest-descriptor"),
        ("normal_row", "manifest-row"),
        ("failure_row", "manifest-row"),
    ),
)
def test_manifest_shapes_reject_missing_and_unknown_fields(
    fixture_name: str, schema_id: str
) -> None:
    fixture = _fixtures()[fixture_name]
    for field in fixture:
        instance = copy.deepcopy(fixture)
        del instance[field]
        with pytest.raises(ContractValidationError, match="required property"):
            validate_instance(schema_id, instance)

    instance = copy.deepcopy(fixture)
    instance["unknown_field"] = True
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance(schema_id, instance)


@pytest.mark.parametrize(
    ("fixture_name", "wrong_schema_id"),
    (
        ("active_pointer", "manifest-descriptor"),
        ("active_pointer", "manifest-row"),
        ("manifest_descriptor", "manifest-active"),
        ("manifest_descriptor", "manifest-row"),
        ("normal_row", "manifest-active"),
        ("normal_row", "manifest-descriptor"),
    ),
)
def test_manifest_contract_shapes_cannot_be_interchanged(
    fixture_name: str, wrong_schema_id: str
) -> None:
    with pytest.raises(ContractValidationError):
        validate_instance(wrong_schema_id, _fixtures()[fixture_name])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("generation_id", "../escape"),
        ("descriptor_path", "../manifest-descriptor.yaml"),
        ("descriptor_path", "/tmp/manifest-descriptor.yaml"),
        ("descriptor_path", "a" * 64 + "/../manifest-descriptor.yaml"),
        ("records_path", "../manifest-records.jsonl"),
        ("records_path", "/tmp/manifest-records.jsonl"),
        ("records_path", "a" * 64 + "/nested/manifest-records.jsonl"),
    ),
)
def test_active_pointer_rejects_unsafe_generation_locations(field: str, value: str) -> None:
    pointer = copy.deepcopy(_fixtures()["active_pointer"])
    pointer[field] = value

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-active", pointer)


def test_failure_row_retains_denominator_identity_and_explicit_unavailability() -> None:
    row = _fixtures()["failure_row"]
    validate_instance("manifest-row", row)

    assert row["upstream_index"] == 1
    assert row["item_id"] == "smb-test-000001"
    assert row["expected_status"] == "unprocessable"
    assert row["processing_status"] == "failed"
    assert row["unprocessable_reason"] == "decode_failed"
    assert row["paired_eligible"] is False
    assert row["paired_ineligibility_reason"] == "decode_failed"
    assert row["encoded_sha256"] is None
    assert row["pixel_sha256"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("upstream_index", None),
        ("item_id", None),
        ("expected_status", None),
        ("paired_ineligibility_reason", None),
    ),
)
def test_failure_row_cannot_lose_denominator_or_pairing_reason(field: str, value: object) -> None:
    row = copy.deepcopy(_fixtures()["failure_row"])
    row[field] = value

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row)


def test_pointer_and_descriptor_freeze_manifest_identity_and_count() -> None:
    fixtures = _fixtures()
    pointer = fixtures["active_pointer"]
    descriptor = fixtures["manifest_descriptor"]

    assert pointer["generation_id"] == descriptor["generation_id"]
    assert pointer["row_schema_id"] == descriptor["row_schema_id"] == "manifest-row"
    assert pointer["row_schema_version"] == descriptor["row_schema_version"] == 1
    assert pointer["row_count"] == descriptor["row_count"] == 685
    assert pointer["records_sha256"] == descriptor["records_sha256"]
    assert pointer["descriptor_path"] == f"{pointer['generation_id']}/manifest-descriptor.yaml"
    assert pointer["records_path"] == f"{pointer['generation_id']}/manifest-records.jsonl"


def test_nested_manifest_objects_are_strict() -> None:
    descriptor = copy.deepcopy(_fixtures()["manifest_descriptor"])
    descriptor["sample_selection"]["outcome_score"] = 1.0
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("manifest-descriptor", descriptor)

    row = copy.deepcopy(_fixtures()["normal_row"])
    row["rights"]["unknown"] = "pending"
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("manifest-row", row)


def test_manifest_item_id_must_match_upstream_index_after_swapping_ids() -> None:
    fixtures = _fixtures()
    first = copy.deepcopy(fixtures["normal_row"])
    second = copy.deepcopy(fixtures["failure_row"])
    first["item_id"], second["item_id"] = second["item_id"], first["item_id"]

    for row in (first, second):
        with pytest.raises(ContractValidationError, match=r"\$\.item_id"):
            validate_instance("manifest-row", row)


@pytest.mark.parametrize(
    ("fixture_name", "mutations"),
    (
        (
            "normal_row",
            {"expected_status": "unprocessable"},
        ),
        (
            "failure_row",
            {"expected_status": "processable"},
        ),
        (
            "normal_row",
            {"unprocessable_reason": "decode_failed"},
        ),
        (
            "failure_row",
            {"unprocessable_reason": None},
        ),
    ),
    ids=(
        "processed-cannot-be-expected-unprocessable",
        "failed-cannot-be-expected-processable",
        "processed-cannot-have-unprocessable-reason",
        "failed-must-have-unprocessable-reason",
    ),
)
def test_manifest_processing_state_union_rejects_contradictions(
    fixture_name: str, mutations: dict[str, object]
) -> None:
    row = copy.deepcopy(_fixtures()[fixture_name])
    row.update(mutations)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row)


@pytest.mark.parametrize(
    ("fixture_name", "mutations"),
    (
        (
            "normal_row",
            {"paired_ineligibility_reason": "invalid_region_annotation"},
        ),
        (
            "normal_row",
            {"paired_eligible": False, "paired_ineligibility_reason": None},
        ),
    ),
    ids=(
        "paired-eligible-cannot-have-reason",
        "paired-ineligible-must-have-reason",
    ),
)
def test_manifest_pairing_state_union_rejects_contradictions(
    fixture_name: str, mutations: dict[str, object]
) -> None:
    row = copy.deepcopy(_fixtures()[fixture_name])
    row.update(mutations)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row)


def test_processed_paired_ineligible_annotation_row_remains_valid() -> None:
    row = copy.deepcopy(_fixtures()["normal_row"])
    row["bbox_valid"] = False
    row["annotation_failures"] = ["region_0_out_of_bounds"]
    row["paired_eligible"] = False
    row["paired_ineligibility_reason"] = "invalid_region_annotation"

    validate_instance("manifest-row", row)


def test_failed_unprocessable_row_remains_valid() -> None:
    validate_instance("manifest-row", _fixtures()["failure_row"])


def test_manifest_candidate_ids_reject_duplicates() -> None:
    row = copy.deepcopy(_fixtures()["normal_row"])
    row["near_duplicate_candidate_ids"] = [
        "candidate-0000000000000000",
        "candidate-0000000000000000",
    ]

    with pytest.raises(ContractValidationError, match=r"\$\.near_duplicate_candidate_ids"):
        validate_instance("manifest-row", row)


def test_manifest_candidate_ids_must_be_in_canonical_order() -> None:
    row = copy.deepcopy(_fixtures()["normal_row"])
    row["near_duplicate_candidate_ids"] = [
        "candidate-0000000000000001",
        "candidate-0000000000000000",
    ]

    with pytest.raises(ContractValidationError, match=r"\$\.near_duplicate_candidate_ids"):
        validate_instance("manifest-row", row)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _provenance_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "proyecto"
    (repository / "src" / "score_super_resolution").mkdir(parents=True)
    (repository / "data" / "schemas" / "v1").mkdir(parents=True)
    (repository / "artifacts").mkdir()
    (repository / "src" / "score_super_resolution" / "alpha.py").write_text(
        "VALUE = 'alpha'\n", encoding="utf-8"
    )
    (repository / "src" / "score_super_resolution" / "beta.py").write_text(
        "VALUE = 'beta'\n", encoding="utf-8"
    )
    (repository / "data" / "schemas" / "v1" / "record.schema.json").write_text(
        '{"type":"object"}\n', encoding="utf-8"
    )
    (repository / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SMB provenance test")
    _git(repository, "config", "user.email", "smb-provenance@example.invalid")
    _git(repository, "add", ".gitignore", "data", "pyproject.toml", "src", "uv.lock")
    _git(repository, "commit", "-q", "-m", "fixture")
    return repository


def test_audit_source_provenance_is_explicit_and_path_order_independent(tmp_path: Path) -> None:
    repository = _provenance_repository(tmp_path)
    paths = smb_audit._authoritative_audit_source_paths(repository)

    provenance = smb_audit.audit_source_provenance(repository)
    forward = smb_audit._source_tree_sha256(repository, paths)
    reverse = smb_audit._source_tree_sha256(repository, reversed(paths))

    assert provenance["source_set_version"] == 1
    assert provenance["algorithm"] == "sha256"
    assert (
        provenance["revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert provenance["dirty"] is False
    assert provenance["source_tree_sha256"] == forward == reverse
    assert set(provenance) == {
        "algorithm",
        "dirty",
        "lock_sha256",
        "patch_sha256",
        "revision",
        "source_set_version",
        "source_tree_sha256",
    }
    assert all(
        len(provenance[field]) == 64
        for field in (
            "source_tree_sha256",
            "patch_sha256",
            "lock_sha256",
        )
    )


def test_tracked_dirty_source_changes_tree_and_patch_without_serializing_content(
    tmp_path: Path,
) -> None:
    repository = _provenance_repository(tmp_path)
    clean = smb_audit.audit_source_provenance(repository)
    secret_sentinel = "_".join(("HF", "TOKEN")) + "=must-not-appear-in-provenance"
    source = repository / "src" / "score_super_resolution" / "alpha.py"
    source.write_text(f"VALUE = {secret_sentinel!r}\n", encoding="utf-8")

    dirty = smb_audit.audit_source_provenance(repository)
    serialized = json.dumps(dirty, sort_keys=True)

    assert dirty["dirty"] is True
    assert dirty["revision"] == clean["revision"]
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
    assert dirty["patch_sha256"] != clean["patch_sha256"]
    assert secret_sentinel not in serialized
    assert str(repository) not in serialized


def test_untracked_relevant_source_changes_actual_tree_and_patch_identity(tmp_path: Path) -> None:
    repository = _provenance_repository(tmp_path)
    clean = smb_audit.audit_source_provenance(repository)
    untracked = repository / "src" / "score_super_resolution" / "new_source.py"
    untracked.write_text("VALUE = 'untracked'\n", encoding="utf-8")

    dirty = smb_audit.audit_source_provenance(repository)

    assert dirty["dirty"] is True
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
    assert dirty["patch_sha256"] != clean["patch_sha256"]


def test_lock_mutation_changes_lock_and_tree_identity(tmp_path: Path) -> None:
    repository = _provenance_repository(tmp_path)
    clean = smb_audit.audit_source_provenance(repository)
    (repository / "uv.lock").write_text("version = 2\n", encoding="utf-8")

    dirty = smb_audit.audit_source_provenance(repository)

    assert dirty["dirty"] is True
    assert dirty["lock_sha256"] != clean["lock_sha256"]
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
    assert dirty["patch_sha256"] != clean["patch_sha256"]


def test_ignored_unrelated_artifact_does_not_change_audit_source_identity(tmp_path: Path) -> None:
    repository = _provenance_repository(tmp_path)
    before = smb_audit.audit_source_provenance(repository)
    (repository / "artifacts" / "runtime-output.bin").write_bytes(b"generated output")

    after = smb_audit.audit_source_provenance(repository)

    assert after == before


@pytest.mark.parametrize("missing", ("git", "source", "lock"))
def test_authoritative_provenance_fails_closed_when_facts_are_missing(
    tmp_path: Path, missing: str
) -> None:
    repository = _provenance_repository(tmp_path)
    if missing == "git":
        target = repository.parent / "without-git"
        target.mkdir()
        (target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (target / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (target / "src" / "score_super_resolution").mkdir(parents=True)
        (target / "src" / "score_super_resolution" / "source.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (target / "data" / "schemas").mkdir(parents=True)
        (target / "data" / "schemas" / "schema.json").write_text("{}\n", encoding="utf-8")
        repository = target
    elif missing == "source":
        (repository / "src" / "score_super_resolution" / "alpha.py").unlink()
        (repository / "src" / "score_super_resolution" / "beta.py").unlink()
    else:
        (repository / "uv.lock").unlink()

    with pytest.raises(RuntimeError, match="provenance"):
        smb_audit.audit_source_provenance(repository)


def _v2_exact_relation(
    *,
    item_id: str = "smb-test-000000",
    counterpart_item_id: str = "smb-test-000001",
) -> dict[str, Any]:
    return {
        "pair_id": "pair-0000000000000001",
        "candidate_type": "exact",
        "item_ids": [item_id, counterpart_item_id],
        "counterpart_item_id": counterpart_item_id,
        "evidence_basis": "cryptographic_equality",
        "evidence": {
            "encoded_sha256": "1" * 64,
            "pixel_sha256": "2" * 64,
        },
        "disposition": "duplicate",
        "reviewer": None,
        "reviewed_at": None,
        "rationale": "Derived from matching encoded and canonical pixel SHA-256 values.",
    }


def _v2_perceptual_relation(
    *,
    pair_id: str = "pair-0000000000000002",
    item_id: str = "smb-test-000000",
    counterpart_item_id: str = "smb-test-000002",
    disposition: str = "related",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "candidate_type": "perceptual",
        "item_ids": [item_id, counterpart_item_id],
        "counterpart_item_id": counterpart_item_id,
        "evidence_basis": "perceptual_hash_plus_human_review",
        "evidence": {
            "algorithm": "phash",
            "version": 1,
            "distance": 3,
        },
        "disposition": disposition,
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-08-18",
        "rationale": "The pages share source material but are not the same page.",
    }


def _v2_row() -> dict[str, Any]:
    exact = _v2_exact_relation()
    perceptual = _v2_perceptual_relation()
    return {
        "schema_version": 2,
        "record_type": "manifest-row",
        "manifest_version": 2,
        "source_key": "smb",
        "source_revision": "a" * 40,
        "split": "test",
        "upstream_index": 0,
        "item_id": "smb-test-000000",
        "source_identity": {
            "original_score_normalized": "bach-bwv-1",
            "original_score_raw_sha256": "3" * 64,
            "page_normalized": "1",
            "page_raw_sha256": "4" * 64,
            "page_texture_normalized": "aged-paper",
            "page_texture_raw_sha256": "5" * 64,
        },
        "source_group_id": "bach-bwv-1",
        "image": {
            "encoded_sha256": "1" * 64,
            "pixel_sha256": "2" * 64,
            "declared_width": 100,
            "declared_height": 200,
            "decoded_width": 100,
            "decoded_height": 200,
            "mode": "RGB",
            "format": "PNG",
            "byte_count": 1024,
        },
        "annotations": {
            "region_count": 1,
            "bbox_valid": True,
            "required_text_present": True,
            "failures": [],
        },
        "automated_audit": {
            "status": "automated",
            "algorithm_version": "smb-audit-v2",
            "quality_flags": [],
        },
        "visual_review": {
            "status": "sampled_human_reviewed",
            "reviewer": "reviewer-1",
            "reviewed_at": "2026-08-18",
            "rationale": "Frozen sample inspection found no material issue.",
            "quality_flags": [],
            "suitability": "suitable",
        },
        "audit_sample_member": True,
        "duplicate_relations": [exact, perceptual],
        "near_duplicate_candidate_ids": [perceptual["pair_id"]],
        "duplicate_summary": {
            "exact_relation_count": 1,
            "perceptual_relation_count": 1,
            "pending_relation_count": 0,
            "duplicate_relation_count": 1,
            "related_relation_count": 1,
            "distinct_relation_count": 0,
            "unavailable_relation_count": 0,
            "group_ids": ["duplicate-group-0000000000000001"],
        },
        "expected_status": "processable",
        "processing_status": "processed",
        "unprocessable_reason": None,
        "rights": {
            "dataset_licence": {
                "status": "confirmed",
                "identifier": "CC-BY-NC-4.0",
                "reference": "https://creativecommons.org/licenses/by-nc/4.0/",
            },
            "item_provenance": {
                "status": "unavailable",
                "reason": "The dataset does not expose a per-item source chain.",
            },
            "access_status": "confirmed",
            "redistribution": {
                "status": "not_established",
                "reviewed_basis_ref": None,
            },
            "figure_reproduction": {
                "status": "prohibited",
                "reviewed_basis_ref": None,
            },
        },
        "paired_eligible": True,
        "paired_ineligibility_reason": None,
    }


def _v2_descriptor() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "record_type": "manifest-descriptor",
        "manifest_id": "smb-evaluation-v2",
        "generation_id": "6" * 64,
        "generation_algorithm": {
            "algorithm": "sha256",
            "version": 2,
            "domain_separator": "smb-manifest-generation-v2",
            "descriptor_canonicalization": "yaml-safe-sort-keys-utf8-v1",
            "records_canonicalization": "jsonl-utf8-sorted-keys-v1",
        },
        "source_key": "smb",
        "source_revision": "a" * 40,
        "creation_command": "python -m score_super_resolution.smb_audit build",
        "source_provenance": {
            "source_set_version": 1,
            "algorithm": "sha256",
            "revision": "b" * 40,
            "dirty": False,
            "source_tree_sha256": "7" * 64,
            "patch_sha256": "8" * 64,
            "lock_sha256": "9" * 64,
        },
        "grouping_unit": "source_score",
        "upstream_split": "test",
        "project_split": "evaluation",
        "deterministic_seed": 17,
        "exclusions": [],
        "row_schema_id": "manifest-row",
        "row_schema_version": 2,
        "row_count": 685,
        "records_sha256": "c" * 64,
        "audit_version": "smb-audit-v2",
        "benchmark_state": "AUDITED_LOCKED",
        "hash_provenance": {
            "encoded": {
                "algorithm": "sha256",
                "version": 1,
                "canonicalization": "encoded-bytes-v1",
            },
            "pixels": {
                "algorithm": "sha256",
                "version": 1,
                "canonicalization": "rgba-uint8-row-major-v1",
            },
        },
        "duplicate_provenance": {
            "exact": {"algorithm": "encoded-and-pixel-sha256", "version": 1},
            "near": {
                "algorithm": "phash",
                "version": 1,
                "library": "ImageHash",
                "library_version": "4.3.2",
                "hash_size": 8,
                "highfreq_factor": 4,
                "maximum_hamming_distance": 6,
            },
        },
        "sample_selection": {
            "algorithm": "sha256-rank",
            "version": 1,
            "seed": 17,
            "population_size": 685,
            "sample_size": 64,
            "identity_fields": ["upstream_index", "item_id"],
            "selection_state": "pre-review",
        },
        "review_inference": {
            "automated_population_audit_count": 685,
            "sampled_human_review_count": 64,
            "targeted_human_review_count": 0,
            "not_visually_reviewed_count": 621,
            "unavailable_visual_review_count": 0,
            "not_applicable_visual_review_count": 0,
            "exact_pair_automated_count": 1,
            "perceptual_pair_count": 14,
            "perceptual_pair_human_review_count": 14,
            "perceptual_pair_pending_count": 0,
            "inference_scope": "sample_observation_only",
            "population_prevalence_inference": "not_supported",
        },
    }


def _v2_active_pointer() -> dict[str, Any]:
    descriptor = _v2_descriptor()
    generation_id = descriptor["generation_id"]
    return {
        "schema_version": 2,
        "record_type": "manifest-active",
        "manifest_id": "smb-evaluation-v2",
        "generation_id": generation_id,
        "descriptor_path": f"{generation_id}/manifest-descriptor.yaml",
        "records_path": f"{generation_id}/manifest-records.jsonl",
        "row_schema_id": "manifest-row",
        "row_schema_version": 2,
        "row_count": 685,
        "records_sha256": descriptor["records_sha256"],
    }


@pytest.mark.parametrize("schema_id", V2_SCHEMA_IDS)
def test_manifest_v2_schemas_self_validate(schema_id: str) -> None:
    schema = load_schema(schema_id, version=2)
    assert schema["$id"] == f"urn:score-super-resolution:schema:v2:{schema_id}"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("instance", "schema_id"),
    (
        (_v2_active_pointer(), "manifest-active"),
        (_v2_descriptor(), "manifest-descriptor"),
        (_v2_row(), "manifest-row"),
    ),
)
def test_manifest_v2_canonical_records_validate(instance: dict[str, Any], schema_id: str) -> None:
    validate_instance(schema_id, instance, version=2)


def test_manifest_v2_compact_row_excludes_raw_upstream_content() -> None:
    schema = load_schema("manifest-row", version=2)
    serialized_schema = json.dumps(schema, sort_keys=True)
    assert '"raw"' not in serialized_schema
    assert "original_score_raw_sha256" in serialized_schema

    compact = _v2_row()
    legacy = copy.deepcopy(_fixtures()["normal_row"])
    for field in ("original_score", "page", "page_texture"):
        legacy[field]["raw"] = "untrusted-upstream-content-" * 200
    assert len(json.dumps(compact, sort_keys=True)) < len(json.dumps(legacy, sort_keys=True)) * 0.7


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    (
        (
            lambda row: row["visual_review"].pop("reviewer"),
            "visual_review",
        ),
        (
            lambda row: row.update(audit_sample_member=False),
            "audit_sample_member",
        ),
        (
            lambda row: row["visual_review"].update(status="not_visually_reviewed"),
            "visual_review",
        ),
        (
            lambda row: row["visual_review"].update(status="targeted_human_reviewed"),
            "visual_review",
        ),
        (
            lambda row: row["visual_review"].update(status="unavailable"),
            "visual_review",
        ),
        (
            lambda row: row["visual_review"].update(status="not_applicable"),
            "visual_review",
        ),
    ),
    ids=(
        "sampled-review-requires-reviewer",
        "sampled-review-requires-frozen-sample-membership",
        "not-visually-reviewed-cannot-retain-human-claims",
        "targeted-review-requires-separate-basis",
        "unavailable-review-cannot-retain-human-claims",
        "not-applicable-review-cannot-retain-human-claims",
    ),
)
def test_manifest_v2_visual_review_union_rejects_contradictions(
    mutation: Any, error_pattern: str
) -> None:
    row = _v2_row()
    mutation(row)
    with pytest.raises(ContractValidationError, match=error_pattern):
        validate_instance("manifest-row", row, version=2)


def test_manifest_v2_unsampled_item_has_no_per_item_visual_claim() -> None:
    row = _v2_row()
    row["audit_sample_member"] = False
    row["visual_review"] = {"status": "not_visually_reviewed"}
    validate_instance("manifest-row", row, version=2)


def test_manifest_v2_targeted_review_is_separate_from_the_frozen_sample() -> None:
    row = _v2_row()
    row["audit_sample_member"] = False
    row["visual_review"] = {
        "status": "targeted_human_reviewed",
        "target_basis_ref": "duplicate-pair:pair-0000000000000002",
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-08-18",
        "rationale": "Reviewed separately to adjudicate one candidate pair.",
        "quality_flags": [],
        "suitability": "not_assessed",
    }
    validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    "automated_audit",
    (
        {"status": "pending", "rationale": "Automated audit has not run."},
        {"status": "unavailable", "reason": "Image bytes could not be decoded."},
        {"status": "not_applicable", "reason": "No image audit applies to this record."},
    ),
    ids=("pending", "unavailable", "not-applicable"),
)
def test_manifest_v2_automated_audit_noncompleted_states_are_closed(
    automated_audit: dict[str, object],
) -> None:
    row = _v2_row()
    row["automated_audit"] = automated_audit
    validate_instance("manifest-row", row, version=2)

    row["automated_audit"] = {"status": automated_audit["status"]}
    with pytest.raises(ContractValidationError, match="automated_audit"):
        validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    "visual_review",
    (
        {"status": "unavailable", "reason": "The image could not be rendered."},
        {"status": "not_applicable", "reason": "Visual review does not apply."},
    ),
    ids=("unavailable", "not-applicable"),
)
def test_manifest_v2_visual_nonreview_states_are_closed(
    visual_review: dict[str, object],
) -> None:
    row = _v2_row()
    row["audit_sample_member"] = False
    row["visual_review"] = visual_review
    validate_instance("manifest-row", row, version=2)

    row["visual_review"] = {"status": visual_review["status"]}
    with pytest.raises(ContractValidationError, match="visual_review"):
        validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda relation: relation.update(disposition="distinct"),
        lambda relation: relation.update(evidence_basis="perceptual_hash_plus_human_review"),
        lambda relation: relation["evidence"].update(encoded_sha256=None, pixel_sha256=None),
        lambda relation: relation.update(reviewer="reviewer-1", reviewed_at="2026-08-18"),
    ),
    ids=(
        "exact-cannot-be-distinct",
        "exact-requires-cryptographic-basis",
        "exact-requires-a-cryptographic-digest",
        "exact-is-automated-not-human-review",
    ),
)
def test_manifest_v2_exact_relation_is_closed_automated_evidence(mutation: Any) -> None:
    row = _v2_row()
    mutation(row["duplicate_relations"][0])
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda relation: relation.update(reviewer=None),
        lambda relation: relation.update(reviewed_at=None),
        lambda relation: relation.update(rationale=""),
        lambda relation: relation.update(evidence_basis="perceptual_hash_candidate"),
    ),
    ids=(
        "reviewed-perceptual-requires-reviewer",
        "reviewed-perceptual-requires-date",
        "reviewed-perceptual-requires-rationale",
        "reviewed-perceptual-requires-human-basis",
    ),
)
def test_manifest_v2_reviewed_perceptual_relation_requires_human_evidence(mutation: Any) -> None:
    row = _v2_row()
    mutation(row["duplicate_relations"][1])
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row, version=2)


def test_manifest_v2_pending_perceptual_relation_cannot_impersonate_human_review() -> None:
    row = _v2_row()
    relation = row["duplicate_relations"][1]
    relation.update(
        evidence_basis="perceptual_hash_candidate",
        disposition="pending",
        reviewer=None,
        reviewed_at=None,
        rationale="",
    )
    row["duplicate_summary"].update(
        pending_relation_count=1,
        related_relation_count=0,
    )
    validate_instance("manifest-row", row, version=2)

    relation["reviewer"] = "reviewer-1"
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row, version=2)


def test_manifest_v2_allows_independent_relations_for_one_item() -> None:
    row = _v2_row()
    row["duplicate_relations"][1] = _v2_perceptual_relation(disposition="distinct")
    row["duplicate_relations"].append(
        _v2_perceptual_relation(
            pair_id="pair-0000000000000003",
            counterpart_item_id="smb-test-000003",
            disposition="related",
        )
    )
    row["near_duplicate_candidate_ids"] = [
        "pair-0000000000000002",
        "pair-0000000000000003",
    ]
    row["duplicate_summary"].update(
        perceptual_relation_count=2,
        related_relation_count=1,
        distinct_relation_count=1,
    )
    validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    ("reuse_field", "status"),
    (
        ("redistribution", "permitted"),
        ("figure_reproduction", "permitted"),
    ),
)
def test_manifest_v2_unavailable_provenance_cannot_imply_permission(
    reuse_field: str, status: str
) -> None:
    row = _v2_row()
    row["rights"][reuse_field]["status"] = status
    with pytest.raises(ContractValidationError, match="reviewed_basis_ref"):
        validate_instance("manifest-row", row, version=2)


def test_manifest_v2_permission_accepts_an_explicit_reviewed_basis() -> None:
    row = _v2_row()
    row["rights"]["figure_reproduction"] = {
        "status": "permitted",
        "reviewed_basis_ref": "rights-review:figure-basis-1",
    }
    validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.update(expected_status="unprocessable"),
        lambda row: row.update(unprocessable_reason="decode_failed"),
        lambda row: row.update(paired_eligible=False),
        lambda row: row.update(paired_ineligibility_reason="invalid_region_annotation"),
    ),
    ids=(
        "processed-cannot-be-expected-unprocessable-v2",
        "processed-cannot-carry-failure-reason-v2",
        "paired-ineligible-requires-reason-v2",
        "paired-eligible-cannot-carry-reason-v2",
    ),
)
def test_manifest_v2_processing_and_pairing_unions_reject_contradictions(mutation: Any) -> None:
    row = _v2_row()
    mutation(row)
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-row", row, version=2)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("automated_population_audit_count", 684),
        ("sampled_human_review_count", 63),
        ("not_visually_reviewed_count", 620),
        ("inference_scope", "population_estimate"),
        ("population_prevalence_inference", "supported"),
    ),
)
def test_manifest_v2_descriptor_freezes_population_sample_inference(
    field: str, value: object
) -> None:
    descriptor = _v2_descriptor()
    descriptor["review_inference"][field] = value
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-descriptor", descriptor, version=2)


def test_manifest_v2_descriptor_requires_plan_01_15_source_provenance() -> None:
    descriptor = _v2_descriptor()
    descriptor["source_provenance"].pop("patch_sha256")
    with pytest.raises(ContractValidationError, match="required property"):
        validate_instance("manifest-descriptor", descriptor, version=2)
