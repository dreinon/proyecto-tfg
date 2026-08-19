from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import score_super_resolution.smb_audit as smb_audit
from score_super_resolution.benchmark_policy import (
    AUDIT_PURPOSES,
    FINAL_EVALUATION_PURPOSES,
    NEVER_ALLOWED_PURPOSES,
    BenchmarkPolicyError,
    BenchmarkPurpose,
    BenchmarkState,
    assert_smb_purpose_allowed,
)
from score_super_resolution.contracts import ContractValidationError, load_schema, validate_instance

_PREREQUISITE_CONTRACTS = {
    "smb_audit_complete": ("smb_audit_control", "smb-freeze-control", 1),
    "evaluation_manifest_frozen": ("manifest_active", "manifest-active", (1, 2)),
    "methods_frozen": ("methods_control", "smb-freeze-control", 1),
    "checkpoints_frozen": ("checkpoints_control", "smb-freeze-control", 1),
    "controlled_conditions_frozen": (
        "controlled_conditions_control",
        "smb-freeze-control",
        1,
    ),
    "metrics_frozen": ("metrics_control", "smb-freeze-control", 1),
    "independent_units_frozen": ("independent_units_control", "smb-freeze-control", 1),
    "exclusions_frozen": ("exclusions_control", "smb-freeze-control", 1),
    "seeds_frozen": ("seeds_control", "smb-freeze-control", 1),
    "qualitative_samples_frozen": (
        "qualitative_samples_control",
        "smb-freeze-control",
        1,
    ),
    "interpretation_rules_frozen": (
        "interpretation_rules_control",
        "smb-freeze-control",
        1,
    ),
    "human_unlock_recorded": ("human_unlock_approval", "smb-freeze-control", 1),
}
_PRECEDING_PREREQUISITES = tuple(_PREREQUISITE_CONTRACTS)[:-1]
_HEX_A = "a" * 64
_HEX_B = "b" * 64
_SOURCE_REVISION = "c" * 40


def _artifact_reference(prerequisite_id: str) -> dict[str, Any]:
    artifact_kind, schema_id, schema_version = _PREREQUISITE_CONTRACTS[prerequisite_id]
    version = schema_version[0] if isinstance(schema_version, tuple) else schema_version
    reference: dict[str, Any] = {
        "prerequisite_id": prerequisite_id,
        "evidence_id": f"synthetic-{prerequisite_id}-v1",
        "artifact_path": f"controls/{prerequisite_id}.json",
        "artifact_sha256": _HEX_A,
        "artifact_kind": artifact_kind,
        "schema_id": schema_id,
        "schema_version": version,
    }
    if prerequisite_id == "evaluation_manifest_frozen":
        reference.update(
            {
                "expected_generation_id": _HEX_B,
                "expected_row_count": 685,
                "expected_records_sha256": _HEX_A,
                "expected_benchmark_state": "AUDITED_LOCKED",
            }
        )
    return reference


def _artifact_backed_unlock() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "smb-evaluation-unlock",
        "review_status": "reviewed",
        "reviewer": "accountable-reviewer",
        "reviewed_at": "2026-08-18",
        "prerequisites": {
            prerequisite_id: _artifact_reference(prerequisite_id)
            for prerequisite_id in _PREREQUISITE_CONTRACTS
        },
    }


def _freeze_content(control_id: str) -> dict[str, Any]:
    contents: dict[str, dict[str, Any]] = {
        "smb_audit_complete": {
            "source_revision": _SOURCE_REVISION,
            "active_generation_id": _HEX_B,
            "row_count": 685,
            "records_sha256": _HEX_A,
            "benchmark_state": "AUDITED_LOCKED",
            "grouping_evidence_sha256": _HEX_A,
            "review_evidence_sha256": _HEX_A,
            "sample_evidence_sha256": _HEX_A,
            "rights_evidence_sha256": _HEX_A,
        },
        "methods_frozen": {
            "methods": [{"role": "synthetic-baseline", "implementation_id": "synthetic-bicubic-v1"}]
        },
        "checkpoints_frozen": {
            "methods": [{"method_id": "synthetic-bicubic-v1", "status": "not_applicable"}]
        },
        "controlled_conditions_frozen": {
            "degradation_control_ids": ["synthetic-degradation-v1"],
            "execution_control_ids": ["synthetic-execution-v1"],
        },
        "metrics_frozen": {
            "quantitative_definition_ids": ["synthetic-psnr-v1"],
            "qualitative_definition_ids": ["synthetic-notation-failure-v1"],
        },
        "independent_units_frozen": {
            "grouping_rule_id": "synthetic-source-score-v1",
            "pairing_rule_id": "synthetic-paired-page-v1",
            "aggregation_rule_id": "synthetic-source-aggregate-v1",
            "uncertainty_rule_id": "synthetic-cluster-bootstrap-v1",
        },
        "exclusions_frozen": {
            "excluded_states": ["synthetic-unprocessable-v1"],
            "failure_handling_rule_id": "synthetic-retain-failures-v1",
        },
        "seeds_frozen": {
            "generation_policy_id": "synthetic-generation-seed-v1",
            "execution_policy_id": "synthetic-execution-seed-v1",
            "sampling_policy_id": "synthetic-sampling-seed-v1",
        },
        "qualitative_samples_frozen": {
            "outcome_independent": True,
            "sampling_rule_id": "synthetic-hash-rank-v1",
            "sampling_rule_version": 1,
            "item_set_sha256": _HEX_A,
            "item_count": 8,
        },
        "interpretation_rules_frozen": {
            "success_rules": ["synthetic-success-v1"],
            "stop_rules": ["synthetic-stop-v1"],
            "contradiction_rules": ["synthetic-contradiction-v1"],
            "claim_rules": ["synthetic-claim-v1"],
        },
        "human_unlock_recorded": {
            "reviewer": "accountable-reviewer",
            "reviewed_at": "2026-08-18",
            "decision": "approved",
            "approval_evidence": {
                "evidence_type": "human-review",
                "evidence_id": "synthetic-human-review-v1",
                "artifact_path": "evidence/human-review.json",
                "artifact_sha256": _HEX_B,
            },
            "prerequisite_digests": {
                prerequisite_id: _HEX_A for prerequisite_id in _PRECEDING_PREREQUISITES
            },
        },
    }
    return contents[control_id]


def _freeze_control(control_id: str) -> dict[str, Any]:
    artifact_kind, _, _ = _PREREQUISITE_CONTRACTS[control_id]
    return {
        "schema_version": 1,
        "record_type": "smb-freeze-control",
        "status": "frozen",
        "artifact_kind": artifact_kind,
        "control_id": control_id,
        "content": _freeze_content(control_id),
    }


def _manifest_rows() -> list[dict[str, Any]]:
    fixture_path = Path(__file__).parent / "fixtures" / "smb" / "records.json"
    template = json.loads(fixture_path.read_text(encoding="utf-8"))["normal_row"]
    rows: list[dict[str, Any]] = []
    for index in range(685):
        row = copy.deepcopy(template)
        row["upstream_index"] = index
        row["item_id"] = f"smb-test-{index:06d}"
        row["original_score"] = {
            "raw": f"synthetic-score-{index:03d}",
            "normalized": f"synthetic-score-{index:03d}",
        }
        row["source_group_id"] = f"synthetic-score-{index:03d}"
        row["audit_sample_member"] = index < 64
        rows.append(row)
    return rows


def _manifest_descriptor() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "smb" / "records.json"
    return copy.deepcopy(
        json.loads(fixture_path.read_text(encoding="utf-8"))["manifest_descriptor"]
    )


def _write_control(path: Path, control: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(control, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _synthetic_evaluation_gate(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    project_root = tmp_path / "project"
    active_path = project_root / "data" / "manifests" / "smb-evaluation-v1.yaml"
    generation_root = project_root / "artifacts" / "smb-manifests" / "generations"
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_manifest_descriptor(),
        rows=_manifest_rows(),
    )
    pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    record = _artifact_backed_unlock()
    manifest_reference = record["prerequisites"]["evaluation_manifest_frozen"]
    manifest_reference.update(
        {
            "artifact_path": active_path.relative_to(project_root).as_posix(),
            "artifact_sha256": hashlib.sha256(active_path.read_bytes()).hexdigest(),
            "expected_generation_id": pointer["generation_id"],
            "expected_row_count": pointer["row_count"],
            "expected_records_sha256": pointer["records_sha256"],
            "expected_benchmark_state": "AUDITED_LOCKED",
        }
    )

    for prerequisite_id in _PRECEDING_PREREQUISITES:
        if prerequisite_id == "evaluation_manifest_frozen":
            continue
        control = _freeze_control(prerequisite_id)
        if prerequisite_id == "smb_audit_complete":
            control["content"].update(
                {
                    "active_generation_id": pointer["generation_id"],
                    "row_count": pointer["row_count"],
                    "records_sha256": pointer["records_sha256"],
                }
            )
        relative = Path("controls") / f"{prerequisite_id}.json"
        digest = _write_control(project_root / relative, control)
        reference = record["prerequisites"][prerequisite_id]
        reference["artifact_path"] = relative.as_posix()
        reference["artifact_sha256"] = digest

    human_control = _freeze_control("human_unlock_recorded")
    human_control["content"]["prerequisite_digests"] = {
        prerequisite_id: record["prerequisites"][prerequisite_id]["artifact_sha256"]
        for prerequisite_id in _PRECEDING_PREREQUISITES
    }
    human_relative = Path("controls") / "human_unlock_recorded.json"
    human_digest = _write_control(project_root / human_relative, human_control)
    human_reference = record["prerequisites"]["human_unlock_recorded"]
    human_reference["artifact_path"] = human_relative.as_posix()
    human_reference["artifact_sha256"] = human_digest
    return record, project_root, generation_root


def _assert_gate_rejected(
    *,
    record: dict[str, Any],
    project_root: Path,
    generation_root: Path,
) -> None:
    calls: list[str] = []
    with pytest.raises(BenchmarkPolicyError):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.METRIC,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            project_root=project_root,
            manifest_generation_root=generation_root,
            callback=lambda: calls.append("called"),
        )
    assert calls == []


def _smb_descriptor() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "data" / "sources" / "smb.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _reviewed_unlock() -> dict[str, Any]:
    return _artifact_backed_unlock()


def test_purpose_partition_is_exhaustive_and_disjoint() -> None:
    partitions = (AUDIT_PURPOSES, FINAL_EVALUATION_PURPOSES, NEVER_ALLOWED_PURPOSES)

    assert set(BenchmarkPurpose) == set().union(*partitions)
    for index, partition in enumerate(partitions):
        for other in partitions[index + 1 :]:
            assert partition.isdisjoint(other)


@pytest.mark.parametrize("purpose", sorted(AUDIT_PURPOSES, key=lambda item: item.value))
def test_locked_smb_allows_each_declared_audit_purpose(purpose: BenchmarkPurpose) -> None:
    result = assert_smb_purpose_allowed(
        source_descriptor=_smb_descriptor(),
        purpose=purpose,
        callback=lambda: purpose.value,
    )

    assert result == purpose.value


@pytest.mark.parametrize(
    "purpose",
    sorted(FINAL_EVALUATION_PURPOSES | NEVER_ALLOWED_PURPOSES, key=lambda item: item.value),
)
def test_locked_smb_rejects_each_non_audit_purpose_before_callback(
    purpose: BenchmarkPurpose,
) -> None:
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match=purpose.value):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=purpose,
            callback=lambda: calls.append(purpose.value),
        )

    assert calls == []


@pytest.mark.parametrize("purpose", sorted(FINAL_EVALUATION_PURPOSES, key=lambda item: item.value))
def test_complete_reviewed_unlock_permits_each_final_evaluation_purpose(
    purpose: BenchmarkPurpose, tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    assert (
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=purpose,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            project_root=project_root,
            manifest_generation_root=generation_root,
            callback=lambda: "executed",
        )
        == "executed"
    )


@pytest.mark.parametrize("purpose", sorted(NEVER_ALLOWED_PURPOSES, key=lambda item: item.value))
def test_adaptive_purposes_remain_forbidden_after_unlock(purpose: BenchmarkPurpose) -> None:
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match=purpose.value):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=purpose,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=_reviewed_unlock(),
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("state", (None, "", "unlocked", "AUDITED_LOCKED", object()))
def test_missing_or_unknown_explicit_state_fails_closed(state: object) -> None:
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match="state"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.INFERENCE,
            state=state,
            unlock_record=_reviewed_unlock(),
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("purpose", (None, "", "unknown", object()))
def test_missing_or_unknown_purpose_fails_closed(purpose: object) -> None:
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match="purpose"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=purpose,
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("schema_version", 2),
        ("record_type", "other-record"),
        ("review_status", "pending"),
        ("reviewer", ""),
        ("reviewed_at", "not-a-date"),
        ("prerequisites", "accepted"),
    ),
)
def test_incomplete_or_malformed_unlock_record_fails_before_callback(
    field: str, invalid_value: object, tmp_path: Path
) -> None:
    record = _reviewed_unlock()
    record[field] = invalid_value
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match="unlock_record"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.METRIC,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            project_root=tmp_path,
            manifest_generation_root=tmp_path,
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("field", tuple(_reviewed_unlock()))
def test_unlock_record_rejects_every_missing_field(field: str, tmp_path: Path) -> None:
    record = _reviewed_unlock()
    del record[field]

    with pytest.raises(BenchmarkPolicyError, match="unlock_record"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.SR_OUTPUT,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            project_root=tmp_path,
            manifest_generation_root=tmp_path,
        )


def test_unlock_record_rejects_unknown_fields(tmp_path: Path) -> None:
    record = _reviewed_unlock()
    record["unknown"] = "must not be ignored"

    with pytest.raises(BenchmarkPolicyError, match="unlock_record"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.DEGRADATION,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            project_root=tmp_path,
            manifest_generation_root=tmp_path,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda descriptor: descriptor.pop("key"),
        lambda descriptor: descriptor.pop("role"),
        lambda descriptor: descriptor.__setitem__("key", "other"),
        lambda descriptor: descriptor.__setitem__("role", "training_data"),
    ),
)
def test_policy_derives_and_validates_smb_identity_from_descriptor(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    descriptor = copy.deepcopy(_smb_descriptor())
    mutate(descriptor)

    with pytest.raises(BenchmarkPolicyError, match="source_descriptor"):
        assert_smb_purpose_allowed(
            source_descriptor=descriptor,
            purpose=BenchmarkPurpose.METADATA_INSPECTION,
        )


def test_guard_without_callback_has_no_side_effect_contract() -> None:
    assert (
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.SCHEMA_AUDIT,
        )
        is None
    )


def test_artifact_unlock_contract_self_validates_and_matches_analysis_protocol() -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / "protocols" / "analysis-v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_ids = [item["id"] for item in protocol["smb_unlock_prerequisites"]]
    schema = load_schema("smb-evaluation-unlock", 1)
    prerequisite_schema = schema["properties"]["prerequisites"]

    assert protocol_ids == list(_PREREQUISITE_CONTRACTS)
    assert prerequisite_schema["required"] == protocol_ids
    assert set(prerequisite_schema["properties"]) == set(protocol_ids)
    validate_instance("smb-evaluation-unlock", _artifact_backed_unlock())


@pytest.mark.parametrize("prerequisite_id", tuple(_PREREQUISITE_CONTRACTS))
def test_artifact_unlock_contract_rejects_each_omitted_prerequisite(
    prerequisite_id: str,
) -> None:
    record = _artifact_backed_unlock()
    del record["prerequisites"][prerequisite_id]

    with pytest.raises(ContractValidationError):
        validate_instance("smb-evaluation-unlock", record)


@pytest.mark.parametrize("prerequisite_id", tuple(_PREREQUISITE_CONTRACTS))
def test_each_prerequisite_owns_its_artifact_contract(prerequisite_id: str) -> None:
    expected_kind, expected_schema, expected_version = _PREREQUISITE_CONTRACTS[prerequisite_id]
    reference = _artifact_backed_unlock()["prerequisites"][prerequisite_id]

    assert reference["artifact_kind"] == expected_kind
    assert reference["schema_id"] == expected_schema
    if isinstance(expected_version, tuple):
        assert reference["schema_version"] in expected_version
    else:
        assert reference["schema_version"] == expected_version


@pytest.mark.parametrize("prerequisite_id", tuple(_PREREQUISITE_CONTRACTS))
def test_prerequisite_rejects_contract_tuple_copied_from_another_key(
    prerequisite_id: str,
) -> None:
    record = _artifact_backed_unlock()
    other_id = next(key for key in _PREREQUISITE_CONTRACTS if key != prerequisite_id)
    other = record["prerequisites"][other_id]
    reference = record["prerequisites"][prerequisite_id]
    for field in ("artifact_kind", "schema_id", "schema_version"):
        reference[field] = other[field]

    with pytest.raises(ContractValidationError):
        validate_instance("smb-evaluation-unlock", record)


@pytest.mark.parametrize(
    ("mutation", "case"),
    (
        (lambda record: record["prerequisites"].__setitem__("unknown", {}), "added key"),
        (
            lambda record: record["prerequisites"]["methods_frozen"].__setitem__(
                "prerequisite_id", "metrics_frozen"
            ),
            "renamed identity",
        ),
        (
            lambda record: record["prerequisites"].__setitem__(
                "methods_frozen", record["prerequisites"]["metrics_frozen"]
            ),
            "duplicated reference",
        ),
        (
            lambda record: record["prerequisites"]["methods_frozen"].__setitem__(
                "artifact_path", "../controls/methods.json"
            ),
            "traversal path",
        ),
        (
            lambda record: record["prerequisites"]["methods_frozen"].__setitem__(
                "artifact_path", "/controls/methods.json"
            ),
            "absolute path",
        ),
        (
            lambda record: record["prerequisites"]["methods_frozen"].__setitem__(
                "artifact_sha256", "not-a-sha"
            ),
            "invalid digest",
        ),
        (
            lambda record: record["prerequisites"]["methods_frozen"].__setitem__("evidence_id", ""),
            "empty evidence identity",
        ),
        (
            lambda record: record["prerequisites"].__setitem__("methods_frozen", "accepted"),
            "bare acceptance string",
        ),
        (
            lambda record: record["prerequisites"]["evaluation_manifest_frozen"].pop(
                "expected_generation_id"
            ),
            "incomplete manifest identity",
        ),
    ),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_artifact_unlock_contract_rejects_malformed_references(
    mutation: Callable[[dict[str, Any]], object], case: str
) -> None:
    record = _artifact_backed_unlock()
    mutation(record)
    assert case

    with pytest.raises(ContractValidationError):
        validate_instance("smb-evaluation-unlock", record)


@pytest.mark.parametrize(
    "control_id",
    tuple(key for key in _PREREQUISITE_CONTRACTS if key != "evaluation_manifest_frozen"),
)
def test_each_non_manifest_freeze_control_branch_validates(control_id: str) -> None:
    validate_instance("smb-freeze-control", _freeze_control(control_id))


@pytest.mark.parametrize(
    ("control_id", "mutation"),
    (
        ("methods_frozen", lambda control: control.__setitem__("content", {})),
        (
            "methods_frozen",
            lambda control: control.__setitem__("artifact_kind", "metrics_control"),
        ),
        (
            "metrics_frozen",
            lambda control: control.__setitem__("control_id", "methods_frozen"),
        ),
        (
            "independent_units_frozen",
            lambda control: control["content"].pop("uncertainty_rule_id"),
        ),
        (
            "qualitative_samples_frozen",
            lambda control: control["content"].__setitem__("outcome_independent", False),
        ),
        (
            "human_unlock_recorded",
            lambda control: control["content"]["prerequisite_digests"].pop(
                "interpretation_rules_frozen"
            ),
        ),
    ),
)
def test_non_manifest_freeze_controls_reject_empty_mismatched_or_incomplete_content(
    control_id: str, mutation: Callable[[dict[str, Any]], object]
) -> None:
    control = _freeze_control(control_id)
    mutation(control)

    with pytest.raises(ContractValidationError):
        validate_instance("smb-freeze-control", control)


def test_legacy_arbitrary_string_unlock_never_reaches_metric_callback(tmp_path: Path) -> None:
    legacy_record = {
        "schema_version": 1,
        "record_type": "smb-evaluation-unlock",
        "review_status": "reviewed",
        "reviewer": "accountable-reviewer",
        "reviewed_at": "2026-08-18",
        "methods": ["fabricated-method"],
        "checkpoints": ["fabricated-checkpoint"],
        "degradations": ["fabricated-degradation"],
        "metrics": ["fabricated-metric"],
        "exclusions": ["fabricated-exclusion"],
        "sample": ["fabricated-sample"],
        "interpretation": ["fabricated-interpretation"],
    }
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.METRIC,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=legacy_record,
            project_root=tmp_path,
            manifest_generation_root=tmp_path,
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("prerequisite_id", tuple(_PREREQUISITE_CONTRACTS))
def test_runtime_gate_rejects_each_missing_prerequisite_artifact(
    prerequisite_id: str, tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"][prerequisite_id]
    (project_root / reference["artifact_path"]).unlink()

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


@pytest.mark.parametrize("prerequisite_id", tuple(_PREREQUISITE_CONTRACTS))
def test_runtime_gate_rejects_each_wrong_prerequisite_digest(
    prerequisite_id: str, tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    record["prerequisites"][prerequisite_id]["artifact_sha256"] = "0" * 64

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda reference: reference.__setitem__("artifact_path", "/etc/hostname"),
        lambda reference: reference.__setitem__("artifact_path", "../outside.json"),
    ),
)
def test_runtime_gate_rejects_absolute_and_traversal_paths_before_callback(
    mutation: Callable[[dict[str, Any]], object], tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    mutation(record["prerequisites"]["methods_frozen"])

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_runtime_gate_rejects_symlinked_artifact(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["methods_frozen"]
    original = project_root / reference["artifact_path"]
    external = tmp_path / "external-methods.json"
    external.write_bytes(original.read_bytes())
    symlink = project_root / "controls" / "symlink-methods.json"
    symlink.symlink_to(external)
    reference["artifact_path"] = symlink.relative_to(project_root).as_posix()
    reference["artifact_sha256"] = hashlib.sha256(external.read_bytes()).hexdigest()

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_runtime_gate_rejects_non_regular_artifact(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["methods_frozen"]
    directory = project_root / "controls" / "directory-artifact"
    directory.mkdir()
    reference["artifact_path"] = directory.relative_to(project_root).as_posix()
    reference["artifact_sha256"] = _HEX_A

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_kind", "metrics_control"),
        ("schema_id", "model-descriptor"),
        ("schema_version", 2),
    ),
)
def test_runtime_gate_rejects_caller_declared_contract_substitution(
    field: str, value: object, tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    record["prerequisites"]["methods_frozen"][field] = value

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_runtime_gate_rejects_cross_key_valid_artifact_substitution(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    methods = record["prerequisites"]["methods_frozen"]
    metrics = record["prerequisites"]["metrics_frozen"]
    methods["artifact_path"] = metrics["artifact_path"]
    methods["artifact_sha256"] = metrics["artifact_sha256"]

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_runtime_gate_rejects_semantically_invalid_correctly_hashed_artifact(
    tmp_path: Path,
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["methods_frozen"]
    artifact_path = project_root / reference["artifact_path"]
    control = json.loads(artifact_path.read_text(encoding="utf-8"))
    control["content"]["methods"] = []
    reference["artifact_sha256"] = _write_control(artifact_path, control)

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_manifest_pointer_hash_is_insufficient_when_generation_is_absent(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["evaluation_manifest_frozen"]
    pointer = yaml.safe_load((project_root / reference["artifact_path"]).read_text())
    descriptor = generation_root / pointer["generation_id"] / "manifest-descriptor.yaml"
    descriptor.unlink()

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_generation_id", "0" * 64),
        ("expected_records_sha256", "0" * 64),
        ("expected_row_count", 684),
    ),
)
def test_manifest_resolution_must_match_expected_identity(
    field: str, value: object, tmp_path: Path
) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    record["prerequisites"]["evaluation_manifest_frozen"][field] = value

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_not_yet_installed_manifest_contract_version_fails_closed(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    record["prerequisites"]["evaluation_manifest_frozen"]["schema_version"] = 2

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_manifest_resolution_rejects_non_locked_descriptor(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["evaluation_manifest_frozen"]
    pointer = yaml.safe_load((project_root / reference["artifact_path"]).read_text())
    descriptor_path = generation_root / pointer["generation_id"] / "manifest-descriptor.yaml"
    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    descriptor["benchmark_state"] = "EVALUATION_UNLOCKED"
    descriptor_path.write_text(yaml.safe_dump(descriptor, sort_keys=True), encoding="utf-8")

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_human_approval_must_digest_every_preceding_artifact(tmp_path: Path) -> None:
    record, project_root, generation_root = _synthetic_evaluation_gate(tmp_path)
    reference = record["prerequisites"]["human_unlock_recorded"]
    artifact_path = project_root / reference["artifact_path"]
    control = json.loads(artifact_path.read_text(encoding="utf-8"))
    control["content"]["prerequisite_digests"]["methods_frozen"] = "0" * 64
    reference["artifact_sha256"] = _write_control(artifact_path, control)

    _assert_gate_rejected(record=record, project_root=project_root, generation_root=generation_root)


def test_evaluation_unlock_requires_explicit_roots(tmp_path: Path) -> None:
    record, _, _ = _synthetic_evaluation_gate(tmp_path)
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.METRIC,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
            callback=lambda: calls.append("called"),
        )

    assert calls == []
