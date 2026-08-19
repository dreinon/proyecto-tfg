from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import score_super_resolution.contracts as contracts
from score_super_resolution.contracts import (
    ContractValidationError,
    load_schema,
    validate_instance,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contracts"
SCHEMA_IDS = (
    "claim-evidence",
    "degradation-trace",
    "experiment-config",
    "failure",
    "method-output",
    "metric-result",
    "model-descriptor",
    "run-record",
    "source-descriptor",
)
EXECUTION_SCHEMA_IDS = {
    "degradation-trace",
    "experiment-config",
    "failure",
    "method-output",
    "metric-result",
    "run-record",
}
MODEL_FORBIDDEN_FIELDS = (
    "selected",
    "selection_score",
    "execution_command",
    "runtime_output",
    "runtime_outputs",
    "smb_result",
    "smb_results",
)


def _fixture(name: str) -> dict[str, dict[str, Any]]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def test_all_v1_schemas_self_validate() -> None:
    for schema_id in SCHEMA_IDS:
        schema = load_schema(schema_id)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"urn:score-super-resolution:schema:v1:{schema_id}"


def test_fixture_bundles_cover_every_non_manifest_schema_family() -> None:
    valid_schema_ids = {case["schema_id"] for case in _fixture("valid-records.json").values()}
    invalid_schema_ids = {case["schema_id"] for case in _fixture("invalid-records.json").values()}

    assert set(SCHEMA_IDS) <= valid_schema_ids
    assert set(SCHEMA_IDS) <= invalid_schema_ids


def test_execution_invalid_bundle_covers_missing_provenance_and_malformed_values() -> None:
    cases = _fixture("invalid-records.json")
    for schema_id in EXECUTION_SCHEMA_IDS:
        prefix = schema_id.replace("-", "_")
        assert f"{prefix}_missing_provenance" in cases
        assert f"{prefix}_malformed_value" in cases


@pytest.mark.parametrize("case", _fixture("valid-records.json").values())
def test_valid_contract_fixtures_pass(case: dict[str, Any]) -> None:
    validate_instance(case["schema_id"], case["instance"])


@pytest.mark.parametrize("case", _fixture("invalid-records.json").values())
def test_invalid_contract_fixtures_fail(case: dict[str, Any]) -> None:
    with pytest.raises(ContractValidationError):
        validate_instance(case["schema_id"], case["instance"])


@pytest.mark.parametrize("case", _fixture("valid-records.json").values())
def test_every_record_type_rejects_missing_and_unknown_fields(case: dict[str, Any]) -> None:
    missing = copy.deepcopy(case["instance"])
    del missing["record_type"]
    with pytest.raises(ContractValidationError):
        validate_instance(case["schema_id"], missing)

    unknown = copy.deepcopy(case["instance"])
    unknown["unknown_field"] = True
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance(case["schema_id"], unknown)


@pytest.mark.parametrize("case", _fixture("valid-records.json").values())
def test_every_record_type_rejects_an_invalid_declared_version(case: dict[str, Any]) -> None:
    instance = copy.deepcopy(case["instance"])
    instance["schema_version"] = 2

    with pytest.raises(ContractValidationError):
        validate_instance(case["schema_id"], instance)


def test_model_invalid_bundle_covers_every_required_provenance_field() -> None:
    cases = _fixture("invalid-records.json")
    required_cases = {
        "model_descriptor_missing_model_id",
        "model_descriptor_missing_source_locator",
        "model_descriptor_missing_source_version",
        "model_descriptor_missing_source_revision",
        "model_descriptor_missing_checkpoint",
        "model_descriptor_checkpoint_missing_identity",
        "model_descriptor_checkpoint_missing_checksum",
        "model_descriptor_missing_license",
        "model_descriptor_missing_input_assumptions",
        "model_descriptor_missing_output_conventions",
        "model_descriptor_missing_verification_date",
    }
    assert required_cases <= cases.keys()


@pytest.mark.parametrize("field", MODEL_FORBIDDEN_FIELDS)
def test_model_descriptor_forbids_selection_execution_and_results(field: str) -> None:
    instance = copy.deepcopy(
        _fixture("valid-records.json")["model_descriptor_checkpoint"]["instance"]
    )
    instance[field] = "forbidden"

    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("model-descriptor", instance)


def test_run_record_requires_complete_execution_provenance() -> None:
    instance = _fixture("valid-records.json")["run_record_non_learned"]["instance"]
    assert {
        "experiment_id",
        "execution_id",
        "started_at",
        "completed_at",
        "repositories",
        "manifest_ids",
        "experiment_config_id",
        "seeds",
        "environment",
        "hardware",
        "paths",
        "model_provenance",
    } <= instance.keys()


@pytest.mark.parametrize(
    "field",
    ("selected_model", "selection_score", "load_model", "execute_model", "smb_results"),
)
def test_run_record_forbids_model_selection_execution_and_results(field: str) -> None:
    instance = copy.deepcopy(
        _fixture("valid-records.json")["run_record_learned_method"]["instance"]
    )
    instance[field] = "forbidden"

    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("run-record", instance)


def test_claim_evidence_has_one_exact_shared_row_shape() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["claim_evidence_draft"]["instance"])
    assert set(instance) == {
        "schema_version",
        "record_type",
        "claim_id",
        "chapter_section",
        "status",
        "evidence_ids",
        "limitations",
        "reviewer",
        "review_date",
    }

    instance["unknown"] = "not part of the CSV contract"
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("claim-evidence", instance)


@pytest.mark.parametrize("field", ("evidence_ids", "limitations", "reviewer", "review_date"))
def test_reviewed_claim_requires_complete_review_evidence(field: str) -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["claim_evidence_reviewed"]["instance"])
    instance[field] = [] if field == "evidence_ids" else ""

    with pytest.raises(ContractValidationError):
        validate_instance("claim-evidence", instance)


def test_claim_review_date_rejects_month_99() -> None:
    instance = _fixture("invalid-records.json")["claim_evidence_impossible_month"]["instance"]

    with pytest.raises(ContractValidationError, match=r"\$\.review_date"):
        validate_instance("claim-evidence", instance)


def test_claim_review_date_rejects_invalid_day_for_month() -> None:
    instance = _fixture("invalid-records.json")["claim_evidence_impossible_day_for_month"][
        "instance"
    ]

    with pytest.raises(ContractValidationError, match=r"\$\.review_date"):
        validate_instance("claim-evidence", instance)


def test_model_verification_date_rejects_non_leap_february_29() -> None:
    instance = _fixture("invalid-records.json")["model_descriptor_non_leap_february_29"]["instance"]

    with pytest.raises(ContractValidationError, match=r"\$\.verification_date"):
        validate_instance("model-descriptor", instance)


def test_valid_leap_day_passes() -> None:
    instance = _fixture("valid-records.json")["claim_evidence_reviewed_leap_day"]["instance"]

    validate_instance("claim-evidence", instance)


def test_run_timestamp_rejects_hour_77() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["run_record_non_learned"]["instance"])
    instance["started_at"] = "2026-08-18T77:00:00Z"

    with pytest.raises(ContractValidationError, match=r"\$\.started_at"):
        validate_instance("run-record", instance)


def test_run_timestamp_rejects_missing_utc_z() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["run_record_non_learned"]["instance"])
    instance["started_at"] = "2026-08-18T12:00:00"

    with pytest.raises(ContractValidationError, match=r"\$\.started_at"):
        validate_instance("run-record", instance)


def test_run_timestamp_rejects_malformed_fractional_seconds() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["run_record_non_learned"]["instance"])
    instance["started_at"] = "2026-08-18T12:00:00.Z"

    with pytest.raises(ContractValidationError, match=r"\$\.started_at"):
        validate_instance("run-record", instance)


def test_run_interval_rejects_completed_before_started() -> None:
    instance = _fixture("invalid-records.json")["run_record_reversed_interval"]["instance"]

    with pytest.raises(ContractValidationError, match=r"\$\.completed_at"):
        validate_instance("run-record", instance)


def test_run_interval_accepts_equal_fractional_instants() -> None:
    instance = _fixture("valid-records.json")["run_record_equal_fractional_boundary"]["instance"]

    validate_instance("run-record", instance)


def test_run_interval_accepts_later_completed_at_without_fractional_seconds() -> None:
    instance = _fixture("valid-records.json")["run_record_non_learned"]["instance"]

    validate_instance("run-record", instance)


@pytest.mark.parametrize(
    ("schema_id", "version"),
    (
        ("../source-descriptor", 1),
        ("v1/source-descriptor", 1),
        ("/tmp/source-descriptor", 1),
        ("source_descriptor", 1),
        ("source-descriptor", 0),
        ("source-descriptor", -1),
        ("source-descriptor", True),
        ("source-descriptor", "1"),
    ),
)
def test_registry_rejects_unsafe_schema_locations(schema_id: str, version: object) -> None:
    with pytest.raises(ContractValidationError):
        load_schema(schema_id, version)  # type: ignore[arg-type]


def test_registry_reports_missing_schema_deterministically() -> None:
    messages = []
    for _ in range(2):
        with pytest.raises(ContractValidationError) as caught:
            load_schema("does-not-exist")
        messages.append(str(caught.value))

    assert messages[0] == messages[1]
    assert messages[0] == "does-not-exist@v1: schema: not found"


def test_registry_rejects_invalid_instance_versions() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["source_descriptor"]["instance"])
    instance["schema_version"] = 2

    with pytest.raises(ContractValidationError) as caught:
        validate_instance("source-descriptor", instance)

    assert caught.value.details
    assert any("schema_version" in detail for detail in caught.value.details)


def test_registry_reports_all_instance_errors_in_stable_order() -> None:
    instance = copy.deepcopy(_fixture("valid-records.json")["claim_evidence_draft"]["instance"])
    del instance["claim_id"]
    instance["unknown"] = True

    messages = []
    for _ in range(2):
        with pytest.raises(ContractValidationError) as caught:
            validate_instance("claim-evidence", instance)
        messages.append(str(caught.value))

    assert messages[0] == messages[1]
    assert "claim_id" in messages[0]
    assert "unknown" in messages[0]


def test_semantic_validation_runs_after_structural_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def semantic_marker(schema_id: str, _instance: object) -> list[str]:
        calls.append(schema_id)
        return ["instance $: semantic marker"]

    monkeypatch.setattr(contracts, "_semantic_validation_errors", semantic_marker)
    invalid = copy.deepcopy(_fixture("valid-records.json")["claim_evidence_draft"]["instance"])
    invalid["unknown"] = True
    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("claim-evidence", invalid)
    assert calls == []

    valid = _fixture("valid-records.json")["claim_evidence_draft"]["instance"]
    with pytest.raises(ContractValidationError, match="semantic marker"):
        validate_instance("claim-evidence", valid)
    assert calls == ["claim-evidence"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ("{", "invalid JSON"),
        ("[]", "root must be a JSON object"),
        (
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": "urn:score-super-resolution:schema:v1:broken",
                    "type": 7,
                }
            ),
            "invalid Draft 2020-12 schema",
        ),
    ),
)
def test_registry_rejects_malformed_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str, expected: str
) -> None:
    schema_root = tmp_path / "schemas"
    version_root = schema_root / "v1"
    version_root.mkdir(parents=True)
    (version_root / "broken.schema.json").write_text(payload, encoding="utf-8")
    monkeypatch.setattr(contracts, "SCHEMA_ROOT", schema_root)

    with pytest.raises(ContractValidationError, match=expected):
        load_schema("broken")


def test_registry_rejects_schema_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema_root = tmp_path / "schemas"
    version_root = schema_root / "v1"
    version_root.mkdir(parents=True)
    (version_root / "broken.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "urn:score-super-resolution:schema:v1:other",
                "type": "object",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(contracts, "SCHEMA_ROOT", schema_root)

    with pytest.raises(ContractValidationError, match="unexpected schema id"):
        load_schema("broken")
