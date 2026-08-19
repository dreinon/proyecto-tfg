from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from score_super_resolution.contracts import (
    ContractValidationError,
    load_schema,
    validate_instance,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "smb" / "records.json"
SCHEMA_IDS = ("manifest-active", "manifest-descriptor", "manifest-row")


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
