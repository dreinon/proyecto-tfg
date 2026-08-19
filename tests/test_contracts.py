from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

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

RECOVERY_REQUIRED_FIELDS = (
    "schema_version",
    "record_type",
    "manifest_id",
    "generation_id",
    "active_pointer_path",
    "active_pointer_sha256",
    "descriptor_path",
    "descriptor_sha256",
    "descriptor_yaml",
    "records_path",
    "row_schema_id",
    "row_schema_version",
    "row_count",
    "records_sha256",
    "source_revision",
    "source_provenance",
    "audit_version",
    "benchmark_state",
    "recovery_records_path",
    "compression",
    "compressed_sha256",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "recovery_command",
    "metadata_sha256",
)

RECOVERY_V2_REQUIRED_FIELDS = (
    "schema_version",
    "record_type",
    "bundle_id",
    "active_pointer_path",
    "manifest_id",
    "generation_id",
    "descriptor_path",
    "descriptor_sha256",
    "descriptor_yaml",
    "records_path",
    "row_schema_id",
    "row_schema_version",
    "row_count",
    "records_sha256",
    "source_revision",
    "source_provenance",
    "audit_version",
    "benchmark_state",
    "recovery_descriptor_path",
    "recovery_records_path",
    "compression",
    "compressed_sha256",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "metadata_sha256",
)


def _canonical_yaml(record: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        record, allow_unicode=True, sort_keys=True, default_flow_style=False
    ).encode("utf-8")


def _recovery_metadata_sha256(record: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "metadata_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"manifest-recovery-metadata-v1\0" + payload).hexdigest()


def _recovery_record() -> dict[str, Any]:
    descriptor = copy.deepcopy(
        json.loads(
            (Path(__file__).parent / "fixtures" / "smb" / "records.json").read_text(
                encoding="utf-8"
            )
        )["manifest_descriptor"]
    )
    descriptor.update(
        {
            "schema_version": 2,
            "manifest_id": "smb-evaluation-v2",
            "generation_algorithm": {
                **descriptor["generation_algorithm"],
                "version": 2,
                "domain_separator": "smb-manifest-generation-v2",
            },
            "source_provenance": {
                "source_set_version": 1,
                "algorithm": "sha256",
                "revision": "b" * 40,
                "dirty": True,
                "source_tree_sha256": "7" * 64,
                "patch_sha256": "8" * 64,
                "lock_sha256": "9" * 64,
            },
            "row_schema_version": 2,
            "audit_version": "smb-audit-v2",
            "review_inference": {
                "automated_population_audit_count": 685,
                "sampled_human_review_count": 64,
                "targeted_human_review_count": 0,
                "not_visually_reviewed_count": 621,
                "unavailable_visual_review_count": 0,
                "not_applicable_visual_review_count": 0,
                "exact_pair_automated_count": 0,
                "perceptual_pair_count": 0,
                "perceptual_pair_human_review_count": 0,
                "perceptual_pair_pending_count": 0,
                "inference_scope": "sample_observation_only",
                "population_prevalence_inference": "not_supported",
            },
        }
    )
    descriptor.pop("code_revision", None)
    records_bytes = b"{}\n"
    descriptor["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
    descriptor.pop("generation_id", None)
    generation_id = hashlib.sha256(
        b"smb-manifest-generation-v2\0" + _canonical_yaml(descriptor) + b"\0" + records_bytes
    ).hexdigest()
    descriptor["generation_id"] = generation_id
    descriptor_bytes = _canonical_yaml(descriptor)
    pointer = {
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
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed_buffer, mode="wb", filename="", compresslevel=9, mtime=0
    ) as handle:
        handle.write(records_bytes)
    compressed = compressed_buffer.getvalue()
    recovery = {
        "schema_version": 1,
        "record_type": "manifest-recovery",
        "manifest_id": "smb-evaluation-v2",
        "generation_id": generation_id,
        "active_pointer_path": "data/manifests/smb-evaluation-v1.yaml",
        "active_pointer_sha256": hashlib.sha256(_canonical_yaml(pointer)).hexdigest(),
        "descriptor_path": pointer["descriptor_path"],
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "descriptor_yaml": descriptor_bytes.decode("utf-8"),
        "records_path": pointer["records_path"],
        "row_schema_id": "manifest-row",
        "row_schema_version": 2,
        "row_count": 685,
        "records_sha256": descriptor["records_sha256"],
        "source_revision": descriptor["source_revision"],
        "source_provenance": descriptor["source_provenance"],
        "audit_version": "smb-audit-v2",
        "benchmark_state": "AUDITED_LOCKED",
        "recovery_records_path": ("data/manifests/smb-evaluation-v1-recovery.jsonl.gz"),
        "compression": {
            "algorithm": "gzip",
            "format_version": 1,
            "compresslevel": 9,
            "mtime": 0,
            "filename": "",
        },
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_size_bytes": len(compressed),
        "uncompressed_size_bytes": len(records_bytes),
        "recovery_command": (
            "uv run python -m score_super_resolution.smb_audit recover-active "
            "--manifest-active data/manifests/smb-evaluation-v1.yaml "
            "--recovery-descriptor data/manifests/smb-evaluation-v1-recovery.yaml "
            "--recovery-records data/manifests/smb-evaluation-v1-recovery.jsonl.gz "
            "--manifest-generation-root artifacts/smb-manifests/generations"
        ),
        "metadata_sha256": "",
    }
    recovery["metadata_sha256"] = _recovery_metadata_sha256(recovery)
    return recovery


def _resign_recovery(record: dict[str, Any]) -> None:
    record["metadata_sha256"] = _recovery_metadata_sha256(record)


def _v2_bundle_id(record: dict[str, Any]) -> str:
    excluded = {
        "bundle_id",
        "recovery_descriptor_path",
        "recovery_records_path",
        "metadata_sha256",
    }
    core = {key: value for key, value in record.items() if key not in excluded}
    payload = json.dumps(
        core,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"manifest-recovery-bundle-v2\0" + payload).hexdigest()


def _recovery_v2_record() -> dict[str, Any]:
    legacy = _recovery_record()
    descriptor = yaml.safe_load(legacy["descriptor_yaml"])
    descriptor["hash_provenance"]["pixels"] = {
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
    descriptor["duplicate_provenance"]["exact"] = {
        "algorithm": "canonical-pixel-sha256",
        "version": 2,
    }
    records_bytes = b"{}\n"
    descriptor.pop("generation_id", None)
    generation_id = hashlib.sha256(
        b"smb-manifest-generation-v2\0" + _canonical_yaml(descriptor) + b"\0" + records_bytes
    ).hexdigest()
    descriptor["generation_id"] = generation_id
    descriptor_bytes = _canonical_yaml(descriptor)
    recovery = {
        key: value
        for key, value in legacy.items()
        if key not in {"active_pointer_sha256", "recovery_command"}
    }
    recovery.update(
        {
            "schema_version": 2,
            "descriptor_yaml": descriptor_bytes.decode("utf-8"),
            "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
            "generation_id": generation_id,
            "descriptor_path": f"{generation_id}/manifest-descriptor.yaml",
            "records_path": f"{generation_id}/manifest-records.jsonl",
            "bundle_id": "",
            "recovery_descriptor_path": "",
            "recovery_records_path": "",
            "metadata_sha256": "",
        }
    )
    recovery["bundle_id"] = _v2_bundle_id(recovery)
    prefix = f"data/manifests/recovery/canonical-pixel-v2/{recovery['bundle_id']}"
    recovery["recovery_descriptor_path"] = f"{prefix}/manifest-recovery.yaml"
    recovery["recovery_records_path"] = f"{prefix}/manifest-records.jsonl.gz"
    recovery["metadata_sha256"] = contracts.recovery_metadata_sha256(recovery, version=2)
    return recovery


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


def test_manifest_recovery_contract_accepts_complete_exact_mapping() -> None:
    recovery = _recovery_record()

    validate_instance("manifest-recovery", recovery, version=1)


@pytest.mark.parametrize("field", RECOVERY_REQUIRED_FIELDS)
def test_manifest_recovery_contract_rejects_every_missing_field(field: str) -> None:
    recovery = _recovery_record()
    del recovery[field]

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", recovery, version=1)


def test_manifest_recovery_contract_rejects_unknown_fields() -> None:
    recovery = _recovery_record()
    recovery["unknown_field"] = True

    with pytest.raises(ContractValidationError, match="additional properties"):
        validate_instance("manifest-recovery", recovery, version=1)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 2),
        ("record_type", "manifest-active"),
        ("generation_id", "not-a-digest"),
        ("active_pointer_path", "/tmp/active.yaml"),
        ("descriptor_path", "../manifest-descriptor.yaml"),
        ("recovery_records_path", "../recovery.gz"),
        ("compressed_sha256", "f" * 63),
        ("compressed_size_bytes", 2_097_153),
        ("uncompressed_size_bytes", 16_777_217),
    ),
)
def test_manifest_recovery_contract_rejects_malformed_or_oversized_fields(
    field: str, value: object
) -> None:
    recovery = _recovery_record()
    recovery[field] = value
    _resign_recovery(recovery)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", recovery, version=1)


@pytest.mark.parametrize(
    "mutation",
    (
        "generation_id",
        "active_pointer_sha256",
        "descriptor_sha256",
        "records_sha256",
        "row_count",
        "row_schema_version",
        "source_revision",
        "source_tree_sha256",
        "patch_sha256",
        "lock_sha256",
        "audit_version",
        "benchmark_state",
    ),
)
def test_manifest_recovery_contract_rejects_independent_cross_field_mutations(
    mutation: str,
) -> None:
    recovery = _recovery_record()
    if mutation in {"source_tree_sha256", "patch_sha256", "lock_sha256"}:
        recovery["source_provenance"][mutation] = "e" * 64
    elif mutation == "row_count":
        recovery[mutation] = 684
    elif mutation == "row_schema_version":
        recovery[mutation] = 1
    elif mutation == "audit_version":
        recovery[mutation] = "smb-audit-v1"
    elif mutation == "benchmark_state":
        recovery[mutation] = "LOCKED"
    elif mutation == "source_revision":
        recovery[mutation] = "e" * 40
    else:
        recovery[mutation] = "e" * 64
    _resign_recovery(recovery)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", recovery, version=1)


def test_manifest_recovery_contract_rejects_embedded_descriptor_mutation() -> None:
    recovery = _recovery_record()
    descriptor = yaml.safe_load(recovery["descriptor_yaml"])
    descriptor["source_revision"] = "e" * 40
    recovery["descriptor_yaml"] = _canonical_yaml(descriptor).decode("utf-8")
    recovery["descriptor_sha256"] = hashlib.sha256(
        recovery["descriptor_yaml"].encode("utf-8")
    ).hexdigest()
    _resign_recovery(recovery)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", recovery, version=1)


def test_manifest_recovery_v2_contract_accepts_non_cyclic_bundle() -> None:
    recovery = _recovery_v2_record()

    validate_instance("manifest-recovery", recovery, version=2)

    assert "active_pointer_sha256" not in recovery
    assert recovery["recovery_descriptor_path"].endswith(
        f"/{recovery['bundle_id']}/manifest-recovery.yaml"
    )
    assert recovery["recovery_records_path"].endswith(
        f"/{recovery['bundle_id']}/manifest-records.jsonl.gz"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "active_pointer_sha256",
        "bundle_id",
        "descriptor_path",
        "records_path",
        "recovery_descriptor_path",
        "recovery_records_path",
        "descriptor_sha256",
        "records_sha256",
        "compressed_sha256",
        "metadata_sha256",
    ),
)
def test_manifest_recovery_v2_rejects_cyclic_or_mismatched_identity(mutation: str) -> None:
    recovery = _recovery_v2_record()
    if mutation == "active_pointer_sha256":
        recovery[mutation] = "f" * 64
    elif mutation == "bundle_id":
        recovery[mutation] = "f" * 64
    elif mutation in {"descriptor_path", "records_path"}:
        recovery[mutation] = recovery[mutation].replace(recovery["generation_id"], "f" * 64)
    elif mutation in {"recovery_descriptor_path", "recovery_records_path"}:
        recovery[mutation] = recovery[mutation].replace(recovery["bundle_id"], "f" * 64)
    else:
        recovery[mutation] = "f" * 64
    if mutation != "metadata_sha256":
        recovery["metadata_sha256"] = contracts.recovery_metadata_sha256(recovery, version=2)

    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", recovery, version=2)


def test_manifest_recovery_versions_do_not_cross_validate() -> None:
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", _recovery_record(), version=2)
    with pytest.raises(ContractValidationError):
        validate_instance("manifest-recovery", _recovery_v2_record(), version=1)
