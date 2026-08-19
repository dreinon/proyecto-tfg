"""Fail-closed validation for versioned scientific evidence contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "data" / "schemas"
_SCHEMA_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DRAFT_2020_12_ID = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_URN_PREFIX = "urn:score-super-resolution:schema"
_RECOVERY_METADATA_DOMAINS = {
    1: b"manifest-recovery-metadata-v1\0",
    2: b"manifest-recovery-metadata-v2\0",
}
_RECOVERY_BUNDLE_DOMAIN_V2 = b"manifest-recovery-bundle-v2\0"


class ContractValidationError(ValueError):
    """Report deterministic schema lookup or instance validation failures."""

    def __init__(self, schema_id: str, version: int, details: Sequence[str]) -> None:
        self.schema_id = schema_id
        self.version = version
        self.details = tuple(details)
        super().__init__(f"{schema_id}@v{version}: {'; '.join(self.details)}")


def _raise(schema_id: object, version: object, detail: str) -> None:
    display_id = schema_id if isinstance(schema_id, str) else repr(schema_id)
    display_version = version if isinstance(version, int) and not isinstance(version, bool) else 0
    raise ContractValidationError(display_id, display_version, (detail,))


def _validated_locator(schema_id: str, version: int) -> tuple[str, int]:
    if not isinstance(schema_id, str) or _SCHEMA_ID_PATTERN.fullmatch(schema_id) is None:
        _raise(schema_id, version, "schema_id: must be a lowercase hyphenated identifier")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        _raise(schema_id, version, "version: must be a positive integer")
    return schema_id, version


def _expected_schema_id(schema_id: str, version: int) -> str:
    return f"{_SCHEMA_URN_PREFIX}:v{version}:{schema_id}"


def _schema_path(schema_id: str, version: int) -> Path:
    root = SCHEMA_ROOT.resolve()
    path = (root / f"v{version}" / f"{schema_id}.schema.json").resolve()
    if not path.is_relative_to(root):
        raise ContractValidationError(schema_id, version, ("schema: path escapes schema root",))
    return path


def _json_path(parts: Sequence[object]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part.isidentifier():
            path += f".{part}"
        else:
            escaped = str(part).replace("~", "~0").replace("/", "~1")
            path += f"/{escaped}"
    return path


def _schema_contract_errors(schema_id: str, version: int, schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("$schema") != _DRAFT_2020_12_ID:
        errors.append(f"schema.$schema: must equal {_DRAFT_2020_12_ID}")
    expected_id = _expected_schema_id(schema_id, version)
    if schema.get("$id") != expected_id:
        errors.append(f"schema.$id: unexpected schema id (expected {expected_id})")

    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping):
        errors.append("schema.properties: must define a record object")
    else:
        schema_version = properties.get("schema_version")
        record_type = properties.get("record_type")
        if not isinstance(schema_version, Mapping) or schema_version.get("const") != version:
            errors.append(f"schema.properties.schema_version: const must equal {version}")
        if not isinstance(record_type, Mapping) or record_type.get("const") != schema_id:
            errors.append(f"schema.properties.record_type: const must equal {schema_id}")
    if not isinstance(required, list) or not {"schema_version", "record_type"} <= set(required):
        errors.append("schema.required: must include schema_version and record_type")
    if schema.get("additionalProperties") is not False:
        errors.append("schema.additionalProperties: must be false")
    return sorted(errors)


def load_schema(schema_id: str, version: int = 1) -> dict[str, Any]:
    """Load and self-check one root-confined Draft 2020-12 schema.

    The registry is file-backed rather than allowlisted. Later owners can register distinct
    ``manifest-active``, ``manifest-descriptor``, and ``manifest-row`` contracts by adding their
    versioned schema files without weakening or bypassing this resolver.
    """

    schema_id, version = _validated_locator(schema_id, version)
    path = _schema_path(schema_id, version)
    if not path.is_file():
        raise ContractValidationError(schema_id, version, ("schema: not found",))

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        detail = f"schema: invalid JSON at line {error.lineno}, column {error.colno}"
        raise ContractValidationError(schema_id, version, (detail,)) from None
    except OSError as error:
        raise ContractValidationError(
            schema_id, version, (f"schema: cannot read ({error.strerror})",)
        ) from None

    if not isinstance(loaded, dict):
        raise ContractValidationError(schema_id, version, ("schema: root must be a JSON object",))

    try:
        Draft202012Validator.check_schema(loaded)
    except SchemaError as error:
        location = _json_path(tuple(error.absolute_path))
        detail = f"schema {location}: invalid Draft 2020-12 schema ({error.message})"
        raise ContractValidationError(schema_id, version, (detail,)) from None

    contract_errors = _schema_contract_errors(schema_id, version, loaded)
    if contract_errors:
        raise ContractValidationError(schema_id, version, contract_errors)
    return loaded


@lru_cache(maxsize=64)
def _cached_validator(
    schema_id: str, version: int, _schema_root_identity: str
) -> Draft202012Validator:
    """Reuse a self-checked immutable schema for row-heavy manifest validation."""

    return Draft202012Validator(load_schema(schema_id, version))


def _parse_date_field(
    instance: Mapping[str, Any], field: str, *, allow_empty: bool = False
) -> tuple[date | None, list[str]]:
    value = instance[field]
    if allow_empty and value == "":
        return None, []
    try:
        return date.fromisoformat(value), []
    except ValueError:
        return None, [f"instance $.{field}: must be a real ISO calendar date"]


def _parse_utc_timestamp_field(
    instance: Mapping[str, Any], field: str
) -> tuple[datetime | None, list[str]]:
    value = instance[field]
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return None, [f"instance $.{field}: must be a real canonical UTC timestamp"]
    return parsed, []


def _manifest_row_semantic_errors(instance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_item_id = f"smb-test-{instance['upstream_index']:06d}"
    if instance["item_id"] != expected_item_id:
        errors.append(f"instance $.item_id: must equal {expected_item_id} for $.upstream_index")

    processing_status = instance["processing_status"]
    expected_status = "processable" if processing_status == "processed" else "unprocessable"
    if instance["expected_status"] != expected_status:
        errors.append(
            f"instance $.expected_status: must be {expected_status} when "
            f"$.processing_status is {processing_status}"
        )

    unprocessable_reason = instance["unprocessable_reason"]
    if processing_status == "processed" and unprocessable_reason is not None:
        errors.append(
            "instance $.unprocessable_reason: must be null when $.processing_status is processed"
        )
    elif processing_status == "failed" and unprocessable_reason is None:
        errors.append(
            "instance $.unprocessable_reason: must name the failure when "
            "$.processing_status is failed"
        )

    paired_eligible = instance["paired_eligible"]
    paired_reason = instance["paired_ineligibility_reason"]
    if paired_eligible and paired_reason is not None:
        errors.append(
            "instance $.paired_ineligibility_reason: must be null when $.paired_eligible is true"
        )
    elif not paired_eligible and paired_reason is None:
        errors.append(
            "instance $.paired_ineligibility_reason: must name the exclusion when "
            "$.paired_eligible is false"
        )

    candidate_ids = instance["near_duplicate_candidate_ids"]
    if candidate_ids != sorted(set(candidate_ids)):
        errors.append(
            "instance $.near_duplicate_candidate_ids: must be unique and in canonical order"
        )

    if instance["schema_version"] == 2:
        visual_review = instance["visual_review"]
        if visual_review["status"] in {
            "sampled_human_reviewed",
            "targeted_human_reviewed",
        }:
            try:
                date.fromisoformat(visual_review["reviewed_at"])
            except ValueError:
                errors.append(
                    "instance $.visual_review.reviewed_at: must be a real ISO calendar date"
                )

        for index, relation in enumerate(instance["duplicate_relations"]):
            if relation["candidate_type"] == "perceptual" and relation["disposition"] in {
                "distinct",
                "duplicate",
                "related",
            }:
                try:
                    date.fromisoformat(relation["reviewed_at"])
                except ValueError:
                    errors.append(
                        "instance "
                        f"$.duplicate_relations[{index}].reviewed_at: "
                        "must be a real ISO calendar date"
                    )
    return errors


def _canonical_yaml(record: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(record), allow_unicode=True, sort_keys=True, default_flow_style=False
    ).encode("utf-8")


def recovery_metadata_sha256(instance: Mapping[str, Any], *, version: int = 1) -> str:
    """Hash the complete recovery mapping except its self-address field."""

    if isinstance(version, bool) or version not in _RECOVERY_METADATA_DOMAINS:
        raise ValueError("unsupported manifest recovery metadata version")
    unsigned = {key: value for key, value in instance.items() if key != "metadata_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_RECOVERY_METADATA_DOMAINS[version] + payload).hexdigest()


def recovery_bundle_id_v2(instance: Mapping[str, Any]) -> str:
    """Address the non-cyclic semantic core of one recovery-v2 bundle."""

    excluded = {
        "bundle_id",
        "recovery_descriptor_path",
        "recovery_records_path",
        "metadata_sha256",
    }
    core = {key: value for key, value in instance.items() if key not in excluded}
    payload = json.dumps(
        core,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_RECOVERY_BUNDLE_DOMAIN_V2 + payload).hexdigest()


def _manifest_recovery_v1_semantic_errors(instance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    descriptor_text = instance["descriptor_yaml"]
    try:
        descriptor = yaml.safe_load(descriptor_text)
    except yaml.YAMLError:
        return ["instance $.descriptor_yaml: must contain canonical YAML"]
    if not isinstance(descriptor, dict):
        return ["instance $.descriptor_yaml: must contain a descriptor mapping"]
    descriptor_bytes = descriptor_text.encode("utf-8")
    if descriptor_bytes != _canonical_yaml(descriptor):
        errors.append("instance $.descriptor_yaml: must use canonical YAML bytes")
    try:
        validate_instance("manifest-descriptor", descriptor, version=2)
    except ContractValidationError as error:
        errors.append(f"instance $.descriptor_yaml: invalid embedded descriptor ({error})")
        return sorted(errors)

    descriptor_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
    if instance["descriptor_sha256"] != descriptor_sha256:
        errors.append("instance $.descriptor_sha256: must hash $.descriptor_yaml")

    matching_fields = (
        "manifest_id",
        "generation_id",
        "row_schema_id",
        "row_schema_version",
        "row_count",
        "records_sha256",
        "source_revision",
        "source_provenance",
        "audit_version",
        "benchmark_state",
    )
    for field in matching_fields:
        if instance[field] != descriptor[field]:
            errors.append(f"instance $.{field}: must equal embedded descriptor $.{field}")

    generation_id = instance["generation_id"]
    expected_descriptor_path = f"{generation_id}/manifest-descriptor.yaml"
    expected_records_path = f"{generation_id}/manifest-records.jsonl"
    if instance["descriptor_path"] != expected_descriptor_path:
        errors.append("instance $.descriptor_path: must name the recovered generation")
    if instance["records_path"] != expected_records_path:
        errors.append("instance $.records_path: must name the recovered generation")

    pointer = {
        "schema_version": 2,
        "record_type": "manifest-active",
        "manifest_id": instance["manifest_id"],
        "generation_id": generation_id,
        "descriptor_path": instance["descriptor_path"],
        "records_path": instance["records_path"],
        "row_schema_id": instance["row_schema_id"],
        "row_schema_version": instance["row_schema_version"],
        "row_count": instance["row_count"],
        "records_sha256": instance["records_sha256"],
    }
    try:
        validate_instance("manifest-active", pointer, version=2)
    except ContractValidationError as error:
        errors.append(f"instance $: invalid reconstructed active pointer ({error})")
    pointer_sha256 = hashlib.sha256(_canonical_yaml(pointer)).hexdigest()
    if instance["active_pointer_sha256"] != pointer_sha256:
        errors.append(
            "instance $.active_pointer_sha256: must hash the reconstructed active pointer"
        )
    if instance["metadata_sha256"] != recovery_metadata_sha256(instance):
        errors.append("instance $.metadata_sha256: must address the canonical recovery metadata")
    return sorted(errors)


def _manifest_recovery_v2_semantic_errors(instance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    descriptor_text = instance["descriptor_yaml"]
    try:
        descriptor = yaml.safe_load(descriptor_text)
    except yaml.YAMLError:
        return ["instance $.descriptor_yaml: must contain canonical YAML"]
    if not isinstance(descriptor, dict):
        return ["instance $.descriptor_yaml: must contain a descriptor mapping"]
    descriptor_bytes = descriptor_text.encode("utf-8")
    if descriptor_bytes != _canonical_yaml(descriptor):
        errors.append("instance $.descriptor_yaml: must use canonical YAML bytes")
    try:
        validate_instance("manifest-descriptor", descriptor, version=2)
    except ContractValidationError as error:
        errors.append(f"instance $.descriptor_yaml: invalid embedded descriptor ({error})")
        return sorted(errors)

    if instance["descriptor_sha256"] != hashlib.sha256(descriptor_bytes).hexdigest():
        errors.append("instance $.descriptor_sha256: must hash $.descriptor_yaml")
    for field in (
        "manifest_id",
        "generation_id",
        "row_schema_id",
        "row_schema_version",
        "row_count",
        "records_sha256",
        "source_revision",
        "source_provenance",
        "audit_version",
        "benchmark_state",
    ):
        if instance[field] != descriptor[field]:
            errors.append(f"instance $.{field}: must equal embedded descriptor $.{field}")

    generation_id = instance["generation_id"]
    if instance["descriptor_path"] != f"{generation_id}/manifest-descriptor.yaml":
        errors.append("instance $.descriptor_path: must name the recovered generation")
    if instance["records_path"] != f"{generation_id}/manifest-records.jsonl":
        errors.append("instance $.records_path: must name the recovered generation")

    expected_bundle_id = recovery_bundle_id_v2(instance)
    if instance["bundle_id"] != expected_bundle_id:
        errors.append("instance $.bundle_id: must address the non-cyclic recovery core")
    prefix = f"data/manifests/recovery/canonical-pixel-v2/{instance['bundle_id']}"
    if instance["recovery_descriptor_path"] != f"{prefix}/manifest-recovery.yaml":
        errors.append("instance $.recovery_descriptor_path: must be bundle-addressed")
    if instance["recovery_records_path"] != f"{prefix}/manifest-records.jsonl.gz":
        errors.append("instance $.recovery_records_path: must be bundle-addressed")
    if instance["metadata_sha256"] != recovery_metadata_sha256(instance, version=2):
        errors.append("instance $.metadata_sha256: must address recovery-v2 metadata")

    recovery_descriptor_sha256 = hashlib.sha256(_canonical_yaml(instance)).hexdigest()
    pointer = {
        "schema_version": 2,
        "record_type": "manifest-active",
        "manifest_id": instance["manifest_id"],
        "generation_id": generation_id,
        "descriptor_path": instance["descriptor_path"],
        "records_path": instance["records_path"],
        "row_schema_id": instance["row_schema_id"],
        "row_schema_version": instance["row_schema_version"],
        "row_count": instance["row_count"],
        "records_sha256": instance["records_sha256"],
        "recovery_descriptor_path": instance["recovery_descriptor_path"],
        "recovery_descriptor_sha256": recovery_descriptor_sha256,
        "recovery_records_path": instance["recovery_records_path"],
        "recovery_records_sha256": instance["compressed_sha256"],
    }
    try:
        validate_instance("manifest-active", pointer, version=2)
    except ContractValidationError as error:
        errors.append(f"instance $: invalid reconstructed active pointer ({error})")
    return sorted(errors)


def _semantic_validation_errors(
    schema_id: str, version: int, instance: Mapping[str, Any]
) -> list[str]:
    if schema_id == "claim-evidence":
        _, errors = _parse_date_field(instance, "review_date", allow_empty=True)
        return errors
    if schema_id == "model-descriptor":
        _, errors = _parse_date_field(instance, "verification_date")
        return errors
    if schema_id == "manifest-row":
        return _manifest_row_semantic_errors(instance)
    if schema_id == "manifest-recovery":
        return (
            _manifest_recovery_v1_semantic_errors(instance)
            if version == 1
            else _manifest_recovery_v2_semantic_errors(instance)
        )
    if schema_id != "run-record":
        return []

    started_at, started_errors = _parse_utc_timestamp_field(instance, "started_at")
    completed_at, completed_errors = _parse_utc_timestamp_field(instance, "completed_at")
    errors = [*started_errors, *completed_errors]
    if not errors and completed_at < started_at:
        errors.append("instance $.completed_at: must not be earlier than $.started_at")
    return errors


def validate_instance(schema_id: str, instance: Mapping[str, Any], version: int = 1) -> None:
    """Validate structural and schema-specific semantic evidence constraints."""

    schema_id, version = _validated_locator(schema_id, version)
    if not isinstance(instance, Mapping):
        raise ContractValidationError(schema_id, version, ("instance $: must be a mapping",))

    validator = _cached_validator(schema_id, version, str(SCHEMA_ROOT.resolve()))
    validation_errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            error.message,
        ),
    )
    if validation_errors:
        details = []
        for error in validation_errors:
            location = _json_path(tuple(error.absolute_path))
            message = error.message[:1].lower() + error.message[1:]
            details.append(f"instance {location}: {message}")
        raise ContractValidationError(schema_id, version, details)

    semantic_errors = _semantic_validation_errors(schema_id, version, instance)
    if semantic_errors:
        raise ContractValidationError(schema_id, version, semantic_errors)
