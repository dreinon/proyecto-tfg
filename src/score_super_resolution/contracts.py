"""Fail-closed validation for versioned scientific evidence contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "data" / "schemas"
_SCHEMA_ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DRAFT_2020_12_ID = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_URN_PREFIX = "urn:score-super-resolution:schema"


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


def validate_instance(schema_id: str, instance: Mapping[str, Any], version: int = 1) -> None:
    """Validate a mapping and raise one deterministic error containing every violation."""

    schema = load_schema(schema_id, version)
    if not isinstance(instance, Mapping):
        raise ContractValidationError(schema_id, version, ("instance $: must be a mapping",))

    validator = Draft202012Validator(schema)
    validation_errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
            error.message,
        ),
    )
    if not validation_errors:
        return

    details = []
    for error in validation_errors:
        location = _json_path(tuple(error.absolute_path))
        message = error.message[:1].lower() + error.message[1:]
        details.append(f"instance {location}: {message}")
    raise ContractValidationError(schema_id, version, details)
