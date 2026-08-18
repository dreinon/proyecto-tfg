"""Deterministic scientific identities and separate runtime-attempt identities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from score_super_resolution.contracts import validate_instance
from score_super_resolution.environment import environment_snapshot

HASH_ALGORITHM = "sha256"
CANONICALIZATION_VERSION = "json-utf8-sorted-keys-v1"
_EXPERIMENT_ID_PATTERN = re.compile(r"experiment-[0-9a-f]{64}\Z")
_SECRET_KEY_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
_SECRET_KEY_NAMES = {"api_key", "apikey", "access_key", "private_key"}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the project's versioned canonical byte representation."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_payload(kind: str, payload: Mapping[str, Any]) -> tuple[str, str]:
    digest = canonical_sha256(
        {
            "algorithm": HASH_ALGORITHM,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "identity_kind": kind,
            "payload": payload,
        }
    )
    return f"{kind}-{digest}", digest


def experiment_identity(
    scientific_controls: Mapping[str, Any],
    *,
    schema_id: str = "experiment-config",
    schema_version: int = 1,
) -> dict[str, str]:
    """Validate and identify scientific intent without runtime-attempt metadata."""

    validate_instance(schema_id, scientific_controls, schema_version)
    identity, digest = _identity_payload(
        "experiment",
        {
            "schema_id": schema_id,
            "schema_version": schema_version,
            "scientific_controls": scientific_controls,
        },
    )
    return {
        "experiment_id": identity,
        "sha256": digest,
        "algorithm": HASH_ALGORITHM,
        "canonicalization_version": CANONICALIZATION_VERSION,
    }


def _secret_like_key(path: Sequence[str], value: Any) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            parts = set(normalized.split("_"))
            if normalized in _SECRET_KEY_NAMES or parts & _SECRET_KEY_PARTS:
                return ".".join((*path, key))
            found = _secret_like_key((*path, key), child)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _secret_like_key((*path, str(index)), child)
            if found is not None:
                return found
    return None


def execution_identity(
    experiment_id: str,
    *,
    started_at: str,
    retry_nonce: str,
    hardware: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Identify one execution attempt while retaining its scientific-identity link."""

    if _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be a canonical experiment identity")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("started_at must be a non-empty string")
    if not isinstance(retry_nonce, str) or not retry_nonce:
        raise ValueError("retry_nonce must be a non-empty string")

    runtime_environment = environment if environment is not None else environment_snapshot()
    attempt = {
        "experiment_id": experiment_id,
        "started_at": started_at,
        "retry_nonce": retry_nonce,
        "environment": runtime_environment,
        "hardware": hardware,
    }
    secret_key = _secret_like_key((), attempt)
    if secret_key is not None:
        raise ValueError(f"execution metadata contains secret-like key: {secret_key}")

    identity, digest = _identity_payload("execution", attempt)
    return {
        "execution_id": identity,
        "experiment_id": experiment_id,
        "sha256": digest,
        "algorithm": HASH_ALGORITHM,
        "canonicalization_version": CANONICALIZATION_VERSION,
    }
