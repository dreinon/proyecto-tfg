"""Fail-closed purpose and unlock policy for the SMB evaluation benchmark."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from enum import StrEnum


class BenchmarkPolicyError(PermissionError):
    """Report a rejected or malformed SMB policy request."""


class BenchmarkPurpose(StrEnum):
    """Every supported reason for crossing the SMB policy boundary."""

    METADATA_INSPECTION = "metadata_inspection"
    SCHEMA_AUDIT = "schema_audit"
    CONTENT_AUDIT = "content_audit"
    GROUPING_AUDIT = "grouping_audit"
    PROVENANCE_AUDIT = "provenance_audit"
    RIGHTS_AUDIT = "rights_audit"
    SUITABILITY_AUDIT = "suitability_audit"
    QUALITY_AUDIT = "quality_audit"
    DUPLICATE_AUDIT = "duplicate_audit"
    FIXED_VISUAL_AUDIT = "fixed_visual_audit"

    DEGRADATION = "degradation"
    INFERENCE = "inference"
    SR_OUTPUT = "sr_output"
    METRIC = "metric"

    TUNING = "tuning"
    FITTING = "fitting"
    SELECTION = "selection"
    OUTCOME_SAMPLING = "outcome_sampling"


class BenchmarkState(StrEnum):
    """States relevant to the SMB outcome quarantine."""

    AUDIT_LOCKED = "AUDIT_LOCKED"
    EVALUATION_UNLOCKED = "EVALUATION_UNLOCKED"


AUDIT_PURPOSES = frozenset(
    {
        BenchmarkPurpose.METADATA_INSPECTION,
        BenchmarkPurpose.SCHEMA_AUDIT,
        BenchmarkPurpose.CONTENT_AUDIT,
        BenchmarkPurpose.GROUPING_AUDIT,
        BenchmarkPurpose.PROVENANCE_AUDIT,
        BenchmarkPurpose.RIGHTS_AUDIT,
        BenchmarkPurpose.SUITABILITY_AUDIT,
        BenchmarkPurpose.QUALITY_AUDIT,
        BenchmarkPurpose.DUPLICATE_AUDIT,
        BenchmarkPurpose.FIXED_VISUAL_AUDIT,
    }
)

FINAL_EVALUATION_PURPOSES = frozenset(
    {
        BenchmarkPurpose.DEGRADATION,
        BenchmarkPurpose.INFERENCE,
        BenchmarkPurpose.SR_OUTPUT,
        BenchmarkPurpose.METRIC,
    }
)

NEVER_ALLOWED_PURPOSES = frozenset(
    {
        BenchmarkPurpose.TUNING,
        BenchmarkPurpose.FITTING,
        BenchmarkPurpose.SELECTION,
        BenchmarkPurpose.OUTCOME_SAMPLING,
    }
)

_SMB_KEY = "smb"
_SMB_ROLE = "evaluation_benchmark"
_UNLOCK_RECORD_TYPE = "smb-evaluation-unlock"
_UNLOCK_IDENTITY_FIELDS = frozenset(
    {
        "methods",
        "checkpoints",
        "degradations",
        "metrics",
        "exclusions",
        "sample",
        "interpretation",
    }
)
_UNLOCK_FIELDS = (
    frozenset(
        {
            "schema_version",
            "record_type",
            "review_status",
            "reviewer",
            "reviewed_at",
        }
    )
    | _UNLOCK_IDENTITY_FIELDS
)


def _policy_error(detail: str) -> BenchmarkPolicyError:
    return BenchmarkPolicyError(f"SMB benchmark policy rejected: {detail}")


def _validate_source_descriptor(source_descriptor: object) -> None:
    if not isinstance(source_descriptor, Mapping):
        raise _policy_error("source_descriptor must be a parsed mapping")
    if source_descriptor.get("key") != _SMB_KEY:
        raise _policy_error("source_descriptor.key must identify smb")
    if source_descriptor.get("role") != _SMB_ROLE:
        raise _policy_error("source_descriptor.role must be evaluation_benchmark")


def _parse_purpose(purpose: object) -> BenchmarkPurpose:
    if isinstance(purpose, BenchmarkPurpose):
        return purpose
    if isinstance(purpose, str):
        try:
            return BenchmarkPurpose(purpose)
        except ValueError:
            pass
    raise _policy_error("purpose is missing or unknown")


def _parse_state(state: object) -> BenchmarkState:
    if isinstance(state, BenchmarkState):
        return state
    if isinstance(state, str):
        try:
            return BenchmarkState(state)
        except ValueError:
            pass
    raise _policy_error("state is missing or unknown")


def _validate_frozen_identities(field: str, value: object) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise _policy_error(
            f"unlock_record.{field} must be a non-empty sequence of unique frozen identities"
        )


def _validate_unlock_record(unlock_record: object) -> None:
    if not isinstance(unlock_record, Mapping):
        raise _policy_error("unlock_record must be a complete reviewed mapping")

    fields = set(unlock_record)
    if fields != _UNLOCK_FIELDS:
        missing = sorted(_UNLOCK_FIELDS - fields)
        unknown = sorted(fields - _UNLOCK_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise _policy_error(f"unlock_record has {' and '.join(details)}")

    if unlock_record["schema_version"] != 1 or isinstance(unlock_record["schema_version"], bool):
        raise _policy_error("unlock_record.schema_version must equal 1")
    if unlock_record["record_type"] != _UNLOCK_RECORD_TYPE:
        raise _policy_error(f"unlock_record.record_type must equal {_UNLOCK_RECORD_TYPE}")
    if unlock_record["review_status"] != "reviewed":
        raise _policy_error("unlock_record.review_status must equal reviewed")

    reviewer = unlock_record["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise _policy_error("unlock_record.reviewer must identify the accountable reviewer")

    reviewed_at = unlock_record["reviewed_at"]
    if not isinstance(reviewed_at, str):
        raise _policy_error("unlock_record.reviewed_at must be an ISO date")
    try:
        date.fromisoformat(reviewed_at)
    except ValueError:
        raise _policy_error("unlock_record.reviewed_at must be an ISO date") from None

    for field in sorted(_UNLOCK_IDENTITY_FIELDS):
        _validate_frozen_identities(field, unlock_record[field])


def assert_smb_purpose_allowed[T](
    *,
    source_descriptor: Mapping[str, object],
    purpose: BenchmarkPurpose | str,
    state: BenchmarkState | str = BenchmarkState.AUDIT_LOCKED,
    unlock_record: Mapping[str, object] | None = None,
    callback: Callable[[], T] | None = None,
) -> T | None:
    """Validate one SMB operation before invoking its optional callback.

    The source identity comes from the already parsed descriptor. Omitted state is locked. A
    caller may name ``EVALUATION_UNLOCKED`` only with an exact, complete, reviewed record for all
    frozen controls. Adaptive purposes remain forbidden because SMB is evaluation-only.
    """

    _validate_source_descriptor(source_descriptor)
    parsed_purpose = _parse_purpose(purpose)
    parsed_state = _parse_state(state)

    if parsed_purpose in NEVER_ALLOWED_PURPOSES:
        raise _policy_error(
            f"purpose={parsed_purpose.value} is forbidden for the evaluation benchmark"
        )

    if parsed_state is BenchmarkState.EVALUATION_UNLOCKED:
        _validate_unlock_record(unlock_record)

    if (
        parsed_purpose in FINAL_EVALUATION_PURPOSES
        and parsed_state is not BenchmarkState.EVALUATION_UNLOCKED
    ):
        raise _policy_error(
            f"purpose={parsed_purpose.value} requires a complete reviewed evaluation unlock"
        )

    if parsed_purpose not in AUDIT_PURPOSES | FINAL_EVALUATION_PURPOSES:
        raise _policy_error(f"purpose={parsed_purpose.value} is not allowed")

    if callback is None:
        return None
    if not callable(callback):
        raise _policy_error("callback must be callable")
    return callback()
