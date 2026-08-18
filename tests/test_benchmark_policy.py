from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from score_super_resolution.benchmark_policy import (
    AUDIT_PURPOSES,
    FINAL_EVALUATION_PURPOSES,
    NEVER_ALLOWED_PURPOSES,
    BenchmarkPolicyError,
    BenchmarkPurpose,
    BenchmarkState,
    assert_smb_purpose_allowed,
)


def _smb_descriptor() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "data" / "sources" / "smb.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _reviewed_unlock() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "smb-evaluation-unlock",
        "review_status": "reviewed",
        "reviewer": "accountable-reviewer",
        "reviewed_at": "2026-08-18",
        "methods": ["methods-v1@sha256:methods"],
        "checkpoints": ["checkpoints-v1@sha256:checkpoints"],
        "degradations": ["degradations-v1@sha256:degradations"],
        "metrics": ["metrics-v1@sha256:metrics"],
        "exclusions": ["exclusions-v1@sha256:exclusions"],
        "sample": ["qualitative-sample-v1@sha256:sample"],
        "interpretation": ["interpretation-v1@sha256:interpretation"],
    }


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
    purpose: BenchmarkPurpose,
) -> None:
    assert (
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=purpose,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=_reviewed_unlock(),
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
        ("methods", []),
        ("checkpoints", None),
        ("degradations", [""]),
        ("metrics", "metrics-v1"),
        ("exclusions", ["duplicate", "duplicate"]),
        ("sample", []),
        ("interpretation", [1]),
    ),
)
def test_incomplete_or_malformed_unlock_record_fails_before_callback(
    field: str, invalid_value: object
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
            callback=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("field", tuple(_reviewed_unlock()))
def test_unlock_record_rejects_every_missing_field(field: str) -> None:
    record = _reviewed_unlock()
    del record[field]

    with pytest.raises(BenchmarkPolicyError, match="unlock_record"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.SR_OUTPUT,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
        )


def test_unlock_record_rejects_unknown_fields() -> None:
    record = _reviewed_unlock()
    record["unknown"] = "must not be ignored"

    with pytest.raises(BenchmarkPolicyError, match="unlock_record"):
        assert_smb_purpose_allowed(
            source_descriptor=_smb_descriptor(),
            purpose=BenchmarkPurpose.DEGRADATION,
            state=BenchmarkState.EVALUATION_UNLOCKED,
            unlock_record=record,
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
