from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from score_super_resolution.identities import (
    canonical_sha256,
    execution_identity,
    experiment_identity,
)

from score_super_resolution.contracts import ContractValidationError, validate_instance

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contracts"


def _fixture(name: str) -> dict[str, Any]:
    cases = json.loads((FIXTURE_ROOT / "valid-records.json").read_text(encoding="utf-8"))
    return copy.deepcopy(cases[name]["instance"])


def _scientific_controls() -> dict[str, Any]:
    return _fixture("experiment_config_learned")


def test_canonical_sha256_is_stable_for_equivalent_key_orders() -> None:
    left = {"outer": {"beta": 2, "alpha": 1}, "items": ["partitura", "música"]}
    right = {"items": ["partitura", "música"], "outer": {"alpha": 1, "beta": 2}}

    assert canonical_sha256(left) == canonical_sha256(right)
    assert len(canonical_sha256(left)) == 64


def test_experiment_identity_hashes_only_validated_scientific_controls() -> None:
    controls = _scientific_controls()
    reordered = dict(reversed(controls.items()))
    reordered["method"] = dict(reversed(controls["method"].items()))

    first = experiment_identity(controls)
    second = experiment_identity(reordered)

    assert first == second
    assert first["algorithm"] == "sha256"
    assert first["canonicalization_version"] == "json-utf8-sorted-keys-v1"
    assert first["experiment_id"] == f"experiment-{first['sha256']}"

    changed = copy.deepcopy(controls)
    changed["degradation_config_id"] = "controlled-x3-v1"
    assert experiment_identity(changed)["experiment_id"] != first["experiment_id"]


def test_experiment_identity_rejects_controls_before_hashing() -> None:
    controls = _scientific_controls()
    controls["runtime_timestamp"] = "2026-08-18T12:00:00Z"

    with pytest.raises(ContractValidationError, match="additional properties"):
        experiment_identity(controls)


def test_runtime_attempt_details_change_only_execution_identity() -> None:
    experiment = experiment_identity(_scientific_controls())
    base = {
        "experiment_id": experiment["experiment_id"],
        "started_at": "2026-08-18T12:00:00Z",
        "retry_nonce": "attempt-1",
        "environment": {"runtime": "local-cpu", "python_version": "3.12.12"},
        "hardware": {"device_type": "cpu", "device_name": "fixture-cpu"},
    }

    first = execution_identity(**base)
    later = execution_identity(**(base | {"started_at": "2026-08-18T12:01:00Z"}))
    retry = execution_identity(**(base | {"retry_nonce": "attempt-2"}))
    gpu = execution_identity(
        **(base | {"hardware": {"device_type": "gpu", "device_name": "fixture-gpu"}})
    )

    assert {
        first["execution_id"],
        later["execution_id"],
        retry["execution_id"],
        gpu["execution_id"],
    } == {f"execution-{identity['sha256']}" for identity in (first, later, retry, gpu)}
    assert (
        len(
            {
                first["execution_id"],
                later["execution_id"],
                retry["execution_id"],
                gpu["execution_id"],
            }
        )
        == 4
    )
    assert all(
        identity["experiment_id"] == experiment["experiment_id"]
        for identity in (first, later, retry, gpu)
    )
    assert experiment_identity(_scientific_controls()) == experiment


@pytest.mark.parametrize("secret_key", ("HF_TOKEN", "api_key", "authorization", "password"))
def test_execution_identity_rejects_secret_bearing_metadata(secret_key: str) -> None:
    experiment = experiment_identity(_scientific_controls())

    with pytest.raises(ValueError, match="secret-like key"):
        execution_identity(
            experiment_id=experiment["experiment_id"],
            started_at="2026-08-18T12:00:00Z",
            retry_nonce="attempt-1",
            environment={secret_key: "must-not-enter-an-identity"},
            hardware={"device_type": "cpu", "device_name": "fixture-cpu"},
        )


def test_identities_link_a_schema_valid_learned_run_record_with_three_repositories() -> None:
    controls = _scientific_controls()
    experiment = experiment_identity(controls)
    execution = execution_identity(
        experiment_id=experiment["experiment_id"],
        started_at="2026-08-18T12:00:00Z",
        retry_nonce="attempt-1",
        environment={"runtime": "local-cpu", "python_version": "3.12.12"},
        hardware={"device_type": "cpu", "device_name": "fixture-cpu"},
    )
    run_record = {
        "schema_version": 1,
        "record_type": "run-record",
        "run_record_id": "run-identity-fixture-0001",
        "experiment_id": experiment["experiment_id"],
        "execution_id": execution["execution_id"],
        "status": "succeeded",
        "started_at": "2026-08-18T12:00:00Z",
        "completed_at": "2026-08-18T12:00:04Z",
        "repositories": [
            {
                "role": "workspace-planning",
                "root": "/workspace",
                "revision_state": "unborn",
                "revision": None,
                "dirty": True,
            },
            {
                "role": "proyecto",
                "root": "/workspace/proyecto",
                "revision_state": "committed",
                "revision": "a" * 40,
                "dirty": True,
            },
            {
                "role": "memoria",
                "root": "/workspace/memoria",
                "revision_state": "committed",
                "revision": "b" * 40,
                "dirty": False,
            },
        ],
        "manifest_ids": controls["manifest_ids"],
        "experiment_config_id": controls["config_id"],
        "method": {
            "method_id": controls["method"]["method_id"],
            "family": controls["method"]["family"],
        },
        "seeds": controls["seeds"],
        "environment": {
            "python_version": "3.12.12",
            "dependency_snapshot_sha256": "c" * 64,
            "runtime": "local-cpu",
        },
        "hardware": {
            "device_type": "cpu",
            "device_name": "fixture-cpu",
            "memory_bytes": 8589934592,
        },
        "paths": {
            "config": "configs/experiments/identity-fixture.json",
            "logs": "artifacts/identity-fixture/logs",
            "outputs": "artifacts/identity-fixture/outputs",
            "metrics": "artifacts/identity-fixture/metrics.jsonl",
            "failures": "artifacts/identity-fixture/failures.jsonl",
        },
        "model_provenance": {
            "status": "descriptor_reference",
            "model_descriptor_id": controls["method"]["model_provenance"]["model_descriptor_id"],
            "checkpoint": {
                "status": "available",
                "identifier": "example-fidelity-model-x4",
                "sha256": "d" * 64,
            },
        },
    }

    validate_instance("run-record", run_record)

    assert [repository["role"] for repository in run_record["repositories"]] == [
        "workspace-planning",
        "proyecto",
        "memoria",
    ]
    assert run_record["model_provenance"]["model_descriptor_id"] == "example-fidelity-model-v1"


def test_run_record_rejects_collapsed_or_borrowed_repository_identities() -> None:
    record = _fixture("run_record_non_learned")
    committed = {
        "role": "workspace-planning",
        "root": "/workspace",
        "revision_state": "committed",
        "revision": "a" * 40,
        "dirty": False,
    }
    record["repositories"] = [
        committed,
        committed | {"root": "/workspace/proyecto"},
        committed | {"root": "/workspace/memoria"},
    ]

    with pytest.raises(ContractValidationError):
        validate_instance("run-record", record)
