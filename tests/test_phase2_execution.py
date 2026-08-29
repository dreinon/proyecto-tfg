from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.execution import (
    ExecutionBusyError,
    ExecutionInterruptedError,
    ReconciliationError,
    artifact_writer_lock,
    audit_no_smb,
    execute_run,
    export_reconciled_run,
    initialize_run,
    load_experiment_config,
    reconcile_run,
    replay_run,
    resume_run,
    snapshot_run,
)
from score_super_resolution.identities import experiment_identity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/phase2-fixture-v1.yaml"


def test_experiment_v2_is_closed_and_identity_is_mutation_sensitive() -> None:
    loaded = load_experiment_config(CONFIG_PATH)
    controls = loaded.payload
    assert loaded.method_ids == (
        "nearest-opencv-exact-v1",
        "bilinear-opencv-exact-v1",
        "bicubic-opencv-v1",
    )
    assert loaded.condition_ids == (
        "x2-clean",
        "x2-moderate",
        "x2-strong",
        "x4-clean",
        "x4-moderate",
        "x4-strong",
    )
    assert experiment_identity(controls, schema_version=2) == experiment_identity(
        dict(reversed(list(controls.items()))), schema_version=2
    )

    scientific_paths = (
        ("seeds", "degradation"),
        ("limits", "max_attempts_per_tuple"),
        ("fixture", "manifest_sha256"),
        ("qualitative_core", "core_sha256"),
        ("controls", "degradation_sha256"),
    )
    original = experiment_identity(controls, schema_version=2)["experiment_id"]
    for section, key in scientific_paths:
        changed = copy.deepcopy(controls)
        value = changed[section][key]
        changed[section][key] = value + 1 if isinstance(value, int) else "0" * 64
        assert experiment_identity(changed, schema_version=2)["experiment_id"] != original

    changed = copy.deepcopy(controls)
    changed["methods"] = list(reversed(changed["methods"]))
    with pytest.raises(ContractValidationError):
        experiment_identity(changed, schema_version=2)


def test_experiment_v2_preserves_v1_contract_validation() -> None:
    fixture = json.loads((PROJECT_ROOT / "tests/fixtures/contracts/valid-records.json").read_text())
    for case in fixture.values():
        validate_instance(case["schema_id"], case["instance"])


def test_ledger_contract_expands_complete_unique_expected_tuple_set(tmp_path: Path) -> None:
    root = tmp_path / "run"
    identity = initialize_run(CONFIG_PATH, root)
    snapshot = snapshot_run(root)
    assert identity.expected_tuple_count == 8 * 6 * 3
    assert snapshot.experiment_id == identity.experiment_id
    assert snapshot.total == 144
    assert snapshot.counts == {"expected": 144}
    assert snapshot.attempt_count == 0
    assert snapshot.integrity_incident_count == 0

    # Reinitialization is a validated no-op, never a second denominator.
    assert initialize_run(CONFIG_PATH, root) == identity
    assert snapshot_run(root) == snapshot


def test_tuple_claim_is_transactional_and_reconciliation_blocks_nonterminal(tmp_path: Path) -> None:
    root = tmp_path / "run"
    initialize_run(CONFIG_PATH, root)
    with pytest.raises(ReconciliationError, match="nonterminal"):
        reconcile_run(CONFIG_PATH, root)
    snapshot = snapshot_run(root)
    assert snapshot.counts == {"expected": 144}
    assert snapshot.attempt_count == 0


def test_writer_lock_is_nonblocking_and_status_snapshot_is_pure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    initialize_run(CONFIG_PATH, root)
    before = snapshot_run(root)
    with artifact_writer_lock(root):
        with pytest.raises(ExecutionBusyError):
            initialize_run(CONFIG_PATH, root)
        assert snapshot_run(root) == before
    assert snapshot_run(root) == before


@pytest.mark.parametrize(
    "schema_id,payload",
    [
        (
            "scientific-result",
            {
                "schema_version": 2,
                "record_type": "scientific-result",
                "scientific_result_id": "scientific-" + "a" * 64,
                "experiment_id": "experiment-" + "b" * 64,
                "tuple_id": "tuple-" + "c" * 64,
                "item_id": "fixture-work-01-page-01",
                "source_group_id": "fixture-work-01",
                "condition_id": "x2-clean",
                "method_id": "bicubic-opencv-v1",
                "output_relative_path": "/absolute/output.png",
                "output_encoded_sha256": "d" * 64,
                "output_pixel_sha256": "e" * 64,
                "degradation_trace": {},
                "baseline_evidence": {},
                "metrics": [],
                "resource": {},
                "scientific_sha256": "f" * 64,
            },
        ),
        (
            "reconciliation-report",
            {
                "schema_version": 2,
                "record_type": "reconciliation-report",
                "reconciliation_id": "reconciliation-" + "a" * 64,
            },
        ),
    ],
)
def test_scientific_and_reconciliation_schemas_fail_closed(
    schema_id: str, payload: dict[str, object]
) -> None:
    with pytest.raises(ContractValidationError):
        validate_instance(schema_id, payload, version=2)


def test_authored_experiment_yaml_is_schema_valid() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text())
    validate_instance("experiment-config", payload, version=2)


def test_resume_unit_skips_committed_valid_success_after_interruption(tmp_path: Path) -> None:
    root = tmp_path / "run"

    def interrupt(boundary: str, _tuple_id: str) -> None:
        if boundary == "after_tuple_commit":
            raise ExecutionInterruptedError("injected interruption")

    with pytest.raises(ExecutionInterruptedError):
        execute_run(CONFIG_PATH, root, boundary_hook=interrupt)
    partial = snapshot_run(root)
    assert partial.counts == {"expected": 143, "succeeded": 1}
    assert partial.attempt_count == 1

    report = resume_run(CONFIG_PATH, root, max_tuples=2)
    assert report.succeeded == 2
    resumed = snapshot_run(root)
    assert resumed.counts == {"expected": 141, "succeeded": 3}
    assert resumed.attempt_count == 3


def test_resume_unit_recovers_failure_before_fsync_with_attempt_history(tmp_path: Path) -> None:
    root = tmp_path / "run"

    def fail(boundary: str, _tuple_id: str) -> None:
        if boundary == "before_output_fsync":
            raise OSError("injected publication failure")

    report = execute_run(CONFIG_PATH, root, max_tuples=1, boundary_hook=fail)
    assert report.failed == 1
    partial = snapshot_run(root)
    assert partial.counts == {"expected": 143, "retry_pending": 1}
    assert partial.attempt_count == 1

    resumed = resume_run(CONFIG_PATH, root, max_tuples=1)
    assert resumed.succeeded == 1
    final = snapshot_run(root)
    assert final.counts == {"expected": 143, "succeeded": 1}
    assert final.attempt_count == 2


def test_integrity_lifecycle_quarantines_and_repairs_corrupted_success(tmp_path: Path) -> None:
    root = tmp_path / "run"
    execute_run(CONFIG_PATH, root, max_tuples=1)
    output = next((root / "outputs").rglob("*.png"))
    original = output.read_bytes()
    output.write_bytes(b"corrupted committed output")

    report = resume_run(CONFIG_PATH, root, max_tuples=1)
    assert report.succeeded == 1
    assert output.read_bytes() == original
    snapshot = snapshot_run(root)
    assert snapshot.counts == {"expected": 143, "succeeded": 1}
    assert snapshot.attempt_count == 2
    assert snapshot.integrity_incident_count == 1
    quarantine = list((root / "quarantine").rglob("*"))
    assert any(
        path.is_file() and path.read_bytes() == b"corrupted committed output" for path in quarantine
    )


def test_corruption_transition_never_counts_missing_committed_output(tmp_path: Path) -> None:
    root = tmp_path / "run"
    execute_run(CONFIG_PATH, root, max_tuples=1)
    output = next((root / "outputs").rglob("*.png"))
    output.unlink()

    # A zero-work resume still performs integrity repair before returning.
    resume_run(CONFIG_PATH, root, max_tuples=0)
    snapshot = snapshot_run(root)
    assert snapshot.counts == {"expected": 143, "retry_pending": 1}
    assert snapshot.integrity_incident_count == 1


def test_report_schema_contracts_fail_closed() -> None:
    for schema_id, record_type in (
        ("replay-report", "replay-report"),
        ("portable-export", "portable-export"),
    ):
        with pytest.raises(ContractValidationError):
            validate_instance(
                schema_id,
                {"schema_version": 2, "record_type": record_type},
                version=2,
            )


def test_cli_contract_emits_stable_json_and_rejects_absolute_root(tmp_path: Path) -> None:
    valid = subprocess.run(
        [
            sys.executable,
            "-m",
            "score_super_resolution.cli",
            "validate",
            "--config",
            "configs/experiments/phase2-fixture-v1.yaml",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    payload = json.loads(valid.stdout)
    assert payload["status"] == "valid"
    assert payload["experiment_id"].startswith("experiment-")

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "score_super_resolution.cli",
            "status",
            "--artifact-root",
            str(tmp_path / "absolute"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert json.loads(rejected.stdout) == {
        "code": "invalid_request",
        "status": "error",
    }


def test_writer_contention_blocks_reconcile_and_export_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "run"
    initialize_run(CONFIG_PATH, root)
    before = snapshot_run(root)
    with artifact_writer_lock(root):
        with pytest.raises(ExecutionBusyError):
            reconcile_run(CONFIG_PATH, root)
        with pytest.raises(ExecutionBusyError):
            export_reconciled_run(CONFIG_PATH, root)
    assert snapshot_run(root) == before


@pytest.fixture(scope="module")
def acceptance_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("phase2-acceptance")
    interrupted = parent / "interrupted"
    clean = parent / "clean"

    def stop_after_commit(boundary: str, _tuple_id: str) -> None:
        if boundary == "after_tuple_commit":
            raise ExecutionInterruptedError("acceptance interruption")

    with pytest.raises(ExecutionInterruptedError):
        execute_run(CONFIG_PATH, interrupted, boundary_hook=stop_after_commit)
    assert snapshot_run(interrupted).counts == {"expected": 143, "succeeded": 1}
    resume_run(CONFIG_PATH, interrupted)
    reconcile_run(CONFIG_PATH, interrupted)
    export_reconciled_run(CONFIG_PATH, interrupted)

    execute_run(CONFIG_PATH, clean)
    reconcile_run(CONFIG_PATH, clean)
    replay_run(CONFIG_PATH, interrupted, clean)
    return interrupted, clean


def test_interrupted_acceptance_run_publishes_closed_export(
    acceptance_runs: tuple[Path, Path],
) -> None:
    interrupted, _clean = acceptance_runs
    assert snapshot_run(interrupted).counts == {"succeeded": 144}
    for relative in (
        "reconciliation-report.json",
        "evidence/aggregate-six-cell.json",
        "evidence/qualitative-membership.json",
        "export/portable-export-manifest.json",
    ):
        assert (interrupted / relative).is_file()
    assert audit_no_smb(interrupted)["status"] == "clean"


def test_second_root_replay_matches_scientific_projection(
    acceptance_runs: tuple[Path, Path],
) -> None:
    interrupted, clean = acceptance_runs
    report = json.loads((interrupted / "replay-report.json").read_text())
    validate_instance("replay-report", report, version=2)
    assert report["status"] == "equivalent"
    assert report["expected_tuple_count"] == 144
    assert snapshot_run(clean).counts == {"succeeded": 144}


@pytest.mark.parametrize(
    "boundary",
    [
        "before_output_fsync",
        "after_output_replace_before_ledger",
        "after_tuple_commit",
    ],
)
def test_failpoint_matrix_resumes_without_losing_attempt_evidence(
    tmp_path: Path, boundary: str
) -> None:
    root = tmp_path / boundary
    fired = False

    def fail(observed: str, _tuple_id: str) -> None:
        nonlocal fired
        if not fired and observed == boundary:
            fired = True
            if boundary == "after_tuple_commit":
                raise ExecutionInterruptedError("interrupted after commit")
            raise OSError("injected publication fault")

    try:
        execute_run(CONFIG_PATH, root, max_tuples=1, boundary_hook=fail)
    except ExecutionInterruptedError:
        pass
    before = snapshot_run(root)
    resume_run(CONFIG_PATH, root, max_tuples=1)
    after = snapshot_run(root)
    assert after.counts.get("succeeded") == 1
    assert after.attempt_count >= before.attempt_count
