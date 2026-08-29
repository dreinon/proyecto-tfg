from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from score_super_resolution.identities import canonical_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks/02-fixture-baseline-review.ipynb"
FIXTURE_ROOT = PROJECT_ROOT / "artifacts/phase2-fixture"


def _copy_fixed_review_inputs(destination: Path) -> Path:
    destination.mkdir(parents=True)
    for relative in (
        "pre-run/qualitative-core-membership.json",
        "reconciliation-report.json",
        "replay-report.json",
        "evidence/aggregate-six-cell.json",
        "evidence/qualitative-membership.json",
        "export/portable-export-manifest.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURE_ROOT / relative, target)
    for directory in ("fixture-input", "outputs", "scientific"):
        shutil.copytree(FIXTURE_ROOT / directory, destination / directory)
    return destination


def test_notebook_source_clean_is_output_free_and_digest_stable() -> None:
    from score_super_resolution.review import notebook_source_sha256

    first = notebook_source_sha256(SOURCE_NOTEBOOK)
    second = notebook_source_sha256(SOURCE_NOTEBOOK)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    assert first == second == canonical_sha256(notebook)
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    serialized = json.dumps(notebook).casefold()
    assert "image/png" not in serialized
    assert "image/jpeg" not in serialized
    assert "application/pdf" not in serialized


def test_notebook_source_clean_rejects_execution_and_embedded_payload(tmp_path: Path) -> None:
    from score_super_resolution.review import FixtureReviewContractError, notebook_source_sha256

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code["execution_count"] = 1
    code["outputs"] = [{"output_type": "display_data", "data": {"image/png": "AAAA"}}]
    dirty = tmp_path / "dirty.ipynb"
    dirty.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(FixtureReviewContractError, match="tracked notebook"):
        notebook_source_sha256(dirty)


def test_consume_acceptance_reports_validates_fixed_replay_without_regeneration() -> None:
    from score_super_resolution.review import validate_fixture_review_inputs

    before = {
        relative: hashlib.sha256((FIXTURE_ROOT / relative).read_bytes()).hexdigest()
        for relative in (
            "reconciliation-report.json",
            "replay-report.json",
            "evidence/aggregate-six-cell.json",
            "evidence/qualitative-membership.json",
        )
    }
    inputs = validate_fixture_review_inputs(PROJECT_ROOT)

    assert inputs["experiment_id"].startswith("experiment-")
    assert inputs["reconciliation_id"].startswith("reconciliation-")
    assert inputs["replay_id"].startswith("replay-")
    assert inputs["condition_ids"] == [
        "x2-clean",
        "x2-moderate",
        "x2-strong",
        "x4-clean",
        "x4-moderate",
        "x4-strong",
    ]
    assert inputs["core_panel_count"] == 12
    assert inputs["additional_panel_count"] <= 12
    assert inputs["requested_panel_count"] <= 24
    assert before == {
        relative: hashlib.sha256((FIXTURE_ROOT / relative).read_bytes()).hexdigest()
        for relative in before
    }


def test_membership_and_report_substitution_fail_closed(tmp_path: Path) -> None:
    from score_super_resolution.review import FixtureReviewContractError, validate_fixture_review_inputs

    artifact_root = _copy_fixed_review_inputs(tmp_path / "phase2-fixture")
    replay_path = artifact_root / "replay-report.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["status"] = "different"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(FixtureReviewContractError):
        validate_fixture_review_inputs(PROJECT_ROOT, artifact_root=artifact_root)

    shutil.copy2(FIXTURE_ROOT / "replay-report.json", replay_path)
    membership_path = artifact_root / "evidence/qualitative-membership.json"
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    membership["core_panels"] = list(reversed(membership["core_panels"]))
    membership_path.write_text(json.dumps(membership), encoding="utf-8")
    with pytest.raises(FixtureReviewContractError):
        validate_fixture_review_inputs(PROJECT_ROOT, artifact_root=artifact_root)


def test_masked_review_contract_is_deterministic_and_content_addressed() -> None:
    from score_super_resolution.review import prepare_fixture_review

    first = prepare_fixture_review(PROJECT_ROOT)
    first_mapping = (FIXTURE_ROOT / "review/method-mapping.json").read_bytes()
    first_panels = {
        panel["panel_id"]: hashlib.sha256((FIXTURE_ROOT / panel["relative_path"]).read_bytes()).hexdigest()
        for panel in first["panels"]
    }
    second = prepare_fixture_review(PROJECT_ROOT)

    assert first == second
    assert (FIXTURE_ROOT / "review/method-mapping.json").read_bytes() == first_mapping
    assert len(first["panels"]) == 24
    assert len(first["mapping"]) == 24
    assert {row["panel_id"] for row in first["mapping"]} == set(first_panels)
    for row in first["mapping"]:
        assert set(row["masked_methods"]) == {"A", "B", "C"}
        assert set(row["masked_methods"].values()) == {
            "nearest-opencv-exact-v1",
            "bilinear-opencv-exact-v1",
            "bicubic-opencv-v1",
        }
    assert first_panels == {
        panel["panel_id"]: hashlib.sha256((FIXTURE_ROOT / panel["relative_path"]).read_bytes()).hexdigest()
        for panel in second["panels"]
    }


def test_working_copy_execute_binds_source_and_creates_no_review() -> None:
    from score_super_resolution.review import execute_fixture_review_notebook, notebook_source_sha256

    source_before = SOURCE_NOTEBOOK.read_bytes()
    manifest = execute_fixture_review_notebook(PROJECT_ROOT)
    working = FIXTURE_ROOT / manifest["working_notebook_relative_path"]

    assert working.is_file()
    assert manifest["notebook_source_sha256"] == notebook_source_sha256(SOURCE_NOTEBOOK)
    assert manifest["working_notebook_sha256"] == hashlib.sha256(working.read_bytes()).hexdigest()
    assert manifest["requested_panel_count"] == 24
    assert manifest["displayable_panel_count"] == 24
    assert manifest["failed_panel_count"] == 0
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
    assert SOURCE_NOTEBOOK.read_bytes() == source_before


def test_thin_notebook_contains_only_review_session_calls() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    required = {
        "FixtureReviewSession",
        "review.summary()",
        "review.panel_widget()",
        "review.review_widget()",
        "review.progress_widget()",
    }
    assert all(token in source for token in required)
    forbidden = {
        "cv2.",
        "numpy",
        "np.",
        "PSNR",
        "SSIM",
        "GaussianBlur",
        "resize(",
        "canonical_sha256",
        "json.dump",
        "open(",
    }
    assert not any(token in source for token in forbidden)


def test_review_contract_does_not_prefill_human_evidence() -> None:
    from score_super_resolution.review import FixtureReviewSession, FixtureReviewContractError

    manifest = execute_fixture_review_notebook_for_contract()
    session = FixtureReviewSession(
        PROJECT_ROOT,
        working_copy_token=manifest["working_copy_token"],
    )
    assert session.summary()["reviewed_panels"] == 0
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
    with pytest.raises(FixtureReviewContractError):
        FixtureReviewSession(
            PROJECT_ROOT,
            working_copy_token="__GENERATED_PHASE2_FIXTURE_REVIEW_WORKING_COPY__",
        )


def execute_fixture_review_notebook_for_contract() -> dict[str, object]:
    from score_super_resolution.review import execute_fixture_review_notebook

    return execute_fixture_review_notebook(PROJECT_ROOT)
