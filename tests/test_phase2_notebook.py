from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from score_super_resolution.identities import canonical_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks/02-fixture-baseline-review.ipynb"
SEMANTIC_SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks/02-semantic-fixture-baseline-review.ipynb"
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
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(
        cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
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
    from score_super_resolution.review import (
        FixtureReviewContractError,
        validate_fixture_review_inputs,
    )

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
        panel["panel_id"]: hashlib.sha256(
            (FIXTURE_ROOT / panel["relative_path"]).read_bytes()
        ).hexdigest()
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
        panel["panel_id"]: hashlib.sha256(
            (FIXTURE_ROOT / panel["relative_path"]).read_bytes()
        ).hexdigest()
        for panel in second["panels"]
    }


def test_working_copy_execute_is_superseded_without_mutation() -> None:
    from score_super_resolution.review import (
        FixtureReviewContractError,
        execute_fixture_review_notebook,
    )

    source_before = SOURCE_NOTEBOOK.read_bytes()
    working = FIXTURE_ROOT / "review/fixture-baseline-review-working.ipynb"
    working_before = working.read_bytes()
    with pytest.raises(FixtureReviewContractError, match=r"D-23.*superseded"):
        execute_fixture_review_notebook(PROJECT_ROOT)
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
    assert SOURCE_NOTEBOOK.read_bytes() == source_before
    assert working.read_bytes() == working_before


def test_thin_notebook_contains_only_review_session_calls() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
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
    from score_super_resolution.review import (
        FixtureReviewContractError,
        FixtureReviewSession,
        prepare_fixture_review,
    )

    prepared = prepare_fixture_review(PROJECT_ROOT)
    with pytest.raises(FixtureReviewContractError, match=r"D-23.*superseded"):
        FixtureReviewSession(
            PROJECT_ROOT,
            working_copy_token=prepared["working_copy_token"],
        )
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
    with pytest.raises(FixtureReviewContractError):
        FixtureReviewSession(
            PROJECT_ROOT,
            working_copy_token="__GENERATED_PHASE2_FIXTURE_REVIEW_WORKING_COPY__",
        )


def test_primitive_semantic_forbidden_but_technical_panels_remain() -> None:
    from score_super_resolution.review import (
        FixtureReviewContractError,
        FixtureReviewSession,
        prepare_fixture_review,
    )

    prepared = prepare_fixture_review(PROJECT_ROOT)
    assert len(prepared["panels"]) == 24
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
    with pytest.raises(FixtureReviewContractError, match=r"D-23.*superseded"):
        FixtureReviewSession(
            PROJECT_ROOT,
            working_copy_token=prepared["working_copy_token"],
        )
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()


def test_semantic_musicxml_applicability_accepts_exact_selected_sources() -> None:
    from score_super_resolution.review import (
        load_semantic_fixture_control,
        validate_semantic_musicxml_source,
    )

    control = load_semantic_fixture_control(PROJECT_ROOT)
    reports = [
        validate_semantic_musicxml_source(
            PROJECT_ROOT / source["source_path"],
            source=source,
            renderer=control["renderer"],
            limits=control["limits"],
        )
        for source in control["sources"]
    ]

    assert [report["source_id"] for report in reports] == [
        "review-work-03-excerpt-01",
        "review-work-04-excerpt-01",
    ]
    assert all(report["coherence_state"] == "structurally-coherent" for report in reports)
    assert all(report["measure_count"] == 2 for report in reports)


@pytest.mark.parametrize("mutation", ["duration", "slur"])
def test_source_coherence_rejects_tag_complete_temporal_or_relation_mutation(
    tmp_path: Path, mutation: str
) -> None:
    from score_super_resolution.review import (
        FixtureReviewContractError,
        load_semantic_fixture_control,
        validate_semantic_musicxml_source,
    )

    control = load_semantic_fixture_control(PROJECT_ROOT)
    source = deepcopy(control["sources"][0])
    original = (PROJECT_ROOT / source["source_path"]).read_text(encoding="utf-8")
    if mutation == "duration":
        changed = original.replace("<duration>3</duration>", "<duration>4</duration>", 1)
    else:
        changed = original.replace('<slur type="stop" number="1"/>', "", 1)
    path = tmp_path / "tag-complete-but-incoherent.musicxml"
    path.write_text(changed, encoding="utf-8")
    source["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(FixtureReviewContractError, match=r"coherence|duration|relation|slur"):
        validate_semantic_musicxml_source(
            path,
            source=source,
            renderer=control["renderer"],
            limits=control["limits"],
        )


def test_semantic_matrix_contract_is_closed_and_identity_sensitive(tmp_path: Path) -> None:
    from score_super_resolution.review import (
        FixtureReviewContractError,
        load_semantic_fixture_control,
    )

    control = load_semantic_fixture_control(PROJECT_ROOT)
    assert control["source_order"] == [
        "review-work-03-excerpt-01",
        "review-work-04-excerpt-01",
    ]
    assert control["condition_order"] == [
        "x2-clean",
        "x2-moderate",
        "x2-strong",
        "x4-clean",
        "x4-moderate",
        "x4-strong",
    ]
    assert control["method_order"] == [
        "nearest-opencv-exact-v1",
        "bilinear-opencv-exact-v1",
        "bicubic-opencv-v1",
    ]
    assert len(control["expected_tuple_keys"]) == 36
    assert len(set(control["expected_tuple_keys"])) == 36
    assert len(control["review_membership"]) == 12

    source_path = PROJECT_ROOT / "configs/experiments/phase2-semantic-fixture-v1.yaml"
    mutated = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    mutated["sources"][0]["roi"]["x"] += 1
    path = tmp_path / "mutated-semantic-control.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    assert canonical_sha256(mutated) != control["semantic_experiment_sha256"]
    with pytest.raises(FixtureReviewContractError, match="source provenance, digest, or ROI"):
        load_semantic_fixture_control(PROJECT_ROOT, path=path)


def _primitive_inventory() -> dict[str, str]:
    return {
        path.relative_to(FIXTURE_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FIXTURE_ROOT.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def test_semantic_matrix_execution_publishes_manifest_before_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import score_super_resolution.review as review

    artifact_root = tmp_path / "phase2-semantic-fixture"
    prepared = review.prepare_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    manifest_path = artifact_root / "semantic-experiment-manifest.json"
    assert manifest_path.exists()
    assert prepared["expected_tuple_count"] == 36
    assert not (artifact_root / "records").exists()

    degradation_calls: list[tuple[str, str]] = []
    baseline_calls: list[tuple[str, str]] = []
    original_degradation = review.apply_degradation
    original_baseline = review.run_baseline

    def checked_degradation(*args: object, **kwargs: object) -> object:
        assert manifest_path.exists()
        degradation_calls.append((str(kwargs["item_id"]), str(kwargs["condition_id"])))
        return original_degradation(*args, **kwargs)

    def checked_baseline(*args: object, **kwargs: object) -> object:
        assert manifest_path.exists()
        baseline_calls.append((str(args[0]), str(kwargs["condition_id"])))
        return original_baseline(*args, **kwargs)

    monkeypatch.setattr(review, "apply_degradation", checked_degradation)
    monkeypatch.setattr(review, "run_baseline", checked_baseline)
    executed = review.execute_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)

    assert executed["terminal_tuple_count"] == 36
    assert len(degradation_calls) == 12
    assert len(baseline_calls) == 36
    assert len(set(degradation_calls)) == 12


def test_semantic_manifest_reconciliation_rejects_unexpected_or_partial_tuple(
    tmp_path: Path,
) -> None:
    from score_super_resolution.review import (
        FixtureReviewContractError,
        execute_semantic_fixture_experiment,
        prepare_semantic_fixture_experiment,
        reconcile_semantic_fixture_experiment,
    )

    artifact_root = tmp_path / "phase2-semantic-fixture"
    prepare_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    execute_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    report = reconcile_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    assert report["expected_tuple_count"] == report["terminal_tuple_count"] == 36
    assert report["counts"] == {"succeeded": 36, "failed": 0, "excluded": 0}

    unexpected = artifact_root / "records/x2-clean/bicubic-opencv-v1/unexpected.json"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}", encoding="utf-8")
    with pytest.raises(FixtureReviewContractError, match=r"unexpected|closed|tuple"):
        reconcile_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)


def test_semantic_masked_panels_bind_exact_systematic_membership(tmp_path: Path) -> None:
    from score_super_resolution.review import (
        execute_semantic_fixture_experiment,
        prepare_semantic_fixture_experiment,
        prepare_semantic_review,
        reconcile_semantic_fixture_experiment,
    )

    artifact_root = tmp_path / "phase2-semantic-fixture"
    prepare_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    execute_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    reconcile_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    prepared = prepare_semantic_review(PROJECT_ROOT, artifact_root=artifact_root)

    assert len(prepared["panels"]) == len(prepared["mapping"]) == 12
    assert [row["source_id"] for row in prepared["panels"]] == [
        *(["review-work-03-excerpt-01"] * 6),
        *(["review-work-04-excerpt-01"] * 6),
    ]
    assert [row["condition_id"] for row in prepared["panels"]] == [
        "x2-clean",
        "x2-moderate",
        "x2-strong",
        "x4-clean",
        "x4-moderate",
        "x4-strong",
    ] * 2
    for row in prepared["mapping"]:
        assert set(row["masked_methods"]) == {"A", "B", "C"}
        assert set(row["masked_methods"].values()) == {
            "nearest-opencv-exact-v1",
            "bilinear-opencv-exact-v1",
            "bicubic-opencv-v1",
        }
    assert not (artifact_root / "review/source-confirmations.json").exists()
    assert not (artifact_root / "review/notation-review.json").exists()


def test_semantic_notebook_source_clean_and_unprefilled() -> None:
    from score_super_resolution.review import semantic_notebook_source_sha256

    first = semantic_notebook_source_sha256(SEMANTIC_SOURCE_NOTEBOOK)
    second = semantic_notebook_source_sha256(SEMANTIC_SOURCE_NOTEBOOK)
    notebook = json.loads(SEMANTIC_SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert first == second == canonical_sha256(notebook)
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "SemanticFixtureReviewSession" in source
    assert "review.source_confirmation_widget()" in source
    assert "review.panel_widget()" in source
    assert "review.review_widget()" in source
    assert "review.progress_widget()" in source
    assert "value=True" not in source
    assert "none_observed" not in source


def test_primitive_evidence_immutable_across_separate_semantic_stream(tmp_path: Path) -> None:
    from score_super_resolution.review import (
        execute_semantic_fixture_experiment,
        prepare_semantic_fixture_experiment,
        prepare_semantic_review,
        reconcile_semantic_fixture_experiment,
    )

    before = _primitive_inventory()
    fixed_before = {
        relative: before[relative]
        for relative in (
            "reconciliation-report.json",
            "replay-report.json",
            "evidence/aggregate-six-cell.json",
            "evidence/qualitative-membership.json",
        )
    }
    artifact_root = tmp_path / "phase2-semantic-fixture"
    prepare_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    execute_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    reconcile_semantic_fixture_experiment(PROJECT_ROOT, artifact_root=artifact_root)
    prepare_semantic_review(PROJECT_ROOT, artifact_root=artifact_root)

    assert _primitive_inventory() == before
    assert {relative: _primitive_inventory()[relative] for relative in fixed_before} == fixed_before
    assert not (FIXTURE_ROOT / "review/notation-review.json").exists()
