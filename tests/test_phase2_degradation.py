from __future__ import annotations

import copy
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parents[1]
CONTROL_PATH = PROJECT_ROOT / "configs/degradations/controlled-score-candidates.yaml"
FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "tests/fixtures/phase2/fixture-manifest-v1.yaml"
EXPECTED_CELLS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)


def _degradation() -> Any:
    return importlib.import_module("score_super_resolution.degradation")


def _yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_degradation_contract_defines_exact_closed_six_cell_candidate() -> None:
    module = _degradation()
    control = module.load_degradation_control(CONTROL_PATH)

    assert control.status == "candidate"
    assert control.version == 1
    assert tuple(control.condition_ids) == EXPECTED_CELLS
    assert control.master_seed == 20260821
    assert control.claim_boundary == "controlled-synthetic-only"
    assert control.image_contract == {
        "mode": "RGB",
        "dtype": "uint8",
        "range": [0, 255],
        "channel_degradation": "achromatic-equal",
        "restoration_preprocessing": "none",
        "forbidden_transformations": [
            "grayscale",
            "background-whitening",
            "contrast-normalization",
            "binarization",
        ],
    }

    conditions = {condition["condition_id"]: condition for condition in control.conditions}
    assert conditions["x2-clean"]["operations"] == ["reduction"]
    assert conditions["x4-clean"]["operations"] == ["reduction"]
    for scale in (2, 4):
        moderate = conditions[f"x{scale}-moderate"]
        strong = conditions[f"x{scale}-strong"]
        expected_order = ["blur", "reduction", "noise", "clip-round", "jpeg"]
        assert moderate["operations"] == expected_order
        assert strong["operations"] == expected_order
        assert moderate["blur"] == {"type": "gaussian", "sigma": 0.8, "kernel": 7}
        assert moderate["noise"] == {"type": "gaussian", "sigma": 3.0}
        assert moderate["jpeg"]["quality"] == 85
        assert strong["blur"] == {"type": "gaussian", "sigma": 1.6, "kernel": 11}
        assert strong["noise"] == {"type": "gaussian", "sigma": 8.0}
        assert strong["jpeg"]["quality"] == 60


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("x3", "condition"),
        ("missing-operator", "operations"),
        ("implicit-default", "additional properties"),
        ("grayscale", "mode"),
        ("whitening", "forbidden"),
        ("non-monotonic", "monotonic"),
        ("duplicate-version", "unique"),
    ),
)
def test_degradation_contract_mutations_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    module = _degradation()
    registry = _yaml(CONTROL_PATH)
    entry = registry["candidates"][0]
    if mutation == "x3":
        entry["conditions"][0]["condition_id"] = "x3-clean"
        entry["conditions"][0]["scale"] = 3
    elif mutation == "missing-operator":
        entry["conditions"][1]["operations"].remove("noise")
    elif mutation == "implicit-default":
        entry["conditions"][0]["interpolation_default"] = True
    elif mutation == "grayscale":
        entry["image_contract"]["mode"] = "L"
    elif mutation == "whitening":
        entry["image_contract"]["forbidden_transformations"].remove("background-whitening")
    elif mutation == "non-monotonic":
        later = copy.deepcopy(entry)
        later["version"] = 0
        registry["candidates"].append(later)
    elif mutation == "duplicate-version":
        registry["candidates"].append(copy.deepcopy(entry))
    path = tmp_path / "candidate.yaml"
    _write_yaml(path, registry)

    with pytest.raises(module.DegradationContractError, match=message):
        module.load_degradation_control(path)


def test_fixture_contract_generates_deterministic_independent_rgb8_bundle(tmp_path: Path) -> None:
    module = _degradation()
    first = module.generate_fixture_bundle(FIXTURE_MANIFEST_PATH, tmp_path / "first")
    second = module.generate_fixture_bundle(FIXTURE_MANIFEST_PATH, tmp_path / "second")

    assert first["manifest_id"] == second["manifest_id"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert len(first["items"]) == 8
    assert len({item["source_group_id"] for item in first["items"]}) == 4
    assert all(item["source_role"] == "pipeline-validation-only" for item in first["items"])
    assert [item["pixel_sha256"] for item in first["items"]] == [
        item["pixel_sha256"] for item in second["items"]
    ]
    for item in first["items"]:
        image = cv2.cvtColor(
            cv2.imread(str(tmp_path / "first" / item["relative_path"]), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.shape[:2] == (item["height"], item["width"])
        roi = item["roi"]
        assert roi["x"] + roi["width"] <= item["width"]
        assert roi["y"] + roi["height"] <= item["height"]


def test_fixture_manifest_records_authorship_licence_checksums_groups_pages_and_rois() -> None:
    manifest = _yaml(FIXTURE_MANIFEST_PATH)
    assert manifest["source_role"] == "pipeline-validation-only"
    assert manifest["content_policy"] == "project-authored-synthetic-no-smb"
    assert len(manifest["items"]) == 8
    grouped: dict[str, list[int]] = {}
    for item in manifest["items"]:
        assert item["origin"] == "generated-by-score-super-resolution-project"
        assert item["author"] == "TFG score-super-resolution project"
        assert item["license"] == "CC0-1.0"
        assert len(item["generated_pixel_sha256"]) == 64
        grouped.setdefault(item["source_group_id"], []).append(item["page_number"])
    assert grouped == {
        "fixture-work-01": [1, 2],
        "fixture-work-02": [1, 2],
        "fixture-work-03": [1, 2],
        "fixture-work-04": [1, 2],
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "absolute-path",
        "traversal",
        "oversized-dimensions",
        "excessive-pixels",
        "too-many-items",
        "smb-origin",
        "smb-role",
        "secret-metadata",
        "changed-checksum",
    ),
)
def test_fixture_mutations_fail_before_generation(tmp_path: Path, mutation: str) -> None:
    module = _degradation()
    manifest = _yaml(FIXTURE_MANIFEST_PATH)
    item = manifest["items"][0]
    if mutation == "absolute-path":
        item["relative_path"] = "/tmp/fixture.png"
    elif mutation == "traversal":
        item["relative_path"] = "../fixture.png"
    elif mutation == "oversized-dimensions":
        item["width"] = 100_000
    elif mutation == "excessive-pixels":
        item["width"] = 5000
        item["height"] = 5000
    elif mutation == "too-many-items":
        manifest["items"] = manifest["items"] * 9
    elif mutation == "smb-origin":
        item["origin"] = "PRAIG/SMB"
    elif mutation == "smb-role":
        manifest["source_role"] = "evaluation-benchmark"
    elif mutation == "secret-metadata":
        item["api_token"] = "not-a-real-token"
    elif mutation == "changed-checksum":
        item["generated_pixel_sha256"] = "0" * 64
    path = tmp_path / "fixture-manifest.yaml"
    _write_yaml(path, manifest)

    with pytest.raises(module.FixtureValidationError):
        module.generate_fixture_bundle(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_fixture_symlink_manifest_and_changed_or_malformed_bytes_fail_closed(
    tmp_path: Path,
) -> None:
    module = _degradation()
    symlink = tmp_path / "manifest-link.yaml"
    symlink.symlink_to(FIXTURE_MANIFEST_PATH)
    with pytest.raises(module.FixtureValidationError, match="symlink"):
        module.generate_fixture_bundle(symlink, tmp_path / "linked-output")

    output = tmp_path / "bundle"
    bundle = module.generate_fixture_bundle(FIXTURE_MANIFEST_PATH, output)
    fixture_path = output / bundle["items"][0]["relative_path"]
    fixture_path.write_bytes(b"not an image")
    with pytest.raises(module.FixtureValidationError, match="digest"):
        module.generate_fixture_bundle(FIXTURE_MANIFEST_PATH, output)


def _neutral_reference(height: int = 131, width: int = 137) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    plane = ((rows * 3 + columns * 5) % 256).astype(np.uint8)
    return np.repeat(plane[..., None], 3, axis=2)


def test_dimensions_alignment_and_rgb8_validation_are_explicit() -> None:
    module = _degradation()
    reference = _neutral_reference()
    aligned_x4 = module.align_reference(reference, 4)

    assert aligned_x4.input_dimensions == (131, 137, 3)
    assert aligned_x4.aligned_dimensions == (128, 136, 3)
    assert aligned_x4.crop == {"top": 0, "left": 0, "bottom": 3, "right": 1}
    assert np.array_equal(aligned_x4.pixels, reference[:128, :136])
    assert aligned_x4.pixels.dtype == np.uint8
    with pytest.raises(module.DegradationContractError):
        module.align_reference(reference.astype(np.float32), 2)
    with pytest.raises(module.DegradationContractError):
        module.align_reference(reference[..., 0], 2)
    with pytest.raises(module.DegradationContractError):
        module.align_reference(reference, 3)


@pytest.mark.parametrize("condition_id", EXPECTED_CELLS)
def test_deterministic_lineage_order_colour_and_encoded_bytes(condition_id: str) -> None:
    module = _degradation()
    control = module.load_degradation_control(CONTROL_PATH)
    reference = _neutral_reference()
    arguments = {
        "control": control,
        "condition_id": condition_id,
        "item_id": "fixture-analytical-page",
        "source_group_id": "fixture-analytical-work",
        "fixture_manifest_id": "phase2-score-fixtures-v1",
        "purpose": "fixture-preview",
    }
    first = module.apply_degradation(reference, **arguments)
    second = module.apply_degradation(reference, **arguments)
    scale = int(condition_id[1])

    assert np.array_equal(first.pixels, second.pixels)
    assert first.encoded_bytes == second.encoded_bytes
    assert first.trace == second.trace
    assert first.pixels.shape == (131 // scale, 137 // scale, 3)
    assert first.pixels.dtype == np.uint8
    assert int(first.pixels.min()) >= 0 and int(first.pixels.max()) <= 255
    assert np.array_equal(first.pixels[..., 0], first.pixels[..., 1])
    assert np.array_equal(first.pixels[..., 1], first.pixels[..., 2])
    expected_order = (
        ["reduction"]
        if condition_id.endswith("clean")
        else ["blur", "reduction", "noise", "clip-round", "jpeg"]
    )
    assert [operation["operator_id"] for operation in first.trace["operations"]] == expected_order
    decoded_bgr = cv2.imdecode(np.frombuffer(first.encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB), first.pixels)


def test_deterministic_seed_and_scientific_mutation_change_identity(tmp_path: Path) -> None:
    module = _degradation()
    control = module.load_degradation_control(CONTROL_PATH)
    seed = module.derive_degradation_seed(
        control.master_seed,
        fixture_manifest_id="phase2-score-fixtures-v1",
        item_id="fixture-work-01-page-01",
        condition_id="x2-moderate",
    )
    assert seed == module.derive_degradation_seed(
        control.master_seed,
        fixture_manifest_id="phase2-score-fixtures-v1",
        item_id="fixture-work-01-page-01",
        condition_id="x2-moderate",
    )
    assert seed != module.derive_degradation_seed(
        control.master_seed,
        fixture_manifest_id="phase2-score-fixtures-v1",
        item_id="fixture-work-01-page-02",
        condition_id="x2-moderate",
    )

    registry = _yaml(CONTROL_PATH)
    registry["candidates"][0]["master_seed"] += 1
    changed_path = tmp_path / "changed.yaml"
    _write_yaml(changed_path, registry)
    changed = module.load_degradation_control(changed_path)
    first = module.apply_degradation(
        _neutral_reference(),
        control=control,
        condition_id="x2-moderate",
        item_id="fixture-work-01-page-01",
        source_group_id="fixture-work-01",
        fixture_manifest_id="phase2-score-fixtures-v1",
        purpose="fixture-preview",
    )
    second = module.apply_degradation(
        _neutral_reference(),
        control=changed,
        condition_id="x2-moderate",
        item_id="fixture-work-01-page-01",
        source_group_id="fixture-work-01",
        fixture_manifest_id="phase2-score-fixtures-v1",
        purpose="fixture-preview",
    )
    assert changed.sha256 != control.sha256
    assert first.trace["trace_id"] != second.trace["trace_id"]
    assert not np.array_equal(first.pixels, second.pixels)
    with pytest.raises(module.DegradationContractError, match="frozen"):
        module.apply_degradation(
            _neutral_reference(),
            control=control,
            condition_id="x2-clean",
            item_id="fixture-work-01-page-01",
            source_group_id="fixture-work-01",
            fixture_manifest_id="phase2-score-fixtures-v1",
            purpose="benchmark",
        )


def test_impulse_and_staff_geometry_remain_lower_right_aligned() -> None:
    module = _degradation()
    control = module.load_degradation_control(CONTROL_PATH)
    reference = np.full((65, 67, 3), 255, dtype=np.uint8)
    reference[8:57:8, 4:64] = 0
    reference[32, 36] = 0
    for scale in (2, 4):
        result = module.apply_degradation(
            reference,
            control=control,
            condition_id=f"x{scale}-clean",
            item_id=f"analytical-x{scale}",
            source_group_id="analytical-work",
            fixture_manifest_id="phase2-score-fixtures-v1",
            purpose="fixture-preview",
        )
        assert result.trace["crop"] == {
            "top": 0,
            "left": 0,
            "bottom": 65 % scale,
            "right": 67 % scale,
        }
        assert result.pixels.shape[:2] == (65 // scale, 67 // scale)
        assert int(result.pixels.min()) < 255


@pytest.fixture(scope="module")
def preview_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    module = _degradation()
    project_root = PROJECT_ROOT
    artifact_root = tmp_path_factory.mktemp("phase2-preview")
    return module.build_degradation_preview(project_root, artifact_root=artifact_root)


def test_preview_has_fixed_two_per_cell_membership_and_exact_panels(
    preview_bundle: dict[str, Any],
) -> None:
    artifact_root = Path(preview_bundle["artifact_root"])
    manifest = json.loads((artifact_root / "preview-manifest.json").read_text(encoding="utf-8"))
    membership = json.loads((artifact_root / "preview-membership.json").read_text(encoding="utf-8"))

    assert len(membership["panels"]) == 12
    assert [panel["condition_id"] for panel in membership["panels"]] == [
        condition_id for condition_id in EXPECTED_CELLS for _ in range(2)
    ]
    for condition_id in EXPECTED_CELLS:
        selected = [
            panel for panel in membership["panels"] if panel["condition_id"] == condition_id
        ]
        assert len(selected) == 2
        assert len({panel["source_group_id"] for panel in selected}) == 2
    assert len(manifest["panels"]) == 12
    assert len(manifest["panel_sha256s"]) == 12
    assert all((artifact_root / panel["relative_path"]).is_file() for panel in manifest["panels"])
    assert "metric" not in json.dumps(manifest).casefold()


def test_working_copy_execute_is_ignored_and_source_only_stays_clean(
    preview_bundle: dict[str, Any],
) -> None:
    module = _degradation()
    artifact_root = Path(preview_bundle["artifact_root"])
    source_path = PROJECT_ROOT / "notebooks/02-degradation-preview.ipynb"
    working_path = artifact_root / "preview-working.ipynb"

    source_digest = module.assert_notebook_source_clean(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    working = json.loads(working_path.read_text(encoding="utf-8"))
    assert all(
        cell.get("execution_count") is None
        for cell in source["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(not cell.get("outputs") for cell in source["cells"] if cell["cell_type"] == "code")
    assert any(
        cell.get("execution_count") is not None
        for cell in working["cells"]
        if cell["cell_type"] == "code"
    )
    assert "phase2_artifact_root" not in working["metadata"]
    assert source_digest == preview_bundle["notebook_source_sha256"]
    assert working_path.is_relative_to(artifact_root)


@pytest.fixture(scope="module")
def relocatable_preview_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    module = _degradation()
    project_root = tmp_path_factory.mktemp("relocatable-phase2-project")
    tracked_inputs = (
        Path("pyproject.toml"),
        Path("configs/degradations/controlled-score-candidates.yaml"),
        Path("tests/fixtures/phase2/fixture-manifest-v1.yaml"),
        Path("notebooks/02-degradation-preview.ipynb"),
    )
    for relative_path in tracked_inputs:
        destination = project_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_path, destination)
    artifact_root = project_root / "artifacts/phase2-degradation-preview"
    preview = module.build_degradation_preview(project_root, artifact_root=artifact_root)
    return project_root, preview


@pytest.mark.parametrize("working_directory", ("notebooks", "artifact-root"))
def test_generated_working_copy_reopens_from_vscode_working_directories(
    relocatable_preview_project: tuple[Path, dict[str, Any]],
    working_directory: str,
) -> None:
    import nbformat
    from nbclient import NotebookClient

    project_root, preview = relocatable_preview_project
    artifact_root = Path(preview["artifact_root"])
    execution_root = (
        project_root / "notebooks" if working_directory == "notebooks" else artifact_root
    )
    notebook = nbformat.read(artifact_root / "preview-working.ipynb", as_version=4)

    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(execution_root)}},
    ).execute()

    assert not (artifact_root / "degradation-decision.json").exists()


def test_notebook_source_clean_rejects_execution_output_and_embedded_payload(
    tmp_path: Path,
) -> None:
    module = _degradation()
    source_path = PROJECT_ROOT / "notebooks/02-degradation-preview.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code["execution_count"] = 1
    code["outputs"] = [
        {
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/png": "aGVsbG8="},
        }
    ]
    dirty = tmp_path / "dirty.ipynb"
    dirty.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(module.NotebookSourceError):
        module.assert_notebook_source_clean(dirty)


def _decision_for_preview(preview_bundle: dict[str, Any], decision: str) -> dict[str, Any]:
    artifact_root = Path(preview_bundle["artifact_root"])
    manifest = json.loads((artifact_root / "preview-manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 2,
        "record_type": "degradation-review",
        "decision": decision,
        "reviewer": "Fixture Test Reviewer",
        "reviewed_at": "2026-08-21T10:00:00Z",
        "rationale": "The fixed analytical panels are suitable for exercising the freeze gate.",
        "candidate_id": manifest["candidate_id"],
        "candidate_sha256": manifest["candidate_sha256"],
        "notebook_source_sha256": manifest["notebook_source_sha256"],
        "preview_manifest_sha256": preview_bundle["preview_manifest_sha256"],
        "membership_sha256": manifest["membership_sha256"],
        "panel_sha256s": manifest["panel_sha256s"],
        "authorship": "human-recorded-in-working-notebook",
    }


def test_freeze_gate_rejects_missing_stale_agent_authored_or_rejected_review(
    tmp_path: Path, preview_bundle: dict[str, Any]
) -> None:
    module = _degradation()
    artifact_root = Path(preview_bundle["artifact_root"])
    preview_manifest = artifact_root / "preview-manifest.json"
    frozen = tmp_path / "controlled-score-v1.yaml"
    reconciliation = tmp_path / "degradation-decision-reconciliation.json"

    with pytest.raises(module.DegradationDecisionError):
        module.freeze_degradation_control(
            CONTROL_PATH,
            tmp_path / "missing.json",
            preview_manifest,
            frozen,
            reconciliation_path=reconciliation,
        )
    agent_authored = _decision_for_preview(preview_bundle, "accept")
    agent_authored["authorship"] = "agent-authored"
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(agent_authored), encoding="utf-8")
    with pytest.raises(module.DegradationDecisionError):
        module.freeze_degradation_control(
            CONTROL_PATH,
            decision_path,
            preview_manifest,
            frozen,
            reconciliation_path=reconciliation,
        )
    rejected = _decision_for_preview(preview_bundle, "reject")
    decision_path.write_text(json.dumps(rejected), encoding="utf-8")
    blocked = module.freeze_degradation_control(
        CONTROL_PATH,
        decision_path,
        preview_manifest,
        frozen,
        reconciliation_path=reconciliation,
    )
    assert blocked["status"] == "blocked-rejected"
    assert not frozen.exists()
    assert not reconciliation.exists()


def test_freeze_gate_accepts_only_exact_content_bound_human_review(
    tmp_path: Path, preview_bundle: dict[str, Any]
) -> None:
    module = _degradation()
    artifact_root = Path(preview_bundle["artifact_root"])
    preview_manifest = artifact_root / "preview-manifest.json"
    decision = _decision_for_preview(preview_bundle, "accept")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    frozen = tmp_path / "controlled-score-v1.yaml"
    reconciliation = tmp_path / "degradation-decision-reconciliation.json"

    result = module.freeze_degradation_control(
        CONTROL_PATH,
        decision_path,
        preview_manifest,
        frozen,
        reconciliation_path=reconciliation,
    )
    assert result["status"] == "frozen"
    assert frozen.is_file()
    assert reconciliation.is_file()
    frozen_control = _yaml(frozen)
    assert frozen_control["status"] == "frozen"
    assert frozen_control["candidate_sha256"] == decision["candidate_sha256"]
    assert frozen_control["decision_reconciliation_sha256"] == result["reconciliation_sha256"]
