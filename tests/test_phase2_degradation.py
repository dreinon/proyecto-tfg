from __future__ import annotations

import copy
import importlib
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
def test_fixture_mutations_fail_before_generation(
    tmp_path: Path, mutation: str
) -> None:
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

