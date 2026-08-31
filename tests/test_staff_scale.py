from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    BenchmarkState,
    assert_smb_purpose_allowed,
)
from score_super_resolution.comparison import ensure_manifest_generation
from score_super_resolution.contracts import load_schema, validate_instance
from score_super_resolution.identities import canonical_sha256
from score_super_resolution.staff_scale import (
    CONDITIONS,
    ESTIMATOR_ID,
    StaffScaleError,
    apply_scale_normalized_degradation,
    canonical_smb_pixel_sha256,
    estimate_staff_spacing,
    load_evaluation_sample_v2,
    load_scale_normalized_control,
)

ROOT = Path(__file__).resolve().parents[1]
V2_CONTROL_ROOT = ROOT / "configs/smb-evaluation-v2"
V2_FREEZE_CONTROLS = tuple(
    path
    for path in sorted(V2_CONTROL_ROOT.glob("*.json"))
    if path.name not in {"human-review.json", "unlock.json"}
)


def _synthetic_staff_page(gap: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    height = max(400, gap * 35)
    width = 700
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    regions: list[dict[str, object]] = []
    for top in (gap * 3, gap * 18):
        for line_index in range(5):
            y = top + line_index * gap
            cv2.line(pixels, (30, y), (width - 30, y), (0, 0, 0), 1, cv2.LINE_8)
        region_top = max(0, top - gap * 2)
        region_height = gap * 9
        regions.append(
            {
                "bbox": {
                    "x": 0.0,
                    "y": 100.0 * region_top / height,
                    "width": 100.0,
                    "height": 100.0 * region_height / height,
                }
            }
        )
    return pixels, regions


def test_v2_sample_is_work_disjoint_from_the_disclosed_v1_pilot() -> None:
    sample = load_evaluation_sample_v2(ROOT)
    with (ROOT / "data/audits/smb-evaluation-v1-source-groups.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        v1_groups = {row["source_group_id"] for row in csv.DictReader(source)}

    assert len(sample) == 64
    assert len({row.item_id for row in sample}) == 64
    assert len({row.source_group_id for row in sample}) == 64
    assert not ({row.source_group_id for row in sample} & v1_groups)
    assert all(row.estimator_id == ESTIMATOR_ID for row in sample)
    assert all(row.staff_sequence_count >= 2 for row in sample)
    assert min(row.staff_spacing_px for row in sample) >= 4.0
    assert max(row.staff_spacing_px for row in sample) <= 32.0


def test_staff_estimator_recovers_scale_instead_of_absolute_pixels() -> None:
    small_pixels, small_regions = _synthetic_staff_page(8)
    large_pixels, large_regions = _synthetic_staff_page(20)

    small = estimate_staff_spacing(small_pixels, small_regions)
    large = estimate_staff_spacing(large_pixels, large_regions)

    assert small.spacing_px == pytest.approx(8.0, abs=0.1)
    assert large.spacing_px == pytest.approx(20.0, abs=0.1)
    assert small.sequence_count >= 2
    assert large.sequence_count >= 2
    with pytest.raises(StaffScaleError, match="fewer than two"):
        estimate_staff_spacing(np.full((200, 300, 3), 255, dtype=np.uint8), small_regions[:1])


def test_canonical_smb_pixel_hash_uses_the_audited_rgba_frame() -> None:
    pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    framed = (
        b"smb-canonical-rgba-frame-v2\0"
        + (2).to_bytes(8, "big")
        + (1).to_bytes(8, "big")
        + b"RGBA8\0"
        + image.convert("RGBA").tobytes()
    )

    assert canonical_smb_pixel_sha256(image) == hashlib.sha256(framed).hexdigest()
    with pytest.raises(StaffScaleError, match="Pillow image"):
        canonical_smb_pixel_sha256(pixels)  # type: ignore[arg-type]


def test_v2_degradation_scales_blur_with_the_frozen_staff_spacing() -> None:
    control = load_scale_normalized_control(ROOT)
    pixels, _ = _synthetic_staff_page(10)
    arguments = {
        "pixels": pixels,
        "control": control,
        "condition_id": "x2-strong",
        "item_id": "smb-test-000000",
        "source_group_id": "synthetic-work",
    }
    small = apply_scale_normalized_degradation(**arguments, staff_spacing_px=8.0)
    repeated = apply_scale_normalized_degradation(**arguments, staff_spacing_px=8.0)
    large = apply_scale_normalized_degradation(**arguments, staff_spacing_px=20.0)

    small_blur = small.trace["operations"][0]
    large_blur = large.trace["operations"][0]
    assert small_blur["effective_sigma_px"] == pytest.approx(1.2)
    assert large_blur["effective_sigma_px"] == pytest.approx(3.0)
    assert np.array_equal(small.pixels, repeated.pixels)
    assert small.trace == repeated.trace
    assert small.pixels.shape[:2] == (pixels.shape[0] // 2, pixels.shape[1] // 2)
    validate_instance("staff-scale-degradation-trace", small.trace, version=1)


def test_v2_control_and_experiment_are_exact_and_outcome_blind() -> None:
    control_payload = yaml.safe_load(
        (ROOT / "configs/degradations/staff-scale-score-v2.yaml").read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        (ROOT / "configs/experiments/smb-pretrained-evaluation-v2.yaml").read_text(encoding="utf-8")
    )
    validate_instance("staff-scale-degradation-control", control_payload, version=1)
    assert canonical_sha256(control_payload) == (
        "69e80f8884746cfe61b2e52e75bccc9a4cf78e929f37de0b9805a90f3ea0d809"
    )
    assert tuple(control_payload["condition_order"]) == CONDITIONS
    assert experiment["status"] == "frozen"
    assert experiment["sample"]["independent_source_group_count"] == 64
    assert experiment["sample"]["excludes_v1_source_groups"] is True
    assert tuple(experiment["conditions"]) == CONDITIONS
    assert canonical_sha256(experiment["qualitative"]["assignment"]) == (
        "bd92071d1664ed0cfe6e773a8e5691af1875c5eb72489b743f24c95df7b8dd83"
    )
    assert "No v2 page or model output was inspected" in control_payload["calibration"]["decision"]


@pytest.mark.parametrize(
    "schema_id", ["staff-scale-degradation-control", "staff-scale-degradation-trace"]
)
def test_v2_staff_scale_schemas_are_registered(schema_id: str) -> None:
    assert load_schema(schema_id, version=1)["$id"].endswith(schema_id)


@pytest.mark.parametrize("path", V2_FREEZE_CONTROLS)
def test_v2_freeze_controls_satisfy_the_policy_schema(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_instance("smb-freeze-control", payload, version=1)
    assert payload["status"] == "frozen"
    assert payload["control_id"] == path.stem


def test_v2_human_unlock_resolves_every_frozen_prerequisite() -> None:
    unlock_path = V2_CONTROL_ROOT / "unlock.json"
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    validate_instance("smb-evaluation-unlock", unlock, version=1)
    source = yaml.safe_load((ROOT / "data/sources/smb.yaml").read_text(encoding="utf-8"))
    calls: list[str] = []

    result = assert_smb_purpose_allowed(
        source_descriptor=source,
        purpose=BenchmarkPurpose.INFERENCE,
        state=BenchmarkState.EVALUATION_UNLOCKED,
        unlock_record=unlock,
        project_root=ROOT,
        manifest_generation_root=ensure_manifest_generation(ROOT),
        callback=lambda: calls.append("allowed") or "v2-ready",
    )

    assert result == "v2-ready"
    assert calls == ["allowed"]
    assert (
        unlock["prerequisites"]["human_unlock_recorded"]["artifact_sha256"]
        == hashlib.sha256((V2_CONTROL_ROOT / "human_unlock_recorded.json").read_bytes()).hexdigest()
    )


def test_v2_notebook_installs_cairosvg_without_replacing_runtime_pillow() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/03-smb-model-comparison-v2.ipynb").read_text(encoding="utf-8")
    )
    setup_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "torch_before" in "".join(cell["source"])
    ]

    assert len(setup_sources) == 1
    setup = setup_sources[0]
    for dependency in (
        '"cairosvg==2.9.0"',
        '"cairocffi==1.7.1"',
        '"cssselect2==0.9.0"',
        '"defusedxml==0.7.1"',
        '"tinycss2==1.5.1"',
        '"webencodings==0.5.1"',
    ):
        assert dependency in setup
    assert '"--no-deps"' in setup
    assert '"pillow==' not in setup.lower()
    assert "import PIL, torchvision, spandrel" in setup
    assert "import cairocffi, cssselect2, cairosvg" in setup


def test_v2_notebook_preflight_checks_manifest_identity_without_repeating_estimation() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/03-smb-model-comparison-v2.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "canonical_smb_pixel_sha256(source_image)" in source
    assert 'manifest_row["image"]["pixel_sha256"]' in source
    assert "preflight.identity_match.all()" in source
    assert "estimate_staff_spacing" not in source


def test_v2_notebook_serializes_numpy_validation_flags_as_builtin_booleans() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/03-smb-model-comparison-v2.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert '"validation": {name: bool(passed) for name, passed in checks.items()}' in source
