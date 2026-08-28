from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import DegradationResult


PROJECT_ROOT = Path(__file__).parents[1]
EVALUATION_CONTROL_PATH = PROJECT_ROOT / "configs/evaluation/evaluation-score-v1.yaml"


def _control() -> dict[str, object]:
    loaded = yaml.safe_load(EVALUATION_CONTROL_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _lr_page(height: int = 18, width: int = 22) -> np.ndarray:
    y, x = np.indices((height, width))
    page = np.full((height, width, 3), 241, dtype=np.uint8)
    page[(y % 7) == 2] = (20, 20, 20)
    page[(x - y) % 13 == 0] = (80, 40, 120)
    return page


def _pixel_sha256(pixels: np.ndarray) -> str:
    import hashlib

    height, width, channels = pixels.shape
    framed = (
        b"phase2-rgb8-v1\0"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + channels.to_bytes(1, "big")
        + pixels.tobytes(order="C")
    )
    return hashlib.sha256(framed).hexdigest()


def _degradation_result(condition_id: str = "x2-clean") -> DegradationResult:
    scale = int(condition_id[1])
    pixels = _lr_page()
    aligned_shape = (pixels.shape[0] * scale, pixels.shape[1] * scale, 3)
    return DegradationResult(
        pixels=pixels,
        encoded_bytes=b"fixture-only",
        trace={
            "condition_id": condition_id,
            "control_sha256": _control()["degradation_control"]["sha256"],
            "output_pixel_sha256": _pixel_sha256(pixels),
            "output_dimensions": {
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "channels": 3,
            },
            "aligned_dimensions": {
                "width": aligned_shape[1],
                "height": aligned_shape[0],
                "channels": 3,
            },
        },
    )


def test_baseline_registry_uses_three_explicit_common_opencv_methods() -> None:
    from score_super_resolution.baselines import BASELINE_METHODS

    assert tuple(BASELINE_METHODS) == (
        "nearest-opencv-exact-v1",
        "bilinear-opencv-exact-v1",
        "bicubic-opencv-v1",
    )
    assert [method.interpolation for method in BASELINE_METHODS.values()] == [
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR_EXACT,
        cv2.INTER_CUBIC,
    ]
    assert BASELINE_METHODS["bicubic-opencv-v1"].role == "principal-simple-reference"


@pytest.mark.parametrize(
    ("method_id", "interpolation"),
    [
        ("nearest-opencv-exact-v1", cv2.INTER_NEAREST_EXACT),
        ("bilinear-opencv-exact-v1", cv2.INTER_LINEAR_EXACT),
        ("bicubic-opencv-v1", cv2.INTER_CUBIC),
    ],
)
def test_baseline_common_contract_matches_explicit_opencv(
    method_id: str, interpolation: int
) -> None:
    from score_super_resolution.baselines import run_baseline

    lr = _lr_page()
    result = run_baseline(method_id, lr, target_shape=(36, 44, 3), condition_id="x2-clean")
    expected = cv2.resize(lr, (44, 36), interpolation=interpolation)

    assert np.array_equal(result.pixels, expected)
    assert result.pixels.dtype == np.uint8
    assert result.pixels.shape == (36, 44, 3)
    assert result.evidence["method_id"] == method_id
    assert result.evidence["condition_id"] == "x2-clean"
    assert result.evidence["backend"] == "opencv"
    assert result.evidence["opencv_version"] == cv2.__version__
    assert result.evidence["input_dimensions"] == {"width": 22, "height": 18, "channels": 3}
    assert result.evidence["output_dimensions"] == {"width": 44, "height": 36, "channels": 3}
    assert isinstance(result.elapsed_ns, int) and result.elapsed_ns > 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.astype(np.float32), "uint8"),
        (lambda value: value[..., 0], "RGB"),
        (lambda value: value[:, :, ::-1], "contiguous"),
        (lambda value: np.empty((9000, 9000, 3), dtype=np.uint8), "pixel"),
    ],
)
def test_baseline_method_contract_rejects_invalid_or_oversized_input(
    mutation: object, message: str
) -> None:
    from score_super_resolution.baselines import BaselineContractError, run_baseline

    invalid = mutation(_lr_page())  # type: ignore[operator]
    with pytest.raises(BaselineContractError, match=message):
        run_baseline(
            "bicubic-opencv-v1", invalid, target_shape=(36, 44, 3), condition_id="x2-clean"
        )


@pytest.mark.parametrize(
    ("method_id", "target_shape", "condition_id", "message"),
    [
        ("unknown-v1", (36, 44, 3), "x2-clean", "method"),
        ("bicubic-opencv-v1", (0, 44, 3), "x2-clean", "target"),
        ("bicubic-opencv-v1", (36, 44, 1), "x2-clean", "target"),
        ("bicubic-opencv-v1", (36, 44, 3), "x3-clean", "condition"),
    ],
)
def test_baseline_method_contract_rejects_unknown_identity_or_shape(
    method_id: str, target_shape: tuple[int, int, int], condition_id: str, message: str
) -> None:
    from score_super_resolution.baselines import BaselineContractError, run_baseline

    with pytest.raises(BaselineContractError, match=message):
        run_baseline(method_id, _lr_page(), target_shape=target_shape, condition_id=condition_id)


def test_baseline_method_contract_rejects_non_positive_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    import score_super_resolution.baselines as module

    values = iter((100, 100))
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(values))
    with pytest.raises(module.BaselineContractError, match="timing"):
        module.run_baseline(
            "bicubic-opencv-v1", _lr_page(), target_shape=(36, 44, 3), condition_id="x2-clean"
        )


def test_resource_protocol_is_full_page_batch_one_and_schema_valid() -> None:
    from score_super_resolution.resources import measure_baseline_resources

    environment = {
        "python_version": "3.12.12",
        "platform": "Linux",
        "machine": "x86_64",
        "cpu_model": "fixture-cpu",
        "logical_cpu_count": 8,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": "a" * 64,
    }
    record = measure_baseline_resources(
        "bicubic-opencv-v1", _degradation_result(), control=_control(), environment=environment
    )

    assert record["protocol"] == {
        "device": "cpu",
        "page_mode": "full-page",
        "batch_size": 1,
        "tiling": "not_applicable",
        "warmup_repetitions": 1,
        "timed_repetitions": 5,
        "timer": "perf_counter_ns",
        "output_encoding_timed": False,
    }
    assert len(record["latency"]["repeats_ns"]) == 5
    assert all(value > 0 for value in record["latency"]["repeats_ns"])
    assert record["latency"]["median_ns"] > 0
    assert record["latency"]["q1_ns"] <= record["latency"]["median_ns"]
    assert record["latency"]["median_ns"] <= record["latency"]["q3_ns"]
    assert record["throughput"]["unit"] == "pages_per_second"
    assert record["memory"]["unit"] == "KiB"
    assert record["memory"]["measurement"] == "isolated_linux_child_peak_rss"
    assert record["memory"]["peak_rss"] >= record["memory"]["baseline_rss"] > 0
    assert record["model_parameters"] == {"state": "not_applicable", "value": None}
    assert record["model_bytes"] == {"state": "not_applicable", "value": None}
    assert record["environment"] == environment
    validate_instance("resource-result", record, version=2)
    json.dumps(record, allow_nan=False, sort_keys=True)


def test_resource_protocol_rejects_corrupted_input_control_environment_and_units() -> None:
    from score_super_resolution.resources import ResourceMeasurementError, measure_baseline_resources

    valid_environment = {
        "python_version": "3.12.12",
        "platform": "Linux",
        "machine": "x86_64",
        "cpu_model": "fixture-cpu",
        "logical_cpu_count": 8,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": "a" * 64,
    }
    corrupted = _degradation_result()
    corrupted.pixels[0, 0, 0] ^= 1
    with pytest.raises(ResourceMeasurementError, match="digest"):
        measure_baseline_resources(
            "bicubic-opencv-v1", corrupted, control=_control(), environment=valid_environment
        )

    wrong_control = copy.deepcopy(_control())
    wrong_control["resource"]["peak_rss_unit"] = "bytes"  # type: ignore[index]
    with pytest.raises(ResourceMeasurementError, match="KiB"):
        measure_baseline_resources(
            "bicubic-opencv-v1",
            _degradation_result(),
            control=wrong_control,
            environment=valid_environment,
        )

    unsafe_environment = {**valid_environment, "api_token": "must-not-appear"}
    with pytest.raises(ResourceMeasurementError, match="secret"):
        measure_baseline_resources(
            "bicubic-opencv-v1",
            _degradation_result(),
            control=_control(),
            environment=unsafe_environment,
        )

    incomplete_environment = copy.deepcopy(valid_environment)
    incomplete_environment.pop("opencv_build_sha256")
    with pytest.raises(ResourceMeasurementError, match="environment"):
        measure_baseline_resources(
            "bicubic-opencv-v1",
            _degradation_result(),
            control=_control(),
            environment=incomplete_environment,
        )


def test_resource_result_schema_is_closed() -> None:
    schema_path = PROJECT_ROOT / "data/schemas/v2/resource-result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    with pytest.raises(ContractValidationError):
        validate_instance(
            "resource-result",
            {"schema_version": 2, "record_type": "resource-result"},
            version=2,
        )
