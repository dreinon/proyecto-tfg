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


def test_baseline_method_contract_rejects_non_positive_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    from score_super_resolution.resources import (
        ResourceMeasurementError,
        measure_baseline_resources,
    )

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


def _metric_control(
    *,
    item_id: str = "fixture-page-01",
    source_group_id: str = "fixture-work-01",
    condition_id: str = "x2-clean",
    method_id: str = "bicubic-opencv-v1",
) -> object:
    from score_super_resolution.evaluation import FidelityControl, load_evaluation_control

    return FidelityControl(
        evaluation=load_evaluation_control(EVALUATION_CONTROL_PATH),
        experiment_id="experiment-fixture-v1",
        item_id=item_id,
        source_group_id=source_group_id,
        condition_id=condition_id,
        method_id=method_id,
        reconstruction_id=f"output-{item_id}-{condition_id}-{method_id}",
        reference_id=f"reference-{item_id}",
    )


def _record(records: tuple[dict[str, object], ...], metric: str, colour: str, domain: str):
    matches = [
        record
        for record in records
        if record["metric_name"] == metric
        and record["colour_id"] == colour
        and record["domain_id"] == domain
    ]
    assert len(matches) == 1
    return matches[0]


def test_metric_control_freezes_primary_and_diagnostic_roles() -> None:
    from score_super_resolution.evaluation import load_evaluation_control

    control = load_evaluation_control(EVALUATION_CONTROL_PATH)
    assert control.evaluation_control_id == "evaluation-score-v1"
    assert control.condition_ids == (
        "x2-clean",
        "x2-moderate",
        "x2-strong",
        "x4-clean",
        "x4-moderate",
        "x4-strong",
    )
    assert control.metric_ids == ("psnr-v1", "ssim-wang11-v1")
    assert control.colour_ids == ("bt601-y-primary-v1", "rgb-diagnostic-v1")
    assert control.domain_ids == (
        "full-page-primary-v1",
        "scale-inner-crop-sensitivity-v1",
    )
    assert control.bootstrap_seed == 20260823
    assert control.bootstrap_repetitions == 10_000
    assert control.reference_method_id == "bicubic-opencv-v1"
    assert len(control.sha256) == 64


def test_metric_bt601_y_anchors_and_exact_match_state_are_explicit() -> None:
    from score_super_resolution.evaluation import bt601_y, compute_fidelity

    black = np.zeros((24, 24, 3), dtype=np.uint8)
    white = np.full((24, 24, 3), 255, dtype=np.uint8)
    red = np.zeros((1, 1, 3), dtype=np.uint8)
    red[..., 0] = 255
    assert float(bt601_y(black)[0, 0]) == pytest.approx(16.0, abs=1e-12)
    assert float(bt601_y(white)[0, 0]) == pytest.approx(235.0, abs=1e-12)
    assert float(bt601_y(red)[0, 0]) == pytest.approx(81.481, abs=1e-3)

    records = compute_fidelity(black, black.copy(), scale=2, control=_metric_control())
    assert len(records) == 8
    for record in records:
        validate_instance("metric-result", record, version=2)
        json.dumps(record, allow_nan=False, sort_keys=True)
        assert record["direction"] == {
            "reference": "aligned-hr",
            "reconstruction": "method-output",
        }
        if record["metric_name"] == "psnr":
            assert record["value_state"] == "positive_infinity"
            assert record["value"] is None
            assert record["exact_match"] is True
        else:
            assert record["value_state"] == "finite"
            assert record["value"] == pytest.approx(1.0)
            assert record["exact_match"] is True


def test_metric_psnr_matches_analytical_rgb_value_and_domains_remain_separate() -> None:
    from score_super_resolution.evaluation import compute_fidelity

    reference = np.zeros((24, 24, 3), dtype=np.uint8)
    estimate = np.full_like(reference, 10)
    records = compute_fidelity(reference, estimate, scale=2, control=_metric_control())
    full_rgb = _record(records, "psnr", "rgb-diagnostic-v1", "full-page-primary-v1")
    crop_rgb = _record(records, "psnr", "rgb-diagnostic-v1", "scale-inner-crop-sensitivity-v1")
    expected = 20.0 * np.log10(255.0 / 10.0)
    assert full_rgb["value"] == pytest.approx(expected, abs=1e-12)
    assert crop_rgb["value"] == pytest.approx(expected, abs=1e-12)
    assert full_rgb["parameters"]["data_range"] == 255.0
    assert full_rgb["parameters"]["channel_axis"] == -1
    assert crop_rgb["parameters"]["crop_pixels_per_edge"] == 2
    assert full_rgb["metric_id"] != crop_rgb["metric_id"]


def _direct_wang_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    from score_super_resolution.evaluation import bt601_y

    x = bt601_y(reference)
    y = bt601_y(estimate)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    ux = cv2.filter2D(x, -1, window, borderType=cv2.BORDER_REFLECT)
    uy = cv2.filter2D(y, -1, window, borderType=cv2.BORDER_REFLECT)
    uxx = cv2.filter2D(x * x, -1, window, borderType=cv2.BORDER_REFLECT)
    uyy = cv2.filter2D(y * y, -1, window, borderType=cv2.BORDER_REFLECT)
    uxy = cv2.filter2D(x * y, -1, window, borderType=cv2.BORDER_REFLECT)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    score = ((2.0 * ux * uy + c1) * (2.0 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(score[5:-5, 5:-5].mean())


def test_metric_ssim_matches_independent_wang_oracle_and_border_sensitivity() -> None:
    from score_super_resolution.evaluation import compute_fidelity

    rng = np.random.Generator(np.random.PCG64(717))
    reference = rng.integers(0, 256, size=(31, 35, 3), dtype=np.uint8)
    estimate = reference.copy()
    estimate[12:16, 14:19] = np.clip(estimate[12:16, 14:19].astype(int) + 17, 0, 255)
    records = compute_fidelity(reference, estimate, scale=2, control=_metric_control())
    primary = _record(records, "ssim", "bt601-y-primary-v1", "full-page-primary-v1")
    assert primary["value"] == pytest.approx(
        _direct_wang_ssim(reference, estimate), rel=0, abs=2e-12
    )
    assert primary["parameters"] == {
        "data_range": 255.0,
        "gaussian_weights": True,
        "sigma": 1.5,
        "window_size": 11,
        "use_sample_covariance": False,
        "k1": 0.01,
        "k2": 0.03,
        "channel_axis": None,
        "crop_pixels_per_edge": 0,
    }

    border = reference.copy()
    border[:2] = 0
    border[-2:] = 0
    border[:, :2] = 0
    border[:, -2:] = 0
    border_records = compute_fidelity(reference, border, scale=2, control=_metric_control())
    full = _record(border_records, "psnr", "rgb-diagnostic-v1", "full-page-primary-v1")
    crop = _record(border_records, "psnr", "rgb-diagnostic-v1", "scale-inner-crop-sensitivity-v1")
    assert full["value_state"] == "finite"
    assert crop["value_state"] == "positive_infinity"


@pytest.mark.parametrize(
    ("reference", "estimate", "scale", "message"),
    [
        (_lr_page(), _lr_page()[:17], 2, "shape"),
        (_lr_page().astype(np.float32), _lr_page(), 2, "uint8"),
        (_lr_page(12, 12), _lr_page(12, 12), 4, "SSIM"),
        (_lr_page(), _lr_page(), 3, "scale"),
    ],
)
def test_metric_contract_fails_closed_on_invalid_arrays_or_crop(
    reference: np.ndarray, estimate: np.ndarray, scale: int, message: str
) -> None:
    from score_super_resolution.evaluation import EvaluationContractError, compute_fidelity

    with pytest.raises(EvaluationContractError, match=message):
        compute_fidelity(reference, estimate, scale=scale, control=_metric_control())


def _synthetic_metric_records(
    *, sources: int = 2
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from score_super_resolution.evaluation import load_evaluation_control

    from score_super_resolution.identities import canonical_sha256

    evaluation = load_evaluation_control(EVALUATION_CONTROL_PATH)
    records: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    for cell_index, condition_id in enumerate(evaluation.condition_ids):
        for source_index in range(sources):
            source_id = f"fixture-work-{source_index + 1:02d}"
            for page_index in range(2):
                item_id = f"{source_id}-page-{page_index + 1:02d}"
                for method_index, method_id in enumerate(evaluation.method_ids):
                    states.append(
                        {
                            "item_id": item_id,
                            "source_group_id": source_id,
                            "condition_id": condition_id,
                            "method_id": method_id,
                            "state": "success",
                            "attempt_count": 1,
                            "exclusion_reason": None,
                        }
                    )
                    for metric_name, metric_id in (("psnr", "psnr-v1"), ("ssim", "ssim-wang11-v1")):
                        for colour_id, role in (
                            ("bt601-y-primary-v1", "primary"),
                            ("rgb-diagnostic-v1", "diagnostic"),
                        ):
                            for domain_id in evaluation.domain_ids:
                                base = 30.0 if metric_name == "psnr" else 0.8
                                value = (
                                    base
                                    + cell_index * 0.01
                                    + source_index * 0.1
                                    + page_index * 0.02
                                    + method_index * 0.03
                                )
                                payload = {
                                    "schema_version": 2,
                                    "record_type": "metric-result",
                                    "experiment_id": "experiment-fixture-v1",
                                    "item_id": item_id,
                                    "source_group_id": source_id,
                                    "condition_id": condition_id,
                                    "method_id": method_id,
                                    "reconstruction_id": f"output-{item_id}-{condition_id}-{method_id}",
                                    "reference_id": f"reference-{item_id}",
                                    "metric_id": f"{metric_id}__{colour_id}__{domain_id}",
                                    "metric_name": metric_name,
                                    "role": role,
                                    "colour_id": colour_id,
                                    "domain_id": domain_id,
                                    "direction": {
                                        "reference": "aligned-hr",
                                        "reconstruction": "method-output",
                                    },
                                    "parameters": {"fixture": True},
                                    "value_state": "finite",
                                    "value": value,
                                    "exact_match": False,
                                    "reference_pixel_sha256": "a" * 64,
                                    "reconstruction_pixel_sha256": "b" * 64,
                                }
                                records.append(
                                    {
                                        **payload,
                                        "metric_result_id": f"metric-{canonical_sha256(payload)}",
                                    }
                                )
    return records, states


def _aggregate_control(records: list[dict[str, object]], states: list[dict[str, object]]) -> object:
    from score_super_resolution.evaluation import AggregateControl, load_evaluation_control

    from score_super_resolution.identities import canonical_sha256

    return AggregateControl(
        evaluation=load_evaluation_control(EVALUATION_CONTROL_PATH),
        experiment_id="experiment-fixture-v1",
        reconciliation_id="reconciliation-fixture-v1",
        reconciliation_sha256="c" * 64,
        raw_metric_input_sha256=canonical_sha256(
            sorted(records, key=lambda row: str(row["metric_result_id"]))
        ),
        tuple_state_input_sha256=canonical_sha256(
            sorted(
                states,
                key=lambda row: (
                    str(row["condition_id"]),
                    str(row["source_group_id"]),
                    str(row["item_id"]),
                    str(row["method_id"]),
                ),
            )
        ),
    )


def test_aggregate_is_six_cell_source_paired_and_deterministic() -> None:
    from score_super_resolution.evaluation import aggregate_paired

    records, states = _synthetic_metric_records()
    control = _aggregate_control(records, states)
    bundle = aggregate_paired(records, states, control=control)
    validate_instance("aggregate-result", bundle, version=2)
    json.dumps(bundle, allow_nan=False, sort_keys=True)

    assert bundle["experiment_id"] == "experiment-fixture-v1"
    assert bundle["reconciliation_id"] == "reconciliation-fixture-v1"
    assert bundle["bicubic_reference_method_id"] == "bicubic-opencv-v1"
    assert [cell["condition_id"] for cell in bundle["cells"]] == _control()["condition_order"]
    assert len(bundle["cells"]) == 6
    for cell in bundle["cells"]:
        assert len(cell["comparisons"]) == 16  # 2 methods x 2 metrics x 2 colours x 2 domains
        for comparison in cell["comparisons"]:
            assert comparison["denominators"] == {
                "expected_pages": 4,
                "pairable_pages": 4,
                "failed_pages": 0,
                "retry_attempts": 0,
                "excluded_pages": 0,
            }
            assert comparison["n_sources"] == 2
            assert comparison["n_pages"] == 4
            assert comparison["paired_difference"]["sample_sd"] == pytest.approx(0.0, abs=1e-14)
            assert comparison["bootstrap"] == {
                "state": "available",
                "generator": "PCG64",
                "seed": 20260823,
                "repetitions": 10_000,
                "confidence_level": 0.95,
                "lower": pytest.approx(-0.06),
                "upper": pytest.approx(-0.06),
            }

    reversed_bundle = aggregate_paired(
        list(reversed(records)), list(reversed(states)), control=control
    )
    assert reversed_bundle == bundle


def test_aggregate_declines_interval_for_one_source_and_preserves_failures() -> None:
    from score_super_resolution.evaluation import aggregate_paired

    records, states = _synthetic_metric_records(sources=1)
    failed = next(
        state
        for state in states
        if state["condition_id"] == "x2-clean"
        and state["method_id"] == "nearest-opencv-exact-v1"
        and state["item_id"].endswith("02")
    )
    failed["state"] = "failed"
    failed["attempt_count"] = 2
    records[:] = [
        record
        for record in records
        if not (
            record["condition_id"] == failed["condition_id"]
            and record["method_id"] == failed["method_id"]
            and record["item_id"] == failed["item_id"]
        )
    ]
    bundle = aggregate_paired(records, states, control=_aggregate_control(records, states))
    comparison = next(
        value
        for value in bundle["cells"][0]["comparisons"]
        if value["method_id"] == "nearest-opencv-exact-v1"
    )
    assert comparison["denominators"] == {
        "expected_pages": 2,
        "pairable_pages": 1,
        "failed_pages": 1,
        "retry_attempts": 1,
        "excluded_pages": 0,
    }
    assert comparison["n_sources"] == 1
    assert comparison["n_pages"] == 1
    assert comparison["bootstrap"] == {
        "state": "unavailable_insufficient_sources",
        "generator": "PCG64",
        "seed": 20260823,
        "repetitions": 10_000,
        "confidence_level": 0.95,
        "lower": None,
        "upper": None,
    }


def test_aggregate_rejects_duplicates_digest_drift_and_incomplete_cells() -> None:
    from score_super_resolution.evaluation import EvaluationContractError, aggregate_paired

    records, states = _synthetic_metric_records()
    control = _aggregate_control(records, states)
    with pytest.raises(EvaluationContractError, match="duplicate"):
        aggregate_paired([*records, records[0]], states, control=control)
    with pytest.raises(EvaluationContractError, match="digest"):
        aggregate_paired(records[:-1], states, control=control)

    reduced_records = [record for record in records if record["condition_id"] != "x4-strong"]
    reduced_states = [state for state in states if state["condition_id"] != "x4-strong"]
    with pytest.raises(EvaluationContractError, match="six cells"):
        aggregate_paired(
            reduced_records,
            reduced_states,
            control=_aggregate_control(reduced_records, reduced_states),
        )
