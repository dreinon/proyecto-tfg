"""Frozen fidelity metrics, source-aware paired analysis, and qualitative selection."""

from __future__ import annotations

import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from score_super_resolution.baselines import BASELINE_METHODS, pixel_sha256, validate_rgb8
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.identities import canonical_sha256

type MetricRecord = dict[str, Any]
type AggregateBundle = dict[str, Any]

EXPECTED_CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
EXPECTED_METRICS = ("psnr-v1", "ssim-wang11-v1")
EXPECTED_COLOURS = ("bt601-y-primary-v1", "rgb-diagnostic-v1")
EXPECTED_DOMAINS = ("full-page-primary-v1", "scale-inner-crop-sensitivity-v1")
_MAX_CONTROL_BYTES = 1_048_576


class EvaluationContractError(ValueError):
    """Evaluation controls, records, or arrays violate the frozen scientific protocol."""


@dataclass(frozen=True)
class EvaluationControl:
    """Validated immutable projection of the authored evaluation control."""

    payload: dict[str, Any]
    evaluation_control_id: str
    condition_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    colour_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    reference_method_id: str
    bootstrap_seed: int
    bootstrap_repetitions: int
    sha256: str


@dataclass(frozen=True)
class FidelityControl:
    """Frozen evaluation definition plus one explicit per-page evidence identity."""

    evaluation: EvaluationControl
    experiment_id: str
    item_id: str
    source_group_id: str
    condition_id: str
    method_id: str
    reconstruction_id: str
    reference_id: str


@dataclass(frozen=True)
class AggregateControl:
    """Reconciled aggregate identity and the exact input digests it authorizes."""

    evaluation: EvaluationControl
    experiment_id: str
    reconciliation_id: str
    reconciliation_sha256: str
    raw_metric_input_sha256: str
    tuple_state_input_sha256: str


def _read_control(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise EvaluationContractError("evaluation control must be a regular non-symlink file")
        if metadata.st_size > _MAX_CONTROL_BYTES:
            raise EvaluationContractError("evaluation control exceeds the byte bound")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            raw = os.read(descriptor, _MAX_CONTROL_BYTES + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if len(raw) > _MAX_CONTROL_BYTES or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise EvaluationContractError("evaluation control changed while being read")
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except EvaluationContractError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvaluationContractError("evaluation control cannot be read safely") from error
    if not isinstance(loaded, dict):
        raise EvaluationContractError("evaluation control root must be a mapping")
    return loaded


def load_evaluation_control(path: Path) -> EvaluationControl:
    """Load and semantically validate every calculation-relevant option before use."""

    payload = _read_control(path)
    try:
        condition_ids = tuple(payload["condition_order"])
        methods = payload["methods"]
        metrics = payload["metrics"]
        metric_ids = tuple(value["metric_id"] for value in metrics["metric_definitions"])
        colour_ids = tuple(value["colour_id"] for value in metrics["colour_spaces"])
        domain_ids = tuple(value["domain_id"] for value in metrics["domains"])
        method_ids = tuple(value["method_id"] for value in methods)
    except (KeyError, TypeError) as error:
        raise EvaluationContractError("evaluation control is incomplete") from error
    if payload.get("schema_version") != 1 or payload.get("record_type") != "evaluation-control":
        raise EvaluationContractError("evaluation control envelope differs")
    if (
        payload.get("evaluation_control_id") != "evaluation-score-v1"
        or payload.get("status") != "frozen"
    ):
        raise EvaluationContractError("evaluation control identity or state differs")
    if payload.get("claim_boundary") != "fixture-instrumentation-only":
        raise EvaluationContractError("evaluation control claim boundary differs")
    if set(payload) != {
        "schema_version",
        "record_type",
        "evaluation_control_id",
        "status",
        "claim_boundary",
        "degradation_control",
        "condition_order",
        "methods",
        "image_contract",
        "metrics",
        "aggregation",
        "bootstrap",
        "resource",
        "qualitative",
    }:
        raise EvaluationContractError("evaluation control contains unknown or missing sections")
    if payload.get("degradation_control") != {
        "control_id": "controlled-score-v1",
        "sha256": "6a61d9a28d2524c9b4e8b2138a23d93bb6fe10e37b8cd1d439cb35dcbdcd9949",
    }:
        raise EvaluationContractError("frozen degradation control identity differs")
    if payload.get("image_contract") != {
        "mode": "RGB",
        "dtype": "uint8",
        "range": [0, 255],
        "aligned_shape_required": True,
    }:
        raise EvaluationContractError("evaluation image contract differs")
    if condition_ids != EXPECTED_CONDITIONS:
        raise EvaluationContractError("evaluation control must retain the exact six cells")
    if method_ids != tuple(BASELINE_METHODS):
        raise EvaluationContractError("evaluation method registry differs")
    for configured, actual in zip(methods, BASELINE_METHODS.values(), strict=True):
        if configured != {
            "method_id": actual.method_id,
            "interpolation": actual.interpolation_name,
            "role": actual.role,
        }:
            raise EvaluationContractError("evaluation method definition differs")
    if (
        metric_ids != EXPECTED_METRICS
        or colour_ids != EXPECTED_COLOURS
        or domain_ids != EXPECTED_DOMAINS
    ):
        raise EvaluationContractError("evaluation metric matrix differs")
    expected_metrics = {
        "direction": {"reference": "aligned-hr", "reconstruction": "method-output"},
        "colour_spaces": [
            {
                "colour_id": "bt601-y-primary-v1",
                "role": "primary",
                "transform": "skimage-rgb2ycbcr-float64-rgb-div-255-y-channel",
                "nominal_range": [16.0, 235.0],
                "data_range": 255.0,
            },
            {
                "colour_id": "rgb-diagnostic-v1",
                "role": "diagnostic",
                "transform": "float64-rgb-0-255",
                "data_range": 255.0,
                "channel_axis": -1,
            },
        ],
        "metric_definitions": [
            {
                "metric_id": "psnr-v1",
                "kind": "psnr",
                "direction": "higher-is-better",
                "implementation": "skimage.metrics.peak_signal_noise_ratio",
                "exact_match_state": "positive_infinity",
            },
            {
                "metric_id": "ssim-wang11-v1",
                "kind": "ssim",
                "direction": "higher-is-better",
                "implementation": "skimage.metrics.structural_similarity",
                "gaussian_weights": True,
                "sigma": 1.5,
                "window_size": 11,
                "use_sample_covariance": False,
                "k1": 0.01,
                "k2": 0.03,
            },
        ],
        "domains": [
            {
                "domain_id": "full-page-primary-v1",
                "role": "primary",
                "crop_hr_pixels_per_edge": 0,
            },
            {
                "domain_id": "scale-inner-crop-sensitivity-v1",
                "role": "sensitivity",
                "crop_hr_pixels_per_edge": "scale",
            },
        ],
    }
    if metrics != expected_metrics:
        raise EvaluationContractError("metric convention matrix differs")
    aggregation = payload.get("aggregation")
    bootstrap = payload.get("bootstrap")
    expected_aggregation = {
        "independent_unit": "source_group_id",
        "page_pair_key": "item_id",
        "reference_method_id": "bicubic-opencv-v1",
        "page_pairing": "complete-method-reference-intersection",
        "within_source": "arithmetic-mean-of-paired-pages",
        "dispersion": [
            "sample-standard-deviation",
            "median",
            "linear-q1",
            "linear-q3",
            "iqr",
        ],
        "p_values": "forbidden",
    }
    if aggregation != expected_aggregation:
        raise EvaluationContractError("source aggregation convention differs")
    expected_bootstrap = {
        "generator": "PCG64",
        "seed": 20260823,
        "repetitions": 10000,
        "resampling_unit": "source_group_id",
        "statistic": "mean-paired-difference",
        "interval": "percentile",
        "confidence_level": 0.95,
        "quantile_method": "linear",
        "minimum_sources": 2,
    }
    if bootstrap != expected_bootstrap:
        raise EvaluationContractError("source bootstrap convention differs")
    return EvaluationControl(
        payload=payload,
        evaluation_control_id="evaluation-score-v1",
        condition_ids=condition_ids,
        method_ids=method_ids,
        metric_ids=metric_ids,
        colour_ids=colour_ids,
        domain_ids=domain_ids,
        reference_method_id="bicubic-opencv-v1",
        bootstrap_seed=20260823,
        bootstrap_repetitions=10000,
        sha256=canonical_sha256(payload),
    )


def bt601_y(rgb: np.ndarray) -> np.ndarray:
    """Return the nominal-range BT.601 Y channel from canonical RGB8 pixels."""

    validate_rgb8(rgb, maximum_pixels=256_000_000)
    value = rgb2ycbcr(rgb.astype(np.float64) / 255.0)[..., 0]
    if not np.isfinite(value).all():
        raise EvaluationContractError("BT.601 Y conversion produced non-finite values")
    return value


def _validate_fidelity_control(control: FidelityControl, scale: int) -> None:
    if not isinstance(control, FidelityControl) or not isinstance(
        control.evaluation, EvaluationControl
    ):
        raise EvaluationContractError("fidelity control must be validated before calculation")
    if canonical_sha256(control.evaluation.payload) != control.evaluation.sha256:
        raise EvaluationContractError("evaluation control changed after validation")
    if scale not in {2, 4}:
        raise EvaluationContractError("metric scale must be x2 or x4")
    if (
        control.condition_id not in control.evaluation.condition_ids
        or int(control.condition_id[1]) != scale
    ):
        raise EvaluationContractError("metric condition and scale differ")
    if control.method_id not in control.evaluation.method_ids:
        raise EvaluationContractError("metric method is unknown")
    for field in (
        control.experiment_id,
        control.item_id,
        control.source_group_id,
        control.reconstruction_id,
        control.reference_id,
    ):
        if not isinstance(field, str) or not field:
            raise EvaluationContractError("metric evidence identity is incomplete")


def _metric_value(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    metric_name: str,
    channel_axis: int | None,
) -> tuple[str, float | None, bool]:
    exact = bool(np.array_equal(reference, reconstruction))
    if metric_name == "psnr":
        if exact:
            return "positive_infinity", None, True
        value = float(peak_signal_noise_ratio(reference, reconstruction, data_range=255.0))
    else:
        value = float(
            structural_similarity(
                reference,
                reconstruction,
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                K1=0.01,
                K2=0.03,
                win_size=11,
                channel_axis=channel_axis,
            )
        )
    if not math.isfinite(value):
        raise EvaluationContractError("metric produced an unsupported non-finite value")
    return "finite", value, exact


def compute_fidelity(
    reference_rgb: np.ndarray,
    reconstruction_rgb: np.ndarray,
    *,
    scale: int,
    control: FidelityControl,
) -> tuple[MetricRecord, ...]:
    """Compute explicit Y-primary and RGB-diagnostic PSNR/SSIM in both domains."""

    _validate_fidelity_control(control, scale)
    try:
        validate_rgb8(reference_rgb, maximum_pixels=256_000_000)
        validate_rgb8(reconstruction_rgb, maximum_pixels=256_000_000)
    except ValueError as error:
        raise EvaluationContractError(str(error)) from error
    if reference_rgb.shape != reconstruction_rgb.shape:
        raise EvaluationContractError("metric reference and reconstruction shape must match")
    if min(reference_rgb.shape[:2]) - 2 * scale < 11:
        raise EvaluationContractError("metric sensitivity crop is too small for Wang SSIM")

    reference_sha = pixel_sha256(reference_rgb)
    reconstruction_sha = pixel_sha256(reconstruction_rgb)
    records: list[MetricRecord] = []
    domains = (("full-page-primary-v1", 0), ("scale-inner-crop-sensitivity-v1", scale))
    for domain_id, crop in domains:
        slices = (slice(crop, -crop), slice(crop, -crop)) if crop else (slice(None), slice(None))
        reference_domain = reference_rgb[slices]
        reconstruction_domain = reconstruction_rgb[slices]
        colour_values = (
            (
                "bt601-y-primary-v1",
                "primary",
                bt601_y(np.ascontiguousarray(reference_domain)),
                bt601_y(np.ascontiguousarray(reconstruction_domain)),
                None,
            ),
            (
                "rgb-diagnostic-v1",
                "diagnostic",
                reference_domain.astype(np.float64),
                reconstruction_domain.astype(np.float64),
                -1,
            ),
        )
        for colour_id, role, reference, reconstruction, channel_axis in colour_values:
            for metric_name, metric_base_id in (("psnr", "psnr-v1"), ("ssim", "ssim-wang11-v1")):
                state, value, exact = _metric_value(
                    reference,
                    reconstruction,
                    metric_name=metric_name,
                    channel_axis=channel_axis,
                )
                parameters: dict[str, Any] = {
                    "data_range": 255.0,
                    "channel_axis": channel_axis,
                    "crop_pixels_per_edge": crop,
                }
                if metric_name == "ssim":
                    parameters = {
                        "data_range": 255.0,
                        "gaussian_weights": True,
                        "sigma": 1.5,
                        "window_size": 11,
                        "use_sample_covariance": False,
                        "k1": 0.01,
                        "k2": 0.03,
                        "channel_axis": channel_axis,
                        "crop_pixels_per_edge": crop,
                    }
                metric_id = f"{metric_base_id}__{colour_id}__{domain_id}"
                payload: MetricRecord = {
                    "schema_version": 2,
                    "record_type": "metric-result",
                    "experiment_id": control.experiment_id,
                    "item_id": control.item_id,
                    "source_group_id": control.source_group_id,
                    "condition_id": control.condition_id,
                    "method_id": control.method_id,
                    "reconstruction_id": control.reconstruction_id,
                    "reference_id": control.reference_id,
                    "metric_id": metric_id,
                    "metric_name": metric_name,
                    "role": role,
                    "colour_id": colour_id,
                    "domain_id": domain_id,
                    "direction": {"reference": "aligned-hr", "reconstruction": "method-output"},
                    "parameters": parameters,
                    "value_state": state,
                    "value": value,
                    "exact_match": exact,
                    "reference_pixel_sha256": reference_sha,
                    "reconstruction_pixel_sha256": reconstruction_sha,
                }
                record = {**payload, "metric_result_id": f"metric-{canonical_sha256(payload)}"}
                try:
                    validate_instance("metric-result", record, version=2)
                except ContractValidationError as error:
                    raise EvaluationContractError(str(error)) from error
                records.append(record)
    return tuple(records)


def _sorted_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records), key=lambda row: str(row["metric_result_id"])
    )


def _state_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["condition_id"]), str(row["item_id"]), str(row["method_id"])


def _sorted_states(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(state) for state in states),
        key=lambda row: (
            str(row["condition_id"]),
            str(row["source_group_id"]),
            str(row["item_id"]),
            str(row["method_id"]),
        ),
    )


def _summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "state": "unavailable_no_pairs",
            "mean": None,
            "sample_sd": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    if not np.isfinite(array).all():
        raise EvaluationContractError("paired source summary contains non-finite observations")
    q1, median, q3 = (
        float(value) for value in np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    )
    return {
        "state": "available",
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size >= 2 else None,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _bootstrap(values: Sequence[float], control: EvaluationControl) -> dict[str, Any]:
    if len(values) < 2:
        return {
            "state": "unavailable_insufficient_sources",
            "generator": "PCG64",
            "seed": control.bootstrap_seed,
            "repetitions": control.bootstrap_repetitions,
            "confidence_level": 0.95,
            "lower": None,
            "upper": None,
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(control.bootstrap_seed))
    indices = rng.integers(0, len(array), size=(control.bootstrap_repetitions, len(array)))
    replicates = array[indices].mean(axis=1)
    lower, upper = np.quantile(replicates, [0.025, 0.975], method="linear")
    return {
        "state": "available",
        "generator": "PCG64",
        "seed": control.bootstrap_seed,
        "repetitions": control.bootstrap_repetitions,
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def aggregate_paired(
    metric_records: Sequence[Mapping[str, Any]],
    tuple_states: Sequence[Mapping[str, Any]],
    *,
    control: AggregateControl,
) -> AggregateBundle:
    """Aggregate complete page pairs inside sources, then bootstrap independent sources."""

    if not isinstance(control, AggregateControl) or not isinstance(
        control.evaluation, EvaluationControl
    ):
        raise EvaluationContractError("aggregate control must be validated")
    if canonical_sha256(control.evaluation.payload) != control.evaluation.sha256:
        raise EvaluationContractError("evaluation control changed after validation")
    if not control.reconciliation_id or len(control.reconciliation_sha256) != 64:
        raise EvaluationContractError("aggregate reconciliation identity is absent")
    records = _sorted_metrics(metric_records)
    states = _sorted_states(tuple_states)
    record_ids = [str(record.get("metric_result_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise EvaluationContractError("duplicate metric result identity")
    state_keys = [_state_key(state) for state in states]
    if len(state_keys) != len(set(state_keys)):
        raise EvaluationContractError("duplicate tuple state identity")
    if canonical_sha256(records) != control.raw_metric_input_sha256:
        raise EvaluationContractError("raw metric input digest differs")
    if canonical_sha256(states) != control.tuple_state_input_sha256:
        raise EvaluationContractError("tuple state input digest differs")
    for record in records:
        try:
            validate_instance("metric-result", record, version=2)
        except ContractValidationError as error:
            raise EvaluationContractError(f"metric record is corrupted: {error}") from error
        if record["experiment_id"] != control.experiment_id:
            raise EvaluationContractError("metric experiment identity differs")
        if record["value_state"] != "finite" or not isinstance(record["value"], (int, float)):
            raise EvaluationContractError("aggregate requires finite metric values")
    for state in states:
        required = {
            "item_id",
            "source_group_id",
            "condition_id",
            "method_id",
            "state",
            "attempt_count",
            "exclusion_reason",
        }
        if set(state) != required:
            raise EvaluationContractError("tuple state fields are incomplete")
        if (
            state["condition_id"] not in control.evaluation.condition_ids
            or state["method_id"] not in control.evaluation.method_ids
        ):
            raise EvaluationContractError("tuple state identity is outside the control")
        if state["state"] not in {"success", "failed", "excluded"}:
            raise EvaluationContractError("tuple state is not reconciled terminal evidence")
        if not isinstance(state["attempt_count"], int) or state["attempt_count"] < 1:
            raise EvaluationContractError("tuple attempt count is invalid")
        if (state["state"] == "excluded") != isinstance(state["exclusion_reason"], str):
            raise EvaluationContractError("tuple exclusion reason is inconsistent")
    observed_cells = {str(state["condition_id"]) for state in states}
    if observed_cells != set(control.evaluation.condition_ids):
        raise EvaluationContractError("aggregate input must contain exactly the six cells")

    state_index = {_state_key(state): state for state in states}
    item_sources: dict[tuple[str, str], str] = {}
    for state in states:
        identity = (str(state["condition_id"]), str(state["item_id"]))
        source = str(state["source_group_id"])
        if identity in item_sources and item_sources[identity] != source:
            raise EvaluationContractError("one item is assigned to multiple sources")
        item_sources[identity] = source
    for condition_id, item_id in item_sources:
        present = {
            method for cell, item, method in state_index if cell == condition_id and item == item_id
        }
        if present != set(control.evaluation.method_ids):
            raise EvaluationContractError("tuple matrix is incomplete for an expected page")

    record_index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    variants: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (
            str(record["condition_id"]),
            str(record["item_id"]),
            str(record["method_id"]),
            str(record["metric_id"]),
        )
        if key in record_index:
            raise EvaluationContractError("duplicate metric tuple")
        record_index[key] = record
        variants.add(
            (
                str(record["metric_id"]),
                str(record["metric_name"]),
                str(record["colour_id"]),
                str(record["domain_id"]),
            )
        )
    expected_variant_count = len(EXPECTED_METRICS) * len(EXPECTED_COLOURS) * len(EXPECTED_DOMAINS)
    if len(variants) != expected_variant_count:
        raise EvaluationContractError("metric variant matrix is incomplete")

    cells: list[dict[str, Any]] = []
    reference_method = control.evaluation.reference_method_id
    for condition_id in control.evaluation.condition_ids:
        comparisons: list[dict[str, Any]] = []
        items = sorted(item for cell, item in item_sources if cell == condition_id)
        for method_id in control.evaluation.method_ids:
            if method_id == reference_method:
                continue
            for metric_id, metric_name, colour_id, domain_id in sorted(variants):
                method_state_rows = [state_index[(condition_id, item, method_id)] for item in items]
                reference_state_rows = [
                    state_index[(condition_id, item, reference_method)] for item in items
                ]
                pairable = [
                    item
                    for item, method_state, reference_state in zip(
                        items, method_state_rows, reference_state_rows, strict=True
                    )
                    if method_state["state"] == reference_state["state"] == "success"
                ]
                failed = sum(
                    method_state["state"] == "failed" or reference_state["state"] == "failed"
                    for method_state, reference_state in zip(
                        method_state_rows, reference_state_rows, strict=True
                    )
                )
                excluded = sum(
                    method_state["state"] == "excluded" or reference_state["state"] == "excluded"
                    for method_state, reference_state in zip(
                        method_state_rows, reference_state_rows, strict=True
                    )
                )
                retries = sum(
                    int(method_state["attempt_count"])
                    - 1
                    + int(reference_state["attempt_count"])
                    - 1
                    for method_state, reference_state in zip(
                        method_state_rows, reference_state_rows, strict=True
                    )
                )
                by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
                role: str | None = None
                for item in pairable:
                    method_record = record_index.get((condition_id, item, method_id, metric_id))
                    reference_record = record_index.get(
                        (condition_id, item, reference_method, metric_id)
                    )
                    if method_record is None or reference_record is None:
                        raise EvaluationContractError(
                            "successful tuple lacks a pairable metric record"
                        )
                    if (
                        method_record["metric_name"] != metric_name
                        or method_record["colour_id"] != colour_id
                        or method_record["domain_id"] != domain_id
                        or reference_record["metric_name"] != metric_name
                        or reference_record["colour_id"] != colour_id
                        or reference_record["domain_id"] != domain_id
                    ):
                        raise EvaluationContractError("metric identity fields contradict metric ID")
                    role = str(method_record["role"])
                    by_source[item_sources[(condition_id, item)]].append(
                        (float(method_record["value"]), float(reference_record["value"]))
                    )
                method_values: list[float] = []
                reference_values: list[float] = []
                differences: list[float] = []
                for source_id in sorted(by_source):
                    values = np.asarray(by_source[source_id], dtype=np.float64)
                    method_mean = float(values[:, 0].mean())
                    reference_mean = float(values[:, 1].mean())
                    method_values.append(method_mean)
                    reference_values.append(reference_mean)
                    differences.append(method_mean - reference_mean)
                identity = f"{condition_id}-{method_id}-vs-{reference_method}-{metric_id}"
                comparisons.append(
                    {
                        "comparison_id": f"comparison-{canonical_sha256(identity)}",
                        "method_id": method_id,
                        "reference_method_id": reference_method,
                        "metric_id": metric_id,
                        "metric_name": metric_name,
                        "role": (
                            role
                            if role is not None
                            else "primary"
                            if colour_id == "bt601-y-primary-v1"
                            else "diagnostic"
                        ),
                        "colour_id": colour_id,
                        "domain_id": domain_id,
                        "denominators": {
                            "expected_pages": len(items),
                            "pairable_pages": len(pairable),
                            "failed_pages": failed,
                            "retry_attempts": retries,
                            "excluded_pages": excluded,
                        },
                        "n_sources": len(by_source),
                        "n_pages": len(pairable),
                        "method_summary": _summary(method_values),
                        "reference_summary": _summary(reference_values),
                        "paired_difference": _summary(differences),
                        "bootstrap": _bootstrap(differences, control.evaluation),
                    }
                )
        cells.append({"condition_id": condition_id, "comparisons": comparisons})

    payload: AggregateBundle = {
        "schema_version": 2,
        "record_type": "aggregate-result",
        "experiment_id": control.experiment_id,
        "reconciliation_id": control.reconciliation_id,
        "reconciliation_sha256": control.reconciliation_sha256,
        "degradation_control_sha256": control.evaluation.payload["degradation_control"]["sha256"],
        "evaluation_control_id": control.evaluation.evaluation_control_id,
        "evaluation_control_sha256": control.evaluation.sha256,
        "bicubic_reference_method_id": reference_method,
        "raw_metric_input_sha256": control.raw_metric_input_sha256,
        "tuple_state_input_sha256": control.tuple_state_input_sha256,
        "cells": cells,
    }
    bundle = {**payload, "aggregate_result_id": f"aggregate-{canonical_sha256(payload)}"}
    try:
        validate_instance("aggregate-result", bundle, version=2)
    except ContractValidationError as error:
        raise EvaluationContractError(str(error)) from error
    return bundle
