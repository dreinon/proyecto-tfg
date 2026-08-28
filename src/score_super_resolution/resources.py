"""Full-page batch-one CPU resource instrumentation for transparent baselines."""

from __future__ import annotations

import math
import multiprocessing
import platform
import resource
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from score_super_resolution.baselines import (
    BASELINE_METHODS,
    pixel_sha256,
    run_baseline,
    validate_rgb8,
)
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import DegradationResult
from score_super_resolution.identities import canonical_sha256

_ENVIRONMENT_FIELDS = {
    "python_version",
    "platform",
    "machine",
    "cpu_model",
    "logical_cpu_count",
    "opencv_version",
    "opencv_build_sha256",
}
_SECRET_PARTS = {"authorization", "credential", "credentials", "password", "secret", "token"}
_SECRET_NAMES = {"api_key", "apikey", "access_key", "private_key"}


class ResourceMeasurementError(ValueError):
    """Resource inputs or measurements are incomplete, ambiguous, or unsafe."""


def _secret_like_key(value: Any, path: tuple[str, ...] = ()) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _SECRET_NAMES or set(normalized.split("_")) & _SECRET_PARTS:
                return ".".join((*path, key))
            found = _secret_like_key(child, (*path, key))
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _secret_like_key(child, (*path, str(index)))
            if found is not None:
                return found
    return None


def _validated_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        raise ResourceMeasurementError("resource environment must be a mapping")
    secret = _secret_like_key(environment)
    if secret is not None:
        raise ResourceMeasurementError(f"secret-like environment key is forbidden: {secret}")
    if set(environment) != _ENVIRONMENT_FIELDS:
        raise ResourceMeasurementError(
            "resource environment must contain the exact allowlisted fields"
        )
    result = dict(environment)
    if result["platform"] != "Linux" or platform.system() != "Linux":
        raise ResourceMeasurementError("resource protocol requires Linux KiB ru_maxrss semantics")
    if result["opencv_version"] != cv2.__version__:
        raise ResourceMeasurementError("resource environment OpenCV version differs from runtime")
    if not isinstance(result["logical_cpu_count"], int) or result["logical_cpu_count"] < 1:
        raise ResourceMeasurementError("resource environment CPU count is invalid")
    for key in ("python_version", "machine", "cpu_model", "opencv_build_sha256"):
        if not isinstance(result[key], str) or not result[key]:
            raise ResourceMeasurementError("resource environment contains an invalid field")
    return result


def _validate_control(control: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    try:
        resource_control = control["resource"]
        degradation_control = control["degradation_control"]
        condition_order = control["condition_order"]
    except (KeyError, TypeError) as error:
        raise ResourceMeasurementError("evaluation control is incomplete") from error
    expected = {
        "device": "cpu",
        "page_mode": "full-page",
        "batch_size": 1,
        "tiling": "not_applicable",
        "warmup_repetitions": 1,
        "timed_repetitions": 5,
        "timer": "perf_counter_ns",
        "output_encoding_timed": False,
        "process_isolation": "linux-child",
        "peak_rss_measurement": "ru_maxrss",
        "peak_rss_unit": "KiB",
        "timeout_seconds": 30,
        "model_parameters": "not_applicable",
        "model_bytes": "not_applicable",
    }
    if resource_control != expected:
        if isinstance(resource_control, Mapping) and resource_control.get("peak_rss_unit") != "KiB":
            raise ResourceMeasurementError("peak RSS unit must be Linux KiB")
        raise ResourceMeasurementError("resource protocol differs from the frozen control")
    if control.get("evaluation_control_id") != "evaluation-score-v1":
        raise ResourceMeasurementError("evaluation control identity differs")
    if trace.get("condition_id") not in condition_order:
        raise ResourceMeasurementError("degradation condition is outside the evaluation control")
    if trace.get("control_sha256") != degradation_control.get("sha256"):
        raise ResourceMeasurementError("degradation control digest differs")
    return dict(resource_control)


def _resource_worker(
    connection: Any,
    method_id: str,
    pixels: np.ndarray,
    target_shape: tuple[int, int, int],
    condition_id: str,
) -> None:
    try:
        baseline_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        run_baseline(method_id, pixels, target_shape=target_shape, condition_id=condition_id)
        repeats = [
            run_baseline(
                method_id, pixels, target_shape=target_shape, condition_id=condition_id
            ).elapsed_ns
            for _ in range(5)
        ]
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        connection.send({"repeats_ns": repeats, "baseline_rss": baseline_rss, "peak_rss": peak_rss})
    except BaseException as error:  # child boundary must report a bounded generic failure
        connection.send({"error": type(error).__name__})
    finally:
        connection.close()


def measure_baseline_resources(
    method: str,
    full_page: DegradationResult,
    *,
    control: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure one validated full LR page in an isolated Linux child process."""

    if method not in BASELINE_METHODS:
        raise ResourceMeasurementError("resource method is unknown")
    if not isinstance(full_page, DegradationResult) or not isinstance(full_page.trace, Mapping):
        raise ResourceMeasurementError("resource full page must be a degradation result")
    validate_rgb8(full_page.pixels)
    trace = full_page.trace
    if trace.get("output_pixel_sha256") != pixel_sha256(full_page.pixels):
        raise ResourceMeasurementError("resource input pixel digest differs from its trace")
    protocol = _validate_control(control, trace)
    safe_environment = _validated_environment(environment)
    output_dimensions = trace.get("aligned_dimensions")
    input_dimensions = trace.get("output_dimensions")
    if not isinstance(output_dimensions, Mapping) or not isinstance(input_dimensions, Mapping):
        raise ResourceMeasurementError("resource dimensions are absent")
    target_shape = (
        int(output_dimensions.get("height", 0)),
        int(output_dimensions.get("width", 0)),
        int(output_dimensions.get("channels", 0)),
    )
    expected_input = {
        "width": full_page.pixels.shape[1],
        "height": full_page.pixels.shape[0],
        "channels": 3,
    }
    if dict(input_dimensions) != expected_input:
        raise ResourceMeasurementError("resource input dimensions differ from its trace")

    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_resource_worker,
        args=(child, method, full_page.pixels, target_shape, str(trace["condition_id"])),
    )
    process.start()
    child.close()
    if not parent.poll(float(protocol["timeout_seconds"])):
        process.terminate()
        process.join(timeout=5)
        raise ResourceMeasurementError("resource child exceeded the timeout")
    payload = parent.recv()
    parent.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise ResourceMeasurementError("resource child did not terminate")
    if process.exitcode != 0 or "error" in payload:
        raise ResourceMeasurementError("resource child failed safely")

    repeats = payload["repeats_ns"]
    if len(repeats) != 5 or any(not isinstance(value, int) or value <= 0 for value in repeats):
        raise ResourceMeasurementError("resource timing repeats are invalid")
    values = np.asarray(repeats, dtype=np.float64)
    median = float(np.median(values))
    q1, q3 = (float(value) for value in np.quantile(values, [0.25, 0.75], method="linear"))
    throughput = 1_000_000_000.0 / median
    if not all(math.isfinite(value) and value > 0 for value in (median, q1, q3, throughput)):
        raise ResourceMeasurementError("resource timing summary is non-finite")
    baseline_rss = int(payload["baseline_rss"])
    peak_rss = int(payload["peak_rss"])
    if baseline_rss < 1 or peak_rss < baseline_rss:
        raise ResourceMeasurementError("resource Linux KiB peak RSS is inconsistent")

    public_protocol = {
        key: protocol[key]
        for key in (
            "device",
            "page_mode",
            "batch_size",
            "tiling",
            "warmup_repetitions",
            "timed_repetitions",
            "timer",
            "output_encoding_timed",
        )
    }
    content = {
        "schema_version": 2,
        "record_type": "resource-result",
        "evaluation_control_id": str(control["evaluation_control_id"]),
        "degradation_control_sha256": str(control["degradation_control"]["sha256"]),
        "condition_id": str(trace["condition_id"]),
        "method_id": method,
        "input_pixel_sha256": pixel_sha256(full_page.pixels),
        "input_dimensions": expected_input,
        "output_dimensions": dict(output_dimensions),
        "protocol": public_protocol,
        "latency": {
            "unit": "ns",
            "repeats_ns": repeats,
            "median_ns": median,
            "q1_ns": q1,
            "q3_ns": q3,
        },
        "throughput": {"unit": "pages_per_second", "value": throughput},
        "memory": {
            "measurement": "isolated_linux_child_peak_rss",
            "unit": "KiB",
            "baseline_rss": baseline_rss,
            "peak_rss": peak_rss,
        },
        "model_parameters": {"state": "not_applicable", "value": None},
        "model_bytes": {"state": "not_applicable", "value": None},
        "environment": safe_environment,
    }
    record = {
        **content,
        "resource_result_id": f"resource-{canonical_sha256(content)}",
    }
    try:
        validate_instance("resource-result", record, version=2)
    except ContractValidationError as error:
        raise ResourceMeasurementError(str(error)) from error
    return record
