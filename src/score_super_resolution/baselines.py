"""Transparent interpolation baselines sharing one strict RGB8 contract."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

EXPECTED_CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
MAX_INPUT_PIXELS = 64_000_000
MAX_OUTPUT_PIXELS = 256_000_000


class BaselineContractError(ValueError):
    """A baseline input, identity, target, or timing violates the common contract."""


@dataclass(frozen=True)
class BaselineMethod:
    """One named OpenCV interpolation primitive and its comparison role."""

    method_id: str
    interpolation: int
    interpolation_name: str
    role: str


@dataclass(frozen=True)
class BaselineResult:
    """Aligned RGB8 output plus operational evidence from the shared boundary."""

    pixels: np.ndarray
    elapsed_ns: int
    evidence: dict[str, Any]


BASELINE_METHODS = MappingProxyType(
    {
        "nearest-opencv-exact-v1": BaselineMethod(
            "nearest-opencv-exact-v1",
            cv2.INTER_NEAREST_EXACT,
            "INTER_NEAREST_EXACT",
            "transparent-low-complexity-reference",
        ),
        "bilinear-opencv-exact-v1": BaselineMethod(
            "bilinear-opencv-exact-v1",
            cv2.INTER_LINEAR_EXACT,
            "INTER_LINEAR_EXACT",
            "transparent-smooth-reference",
        ),
        "bicubic-opencv-v1": BaselineMethod(
            "bicubic-opencv-v1",
            cv2.INTER_CUBIC,
            "INTER_CUBIC",
            "principal-simple-reference",
        ),
    }
)


def pixel_sha256(pixels: np.ndarray) -> str:
    """Hash canonical RGB8 pixels with explicit dimensions and a domain separator."""

    height, width, channels = pixels.shape
    framed = (
        b"phase2-rgb8-v1\0"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + channels.to_bytes(1, "big")
        + pixels.tobytes(order="C")
    )
    return hashlib.sha256(framed).hexdigest()


def _dimensions(pixels: np.ndarray) -> dict[str, int]:
    return {"width": pixels.shape[1], "height": pixels.shape[0], "channels": pixels.shape[2]}


def validate_rgb8(pixels: np.ndarray, *, maximum_pixels: int = MAX_INPUT_PIXELS) -> None:
    """Reject anything other than a bounded, owned-layout RGB uint8 array."""

    if not isinstance(pixels, np.ndarray):
        raise BaselineContractError("baseline input must be a NumPy array")
    if pixels.dtype != np.uint8:
        raise BaselineContractError("baseline input dtype must be uint8")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise BaselineContractError("baseline input must be an RGB array with three channels")
    if pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise BaselineContractError("baseline input dimensions must be positive")
    if pixels.shape[0] * pixels.shape[1] > maximum_pixels:
        raise BaselineContractError("baseline input pixel count exceeds the safety bound")
    if not pixels.flags.c_contiguous:
        raise BaselineContractError("baseline input must be C-contiguous")


def _validate_target(target_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        not isinstance(target_shape, tuple)
        or len(target_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in target_shape)
    ):
        raise BaselineContractError("target shape must be an explicit (height, width, 3) tuple")
    height, width, channels = target_shape
    if height < 1 or width < 1 or channels != 3:
        raise BaselineContractError("target shape must contain positive RGB dimensions")
    if height * width > MAX_OUTPUT_PIXELS:
        raise BaselineContractError("target pixel count exceeds the safety bound")
    return height, width, channels


def run_baseline(
    method_id: str,
    lr_rgb: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    condition_id: str,
) -> BaselineResult:
    """Resize a validated LR page to explicit aligned-HR dimensions through one boundary."""

    method = BASELINE_METHODS.get(method_id)
    if method is None:
        raise BaselineContractError("baseline method is unknown")
    if condition_id not in EXPECTED_CONDITIONS:
        raise BaselineContractError("baseline condition is outside the frozen six-cell control")
    validate_rgb8(lr_rgb)
    height, width, _ = _validate_target(target_shape)
    expected_scale = int(condition_id[1])
    if height != lr_rgb.shape[0] * expected_scale or width != lr_rgb.shape[1] * expected_scale:
        raise BaselineContractError("target shape is not aligned with the condition scale")

    started = time.perf_counter_ns()
    output = cv2.resize(lr_rgb, (width, height), interpolation=method.interpolation)
    finished = time.perf_counter_ns()
    elapsed_ns = finished - started
    if elapsed_ns <= 0 or not np.isfinite(float(elapsed_ns)):
        raise BaselineContractError("baseline timing must be finite and positive")
    if output.dtype != np.uint8 or output.shape != (height, width, 3):
        raise BaselineContractError("OpenCV returned an invalid aligned RGB8 output")
    output = np.ascontiguousarray(output)
    evidence = {
        "method_id": method.method_id,
        "role": method.role,
        "condition_id": condition_id,
        "backend": "opencv",
        "opencv_version": cv2.__version__,
        "interpolation": method.interpolation_name,
        "input_pixel_sha256": pixel_sha256(lr_rgb),
        "output_pixel_sha256": pixel_sha256(output),
        "input_dimensions": _dimensions(lr_rgb),
        "output_dimensions": _dimensions(output),
    }
    if not re.fullmatch(r"[0-9a-f]{64}", evidence["output_pixel_sha256"]):
        raise BaselineContractError("baseline output digest is invalid")
    return BaselineResult(pixels=output, elapsed_ns=elapsed_ns, evidence=evidence)
