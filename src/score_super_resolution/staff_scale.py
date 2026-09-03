"""Staff-scale-normalized degradation for the corrected SMB evaluation protocol."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from score_super_resolution.baselines import pixel_sha256, validate_rgb8
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import align_reference, derive_degradation_seed
from score_super_resolution.identities import canonical_sha256

CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
ESTIMATOR_ID = "region-deskew-horizontal-morphology-v1"
FULL_PAGE_ESTIMATOR_ID = "full-page-hybrid-horizontal-v2"
SAMPLE_RELATIVE_PATH = Path("data/audits/smb-evaluation-sample-v2.csv")
CONTROL_RELATIVE_PATH = Path("configs/degradations/staff-scale-score-v2.yaml")
SAMPLE_SIZE = 64


class StaffScaleError(ValueError):
    """The staff scale, frozen sample, or normalized degradation is invalid."""


@dataclass(frozen=True)
class StaffSpacingEstimate:
    """One deterministic page-level staff-space estimate in HR pixels."""

    spacing_px: float
    estimator_id: str
    sequence_count: int
    contributing_regions: int
    median_deskew_degrees: float


@dataclass(frozen=True)
class EvaluationSampleV2Row:
    """One work-disjoint SMB evaluation page with a frozen staff scale."""

    upstream_index: int
    item_id: str
    source_group_id: str
    staff_spacing_px: float
    estimator_id: str
    staff_sequence_count: int
    selection_rank: int


@dataclass(frozen=True)
class StaffScaleControl:
    """Validated immutable parameters for the v2 degradation."""

    control_id: str
    master_seed: int
    staff_spacing_min_px: float
    staff_spacing_max_px: float
    conditions: tuple[dict[str, Any], ...]
    sha256: str


@dataclass(frozen=True)
class StaffScaleDegradationResult:
    """One LR result and its effective page-specific parameters."""

    pixels: np.ndarray
    encoded: bytes
    trace: dict[str, Any]


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    index = int(np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() / 2.0))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _validated_region_crop(pixels: np.ndarray, raw_region: object) -> np.ndarray:
    if not isinstance(raw_region, Mapping) or not isinstance(raw_region.get("bbox"), Mapping):
        raise StaffScaleError("SMB region must contain a normalized bounding box")
    bbox = raw_region["bbox"]
    try:
        x, y, width, height = (float(bbox[name]) for name in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as error:
        raise StaffScaleError("SMB region bounding box is malformed") from error
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise StaffScaleError("SMB region bounding box must be finite")
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 100.5 or y + height > 100.5:
        raise StaffScaleError("SMB region bounding box is outside the page")
    page_height, page_width = pixels.shape[:2]
    left = max(0, round(x / 100.0 * page_width))
    top = max(0, round(y / 100.0 * page_height))
    right = min(page_width, left + round(width / 100.0 * page_width))
    bottom = min(page_height, top + round(height / 100.0 * page_height))
    if right - left < 100 or bottom - top < 20:
        raise StaffScaleError("SMB region is too small for staff-scale estimation")
    return np.ascontiguousarray(pixels[top:bottom, left:right])


def _deskew_region(gray: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 40, 120, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(25, width // 12),
        minLineLength=max(40, round(width * 0.35)),
        maxLineGap=max(8, round(width * 0.10)),
    )
    angles: list[float] = []
    weights: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            delta_x = int(x2) - int(x1)
            delta_y = int(y2) - int(y1)
            angle = math.degrees(math.atan2(delta_y, delta_x))
            if abs(delta_x) > width * 0.30 and abs(angle) < 5.0:
                angles.append(angle)
                weights.append(abs(delta_x))
    angle = _weighted_median(angles, weights) if angles else 0.0
    transform = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, angle


def _equidistant_five_line_sequences(rows: Sequence[float]) -> list[float]:
    candidates: list[float] = []
    for index in range(len(rows) - 4):
        gaps = np.diff(np.asarray(rows[index : index + 5], dtype=np.float64))
        median = float(np.median(gaps))
        tolerance = max(1.25, 0.15 * median)
        if 3.0 <= median <= 40.0 and float(np.max(np.abs(gaps - median))) <= tolerance:
            candidates.append(median)
    return candidates


def _region_staff_candidates(region: np.ndarray) -> tuple[list[float], float]:
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    deskewed, angle = _deskew_region(gray)
    _, binary = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    width = binary.shape[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, round(width * 0.25)), 1)),
    )
    row_score = np.count_nonzero(horizontal, axis=1)
    active = np.flatnonzero(row_score >= max(20, round(width * 0.18)))
    if not active.size:
        return [], angle
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    centers = [
        float(np.average(group, weights=row_score[group] + 1))
        for group in groups
        if 1 <= len(group) <= 5
    ]
    return _equidistant_five_line_sequences(centers), angle


def _projection_staff_candidate(region: np.ndarray) -> float | None:
    """Estimate staff periodicity when compression breaks long morphological line support."""

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    deskewed, _angle = _deskew_region(gray)
    darkness = 255.0 - deskewed.astype(np.float64)
    signal = darkness.mean(axis=1)
    trend = cv2.GaussianBlur(signal[:, None], (1, 0), 20.0).ravel()
    centered = signal - trend
    deviation = float(centered.std())
    if not math.isfinite(deviation) or deviation < 1e-6:
        return None
    normalized = (centered - centered.mean()) / deviation
    maximum_lag = min(32, len(normalized) // 4)
    correlations = {
        lag: float(np.mean(normalized[:-lag] * normalized[lag:]))
        for lag in range(4, maximum_lag + 1)
    }
    if not correlations:
        return None
    best_lag = max(correlations, key=correlations.get)
    best_score = correlations[best_lag]
    if best_lag % 2 == 0 and correlations.get(best_lag // 2, -math.inf) >= 0.75 * best_score:
        best_lag //= 2
        best_score = correlations[best_lag]
    if best_score < 0.20:
        return None
    return float(best_lag)


def estimate_staff_spacing(pixels: np.ndarray, regions: object) -> StaffSpacingEstimate:
    """Estimate staff-space from annotated systems without using SR outputs or metrics."""

    validate_rgb8(pixels)
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)) or not regions:
        raise StaffScaleError("staff-scale estimation requires SMB system regions")
    candidates: list[float] = []
    contributing_regions = 0
    angles: list[float] = []
    for raw_region in regions:
        region = _validated_region_crop(pixels, raw_region)
        region_candidates, angle = _region_staff_candidates(region)
        angles.append(angle)
        if region_candidates:
            contributing_regions += 1
            candidates.extend(region_candidates)
    if len(candidates) < 2:
        raise StaffScaleError("fewer than two staff sequences support the page scale")
    initial_median = float(np.median(candidates))
    tolerance = max(1.5, 0.20 * initial_median)
    inliers = [value for value in candidates if abs(value - initial_median) <= tolerance]
    if len(inliers) < 2:
        raise StaffScaleError("staff-spacing candidates are not mutually consistent")
    spacing = float(np.median(inliers))
    if not 4.0 <= spacing <= 32.0:
        raise StaffScaleError("staff spacing is outside the validated estimator range")
    return StaffSpacingEstimate(
        spacing_px=spacing,
        estimator_id=ESTIMATOR_ID,
        sequence_count=len(inliers),
        contributing_regions=contributing_regions,
        median_deskew_degrees=float(np.median(angles)) if angles else 0.0,
    )


def estimate_staff_spacing_full_page(pixels: np.ndarray) -> StaffSpacingEstimate:
    """Estimate staff space without dataset annotations for external professional inputs."""

    validate_rgb8(pixels)
    height, width = pixels.shape[:2]
    if height < 120 or width < 200:
        raise StaffScaleError("full-page staff estimation requires a page-sized image")
    left = round(width * 0.03)
    right = round(width * 0.97)
    top = round(height * 0.03)
    bottom = round(height * 0.97)
    page = np.ascontiguousarray(pixels[top:bottom, left:right])
    midpoint = page.shape[0] // 2
    regions = (
        page,
        np.ascontiguousarray(page[:midpoint]),
        np.ascontiguousarray(page[midpoint:]),
    )
    candidates: list[float] = []
    angles: list[float] = []
    contributing_regions = 0
    for region in regions:
        region_candidates, angle = _region_staff_candidates(region)
        angles.append(angle)
        if region_candidates:
            contributing_regions += 1
            candidates.extend(region_candidates)
    if len(candidates) >= 2:
        initial_median = float(np.median(candidates))
        tolerance = max(1.5, 0.20 * initial_median)
        inliers = [value for value in candidates if abs(value - initial_median) <= tolerance]
    else:
        projection_candidates = [
            candidate
            for region in regions
            if (candidate := _projection_staff_candidate(region)) is not None
        ]
        if len(projection_candidates) < 2:
            raise StaffScaleError("fewer than two staff sequences support the full-page scale")
        initial_median = float(np.median(projection_candidates))
        tolerance = max(1.5, 0.20 * initial_median)
        inliers = [
            value for value in projection_candidates if abs(value - initial_median) <= tolerance
        ]
        contributing_regions = len(inliers)
    if len(inliers) < 2:
        raise StaffScaleError("full-page staff-spacing candidates are not mutually consistent")
    spacing = float(np.median(inliers))
    if not 4.0 <= spacing <= 64.0:
        raise StaffScaleError("full-page staff spacing is outside the supported range")
    return StaffSpacingEstimate(
        spacing_px=spacing,
        estimator_id=FULL_PAGE_ESTIMATOR_ID,
        sequence_count=len(inliers),
        contributing_regions=contributing_regions,
        median_deskew_degrees=float(np.median(angles)),
    )


def load_scale_normalized_control(project_root: Path) -> StaffScaleControl:
    """Load and validate the immutable v2 page-relative degradation control."""

    path = Path(project_root).resolve() / CONTROL_RELATIVE_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_instance("staff-scale-degradation-control", payload, version=1)
    except (OSError, UnicodeError, yaml.YAMLError, ContractValidationError) as error:
        raise StaffScaleError("staff-scale degradation control cannot be validated") from error
    return StaffScaleControl(
        control_id=str(payload["control_id"]),
        master_seed=int(payload["master_seed"]),
        staff_spacing_min_px=float(payload["staff_spacing"]["accepted_range_px"][0]),
        staff_spacing_max_px=float(payload["staff_spacing"]["accepted_range_px"][1]),
        conditions=tuple(dict(condition) for condition in payload["conditions"]),
        sha256=canonical_sha256(payload),
    )


def load_evaluation_sample_v2(project_root: Path) -> tuple[EvaluationSampleV2Row, ...]:
    """Load the frozen 64-page sample with one page per fresh musical work."""

    project_root = Path(project_root).resolve()
    control_path = project_root / CONTROL_RELATIVE_PATH
    try:
        control_payload = yaml.safe_load(control_path.read_text(encoding="utf-8"))
        expected_sha256 = str(control_payload["sample"]["sha256"])
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError) as error:
        raise StaffScaleError("v2 sample identity cannot be resolved") from error
    path = project_root / SAMPLE_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise StaffScaleError("v2 evaluation sample is unavailable")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise StaffScaleError("v2 evaluation sample digest differs")
    try:
        raw_rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as error:
        raise StaffScaleError("v2 evaluation sample cannot be parsed") from error
    fields = {
        "upstream_index",
        "item_id",
        "source_group_id",
        "staff_spacing_px",
        "estimator_id",
        "staff_sequence_count",
        "selection_rank",
    }
    if len(raw_rows) != SAMPLE_SIZE or not raw_rows or set(raw_rows[0]) != fields:
        raise StaffScaleError("v2 evaluation sample shape differs")
    try:
        rows = tuple(
            EvaluationSampleV2Row(
                upstream_index=int(row["upstream_index"]),
                item_id=row["item_id"],
                source_group_id=row["source_group_id"],
                staff_spacing_px=float(row["staff_spacing_px"]),
                estimator_id=row["estimator_id"],
                staff_sequence_count=int(row["staff_sequence_count"]),
                selection_rank=int(row["selection_rank"]),
            )
            for row in raw_rows
        )
    except (TypeError, ValueError) as error:
        raise StaffScaleError("v2 evaluation sample values are malformed") from error
    if (
        len({row.item_id for row in rows}) != SAMPLE_SIZE
        or len({row.source_group_id for row in rows}) != SAMPLE_SIZE
        or {row.selection_rank for row in rows} != set(range(1, SAMPLE_SIZE + 1))
        or any(row.estimator_id != ESTIMATOR_ID or row.staff_sequence_count < 2 for row in rows)
    ):
        raise StaffScaleError("v2 sample independence or estimator evidence differs")
    return rows


def canonical_smb_pixel_sha256(image: Image.Image) -> str:
    """Hash one SMB source image with the audited canonical RGBA v2 framing."""

    if not isinstance(image, Image.Image):
        raise StaffScaleError("SMB pixel identity requires a Pillow image")
    canonical = image.convert("RGBA")
    width, height = canonical.size
    framed = (
        b"smb-canonical-rgba-frame-v2\0"
        + width.to_bytes(8, "big")
        + height.to_bytes(8, "big")
        + b"RGBA8\0"
        + canonical.tobytes()
    )
    return hashlib.sha256(framed).hexdigest()


def _condition(control: StaffScaleControl, condition_id: str) -> dict[str, Any]:
    matches = [entry for entry in control.conditions if entry["condition_id"] == condition_id]
    if len(matches) != 1:
        raise StaffScaleError("condition must be one of the exact six v2 cells")
    return dict(matches[0])


def _encode_jpeg_rgb(pixels: np.ndarray, quality: int) -> tuple[np.ndarray, bytes]:
    options = [
        cv2.IMWRITE_JPEG_QUALITY,
        quality,
        cv2.IMWRITE_JPEG_PROGRESSIVE,
        0,
        cv2.IMWRITE_JPEG_OPTIMIZE,
        0,
    ]
    sampling_key = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    sampling_444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
    if sampling_key is None or sampling_444 is None:
        raise StaffScaleError("OpenCV lacks explicit JPEG 4:4:4 controls")
    options.extend([sampling_key, sampling_444])
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), options)
    if not success:
        raise StaffScaleError("v2 JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise StaffScaleError("v2 JPEG decode verification failed")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)), encoded.tobytes()


def apply_scale_normalized_degradation(
    pixels: np.ndarray,
    *,
    control: StaffScaleControl,
    condition_id: str,
    item_id: str,
    source_group_id: str,
    staff_spacing_px: float,
) -> StaffScaleDegradationResult:
    """Apply v2 with blur expressed as a fixed fraction of each page's staff space."""

    validate_rgb8(pixels)
    if not math.isfinite(staff_spacing_px) or not (
        control.staff_spacing_min_px <= staff_spacing_px <= control.staff_spacing_max_px
    ):
        raise StaffScaleError("frozen staff spacing is outside the control range")
    condition = _condition(control, condition_id)
    scale = int(condition["scale"])
    aligned = align_reference(pixels, scale)
    derived_seed = derive_degradation_seed(
        control.master_seed,
        fixture_manifest_id="smb-staff-scale-evaluation-v2",
        item_id=item_id,
        condition_id=condition_id,
    )
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    working = aligned.pixels.astype(np.float64)
    operations: list[dict[str, Any]] = []
    blur = condition["blur"]
    if blur is not None:
        sigma = max(float(blur["minimum_sigma_px"]), staff_spacing_px * float(blur["staff_ratio"]))
        kernel = 2 * math.ceil(float(blur["kernel_radius_sigma"]) * sigma) + 1
        working = cv2.GaussianBlur(
            working,
            (kernel, kernel),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        operations.append(
            {
                "operator_id": "staff-scale-gaussian-blur",
                "staff_spacing_px": staff_spacing_px,
                "staff_ratio": float(blur["staff_ratio"]),
                "effective_sigma_px": sigma,
                "kernel": kernel,
            }
        )
    target = (aligned.pixels.shape[1] // scale, aligned.pixels.shape[0] // scale)
    working = cv2.resize(working, target, interpolation=cv2.INTER_AREA)
    operations.append({"operator_id": "reduction", "scale": scale, "interpolation": "INTER_AREA"})
    noise = condition["noise"]
    encoded = b""
    if noise is not None:
        noise_sigma = float(noise["sigma"])
        perturbation = np.random.Generator(np.random.PCG64(derived_seed)).normal(
            0.0, noise_sigma, size=(*working.shape[:2], 1)
        )
        working = np.clip(np.rint(working + perturbation), 0, 255).astype(np.uint8)
        operations.append(
            {
                "operator_id": "gaussian-noise",
                "sigma": noise_sigma,
                "channel_mode": "achromatic-equal",
                "generator": "PCG64",
            }
        )
        quality = int(condition["jpeg"]["quality"])
        output, encoded = _encode_jpeg_rgb(working, quality)
        operations.append(
            {
                "operator_id": "jpeg",
                "quality": quality,
                "sampling_factor": "4:4:4",
            }
        )
    else:
        output = np.clip(np.rint(working), 0, 255).astype(np.uint8)
    trace_payload = {
        "schema_version": 1,
        "record_type": "staff-scale-degradation-trace",
        "control_id": control.control_id,
        "control_sha256": control.sha256,
        "item_id": item_id,
        "source_group_id": source_group_id,
        "condition_id": condition_id,
        "scale": scale,
        "staff_spacing_px": staff_spacing_px,
        "derived_seed": derived_seed,
        "input_pixel_sha256": aligned.input_pixel_sha256,
        "aligned_pixel_sha256": aligned.aligned_pixel_sha256,
        "output_pixel_sha256": pixel_sha256(output),
        "operations": operations,
    }
    trace = dict(trace_payload)
    trace["trace_id"] = f"staff-scale-degradation-{canonical_sha256(trace_payload)}"
    return StaffScaleDegradationResult(pixels=output, encoded=encoded, trace=trace)
