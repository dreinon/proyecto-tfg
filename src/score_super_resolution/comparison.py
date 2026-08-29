"""Compact, outcome-blind helpers for the fixed SMB pretrained comparison."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import DegradationControl
from score_super_resolution.identities import canonical_sha256

SAMPLE_RELATIVE_PATH = Path("data/audits/smb-visual-sample-v1.csv")
SAMPLE_SHA256 = "993a9d03fceb733982ee2091de13b2991e57c511013e2f263041835b6e508970"
SAMPLE_SIZE = 64
CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
QUALITATIVE_ASSIGNMENT = (
    ("smb-test-000683", "x2-clean"),
    ("smb-test-000295", "x2-moderate"),
    ("smb-test-000513", "x2-strong"),
    ("smb-test-000519", "x4-clean"),
    ("smb-test-000304", "x4-moderate"),
    ("smb-test-000278", "x4-strong"),
)


class ComparisonContractError(ValueError):
    """A frozen sample, control, image, or metric pair is inconsistent."""


@dataclass(frozen=True)
class EvaluationSampleRow:
    upstream_index: int
    item_id: str
    source_group_id: str


def load_evaluation_sample(project_root: Path) -> tuple[EvaluationSampleRow, ...]:
    """Load the pre-review SHA-ranked sample and require 64 independent source groups."""

    path = Path(project_root).resolve() / SAMPLE_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ComparisonContractError("fixed SMB sample is unavailable")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SAMPLE_SHA256:
        raise ComparisonContractError("fixed SMB sample digest differs")
    try:
        rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    except (UnicodeError, csv.Error) as error:
        raise ComparisonContractError("fixed SMB sample cannot be parsed") from error
    expected_fields = {
        "upstream_index",
        "item_id",
        "source_group_id",
        "processing_status",
        "audit_sample_member",
    }
    if len(rows) != SAMPLE_SIZE or set(rows[0]) != expected_fields:
        raise ComparisonContractError("fixed SMB sample shape differs")
    if any(
        row["processing_status"] != "processed" or row["audit_sample_member"] != "True"
        for row in rows
    ):
        raise ComparisonContractError("fixed SMB sample contains an ineligible row")
    parsed = tuple(
        EvaluationSampleRow(
            upstream_index=int(row["upstream_index"]),
            item_id=row["item_id"],
            source_group_id=row["source_group_id"],
        )
        for row in rows
    )
    if (
        len({row.item_id for row in parsed}) != SAMPLE_SIZE
        or len({row.source_group_id for row in parsed}) != SAMPLE_SIZE
    ):
        raise ComparisonContractError("fixed SMB sample is not independent by source group")
    if not {item_id for item_id, _ in QUALITATIVE_ASSIGNMENT} <= {row.item_id for row in parsed}:
        raise ComparisonContractError("qualitative membership is outside the fixed sample")
    return parsed


def ensure_manifest_generation(project_root: Path) -> Path:
    """Restore the ignored active SMB generation from its tracked recovery bundle if needed."""

    project_root = Path(project_root).resolve()
    active_path = project_root / "data/manifests/smb-evaluation-v1.yaml"
    try:
        pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ComparisonContractError("active SMB manifest pointer cannot be read") from error
    if not isinstance(pointer, dict):
        raise ComparisonContractError("active SMB manifest pointer is malformed")
    generation_id = pointer.get("generation_id")
    if not isinstance(generation_id, str) or len(generation_id) != 64:
        raise ComparisonContractError("active SMB generation identity is malformed")
    generation_root = project_root / "artifacts/smb-manifests/generations"
    generation_path = generation_root / generation_id
    expected = (
        generation_path / "manifest-descriptor.yaml",
        generation_path / "manifest-records.jsonl",
    )
    if all(path.is_file() and not path.is_symlink() for path in expected):
        return generation_root

    def recovery_path(field: str) -> Path:
        relative = Path(str(pointer.get(field, "")))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ComparisonContractError("SMB recovery path is not project-relative")
        resolved = (project_root / relative).resolve()
        if not resolved.is_relative_to(project_root) or not resolved.is_file():
            raise ComparisonContractError("SMB recovery artifact is unavailable")
        return resolved

    from score_super_resolution.smb_audit import recover_active_manifest

    try:
        recovered = recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_path("recovery_descriptor_path"),
            recovery_records_path=recovery_path("recovery_records_path"),
            generation_root=generation_root,
        )
    except Exception as error:
        raise ComparisonContractError("active SMB generation recovery failed") from error
    if (
        recovered.get("generation_id") != generation_id
        or recovered.get("row_count") != pointer.get("row_count")
        or recovered.get("records_sha256") != pointer.get("records_sha256")
    ):
        raise ComparisonContractError("recovered SMB generation identity differs")
    return generation_root


def load_frozen_degradation(project_root: Path) -> DegradationControl:
    """Load the accepted frozen degradation control used by the SMB comparison."""

    path = Path(project_root).resolve() / "configs/degradations/controlled-score-v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_instance("degradation-control", payload, version=2)
    except (OSError, UnicodeError, yaml.YAMLError, ContractValidationError) as error:
        raise ComparisonContractError("frozen degradation control cannot be validated") from error
    if (
        payload.get("status") != "frozen"
        or payload.get("control_id") != "controlled-score-v1"
        or tuple(payload.get("condition_order", ())) != CONDITIONS
    ):
        raise ComparisonContractError("frozen degradation control identity differs")
    return DegradationControl(
        version=int(payload["version"]),
        candidate_id=str(payload["candidate_id"]),
        status=str(payload["status"]),
        claim_boundary=str(payload["claim_boundary"]),
        master_seed=int(payload["master_seed"]),
        image_contract=dict(payload["image_contract"]),
        alignment=dict(payload["alignment"]),
        runtime=dict(payload["runtime"]),
        condition_ids=tuple(payload["condition_order"]),
        conditions=tuple(dict(condition) for condition in payload["conditions"]),
        sha256=canonical_sha256(payload),
    )


def as_rgb8(image: Any) -> np.ndarray:
    """Normalize a Hugging Face/Pillow/NumPy image to owned contiguous RGB uint8 pixels."""

    if hasattr(image, "convert"):
        image = image.convert("RGB")
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
        raise ComparisonContractError("SMB image cannot be represented as RGB")
    pixels = pixels[..., :3]
    if pixels.dtype != np.uint8:
        if not np.issubdtype(pixels.dtype, np.integer):
            raise ComparisonContractError("SMB image dtype is not an integer RGB representation")
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(pixels)


def fidelity_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Compute the frozen full-page Y-primary and RGB-diagnostic PSNR/SSIM pair."""

    for label, pixels in (("reference", reference), ("reconstruction", reconstruction)):
        if (
            not isinstance(pixels, np.ndarray)
            or pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
        ):
            raise ComparisonContractError(f"{label} must be an RGB uint8 array")
    if reference.shape != reconstruction.shape or min(reference.shape[:2]) < 11:
        raise ComparisonContractError("metric pair must be aligned and at least 11 pixels wide")
    reference_float = reference.astype(np.float64)
    reconstruction_float = reconstruction.astype(np.float64)
    reference_y = rgb2ycbcr(reference_float / 255.0)[..., 0]
    reconstruction_y = rgb2ycbcr(reconstruction_float / 255.0)[..., 0]
    with np.errstate(divide="ignore"):
        psnr_y = float(peak_signal_noise_ratio(reference_y, reconstruction_y, data_range=255.0))
        psnr_rgb = float(
            peak_signal_noise_ratio(reference_float, reconstruction_float, data_range=255.0)
        )
    return {
        "psnr_y": psnr_y,
        "ssim_y": float(
            structural_similarity(
                reference_y,
                reconstruction_y,
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                win_size=11,
                use_sample_covariance=False,
                K1=0.01,
                K2=0.03,
            )
        ),
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": float(
            structural_similarity(
                reference_float,
                reconstruction_float,
                data_range=255.0,
                channel_axis=-1,
                gaussian_weights=True,
                sigma=1.5,
                win_size=11,
                use_sample_covariance=False,
                K1=0.01,
                K2=0.03,
            )
        ),
    }


def technical_smoke_image() -> np.ndarray:
    """Return a deterministic non-evidence image used only to validate model plumbing."""

    canvas = np.full((96, 128, 3), 248, dtype=np.uint8)
    for y in range(18, 79, 12):
        cv2.line(canvas, (8, y), (119, y), (30, 30, 30), 1, cv2.LINE_8)
    cv2.circle(canvas, (42, 36), 5, (20, 20, 20), -1, cv2.LINE_8)
    cv2.line(canvas, (47, 36), (47, 20), (20, 20, 20), 2, cv2.LINE_8)
    cv2.rectangle(canvas, (74, 48), (87, 58), (20, 20, 20), -1)
    return np.ascontiguousarray(canvas)
