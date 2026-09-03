"""Outcome-blind external-corpus pilot for the professional demonstrator."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from score_super_resolution.application import ProfessionalInferenceService
from score_super_resolution.baselines import pixel_sha256
from score_super_resolution.comparison import fidelity_metrics
from score_super_resolution.degradation import align_reference
from score_super_resolution.edsr_finetuning import (
    ADAPTED_METHOD_ID,
    BICUBIC_METHOD_ID,
    PRETRAINED_METHOD_ID,
    RESULT_FIELDS,
    analyze_adaptation_results,
    write_run_manifest,
)
from score_super_resolution.identities import canonical_sha256
from score_super_resolution.pretrained import CHECKPOINTS, PretrainedSRRunner
from score_super_resolution.staff_scale import (
    CONDITIONS,
    StaffScaleError,
    apply_scale_normalized_degradation,
    estimate_staff_spacing_full_page,
    load_scale_normalized_control,
)

SOURCE_ROOT_RELATIVE_PATH = Path("data/raw/professional-pilot-v1")
METADATA_FILENAME = "source-metadata.csv"
ARTIFACT_ROOT_RELATIVE_PATH = Path("artifacts/professional-pilot-v1")
SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MAXIMUM_SOURCE_BYTES = 100_000_000
MAXIMUM_REFERENCE_PIXELS = 32_000_000
EXPECTED_TEST_WORKS = 12
MINIMUM_ENGINEERING_WORKS = 3
MAXIMUM_ENGINEERING_WORKS = 5
SELECTION_SEED = 20260903
BOOTSTRAP_REPETITIONS = 2_000
METADATA_FIELDS = (
    "role",
    "work_id",
    "file_name",
    "genre",
    "instrument",
    "orientation",
    "source_type",
    "rights_basis",
    "source_reference",
    "notation_density",
    "document_condition",
    "text_present",
    "notes",
)


class ExternalPilotError(ValueError):
    """External source, selection, execution, or evidence is not safe to report."""


@dataclass(frozen=True)
class ExternalPilotPage:
    """One input-only selected work with enough identity to reproduce the pilot."""

    role: str
    work_id: str
    item_id: str
    path: Path
    file_name: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    staff_spacing_px: float
    staff_estimator_id: str
    staff_sequence_count: int
    genre: str
    instrument: str
    orientation: str
    source_type: str
    rights_basis: str
    source_reference: str
    notation_density: str
    document_condition: str
    text_present: bool
    notes: str

    def evidence_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "work_id": self.work_id,
            "item_id": self.item_id,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "pixel_sha256": self.pixel_sha256,
            "width": self.width,
            "height": self.height,
            "staff_spacing_px": self.staff_spacing_px,
            "staff_estimator_id": self.staff_estimator_id,
            "staff_sequence_count": self.staff_sequence_count,
            "genre": self.genre,
            "instrument": self.instrument,
            "orientation": self.orientation,
            "source_type": self.source_type,
            "rights_basis": self.rights_basis,
            "source_reference": self.source_reference,
            "notation_density": self.notation_density,
            "document_condition": self.document_condition,
            "text_present": self.text_present,
            "notes": self.notes,
        }


def _read_source_pixels(path: Path) -> np.ndarray:
    try:
        if path.is_symlink() or not path.is_file():
            raise ExternalPilotError("an external source must be a regular non-symlink file")
        metadata = path.stat()
        if metadata.st_size < 1 or metadata.st_size > MAXIMUM_SOURCE_BYTES:
            raise ExternalPilotError("an external source exceeds the file-size contract")
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ExternalPilotError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ExternalPilotError("an external source is not a readable image") from error
    pixels = np.ascontiguousarray(pixels)
    if pixels.shape[0] * pixels.shape[1] > MAXIMUM_REFERENCE_PIXELS:
        raise ExternalPilotError("an external reference exceeds the pixel safety bound")
    return pixels


def _load_metadata(path: Path) -> list[dict[str, str]]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ExternalPilotError("source-metadata.csv is missing")
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except ExternalPilotError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise ExternalPilotError("external source metadata cannot be read") from error
    if not rows or tuple(rows[0]) != METADATA_FIELDS:
        raise ExternalPilotError("external source metadata has an unexpected schema")
    return rows


def load_external_pilot_pages(
    project_root: Path,
    *,
    source_root: Path | None = None,
) -> tuple[ExternalPilotPage, ...]:
    """Validate the local rights metadata and freeze input-only page evidence in memory."""

    project_root = Path(project_root).resolve()
    source_root = (
        Path(source_root).resolve()
        if source_root is not None
        else project_root / SOURCE_ROOT_RELATIVE_PATH
    )
    rows = _load_metadata(source_root / METADATA_FILENAME)
    role_counts = {
        role: sum(row["role"] == role for row in rows) for role in ("engineering", "test")
    }
    if role_counts["test"] != EXPECTED_TEST_WORKS or not (
        MINIMUM_ENGINEERING_WORKS <= role_counts["engineering"] <= MAXIMUM_ENGINEERING_WORKS
    ):
        raise ExternalPilotError(
            "the pilot requires 3-5 engineering works and exactly 12 test works"
        )
    if any(row["role"] not in {"engineering", "test"} for row in rows):
        raise ExternalPilotError("external source role must be engineering or test")
    if len({row["work_id"] for row in rows}) != len(rows) or len(
        {row["file_name"] for row in rows}
    ) != len(rows):
        raise ExternalPilotError("external work and file identities must be unique")

    pages: list[ExternalPilotPage] = []
    for row in rows:
        if (
            not row["work_id"]
            or not row["work_id"].replace("-", "").replace("_", "").isalnum()
            or Path(row["file_name"]).name != row["file_name"]
            or Path(row["file_name"]).suffix.casefold() not in SUPPORTED_SUFFIXES
        ):
            raise ExternalPilotError("external work or file identity is unsafe")
        if row["source_type"] not in {"scan", "born-digital"}:
            raise ExternalPilotError("source_type must be scan or born-digital")
        if not row["genre"] or not row["instrument"]:
            raise ExternalPilotError("genre and instrument are required input-only strata")
        if row["orientation"] not in {"horizontal", "vertical"}:
            raise ExternalPilotError("orientation must be horizontal or vertical")
        if row["notation_density"] not in {"sparse", "medium", "dense"}:
            raise ExternalPilotError("notation_density must be sparse, medium, or dense")
        if row["document_condition"] not in {"clean", "aged", "mixed"}:
            raise ExternalPilotError("document_condition must be clean, aged, or mixed")
        if row["text_present"] not in {"yes", "no"}:
            raise ExternalPilotError("text_present must be yes or no")
        if not row["rights_basis"] or not row["source_reference"]:
            raise ExternalPilotError("rights basis and source reference are required")
        path = source_root / row["file_name"]
        try:
            pixels = _read_source_pixels(path)
        except ExternalPilotError as error:
            raise ExternalPilotError(
                f"source image for {row['work_id']} cannot be loaded: {error}"
            ) from error
        try:
            spacing = estimate_staff_spacing_full_page(pixels)
        except StaffScaleError as error:
            raise ExternalPilotError(
                f"staff scale cannot be measured for {row['work_id']}"
            ) from error
        if not 4.0 <= spacing.spacing_px <= 32.0:
            raise ExternalPilotError(
                f"staff scale for {row['work_id']} is outside the frozen degradation range"
            )
        pages.append(
            ExternalPilotPage(
                role=row["role"],
                work_id=row["work_id"],
                item_id=f"external-{row['work_id']}",
                path=path,
                file_name=row["file_name"],
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                pixel_sha256=pixel_sha256(pixels),
                width=pixels.shape[1],
                height=pixels.shape[0],
                staff_spacing_px=spacing.spacing_px,
                staff_estimator_id=spacing.estimator_id,
                staff_sequence_count=spacing.sequence_count,
                genre=row["genre"],
                instrument=row["instrument"],
                orientation=row["orientation"],
                source_type=row["source_type"],
                rights_basis=row["rights_basis"],
                source_reference=row["source_reference"],
                notation_density=row["notation_density"],
                document_condition=row["document_condition"],
                text_present=row["text_present"] == "yes",
                notes=row["notes"],
            )
        )
    return tuple(sorted(pages, key=lambda page: (page.role, page.work_id)))


def freeze_external_pilot_manifest(
    pages: tuple[ExternalPilotPage, ...], output_root: Path
) -> tuple[Path, str]:
    """Write the outcome-blind source manifest before any model output is opened."""

    records = [page.evidence_record() for page in pages]
    manifest_sha256 = canonical_sha256(records)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "input-manifest.csv"
    frame = pd.DataFrame(records)
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    if path.is_file():
        if path.read_bytes() == temporary.read_bytes():
            temporary.unlink()
            return path, manifest_sha256
        if (output_root / "evaluation-identity.json").exists() or (
            output_root / "raw-metrics.csv"
        ).exists():
            temporary.unlink()
            raise ExternalPilotError("external inputs changed after evaluation began")
    os.replace(temporary, path)
    return path, manifest_sha256


def qualitative_assignment(pages: tuple[ExternalPilotPage, ...]) -> dict[str, str]:
    """Assign each of twelve test works once, balancing two works per condition."""

    test_pages = [page for page in pages if page.role == "test"]
    if len(test_pages) != EXPECTED_TEST_WORKS:
        raise ExternalPilotError("qualitative assignment requires the exact test population")
    ranked = sorted(
        test_pages,
        key=lambda page: hashlib.sha256(f"{SELECTION_SEED}|{page.work_id}".encode()).hexdigest(),
    )
    ordered_conditions = tuple(CONDITIONS) * 2
    return {
        page.work_id: condition for page, condition in zip(ranked, ordered_conditions, strict=True)
    }


def _save_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_revision(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExternalPilotError("the project Git revision cannot be recorded") from error
    revision = result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ExternalPilotError("the project Git revision is malformed")
    return revision


def _git_dirty(project_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExternalPilotError("the project Git dirty state cannot be recorded") from error
    return bool(result.stdout.strip())


def _source_identity(project_root: Path) -> dict[str, object]:
    relative_paths = (
        "src/score_super_resolution/application.py",
        "src/score_super_resolution/degradation.py",
        "src/score_super_resolution/edsr_finetuning.py",
        "src/score_super_resolution/external_pilot.py",
        "src/score_super_resolution/pretrained.py",
        "src/score_super_resolution/staff_scale.py",
    )
    files: dict[str, str] = {}
    for relative_path in relative_paths:
        path = project_root / relative_path
        try:
            files[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ExternalPilotError("the external-pilot source identity cannot be read") from error
    return {"files": files, "sha256": canonical_sha256(files)}


def _runtime_evidence(device: torch.device) -> dict[str, object]:
    packages = {}
    for name in ("numpy", "opencv-python-headless", "pandas", "Pillow", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "device": str(device),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "packages": packages,
    }


def evaluate_external_pilot(
    project_root: Path,
    pages: tuple[ExternalPilotPage, ...],
    *,
    output_root: Path,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Run the exact three-method, six-condition external test with resumable evidence."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_pages = tuple(page for page in pages if page.role == "test")
    if len(test_pages) != EXPECTED_TEST_WORKS:
        raise ExternalPilotError("external evaluation requires exactly twelve test works")
    control = load_scale_normalized_control(project_root)
    pretrained = PretrainedSRRunner(project_root, device=selected_device)
    adapted = ProfessionalInferenceService(project_root, device=selected_device)
    assignment = qualitative_assignment(pages)
    results_path = output_root / "raw-metrics.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "experiment_id": "professional-pilot-v1",
        "git_revision": _git_revision(project_root),
        "git_dirty": _git_dirty(project_root),
        "source_identity": _source_identity(project_root),
        "input_manifest_sha256": canonical_sha256([page.evidence_record() for page in pages]),
        "degradation_control_id": control.control_id,
        "degradation_control_sha256": control.sha256,
        "conditions": list(CONDITIONS),
        "methods": [BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID],
        "official_checkpoint_sha256": {
            str(scale): CHECKPOINTS[(PRETRAINED_METHOD_ID, scale)].sha256 for scale in (2, 4)
        },
        "adapted_checkpoint_sha256": {
            str(scale): adapted.model_identity(scale) for scale in (2, 4)
        },
        "selection_seed": SELECTION_SEED,
        "qualitative_assignment": assignment,
        "device": str(selected_device),
    }
    identity_path = output_root / "evaluation-identity.json"
    if identity_path.is_file():
        try:
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ExternalPilotError("partial external identity cannot be read") from error
        if existing_identity != identity:
            raise ExternalPilotError("partial external metrics belong to another frozen run")
    else:
        _atomic_write_json(identity_path, identity)
    _atomic_write_json(output_root / "runtime-evidence.json", _runtime_evidence(selected_device))
    if results_path.is_file():
        results = pd.read_csv(results_path)
        if tuple(results.columns) != RESULT_FIELDS:
            raise ExternalPilotError("partial external metrics have an unexpected schema")
        records = results.to_dict("records")
    else:
        records = []
    completed = {
        (str(row["item_id"]), str(row["condition_id"]), str(row["method_id"])) for row in records
    }
    recorded_by_key = {
        (str(row["item_id"]), str(row["condition_id"]), str(row["method_id"])): row
        for row in records
    }
    qualitative_rows: list[dict[str, str]] = []

    for page_index, page in enumerate(test_pages):
        reference_original = _read_source_pixels(page.path)
        if pixel_sha256(reference_original) != page.pixel_sha256:
            raise ExternalPilotError("external source pixels changed after manifest freeze")
        for condition_id in CONDITIONS:
            scale = int(condition_id[1])
            degraded = apply_scale_normalized_degradation(
                reference_original,
                control=control,
                condition_id=condition_id,
                item_id=page.item_id,
                source_group_id=page.work_id,
                staff_spacing_px=page.staff_spacing_px,
            )
            reference = align_reference(reference_original, scale).pixels
            outputs: dict[str, np.ndarray] = {}
            retain_qualitative = assignment[page.work_id] == condition_id
            for method_id in (BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID):
                key = (page.item_id, condition_id, method_id)
                if key in completed and not retain_qualitative:
                    continue
                if method_id == ADAPTED_METHOD_ID:
                    result = adapted.enhance(degraded.pixels, scale=scale)
                    output = result.pixels
                    runtime = result.elapsed_seconds
                    checkpoint_digest: str | float = result.checkpoint_sha256
                else:
                    result = pretrained.run(
                        method_id,
                        degraded.pixels,
                        target_shape=reference.shape,
                        condition_id=condition_id,
                    )
                    output = result.pixels
                    runtime = result.elapsed_ns / 1e9
                    checkpoint_digest = result.evidence.get("checkpoint_sha256", math.nan)
                outputs[method_id] = output
                output_digest = pixel_sha256(output)
                if key in completed:
                    if str(recorded_by_key[key]["output_sha256"]) != output_digest:
                        raise ExternalPilotError("replayed external output differs from its record")
                    continue
                records.append(
                    {
                        "upstream_index": page_index,
                        "item_id": page.item_id,
                        "source_group_id": page.work_id,
                        "condition_id": condition_id,
                        "scale": scale,
                        "profile": condition_id.split("-", 1)[1],
                        "method_id": method_id,
                        **fidelity_metrics(reference, output),
                        "runtime_seconds": runtime,
                        "output_sha256": output_digest,
                        "checkpoint_sha256": checkpoint_digest,
                    }
                )
                completed.add(key)
                recorded_by_key[key] = records[-1]
            frame = pd.DataFrame(records, columns=RESULT_FIELDS)
            temporary = results_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, results_path)

            if retain_qualitative:
                case_root = output_root / "qualitative" / page.work_id / condition_id
                display_lr = cv2.resize(
                    degraded.pixels,
                    (reference.shape[1], reference.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                images = {
                    "reference-hr": reference,
                    "input-lr-nearest": display_lr,
                    **outputs,
                }
                for role, pixels in images.items():
                    path = case_root / f"{role}.png"
                    _save_image(path, pixels)
                    qualitative_rows.append(
                        {
                            "item_id": page.item_id,
                            "source_group_id": page.work_id,
                            "condition_id": condition_id,
                            "image_role": role,
                            "path": path.relative_to(output_root).as_posix(),
                            "pixel_sha256": pixel_sha256(pixels),
                        }
                    )

    results = pd.DataFrame(records, columns=RESULT_FIELDS).sort_values(
        ["item_id", "condition_id", "method_id"]
    )
    expected = EXPECTED_TEST_WORKS * len(CONDITIONS) * 3
    if (
        len(results) != expected
        or len(results.drop_duplicates(["item_id", "condition_id", "method_id"])) != expected
    ):
        raise ExternalPilotError("external evaluation did not reconcile all 216 outputs")
    results.to_csv(results_path, index=False)
    if qualitative_rows:
        pd.DataFrame(qualitative_rows).sort_values(
            ["condition_id", "source_group_id", "image_role"]
        ).to_csv(output_root / "qualitative-index.csv", index=False)
    return results.reset_index(drop=True)


def analyze_external_pilot(
    results: pd.DataFrame, output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute aggregate and source-bootstrap evidence with the frozen controls."""

    aggregate, paired = analyze_adaptation_results(
        results,
        seed=SELECTION_SEED,
        repetitions=BOOTSTRAP_REPETITIONS,
    )
    output_root = Path(output_root).resolve()
    aggregate.to_csv(output_root / "aggregate-metrics.csv", index=False)
    paired.to_csv(output_root / "paired-bootstrap.csv", index=False)
    identity_path = output_root / "evaluation-identity.json"
    if not identity_path.is_file():
        raise ExternalPilotError("external evaluation identity is missing")
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExternalPilotError("external evaluation identity cannot be read") from error
    write_run_manifest(
        output_root,
        {
            "experiment_id": "professional-pilot-v1",
            "input_manifest_sha256": identity["input_manifest_sha256"],
            "git_revision": identity["git_revision"],
            "git_dirty": identity["git_dirty"],
            "source_identity_sha256": identity["source_identity"]["sha256"],
            "rows": len(results),
            "independent_test_works": int(results["source_group_id"].nunique()),
            "selection_seed": SELECTION_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        },
    )
    return aggregate, paired
