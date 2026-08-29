"""Masked, content-addressed review of the fixed Phase 2 fixture evidence."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import ipywidgets as widgets
import nbformat
import numpy as np
from IPython.display import clear_output, display
from nbclient import NotebookClient

from score_super_resolution.baselines import pixel_sha256
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.evaluation import NOTATION_TAXONOMY, validate_notation_review
from score_super_resolution.identities import canonical_sha256

EXPECTED_CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
EXPECTED_METHODS = (
    "nearest-opencv-exact-v1",
    "bilinear-opencv-exact-v1",
    "bicubic-opencv-v1",
)
MASK_LABELS = ("A", "B", "C")
WORKING_COPY_MARKER = "__GENERATED_PHASE2_FIXTURE_REVIEW_WORKING_COPY__"
_GENERATED_TOKEN = re.compile(r"generated:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_PIXELS = 4_194_304


class FixtureReviewContractError(ValueError):
    """Fixed evidence, notebook, panel, or review state violates the review contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_regular(path: Path, *, maximum_bytes: int, kind: str) -> bytes:
    path = Path(path)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FixtureReviewContractError(f"{kind} must be a regular non-symlink file")
        if metadata.st_size > maximum_bytes:
            raise FixtureReviewContractError(f"{kind} exceeds its byte bound")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - consumed))
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > maximum_bytes:
                    raise FixtureReviewContractError(f"{kind} exceeds its byte bound")
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except FixtureReviewContractError:
        raise
    except OSError as error:
        raise FixtureReviewContractError(f"{kind} cannot be read safely") from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    if identity(before) != identity(after):
        raise FixtureReviewContractError(f"{kind} changed while being read")
    return b"".join(chunks)


def _read_json(path: Path, *, kind: str, maximum_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular(path, maximum_bytes=maximum_bytes, kind=kind))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise FixtureReviewContractError(f"{kind} is malformed") from error
    if not isinstance(value, dict):
        raise FixtureReviewContractError(f"{kind} root must be an object")
    return value


def _safe_relative(root: Path, relative: str, *, kind: str) -> Path:
    if not isinstance(relative, str) or not relative or relative.startswith(("/", "\\")):
        raise FixtureReviewContractError(f"{kind} path must be canonical and relative")
    parts = Path(relative).parts
    if any(part in {"", ".", ".."} for part in parts) or "\\" in relative:
        raise FixtureReviewContractError(f"{kind} path contains traversal")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise FixtureReviewContractError(f"{kind} path escapes its root")
    current = root.resolve()
    for part in parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise FixtureReviewContractError(f"{kind} path contains a symlink")
    return path


def _self_digest(record: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: value for key, value in record.items() if key != field})


def _require_self_digest(record: Mapping[str, Any], field: str, *, kind: str) -> None:
    if record.get(field) != _self_digest(record, field):
        raise FixtureReviewContractError(f"{kind} content digest differs")


def _decode_rgb(
    path: Path,
    *,
    encoded_sha256: str | None,
    pixel_digest: str | None,
    kind: str,
) -> np.ndarray:
    raw = _read_regular(path, maximum_bytes=_MAX_IMAGE_BYTES, kind=kind)
    if encoded_sha256 is not None and hashlib.sha256(raw).hexdigest() != encoded_sha256:
        raise FixtureReviewContractError(f"{kind} encoded digest differs")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise FixtureReviewContractError(f"{kind} cannot be decoded as RGB8")
    if int(decoded.shape[0]) * int(decoded.shape[1]) > _MAX_PIXELS:
        raise FixtureReviewContractError(f"{kind} exceeds its pixel bound")
    rgb = np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    if pixel_digest is not None and pixel_sha256(rgb) != pixel_digest:
        raise FixtureReviewContractError(f"{kind} pixel digest differs")
    return rgb


def _degradation_pixel_sha256(pixels: np.ndarray) -> str:
    height, width, channels = pixels.shape
    framed = (
        b"phase2-degradation-rgb8-v1\0"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + channels.to_bytes(1, "big")
        + pixels.tobytes(order="C")
    )
    return hashlib.sha256(framed).hexdigest()


def notebook_source_sha256(path: Path) -> str:
    """Return the normalized digest of a source-only notebook or fail closed."""

    notebook = _read_json(Path(path), kind="tracked notebook", maximum_bytes=4 * 1024 * 1024)
    if (
        not isinstance(notebook.get("cells"), list)
        or not isinstance(notebook.get("metadata"), dict)
        or not isinstance(notebook.get("nbformat"), int)
        or isinstance(notebook.get("nbformat"), bool)
    ):
        raise FixtureReviewContractError("tracked notebook structure is invalid")
    serialized = json.dumps(notebook, sort_keys=True).casefold()
    if any(
        value in serialized for value in ("image/png", "image/jpeg", "application/pdf", "base64,")
    ):
        raise FixtureReviewContractError("tracked notebook contains an embedded image payload")
    marker_count = 0
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or not isinstance(cell.get("metadata"), dict):
            raise FixtureReviewContractError("tracked notebook cell structure is invalid")
        source = cell.get("source")
        if not isinstance(source, (str, list)) or (
            isinstance(source, list) and not all(isinstance(value, str) for value in source)
        ):
            raise FixtureReviewContractError("tracked notebook cell source is invalid")
        marker_count += (source if isinstance(source, str) else "".join(source)).count(
            WORKING_COPY_MARKER
        )
        if cell.get("cell_type") == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs") != []
        ):
            raise FixtureReviewContractError("tracked notebook must be unexecuted and output-free")
    if marker_count != 1:
        raise FixtureReviewContractError("tracked notebook working-copy guard is absent")
    return canonical_sha256(notebook)


def _logical_working_sha256(working_path: Path, source_path: Path, token: str) -> str:
    working = _read_json(working_path, kind="working notebook", maximum_bytes=16 * 1024 * 1024)
    source = _read_json(source_path, kind="tracked notebook", maximum_bytes=4 * 1024 * 1024)

    def projection(notebook: dict[str, Any], *, working_copy: bool) -> dict[str, Any]:
        value = copy.deepcopy(notebook)
        value.get("metadata", {}).pop("widgets", None)
        language = value.get("metadata", {}).get("language_info")
        if isinstance(language, dict):
            for field in (
                "codemirror_mode",
                "file_extension",
                "mimetype",
                "nbconvert_exporter",
                "pygments_lexer",
                "version",
            ):
                language.pop(field, None)
        kernelspec = value.get("metadata", {}).get("kernelspec")
        if not isinstance(kernelspec, dict) or kernelspec.get("name") != "python3":
            raise FixtureReviewContractError("working notebook kernelspec differs")
        if kernelspec.get("language") != "python" or kernelspec.get("display_name") not in {
            "Python 3 (score-super-resolution)",
            "score-super-resolution (3.12.12)",
        }:
            raise FixtureReviewContractError("working notebook kernel identity differs")
        kernelspec["display_name"] = "score-super-resolution (3.12.12)"
        replacements = 0
        for cell in value["cells"]:
            cell["metadata"].pop("execution", None)
            for field in ("collapsed", "scrolled", "trusted"):
                cell["metadata"].pop(field, None)
            cell_source = cell["source"]
            joined = cell_source if isinstance(cell_source, str) else "".join(cell_source)
            expected = token if working_copy else WORKING_COPY_MARKER
            replacements += joined.count(expected)
            joined = joined.replace(expected, WORKING_COPY_MARKER)
            cell["source"] = (
                joined if isinstance(cell_source, str) else joined.splitlines(keepends=True)
            )
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        if replacements != 1:
            raise FixtureReviewContractError("working notebook token differs")
        return value

    working_projection = projection(working, working_copy=True)
    source_projection = projection(source, working_copy=False)
    if working_projection != source_projection:
        raise FixtureReviewContractError("working notebook differs from its tracked source")
    return canonical_sha256(
        {"domain": "phase2-fixture-review-logical-notebook-v1", "notebook": working_projection}
    )


def _validate_export_binding(
    artifact_root: Path,
    export: Mapping[str, Any],
    relative: str,
    record: Mapping[str, Any],
) -> None:
    evidence = {item["relative_path"]: item for item in export["evidence"]}
    if relative not in evidence:
        raise FixtureReviewContractError("portable export omits required review evidence")
    path = _safe_relative(artifact_root, relative, kind="portable evidence")
    raw = _read_regular(path, maximum_bytes=_MAX_JSON_BYTES, kind="portable evidence")
    if hashlib.sha256(raw).hexdigest() != evidence[relative]["sha256"]:
        raise FixtureReviewContractError("portable evidence byte digest differs")
    if relative == "reconciliation-report.json":
        expected_count = int(record["expected_tuple_count"])
    elif relative == "evidence/aggregate-six-cell.json":
        expected_count = 6
    else:
        expected_count = len(record["core_panels"]) + len(record["additional_panels"])
    if evidence[relative]["record_count"] != expected_count:
        raise FixtureReviewContractError("portable evidence record count differs")


def validate_fixture_review_inputs(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Validate fixed Plan 03 evidence without running, repairing, exporting, or selecting."""

    project_root = Path(project_root).resolve()
    root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else project_root / "artifacts/phase2-fixture"
    )
    if not root.is_dir() or root.is_symlink():
        raise FixtureReviewContractError("fixture artifact root is unavailable or unsafe")
    paths = {
        "core": root / "pre-run/qualitative-core-membership.json",
        "reconciliation": root / "reconciliation-report.json",
        "replay": root / "replay-report.json",
        "aggregate": root / "evidence/aggregate-six-cell.json",
        "membership": root / "evidence/qualitative-membership.json",
        "export": root / "export/portable-export-manifest.json",
    }
    records = {key: _read_json(path, kind=key) for key, path in paths.items()}
    core = records["core"]
    reconciliation = records["reconciliation"]
    replay = records["replay"]
    aggregate = records["aggregate"]
    membership = records["membership"]
    export = records["export"]
    try:
        validate_instance("qualitative-sample", core, version=2)
        validate_instance("reconciliation-report", reconciliation, version=2)
        validate_instance("replay-report", replay, version=2)
        validate_instance("aggregate-result", aggregate, version=2)
        validate_instance("qualitative-sample", membership, version=2)
        validate_instance("portable-export", export, version=2)
    except ContractValidationError as error:
        raise FixtureReviewContractError("fixed fixture evidence fails its schema") from error
    _require_self_digest(reconciliation, "report_sha256", kind="reconciliation report")
    _require_self_digest(replay, "report_sha256", kind="replay report")
    _require_self_digest(export, "manifest_sha256", kind="portable export")
    if core.get("core_sha256") != _self_digest(core, "core_sha256"):
        raise FixtureReviewContractError("pre-run core digest differs")
    if membership.get("final_membership_sha256") != _self_digest(
        membership, "final_membership_sha256"
    ):
        raise FixtureReviewContractError("final membership digest differs")
    if aggregate.get("aggregate_result_id") != (
        "aggregate-" + _self_digest(aggregate, "aggregate_result_id")
    ):
        raise FixtureReviewContractError("aggregate content identity differs")
    if replay.get("status") != "equivalent" or (
        replay.get("primary_scientific_projection_sha256")
        != replay.get("replay_scientific_projection_sha256")
        or replay.get("primary_output_pixels_sha256") != replay.get("replay_output_pixels_sha256")
    ):
        raise FixtureReviewContractError("fixed replay does not prove scientific equivalence")
    if (
        reconciliation.get("expected_tuple_count") != 144
        or reconciliation.get("terminal_tuple_count") != 144
    ):
        raise FixtureReviewContractError("reconciliation denominator differs")
    if reconciliation.get("counts") != {"succeeded": 144, "failed": 0, "excluded": 0}:
        raise FixtureReviewContractError("reconciliation terminal counts differ")
    if replay.get("expected_tuple_count") != 144:
        raise FixtureReviewContractError("replay denominator differs")
    if export.get("dataset_role") != "pipeline-validation-only":
        raise FixtureReviewContractError("portable export crosses the fixture-only boundary")
    if (
        export.get("experiment_id") != reconciliation.get("experiment_id")
        or aggregate.get("experiment_id") != reconciliation.get("experiment_id")
        or membership.get("experiment_id") != reconciliation.get("experiment_id")
        or replay.get("experiment_id") != reconciliation.get("experiment_id")
    ):
        raise FixtureReviewContractError("fixture experiment identity differs across evidence")
    if (
        export.get("reconciliation_id") != reconciliation.get("reconciliation_id")
        or export.get("reconciliation_sha256") != reconciliation.get("report_sha256")
        or aggregate.get("reconciliation_id") != reconciliation.get("reconciliation_id")
        or aggregate.get("reconciliation_sha256") != reconciliation.get("report_sha256")
        or replay.get("primary_reconciliation_id") != reconciliation.get("reconciliation_id")
        or replay.get("primary_reconciliation_sha256") != reconciliation.get("report_sha256")
    ):
        raise FixtureReviewContractError("reconciliation lineage differs across evidence")
    if [cell.get("condition_id") for cell in aggregate.get("cells", [])] != list(
        EXPECTED_CONDITIONS
    ) or len(aggregate.get("cells", [])) != 6:
        raise FixtureReviewContractError("aggregate is not the exact six-cell result")
    if (
        membership.get("membership_stage") != "final"
        or core.get("membership_stage") != "pre-run-core"
        or membership.get("core_membership_id") != core.get("core_membership_id")
        or membership.get("core_sha256") != core.get("core_sha256")
        or membership.get("core_panels") != core.get("core_panels")
        or len(core.get("core_panels", [])) != 12
        or len(membership.get("additional_panels", [])) > 12
    ):
        raise FixtureReviewContractError(
            "final membership does not preserve the exact pre-run core"
        )
    panels = [*membership["core_panels"], *membership["additional_panels"]]
    panel_ids = [panel["panel_id"] for panel in panels]
    if len(panel_ids) != len(set(panel_ids)) or len(panels) > 24:
        raise FixtureReviewContractError("final panel membership is duplicate or over-bound")
    denominators = membership.get("denominators")
    if denominators != {
        "requested_panels": len(panels),
        "displayable_panels": len(panels),
        "reviewed_panels": 0,
        "skipped_panels": 0,
        "failed_panels": 0,
    }:
        raise FixtureReviewContractError("final membership denominators differ")
    if any(tuple(panel.get("method_ids", ())) != EXPECTED_METHODS for panel in panels):
        raise FixtureReviewContractError("panel method membership differs")
    _validate_export_binding(root, export, "reconciliation-report.json", reconciliation)
    _validate_export_binding(root, export, "evidence/aggregate-six-cell.json", aggregate)
    _validate_export_binding(root, export, "evidence/qualitative-membership.json", membership)
    controls = export.get("controls", {})
    if (
        controls.get("core_membership_id") != core.get("core_membership_id")
        or controls.get("core_sha256") != core.get("core_sha256")
        or controls.get("degradation_sha256") != reconciliation.get("degradation_control_sha256")
        or controls.get("evaluation_sha256") != reconciliation.get("evaluation_control_sha256")
        or controls.get("fixture_manifest_sha256") != reconciliation.get("fixture_manifest_sha256")
        or aggregate.get("degradation_control_sha256") != controls.get("degradation_sha256")
        or aggregate.get("evaluation_control_sha256") != controls.get("evaluation_sha256")
        or membership.get("evaluation_control_sha256") != controls.get("evaluation_sha256")
        or membership.get("fixture_manifest_sha256") != controls.get("fixture_manifest_sha256")
    ):
        raise FixtureReviewContractError("fixed control or input digest differs")
    return {
        "artifact_root": root,
        "experiment_id": reconciliation["experiment_id"],
        "reconciliation_id": reconciliation["reconciliation_id"],
        "reconciliation_sha256": reconciliation["report_sha256"],
        "replay_id": replay["replay_id"],
        "replay_sha256": replay["report_sha256"],
        "aggregate_id": aggregate["aggregate_result_id"],
        "aggregate_file_sha256": hashlib.sha256(paths["aggregate"].read_bytes()).hexdigest(),
        "membership_id": membership["final_membership_id"],
        "membership_sha256": membership["final_membership_sha256"],
        "membership_file_sha256": hashlib.sha256(paths["membership"].read_bytes()).hexdigest(),
        "core_membership_id": core["core_membership_id"],
        "core_sha256": core["core_sha256"],
        "degradation_control_sha256": controls["degradation_sha256"],
        "evaluation_control_sha256": controls["evaluation_sha256"],
        "fixture_manifest_sha256": controls["fixture_manifest_sha256"],
        "portable_export_id": export["portable_export_id"],
        "portable_export_sha256": export["manifest_sha256"],
        "condition_ids": list(EXPECTED_CONDITIONS),
        "core_panel_count": 12,
        "additional_panel_count": len(membership["additional_panels"]),
        "requested_panel_count": len(panels),
        "panels": copy.deepcopy(panels),
    }


def _method_mapping(panel_id: str, membership_sha256: str) -> dict[str, str]:
    ranked = sorted(
        EXPECTED_METHODS,
        key=lambda method: canonical_sha256(
            {
                "domain": "phase2-fixture-masked-method-v1",
                "membership_sha256": membership_sha256,
                "panel_id": panel_id,
                "method_id": method,
            }
        ),
    )
    return dict(zip(MASK_LABELS, ranked, strict=True))


def _validated_panel_images(
    artifact_root: Path, panel: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    condition_id = str(panel["condition_id"])
    item_id = str(panel["item_id"])
    records: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray] = {}
    for method_id in EXPECTED_METHODS:
        record_path = artifact_root / f"scientific/{condition_id}/{method_id}/{item_id}.json"
        record = _read_json(record_path, kind="scientific panel record")
        try:
            validate_instance("scientific-result", record, version=2)
        except ContractValidationError as error:
            raise FixtureReviewContractError("scientific panel record fails its schema") from error
        scientific_digest = canonical_sha256(
            {
                key: value
                for key, value in record.items()
                if key not in {"scientific_result_id", "scientific_sha256"}
            }
        )
        if (
            record.get("condition_id") != condition_id
            or record.get("item_id") != item_id
            or record.get("source_group_id") != panel["source_group_id"]
            or record.get("method_id") != method_id
            or record.get("scientific_sha256") != scientific_digest
            or record.get("scientific_result_id") != "scientific-" + record["scientific_sha256"]
        ):
            raise FixtureReviewContractError("scientific panel identity differs")
        expected_relative = f"outputs/{condition_id}/{method_id}/{item_id}.png"
        if record.get("output_relative_path") != expected_relative:
            raise FixtureReviewContractError("scientific panel output path differs")
        outputs[method_id] = _decode_rgb(
            _safe_relative(artifact_root, expected_relative, kind="method output"),
            encoded_sha256=record["output_encoded_sha256"],
            pixel_digest=record["output_pixel_sha256"],
            kind="method output",
        )
        records[method_id] = record
    traces = [record["degradation_trace"] for record in records.values()]
    if any(trace != traces[0] for trace in traces[1:]):
        raise FixtureReviewContractError("panel methods do not share one degradation trace")
    trace = traces[0]
    reference_path = artifact_root / f"fixture-input/images/{item_id}.png"
    reference = _decode_rgb(
        reference_path,
        encoded_sha256=None,
        pixel_digest=None,
        kind="fixture reference",
    )
    if _degradation_pixel_sha256(reference) != trace["input_pixel_sha256"]:
        raise FixtureReviewContractError("fixture reference pixel digest differs")
    crop = trace["crop"]
    if crop.get("top") != 0 or crop.get("left") != 0:
        raise FixtureReviewContractError("fixture alignment crop is not lower/right only")
    bottom = int(crop["bottom"])
    right = int(crop["right"])
    aligned = np.ascontiguousarray(
        reference[
            : reference.shape[0] - bottom if bottom else None,
            : reference.shape[1] - right if right else None,
        ]
    )
    if _degradation_pixel_sha256(aligned) != trace["aligned_pixel_sha256"]:
        raise FixtureReviewContractError("aligned fixture reference digest differs")
    for record in records.values():
        if any(
            metric["reference_pixel_sha256"] != pixel_sha256(aligned)
            for metric in record["metrics"]
        ):
            raise FixtureReviewContractError("panel metric reference digest differs")
    if any(image.shape != aligned.shape for image in outputs.values()):
        raise FixtureReviewContractError("panel method geometry differs from aligned reference")
    scale = int(condition_id[1])
    lr = np.ascontiguousarray(outputs["nearest-opencv-exact-v1"][::scale, ::scale])
    if (
        list(lr.shape)
        != [
            trace["output_dimensions"]["height"],
            trace["output_dimensions"]["width"],
            trace["output_dimensions"]["channels"],
        ]
        or _degradation_pixel_sha256(lr) != trace["output_pixel_sha256"]
    ):
        raise FixtureReviewContractError("native LR reconstruction differs from fixed evidence")
    roi = panel["roi"]
    if roi["x"] + roi["width"] > aligned.shape[1] or roi["y"] + roi["height"] > aligned.shape[0]:
        raise FixtureReviewContractError("panel ROI escapes the aligned reference")
    return aligned, lr, outputs


def _fit(image: np.ndarray, width: int, height: int, *, allow_upscale: bool = True) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    if not allow_upscale:
        scale = min(scale, 1.0)
    target = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1 else cv2.INTER_NEAREST
    resized = cv2.resize(image, target, interpolation=interpolation)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _render_panel(
    panel: Mapping[str, Any],
    mapping: Mapping[str, str],
    reference: np.ndarray,
    lr: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> bytes:
    columns = [
        ("HR", reference),
        ("LR", lr),
        *[(label, outputs[method]) for label, method in mapping.items()],
    ]
    column_width = 300
    context_height = 220
    roi_height = 190
    label_height = 34
    gutter = 12
    header_height = 52
    canvas_width = len(columns) * column_width + (len(columns) + 1) * gutter
    canvas_height = (
        header_height + label_height + context_height + label_height + roi_height + 3 * gutter
    )
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{panel['condition_id']} | {panel['item_id']} | masked methods",
        (gutter, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    roi = panel["roi"]
    for index, (label, image) in enumerate(columns):
        left = gutter + index * (column_width + gutter)
        cv2.putText(
            canvas,
            label,
            (left, header_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        context_top = header_height + label_height
        canvas[context_top : context_top + context_height, left : left + column_width] = _fit(
            image,
            column_width,
            context_height,
            allow_upscale=label != "LR",
        )
        cv2.putText(
            canvas,
            "corresponding native ROI" if label == "LR" else "fixed ROI",
            (left, context_top + context_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        if label == "LR":
            scale = int(str(panel["condition_id"])[1])
            crop = image[
                roi["y"] // scale : (roi["y"] + roi["height"] + scale - 1) // scale,
                roi["x"] // scale : (roi["x"] + roi["width"] + scale - 1) // scale,
            ]
        else:
            crop = image[
                roi["y"] : roi["y"] + roi["height"],
                roi["x"] : roi["x"] + roi["width"],
            ]
        roi_top = context_top + context_height + label_height
        canvas[roi_top : roi_top + roi_height, left : left + column_width] = _fit(
            crop,
            column_width,
            roi_height,
            allow_upscale=label != "LR",
        )
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise FixtureReviewContractError("review panel PNG encoding failed")
    return bytes(encoded)


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short durable write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_identical(path: Path, payload: bytes, *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise FixtureReviewContractError(f"{kind} parent is unsafe")
    if path.exists() or path.is_symlink():
        existing = _read_regular(path, maximum_bytes=max(len(payload), 1) + 1, kind=kind)
        if existing != payload:
            raise FixtureReviewContractError(f"{kind} already contains different evidence")
        return
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        _write_new(temporary, payload)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_fixture_review(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Publish only deterministic masked panels and mapping from fixed reconciled evidence."""

    inputs = validate_fixture_review_inputs(project_root, artifact_root=artifact_root)
    root = Path(inputs["artifact_root"])
    review_root = root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    if review_root.is_symlink():
        raise FixtureReviewContractError("review root must not be a symlink")
    mapping_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for panel in inputs["panels"]:
        mapping = _method_mapping(panel["panel_id"], inputs["membership_sha256"])
        reference, lr, outputs = _validated_panel_images(root, panel)
        encoded = _render_panel(panel, mapping, reference, lr, outputs)
        relative = f"review/panels/{panel['panel_id']}.png"
        path = _safe_relative(root, relative, kind="review panel")
        _publish_identical(path, encoded, kind="review panel")
        panel_rows.append(
            {
                "panel_id": panel["panel_id"],
                "condition_id": panel["condition_id"],
                "item_id": panel["item_id"],
                "source_group_id": panel["source_group_id"],
                "roi": copy.deepcopy(panel["roi"]),
                "relative_path": relative,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        mapping_rows.append({"panel_id": panel["panel_id"], "masked_methods": mapping})
    mapping_core = {
        "schema_version": 1,
        "record_type": "fixture-method-mapping",
        "experiment_id": inputs["experiment_id"],
        "membership_id": inputs["membership_id"],
        "membership_sha256": inputs["membership_sha256"],
        "mapping": mapping_rows,
    }
    mapping_record = {**mapping_core, "mapping_sha256": canonical_sha256(mapping_core)}
    _publish_identical(
        review_root / "method-mapping.json",
        _canonical_json(mapping_record),
        kind="method mapping",
    )
    static_identity = {
        "experiment_id": inputs["experiment_id"],
        "reconciliation_sha256": inputs["reconciliation_sha256"],
        "replay_sha256": inputs["replay_sha256"],
        "aggregate_file_sha256": inputs["aggregate_file_sha256"],
        "membership_sha256": inputs["membership_sha256"],
        "mapping_sha256": mapping_record["mapping_sha256"],
        "panel_sha256s": [panel["sha256"] for panel in panel_rows],
    }
    return {
        **{key: value for key, value in inputs.items() if key != "panels"},
        "mapping": mapping_rows,
        "mapping_sha256": mapping_record["mapping_sha256"],
        "panels": panel_rows,
        "working_copy_token": f"generated:{canonical_sha256(static_identity)}",
    }


def execute_fixture_review_notebook(project_root: Path) -> dict[str, Any]:
    """Execute the clean source out of place and bind its ignored working copy."""

    project_root = Path(project_root).resolve()
    prepared = prepare_fixture_review(project_root)
    root = Path(prepared["artifact_root"])
    source_path = project_root / "notebooks/02-fixture-baseline-review.ipynb"
    source_sha256 = notebook_source_sha256(source_path)
    notebook = nbformat.read(source_path, as_version=4)
    replacements = 0
    for cell in notebook.cells:
        if WORKING_COPY_MARKER in cell.source:
            cell.source = cell.source.replace(WORKING_COPY_MARKER, prepared["working_copy_token"])
            replacements += 1
    if replacements != 1:
        raise FixtureReviewContractError("tracked notebook working-copy guard is ambiguous")
    try:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(project_root)}},
        )
        client.execute()
    except Exception as error:
        raise FixtureReviewContractError("ignored review notebook execution failed") from error
    for cell in notebook.cells:
        if isinstance(cell.get("metadata"), dict):
            cell.metadata.pop("execution", None)
    working_bytes = nbformat.writes(notebook).encode("utf-8")
    working_path = root / "review/fixture-baseline-review-working.ipynb"
    if working_path.exists():
        existing_logical = _logical_working_sha256(
            working_path, source_path, prepared["working_copy_token"]
        )
        temporary = working_path.with_name(f".{working_path.name}.tmp-{uuid4().hex}")
        try:
            _write_new(temporary, working_bytes)
            replacement_logical = _logical_working_sha256(
                temporary, source_path, prepared["working_copy_token"]
            )
            if replacement_logical != existing_logical:
                raise FixtureReviewContractError("working notebook logical source changed")
            os.replace(temporary, working_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        _publish_identical(working_path, working_bytes, kind="working notebook")
    logical_sha256 = _logical_working_sha256(
        working_path, source_path, prepared["working_copy_token"]
    )
    manifest_core = {
        "schema_version": 1,
        "record_type": "fixture-review-session",
        "experiment_id": prepared["experiment_id"],
        "reconciliation_id": prepared["reconciliation_id"],
        "reconciliation_sha256": prepared["reconciliation_sha256"],
        "replay_id": prepared["replay_id"],
        "replay_sha256": prepared["replay_sha256"],
        "aggregate_id": prepared["aggregate_id"],
        "aggregate_file_sha256": prepared["aggregate_file_sha256"],
        "membership_id": prepared["membership_id"],
        "membership_sha256": prepared["membership_sha256"],
        "core_membership_id": prepared["core_membership_id"],
        "core_sha256": prepared["core_sha256"],
        "degradation_control_sha256": prepared["degradation_control_sha256"],
        "evaluation_control_sha256": prepared["evaluation_control_sha256"],
        "fixture_manifest_sha256": prepared["fixture_manifest_sha256"],
        "portable_export_id": prepared["portable_export_id"],
        "portable_export_sha256": prepared["portable_export_sha256"],
        "notebook_source_relative_path": "notebooks/02-fixture-baseline-review.ipynb",
        "notebook_source_sha256": source_sha256,
        "working_notebook_relative_path": "review/fixture-baseline-review-working.ipynb",
        "working_notebook_sha256": hashlib.sha256(working_path.read_bytes()).hexdigest(),
        "working_notebook_logical_sha256": logical_sha256,
        "working_copy_token": prepared["working_copy_token"],
        "mapping_relative_path": "review/method-mapping.json",
        "mapping_sha256": prepared["mapping_sha256"],
        "panels": prepared["panels"],
        "requested_panel_count": prepared["requested_panel_count"],
        "displayable_panel_count": prepared["requested_panel_count"],
        "failed_panel_count": 0,
        "review_relative_path": "review/notation-review.json",
    }
    manifest = {**manifest_core, "session_sha256": canonical_sha256(manifest_core)}
    manifest_path = root / "review/review-session-manifest.json"
    payload = _canonical_json(manifest)
    if manifest_path.exists():
        existing = _read_json(manifest_path, kind="review session manifest")
        stable_fields = {
            key: value
            for key, value in manifest.items()
            if key not in {"working_notebook_sha256", "session_sha256"}
        }
        observed = {
            key: value
            for key, value in existing.items()
            if key not in {"working_notebook_sha256", "session_sha256"}
        }
        if stable_fields != observed:
            raise FixtureReviewContractError("existing review session identity differs")
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{uuid4().hex}")
        try:
            _write_new(temporary, payload)
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        _publish_identical(manifest_path, payload, kind="review session manifest")
    if notebook_source_sha256(source_path) != source_sha256:
        raise FixtureReviewContractError("tracked notebook changed during out-of-place execution")
    return manifest


def _review_digest(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    return hashlib.sha256(
        _read_regular(path, maximum_bytes=_MAX_JSON_BYTES, kind="notation review")
    ).hexdigest()


def _load_review_bundle(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    bundle = _read_json(path, kind="notation review")
    if bundle.get("review_sha256") != _self_digest(bundle, "review_sha256"):
        raise FixtureReviewContractError("notation review digest differs")
    required = {
        "schema_version",
        "record_type",
        "session_sha256",
        "sample_membership_id",
        "sample_sha256",
        "mapping_sha256",
        "notebook_source_sha256",
        "working_notebook_logical_sha256",
        "reviews",
        "denominators",
        "review_sha256",
    }
    if (
        set(bundle) != required
        or bundle.get("schema_version") != 1
        or bundle.get("record_type") != "fixture-notation-review-bundle"
    ):
        raise FixtureReviewContractError("notation review envelope differs")
    if (
        bundle["session_sha256"] != manifest["session_sha256"]
        or bundle["sample_membership_id"] != manifest["membership_id"]
        or bundle["sample_sha256"] != manifest["membership_sha256"]
        or bundle["mapping_sha256"] != manifest["mapping_sha256"]
        or bundle["notebook_source_sha256"] != manifest["notebook_source_sha256"]
        or bundle["working_notebook_logical_sha256"] != manifest["working_notebook_logical_sha256"]
    ):
        raise FixtureReviewContractError("notation review lineage differs")
    panel_ids = [panel["panel_id"] for panel in manifest["panels"]]
    reviews = bundle["reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(
        {row.get("panel_id") for row in reviews}
    ):
        raise FixtureReviewContractError("notation review contains duplicate panels")
    if [row.get("panel_id") for row in reviews] != [
        panel for panel in panel_ids if panel in {row.get("panel_id") for row in reviews}
    ]:
        raise FixtureReviewContractError("notation review panel order differs")
    for review in reviews:
        try:
            validate_notation_review(review)
        except (ContractValidationError, ValueError) as error:
            raise FixtureReviewContractError("notation review row is invalid") from error
        if (
            review["sample_membership_id"] != manifest["membership_id"]
            or review["sample_sha256"] != manifest["membership_sha256"]
        ):
            raise FixtureReviewContractError("notation review sample binding differs")
    expected_denominators = {
        "requested_panels": manifest["requested_panel_count"],
        "displayable_panels": manifest["displayable_panel_count"],
        "reviewed_panels": len(reviews),
        "skipped_panels": 0,
        "failed_panels": manifest["failed_panel_count"],
    }
    if bundle["denominators"] != expected_denominators:
        raise FixtureReviewContractError("notation review denominators differ")
    return bundle


def _durable_cas_json(path: Path, payload: bytes, *, expected_sha256: str | None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".notation-review.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    temporary: Path | None = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FixtureReviewContractError("notation review lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        observed = _review_digest(path)
        if observed != expected_sha256:
            raise FixtureReviewContractError("stale notation review session; reload required")
        temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        _write_new(temporary, payload)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


class FixtureReviewSession:
    """Review fixed masked panels while keeping all human state ignored and CAS-protected."""

    def __init__(
        self,
        project_root: Path | str = ".",
        *,
        working_copy_token: str,
        artifact_root: Path | None = None,
    ) -> None:
        start = Path(project_root).resolve()
        self.project_root = next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
                and (candidate / "configs/experiments/phase2-fixture-v1.yaml").is_file()
            ),
            None,
        )
        if self.project_root is None:
            raise FixtureReviewContractError("could not locate the proyecto root")
        if (
            not isinstance(working_copy_token, str)
            or _GENERATED_TOKEN.fullmatch(working_copy_token) is None
        ):
            raise FixtureReviewContractError("execute only the generated ignored working notebook")
        self.prepared = prepare_fixture_review(
            self.project_root,
            artifact_root=artifact_root,
        )
        if working_copy_token != self.prepared["working_copy_token"]:
            raise FixtureReviewContractError("working notebook token does not bind fixed evidence")
        self.working_copy_token = working_copy_token
        self.artifact_root = Path(self.prepared["artifact_root"])
        self.manifest_path = self.artifact_root / "review/review-session-manifest.json"
        self.review_path = self.artifact_root / "review/notation-review.json"
        self.manifest = (
            _read_json(self.manifest_path, kind="review session manifest")
            if self.manifest_path.exists()
            else None
        )
        if self.manifest is not None:
            if self.manifest.get("session_sha256") != _self_digest(self.manifest, "session_sha256"):
                raise FixtureReviewContractError("review session manifest digest differs")
            if (
                self.manifest.get("working_copy_token") != working_copy_token
                or self.manifest.get("membership_sha256") != self.prepared["membership_sha256"]
                or self.manifest.get("mapping_sha256") != self.prepared["mapping_sha256"]
                or self.manifest.get("panels") != self.prepared["panels"]
            ):
                raise FixtureReviewContractError("review session manifest binding differs")
        self.reload()

    def reload(self) -> str | None:
        if self.manifest is None and self.manifest_path.exists():
            self.manifest = _read_json(self.manifest_path, kind="review session manifest")
        self.review = (
            _load_review_bundle(self.review_path, self.manifest)
            if self.manifest is not None
            else None
        )
        self.expected_review_sha256 = _review_digest(self.review_path)
        return self.expected_review_sha256

    def summary(self) -> dict[str, Any]:
        reviewed = len(self.review["reviews"]) if self.review is not None else 0
        return {
            "dataset_role": "pipeline-validation-only",
            "experiment_id": self.prepared["experiment_id"],
            "reconciliation_id": self.prepared["reconciliation_id"],
            "replay_id": self.prepared["replay_id"],
            "requested_panels": self.prepared["requested_panel_count"],
            "displayable_panels": self.prepared["requested_panel_count"],
            "reviewed_panels": reviewed,
            "skipped_panels": 0,
            "failed_panels": 0,
            "methods_masked": True,
        }

    def _require_manifest(self) -> dict[str, Any]:
        if self.manifest is None:
            raise FixtureReviewContractError(
                "review manifest is finalized after notebook generation"
            )
        source_path = self.project_root / self.manifest["notebook_source_relative_path"]
        working_path = self.artifact_root / self.manifest["working_notebook_relative_path"]
        if notebook_source_sha256(source_path) != self.manifest["notebook_source_sha256"]:
            raise FixtureReviewContractError("tracked review notebook source changed")
        if (
            _logical_working_sha256(working_path, source_path, self.working_copy_token)
            != self.manifest["working_notebook_logical_sha256"]
        ):
            raise FixtureReviewContractError("ignored working notebook source changed")
        return self.manifest

    def save_panel_review(
        self,
        *,
        panel_id: str,
        reviewer: str,
        labels: Sequence[str],
        severity: str | None,
        rationale: str,
    ) -> str:
        manifest = self._require_manifest()
        panel_ids = [panel["panel_id"] for panel in manifest["panels"]]
        if panel_id not in panel_ids:
            raise FixtureReviewContractError("review panel is outside fixed membership")
        payload = {
            "schema_version": 2,
            "record_type": "notation-review",
            "sample_membership_id": manifest["membership_id"],
            "sample_sha256": manifest["membership_sha256"],
            "panel_id": panel_id,
            "reviewer": reviewer.strip(),
            "reviewed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "labels": list(labels),
            "severity": severity,
            "rationale": rationale.strip(),
        }
        record = {**payload, "review_id": f"review-{canonical_sha256(payload)}"}
        try:
            validated = validate_notation_review(record)
        except (ContractValidationError, ValueError) as error:
            raise FixtureReviewContractError("notation review input is invalid") from error
        reviews_by_panel = {
            review["panel_id"]: review
            for review in (self.review["reviews"] if self.review is not None else [])
        }
        reviews_by_panel[panel_id] = validated
        reviews = [reviews_by_panel[value] for value in panel_ids if value in reviews_by_panel]
        core = {
            "schema_version": 1,
            "record_type": "fixture-notation-review-bundle",
            "session_sha256": manifest["session_sha256"],
            "sample_membership_id": manifest["membership_id"],
            "sample_sha256": manifest["membership_sha256"],
            "mapping_sha256": manifest["mapping_sha256"],
            "notebook_source_sha256": manifest["notebook_source_sha256"],
            "working_notebook_logical_sha256": manifest["working_notebook_logical_sha256"],
            "reviews": reviews,
            "denominators": {
                "requested_panels": manifest["requested_panel_count"],
                "displayable_panels": manifest["displayable_panel_count"],
                "reviewed_panels": len(reviews),
                "skipped_panels": 0,
                "failed_panels": manifest["failed_panel_count"],
            },
        }
        bundle = {**core, "review_sha256": canonical_sha256(core)}
        self.expected_review_sha256 = _durable_cas_json(
            self.review_path,
            _canonical_json(bundle),
            expected_sha256=self.expected_review_sha256,
        )
        self.review = bundle
        return self.expected_review_sha256

    def panel_widget(self) -> widgets.Widget:
        panels = self.prepared["panels"]
        position = widgets.IntSlider(
            value=0,
            min=0,
            max=len(panels) - 1,
            description="Panel:",
            continuous_update=False,
        )
        previous = widgets.Button(description="← Anterior")
        following = widgets.Button(description="Siguiente →")
        show = widgets.Button(description="Mostrar panel", button_style="info")
        output = widgets.Output()

        def render(*_args: object) -> None:
            panel = panels[position.value]
            path = self.artifact_root / panel["relative_path"]
            raw = _read_regular(path, maximum_bytes=_MAX_IMAGE_BYTES, kind="review panel")
            if hashlib.sha256(raw).hexdigest() != panel["sha256"]:
                raise FixtureReviewContractError("review panel digest differs")
            with output:
                clear_output(wait=True)
                display(widgets.Image(value=raw, format="png", width=1250))
                print(
                    f"{position.value + 1}/{len(panels)} · {panel['condition_id']} · "
                    f"{panel['item_id']} · métodos A/B/C enmascarados"
                )

        show.on_click(render)
        previous.on_click(lambda _button: setattr(position, "value", max(0, position.value - 1)))
        following.on_click(
            lambda _button: setattr(position, "value", min(position.max, position.value + 1))
        )
        return widgets.VBox([widgets.HBox([position, previous, show, following]), output])

    def review_widget(self) -> widgets.Widget:
        panel_ids = [panel["panel_id"] for panel in self.prepared["panels"]]
        panels_by_id = {panel["panel_id"]: panel for panel in self.prepared["panels"]}
        panel = widgets.Dropdown(options=panel_ids, description="Panel:")
        preview = widgets.Output()
        reviewer = widgets.Text(description="Revisor:", placeholder="Nombre")
        checkboxes = {
            label: widgets.Checkbox(value=False, description=label, indent=False)
            for label in NOTATION_TAXONOMY
        }
        severity = widgets.Dropdown(
            options=[("Sin severidad", None), "minor", "material", "unusable"],
            description="Severidad:",
        )
        rationale = widgets.Textarea(
            description="Justificación:", layout=widgets.Layout(width="1000px", height="90px")
        )
        save = widgets.Button(description="Guardar revisión", button_style="success")
        status = widgets.Output()

        def render_selected(_change: object | None = None) -> None:
            selected = panels_by_id[str(panel.value)]
            path = self.artifact_root / selected["relative_path"]
            raw = _read_regular(path, maximum_bytes=_MAX_IMAGE_BYTES, kind="review panel")
            if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
                raise FixtureReviewContractError("review panel digest differs")
            with preview:
                clear_output(wait=True)
                display(widgets.Image(value=raw, format="png", width=1250))
                print(
                    f"{panel_ids.index(str(panel.value)) + 1}/{len(panel_ids)} · "
                    f"{selected['condition_id']} · {selected['item_id']} · "
                    "métodos A/B/C enmascarados"
                )

        def persist(_button: widgets.Button) -> None:
            with status:
                clear_output()
                try:
                    selected = [label for label, checkbox in checkboxes.items() if checkbox.value]
                    self.save_panel_review(
                        panel_id=str(panel.value),
                        reviewer=reviewer.value,
                        labels=selected,
                        severity=severity.value,
                        rationale=rationale.value,
                    )
                    summary = self.summary()
                    print(
                        f"Guardado: {summary['reviewed_panels']}/"
                        f"{summary['displayable_panels']} paneles revisados."
                    )
                    if panel_ids.index(str(panel.value)) + 1 < len(panel_ids):
                        panel.value = panel_ids[panel_ids.index(str(panel.value)) + 1]
                    for checkbox in checkboxes.values():
                        checkbox.value = False
                    severity.value = None
                    rationale.value = ""
                except Exception as error:
                    print(f"No se guardó: {error}")

        save.on_click(persist)
        panel.observe(render_selected, names="value")
        render_selected()
        return widgets.VBox(
            [
                preview,
                widgets.HBox([panel, reviewer]),
                widgets.GridBox(
                    list(checkboxes.values()),
                    layout=widgets.Layout(grid_template_columns="repeat(2, minmax(320px, 1fr))"),
                ),
                severity,
                rationale,
                save,
                status,
            ]
        )

    def progress_widget(self) -> widgets.Widget:
        button = widgets.Button(description="Comprobar progreso", button_style="info")
        output = widgets.Output()

        def check(_button: widgets.Button) -> None:
            with output:
                clear_output()
                self.reload()
                summary = self.summary()
                print(
                    f"Solicitados: {summary['requested_panels']} · "
                    f"mostrables: {summary['displayable_panels']} · "
                    f"revisados: {summary['reviewed_panels']} · "
                    f"omitidos: {summary['skipped_panels']} · "
                    f"fallidos: {summary['failed_panels']}"
                )
                if summary["reviewed_panels"] == summary["displayable_panels"]:
                    print("Revisión completa. Comunica a Codex: review complete")

        button.on_click(check)
        return widgets.VBox([button, output])
