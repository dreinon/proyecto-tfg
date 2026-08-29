"""Masked, content-addressed review of the fixed Phase 2 fixture evidence."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import ipywidgets as widgets
import nbformat
import numpy as np
import yaml
from defusedxml import ElementTree
from IPython.display import clear_output, display
from nbclient import NotebookClient

from score_super_resolution.baselines import pixel_sha256, run_baseline
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import (
    DegradationControl,
    align_reference,
    apply_degradation,
    generate_visual_fixture_bundle,
    native_physical_review_rois,
)
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
SEMANTIC_CONFIG_RELATIVE = Path("configs/experiments/phase2-semantic-fixture-v1.yaml")
SEMANTIC_SOURCE_IDS = (
    "review-work-03-excerpt-01",
    "review-work-04-excerpt-01",
)
SEMANTIC_WORKING_COPY_MARKER = "__GENERATED_PHASE2_SEMANTIC_REVIEW_WORKING_COPY__"
_D23_ERROR = (
    "D-23 superseded checkpoint: primitive fixtures are technical evidence only; "
    "semantic review persistence, reveal, and reconciliation are forbidden"
)


class FixtureReviewContractError(ValueError):
    """Fixed evidence, notebook, panel, or review state violates the review contract."""


def _read_yaml(path: Path, *, kind: str, maximum_bytes: int = 1024 * 1024) -> dict[str, Any]:
    try:
        value = yaml.safe_load(_read_regular(path, maximum_bytes=maximum_bytes, kind=kind))
    except (UnicodeError, yaml.YAMLError) as error:
        raise FixtureReviewContractError(f"{kind} is malformed") from error
    if not isinstance(value, dict):
        raise FixtureReviewContractError(f"{kind} root must be a mapping")
    return value


def _assert_no_secret_or_remote_identity(value: Any, *, kind: str) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    forbidden = ("hf_token", "authorization", "api_key", "praig/smb", "load_dataset")
    if any(token in serialized for token in forbidden):
        raise FixtureReviewContractError(f"{kind} contains a secret-like or remote identity")


def load_semantic_fixture_control(
    project_root: Path,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the exact two-source applicability denominator before any semantic output exists."""

    project_root = Path(project_root).resolve()
    config_path = project_root / SEMANTIC_CONFIG_RELATIVE if path is None else Path(path).resolve()
    control = _read_yaml(config_path, kind="semantic fixture control")
    _assert_no_secret_or_remote_identity(control, kind="semantic fixture control")
    try:
        validate_instance("semantic-fixture-experiment", control, version=2)
    except ContractValidationError as error:
        raise FixtureReviewContractError("semantic fixture control fails its schema") from error

    manifest_path = project_root / control["visual_manifest"]["manifest_path"]
    manifest = _read_yaml(manifest_path, kind="visual fixture manifest")
    if (
        manifest.get("manifest_id") != control["visual_manifest"]["manifest_id"]
        or canonical_sha256(manifest) != control["visual_manifest"]["manifest_sha256"]
    ):
        raise FixtureReviewContractError("semantic visual manifest identity differs")
    selected = {
        item["item_id"]: item
        for item in manifest.get("items", [])
        if item.get("item_id") in SEMANTIC_SOURCE_IDS
    }
    if (
        tuple(control["source_order"]) != SEMANTIC_SOURCE_IDS
        or tuple(selected) != SEMANTIC_SOURCE_IDS
    ):
        raise FixtureReviewContractError("semantic source order differs")
    expected_sources: list[dict[str, Any]] = []
    for source_id in SEMANTIC_SOURCE_IDS:
        item = selected[source_id]
        expected_sources.append(
            {
                "source_id": source_id,
                "source_group_id": item["source_group_id"],
                "source_role": item["source_role"],
                "origin": item["origin"],
                "author": item["author"],
                "license": item["license"],
                "source_path": f"tests/fixtures/phase2/{item['source_relative_path']}",
                "source_sha256": item["source_sha256"],
                "rendered_relative_path": f"inputs/{source_id}.png",
                "rendered_pixel_sha256": item["rendered_pixel_sha256"],
                "roi": item["roi"],
            }
        )
    if control["sources"] != expected_sources:
        raise FixtureReviewContractError("semantic source provenance, digest, or ROI differs")
    expected_renderer = copy.deepcopy(manifest["renderer"])
    expected_renderer["engraver"].pop("project_url", None)
    expected_renderer["rasterizer"].pop("project_url", None)
    if control["renderer"] != expected_renderer:
        raise FixtureReviewContractError("semantic renderer contract differs")
    if importlib.metadata.version("verovio") != control["renderer"]["engraver"]["version"]:
        raise FixtureReviewContractError("semantic Verovio runtime differs")
    if importlib.metadata.version("cairosvg") != control["renderer"]["rasterizer"]["version"]:
        raise FixtureReviewContractError("semantic CairoSVG runtime differs")

    if tuple(control["condition_order"]) != EXPECTED_CONDITIONS:
        raise FixtureReviewContractError("semantic condition order differs")
    if tuple(control["method_order"]) != EXPECTED_METHODS:
        raise FixtureReviewContractError("semantic method order differs")
    expected_tuples = [
        f"{source_id}|{condition_id}|{method_id}"
        for source_id in SEMANTIC_SOURCE_IDS
        for condition_id in EXPECTED_CONDITIONS
        for method_id in EXPECTED_METHODS
    ]
    if control["expected_tuple_keys"] != expected_tuples:
        raise FixtureReviewContractError("semantic expected tuple order differs")
    expected_membership = [
        {
            "panel_id": f"panel-{index:02d}",
            "condition_id": condition_id,
            "source_id": source_id,
        }
        for index, (source_id, condition_id) in enumerate(
            (
                (source_id, condition_id)
                for source_id in SEMANTIC_SOURCE_IDS
                for condition_id in EXPECTED_CONDITIONS
            ),
            start=1,
        )
    ]
    if control["review_membership"] != expected_membership:
        raise FixtureReviewContractError("semantic review membership differs")

    for control_name, digest_name in (
        ("degradation", "degradation_sha256"),
        ("evaluation", "evaluation_sha256"),
    ):
        control_file = _read_yaml(
            project_root / control["controls"][f"{control_name}_path"],
            kind=f"{control_name} control",
        )
        if canonical_sha256(control_file) != control["controls"][digest_name]:
            raise FixtureReviewContractError(f"semantic {control_name} control digest differs")
    if control["controls"]["degradation_seed"] != 20260821:
        raise FixtureReviewContractError("semantic degradation seed differs")

    identity = canonical_sha256(control)
    return copy.deepcopy(
        {
            **control,
            "semantic_experiment_id": f"semantic-experiment-{identity}",
            "semantic_experiment_sha256": identity,
        }
    )


def _local_name(element: Any) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _children(element: Any, name: str) -> list[Any]:
    return [child for child in list(element) if _local_name(child) == name]


def _child(element: Any, name: str) -> Any | None:
    matches = _children(element, name)
    if len(matches) > 1:
        raise FixtureReviewContractError(f"MusicXML coherence has duplicate {name} elements")
    return matches[0] if matches else None


def _text(element: Any, name: str, *, required: bool = True) -> str | None:
    child = _child(element, name)
    if child is None:
        if required:
            raise FixtureReviewContractError(f"MusicXML coherence requires {name}")
        return None
    value = (child.text or "").strip()
    if required and not value:
        raise FixtureReviewContractError(f"MusicXML coherence requires non-empty {name}")
    return value or None


_TYPE_DURATION = {
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
}


def validate_semantic_musicxml_source(
    source_path: Path,
    *,
    source: Mapping[str, Any],
    renderer: Mapping[str, Any],
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject tag-complete but temporally or relationally incoherent MusicXML."""

    source_id = source.get("source_id")
    if source_id not in SEMANTIC_SOURCE_IDS:
        raise FixtureReviewContractError("MusicXML source is outside the semantic denominator")
    payload = _read_regular(
        Path(source_path),
        maximum_bytes=int(limits["max_source_bytes"]),
        kind="semantic MusicXML source",
    )
    if hashlib.sha256(payload).hexdigest() != source.get("source_sha256"):
        raise FixtureReviewContractError("MusicXML source digest differs")
    if (
        source.get("source_role") != "visual-degradation-review"
        or source.get("origin") != "authored-for-this-tfg-fixture-suite"
        or source.get("author") != "TFG score-super-resolution project"
        or source.get("license") != "CC0-1.0"
    ):
        raise FixtureReviewContractError("MusicXML provenance or licence differs")
    if (
        renderer.get("engraver", {}).get("source_format") != "MusicXML 4.0 partwise"
        or renderer.get("engraver", {}).get("version") != "6.2.1"
        or renderer.get("rasterizer", {}).get("version") != "2.9.0"
    ):
        raise FixtureReviewContractError("MusicXML renderer identity differs")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise FixtureReviewContractError("MusicXML coherence parse failed") from error
    nodes = list(root.iter())
    if len(nodes) > int(limits["max_xml_nodes"]):
        raise FixtureReviewContractError("MusicXML coherence node bound exceeded")

    def depth(element: Any, current: int = 1) -> int:
        return max([current, *(depth(child, current + 1) for child in list(element))])

    if depth(root) > int(limits["max_xml_depth"]):
        raise FixtureReviewContractError("MusicXML coherence depth bound exceeded")
    if _local_name(root) != "score-partwise" or root.attrib.get("version") != "4.0":
        raise FixtureReviewContractError("MusicXML coherence requires score-partwise 4.0")

    part_list = _child(root, "part-list")
    if part_list is None:
        raise FixtureReviewContractError("MusicXML coherence requires part-list")
    declared_parts = [part.attrib.get("id") for part in _children(part_list, "score-part")]
    actual_parts = [part.attrib.get("id") for part in _children(root, "part")]
    if (
        not declared_parts
        or any(not value for value in declared_parts)
        or len(set(declared_parts)) != len(declared_parts)
        or actual_parts != declared_parts
    ):
        raise FixtureReviewContractError("MusicXML coherence part references differ")

    open_beams: set[tuple[str, str, int, str]] = set()
    open_slurs: set[tuple[str, str, int, str]] = set()
    open_ties: set[tuple[str, str, int, str]] = set()
    open_tieds: set[tuple[str, str, int, str]] = set()
    measure_count = 0
    note_count = 0
    for part in _children(root, "part"):
        part_id = str(part.attrib["id"])
        divisions: int | None = None
        expected_measure_duration: Fraction | None = None
        staves = 1
        measures = _children(part, "measure")
        if not measures or len(measures) > int(limits["max_measures_per_part"]):
            raise FixtureReviewContractError("MusicXML coherence measure count is invalid")
        measure_count += len(measures)
        for measure in measures:
            attributes = _child(measure, "attributes")
            if attributes is not None:
                divisions_text = _text(attributes, "divisions", required=False)
                if divisions_text is not None:
                    divisions = int(divisions_text)
                    if divisions <= 0:
                        raise FixtureReviewContractError("MusicXML divisions must be positive")
                staves_text = _text(attributes, "staves", required=False)
                if staves_text is not None:
                    staves = int(staves_text)
                    if staves < 1:
                        raise FixtureReviewContractError("MusicXML staves must be positive")
                time = _child(attributes, "time")
                if time is not None:
                    if divisions is None:
                        raise FixtureReviewContractError("MusicXML time requires active divisions")
                    beats = int(_text(time, "beats"))
                    beat_type = int(_text(time, "beat-type"))
                    duration = Fraction(divisions * beats * 4, beat_type)
                    if duration.denominator != 1 or duration <= 0:
                        raise FixtureReviewContractError(
                            "MusicXML measure duration is not integral"
                        )
                    expected_measure_duration = duration
            if divisions is None or expected_measure_duration is None:
                raise FixtureReviewContractError("MusicXML measure lacks active timing")

            cursor = Fraction(0)
            intervals: dict[tuple[str, int], list[tuple[Fraction, Fraction]]] = {}
            last_note: tuple[str, int, Fraction, Fraction] | None = None
            for event in list(measure):
                tag = _local_name(event)
                if tag == "backup":
                    duration = Fraction(int(_text(event, "duration")))
                    if duration <= 0 or cursor - duration < 0:
                        raise FixtureReviewContractError("MusicXML backup duration is invalid")
                    cursor -= duration
                    last_note = None
                    continue
                if tag == "forward":
                    duration = Fraction(int(_text(event, "duration")))
                    if duration <= 0:
                        raise FixtureReviewContractError("MusicXML forward duration is invalid")
                    cursor += duration
                    last_note = None
                    continue
                if tag != "note":
                    continue
                note_count += 1
                pitch = _child(event, "pitch")
                rest = _child(event, "rest")
                if (pitch is None) == (rest is None):
                    raise FixtureReviewContractError(
                        "MusicXML note requires pitch/rest exclusivity"
                    )
                duration = Fraction(int(_text(event, "duration")))
                if duration <= 0:
                    raise FixtureReviewContractError("MusicXML note duration must be positive")
                note_type = _text(event, "type")
                if note_type not in _TYPE_DURATION:
                    raise FixtureReviewContractError("MusicXML note type is unsupported")
                dots = len(_children(event, "dot"))
                dot_factor = sum((Fraction(1, 2**index) for index in range(dots + 1)), Fraction())
                if duration != _TYPE_DURATION[note_type] * divisions * dot_factor:
                    raise FixtureReviewContractError("MusicXML duration/type/dot coherence differs")
                voice = str(_text(event, "voice"))
                staff_text = _text(event, "staff", required=False)
                staff = int(staff_text) if staff_text is not None else 1
                if staff < 1 or staff > staves:
                    raise FixtureReviewContractError("MusicXML staff reference is invalid")
                chord = _child(event, "chord") is not None
                if chord:
                    if (
                        last_note is None
                        or last_note[:2] != (voice, staff)
                        or last_note[3] != duration
                    ):
                        raise FixtureReviewContractError("MusicXML chord relation is inconsistent")
                    onset = last_note[2]
                else:
                    onset = cursor
                    cursor += duration
                    last_note = (voice, staff, onset, duration)
                    intervals.setdefault((voice, staff), []).append((onset, onset + duration))
                pitch_key = "rest"
                if pitch is not None:
                    pitch_key = ":".join(
                        (
                            str(_text(pitch, "step")),
                            str(_text(pitch, "alter", required=False) or "0"),
                            str(_text(pitch, "octave")),
                        )
                    )
                relation_prefix = (part_id, voice, staff, pitch_key)
                tie_types = {tie.attrib.get("type") for tie in _children(event, "tie")}
                notations = _child(event, "notations")
                tied_types = (
                    {tied.attrib.get("type") for tied in _children(notations, "tied")}
                    if notations is not None
                    else set()
                )
                if tie_types != tied_types:
                    raise FixtureReviewContractError("MusicXML tie/tied relation differs")
                for relation_type in tie_types:
                    if relation_type == "start":
                        if relation_prefix in open_ties or relation_prefix in open_tieds:
                            raise FixtureReviewContractError("MusicXML tie relation starts twice")
                        open_ties.add(relation_prefix)
                        open_tieds.add(relation_prefix)
                    elif relation_type == "stop":
                        if relation_prefix not in open_ties or relation_prefix not in open_tieds:
                            raise FixtureReviewContractError("MusicXML tie relation stops unopened")
                        open_ties.remove(relation_prefix)
                        open_tieds.remove(relation_prefix)
                    else:
                        raise FixtureReviewContractError("MusicXML tie relation type is invalid")
                for beam in _children(event, "beam"):
                    key = (part_id, voice, staff, beam.attrib.get("number", "1"))
                    value = (beam.text or "").strip()
                    if value == "begin":
                        if key in open_beams:
                            raise FixtureReviewContractError("MusicXML beam relation starts twice")
                        open_beams.add(key)
                    elif value == "continue":
                        if key not in open_beams:
                            raise FixtureReviewContractError(
                                "MusicXML beam relation continues unopened"
                            )
                    elif value == "end":
                        if key not in open_beams:
                            raise FixtureReviewContractError("MusicXML beam relation ends unopened")
                        open_beams.remove(key)
                    else:
                        raise FixtureReviewContractError("MusicXML beam relation type is invalid")
                if notations is not None:
                    for slur in _children(notations, "slur"):
                        key = (part_id, voice, staff, slur.attrib.get("number", "1"))
                        relation_type = slur.attrib.get("type")
                        if relation_type == "start":
                            if key in open_slurs:
                                raise FixtureReviewContractError(
                                    "MusicXML slur relation starts twice"
                                )
                            open_slurs.add(key)
                        elif relation_type == "stop":
                            if key not in open_slurs:
                                raise FixtureReviewContractError(
                                    "MusicXML slur relation stops unopened"
                                )
                            open_slurs.remove(key)
                        else:
                            raise FixtureReviewContractError(
                                "MusicXML slur relation type is invalid"
                            )

            expected = expected_measure_duration
            if cursor != expected:
                raise FixtureReviewContractError("MusicXML measure duration coherence differs")
            for voice_staff, spans in intervals.items():
                ordered = sorted(set(spans))
                if not ordered or ordered[0][0] != 0 or ordered[-1][1] != expected:
                    raise FixtureReviewContractError(
                        f"MusicXML voice/staff duration coherence differs for {voice_staff}"
                    )
                if any(left[1] != right[0] for left, right in pairwise(ordered)):
                    raise FixtureReviewContractError("MusicXML voice/staff contains a timing gap")

    if open_beams or open_slurs or open_ties or open_tieds:
        raise FixtureReviewContractError("MusicXML coherence has an unbalanced relation")
    roi = source.get("roi")
    if (
        not isinstance(roi, Mapping)
        or any(
            isinstance(roi.get(key), bool) or not isinstance(roi.get(key), int)
            for key in ("x", "y", "width", "height")
        )
        or roi["x"] < 0
        or roi["y"] < 0
        or roi["width"] <= 0
        or roi["height"] <= 0
        or roi["x"] + roi["width"] > int(renderer["rasterizer"]["canvas_width"])
        or roi["y"] + roi["height"] > int(renderer["rasterizer"]["canvas_height"])
    ):
        raise FixtureReviewContractError("MusicXML rendered ROI bounds differ")
    return {
        "source_id": source_id,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "coherence_state": "structurally-coherent",
        "measure_count": measure_count,
        "note_count": note_count,
        "renderer_sha256": canonical_sha256(renderer),
        "roi": copy.deepcopy(dict(roi)),
    }


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
    """Reject the superseded primitive semantic notebook without mutating its bytes."""

    project_root = Path(project_root).resolve()
    review_path = project_root / "artifacts/phase2-fixture/review/notation-review.json"
    if review_path.exists() or review_path.is_symlink():
        raise FixtureReviewContractError(f"{_D23_ERROR}; invalid notation-review.json exists")
    raise FixtureReviewContractError(_D23_ERROR)


def _execute_fixture_review_notebook_legacy(project_root: Path) -> dict[str, Any]:
    """Historical implementation retained only to explain committed technical lineage."""

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
        primitive_review = (
            self.project_root / "artifacts/phase2-fixture/review/notation-review.json"
        )
        if primitive_review.exists() or primitive_review.is_symlink():
            raise FixtureReviewContractError(f"{_D23_ERROR}; invalid notation-review.json exists")
        raise FixtureReviewContractError(_D23_ERROR)
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

    # These four unreachable primitive-class methods are UI templates reused by the
    # coherent-notation session below; FixtureReviewSession.__init__ always fails D-23.
    def _semantic_source_confirmation_widget_template(self) -> widgets.Widget:
        reviewer = widgets.Text(description="Revisor:", placeholder="Nombre")
        checks = {
            source_id: widgets.Checkbox(
                value=False,
                description=f"Confirmo música coherente: {source_id}",
                indent=False,
            )
            for source_id in SEMANTIC_SOURCE_IDS
        }
        rationales = {
            source_id: widgets.Textarea(
                description="Motivo:",
                placeholder="Describe brevemente por qué la notación es coherente y reconocible",
                layout=widgets.Layout(width="1000px", height="70px"),
            )
            for source_id in SEMANTIC_SOURCE_IDS
        }
        show = widgets.Button(description="Mostrar anclas HR", button_style="info")
        save = widgets.Button(description="Guardar confirmaciones", button_style="success")
        preview = widgets.Output()
        status = widgets.Output()

        def render(_button: widgets.Button) -> None:
            manifest = _load_semantic_manifest(self.artifact_root)
            with preview:
                clear_output(wait=True)
                for row in manifest["inputs"]:
                    raw = _read_regular(
                        self.artifact_root / row["relative_path"],
                        maximum_bytes=_MAX_IMAGE_BYTES,
                        kind="semantic HR anchor",
                    )
                    if hashlib.sha256(raw).hexdigest() != row["encoded_sha256"]:
                        raise FixtureReviewContractError("semantic HR anchor digest differs")
                    print(f"{row['source_id']} · HR completa")
                    display(widgets.Image(value=raw, format="png", width=1000))

        def persist(_button: widgets.Button) -> None:
            with status:
                clear_output()
                try:
                    selected = [source_id for source_id, check in checks.items() if check.value]
                    self.save_source_confirmations(
                        reviewer=reviewer.value,
                        rationales={key: value.value for key, value in rationales.items()},
                        confirmed_sources=selected,
                    )
                    print("Confirmaciones HR guardadas: 2/2.")
                except Exception as error:
                    print(f"No se guardó: {error}")

        show.on_click(render)
        save.on_click(persist)
        rows: list[widgets.Widget] = [reviewer, show]
        for source_id in SEMANTIC_SOURCE_IDS:
            rows.extend([checks[source_id], rationales[source_id]])
        rows.extend([save, preview, status])
        return widgets.VBox(rows)

    def _semantic_panel_widget_template(self) -> widgets.Widget:
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
            raw = _read_regular(
                self.artifact_root / panel["relative_path"],
                maximum_bytes=_MAX_IMAGE_BYTES,
                kind="semantic review panel",
            )
            if hashlib.sha256(raw).hexdigest() != panel["sha256"]:
                raise FixtureReviewContractError("semantic review panel digest differs")
            with output:
                clear_output(wait=True)
                display(widgets.Image(value=raw, format="png", width=1450))
                print(
                    f"{position.value + 1}/12 · {panel['source_id']} · "
                    f"{panel['condition_id']} · métodos A/B/C enmascarados"
                )

        show.on_click(render)
        previous.on_click(lambda _button: setattr(position, "value", max(0, position.value - 1)))
        following.on_click(
            lambda _button: setattr(position, "value", min(position.max, position.value + 1))
        )
        return widgets.VBox([widgets.HBox([position, previous, show, following]), output])

    def _semantic_review_widget_template(self) -> widgets.Widget:
        panel_ids = [row["panel_id"] for row in self.prepared["panels"]]
        panels_by_id = {row["panel_id"]: row for row in self.prepared["panels"]}
        panel = widgets.Dropdown(options=panel_ids, description="Panel:")
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
        show = widgets.Button(description="Mostrar seleccionado", button_style="info")
        save = widgets.Button(description="Guardar revisión", button_style="success")
        preview = widgets.Output()
        status = widgets.Output()

        def render(_button: widgets.Button) -> None:
            selected = panels_by_id[str(panel.value)]
            raw = _read_regular(
                self.artifact_root / selected["relative_path"],
                maximum_bytes=_MAX_IMAGE_BYTES,
                kind="semantic review panel",
            )
            if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
                raise FixtureReviewContractError("semantic review panel digest differs")
            with preview:
                clear_output(wait=True)
                display(widgets.Image(value=raw, format="png", width=1450))
                print(
                    f"{panel_ids.index(str(panel.value)) + 1}/12 · {selected['source_id']} · "
                    f"{selected['condition_id']} · métodos A/B/C enmascarados"
                )

        def persist(_button: widgets.Button) -> None:
            with status:
                clear_output()
                try:
                    labels = [label for label, check in checkboxes.items() if check.value]
                    self.save_panel_review(
                        panel_id=str(panel.value),
                        reviewer=reviewer.value,
                        labels=labels,
                        severity=severity.value,
                        rationale=rationale.value,
                    )
                    print(f"Guardado: {self.summary()['reviewed_panels']}/12 paneles revisados.")
                    index = panel_ids.index(str(panel.value))
                    if index + 1 < len(panel_ids):
                        panel.value = panel_ids[index + 1]
                    for check in checkboxes.values():
                        check.value = False
                    severity.value = None
                    rationale.value = ""
                except Exception as error:
                    print(f"No se guardó: {error}")

        show.on_click(render)
        save.on_click(persist)
        return widgets.VBox(
            [
                widgets.HBox([panel, reviewer, show]),
                preview,
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

    def _semantic_progress_widget_template(self) -> widgets.Widget:
        button = widgets.Button(description="Comprobar progreso", button_style="info")
        output = widgets.Output()

        def check(_button: widgets.Button) -> None:
            with output:
                clear_output()
                self.reload()
                summary = self.summary()
                print(
                    f"Fuentes HR confirmadas: {summary['confirmed_sources']}/2 · "
                    f"paneles revisados: {summary['reviewed_panels']}/12 · "
                    f"omitidos: {summary['skipped_panels']} · fallidos: {summary['failed_panels']}"
                )
                if summary["confirmed_sources"] == 2 and summary["reviewed_panels"] == 12:
                    print("Revisión semántica completa. Comunica: semantic review complete")

        button.on_click(check)
        return widgets.VBox([button, output])

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


def _fixture_pixel_sha256(pixels: np.ndarray) -> str:
    height, width, channels = pixels.shape
    framed = (
        b"phase2-fixture-rgb8-v1\0"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + channels.to_bytes(1, "big")
        + pixels.tobytes(order="C")
    )
    return hashlib.sha256(framed).hexdigest()


def _semantic_artifact_root(project_root: Path, artifact_root: Path | None) -> Path:
    project_root = Path(project_root).resolve()
    root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else project_root / "artifacts/phase2-semantic-fixture"
    )
    if root == project_root or root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FixtureReviewContractError("semantic artifact root is unavailable or unsafe")
    return root


def _artifact_inventory(root: Path, *, kind: str) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise FixtureReviewContractError(f"{kind} root is unavailable or unsafe")
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise FixtureReviewContractError(f"{kind} contains a symlink: {relative}")
        if path.is_file():
            payload = _read_regular(path, maximum_bytes=128 * 1024 * 1024, kind=kind)
            inventory[relative] = hashlib.sha256(payload).hexdigest()
        elif not path.is_dir():
            raise FixtureReviewContractError(f"{kind} contains a non-regular entry")
    return inventory


def _degradation_control(project_root: Path, semantic: Mapping[str, Any]) -> DegradationControl:
    raw = _read_yaml(
        project_root / semantic["controls"]["degradation_path"],
        kind="frozen degradation control",
    )
    if canonical_sha256(raw) != semantic["controls"]["degradation_sha256"]:
        raise FixtureReviewContractError("frozen degradation control identity differs")
    if (
        raw.get("control_id") != semantic["controls"]["degradation_id"]
        or raw.get("status") != "frozen"
        or raw.get("master_seed") != semantic["controls"]["degradation_seed"]
        or tuple(raw.get("condition_order", ())) != EXPECTED_CONDITIONS
    ):
        raise FixtureReviewContractError("frozen degradation control content differs")
    return DegradationControl(
        version=int(raw["version"]),
        candidate_id=str(raw["candidate_id"]),
        status=str(raw["status"]),
        claim_boundary=str(raw["claim_boundary"]),
        master_seed=int(raw["master_seed"]),
        image_contract=copy.deepcopy(raw["image_contract"]),
        alignment=copy.deepcopy(raw["alignment"]),
        runtime=copy.deepcopy(raw["runtime"]),
        condition_ids=tuple(raw["condition_order"]),
        conditions=tuple(copy.deepcopy(raw["conditions"])),
        sha256=canonical_sha256(raw),
    )


def _load_semantic_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "semantic-experiment-manifest.json", kind="semantic manifest")
    _require_self_digest(manifest, "manifest_sha256", kind="semantic manifest")
    if manifest.get("record_type") != "semantic-experiment-manifest":
        raise FixtureReviewContractError("semantic manifest record type differs")
    return manifest


def prepare_semantic_fixture_experiment(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Publish the exact applicability inputs and expected 36-key matrix before compute."""

    project_root = Path(project_root).resolve()
    semantic = load_semantic_fixture_control(project_root)
    root = _semantic_artifact_root(project_root, artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    primitive_root = project_root / "artifacts/phase2-fixture"
    primitive_review = primitive_root / "review/notation-review.json"
    if primitive_review.exists() or primitive_review.is_symlink():
        raise FixtureReviewContractError(f"{_D23_ERROR}; invalid notation-review.json exists")
    primitive_inventory = _artifact_inventory(primitive_root, kind="primitive evidence")
    primitive_fixed = validate_fixture_review_inputs(project_root)

    applicability_rows = [
        validate_semantic_musicxml_source(
            project_root / source["source_path"],
            source=source,
            renderer=semantic["renderer"],
            limits=semantic["limits"],
        )
        for source in semantic["sources"]
    ]
    manifest_path = project_root / semantic["visual_manifest"]["manifest_path"]
    source_root = project_root / "tests/fixtures/phase2"
    with tempfile.TemporaryDirectory(prefix="phase2-semantic-render-") as temporary_name:
        temporary = Path(temporary_name)
        first = generate_visual_fixture_bundle(
            manifest_path, source_root=source_root, output_root=temporary / "first"
        )
        second = generate_visual_fixture_bundle(
            manifest_path, source_root=source_root, output_root=temporary / "second"
        )
        first_by_id = {row["item_id"]: row for row in first["items"]}
        second_by_id = {row["item_id"]: row for row in second["items"]}
        input_rows: list[dict[str, Any]] = []
        for source in semantic["sources"]:
            source_id = source["source_id"]
            left = first_by_id[source_id]
            right = second_by_id[source_id]
            left_payload = _read_regular(
                temporary / "first" / left["relative_path"],
                maximum_bytes=semantic["limits"]["max_output_bytes"],
                kind="first deterministic engraving",
            )
            right_payload = _read_regular(
                temporary / "second" / right["relative_path"],
                maximum_bytes=semantic["limits"]["max_output_bytes"],
                kind="second deterministic engraving",
            )
            if (
                left != right
                or left_payload != right_payload
                or left["pixel_sha256"] != source["rendered_pixel_sha256"]
            ):
                raise FixtureReviewContractError("semantic engraving is not deterministic")
            destination = _safe_relative(root, source["rendered_relative_path"], kind="HR input")
            _publish_identical(destination, left_payload, kind="semantic HR input")
            input_rows.append(
                {
                    "source_id": source_id,
                    "source_group_id": source["source_group_id"],
                    "source_sha256": source["source_sha256"],
                    "relative_path": source["rendered_relative_path"],
                    "encoded_sha256": hashlib.sha256(left_payload).hexdigest(),
                    "pixel_sha256": left["pixel_sha256"],
                    "width": left["width"],
                    "height": left["height"],
                    "roi": copy.deepcopy(source["roi"]),
                }
            )

    applicability_core = {
        "schema_version": 2,
        "record_type": "semantic-applicability",
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "claim_boundary": semantic["claim_boundary"],
        "renderer": copy.deepcopy(semantic["renderer"]),
        "sources": applicability_rows,
        "deterministic_render_count_per_source": 2,
        "input_records": input_rows,
        "non_claims": copy.deepcopy(semantic["non_claims"]),
    }
    applicability = {
        **applicability_core,
        "applicability_sha256": canonical_sha256(applicability_core),
    }
    _publish_identical(
        root / semantic["paths"]["applicability"],
        _canonical_json(applicability),
        kind="semantic applicability",
    )

    manifest_core = {
        "schema_version": 2,
        "record_type": "semantic-experiment-manifest",
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "claim_boundary": semantic["claim_boundary"],
        "expected_tuple_keys": copy.deepcopy(semantic["expected_tuple_keys"]),
        "expected_tuple_count": 36,
        "source_order": copy.deepcopy(semantic["source_order"]),
        "condition_order": copy.deepcopy(semantic["condition_order"]),
        "method_order": copy.deepcopy(semantic["method_order"]),
        "review_membership": copy.deepcopy(semantic["review_membership"]),
        "controls": copy.deepcopy(semantic["controls"]),
        "inputs": input_rows,
        "applicability_sha256": applicability["applicability_sha256"],
        "primitive_inventory": primitive_inventory,
        "primitive_inventory_sha256": canonical_sha256(primitive_inventory),
        "primitive_fixed_identities": {
            "experiment_id": primitive_fixed["experiment_id"],
            "reconciliation_id": primitive_fixed["reconciliation_id"],
            "reconciliation_sha256": primitive_fixed["reconciliation_sha256"],
            "replay_id": primitive_fixed["replay_id"],
            "replay_sha256": primitive_fixed["replay_sha256"],
            "aggregate_file_sha256": primitive_fixed["aggregate_file_sha256"],
            "membership_sha256": primitive_fixed["membership_sha256"],
        },
        "paths": copy.deepcopy(semantic["paths"]),
        "limits": copy.deepcopy(semantic["limits"]),
        "non_claims": copy.deepcopy(semantic["non_claims"]),
    }
    manifest = {**manifest_core, "manifest_sha256": canonical_sha256(manifest_core)}
    _publish_identical(
        root / semantic["paths"]["manifest"],
        _canonical_json(manifest),
        kind="semantic experiment manifest",
    )
    return {**manifest, "artifact_root": root}


def _encode_png(pixels: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not ok:
        raise FixtureReviewContractError("semantic PNG encoding failed")
    return bytes(encoded)


def execute_semantic_fixture_experiment(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Run only the predeclared semantic matrix through the frozen public operators."""

    project_root = Path(project_root).resolve()
    semantic = load_semantic_fixture_control(project_root)
    root = _semantic_artifact_root(project_root, artifact_root)
    manifest = _load_semantic_manifest(root)
    if (
        manifest["semantic_experiment_sha256"] != semantic["semantic_experiment_sha256"]
        or manifest["expected_tuple_keys"] != semantic["expected_tuple_keys"]
    ):
        raise FixtureReviewContractError("semantic manifest no longer binds the frozen matrix")
    control = _degradation_control(project_root, semantic)
    inputs = {row["source_id"]: row for row in manifest["inputs"]}
    terminal: list[str] = []
    for source in semantic["sources"]:
        source_id = source["source_id"]
        input_record = inputs[source_id]
        reference = _decode_rgb(
            root / input_record["relative_path"],
            encoded_sha256=input_record["encoded_sha256"],
            pixel_digest=None,
            kind="semantic HR input",
        )
        if _fixture_pixel_sha256(reference) != input_record["pixel_sha256"]:
            raise FixtureReviewContractError("semantic HR input pixel digest differs")
        for condition_id in semantic["condition_order"]:
            degraded = apply_degradation(
                reference,
                control=control,
                condition_id=condition_id,
                item_id=source_id,
                source_group_id=source["source_group_id"],
                fixture_manifest_id=semantic["visual_manifest"]["manifest_id"],
                purpose="benchmark",
            )
            trace_relative = semantic["paths"]["traces"].format(
                condition_id=condition_id, source_id=source_id
            )
            _publish_identical(
                _safe_relative(root, trace_relative, kind="semantic trace"),
                _canonical_json(degraded.trace),
                kind="semantic degradation trace",
            )
            extension = "jpg" if degraded.encoded_bytes[:2] == b"\xff\xd8" else "png"
            lr_relative = f"lr/{condition_id}/{source_id}.{extension}"
            _publish_identical(
                _safe_relative(root, lr_relative, kind="semantic LR"),
                degraded.encoded_bytes,
                kind="semantic LR",
            )
            aligned = align_reference(reference, int(condition_id[1]))
            for method_id in semantic["method_order"]:
                tuple_key = f"{source_id}|{condition_id}|{method_id}"
                if tuple_key not in semantic["expected_tuple_keys"]:
                    raise FixtureReviewContractError("semantic execution escaped expected matrix")
                baseline = run_baseline(
                    method_id,
                    degraded.pixels,
                    target_shape=tuple(int(value) for value in aligned.pixels.shape),
                    condition_id=condition_id,
                )
                output_relative = semantic["paths"]["outputs"].format(
                    condition_id=condition_id,
                    method_id=method_id,
                    source_id=source_id,
                )
                output_payload = _encode_png(baseline.pixels)
                _publish_identical(
                    _safe_relative(root, output_relative, kind="semantic output"),
                    output_payload,
                    kind="semantic baseline output",
                )
                record_core = {
                    "schema_version": 2,
                    "record_type": "semantic-scientific-result",
                    "tuple_key": tuple_key,
                    "semantic_experiment_id": semantic["semantic_experiment_id"],
                    "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
                    "source_id": source_id,
                    "source_group_id": source["source_group_id"],
                    "source_sha256": source["source_sha256"],
                    "input_relative_path": input_record["relative_path"],
                    "input_encoded_sha256": input_record["encoded_sha256"],
                    "input_pixel_sha256": input_record["pixel_sha256"],
                    "condition_id": condition_id,
                    "method_id": method_id,
                    "degradation_control_sha256": semantic["controls"]["degradation_sha256"],
                    "evaluation_control_sha256": semantic["controls"]["evaluation_sha256"],
                    "master_seed": semantic["controls"]["degradation_seed"],
                    "degradation_trace": copy.deepcopy(degraded.trace),
                    "trace_relative_path": trace_relative,
                    "lr_relative_path": lr_relative,
                    "lr_encoded_sha256": hashlib.sha256(degraded.encoded_bytes).hexdigest(),
                    "lr_pixel_sha256": _degradation_pixel_sha256(degraded.pixels),
                    "baseline_evidence": copy.deepcopy(baseline.evidence),
                    "output_relative_path": output_relative,
                    "output_encoded_sha256": hashlib.sha256(output_payload).hexdigest(),
                    "output_pixel_sha256": pixel_sha256(baseline.pixels),
                    "status": "succeeded",
                    "claim_boundary": semantic["claim_boundary"],
                    "non_claims": copy.deepcopy(semantic["non_claims"]),
                }
                scientific_sha256 = canonical_sha256(record_core)
                record = {
                    **record_core,
                    "scientific_result_id": f"semantic-result-{scientific_sha256}",
                    "scientific_sha256": scientific_sha256,
                }
                record_relative = semantic["paths"]["records"].format(
                    condition_id=condition_id,
                    method_id=method_id,
                    source_id=source_id,
                )
                _publish_identical(
                    _safe_relative(root, record_relative, kind="semantic record"),
                    _canonical_json(record),
                    kind="semantic scientific record",
                )
                terminal.append(tuple_key)
    if terminal != semantic["expected_tuple_keys"]:
        raise FixtureReviewContractError("semantic execution terminal order differs")
    return {
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "expected_tuple_count": 36,
        "terminal_tuple_count": len(terminal),
    }


def reconcile_semantic_fixture_experiment(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Fail closed unless the separate stream contains exactly 36 valid terminal tuples."""

    project_root = Path(project_root).resolve()
    semantic = load_semantic_fixture_control(project_root)
    root = _semantic_artifact_root(project_root, artifact_root)
    manifest = _load_semantic_manifest(root)
    expected_record_paths = {
        semantic["paths"]["records"].format(
            source_id=source_id, condition_id=condition_id, method_id=method_id
        )
        for source_id in semantic["source_order"]
        for condition_id in semantic["condition_order"]
        for method_id in semantic["method_order"]
    }
    expected_output_paths = {
        semantic["paths"]["outputs"].format(
            source_id=source_id, condition_id=condition_id, method_id=method_id
        )
        for source_id in semantic["source_order"]
        for condition_id in semantic["condition_order"]
        for method_id in semantic["method_order"]
    }
    expected_trace_paths = {
        semantic["paths"]["traces"].format(source_id=source_id, condition_id=condition_id)
        for source_id in semantic["source_order"]
        for condition_id in semantic["condition_order"]
    }
    for directory_name, expected in (
        ("records", expected_record_paths),
        ("outputs", expected_output_paths),
        ("traces", expected_trace_paths),
    ):
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise FixtureReviewContractError(f"semantic {directory_name} directory is unsafe")
        observed: set[str] = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise FixtureReviewContractError(f"semantic {directory_name} contains a symlink")
            if path.is_file():
                observed.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise FixtureReviewContractError(
                    f"semantic {directory_name} contains a non-regular entry"
                )
        if observed != expected:
            raise FixtureReviewContractError(
                f"semantic {directory_name} has an unexpected, missing, or partial tuple"
            )
    lr_paths: set[str] = set()
    lr_root = root / "lr"
    if not lr_root.is_dir() or lr_root.is_symlink():
        raise FixtureReviewContractError("semantic LR directory is unsafe")
    for path in lr_root.rglob("*"):
        if path.is_symlink():
            raise FixtureReviewContractError("semantic LR contains a symlink")
        if path.is_file():
            lr_paths.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise FixtureReviewContractError("semantic LR contains a non-regular entry")
    if len(lr_paths) != 12:
        raise FixtureReviewContractError("semantic LR denominator differs")

    input_records = {row["source_id"]: row for row in manifest["inputs"]}
    scientific_rows: list[dict[str, str]] = []
    observed_keys: list[str] = []
    expected_lr_paths: set[str] = set()
    for tuple_key in semantic["expected_tuple_keys"]:
        source_id, condition_id, method_id = tuple_key.split("|")
        record_relative = semantic["paths"]["records"].format(
            source_id=source_id, condition_id=condition_id, method_id=method_id
        )
        record = _read_json(root / record_relative, kind="semantic scientific record")
        record_core = {
            key: value
            for key, value in record.items()
            if key not in {"scientific_result_id", "scientific_sha256"}
        }
        digest = canonical_sha256(record_core)
        if (
            record.get("tuple_key") != tuple_key
            or record.get("scientific_sha256") != digest
            or record.get("scientific_result_id") != f"semantic-result-{digest}"
            or record.get("semantic_experiment_sha256") != semantic["semantic_experiment_sha256"]
            or record.get("status") != "succeeded"
            or record.get("non_claims") != semantic["non_claims"]
        ):
            raise FixtureReviewContractError("semantic scientific identity differs")
        trace = record["degradation_trace"]
        trace_core = {key: value for key, value in trace.items() if key != "trace_id"}
        if (
            trace.get("trace_id") != f"degradation-{canonical_sha256(trace_core)}"
            or trace.get("control_sha256") != semantic["controls"]["degradation_sha256"]
            or trace.get("master_seed") != semantic["controls"]["degradation_seed"]
        ):
            raise FixtureReviewContractError("semantic degradation trace identity differs")
        trace_file = _read_json(root / record["trace_relative_path"], kind="semantic trace")
        if trace_file != trace:
            raise FixtureReviewContractError("semantic shared trace differs")
        lr_payload = _read_regular(
            root / record["lr_relative_path"],
            maximum_bytes=semantic["limits"]["max_output_bytes"],
            kind="semantic LR",
        )
        if hashlib.sha256(lr_payload).hexdigest() != record["lr_encoded_sha256"]:
            raise FixtureReviewContractError("semantic LR encoded digest differs")
        lr = _decode_rgb(
            root / record["lr_relative_path"],
            encoded_sha256=record["lr_encoded_sha256"],
            pixel_digest=None,
            kind="semantic LR",
        )
        if _degradation_pixel_sha256(lr) != record["lr_pixel_sha256"] or record[
            "baseline_evidence"
        ]["input_pixel_sha256"] != pixel_sha256(lr):
            raise FixtureReviewContractError("semantic LR pixel lineage differs")
        expected_lr_paths.add(record["lr_relative_path"])
        output = _decode_rgb(
            root / record["output_relative_path"],
            encoded_sha256=record["output_encoded_sha256"],
            pixel_digest=record["output_pixel_sha256"],
            kind="semantic output",
        )
        if record["baseline_evidence"]["output_pixel_sha256"] != pixel_sha256(output):
            raise FixtureReviewContractError("semantic output baseline lineage differs")
        input_record = input_records[source_id]
        if (
            record["source_sha256"]
            != next(
                row["source_sha256"] for row in semantic["sources"] if row["source_id"] == source_id
            )
            or record["input_encoded_sha256"] != input_record["encoded_sha256"]
            or record["input_pixel_sha256"] != input_record["pixel_sha256"]
        ):
            raise FixtureReviewContractError("semantic source lineage differs")
        observed_keys.append(tuple_key)
        scientific_rows.append(
            {
                "tuple_key": tuple_key,
                "scientific_sha256": digest,
                "output_encoded_sha256": record["output_encoded_sha256"],
                "output_pixel_sha256": record["output_pixel_sha256"],
                "lr_encoded_sha256": record["lr_encoded_sha256"],
                "lr_pixel_sha256": record["lr_pixel_sha256"],
                "trace_id": trace["trace_id"],
            }
        )
    if observed_keys != semantic["expected_tuple_keys"] or lr_paths != expected_lr_paths:
        raise FixtureReviewContractError("semantic closed tuple matrix differs")
    current_primitive = _artifact_inventory(
        project_root / "artifacts/phase2-fixture", kind="primitive evidence"
    )
    if (
        current_primitive != manifest["primitive_inventory"]
        or canonical_sha256(current_primitive) != manifest["primitive_inventory_sha256"]
    ):
        raise FixtureReviewContractError("primitive evidence changed during semantic execution")
    fixed = validate_fixture_review_inputs(project_root)
    if manifest["primitive_fixed_identities"] != {
        "experiment_id": fixed["experiment_id"],
        "reconciliation_id": fixed["reconciliation_id"],
        "reconciliation_sha256": fixed["reconciliation_sha256"],
        "replay_id": fixed["replay_id"],
        "replay_sha256": fixed["replay_sha256"],
        "aggregate_file_sha256": fixed["aggregate_file_sha256"],
        "membership_sha256": fixed["membership_sha256"],
    }:
        raise FixtureReviewContractError("primitive fixed identities changed")
    report_core = {
        "schema_version": 2,
        "record_type": "semantic-reconciliation",
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "applicability_sha256": manifest["applicability_sha256"],
        "expected_tuple_count": 36,
        "terminal_tuple_count": 36,
        "counts": {"succeeded": 36, "failed": 0, "excluded": 0},
        "tuples": scientific_rows,
        "primitive_inventory_sha256_before": manifest["primitive_inventory_sha256"],
        "primitive_inventory_sha256_after": canonical_sha256(current_primitive),
        "primitive_fixed_identities": copy.deepcopy(manifest["primitive_fixed_identities"]),
        "claim_boundary": semantic["claim_boundary"],
        "non_claims": copy.deepcopy(semantic["non_claims"]),
    }
    report = {**report_core, "reconciliation_sha256": canonical_sha256(report_core)}
    _publish_identical(
        root / semantic["paths"]["reconciliation"],
        _canonical_json(report),
        kind="semantic reconciliation",
    )
    return report


def _semantic_method_mapping(panel_id: str, semantic_sha256: str) -> dict[str, str]:
    ranked = sorted(
        EXPECTED_METHODS,
        key=lambda method_id: canonical_sha256(
            {
                "domain": "phase2-semantic-masked-method-v1",
                "semantic_experiment_sha256": semantic_sha256,
                "panel_id": panel_id,
                "method_id": method_id,
            }
        ),
    )
    return dict(zip(MASK_LABELS, ranked, strict=True))


def _render_semantic_panel(
    panel: Mapping[str, Any],
    mapping: Mapping[str, str],
    reference: np.ndarray,
    lr: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> tuple[bytes, dict[str, Any]]:
    scale = int(str(panel["condition_id"])[1])
    roi = panel["roi"]
    hr_roi, corresponding_lr, roi_evidence = native_physical_review_rois(
        reference, lr, roi=roi, scale=scale
    )
    method_rois = {
        label: outputs[method][
            roi["y"] : roi["y"] + roi["height"],
            roi["x"] : roi["x"] + roi["width"],
        ]
        for label, method in mapping.items()
    }
    columns = [
        ("HR", reference, hr_roi, f"HR {reference.shape[1]}x{reference.shape[0]} | fixed ROI"),
        (
            "LR",
            lr,
            corresponding_lr,
            f"native LR {lr.shape[1]}x{lr.shape[0]} | exact nearest corresponding ROI",
        ),
        *[
            (label, outputs[method], method_rois[label], f"masked {label} | fixed HR-sized ROI")
            for label, method in mapping.items()
        ],
    ]
    column_width, context_height, roi_height, gutter = 310, 215, 250, 12
    header_height, label_height = 54, 54
    width = len(columns) * column_width + (len(columns) + 1) * gutter
    height = header_height + label_height + context_height + label_height + roi_height + 3 * gutter
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{panel['panel_id']} | {panel['condition_id']} | {panel['source_id']} | methods masked",
        (gutter, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for index, (label, context, crop, description) in enumerate(columns):
        left = gutter + index * (column_width + gutter)
        cv2.putText(
            canvas,
            label,
            (left, header_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            description[:48],
            (left, header_height + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        context_top = header_height + label_height
        canvas[context_top : context_top + context_height, left : left + column_width] = _fit(
            context, column_width, context_height, allow_upscale=label != "LR"
        )
        cv2.putText(
            canvas,
            "native/corresponding ROI" if label == "LR" else "same fixed ROI",
            (left, context_top + context_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        roi_top = context_top + context_height + label_height
        canvas[roi_top : roi_top + roi_height, left : left + column_width] = _fit(
            crop, column_width, roi_height, allow_upscale=True
        )
    return _encode_png(canvas), roi_evidence


def prepare_semantic_review(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Publish the exact masked twelve-panel review without human evidence."""

    project_root = Path(project_root).resolve()
    semantic = load_semantic_fixture_control(project_root)
    root = _semantic_artifact_root(project_root, artifact_root)
    manifest = _load_semantic_manifest(root)
    reconciliation = reconcile_semantic_fixture_experiment(project_root, artifact_root=root)
    if reconciliation.get("terminal_tuple_count") != 36:
        raise FixtureReviewContractError("semantic reconciliation is incomplete")
    records = {
        row["tuple_key"]: _read_json(
            root
            / semantic["paths"]["records"].format(
                source_id=row["tuple_key"].split("|")[0],
                condition_id=row["tuple_key"].split("|")[1],
                method_id=row["tuple_key"].split("|")[2],
            ),
            kind="semantic scientific record",
        )
        for row in reconciliation["tuples"]
    }
    inputs = {row["source_id"]: row for row in manifest["inputs"]}
    mapping_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    for membership in semantic["review_membership"]:
        panel_id = membership["panel_id"]
        source_id = membership["source_id"]
        condition_id = membership["condition_id"]
        source = next(row for row in semantic["sources"] if row["source_id"] == source_id)
        input_record = inputs[source_id]
        reference_full = _decode_rgb(
            root / input_record["relative_path"],
            encoded_sha256=input_record["encoded_sha256"],
            pixel_digest=None,
            kind="semantic HR input",
        )
        aligned = align_reference(reference_full, int(condition_id[1])).pixels
        method_records = {
            method_id: records[f"{source_id}|{condition_id}|{method_id}"]
            for method_id in semantic["method_order"]
        }
        first_record = method_records[semantic["method_order"][0]]
        lr = _decode_rgb(
            root / first_record["lr_relative_path"],
            encoded_sha256=first_record["lr_encoded_sha256"],
            pixel_digest=None,
            kind="semantic LR",
        )
        outputs = {
            method_id: _decode_rgb(
                root / record["output_relative_path"],
                encoded_sha256=record["output_encoded_sha256"],
                pixel_digest=record["output_pixel_sha256"],
                kind="semantic output",
            )
            for method_id, record in method_records.items()
        }
        mapping = _semantic_method_mapping(panel_id, semantic["semantic_experiment_sha256"])
        panel = {
            "panel_id": panel_id,
            "source_id": source_id,
            "source_group_id": source["source_group_id"],
            "condition_id": condition_id,
            "roi": copy.deepcopy(source["roi"]),
        }
        encoded, roi_evidence = _render_semantic_panel(panel, mapping, aligned, lr, outputs)
        relative = f"review/panels/{panel_id}.png"
        _publish_identical(
            _safe_relative(root, relative, kind="semantic review panel"),
            encoded,
            kind="semantic review panel",
        )
        tuple_bindings = [
            {
                "masked_label": label,
                "tuple_key": f"{source_id}|{condition_id}|{method_id}",
                "scientific_sha256": method_records[method_id]["scientific_sha256"],
                "output_pixel_sha256": method_records[method_id]["output_pixel_sha256"],
            }
            for label, method_id in mapping.items()
        ]
        panel_rows.append(
            {
                **panel,
                "relative_path": relative,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "input_encoded_sha256": input_record["encoded_sha256"],
                "input_pixel_sha256": input_record["pixel_sha256"],
                "lr_relative_path": first_record["lr_relative_path"],
                "lr_encoded_sha256": first_record["lr_encoded_sha256"],
                "lr_pixel_sha256": first_record["lr_pixel_sha256"],
                "roi_evidence": roi_evidence,
                "tuple_bindings": tuple_bindings,
                "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
                "applicability_sha256": manifest["applicability_sha256"],
                "reconciliation_sha256": reconciliation["reconciliation_sha256"],
            }
        )
        mapping_rows.append({"panel_id": panel_id, "masked_methods": mapping})
    membership_core = {
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "review_membership": copy.deepcopy(semantic["review_membership"]),
    }
    membership_sha256 = canonical_sha256(membership_core)
    mapping_core = {
        "schema_version": 2,
        "record_type": "semantic-method-mapping",
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "membership_sha256": membership_sha256,
        "mapping": mapping_rows,
    }
    mapping = {**mapping_core, "mapping_sha256": canonical_sha256(mapping_core)}
    _publish_identical(
        root / "review/method-mapping.json",
        _canonical_json(mapping),
        kind="semantic method mapping",
    )
    static_identity = {
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "applicability_sha256": manifest["applicability_sha256"],
        "reconciliation_sha256": reconciliation["reconciliation_sha256"],
        "mapping_sha256": mapping["mapping_sha256"],
        "panel_sha256s": [row["sha256"] for row in panel_rows],
    }
    return {
        "artifact_root": root,
        "semantic_experiment_id": semantic["semantic_experiment_id"],
        "semantic_experiment_sha256": semantic["semantic_experiment_sha256"],
        "applicability_sha256": manifest["applicability_sha256"],
        "reconciliation_sha256": reconciliation["reconciliation_sha256"],
        "membership_id": f"semantic-membership-{membership_sha256}",
        "membership_sha256": membership_sha256,
        "mapping": mapping_rows,
        "mapping_sha256": mapping["mapping_sha256"],
        "panels": panel_rows,
        "requested_panel_count": 12,
        "working_copy_token": f"generated:{canonical_sha256(static_identity)}",
        "claim_boundary": semantic["claim_boundary"],
        "non_claims": copy.deepcopy(semantic["non_claims"]),
    }


def semantic_notebook_source_sha256(path: Path) -> str:
    """Return the normalized digest of the source-only semantic review notebook."""

    notebook = _read_json(
        Path(path), kind="semantic tracked notebook", maximum_bytes=4 * 1024 * 1024
    )
    if not isinstance(notebook.get("cells"), list) or not isinstance(
        notebook.get("metadata"), dict
    ):
        raise FixtureReviewContractError("semantic tracked notebook structure is invalid")
    serialized = json.dumps(notebook, sort_keys=True).casefold()
    if any(
        value in serialized for value in ("image/png", "image/jpeg", "application/pdf", "base64,")
    ):
        raise FixtureReviewContractError("semantic tracked notebook contains embedded payload")
    marker_count = 0
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or not isinstance(cell.get("metadata"), dict):
            raise FixtureReviewContractError("semantic tracked notebook cell is invalid")
        source = cell.get("source")
        if not isinstance(source, (str, list)) or (
            isinstance(source, list) and not all(isinstance(value, str) for value in source)
        ):
            raise FixtureReviewContractError("semantic tracked notebook source is invalid")
        marker_count += (source if isinstance(source, str) else "".join(source)).count(
            SEMANTIC_WORKING_COPY_MARKER
        )
        if cell.get("cell_type") == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs") != []
        ):
            raise FixtureReviewContractError(
                "semantic tracked notebook must be unexecuted and output-free"
            )
    if marker_count != 1:
        raise FixtureReviewContractError("semantic working-copy guard is absent or ambiguous")
    return canonical_sha256(notebook)


def _semantic_logical_working_sha256(working_path: Path, source_path: Path, token: str) -> str:
    working = _read_json(
        working_path, kind="semantic working notebook", maximum_bytes=16 * 1024 * 1024
    )
    source = _read_json(
        source_path, kind="semantic tracked notebook", maximum_bytes=4 * 1024 * 1024
    )

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
            raise FixtureReviewContractError("semantic working notebook kernelspec differs")
        if kernelspec.get("language") != "python" or kernelspec.get("display_name") not in {
            "Python 3 (score-super-resolution)",
            "score-super-resolution (3.12.12)",
        }:
            raise FixtureReviewContractError("semantic working notebook kernel identity differs")
        kernelspec["display_name"] = "score-super-resolution (3.12.12)"
        replacements = 0
        for cell in value["cells"]:
            cell.get("metadata", {}).pop("execution", None)
            for field in ("collapsed", "scrolled", "trusted"):
                cell.get("metadata", {}).pop(field, None)
            cell_source = cell["source"]
            joined = cell_source if isinstance(cell_source, str) else "".join(cell_source)
            expected = token if working_copy else SEMANTIC_WORKING_COPY_MARKER
            replacements += joined.count(expected)
            joined = joined.replace(expected, SEMANTIC_WORKING_COPY_MARKER)
            cell["source"] = (
                joined if isinstance(cell_source, str) else joined.splitlines(keepends=True)
            )
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        if replacements != 1:
            raise FixtureReviewContractError("semantic working notebook token differs")
        return value

    working_projection = projection(working, working_copy=True)
    source_projection = projection(source, working_copy=False)
    if working_projection != source_projection:
        raise FixtureReviewContractError("semantic working notebook differs from tracked source")
    return canonical_sha256(
        {"domain": "phase2-semantic-review-logical-notebook-v1", "notebook": working_projection}
    )


def execute_semantic_review_notebook(project_root: Path) -> dict[str, Any]:
    """Execute the clean semantic notebook out of place without fabricating human evidence."""

    project_root = Path(project_root).resolve()
    prepared = prepare_semantic_review(project_root)
    root = Path(prepared["artifact_root"])
    source_path = project_root / "notebooks/02-semantic-fixture-baseline-review.ipynb"
    source_sha256 = semantic_notebook_source_sha256(source_path)
    notebook = nbformat.read(source_path, as_version=4)
    replacements = 0
    for cell in notebook.cells:
        if SEMANTIC_WORKING_COPY_MARKER in cell.source:
            cell.source = cell.source.replace(
                SEMANTIC_WORKING_COPY_MARKER, prepared["working_copy_token"]
            )
            replacements += 1
    if replacements != 1:
        raise FixtureReviewContractError("semantic tracked notebook guard is ambiguous")
    try:
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(project_root)}},
        ).execute()
    except Exception as error:
        raise FixtureReviewContractError("semantic review notebook execution failed") from error
    for cell in notebook.cells:
        if isinstance(cell.get("metadata"), dict):
            cell.metadata.pop("execution", None)
    working_payload = nbformat.writes(notebook).encode("utf-8")
    working_path = root / "review/semantic-fixture-baseline-review-working.ipynb"
    if working_path.exists():
        existing_logical = _semantic_logical_working_sha256(
            working_path, source_path, prepared["working_copy_token"]
        )
        temporary = working_path.with_name(f".{working_path.name}.tmp-{uuid4().hex}")
        try:
            _write_new(temporary, working_payload)
            replacement_logical = _semantic_logical_working_sha256(
                temporary, source_path, prepared["working_copy_token"]
            )
            if replacement_logical != existing_logical:
                raise FixtureReviewContractError("semantic working notebook logical source changed")
            os.replace(temporary, working_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        _publish_identical(working_path, working_payload, kind="semantic working notebook")
    logical_sha256 = _semantic_logical_working_sha256(
        working_path, source_path, prepared["working_copy_token"]
    )
    session_core = {
        "schema_version": 2,
        "record_type": "semantic-review-session",
        "semantic_experiment_id": prepared["semantic_experiment_id"],
        "semantic_experiment_sha256": prepared["semantic_experiment_sha256"],
        "applicability_sha256": prepared["applicability_sha256"],
        "reconciliation_sha256": prepared["reconciliation_sha256"],
        "notebook_source_sha256": source_sha256,
        "working_notebook_logical_sha256": logical_sha256,
        "working_copy_token": prepared["working_copy_token"],
        "mapping_sha256": prepared["mapping_sha256"],
        "panels": prepared["panels"],
        "requested_panel_count": 12,
        "displayable_panel_count": 12,
        "failed_panel_count": 0,
        "confirmation_relative_path": "review/source-confirmations.json",
        "review_relative_path": "review/notation-review.json",
    }
    session = {
        **session_core,
        "session_id": f"semantic-session-{canonical_sha256(session_core)}",
    }
    try:
        validate_instance("semantic-review-session", session, version=2)
    except ContractValidationError as error:
        raise FixtureReviewContractError("semantic review session fails its schema") from error
    _publish_identical(
        root / "review/review-session-manifest.json",
        _canonical_json(session),
        kind="semantic review session",
    )
    if semantic_notebook_source_sha256(source_path) != source_sha256:
        raise FixtureReviewContractError("semantic tracked notebook changed during execution")
    for forbidden in (
        root / "review/source-confirmations.json",
        root / "review/notation-review.json",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise FixtureReviewContractError("semantic notebook fabricated human evidence")
    return session


class SemanticFixtureReviewSession:
    """Content-bound review of coherent notation with genuine source confirmations."""

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
                and (candidate / SEMANTIC_CONFIG_RELATIVE).is_file()
            ),
            None,
        )
        if self.project_root is None:
            raise FixtureReviewContractError("could not locate the proyecto root")
        if (
            not isinstance(working_copy_token, str)
            or _GENERATED_TOKEN.fullmatch(working_copy_token) is None
        ):
            raise FixtureReviewContractError("execute only the generated semantic working notebook")
        self.prepared = prepare_semantic_review(self.project_root, artifact_root=artifact_root)
        if working_copy_token != self.prepared["working_copy_token"]:
            raise FixtureReviewContractError("semantic working token does not bind fixed evidence")
        self.working_copy_token = working_copy_token
        self.artifact_root = Path(self.prepared["artifact_root"])
        self.manifest_path = self.artifact_root / "review/review-session-manifest.json"
        self.confirmation_path = self.artifact_root / "review/source-confirmations.json"
        self.review_path = self.artifact_root / "review/notation-review.json"
        self.manifest = None
        if self.manifest_path.exists():
            self.manifest = _read_json(self.manifest_path, kind="semantic review session")
            try:
                validate_instance("semantic-review-session", self.manifest, version=2)
            except ContractValidationError as error:
                raise FixtureReviewContractError(
                    "semantic review session fails its schema"
                ) from error
            session_core = {
                key: value for key, value in self.manifest.items() if key != "session_id"
            }
            if (
                self.manifest["session_id"] != f"semantic-session-{canonical_sha256(session_core)}"
                or self.manifest["semantic_experiment_sha256"]
                != self.prepared["semantic_experiment_sha256"]
                or self.manifest["applicability_sha256"] != self.prepared["applicability_sha256"]
                or self.manifest["reconciliation_sha256"] != self.prepared["reconciliation_sha256"]
                or self.manifest["working_copy_token"] != working_copy_token
                or self.manifest["panels"] != self.prepared["panels"]
                or self.manifest["mapping_sha256"] != self.prepared["mapping_sha256"]
            ):
                raise FixtureReviewContractError("semantic review session binding differs")
        self.reload()

    def _require_manifest(self) -> dict[str, Any]:
        if self.manifest is None and self.manifest_path.exists():
            self.manifest = _read_json(self.manifest_path, kind="semantic review session")
        if self.manifest is None:
            raise FixtureReviewContractError("semantic session finalizes after notebook generation")
        source_path = self.project_root / "notebooks/02-semantic-fixture-baseline-review.ipynb"
        working_path = self.artifact_root / "review/semantic-fixture-baseline-review-working.ipynb"
        if semantic_notebook_source_sha256(source_path) != self.manifest["notebook_source_sha256"]:
            raise FixtureReviewContractError("semantic tracked notebook source changed")
        if (
            _semantic_logical_working_sha256(working_path, source_path, self.working_copy_token)
            != self.manifest["working_notebook_logical_sha256"]
        ):
            raise FixtureReviewContractError("semantic ignored notebook source changed")
        return self.manifest

    def _load_confirmations(self) -> dict[str, Any] | None:
        if not self.confirmation_path.exists() and not self.confirmation_path.is_symlink():
            return None
        record = _read_json(self.confirmation_path, kind="semantic source confirmations")
        if record.get("confirmation_sha256") != _self_digest(record, "confirmation_sha256"):
            raise FixtureReviewContractError("semantic source confirmation digest differs")
        confirmations = record.get("confirmations")
        if (
            record.get("record_type") != "semantic-source-confirmations"
            or record.get("session_id") != self._require_manifest()["session_id"]
            or not isinstance(confirmations, list)
            or [row.get("source_id") for row in confirmations] != list(SEMANTIC_SOURCE_IDS)
            or any(row.get("confirmed") is not True for row in confirmations)
            or any(
                not isinstance(row.get("reviewer"), str)
                or not row["reviewer"].strip()
                or not isinstance(row.get("rationale"), str)
                or not row["rationale"].strip()
                or not isinstance(row.get("confirmed_at"), str)
                or not row["confirmed_at"].endswith("Z")
                for row in confirmations
            )
        ):
            raise FixtureReviewContractError("semantic source confirmations differ")
        return record

    def _load_semantic_review(self) -> dict[str, Any] | None:
        if not self.review_path.exists() and not self.review_path.is_symlink():
            return None
        bundle = _read_json(self.review_path, kind="semantic notation review")
        if bundle.get("review_sha256") != _self_digest(bundle, "review_sha256"):
            raise FixtureReviewContractError("semantic notation review digest differs")
        if (
            bundle.get("record_type") != "semantic-notation-review-bundle"
            or bundle.get("session_id") != self._require_manifest()["session_id"]
            or bundle.get("membership_id") != self.prepared["membership_id"]
            or bundle.get("membership_sha256") != self.prepared["membership_sha256"]
        ):
            raise FixtureReviewContractError("semantic notation review lineage differs")
        reviews = bundle.get("reviews")
        panel_ids = [row["panel_id"] for row in self.prepared["panels"]]
        if not isinstance(reviews, list) or [row.get("panel_id") for row in reviews] != [
            panel_id
            for panel_id in panel_ids
            if panel_id in {row.get("panel_id") for row in reviews}
        ]:
            raise FixtureReviewContractError("semantic notation review panel order differs")
        for row in reviews:
            try:
                validate_notation_review(row)
            except (ContractValidationError, ValueError) as error:
                raise FixtureReviewContractError(
                    "semantic notation review row is invalid"
                ) from error
            if (
                row["sample_membership_id"] != self.prepared["membership_id"]
                or row["sample_sha256"] != self.prepared["membership_sha256"]
            ):
                raise FixtureReviewContractError("semantic notation review sample differs")
        if bundle.get("denominators") != {
            "requested_panels": 12,
            "displayable_panels": 12,
            "reviewed_panels": len(reviews),
            "skipped_panels": 0,
            "failed_panels": 0,
        }:
            raise FixtureReviewContractError("semantic notation review denominators differ")
        return bundle

    def reload(self) -> None:
        self.confirmations = self._load_confirmations()
        self.review = self._load_semantic_review()
        self.expected_confirmation_sha256 = _review_digest(self.confirmation_path)
        self.expected_review_sha256 = _review_digest(self.review_path)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_role": "authored-coherent-notation-applicability-only",
            "semantic_experiment_id": self.prepared["semantic_experiment_id"],
            "confirmed_sources": (
                len(self.confirmations["confirmations"]) if self.confirmations is not None else 0
            ),
            "requested_panels": 12,
            "displayable_panels": 12,
            "reviewed_panels": len(self.review["reviews"]) if self.review is not None else 0,
            "skipped_panels": 0,
            "failed_panels": 0,
            "methods_masked": True,
            "claim_boundary": self.prepared["claim_boundary"],
        }

    def save_source_confirmations(
        self, *, reviewer: str, rationales: Mapping[str, str], confirmed_sources: Sequence[str]
    ) -> str:
        manifest = self._require_manifest()
        reviewer_value = reviewer.strip()
        if (
            tuple(confirmed_sources) != SEMANTIC_SOURCE_IDS
            or not reviewer_value
            or len(reviewer_value) > 200
            or any(ord(character) < 32 for character in reviewer_value)
        ):
            raise FixtureReviewContractError(
                "both semantic HR sources require genuine confirmation"
            )
        rows = []
        reviewed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for source_id in SEMANTIC_SOURCE_IDS:
            rationale = str(rationales.get(source_id, "")).strip()
            if (
                not rationale
                or len(rationale) > 2000
                or any(ord(character) < 32 and character not in "\n\t" for character in rationale)
            ):
                raise FixtureReviewContractError("each semantic HR source requires a rationale")
            rows.append(
                {
                    "source_id": source_id,
                    "confirmed": True,
                    "reviewer": reviewer_value,
                    "confirmed_at": reviewed_at,
                    "rationale": rationale,
                }
            )
        core = {
            "schema_version": 2,
            "record_type": "semantic-source-confirmations",
            "session_id": manifest["session_id"],
            "semantic_experiment_sha256": self.prepared["semantic_experiment_sha256"],
            "applicability_sha256": self.prepared["applicability_sha256"],
            "confirmations": rows,
        }
        record = {**core, "confirmation_sha256": canonical_sha256(core)}
        self.expected_confirmation_sha256 = _durable_cas_json(
            self.confirmation_path,
            _canonical_json(record),
            expected_sha256=self.expected_confirmation_sha256,
        )
        self.confirmations = record
        return self.expected_confirmation_sha256

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
        if self._load_confirmations() is None:
            raise FixtureReviewContractError(
                "confirm both coherent HR sources before reviewing panels"
            )
        panel_ids = [row["panel_id"] for row in self.prepared["panels"]]
        if panel_id not in panel_ids:
            raise FixtureReviewContractError("semantic review panel is outside fixed membership")
        payload = {
            "schema_version": 2,
            "record_type": "notation-review",
            "sample_membership_id": self.prepared["membership_id"],
            "sample_sha256": self.prepared["membership_sha256"],
            "panel_id": panel_id,
            "reviewer": reviewer.strip(),
            "reviewed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "labels": list(labels),
            "severity": severity,
            "rationale": rationale.strip(),
        }
        row = {**payload, "review_id": f"review-{canonical_sha256(payload)}"}
        try:
            row = validate_notation_review(row)
        except (ContractValidationError, ValueError) as error:
            raise FixtureReviewContractError("semantic notation review input is invalid") from error
        rows_by_id = {
            item["panel_id"]: item for item in (self.review["reviews"] if self.review else [])
        }
        rows_by_id[panel_id] = row
        rows = [rows_by_id[value] for value in panel_ids if value in rows_by_id]
        core = {
            "schema_version": 2,
            "record_type": "semantic-notation-review-bundle",
            "session_id": manifest["session_id"],
            "semantic_experiment_sha256": self.prepared["semantic_experiment_sha256"],
            "applicability_sha256": self.prepared["applicability_sha256"],
            "reconciliation_sha256": self.prepared["reconciliation_sha256"],
            "mapping_sha256": self.prepared["mapping_sha256"],
            "membership_id": self.prepared["membership_id"],
            "membership_sha256": self.prepared["membership_sha256"],
            "confirmation_sha256": self.confirmations["confirmation_sha256"],
            "reviews": rows,
            "denominators": {
                "requested_panels": 12,
                "displayable_panels": 12,
                "reviewed_panels": len(rows),
                "skipped_panels": 0,
                "failed_panels": 0,
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

    def source_confirmation_widget(self) -> widgets.Widget:
        return FixtureReviewSession._semantic_source_confirmation_widget_template(self)

    def panel_widget(self) -> widgets.Widget:
        return FixtureReviewSession._semantic_panel_widget_template(self)

    def review_widget(self) -> widgets.Widget:
        return FixtureReviewSession._semantic_review_widget_template(self)

    def progress_widget(self) -> widgets.Widget:
        return FixtureReviewSession._semantic_progress_widget_template(self)
