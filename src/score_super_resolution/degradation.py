"""Deterministic controlled degradation and project-authored fixture contracts.

This module deliberately has no dataset or network loader.  Its only accepted source material is
the closed project-authored fixture manifest used to review Phase 2 controls while SMB remains
``AUDITED_LOCKED``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cairosvg
import cv2
import numpy as np
import verovio
import yaml
from defusedxml import ElementTree

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.identities import canonical_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "configs/degradations/controlled-score-candidates.yaml"
DEFAULT_FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "tests/fixtures/phase2/fixture-manifest-v1.yaml"
DEFAULT_VISUAL_FIXTURE_MANIFEST_PATH = (
    PROJECT_ROOT / "tests/fixtures/phase2/visual-fixture-manifest-v2.yaml"
)
LEGACY_CANDIDATE_ID = "controlled-score-v1-candidate"
CURRENT_CANDIDATE_ID = "controlled-score-v2-candidate"
LEGACY_CANDIDATE_RAW_SHA256 = "cabc3ad9ff1564ff2d08808c42a8e34784bebe8ff47beabaa979a7a167548536"
LEGACY_CANDIDATE_CANONICAL_SHA256 = (
    "52cb18aa12de1a11791e7249f8086df3a25030b2e5c4c5ef7c29948c2e22f237"
)

EXPECTED_CONDITION_IDS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
_COMPOUND_ORDER = ("blur", "reduction", "noise", "clip-round", "jpeg")
_SECRET_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
_SECRET_NAMES = {"api_key", "apikey", "access_key", "private_key"}
_MAX_CONTROL_BYTES = 1_048_576
_MAX_MANIFEST_BYTES = 1_048_576
_MAX_REVIEW_BYTES = 1_048_576
_MAX_EVIDENCE_BYTES = 32 * 1_048_576


class DegradationContractError(ValueError):
    """A candidate registry is incomplete, unsafe, or scientifically inconsistent."""


class FixtureValidationError(ValueError):
    """A fixture manifest or generated fixture fails the authored-fixture contract."""


class NotebookSourceError(ValueError):
    """A tracked notebook contains state or payloads that belong only in ignored artifacts."""


class DegradationDecisionError(ValueError):
    """A degradation decision is absent, stale, inauthentic, or content-mismatched."""


@dataclass(frozen=True)
class DegradationControl:
    """One validated immutable candidate projection."""

    version: int
    candidate_id: str
    status: str
    claim_boundary: str
    master_seed: int
    image_contract: dict[str, Any]
    alignment: dict[str, Any]
    runtime: dict[str, Any]
    condition_ids: tuple[str, ...]
    conditions: tuple[dict[str, Any], ...]
    sha256: str


@dataclass(frozen=True)
class AlignedReference:
    """Canonical RGB8 reference and its lower/right divisibility crop evidence."""

    pixels: np.ndarray
    input_pixel_sha256: str
    aligned_pixel_sha256: str
    input_dimensions: tuple[int, int, int]
    aligned_dimensions: tuple[int, int, int]
    crop: dict[str, int]


@dataclass(frozen=True)
class DegradationResult:
    """A degraded RGB8 array plus encoded bytes and JSON-safe scientific trace."""

    pixels: np.ndarray
    encoded_bytes: bytes
    trace: dict[str, Any]


_PROJECT_ROOT_MARKERS = (
    Path("pyproject.toml"),
    Path("configs/degradations/controlled-score-candidates.yaml"),
    Path("notebooks/02-degradation-preview.ipynb"),
)


def _is_regular_non_symlink(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def _discover_project_root(start: Path) -> Path:
    """Find the nearest checked-out project root without trusting the process cwd as that root."""

    candidate = Path(start).absolute()
    if _is_regular_non_symlink(candidate):
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if all(_is_regular_non_symlink(directory / marker) for marker in _PROJECT_ROOT_MARKERS):
            return directory.resolve()
    raise DegradationDecisionError(
        "project root cannot be discovered from the notebook working directory"
    )


def _secret_like_key(path: tuple[str, ...], value: Any) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _SECRET_NAMES or set(normalized.split("_")) & _SECRET_PARTS:
                return ".".join((*path, key))
            found = _secret_like_key((*path, key), child)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _secret_like_key((*path, str(index)), child)
            if found is not None:
                return found
    return None


def _read_regular_yaml(path: Path, *, maximum_bytes: int, kind: str) -> dict[str, Any]:
    path = Path(path)
    try:
        if path.is_symlink():
            raise ValueError(f"{kind} path must not be a symlink")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{kind} path must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ValueError(f"{kind} exceeds the encoded byte limit")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(descriptor, min(65_536, maximum_bytes + 1 - total)):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(f"{kind} exceeds the encoded byte limit")
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{kind} changed while being read")
        loaded = yaml.safe_load(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"{kind} cannot be read safely") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{kind} root must be a mapping")
    return loaded


def _validate_registry_semantics(registry: Mapping[str, Any]) -> None:
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DegradationContractError("candidate registry must contain entries")
    versions = [
        candidate.get("version") for candidate in candidates if isinstance(candidate, Mapping)
    ]
    if len(versions) != len(candidates):
        raise DegradationContractError("candidate registry entries must be mappings")
    if len(set(versions)) != len(versions):
        raise DegradationContractError("candidate versions must be unique")
    if versions != list(range(1, len(versions) + 1)):
        raise DegradationContractError("candidate versions must be strictly monotonic from one")


def _validate_registry_bytes(raw: bytes, candidates: Sequence[Mapping[str, Any]]) -> None:
    first_marker = b"  - version: 1\n"
    second_marker = b"  - version: 2\n"
    try:
        first_start = raw.index(first_marker)
        first_end = raw.index(second_marker, first_start + len(first_marker))
    except ValueError as error:
        raise DegradationContractError(
            "candidate registry must contain the exact v1/v2 append"
        ) from error
    if hashlib.sha256(raw[first_start:first_end]).hexdigest() != LEGACY_CANDIDATE_RAW_SHA256:
        raise DegradationContractError("candidate 1 raw bytes differ from immutable history")
    if canonical_sha256(candidates[0]) != LEGACY_CANDIDATE_CANONICAL_SHA256:
        raise DegradationContractError(
            "candidate 1 canonical content differs from immutable history"
        )


def _validate_candidate_semantics(entry: Mapping[str, Any]) -> None:
    conditions = entry["conditions"]
    ids = tuple(condition["condition_id"] for condition in conditions)
    if ids != EXPECTED_CONDITION_IDS or tuple(entry["condition_order"]) != EXPECTED_CONDITION_IDS:
        raise DegradationContractError("condition IDs and order must equal the exact six-cell grid")
    for condition in conditions:
        condition_id = condition["condition_id"]
        scale_text, severity = condition_id.split("-", maxsplit=1)
        if condition["scale"] != int(scale_text[1:]) or condition["severity"] != severity:
            raise DegradationContractError(f"condition {condition_id} identity is inconsistent")
        if condition["reduction"] != {"interpolation": "INTER_AREA"}:
            raise DegradationContractError(f"condition {condition_id} reduction is not explicit")
        if severity == "clean":
            if (
                tuple(condition["operations"]) != ("reduction",)
                or condition["blur"] is not None
                or condition["noise"] is not None
                or condition["jpeg"] is not None
            ):
                raise DegradationContractError(f"condition {condition_id} clean operations differ")
            continue
        version = entry["version"]
        if version == 1:
            expected_by_severity = {
                "moderate": (
                    {"type": "gaussian", "sigma": 0.8, "kernel": 7},
                    {"type": "gaussian", "sigma": 3.0},
                    85,
                ),
                "strong": (
                    {"type": "gaussian", "sigma": 1.6, "kernel": 11},
                    {"type": "gaussian", "sigma": 8.0},
                    60,
                ),
            }
        elif version == 2:
            expected_by_severity = {
                "moderate": (
                    {"type": "gaussian", "sigma": 1.2, "kernel": 9},
                    {"type": "gaussian", "sigma": 5.0},
                    75,
                ),
                "strong": (
                    {"type": "gaussian", "sigma": 2.4, "kernel": 17},
                    {"type": "gaussian", "sigma": 14.0},
                    40,
                ),
            }
        else:
            raise DegradationContractError("candidate version has no implemented semantics")
        expected = expected_by_severity[severity]
        if tuple(condition["operations"]) != _COMPOUND_ORDER:
            raise DegradationContractError(f"condition {condition_id} operations order is invalid")
        if condition["blur"] != expected[0] or condition["noise"] != expected[1]:
            raise DegradationContractError(f"condition {condition_id} parameters are invalid")
        if condition["jpeg"]["quality"] != expected[2]:
            raise DegradationContractError(f"condition {condition_id} JPEG quality is invalid")


def load_degradation_control(
    path: Path = DEFAULT_CONTROL_PATH, *, version: int | None = None
) -> DegradationControl:
    """Load one explicitly versioned candidate after closed registry validation."""

    try:
        raw = _read_regular_bytes(
            path, maximum_bytes=_MAX_CONTROL_BYTES, kind="degradation control"
        )
        registry = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(registry, dict):
            raise DegradationContractError("degradation control root must be a mapping")
        secret = _secret_like_key((), registry)
        if secret is not None:
            raise DegradationContractError(f"secret-like key is forbidden: {secret}")
        _validate_registry_semantics(registry)
        validate_instance("degradation-control", registry, version=2)
        candidates = registry["candidates"]
        _validate_registry_bytes(raw, candidates)
        for index, candidate in enumerate(candidates):
            _validate_candidate_semantics(candidate)
            if index == 0:
                if "previous_candidate_sha256" in candidate:
                    raise DegradationContractError("first candidate must not declare a predecessor")
            elif candidate.get("previous_candidate_sha256") != canonical_sha256(
                candidates[index - 1]
            ):
                raise DegradationContractError(
                    "candidate predecessor digest detects prior mutation"
                )
        selected_version = len(candidates) if version is None else version
        if isinstance(selected_version, bool) or not isinstance(selected_version, int):
            raise DegradationContractError("candidate version must be an integer")
        matches = [entry for entry in candidates if entry["version"] == selected_version]
        if len(matches) != 1:
            raise DegradationContractError("candidate version is not uniquely declared")
        entry = copy.deepcopy(matches[0])
    except DegradationContractError:
        raise
    except (
        ContractValidationError,
        KeyError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise DegradationContractError(str(error)) from error
    return DegradationControl(
        version=entry["version"],
        candidate_id=entry["candidate_id"],
        status=entry["status"],
        claim_boundary=entry["claim_boundary"],
        master_seed=entry["master_seed"],
        image_contract=entry["image_contract"],
        alignment=entry["alignment"],
        runtime=entry["runtime"],
        condition_ids=tuple(entry["condition_order"]),
        conditions=tuple(entry["conditions"]),
        sha256=canonical_sha256(entry),
    )


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


def _validate_fixture_semantics(manifest: Mapping[str, Any]) -> None:
    limits = manifest["limits"]
    items = manifest["items"]
    if len(items) > limits["max_items"]:
        raise FixtureValidationError("fixture item count exceeds the declared bound")
    item_ids: set[str] = set()
    paths: set[str] = set()
    group_pages: dict[str, list[int]] = {}
    prior_key: tuple[str, int] | None = None
    for item in items:
        item_id = item["item_id"]
        relative = Path(item["relative_path"])
        if item_id in item_ids or item["relative_path"] in paths:
            raise FixtureValidationError("fixture IDs and paths must be unique")
        item_ids.add(item_id)
        paths.add(item["relative_path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise FixtureValidationError("fixture paths must be canonical relative paths")
        width, height = item["width"], item["height"]
        if width > limits["max_width"] or height > limits["max_height"]:
            raise FixtureValidationError("fixture dimensions exceed the declared bound")
        if width * height > limits["max_pixels"]:
            raise FixtureValidationError("fixture pixels exceed the declared bound")
        roi = item["roi"]
        if roi["x"] + roi["width"] > width or roi["y"] + roi["height"] > height:
            raise FixtureValidationError("fixture ROI escapes the image")
        group_page = (item["source_group_id"], item["page_number"])
        if prior_key is not None and group_page <= prior_key:
            raise FixtureValidationError("fixture source group must precede canonical page order")
        prior_key = group_page
        group_pages.setdefault(item["source_group_id"], []).append(item["page_number"])
        if item_id != f"{item['source_group_id']}-page-{item['page_number']:02d}":
            raise FixtureValidationError("fixture item identity must bind group before page")
    if len(group_pages) < 4 or any(pages != [1, 2] for pages in group_pages.values()):
        raise FixtureValidationError("fixtures require at least four groups with two pages each")


def _colour(value: Sequence[int]) -> tuple[int, int, int]:
    return int(value[0]), int(value[1]), int(value[2])


def _render_fixture(item: Mapping[str, Any]) -> np.ndarray:
    height, width = int(item["height"]), int(item["width"])
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = np.asarray(item["background"], dtype=np.uint8)
    for primitive in item["primitives"]:
        kind = primitive["type"]
        colour = _colour(primitive.get("colour", [0, 0, 0]))
        if kind == "gradient":
            start, end = primitive["value_start"], primitive["value_end"]
            gradient = np.rint(np.linspace(start, end, width, dtype=np.float64))[None, :, None]
            pixels = np.clip(pixels.astype(np.int16) + gradient.astype(np.int16), 0, 255).astype(
                np.uint8
            )
        elif kind == "staff":
            x, y = primitive["origin"]
            for offset in range(primitive["count"]):
                row = y + offset * primitive["spacing"]
                cv2.line(
                    pixels,
                    (x, row),
                    (x + primitive["length"], row),
                    colour,
                    primitive["thickness"],
                    cv2.LINE_8,
                )
        elif kind == "note":
            x, y = primitive["centre"]
            radius = primitive["radius"]
            cv2.ellipse(pixels, (x, y), (radius, max(2, radius - 2)), -18, 0, 360, colour, -1)
            if primitive["stem"] == "up":
                cv2.line(
                    pixels,
                    (x + radius - 1, y),
                    (x + radius - 1, y - radius * 5),
                    colour,
                    primitive["thickness"],
                    cv2.LINE_8,
                )
            elif primitive["stem"] == "down":
                cv2.line(
                    pixels,
                    (x - radius + 1, y),
                    (x - radius + 1, y + radius * 5),
                    colour,
                    primitive["thickness"],
                    cv2.LINE_8,
                )
        elif kind == "line":
            cv2.line(
                pixels,
                tuple(primitive["origin"]),
                tuple(primitive["end"]),
                colour,
                primitive["thickness"],
                cv2.LINE_8,
            )
        elif kind == "ellipse":
            cv2.ellipse(
                pixels,
                tuple(primitive["centre"]),
                tuple(primitive["axes"]),
                0,
                primitive["angle_start"],
                primitive["angle_end"],
                colour,
                primitive["thickness"],
                cv2.LINE_AA,
            )
        elif kind == "text":
            cv2.putText(
                pixels,
                primitive["text"],
                tuple(primitive["origin"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                primitive["scale"],
                colour,
                primitive["thickness"],
                cv2.LINE_AA,
            )
        elif kind == "impulse":
            x, y = primitive["centre"]
            pixels[y, x] = colour
        elif kind == "border":
            inset = primitive["thickness"] // 2
            cv2.rectangle(
                pixels,
                (inset, inset),
                (width - 1 - inset, height - 1 - inset),
                colour,
                primitive["thickness"],
                cv2.LINE_8,
            )
        else:  # pragma: no cover - schema closes the vocabulary
            raise FixtureValidationError(f"unsupported fixture primitive: {kind}")
    return np.ascontiguousarray(pixels)


def _encode_fixture_png(pixels: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    if not success:
        raise FixtureValidationError("fixture PNG encoding failed")
    return encoded.tobytes()


def _write_new_regular(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short fixture write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def generate_fixture_bundle(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    """Validate and materialize the small authored fixture bundle deterministically.

    All manifest, identity, bounds, and expected pixel hashes are checked before the output root is
    created. Existing files must be byte-identical; the function never overwrites them.
    """

    output_root = Path(output_root)
    try:
        manifest = _read_regular_yaml(
            manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES, kind="fixture manifest"
        )
        secret = _secret_like_key((), manifest)
        if secret is not None:
            raise FixtureValidationError(f"secret-like fixture metadata is forbidden: {secret}")
        serialized = yaml.safe_dump(manifest, sort_keys=True).casefold()
        if "praig/smb" in serialized or "evaluation_benchmark" in serialized:
            raise FixtureValidationError("SMB identity or role is forbidden in fixture manifests")
        validate_instance("fixture-manifest", manifest, version=2)
        _validate_fixture_semantics(manifest)
        prepared: list[tuple[Mapping[str, Any], np.ndarray, bytes, str, str]] = []
        for item in manifest["items"]:
            pixels = _render_fixture(item)
            pixel_sha256 = _fixture_pixel_sha256(pixels)
            if pixel_sha256 != item["generated_pixel_sha256"]:
                raise FixtureValidationError(
                    f"fixture generated pixel digest mismatch for {item['item_id']}"
                )
            encoded = _encode_fixture_png(pixels)
            if len(encoded) > manifest["limits"]["max_encoded_bytes"]:
                raise FixtureValidationError("fixture encoded bytes exceed the declared bound")
            prepared.append(
                (item, pixels, encoded, pixel_sha256, hashlib.sha256(encoded).hexdigest())
            )
        if output_root.is_symlink():
            raise FixtureValidationError("fixture output root must not be a symlink")
        if output_root.exists() and not output_root.is_dir():
            raise FixtureValidationError("fixture output root must be a directory")
        records: list[dict[str, Any]] = []
        for item, pixels, encoded, pixel_sha256, encoded_sha256 in prepared:
            relative = Path(item["relative_path"])
            destination = output_root / relative
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_file():
                    raise FixtureValidationError(
                        "fixture output must be a regular non-symlink file"
                    )
                metadata = destination.stat()
                if metadata.st_size > manifest["limits"]["max_encoded_bytes"]:
                    raise FixtureValidationError("fixture encoded bytes exceed the declared bound")
                existing = destination.read_bytes()
                if hashlib.sha256(existing).hexdigest() != encoded_sha256:
                    raise FixtureValidationError("fixture encoded digest mismatch before decode")
            else:
                _write_new_regular(destination, encoded)
            records.append(
                {
                    "item_id": item["item_id"],
                    "source_group_id": item["source_group_id"],
                    "page_number": item["page_number"],
                    "source_role": item["source_role"],
                    "relative_path": item["relative_path"],
                    "width": pixels.shape[1],
                    "height": pixels.shape[0],
                    "roi": copy.deepcopy(item["roi"]),
                    "pixel_sha256": pixel_sha256,
                    "encoded_sha256": encoded_sha256,
                }
            )
    except FixtureValidationError:
        raise
    except (ContractValidationError, KeyError, TypeError, ValueError) as error:
        raise FixtureValidationError(str(error)) from error
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": canonical_sha256(manifest),
        "source_role": manifest["source_role"],
        "items": records,
    }


def _musicxml_semantics(payload: bytes) -> set[str]:
    """Derive review semantics from MusicXML rather than trusting manifest labels."""

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise FixtureValidationError("visual fixture source is not valid XML") from error
    names = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
    present = set(names)
    semantics: set[str] = set()
    rules = {
        "clefs": "clef",
        "key-signatures": "fifths",
        "time-signatures": "time",
        "staff-lines": "staff-lines",
        "barlines": "barline",
        "pitched-notes": "pitch",
        "stems": "stem",
        "beams": "beam",
        "rests": "rest",
        "accidentals": "accidental",
        "slurs": "slur",
        "ties": "tied",
        "articulations": "articulations",
        "dynamics": "dynamics",
    }
    semantics.update(label for label, tag in rules.items() if tag in present)
    text = " ".join((element.text or "") for element in root.iter()).strip()
    if present & {"words", "rehearsal", "lyric"} and any(character.isdigit() for character in text):
        semantics.add("text-lyrics-digits")
    if {"staves", "backup", "chord"} <= present:
        semantics.add("grand-staff-polyphony-chords")
    return semantics


def _render_engraved_fixture(
    source: bytes, *, options: Mapping[str, Any], rasterizer: Mapping[str, Any]
) -> np.ndarray:
    """Render one bounded MusicXML source onto a fixed white RGB review canvas."""

    toolkit = verovio.toolkit()
    toolkit.setOptions(dict(options))
    if not toolkit.loadData(source.decode("utf-8")) or toolkit.getPageCount() != 1:
        raise FixtureValidationError("visual fixture must engrave as exactly one page")
    svg = toolkit.renderToSVG(1)
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="#ffffff")
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise FixtureValidationError("engraved fixture rasterization failed")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    mask = np.any(rgb < int(rasterizer["ink_threshold"]), axis=2).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        raise FixtureValidationError("engraved fixture contains no visible notation")
    x, y, width, height = cv2.boundingRect(points)
    margin = int(rasterizer["crop_margin"])
    left, top = max(0, x - margin), max(0, y - margin)
    right = min(rgb.shape[1], x + width + margin)
    bottom = min(rgb.shape[0], y + height + margin)
    cropped = np.ascontiguousarray(rgb[top:bottom, left:right])
    canvas_width = int(rasterizer["canvas_width"])
    canvas_height = int(rasterizer["canvas_height"])
    inset = int(rasterizer["canvas_inset"])
    ratio = min(
        (canvas_width - 2 * inset) / cropped.shape[1],
        (canvas_height - 2 * inset) / cropped.shape[0],
    )
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        cropped,
        (round(cropped.shape[1] * ratio), round(cropped.shape[0] * ratio)),
        interpolation=interpolation,
    )
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    offset_x = (canvas_width - resized.shape[1]) // 2
    offset_y = (canvas_height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return np.ascontiguousarray(canvas)


def generate_visual_fixture_bundle(
    manifest_path: Path, *, source_root: Path, output_root: Path
) -> dict[str, Any]:
    """Validate, engrave, and materialize the separate human-review fixture set."""

    manifest_path = Path(manifest_path)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root)
    try:
        manifest = _read_regular_yaml(
            manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES, kind="visual fixture manifest"
        )
        if manifest.get("source_role") != "visual-degradation-review":
            raise FixtureValidationError(
                "visual fixtures require source_role visual-degradation-review"
            )
        secret = _secret_like_key((), manifest)
        if secret is not None:
            raise FixtureValidationError(f"secret-like fixture metadata is forbidden: {secret}")
        serialized = yaml.safe_dump(manifest, sort_keys=True).casefold()
        if "praig/smb" in serialized or "evaluation_benchmark" in serialized or "hf_" in serialized:
            raise FixtureValidationError(
                "SMB identity, loader, or role is forbidden in visual fixtures"
            )
        validate_instance("visual-fixture-manifest", manifest, version=2)
        configured = manifest["renderer"]
        if importlib.metadata.version("verovio") != configured["engraver"]["version"]:
            raise FixtureValidationError("Verovio runtime version differs from visual manifest")
        if importlib.metadata.version("cairosvg") != configured["rasterizer"]["version"]:
            raise FixtureValidationError("CairoSVG runtime version differs from visual manifest")
        required = set(manifest["required_semantics"])
        item_ids = {item["item_id"] for item in manifest["items"]}
        groups = {item["source_group_id"] for item in manifest["items"]}
        if len(groups) < 4:
            raise FixtureValidationError("visual review requires four independent source groups")
        declared_union: set[str] = set()
        prepared: list[tuple[Mapping[str, Any], np.ndarray, bytes, bytes]] = []
        for item in manifest["items"]:
            source_path = (source_root / item["source_relative_path"]).resolve()
            if not source_path.is_relative_to(source_root):
                raise FixtureValidationError("visual fixture source path escapes source root")
            source = _read_regular_bytes(
                source_path,
                maximum_bytes=manifest["limits"]["max_source_bytes"],
                kind="MusicXML source",
            )
            if hashlib.sha256(source).hexdigest() != item["source_sha256"]:
                raise FixtureValidationError(f"source digest mismatch for {item['item_id']}")
            actual = _musicxml_semantics(source)
            declared = set(item["semantic_features"])
            if actual != declared:
                raise FixtureValidationError(
                    f"declared notation semantics differ from source for {item['item_id']}"
                )
            declared_union.update(declared)
            pixels = _render_engraved_fixture(
                source,
                options=configured["engraver"]["options"],
                rasterizer=configured["rasterizer"],
            )
            pixel_sha256 = _fixture_pixel_sha256(pixels)
            if pixel_sha256 != item["rendered_pixel_sha256"]:
                raise FixtureValidationError(
                    f"rendered pixel digest mismatch for {item['item_id']}"
                )
            encoded = _encode_fixture_png(pixels)
            prepared.append((item, pixels, encoded, source))
        if declared_union != required:
            raise FixtureValidationError(
                "visual fixture set does not prove every required semantic"
            )
        panels = manifest["review_membership"]
        if len(panels) != 12 or {panel["item_id"] for panel in panels} - item_ids:
            raise FixtureValidationError("visual review membership must contain 12 known items")
        if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
            raise FixtureValidationError("visual fixture output root must be a directory")
        records: list[dict[str, Any]] = []
        for item, pixels, encoded, source in prepared:
            destination = output_root / item["relative_path"]
            digest = hashlib.sha256(encoded).hexdigest()
            if destination.exists():
                if (
                    destination.is_symlink()
                    or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
                ):
                    _write_atomic(destination, encoded)
            else:
                _write_new_regular(destination, encoded)
            records.append(
                {
                    "item_id": item["item_id"],
                    "source_group_id": item["source_group_id"],
                    "source_role": item["source_role"],
                    "source_relative_path": item["source_relative_path"],
                    "source_sha256": hashlib.sha256(source).hexdigest(),
                    "relative_path": item["relative_path"],
                    "width": pixels.shape[1],
                    "height": pixels.shape[0],
                    "roi": copy.deepcopy(item["roi"]),
                    "semantic_features": copy.deepcopy(item["semantic_features"]),
                    "pixel_sha256": _fixture_pixel_sha256(pixels),
                    "encoded_sha256": digest,
                }
            )
    except FixtureValidationError:
        raise
    except (
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        raise FixtureValidationError(str(error)) from error
    item_records = {item["item_id"]: item for item in records}
    review_membership = []
    for panel in manifest["review_membership"]:
        item = item_records[panel["item_id"]]
        review_membership.append(
            {
                **copy.deepcopy(panel),
                "fixture_source_role": manifest["source_role"],
                "source_relative_path": item["relative_path"],
                "roi": copy.deepcopy(item["roi"]),
            }
        )
    engraver = copy.deepcopy(configured["engraver"])
    engraver["runtime_version"] = importlib.metadata.version("verovio")
    engraver["runtime_toolkit_version"] = verovio.toolkit().getVersion()
    rasterizer = copy.deepcopy(configured["rasterizer"])
    rasterizer["runtime_version"] = importlib.metadata.version("cairosvg")
    rasterizer["opencv_version"] = cv2.__version__
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": canonical_sha256(manifest),
        "source_role": manifest["source_role"],
        "required_semantics": sorted(required),
        "renderer": {"engraver": engraver, "rasterizer": rasterizer},
        "review_membership": review_membership,
        "items": records,
    }


def _pixel_sha256(pixels: np.ndarray) -> str:
    height, width, channels = pixels.shape
    framed = (
        b"phase2-degradation-rgb8-v1\0"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + channels.to_bytes(1, "big")
        + pixels.tobytes(order="C")
    )
    return hashlib.sha256(framed).hexdigest()


def _dimensions(pixels: np.ndarray) -> dict[str, int]:
    return {
        "height": int(pixels.shape[0]),
        "width": int(pixels.shape[1]),
        "channels": int(pixels.shape[2]),
    }


def align_reference(pixels: np.ndarray, scale: int) -> AlignedReference:
    """Validate RGB8 input and crop only its lower/right divisibility remainder."""

    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise DegradationContractError("reference must be a uint8 NumPy array")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise DegradationContractError("reference must use explicit RGB channels")
    if scale not in (2, 4):
        raise DegradationContractError("scale must be exactly 2 or 4")
    bottom = int(pixels.shape[0] % scale)
    right = int(pixels.shape[1] % scale)
    height = pixels.shape[0] - bottom
    width = pixels.shape[1] - right
    if height < scale or width < scale:
        raise DegradationContractError("reference is too small for the requested scale")
    source = np.ascontiguousarray(pixels)
    aligned = np.ascontiguousarray(source[:height, :width])
    return AlignedReference(
        pixels=aligned,
        input_pixel_sha256=_pixel_sha256(source),
        aligned_pixel_sha256=_pixel_sha256(aligned),
        input_dimensions=tuple(int(value) for value in source.shape),
        aligned_dimensions=tuple(int(value) for value in aligned.shape),
        crop={"top": 0, "left": 0, "bottom": bottom, "right": right},
    )


def derive_degradation_seed(
    master_seed: int,
    *,
    fixture_manifest_id: str,
    item_id: str,
    condition_id: str,
) -> int:
    """Derive a stable PCG64 seed from the complete fixture-condition identity."""

    payload = {
        "domain": "phase2-controlled-degradation-seed-v1",
        "master_seed": master_seed,
        "fixture_manifest_id": fixture_manifest_id,
        "item_id": item_id,
        "condition_id": condition_id,
    }
    return int.from_bytes(bytes.fromhex(canonical_sha256(payload))[:8], "big")


def _condition(control: DegradationControl, condition_id: str) -> dict[str, Any]:
    matches = [entry for entry in control.conditions if entry["condition_id"] == condition_id]
    if len(matches) != 1:
        raise DegradationContractError("condition must be one of the exact six controlled cells")
    return copy.deepcopy(matches[0])


def _encode_rgb(pixels: np.ndarray, condition: Mapping[str, Any]) -> tuple[np.ndarray, bytes]:
    if condition["jpeg"] is None:
        success, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
    else:
        jpeg = condition["jpeg"]
        options = [
            cv2.IMWRITE_JPEG_QUALITY,
            int(jpeg["quality"]),
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            0,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            0,
        ]
        sampling_key = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        sampling_444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if sampling_key is None or sampling_444 is None:
            raise DegradationContractError("OpenCV lacks explicit JPEG 4:4:4 controls")
        options.extend([sampling_key, sampling_444])
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), options)
    if not success:
        raise DegradationContractError("controlled image encoding failed")
    payload = encoded.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise DegradationContractError("controlled image decode verification failed")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)), payload


def apply_degradation(
    pixels: np.ndarray,
    *,
    control: DegradationControl,
    condition_id: str,
    item_id: str,
    source_group_id: str,
    fixture_manifest_id: str,
    purpose: str,
) -> DegradationResult:
    """Apply an explicit deterministic cell and emit its complete scientific lineage."""

    if purpose not in {"fixture-preview", "benchmark"}:
        raise DegradationContractError("degradation purpose must be explicit")
    if purpose == "benchmark" and control.status != "frozen":
        raise DegradationContractError("benchmark degradation requires a frozen control")
    if purpose == "fixture-preview" and control.status != "candidate":
        raise DegradationContractError("fixture preview requires a candidate control")
    condition = _condition(control, condition_id)
    scale = int(condition["scale"])
    aligned = align_reference(pixels, scale)
    derived_seed = derive_degradation_seed(
        control.master_seed,
        fixture_manifest_id=fixture_manifest_id,
        item_id=item_id,
        condition_id=condition_id,
    )

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    working = aligned.pixels.astype(np.float64)
    operations: list[dict[str, Any]] = []
    if condition["blur"] is not None:
        blur = condition["blur"]
        working = cv2.GaussianBlur(
            working,
            (int(blur["kernel"]), int(blur["kernel"])),
            sigmaX=float(blur["sigma"]),
            sigmaY=float(blur["sigma"]),
            borderType=cv2.BORDER_REFLECT_101,
        )
        operations.append(
            {
                "order": 1,
                "operator_id": "blur",
                "parameters": {
                    "type": "gaussian",
                    "sigma": float(blur["sigma"]),
                    "kernel": int(blur["kernel"]),
                    "border": "reflect-101",
                },
            }
        )
    target = (aligned.pixels.shape[1] // scale, aligned.pixels.shape[0] // scale)
    working = cv2.resize(working, target, interpolation=cv2.INTER_AREA)
    operations.append(
        {
            "order": len(operations) + 1,
            "operator_id": "reduction",
            "parameters": {"scale": scale, "interpolation": "INTER_AREA"},
        }
    )
    if condition["noise"] is not None:
        sigma = float(condition["noise"]["sigma"])
        noise = np.random.Generator(np.random.PCG64(derived_seed)).normal(
            0.0, sigma, size=(*working.shape[:2], 1)
        )
        working = working + noise
        operations.append(
            {
                "order": len(operations) + 1,
                "operator_id": "noise",
                "parameters": {
                    "type": "gaussian",
                    "sigma": sigma,
                    "channel_mode": "achromatic-equal",
                    "generator": "PCG64",
                },
            }
        )
        working = np.clip(np.rint(working), 0, 255).astype(np.uint8)
        operations.append(
            {
                "order": len(operations) + 1,
                "operator_id": "clip-round",
                "parameters": {"range": [0, 255], "rounding": "rint", "dtype": "uint8"},
            }
        )
        operations.append(
            {
                "order": len(operations) + 1,
                "operator_id": "jpeg",
                "parameters": {
                    "quality": int(condition["jpeg"]["quality"]),
                    "sampling_factor": "4:4:4",
                    "progressive": False,
                    "optimize": False,
                    "colour_order": "RGB-BGR-explicit",
                },
            }
        )
    else:
        working = np.clip(np.rint(working), 0, 255).astype(np.uint8)

    output, encoded = _encode_rgb(working, condition)
    trace_payload = {
        "schema_version": 2,
        "record_type": "degradation-trace",
        "item_id": item_id,
        "source_group_id": source_group_id,
        "fixture_manifest_id": fixture_manifest_id,
        "condition_id": condition_id,
        "control_sha256": control.sha256,
        "master_seed": control.master_seed,
        "derived_seed": derived_seed,
        "input_pixel_sha256": aligned.input_pixel_sha256,
        "aligned_pixel_sha256": aligned.aligned_pixel_sha256,
        "output_pixel_sha256": _pixel_sha256(output),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "input_dimensions": _dimensions(np.asarray(pixels)),
        "aligned_dimensions": _dimensions(aligned.pixels),
        "output_dimensions": _dimensions(output),
        "crop": aligned.crop,
        "operations": operations,
        "runtime": {
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "opencl": bool(cv2.ocl.useOpenCL()),
        },
    }
    trace = {"trace_id": f"degradation-{canonical_sha256(trace_payload)}", **trace_payload}
    try:
        validate_instance("degradation-trace", trace, version=2)
    except ContractValidationError as error:
        raise DegradationContractError(str(error)) from error
    return DegradationResult(pixels=output, encoded_bytes=encoded, trace=trace)


def assert_notebook_source_clean(path: Path) -> str:
    """Reject tracked notebook execution state or embedded binary display payloads."""

    try:
        notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NotebookSourceError("notebook source is not readable canonical JSON") from error
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise NotebookSourceError("notebook source has no cell list")
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs")
        ):
            raise NotebookSourceError("tracked notebook must be output-free and unexecuted")
        serialized = json.dumps(cell, sort_keys=True).casefold()
        if any(mime in serialized for mime in ("image/png", "image/jpeg", "application/pdf")):
            raise NotebookSourceError("tracked notebook contains an embedded binary payload")
    return canonical_sha256(notebook)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write_atomic(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise OSError(f"temporary output already exists: {temporary}")
    _write_new_regular(temporary, payload)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _regular_file_inventory(root: Path, *, exclude_candidates: bool = False) -> dict[str, str]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise DegradationDecisionError("evidence root must be a regular directory, not a symlink")
    inventory: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in tuple(directory_names):
            child = current / name
            if child.is_symlink():
                raise DegradationDecisionError("evidence inventory contains a symlink directory")
            if exclude_candidates and child == root / "candidates":
                directory_names.remove(name)
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink():
                raise DegradationDecisionError("evidence inventory contains a symlink file")
            relative = path.relative_to(root).as_posix()
            payload = _read_regular_bytes(
                path, maximum_bytes=_MAX_EVIDENCE_BYTES, kind="evidence file"
            )
            inventory[relative] = hashlib.sha256(payload).hexdigest()
            if len(inventory) > 128:
                raise DegradationDecisionError("evidence inventory exceeds the file-count bound")
    return dict(sorted(inventory.items()))


def _validate_legacy_evidence_root(root: Path) -> dict[str, str]:
    inventory = _regular_file_inventory(root, exclude_candidates=True)
    required = {
        "degradation-decision.json",
        "fixture-bundle.json",
        "preview-manifest.json",
        "preview-mapping.json",
        "preview-membership.json",
        "preview-working.ipynb",
    }
    if not required <= inventory.keys():
        raise DegradationDecisionError("legacy candidate-1 evidence is incomplete")
    manifest = _read_regular_json(root / "preview-manifest.json", kind="legacy preview manifest")
    decision = _read_regular_json(root / "degradation-decision.json", kind="legacy decision")
    membership = _read_regular_json(root / "preview-membership.json", kind="legacy membership")
    mapping = _read_regular_json(root / "preview-mapping.json", kind="legacy mapping")
    if manifest.get("candidate_id") != LEGACY_CANDIDATE_ID:
        raise DegradationDecisionError("legacy preview candidate identity differs")
    if manifest.get("candidate_sha256") != LEGACY_CANDIDATE_CANONICAL_SHA256:
        raise DegradationDecisionError("legacy preview candidate digest differs")
    if decision.get("decision") != "reject" or decision.get("candidate_id") != LEGACY_CANDIDATE_ID:
        raise DegradationDecisionError("legacy decision must remain the human rejection")
    expected_decision = {
        "candidate_sha256": LEGACY_CANDIDATE_CANONICAL_SHA256,
        "notebook_source_sha256": manifest.get("notebook_source_sha256"),
        "preview_manifest_sha256": canonical_sha256(manifest),
        "membership_sha256": canonical_sha256(membership),
        "panel_sha256s": manifest.get("panel_sha256s"),
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise DegradationDecisionError(f"legacy decision differs at {key}")
    if manifest.get("mapping_sha256") != canonical_sha256(mapping):
        raise DegradationDecisionError("legacy preview mapping differs")
    panels = manifest.get("panels")
    if not isinstance(panels, list) or len(panels) != 12:
        raise DegradationDecisionError("legacy preview must retain twelve panels")
    actual_panel_sha256s: list[str] = []
    for panel in panels:
        relative = Path(panel["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DegradationDecisionError("legacy panel path escapes its evidence root")
        payload = _read_regular_bytes(
            root / relative, maximum_bytes=_MAX_EVIDENCE_BYTES, kind="legacy panel"
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != panel["sha256"]:
            raise DegradationDecisionError("legacy panel bytes differ")
        actual_panel_sha256s.append(digest)
    if actual_panel_sha256s != manifest["panel_sha256s"]:
        raise DegradationDecisionError("legacy panel ordering differs")
    return inventory


def _validate_legacy_archive(legacy_root: Path, archive_root: Path) -> dict[str, Any]:
    source_inventory = _validate_legacy_evidence_root(legacy_root)
    archive_inventory = _regular_file_inventory(archive_root)
    reconciliation_name = "legacy-evidence-reconciliation.json"
    if reconciliation_name not in archive_inventory:
        raise DegradationDecisionError("legacy archive reconciliation is missing")
    archived_files = {
        path: digest for path, digest in archive_inventory.items() if path != reconciliation_name
    }
    if archived_files != source_inventory:
        raise DegradationDecisionError("legacy archive bytes differ from candidate-1 evidence")
    reconciliation = _read_regular_json(
        archive_root / reconciliation_name, kind="legacy evidence reconciliation"
    )
    expected_files = [
        {"relative_path": path, "sha256": digest} for path, digest in source_inventory.items()
    ]
    if reconciliation != {
        "schema_version": 1,
        "record_type": "legacy-degradation-evidence-reconciliation",
        "candidate_id": LEGACY_CANDIDATE_ID,
        "candidate_sha256": LEGACY_CANDIDATE_CANONICAL_SHA256,
        "source_files": expected_files,
        "source_inventory_sha256": canonical_sha256(expected_files),
    }:
        raise DegradationDecisionError("legacy archive reconciliation content differs")
    return reconciliation


def archive_legacy_degradation_evidence(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Copy immutable candidate-1 evidence into a candidate-scoped reconciled archive."""

    project_root = Path(project_root).resolve()
    legacy_root = project_root / "artifacts/phase2-degradation-preview"
    destination_base = legacy_root if artifact_root is None else Path(artifact_root).resolve()
    candidates_root = destination_base / "candidates"
    archive_root = candidates_root / LEGACY_CANDIDATE_ID
    if archive_root.exists() or archive_root.is_symlink():
        reconciliation = _validate_legacy_archive(legacy_root, archive_root)
        return {
            "archive_root": str(archive_root),
            "reconciliation_sha256": canonical_sha256(reconciliation),
        }

    source_inventory = _validate_legacy_evidence_root(legacy_root)
    if candidates_root.is_symlink():
        raise DegradationDecisionError("candidate archive parent must not be a symlink")
    candidates_root.mkdir(parents=True, exist_ok=True)
    temporary = candidates_root / f".{LEGACY_CANDIDATE_ID}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    for relative in source_inventory:
        payload = _read_regular_bytes(
            legacy_root / relative,
            maximum_bytes=_MAX_EVIDENCE_BYTES,
            kind="legacy evidence source",
        )
        _write_new_regular(temporary / relative, payload)
    source_files = [
        {"relative_path": path, "sha256": digest} for path, digest in source_inventory.items()
    ]
    reconciliation = {
        "schema_version": 1,
        "record_type": "legacy-degradation-evidence-reconciliation",
        "candidate_id": LEGACY_CANDIDATE_ID,
        "candidate_sha256": LEGACY_CANDIDATE_CANONICAL_SHA256,
        "source_files": source_files,
        "source_inventory_sha256": canonical_sha256(source_files),
    }
    _write_new_regular(
        temporary / "legacy-evidence-reconciliation.json", _json_bytes(reconciliation)
    )
    _validate_legacy_archive(legacy_root, temporary)
    os.rename(temporary, archive_root)
    directory = os.open(candidates_root, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "archive_root": str(archive_root),
        "reconciliation_sha256": canonical_sha256(reconciliation),
    }


def _load_rgb(path: Path) -> np.ndarray:
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise FixtureValidationError(f"fixture image failed to decode: {path.name}")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _panel_image(
    reference: np.ndarray, degraded: np.ndarray, label: str, roi: Mapping[str, int], scale: int
) -> np.ndarray:
    canvas_width = 1120
    canvas_height = 820
    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    available_height = 390

    def place(image: np.ndarray, left: int, title: str, top_base: int = 80) -> None:
        ratio = min(500 / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        top = top_base + (available_height - resized.shape[0]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            title,
            (left, top_base - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (30, 30, 30),
            1,
        )

    x = min(int(roi["x"]), reference.shape[1] - 1)
    y = min(int(roi["y"]), reference.shape[0] - 1)
    width = min(int(roi["width"]), reference.shape[1] - x)
    height = min(int(roi["height"]), reference.shape[0] - y)
    lr_x, lr_y = x // scale, y // scale
    lr_width = max(1, width // scale)
    lr_height = max(1, height // scale)
    reference_context = reference.copy()
    degraded_context = degraded.copy()
    cv2.rectangle(reference_context, (x, y), (x + width, y + height), (190, 30, 30), 4)
    cv2.rectangle(
        degraded_context,
        (lr_x, lr_y),
        (lr_x + lr_width, lr_y + lr_height),
        (190, 30, 30),
        max(1, 4 // scale),
    )
    place(reference_context, 20, "Whole engraved HR system (fixed ROI in red)")
    place(degraded_context, 590, "Corresponding whole LR system")
    hr_roi = reference[y : y + height, x : x + width]
    lr_roi = degraded[lr_y : lr_y + lr_height, lr_x : lr_x + lr_width]
    roi_height = 235
    hr_zoom = cv2.resize(hr_roi, (500, roi_height), interpolation=cv2.INTER_NEAREST)
    lr_zoom = cv2.resize(lr_roi, (500, roi_height), interpolation=cv2.INTER_NEAREST)
    canvas[555 : 555 + roi_height, 20:520] = hr_zoom
    canvas[555 : 555 + roi_height, 590:1090] = lr_zoom
    cv2.putText(canvas, "Fixed HR ROI", (20, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1)
    cv2.putText(
        canvas,
        "Corresponding LR ROI",
        (590, 540),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        1,
    )
    return canvas


def _build_degradation_preview_directory(
    project_root: Path,
    *,
    artifact_root: Path,
    legacy_reconciliation_sha256: str,
) -> dict[str, Any]:
    """Build one complete candidate-2 directory before atomic publication."""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    source_path = project_root / "notebooks/02-degradation-preview.ipynb"
    source_sha256 = assert_notebook_source_clean(source_path)
    control_path = project_root / "configs/degradations/controlled-score-candidates.yaml"
    control = load_degradation_control(control_path)
    if control.candidate_id != CURRENT_CANDIDATE_ID or control.version != 2:
        raise DegradationDecisionError("preview publication requires exact candidate 2")
    fixture_root = artifact_root / "visual-fixtures"
    fixture_manifest_path = project_root / "tests/fixtures/phase2/visual-fixture-manifest-v2.yaml"
    bundle = generate_visual_fixture_bundle(
        fixture_manifest_path,
        source_root=fixture_manifest_path.parent,
        output_root=fixture_root,
    )
    membership = {
        "schema_version": 1,
        "record_type": "degradation-preview-membership",
        "selection_policy": "paired-fixed-anchors-v2",
        "control_sha256": control.sha256,
        "fixture_manifest_id": bundle["manifest_id"],
        "fixture_manifest_sha256": bundle["manifest_sha256"],
        "panels": bundle["review_membership"],
    }
    _write_atomic(artifact_root / "fixture-bundle.json", _json_bytes(bundle))
    _write_atomic(artifact_root / "preview-membership.json", _json_bytes(membership))

    item_records = {item["item_id"]: item for item in bundle["items"]}
    panel_records: list[dict[str, Any]] = []
    mapping_records: list[dict[str, Any]] = []
    panel_sha256s: list[str] = []
    panels_root = artifact_root / "panels"
    for index, member in enumerate(membership["panels"], start=1):
        item = item_records[member["item_id"]]
        reference = _load_rgb(fixture_root / item["relative_path"])
        result = apply_degradation(
            reference,
            control=control,
            condition_id=member["condition_id"],
            item_id=member["item_id"],
            source_group_id=member["source_group_id"],
            fixture_manifest_id=bundle["manifest_id"],
            purpose="fixture-preview",
        )
        aligned = align_reference(reference, int(member["condition_id"][1]))
        label = f"{member['condition_id']} | {member['item_id']} | panel {index:02d}/12"
        panel_pixels = _panel_image(
            aligned.pixels,
            result.pixels,
            label,
            member["roi"],
            int(member["condition_id"][1]),
        )
        panel_bytes = _encode_fixture_png(panel_pixels)
        panel_path = panels_root / f"{member['panel_id']}-{member['condition_id']}.png"
        if panel_path.exists():
            if (
                hashlib.sha256(panel_path.read_bytes()).hexdigest()
                != hashlib.sha256(panel_bytes).hexdigest()
            ):
                _write_atomic(panel_path, panel_bytes)
        else:
            _write_new_regular(panel_path, panel_bytes)
        panel_sha256 = hashlib.sha256(panel_bytes).hexdigest()
        panel_sha256s.append(panel_sha256)
        panel_records.append(
            {
                "condition_id": member["condition_id"],
                "item_id": member["item_id"],
                "source_group_id": member["source_group_id"],
                "fixture_source_role": bundle["source_role"],
                "relative_path": panel_path.relative_to(artifact_root).as_posix(),
                "sha256": panel_sha256,
                "degradation_trace_id": result.trace["trace_id"],
            }
        )
        mapping_records.append(
            {
                "condition_id": member["condition_id"],
                "item_id": member["item_id"],
                "source_group_id": member["source_group_id"],
                "source_pixel_sha256": item["pixel_sha256"],
                "degradation_trace": result.trace,
                "panel_sha256": panel_sha256,
            }
        )
    mapping = {
        "schema_version": 1,
        "record_type": "degradation-preview-mapping",
        "panels": mapping_records,
    }
    _write_atomic(artifact_root / "preview-mapping.json", _json_bytes(mapping))
    manifest = {
        "schema_version": 1,
        "record_type": "degradation-preview-manifest",
        "candidate_id": control.candidate_id,
        "candidate_sha256": control.sha256,
        "candidate_registry_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "candidate_registry_relative_path": "configs/degradations/controlled-score-candidates.yaml",
        "notebook_source_sha256": source_sha256,
        "notebook_source_relative_path": "notebooks/02-degradation-preview.ipynb",
        "fixture_manifest_id": bundle["manifest_id"],
        "fixture_manifest_sha256": bundle["manifest_sha256"],
        "fixture_bundle_sha256": canonical_sha256(bundle),
        "fixture_source_role": bundle["source_role"],
        "renderer": bundle["renderer"],
        "membership_sha256": canonical_sha256(membership),
        "mapping_sha256": canonical_sha256(mapping),
        "legacy_evidence_reconciliation_sha256": legacy_reconciliation_sha256,
        "panels": panel_records,
        "panel_sha256s": panel_sha256s,
    }
    manifest_bytes = _json_bytes(manifest)
    _write_atomic(artifact_root / "preview-manifest.json", manifest_bytes)

    try:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(source_path, as_version=4)
        marker = "__GENERATED_PHASE2_VISUAL_REVIEW_WORKING_COPY__"
        replacements = 0
        for cell in notebook.cells:
            if marker in cell.source:
                cell.source = cell.source.replace(marker, f"generated:{canonical_sha256(manifest)}")
                replacements += 1
        if replacements != 1:
            raise NotebookSourceError("tracked notebook working-copy guard is absent or ambiguous")
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(project_root)}},
        )
        environment_key = "SCORE_SR_PHASE2_PREVIEW_ROOT"
        prior_artifact_root = os.environ.get(environment_key)
        os.environ[environment_key] = str(artifact_root)
        try:
            client.execute()
        finally:
            if prior_artifact_root is None:
                os.environ.pop(environment_key, None)
            else:
                os.environ[environment_key] = prior_artifact_root
        working_path = artifact_root / "preview-working.ipynb"
        payload = nbformat.writes(notebook).encode("utf-8")
        _write_atomic(working_path, payload)
    except Exception as error:
        raise NotebookSourceError(f"ignored preview notebook execution failed: {error}") from error

    assert_notebook_source_clean(source_path)
    publication = {
        "schema_version": 1,
        "record_type": "degradation-preview-publication",
        "candidate_id": control.candidate_id,
        "preview_manifest_sha256": canonical_sha256(manifest),
        "working_notebook_sha256": hashlib.sha256(
            (artifact_root / "preview-working.ipynb").read_bytes()
        ).hexdigest(),
    }
    _write_new_regular(artifact_root / "preview-publication.json", _json_bytes(publication))
    return {
        "artifact_root": str(artifact_root),
        "project_root": str(project_root),
        "notebook_source_sha256": source_sha256,
        "preview_manifest_sha256": canonical_sha256(manifest),
    }


def build_degradation_preview(
    project_root: Path, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """Archive candidate 1 and atomically publish or validate the paired candidate-2 preview."""

    project_root = Path(project_root).resolve()
    expected_base = project_root / "artifacts/phase2-degradation-preview"
    artifact_base = expected_base if artifact_root is None else Path(artifact_root).resolve()
    if artifact_base != expected_base:
        raise DegradationDecisionError("preview artifact root must be the project candidate base")
    archive = archive_legacy_degradation_evidence(project_root, artifact_root=artifact_base)
    candidates_root = artifact_base / "candidates"
    candidate_root = candidates_root / CURRENT_CANDIDATE_ID
    if candidate_root.exists() or candidate_root.is_symlink():
        manifest = _reconcile_preview_evidence(candidate_root / "preview-manifest.json")
        return {
            "artifact_root": str(candidate_root),
            "project_root": str(project_root),
            "notebook_source_sha256": manifest["notebook_source_sha256"],
            "preview_manifest_sha256": canonical_sha256(manifest),
        }
    if candidates_root.is_symlink():
        raise DegradationDecisionError("candidate preview parent must not be a symlink")
    candidates_root.mkdir(parents=True, exist_ok=True)
    temporary = candidates_root / f".{CURRENT_CANDIDATE_ID}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    built = _build_degradation_preview_directory(
        project_root,
        artifact_root=temporary,
        legacy_reconciliation_sha256=archive["reconciliation_sha256"],
    )
    _reconcile_preview_evidence(temporary / "preview-manifest.json", allow_temporary_root=True)
    os.rename(temporary, candidate_root)
    directory = os.open(candidates_root, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    built["artifact_root"] = str(candidate_root)
    return built


def _read_regular_json(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        raw = _read_regular_bytes(path, maximum_bytes=_MAX_REVIEW_BYTES, kind=kind)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DegradationDecisionError(f"{kind} cannot be read safely") from error
    if not isinstance(value, dict):
        raise DegradationDecisionError(f"{kind} root must be an object")
    return value


def _read_regular_bytes(path: Path, *, maximum_bytes: int, kind: str) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{kind} path must not be a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
        raise ValueError(f"{kind} must be a bounded regular file")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, maximum_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) > maximum_bytes or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{kind} changed or exceeded bounds while being read")
    return raw


def _reconcile_preview_evidence(
    preview_manifest_path: Path, *, allow_temporary_root: bool = False
) -> dict[str, Any]:
    manifest_path = Path(preview_manifest_path).resolve()
    candidate_root = manifest_path.parent
    temporary_prefix = f".{CURRENT_CANDIDATE_ID}.tmp-"
    valid_root_name = candidate_root.name == CURRENT_CANDIDATE_ID or (
        allow_temporary_root and candidate_root.name.startswith(temporary_prefix)
    )
    if (
        manifest_path.name != "preview-manifest.json"
        or not valid_root_name
        or candidate_root.parent.name != "candidates"
        or candidate_root.parent.parent.name != "phase2-degradation-preview"
        or candidate_root.parent.parent.parent.name != "artifacts"
    ):
        raise DegradationDecisionError("preview manifest is outside the candidate-2 scoped root")
    project_root = candidate_root.parents[3]
    manifest = _read_regular_json(manifest_path, kind="preview manifest")
    membership = _read_regular_json(
        candidate_root / "preview-membership.json", kind="preview membership"
    )
    mapping = _read_regular_json(candidate_root / "preview-mapping.json", kind="preview mapping")
    fixture_bundle = _read_regular_json(
        candidate_root / "fixture-bundle.json", kind="visual fixture bundle"
    )
    publication = _read_regular_json(
        candidate_root / "preview-publication.json", kind="preview publication"
    )
    if manifest.get("candidate_id") != CURRENT_CANDIDATE_ID:
        raise DegradationDecisionError("preview manifest does not identify candidate 2")
    control_path = project_root / "configs/degradations/controlled-score-candidates.yaml"
    control = load_degradation_control(control_path)
    registry_bytes = _read_regular_bytes(
        control_path, maximum_bytes=_MAX_CONTROL_BYTES, kind="candidate registry"
    )
    expected_manifest = {
        "candidate_sha256": control.sha256,
        "candidate_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "candidate_registry_relative_path": "configs/degradations/controlled-score-candidates.yaml",
        "notebook_source_relative_path": "notebooks/02-degradation-preview.ipynb",
        "fixture_bundle_sha256": canonical_sha256(fixture_bundle),
        "fixture_manifest_sha256": fixture_bundle.get("manifest_sha256"),
        "membership_sha256": canonical_sha256(membership),
        "mapping_sha256": canonical_sha256(mapping),
    }
    if control.candidate_id != CURRENT_CANDIDATE_ID or control.version != 2:
        raise DegradationDecisionError("preview registry no longer selects exact candidate 2")
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise DegradationDecisionError(f"preview manifest differs at {key}")
    source_path = project_root / manifest["notebook_source_relative_path"]
    if assert_notebook_source_clean(source_path) != manifest.get("notebook_source_sha256"):
        raise DegradationDecisionError("preview source notebook digest differs")

    archive_root = candidate_root.parent / LEGACY_CANDIDATE_ID
    legacy_root = candidate_root.parent.parent
    archive_reconciliation = _validate_legacy_archive(legacy_root, archive_root)
    if manifest.get("legacy_evidence_reconciliation_sha256") != canonical_sha256(
        archive_reconciliation
    ):
        raise DegradationDecisionError("legacy archive lineage differs")

    fixture_manifest_path = project_root / "tests/fixtures/phase2/visual-fixture-manifest-v2.yaml"
    fixture_manifest = _read_regular_yaml(
        fixture_manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        kind="visual fixture manifest",
    )
    if fixture_bundle.get("manifest_id") != "phase2-engraved-visual-fixtures-v2":
        raise DegradationDecisionError("visual fixture bundle identity differs")
    if fixture_bundle.get("manifest_sha256") != canonical_sha256(fixture_manifest):
        raise DegradationDecisionError("visual fixture manifest bytes differ")
    items = fixture_bundle.get("items")
    if not isinstance(items, list) or len({item.get("source_group_id") for item in items}) != 4:
        raise DegradationDecisionError("visual fixture bundle must retain four source groups")
    expected_files = {
        "fixture-bundle.json",
        "preview-manifest.json",
        "preview-mapping.json",
        "preview-membership.json",
        "preview-publication.json",
        "preview-working.ipynb",
    }
    for item in items:
        source_relative = Path(item["source_relative_path"])
        image_relative = Path("visual-fixtures") / item["relative_path"]
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise DegradationDecisionError("visual fixture source path escapes its root")
        if image_relative.is_absolute() or ".." in image_relative.parts:
            raise DegradationDecisionError("visual fixture image path escapes its root")
        source_bytes = _read_regular_bytes(
            fixture_manifest_path.parent / source_relative,
            maximum_bytes=_MAX_MANIFEST_BYTES,
            kind="visual fixture source",
        )
        if hashlib.sha256(source_bytes).hexdigest() != item["source_sha256"]:
            raise DegradationDecisionError("visual fixture source bytes differ")
        image_bytes = _read_regular_bytes(
            candidate_root / image_relative,
            maximum_bytes=_MAX_EVIDENCE_BYTES,
            kind="visual fixture image",
        )
        if hashlib.sha256(image_bytes).hexdigest() != item["encoded_sha256"]:
            raise DegradationDecisionError("visual fixture image bytes differ")
        expected_files.add(image_relative.as_posix())

    members = membership.get("panels")
    bundle_members = fixture_bundle.get("review_membership")
    if (
        membership.get("selection_policy") != "paired-fixed-anchors-v2"
        or members != bundle_members
        or not isinstance(members, list)
        or len(members) != 12
    ):
        raise DegradationDecisionError("paired preview membership differs")
    expected_items = ["review-work-01-excerpt-01", "review-work-04-excerpt-01"]
    expected_rois: list[Mapping[str, Any]] | None = None
    for offset, condition_id in enumerate(EXPECTED_CONDITION_IDS):
        pair = members[offset * 2 : offset * 2 + 2]
        if [member.get("condition_id") for member in pair] != [condition_id, condition_id]:
            raise DegradationDecisionError("paired preview condition order differs")
        if [member.get("item_id") for member in pair] != expected_items:
            raise DegradationDecisionError("paired preview anchors differ")
        rois = [member.get("roi") for member in pair]
        if expected_rois is None:
            expected_rois = rois
        elif rois != expected_rois:
            raise DegradationDecisionError("paired preview ROI identities differ")

    panels = manifest.get("panels")
    mapping_panels = mapping.get("panels")
    if not isinstance(panels, list) or not isinstance(mapping_panels, list):
        raise DegradationDecisionError("preview panel evidence is malformed")
    if len(panels) != 12 or len(mapping_panels) != 12:
        raise DegradationDecisionError("preview must contain twelve paired panels")
    panel_sha256s: list[str] = []
    for member, panel, mapped in zip(members, panels, mapping_panels, strict=True):
        identity = (member["condition_id"], member["item_id"], member["source_group_id"])
        if identity != (
            panel.get("condition_id"),
            panel.get("item_id"),
            panel.get("source_group_id"),
        ) or identity != (
            mapped.get("condition_id"),
            mapped.get("item_id"),
            mapped.get("source_group_id"),
        ):
            raise DegradationDecisionError("preview membership, mapping, and manifest differ")
        relative = Path(panel["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("panels",):
            raise DegradationDecisionError("panel path escapes the candidate root")
        panel_bytes = _read_regular_bytes(
            candidate_root / relative,
            maximum_bytes=_MAX_EVIDENCE_BYTES,
            kind="candidate-2 panel",
        )
        digest = hashlib.sha256(panel_bytes).hexdigest()
        if digest != panel.get("sha256") or digest != mapped.get("panel_sha256"):
            raise DegradationDecisionError("candidate-2 panel bytes differ")
        panel_sha256s.append(digest)
        expected_files.add(relative.as_posix())
    if panel_sha256s != manifest.get("panel_sha256s"):
        raise DegradationDecisionError("candidate-2 panel digest order differs")

    working_path = candidate_root / "preview-working.ipynb"
    working_sha256 = hashlib.sha256(
        _read_regular_bytes(
            working_path, maximum_bytes=_MAX_EVIDENCE_BYTES, kind="candidate-2 working notebook"
        )
    ).hexdigest()
    expected_publication = {
        "schema_version": 1,
        "record_type": "degradation-preview-publication",
        "candidate_id": CURRENT_CANDIDATE_ID,
        "preview_manifest_sha256": canonical_sha256(manifest),
        "working_notebook_sha256": working_sha256,
    }
    if publication != expected_publication:
        raise DegradationDecisionError("candidate-2 publication reconciliation differs")
    inventory = _regular_file_inventory(candidate_root)
    allowed_optional = {
        "degradation-decision.json",
        "degradation-decision-reconciliation.json",
    }
    unexpected = set(inventory) - expected_files - allowed_optional
    if unexpected or not expected_files <= inventory.keys():
        raise DegradationDecisionError("candidate-2 publication is partial or has extra files")
    serialized = json.dumps(
        (manifest, membership, mapping, fixture_bundle), sort_keys=True
    ).casefold()
    if (
        "praig/smb" in serialized
        or "load_dataset" in serialized
        or "data/sources/smb" in serialized
    ):
        raise DegradationDecisionError("SMB identity or loader is forbidden in candidate preview")
    return manifest


def validate_degradation_decision(
    decision_path: Path, preview_manifest_path: Path
) -> dict[str, Any]:
    """Validate human authorship and bind a decision to every preview content identity."""

    try:
        decision_path = Path(decision_path).resolve()
        preview_manifest_path = Path(preview_manifest_path).resolve()
        if (
            decision_path.name != "degradation-decision.json"
            or preview_manifest_path.name != "preview-manifest.json"
            or decision_path.parent != preview_manifest_path.parent
        ):
            raise DegradationDecisionError(
                "decision and manifest must share the candidate-2 scoped root"
            )
        manifest = _reconcile_preview_evidence(preview_manifest_path)
        decision = _read_regular_json(decision_path, kind="degradation decision")
        validate_instance("degradation-review", decision, version=2)
        if not decision["reviewer"].strip() or not decision["rationale"].strip():
            raise DegradationDecisionError("reviewer and rationale must be non-empty")
        reviewed_at = decision["reviewed_at"]
        if not reviewed_at.endswith("Z"):
            raise DegradationDecisionError("reviewed_at must be canonical UTC")
        datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
        expected = {
            "candidate_id": CURRENT_CANDIDATE_ID,
            "candidate_sha256": manifest["candidate_sha256"],
            "notebook_source_sha256": manifest["notebook_source_sha256"],
            "preview_manifest_sha256": canonical_sha256(manifest),
            "membership_sha256": manifest["membership_sha256"],
            "panel_sha256s": manifest["panel_sha256s"],
        }
        for field, value in expected.items():
            if decision[field] != value:
                raise DegradationDecisionError(f"decision is stale or mismatched at {field}")
    except DegradationDecisionError:
        raise
    except (ContractValidationError, KeyError, TypeError, ValueError) as error:
        raise DegradationDecisionError(str(error)) from error
    return decision


def freeze_degradation_control(
    control_path: Path,
    decision_path: Path,
    preview_manifest_path: Path,
    output_path: Path,
    *,
    reconciliation_path: Path,
) -> dict[str, Any]:
    """Publish a content-bound frozen control only after an exact human acceptance."""

    decision = validate_degradation_decision(decision_path, preview_manifest_path)
    if decision["decision"] == "reject":
        return {
            "status": "blocked-rejected",
            "next_step": "revise one new append-only candidate and repeat the fixed preview review",
        }
    control = load_degradation_control(control_path)
    if (
        control.candidate_id != CURRENT_CANDIDATE_ID
        or control.version != 2
        or control.sha256 != decision["candidate_sha256"]
    ):
        raise DegradationDecisionError("accepted decision does not bind the current candidate")
    manifest = _reconcile_preview_evidence(preview_manifest_path)
    reconciliation = {
        "schema_version": 1,
        "record_type": "degradation-decision-reconciliation",
        "decision_sha256": canonical_sha256(decision),
        "decision": decision["decision"],
        "candidate_id": CURRENT_CANDIDATE_ID,
        "candidate_sha256": decision["candidate_sha256"],
        "candidate_registry_sha256": manifest["candidate_registry_sha256"],
        "preview_manifest_sha256": decision["preview_manifest_sha256"],
        "membership_sha256": decision["membership_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "legacy_evidence_reconciliation_sha256": manifest["legacy_evidence_reconciliation_sha256"],
        "panel_sha256s": decision["panel_sha256s"],
        "reconciled_at": decision["reviewed_at"],
    }
    reconciliation_sha256 = canonical_sha256(reconciliation)
    frozen = {
        "schema_version": 2,
        "record_type": "degradation-control",
        "control_id": "controlled-score-v1",
        "version": 1,
        "status": "frozen",
        "candidate_id": control.candidate_id,
        "candidate_version": control.version,
        "claim_boundary": control.claim_boundary,
        "master_seed": control.master_seed,
        "candidate_sha256": control.sha256,
        "decision_sha256": canonical_sha256(decision),
        "decision_reconciliation_sha256": reconciliation_sha256,
        "image_contract": control.image_contract,
        "alignment": control.alignment,
        "runtime": control.runtime,
        "condition_order": list(control.condition_ids),
        "conditions": list(control.conditions),
    }
    if Path(output_path).exists() or Path(reconciliation_path).exists():
        raise DegradationDecisionError("freeze outputs already exist; refusing overwrite")
    _write_new_regular(Path(reconciliation_path), _json_bytes(reconciliation))
    try:
        _write_new_regular(
            Path(output_path), yaml.safe_dump(frozen, sort_keys=False).encode("utf-8")
        )
    except Exception:
        Path(reconciliation_path).unlink(missing_ok=True)
        raise
    return {"status": "frozen", "reconciliation_sha256": reconciliation_sha256}


class DegradationPreviewSession:
    """Notebook-only helper for rendering panels and recording a human decision."""

    def __init__(self, project_root: Path | None = None, artifact_root: Path | None = None) -> None:
        discovery_start = Path.cwd() if project_root is None else Path(project_root)
        self.project_root = _discover_project_root(discovery_start)
        configured = os.environ.get("SCORE_SR_PHASE2_PREVIEW_ROOT")
        self.artifact_root = (
            Path(artifact_root).resolve()
            if artifact_root is not None
            else Path(configured).resolve()
            if configured
            else self.project_root
            / "artifacts/phase2-degradation-preview/candidates"
            / CURRENT_CANDIDATE_ID
        )
        self.manifest_path = self.artifact_root / "preview-manifest.json"
        self.manifest = _read_regular_json(self.manifest_path, kind="preview manifest")
        if self.manifest.get("candidate_id") != CURRENT_CANDIDATE_ID:
            raise DegradationDecisionError("working notebook must open candidate 2 only")

    def summary(self) -> None:
        from IPython.display import Image, Markdown, display

        display(Markdown("## Paired candidate-2 degradation review"))
        for panel in self.manifest["panels"]:
            display(Markdown(f"### {panel['condition_id']} — {panel['item_id']}"))
            display(Image(filename=str(self.artifact_root / panel["relative_path"])))

    def decision_widget(self) -> Any:
        import ipywidgets as widgets
        from IPython.display import display

        reviewer = widgets.Text(description="Reviewer:")
        decision = widgets.ToggleButtons(
            options=[("Choose…", ""), ("Accept", "accept"), ("Reject", "reject")],
            description="Decision:",
        )
        rationale = widgets.Textarea(description="Rationale:", layout={"width": "90%"})
        save = widgets.Button(description="Save scientific decision", button_style="warning")
        status = widgets.HTML()

        def record(_button: Any) -> None:
            if not reviewer.value.strip() or decision.value not in {"accept", "reject"}:
                status.value = "<b>Not saved:</b> reviewer and accept/reject are required."
                return
            if not rationale.value.strip():
                status.value = "<b>Not saved:</b> a non-empty rationale is required."
                return
            manifest = _read_regular_json(self.manifest_path, kind="preview manifest")
            record_value = {
                "schema_version": 2,
                "record_type": "degradation-review",
                "decision": decision.value,
                "reviewer": reviewer.value.strip(),
                "reviewed_at": datetime.now(UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "rationale": rationale.value.strip(),
                "candidate_id": manifest["candidate_id"],
                "candidate_sha256": manifest["candidate_sha256"],
                "notebook_source_sha256": manifest["notebook_source_sha256"],
                "preview_manifest_sha256": canonical_sha256(manifest),
                "membership_sha256": manifest["membership_sha256"],
                "panel_sha256s": manifest["panel_sha256s"],
                "authorship": "human-recorded-in-working-notebook",
            }
            try:
                validate_instance("degradation-review", record_value, version=2)
                _write_atomic(
                    self.artifact_root / "degradation-decision.json", _json_bytes(record_value)
                )
            except Exception as error:
                status.value = f"<b>Not saved:</b> {error}"
                return
            status.value = (
                "<b>Decision saved durably for controlled-score-v2-candidate.</b> "
                "Return to Codex and type <code>decision recorded</code>."
            )

        save.on_click(record)
        interface = widgets.VBox([reviewer, decision, rationale, save, status])
        display(interface)
        return interface
