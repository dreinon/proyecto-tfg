"""Deterministic controlled degradation and project-authored fixture contracts.

This module deliberately has no dataset or network loader.  Its only accepted source material is
the closed project-authored fixture manifest used to review Phase 2 controls while SMB remains
``AUDITED_LOCKED``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.identities import canonical_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROL_PATH = PROJECT_ROOT / "configs/degradations/controlled-score-candidates.yaml"
DEFAULT_FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "tests/fixtures/phase2/fixture-manifest-v1.yaml"

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


class DegradationContractError(ValueError):
    """A candidate registry is incomplete, unsafe, or scientifically inconsistent."""


class FixtureValidationError(ValueError):
    """A fixture manifest or generated fixture fails the authored-fixture contract."""


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
        expected = (
            (
                {"type": "gaussian", "sigma": 0.8, "kernel": 7},
                {"type": "gaussian", "sigma": 3.0},
                85,
            )
            if severity == "moderate"
            else (
                {"type": "gaussian", "sigma": 1.6, "kernel": 11},
                {"type": "gaussian", "sigma": 8.0},
                60,
            )
        )
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
        registry = _read_regular_yaml(
            path, maximum_bytes=_MAX_CONTROL_BYTES, kind="degradation control"
        )
        secret = _secret_like_key((), registry)
        if secret is not None:
            raise DegradationContractError(f"secret-like key is forbidden: {secret}")
        _validate_registry_semantics(registry)
        validate_instance("degradation-control", registry, version=2)
        candidates = registry["candidates"]
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
    except (ContractValidationError, KeyError, TypeError, ValueError) as error:
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
