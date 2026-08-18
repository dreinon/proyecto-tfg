"""Fixture-testable SMB audit and immutable manifest generation boundary."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import imagehash
import yaml
from PIL import Image, UnidentifiedImageError

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    assert_smb_purpose_allowed,
)
from score_super_resolution.contracts import ContractValidationError, validate_instance

EXPECTED_ROW_COUNT = 685
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_MAX_ENCODED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PIXELS = 100_000_000
HASH_SIZE = 8
HIGHFREQ_FACTOR = 4
MAXIMUM_HAMMING_DISTANCE = 6
IMAGEHASH_VERSION = importlib.metadata.version("ImageHash")
GENERATION_DOMAIN = b"smb-manifest-generation-v1\0"
REVIEW_CSV_FIELDS = (
    "review_kind",
    "review_key",
    "item_id",
    "candidate_item_id",
    "review_status",
    "reviewer",
    "reviewed_at",
    "rationale",
    "source_group_id",
    "quality_disposition",
    "suitability_disposition",
    "duplicate_disposition",
    "dataset_licence_status",
    "item_provenance_status",
    "access_status",
    "redistribution_status",
    "figure_reproduction_status",
)
PUBLICATION_BOUNDARIES = (
    "generation_records_written",
    "generation_records_fsynced",
    "generation_descriptor_written",
    "generation_descriptor_fsynced",
    "temporary_generation_directory_fsynced",
    "generation_renamed",
    "generations_parent_fsynced",
    "pointer_written",
    "pointer_fsynced",
    "pointer_replaced",
    "active_parent_fsynced",
)

_SAFE_METADATA_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SAFE_UPSTREAM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class ManifestPublicationError(RuntimeError):
    """Report a generation integrity/publication failure and pointer commit state."""

    def __init__(self, message: str, *, committed: bool) -> None:
        self.committed = committed
        super().__init__(message)


class ReviewFinalizationError(ValueError):
    """Report an invalid or incomplete human review without changing evidence."""


def _review_error(detail: str) -> ReviewFinalizationError:
    return ReviewFinalizationError(f"SMB review error: {detail}")


def _publication_error(detail: str, *, committed: bool = False) -> ManifestPublicationError:
    return ManifestPublicationError(f"SMB manifest error: {detail}", committed=committed)


def _canonical_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _canonical_descriptor(descriptor: Mapping[str, object]) -> bytes:
    return yaml.safe_dump(
        dict(descriptor), allow_unicode=True, sort_keys=True, default_flow_style=False
    ).encode("utf-8")


def _generation_id(descriptor: Mapping[str, object], records_bytes: bytes) -> str:
    identity_descriptor = dict(descriptor)
    identity_descriptor.pop("generation_id", None)
    payload = GENERATION_DOMAIN + _canonical_descriptor(identity_descriptor) + b"\0" + records_bytes
    return hashlib.sha256(payload).hexdigest()


def _normalize_metadata(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    if not text:
        return None
    return text[:128] if _SAFE_METADATA_PATTERN.fullmatch(text[:128]) else None


def _raw_normalized(value: object) -> dict[str, object]:
    raw = value if isinstance(value, (str, int)) and not isinstance(value, bool) else None
    return {"raw": raw, "normalized": _normalize_metadata(raw)}


def _safe_item_id(upstream_index: int) -> str:
    if isinstance(upstream_index, bool) or not isinstance(upstream_index, int):
        raise ValueError("upstream_index must be an integer")
    if not 0 <= upstream_index < EXPECTED_ROW_COUNT:
        raise ValueError("upstream_index must be between 0 and 684")
    return f"smb-test-{upstream_index:06d}"


def _nullable_positive_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _encoded_image(record: Mapping[str, object]) -> bytes:
    image = record.get("image")
    if isinstance(image, bytes):
        return image
    if isinstance(image, bytearray):
        return bytes(image)
    if isinstance(image, Mapping):
        encoded = image.get("bytes")
        if isinstance(encoded, bytes):
            return encoded
        if isinstance(encoded, bytearray):
            return bytes(encoded)
    raise ValueError("image_bytes_unavailable")


def _regions_valid(regions: object, width: int, height: int) -> tuple[int, bool, list[str]]:
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes, bytearray)):
        return 0, False, ["regions_not_sequence"]
    failures: list[str] = []
    for index, region in enumerate(regions):
        bbox = region.get("bbox") if isinstance(region, Mapping) else None
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes, bytearray))
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
        ):
            failures.append(f"region_{index}_invalid_bbox")
            continue
        left, top, right, bottom = bbox
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            failures.append(f"region_{index}_out_of_bounds")
    return len(regions), not failures, failures


def _rights(*, item_provenance: str = "pending") -> dict[str, str]:
    return {
        "dataset_licence_status": "pending",
        "item_provenance_status": item_provenance,
        "access_status": "confirmed",
        "redistribution_status": "pending",
        "figure_reproduction_status": "pending",
    }


def _duplicate_review(*, available: bool = True) -> dict[str, object]:
    return {
        "review_status": "pending" if available else "not_applicable",
        "disposition": "pending" if available else "unavailable",
        "reviewer": None,
        "reviewed_at": None,
        "rationale": "" if available else "unprocessable",
    }


def _base_row(record: Mapping[str, object], upstream_index: int) -> dict[str, object]:
    original_score = _raw_normalized(record.get("original_score"))
    page = _raw_normalized(record.get("page"))
    page_texture = _raw_normalized(record.get("page_texture"))
    raw_upstream_id = record.get("id")
    annotations: list[str] = []
    if (
        not isinstance(raw_upstream_id, str)
        or _SAFE_UPSTREAM_ID_PATTERN.fullmatch(raw_upstream_id) is None
    ):
        annotations.append("unsafe_upstream_id")
    expected_status = record.get("expected_status", "processable")
    if expected_status not in {"processable", "unprocessable"}:
        expected_status = "unprocessable"
        annotations.append("invalid_expected_status")
    return {
        "schema_version": 1,
        "record_type": "manifest-row",
        "manifest_version": 1,
        "source_key": "smb",
        "source_revision": "",
        "split": "test",
        "upstream_index": upstream_index,
        "item_id": _safe_item_id(upstream_index),
        "original_score": original_score,
        "source_group_id": original_score["normalized"],
        "page": page,
        "page_texture": page_texture,
        "encoded_sha256": None,
        "pixel_sha256": None,
        "declared_width": _nullable_positive_integer(record.get("original_width")),
        "declared_height": _nullable_positive_integer(record.get("original_height")),
        "decoded_width": None,
        "decoded_height": None,
        "image_mode": None,
        "image_format": None,
        "byte_count": None,
        "region_count": None,
        "bbox_valid": None,
        "required_text_present": None,
        "annotation_failures": annotations,
        "quality": {
            "review_status": "pending",
            "flags": [],
            "suitability_disposition": "pending",
            "notes": "",
        },
        "exact_duplicate_group": None,
        "near_duplicate_candidate_ids": [],
        "duplicate_review": _duplicate_review(),
        "expected_status": expected_status,
        "processing_status": "failed",
        "unprocessable_reason": "audit_not_completed",
        "audit_sample_member": False,
        "rights": _rights(),
        "paired_eligible": False,
        "paired_ineligibility_reason": "audit_not_completed",
    }


def _mark_failure(
    row: dict[str, object], reason: str, *, quality_flag: str = "unprocessable"
) -> dict[str, object]:
    row["processing_status"] = "failed"
    row["unprocessable_reason"] = reason
    row["paired_eligible"] = False
    row["paired_ineligibility_reason"] = reason
    row["annotation_failures"] = sorted(set([*row["annotation_failures"], reason]))  # type: ignore[arg-type]
    row["quality"] = {
        "review_status": "not_applicable",
        "flags": [quality_flag],
        "suitability_disposition": "unavailable",
        "notes": reason,
    }
    row["duplicate_review"] = _duplicate_review(available=False)
    row["rights"] = _rights(item_provenance="unavailable")
    validate_instance("manifest-row", row)
    return row


def _audit_after_guard(
    record: Mapping[str, object],
    *,
    upstream_index: int,
    source_revision: str,
    max_encoded_bytes: int,
    max_pixels: int,
) -> tuple[dict[str, object], imagehash.ImageHash | None]:
    row = _base_row(record, upstream_index)
    row["source_revision"] = source_revision
    try:
        encoded = _encoded_image(record)
    except ValueError:
        return _mark_failure(row, "image_bytes_unavailable"), None
    row["byte_count"] = len(encoded)
    if len(encoded) > max_encoded_bytes:
        return _mark_failure(row, "encoded_image_too_large", quality_flag="oversized"), None
    row["encoded_sha256"] = hashlib.sha256(encoded).hexdigest()

    try:
        with Image.open(io.BytesIO(encoded)) as opened:
            width, height = opened.size
            row["decoded_width"] = width
            row["decoded_height"] = height
            row["image_mode"] = opened.mode
            row["image_format"] = opened.format
            if width < 1 or height < 1 or width * height > max_pixels:
                return _mark_failure(row, "image_too_large", quality_flag="oversized"), None
            opened.load()
            canonical = opened.convert("RGBA")
            pixel_bytes = canonical.tobytes()
            row["pixel_sha256"] = hashlib.sha256(pixel_bytes).hexdigest()
            perceptual_hash = imagehash.phash(
                canonical, hash_size=HASH_SIZE, highfreq_factor=HIGHFREQ_FACTOR
            )
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return _mark_failure(row, "decode_failed"), None

    region_count, bbox_valid, region_failures = _regions_valid(record.get("regions"), width, height)
    row["region_count"] = region_count
    row["bbox_valid"] = bbox_valid
    row["annotation_failures"] = sorted(
        set([*row["annotation_failures"], *region_failures])  # type: ignore[arg-type]
    )
    required_text_present = all(
        metadata["normalized"] is not None
        for metadata in (row["original_score"], row["page"], row["page_texture"])
    )
    row["required_text_present"] = required_text_present
    row["processing_status"] = "processed"
    row["unprocessable_reason"] = None
    if not required_text_present:
        paired_reason = "missing_required_metadata"
    elif not bbox_valid:
        paired_reason = "invalid_region_annotation"
    else:
        paired_reason = None
    row["paired_eligible"] = paired_reason is None
    row["paired_ineligibility_reason"] = paired_reason
    validate_instance("manifest-row", row)
    return row, perceptual_hash


def audit_item(
    record: Mapping[str, object],
    *,
    upstream_index: int,
    source_descriptor: Mapping[str, object],
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, object]:
    """Audit one item after the locked-benchmark guard, returning one strict row."""

    result = assert_smb_purpose_allowed(
        source_descriptor=source_descriptor,
        purpose=BenchmarkPurpose.CONTENT_AUDIT,
        callback=lambda: _audit_after_guard(
            record,
            upstream_index=upstream_index,
            source_revision=str(source_descriptor.get("revision", "")),
            max_encoded_bytes=max_encoded_bytes,
            max_pixels=max_pixels,
        ),
    )
    if not isinstance(result, tuple):
        raise RuntimeError("SMB audit guard returned no item result")
    return result[0]


def select_visual_sample(
    rows: Sequence[Mapping[str, object]], *, seed: int, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> list[str]:
    """Select a fixed pre-review sample using identity fields only."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer")
    if sample_size > len(rows):
        raise ValueError("sample_size cannot exceed the population")
    ranked: list[tuple[str, str]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        upstream_index = row.get("upstream_index")
        item_id = row.get("item_id")
        if (
            isinstance(upstream_index, bool)
            or not isinstance(upstream_index, int)
            or not isinstance(item_id, str)
        ):
            raise ValueError("sample identities require integer upstream_index and string item_id")
        identity = (upstream_index, item_id)
        if identity in seen:
            raise ValueError("sample identities must be unique")
        seen.add(identity)
        rank = hashlib.sha256(
            f"sha256-rank-v1\0{seed}\0{upstream_index}\0{item_id}".encode()
        ).hexdigest()
        ranked.append((rank, item_id))
    return [item_id for _, item_id in sorted(ranked)[:sample_size]]


def _candidate_id(first_item_id: str, second_item_id: str) -> str:
    first, second = sorted((first_item_id, second_item_id))
    digest = hashlib.sha256(f"near-duplicate-v1\0{first}\0{second}".encode()).hexdigest()
    return f"candidate-{digest[:16]}"


def audit_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    source_descriptor: Mapping[str, object],
    deterministic_seed: int = 20260818,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, object]]:
    """Audit fixtures safely, preserve every input, and label duplicate candidates."""

    audited: list[tuple[dict[str, object], imagehash.ImageHash | None]] = []
    for upstream_index, record in enumerate(records):
        result = assert_smb_purpose_allowed(
            source_descriptor=source_descriptor,
            purpose=BenchmarkPurpose.CONTENT_AUDIT,
            callback=lambda record=record, upstream_index=upstream_index: _audit_after_guard(
                record,
                upstream_index=upstream_index,
                source_revision=str(source_descriptor.get("revision", "")),
                max_encoded_bytes=max_encoded_bytes,
                max_pixels=max_pixels,
            ),
        )
        if not isinstance(result, tuple):
            raise RuntimeError("SMB audit guard returned no item result")
        audited.append(result)

    rows = [row for row, _ in audited]
    selected = set(select_visual_sample(rows, seed=deterministic_seed, sample_size=sample_size))
    for row in rows:
        row["audit_sample_member"] = row["item_id"] in selected

    exact_groups: dict[tuple[object, object], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["processing_status"] == "processed":
            exact_groups[(row["encoded_sha256"], row["pixel_sha256"])].append(index)
    for exact_key, indices in exact_groups.items():
        if len(indices) < 2:
            continue
        digest = hashlib.sha256(
            f"exact-duplicate-v1\0{exact_key[0]}\0{exact_key[1]}".encode()
        ).hexdigest()
        group_id = f"exact-{digest[:16]}"
        for index in indices:
            rows[index]["exact_duplicate_group"] = group_id

    for first_index, (first_row, first_hash) in enumerate(audited):
        if first_hash is None:
            continue
        for second_index in range(first_index + 1, len(audited)):
            second_row, second_hash = audited[second_index]
            if second_hash is None or first_row["pixel_sha256"] == second_row["pixel_sha256"]:
                continue
            if first_hash - second_hash <= MAXIMUM_HAMMING_DISTANCE:
                candidate_id = _candidate_id(str(first_row["item_id"]), str(second_row["item_id"]))
                first_row["near_duplicate_candidate_ids"].append(candidate_id)  # type: ignore[union-attr]
                second_row["near_duplicate_candidate_ids"].append(candidate_id)  # type: ignore[union-attr]

    for row in rows:
        row["near_duplicate_candidate_ids"] = sorted(row["near_duplicate_candidate_ids"])  # type: ignore[arg-type]
        validate_instance("manifest-row", row)
    return rows


def _publication_boundary(name: str, hook: Callable[[str], None] | None) -> None:
    if name not in PUBLICATION_BOUNDARIES:
        raise ValueError(f"unknown publication boundary: {name}")
    if hook is not None:
        hook(name)
    failpoint = os.environ.get("SCORE_SR_SMB_PUBLICATION_FAILPOINT")
    if failpoint == f"{name}:raise":
        raise OSError(f"injected publication failure after {name}")
    if failpoint == f"{name}:exit":
        os._exit(91)


def _write_fsynced(
    path: Path,
    content: bytes,
    *,
    written_boundary: str,
    fsynced_boundary: str,
    boundary_hook: Callable[[str], None] | None,
) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        _publication_boundary(written_boundary, boundary_hook)
        os.fsync(handle.fileno())
        _publication_boundary(fsynced_boundary, boundary_hook)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_generation_inputs(
    descriptor: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    try:
        for index, row in enumerate(rows):
            try:
                validate_instance("manifest-row", row)
            except ContractValidationError as error:
                raise _publication_error(
                    f"manifest-row[{index}] failed validation: {error}"
                ) from error
        records_bytes = _canonical_jsonl(rows)
        records_sha256 = hashlib.sha256(records_bytes).hexdigest()
        completed_descriptor = dict(descriptor)
        completed_descriptor["row_count"] = len(rows)
        completed_descriptor["records_sha256"] = records_sha256
        completed_descriptor.pop("generation_id", None)
        duplicate_provenance = completed_descriptor.get("duplicate_provenance")
        near_provenance = (
            duplicate_provenance.get("near") if isinstance(duplicate_provenance, Mapping) else None
        )
        if (
            not isinstance(near_provenance, Mapping)
            or near_provenance.get("library_version") != IMAGEHASH_VERSION
        ):
            raise _publication_error(
                "manifest ImageHash provenance does not match the installed audit library"
            )
        generation_id = _generation_id(completed_descriptor, records_bytes)
        completed_descriptor["generation_id"] = generation_id
        validate_instance("manifest-descriptor", completed_descriptor)
        descriptor_bytes = _canonical_descriptor(completed_descriptor)
        pointer = {
            "schema_version": 1,
            "record_type": "manifest-active",
            "manifest_id": completed_descriptor["manifest_id"],
            "generation_id": generation_id,
            "descriptor_path": f"{generation_id}/manifest-descriptor.yaml",
            "records_path": f"{generation_id}/manifest-records.jsonl",
            "row_schema_id": completed_descriptor["row_schema_id"],
            "row_schema_version": completed_descriptor["row_schema_version"],
            "row_count": completed_descriptor["row_count"],
            "records_sha256": records_sha256,
        }
        validate_instance("manifest-active", pointer)
    except ManifestPublicationError:
        raise
    except (ContractValidationError, TypeError, ValueError) as error:
        raise _publication_error(f"generation validation failed: {error}") from error
    return completed_descriptor, pointer, descriptor_bytes, records_bytes


def publish_manifest_generation(
    *,
    active_path: Path,
    generation_root: Path,
    descriptor: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    boundary_hook: Callable[[str], None] | None = None,
) -> None:
    """Validate and publish one immutable content-addressed generation through one pointer."""

    _, pointer, descriptor_bytes, records_bytes = _validate_generation_inputs(descriptor, rows)
    generation_id = str(pointer["generation_id"])
    root = generation_root.resolve()
    generation_path = (root / generation_id).resolve()
    if generation_path.parent != root:
        raise _publication_error("generation path escapes generation root")
    temp_generation = root / f".tmp-{generation_id}-{uuid.uuid4().hex}"
    committed = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        if generation_path.exists():
            descriptor_path = generation_path / "manifest-descriptor.yaml"
            records_path = generation_path / "manifest-records.jsonl"
            if (
                not descriptor_path.is_file()
                or not records_path.is_file()
                or descriptor_path.read_bytes() != descriptor_bytes
                or records_path.read_bytes() != records_bytes
            ):
                raise _publication_error("existing generation is not byte-identical")
        else:
            temp_generation.mkdir()
            _write_fsynced(
                temp_generation / "manifest-records.jsonl",
                records_bytes,
                written_boundary="generation_records_written",
                fsynced_boundary="generation_records_fsynced",
                boundary_hook=boundary_hook,
            )
            _write_fsynced(
                temp_generation / "manifest-descriptor.yaml",
                descriptor_bytes,
                written_boundary="generation_descriptor_written",
                fsynced_boundary="generation_descriptor_fsynced",
                boundary_hook=boundary_hook,
            )
            _fsync_directory(temp_generation)
            _publication_boundary("temporary_generation_directory_fsynced", boundary_hook)
            os.replace(temp_generation, generation_path)
            _publication_boundary("generation_renamed", boundary_hook)
            _fsync_directory(root)
            _publication_boundary("generations_parent_fsynced", boundary_hook)

        pointer_bytes = _canonical_descriptor(pointer)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        if active_path.is_file() and active_path.read_bytes() == pointer_bytes:
            return
        pointer_temp = active_path.parent / f".{active_path.name}.tmp-{uuid.uuid4().hex}"
        try:
            _write_fsynced(
                pointer_temp,
                pointer_bytes,
                written_boundary="pointer_written",
                fsynced_boundary="pointer_fsynced",
                boundary_hook=boundary_hook,
            )
            os.replace(pointer_temp, active_path)
            committed = True
            _publication_boundary("pointer_replaced", boundary_hook)
            _fsync_directory(active_path.parent)
            _publication_boundary("active_parent_fsynced", boundary_hook)
        finally:
            if pointer_temp.exists():
                pointer_temp.unlink()
    except ManifestPublicationError:
        raise
    except Exception as error:
        detail = error.strerror if isinstance(error, OSError) else type(error).__name__
        raise _publication_error(f"publication failed: {detail}", committed=committed) from error
    finally:
        if temp_generation.exists():
            shutil.rmtree(temp_generation)


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise _publication_error(
            f"cannot read {label}: {type(error).__name__}", committed=True
        ) from error
    if not isinstance(loaded, dict):
        raise _publication_error(f"{label} must be a mapping", committed=True)
    return loaded


def _resolved_named_path(root: Path, relative: object, *, expected: str) -> Path:
    if relative != expected:
        raise _publication_error(
            "active pointer names a non-canonical generation path", committed=True
        )
    root = root.resolve()
    resolved = (root / expected).resolve()
    if not resolved.is_relative_to(root):
        raise _publication_error("active pointer path escapes generation root", committed=True)
    return resolved


def resolve_active_manifest(
    *, active_path: Path, generation_root: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Resolve and validate exactly the generation named by the active pointer."""

    pointer = _load_yaml_mapping(active_path, label="active pointer")
    validate_instance("manifest-active", pointer)
    generation_id = str(pointer["generation_id"])
    descriptor_relative = f"{generation_id}/manifest-descriptor.yaml"
    records_relative = f"{generation_id}/manifest-records.jsonl"
    descriptor_path = _resolved_named_path(
        generation_root, pointer["descriptor_path"], expected=descriptor_relative
    )
    records_path = _resolved_named_path(
        generation_root, pointer["records_path"], expected=records_relative
    )
    descriptor = _load_yaml_mapping(descriptor_path, label="generation descriptor")
    validate_instance("manifest-descriptor", descriptor)
    try:
        records_bytes = records_path.read_bytes()
    except OSError as error:
        raise _publication_error("cannot read generation records", committed=True) from error
    records_sha256 = hashlib.sha256(records_bytes).hexdigest()
    if (
        records_sha256 != pointer["records_sha256"]
        or records_sha256 != descriptor["records_sha256"]
    ):
        raise _publication_error("generation records checksum mismatch", committed=True)
    expected_generation_id = _generation_id(descriptor, records_bytes)
    if expected_generation_id != generation_id or descriptor["generation_id"] != generation_id:
        raise _publication_error(
            "content address does not match generation identity", committed=True
        )

    rows: list[dict[str, object]] = []
    try:
        text = records_bytes.decode("utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("records JSONL lacks final newline")
        for index, line in enumerate(text.splitlines()):
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"row {index} is not an object")
            validate_instance("manifest-row", loaded)
            rows.append(loaded)
    except (UnicodeError, json.JSONDecodeError, ValueError, ContractValidationError) as error:
        raise _publication_error(
            f"generation records failed validation: {error}", committed=True
        ) from error

    agreement_fields = (
        "manifest_id",
        "generation_id",
        "row_schema_id",
        "row_schema_version",
        "row_count",
        "records_sha256",
    )
    if any(pointer[field] != descriptor[field] for field in agreement_fields):
        raise _publication_error("pointer and descriptor disagree", committed=True)
    if len(rows) != pointer["row_count"]:
        raise _publication_error("generation row count mismatch", committed=True)
    return descriptor, rows


def reconcile_manifest(
    *,
    active_path: Path,
    generation_root: Path,
    expected_indices: Iterable[int] = range(EXPECTED_ROW_COUNT),
) -> dict[str, int]:
    """Reconcile all expected indices only after active-generation resolution succeeds."""

    descriptor, rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    if descriptor["benchmark_state"] != "AUDITED_LOCKED":
        raise _publication_error("manifest benchmark state is not AUDITED_LOCKED", committed=True)
    expected = set(expected_indices)
    counts = Counter(row["upstream_index"] for row in rows)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    actual = set(counts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if duplicates or missing or unexpected:
        raise _publication_error(
            f"duplicate indices={duplicates}; missing indices={missing}; "
            f"unexpected indices={unexpected}",
            committed=True,
        )
    return {
        "row_count": len(rows),
        "processed": sum(row["processing_status"] == "processed" for row in rows),
        "failed": sum(row["processing_status"] == "failed" for row in rows),
        "paired_eligible": sum(row["paired_eligible"] is True for row in rows),
    }


def _neutralize_display(value: object) -> str:
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(_FORMULA_PREFIXES) else text


def _review_row(row: Mapping[str, object]) -> dict[str, str]:
    quality = row["quality"]
    duplicate_review = row["duplicate_review"]
    rights = row["rights"]
    assert isinstance(quality, Mapping)
    assert isinstance(duplicate_review, Mapping)
    assert isinstance(rights, Mapping)
    return {
        "review_kind": "item",
        "review_key": str(row["item_id"]),
        "item_id": str(row["item_id"]),
        "candidate_item_id": "",
        "review_status": str(quality["review_status"]),
        "reviewer": _neutralize_display(duplicate_review["reviewer"]),
        "reviewed_at": _neutralize_display(duplicate_review["reviewed_at"]),
        "rationale": _neutralize_display(duplicate_review["rationale"]),
        "source_group_id": str(row["source_group_id"] or ""),
        "quality_disposition": ";".join(str(flag) for flag in quality["flags"]),
        "suitability_disposition": str(quality["suitability_disposition"]),
        "duplicate_disposition": str(duplicate_review["disposition"]),
        "dataset_licence_status": str(rights["dataset_licence_status"]),
        "item_provenance_status": str(rights["item_provenance_status"]),
        "access_status": str(rights["access_status"]),
        "redistribution_status": str(rights["redistribution_status"]),
        "figure_reproduction_status": str(rights["figure_reproduction_status"]),
    }


def _candidate_review_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    candidates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        candidate_ids = row["near_duplicate_candidate_ids"]
        assert isinstance(candidate_ids, Sequence)
        for candidate_id in candidate_ids:
            candidates[str(candidate_id)].append(str(row["item_id"]))
    review_rows: list[dict[str, str]] = []
    for candidate_id, item_ids in sorted(candidates.items()):
        unique_ids = sorted(set(item_ids))
        if len(unique_ids) != 2 or len(item_ids) != 2:
            raise _review_error(
                f"candidate {candidate_id} must identify exactly two unique manifest rows"
            )
        if candidate_id != _candidate_id(*unique_ids):
            raise _review_error(f"candidate {candidate_id} is not the canonical key for its pair")
        review_rows.append(
            {
                **{field: "" for field in REVIEW_CSV_FIELDS},
                "review_kind": "candidate",
                "review_key": candidate_id,
                "item_id": unique_ids[0],
                "candidate_item_id": unique_ids[1],
                "review_status": "pending",
                "duplicate_disposition": "pending",
            }
        )
    return review_rows


def _expected_review_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    return [*(_review_row(row) for row in rows), *_candidate_review_rows(rows)]


def _write_review_rows(path: Path, review_rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(review_rows)


def write_review_csv(*, active_path: Path, generation_root: Path, output_path: Path) -> None:
    """Write redacted review rows derived exclusively from validated active resolution."""

    _, rows = resolve_active_manifest(active_path=active_path, generation_root=generation_root)
    _write_review_rows(output_path, _expected_review_rows(rows))


def _write_csv(
    path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def emit_review_evidence_from_active_manifest(
    *,
    active_path: Path,
    generation_root: Path,
    audit_descriptor_path: Path,
    audit_records_path: Path,
    sample_path: Path,
    review_path: Path,
) -> None:
    """Regenerate tracked, redacted review evidence from one validated active generation."""

    descriptor, rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    audit_descriptor = {
        "audit_version": descriptor["audit_version"],
        "benchmark_state": descriptor["benchmark_state"],
        "manifest_id": descriptor["manifest_id"],
        "record_type": "smb-audit-export",
        "records_sha256": descriptor["records_sha256"],
        "row_count": len(rows),
        "schema_version": 1,
        "source_key": descriptor["source_key"],
        "source_revision": descriptor["source_revision"],
    }
    redacted_rows = [
        {
            "audit_sample_member": row["audit_sample_member"],
            "item_id": row["item_id"],
            "processing_status": row["processing_status"],
            "source_group_id": row["source_group_id"],
            "upstream_index": row["upstream_index"],
        }
        for row in rows
    ]
    sample_rows = [row for row in redacted_rows if row["audit_sample_member"] is True]
    audit_descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    audit_descriptor_path.write_bytes(_canonical_descriptor(audit_descriptor))
    audit_records_path.parent.mkdir(parents=True, exist_ok=True)
    audit_records_path.write_bytes(_canonical_jsonl(redacted_rows))
    _write_csv(
        sample_path,
        fieldnames=(
            "upstream_index",
            "item_id",
            "source_group_id",
            "processing_status",
            "audit_sample_member",
        ),
        rows=sample_rows,
    )
    _write_review_rows(review_path, _expected_review_rows(rows))


def _read_review_rows(review_path: Path) -> list[dict[str, str]]:
    try:
        with review_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REVIEW_CSV_FIELDS:
                raise _review_error("CSV header does not match the exact review contract")
            rows = list(reader)
    except ReviewFinalizationError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise _review_error(f"cannot read review CSV: {type(error).__name__}") from error
    if any(None in row or set(row) != set(REVIEW_CSV_FIELDS) for row in rows):
        raise _review_error("review CSV row does not match the exact header")
    return [{field: str(row[field]) for field in REVIEW_CSV_FIELDS} for row in rows]


_QUALITY_DISPOSITIONS = {
    "blurred",
    "low_contrast",
    "oversized",
    "skewed",
    "unprocessable",
}
_SUITABILITY_DISPOSITIONS = {"suitable", "unsuitable", "unavailable"}
_DUPLICATE_DISPOSITIONS = {"distinct", "duplicate", "related", "unavailable"}
_DATASET_LICENCE_STATUSES = {"confirmed", "restricted"}
_ITEM_PROVENANCE_STATUSES = {"confirmed", "unavailable"}
_ACCESS_STATUSES = {"confirmed", "restricted"}
_REDISTRIBUTION_STATUSES = {"permitted", "prohibited"}
_FIGURE_REPRODUCTION_STATUSES = {"permitted", "prohibited"}


def _quality_flags(value: str, *, review_key: str) -> list[str]:
    if value == "acceptable":
        return []
    flags = value.split(";")
    if not flags or any(flag not in _QUALITY_DISPOSITIONS for flag in flags):
        raise _review_error(f"{review_key}: invalid quality_disposition")
    if flags != sorted(set(flags)):
        raise _review_error(f"{review_key}: quality_disposition must be unique and canonical")
    return flags


def _require_enum(row: Mapping[str, str], field: str, allowed: set[str]) -> None:
    if row[field] not in allowed:
        raise _review_error(f"{row['review_key']}: invalid {field}")


def _validated_review_rows(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    expected_rows = _expected_review_rows(rows)
    expected = {row["review_key"]: row for row in expected_rows}
    seen: dict[str, dict[str, str]] = {}
    for untrusted in review_rows:
        if set(untrusted) != set(REVIEW_CSV_FIELDS):
            raise _review_error("review row does not match the exact header")
        row = {field: str(untrusted[field]) for field in REVIEW_CSV_FIELDS}
        key = row["review_key"]
        if key in seen:
            raise _review_error(f"duplicate review key: {key}")
        if key not in expected:
            raise _review_error(f"unknown review key: {key}")
        expected_row = expected[key]
        if row["review_kind"] != expected_row["review_kind"]:
            raise _review_error(f"{key}: review_kind does not match the emitted key")
        if row["review_kind"] == "candidate" and (row["item_id"], row["candidate_item_id"]) != (
            expected_row["item_id"],
            expected_row["candidate_item_id"],
        ):
            raise _review_error(f"{key}: candidate pair is not in canonical order")
        if row["item_id"] != expected_row["item_id"]:
            raise _review_error(f"{key}: item_id does not match the emitted key")
        if row["candidate_item_id"] != expected_row["candidate_item_id"]:
            raise _review_error(f"{key}: candidate_item_id does not match the emitted key")
        if row["review_status"] != "reviewed":
            raise _review_error(f"{key}: review_status must be reviewed")
        for field in ("reviewer", "rationale"):
            if not row[field].strip():
                raise _review_error(f"{key}: {field} is required")
        try:
            if date.fromisoformat(row["reviewed_at"]).isoformat() != row["reviewed_at"]:
                raise ValueError
        except ValueError:
            raise _review_error(f"{key}: reviewed_at must be an ISO date") from None

        if row["review_kind"] == "candidate":
            _require_enum(row, "duplicate_disposition", _DUPLICATE_DISPOSITIONS)
            irrelevant = (
                "source_group_id",
                "quality_disposition",
                "suitability_disposition",
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            )
            if any(row[field] for field in irrelevant):
                raise _review_error(f"{key}: candidate row contains item-only dispositions")
        else:
            _quality_flags(row["quality_disposition"], review_key=key)
            _require_enum(row, "suitability_disposition", _SUITABILITY_DISPOSITIONS)
            _require_enum(row, "duplicate_disposition", _DUPLICATE_DISPOSITIONS)
            _require_enum(row, "dataset_licence_status", _DATASET_LICENCE_STATUSES)
            _require_enum(row, "item_provenance_status", _ITEM_PROVENANCE_STATUSES)
            _require_enum(row, "access_status", _ACCESS_STATUSES)
            _require_enum(row, "redistribution_status", _REDISTRIBUTION_STATUSES)
            _require_enum(row, "figure_reproduction_status", _FIGURE_REPRODUCTION_STATUSES)
            source_group_id = row["source_group_id"]
            if source_group_id and _SAFE_METADATA_PATTERN.fullmatch(source_group_id) is None:
                raise _review_error(f"{key}: invalid source_group_id")
        seen[key] = row

    missing = sorted(set(expected) - set(seen))
    if missing:
        raise _review_error(f"missing review key: {missing[0]}")

    item_dispositions = {
        key: row["duplicate_disposition"]
        for key, row in seen.items()
        if row["review_kind"] == "item"
    }
    candidate_updates: dict[str, dict[str, str]] = {}
    for row in seen.values():
        if row["review_kind"] != "candidate":
            continue
        for item_id in (row["item_id"], row["candidate_item_id"]):
            if item_dispositions[item_id] != row["duplicate_disposition"]:
                raise _review_error(f"{item_id}: ambiguous duplicate dispositions")
            existing = candidate_updates.get(item_id)
            if (
                existing is not None
                and existing["duplicate_disposition"] != row["duplicate_disposition"]
            ):
                raise _review_error(f"{item_id}: ambiguous candidate dispositions")
            candidate_updates[item_id] = row
    return [seen[row["review_key"]] for row in expected_rows]


def apply_review_dispositions(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    """Apply one complete stable-key review to copied manifest rows."""

    validated = _validated_review_rows(rows, review_rows)
    updated = copy.deepcopy(list(rows))
    by_id = {str(row["item_id"]): row for row in updated}
    if len(by_id) != len(updated):
        raise _review_error("manifest contains duplicate item identities")
    for review in validated:
        if review["review_kind"] != "item":
            continue
        row = by_id[review["item_id"]]
        row["source_group_id"] = review["source_group_id"] or None
        quality = row["quality"]
        duplicate_review = row["duplicate_review"]
        rights = row["rights"]
        assert isinstance(quality, dict)
        assert isinstance(duplicate_review, dict)
        assert isinstance(rights, dict)
        quality.update(
            {
                "review_status": "reviewed",
                "flags": _quality_flags(
                    review["quality_disposition"], review_key=review["review_key"]
                ),
                "suitability_disposition": review["suitability_disposition"],
                "notes": review["rationale"],
            }
        )
        duplicate_review.update(
            {
                "review_status": "reviewed",
                "disposition": review["duplicate_disposition"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "rationale": review["rationale"],
            }
        )
        for field in (
            "dataset_licence_status",
            "item_provenance_status",
            "access_status",
            "redistribution_status",
            "figure_reproduction_status",
        ):
            rights[field] = review[field]

    for review in validated:
        if review["review_kind"] != "candidate":
            continue
        disposition = {
            "review_status": "reviewed",
            "disposition": review["duplicate_disposition"],
            "reviewer": review["reviewer"],
            "reviewed_at": review["reviewed_at"],
            "rationale": review["rationale"],
        }
        for item_id in (review["item_id"], review["candidate_item_id"]):
            by_id[item_id]["duplicate_review"] = copy.deepcopy(disposition)

    for index, row in enumerate(updated):
        try:
            validate_instance("manifest-row", row)
        except ContractValidationError as error:
            raise _review_error(
                f"updated manifest-row[{index}] failed validation: {error}"
            ) from error
    return updated


def validate_review_from_active_manifest(
    *, review_path: Path, active_path: Path, generation_root: Path
) -> None:
    """Validate an existing review against the active manifest without writing any file."""

    _, rows = resolve_active_manifest(active_path=active_path, generation_root=generation_root)
    apply_review_dispositions(rows, _read_review_rows(review_path))


def _require_complete_denominator(rows: Sequence[Mapping[str, object]]) -> None:
    expected_ids = {f"smb-test-{index:06d}" for index in range(EXPECTED_ROW_COUNT)}
    actual_ids = [str(row["item_id"]) for row in rows]
    actual_indices = [row["upstream_index"] for row in rows]
    if (
        len(rows) != EXPECTED_ROW_COUNT
        or len(set(actual_ids)) != EXPECTED_ROW_COUNT
        or set(actual_ids) != expected_ids
        or set(actual_indices) != set(range(EXPECTED_ROW_COUNT))
    ):
        raise _review_error("finalized manifest does not preserve all 685 identities")


def finalize_reviewed_manifest(
    *, review_path: Path, active_path: Path, generation_root: Path
) -> None:
    """Resolve, validate, apply, and atomically publish one completed review."""

    descriptor, rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    updated = apply_review_dispositions(rows, _read_review_rows(review_path))
    _require_complete_denominator(updated)
    if descriptor["benchmark_state"] != "AUDITED_LOCKED":
        raise _review_error("finalization cannot change a non-locked benchmark")
    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=updated,
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command arguments\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Validate and reconcile an active SMB manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--manifest-active", type=Path, required=True)
    reconcile.add_argument("--manifest-generation-root", type=Path, required=True)
    review = commands.add_parser("write-review")
    review.add_argument("--manifest-active", type=Path, required=True)
    review.add_argument("--manifest-generation-root", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-review")
    prepare.add_argument("--manifest-active", type=Path, required=True)
    prepare.add_argument("--manifest-generation-root", type=Path, required=True)
    prepare.add_argument("--audit-descriptor", type=Path, required=True)
    prepare.add_argument("--audit-records", type=Path, required=True)
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--review", type=Path, required=True)
    validate_review = commands.add_parser("validate-review")
    validate_review.add_argument("--review", type=Path, required=True)
    validate_review.add_argument("--manifest-active", type=Path, required=True)
    validate_review.add_argument("--manifest-generation-root", type=Path, required=True)
    finalize = commands.add_parser("finalize-review")
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--manifest-active", type=Path, required=True)
    finalize.add_argument("--manifest-generation-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "reconcile":
        report = reconcile_manifest(
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "write-review":
        write_review_csv(
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
            output_path=arguments.output,
        )
    elif arguments.command == "prepare-review":
        emit_review_evidence_from_active_manifest(
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
            audit_descriptor_path=arguments.audit_descriptor,
            audit_records_path=arguments.audit_records,
            sample_path=arguments.sample,
            review_path=arguments.review,
        )
    elif arguments.command == "validate-review":
        validate_review_from_active_manifest(
            review_path=arguments.review,
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
        )
    else:
        finalize_reviewed_manifest(
            review_path=arguments.review,
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
