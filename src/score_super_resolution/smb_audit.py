"""Fixture-testable SMB audit and immutable manifest generation boundary."""

from __future__ import annotations

import argparse
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
from collections.abc import Iterable, Mapping, Sequence
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

_SAFE_METADATA_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SAFE_UPSTREAM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class ManifestPublicationError(RuntimeError):
    """Report a generation integrity/publication failure and pointer commit state."""

    def __init__(self, message: str, *, committed: bool) -> None:
        self.committed = committed
        super().__init__(message)


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


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


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
            _write_fsynced(temp_generation / "manifest-descriptor.yaml", descriptor_bytes)
            _write_fsynced(temp_generation / "manifest-records.jsonl", records_bytes)
            _fsync_directory(temp_generation)
            os.replace(temp_generation, generation_path)
            _fsync_directory(root)

        pointer_bytes = _canonical_descriptor(pointer)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_temp = active_path.parent / f".{active_path.name}.tmp-{uuid.uuid4().hex}"
        try:
            _write_fsynced(pointer_temp, pointer_bytes)
            os.replace(pointer_temp, active_path)
            committed = True
            _fsync_directory(active_path.parent)
        finally:
            if pointer_temp.exists():
                pointer_temp.unlink()
    except ManifestPublicationError:
        raise
    except OSError as error:
        raise _publication_error(
            f"publication I/O failed: {error.strerror}", committed=committed
        ) from error
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


def write_review_csv(*, active_path: Path, generation_root: Path, output_path: Path) -> None:
    """Write redacted review rows derived exclusively from validated active resolution."""

    _, rows = resolve_active_manifest(active_path=active_path, generation_root=generation_root)
    review_rows = [_review_row(row) for row in rows]
    candidates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for candidate_id in row["near_duplicate_candidate_ids"]:  # type: ignore[union-attr]
            candidates[str(candidate_id)].append(str(row["item_id"]))
    for candidate_id, item_ids in sorted(candidates.items()):
        unique_ids = sorted(set(item_ids))
        if len(unique_ids) != 2:
            raise _publication_error(
                "near-duplicate candidate does not identify two rows", committed=True
            )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(review_rows)


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
    write_review_csv(
        active_path=arguments.manifest_active,
        generation_root=arguments.manifest_generation_root,
        output_path=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
