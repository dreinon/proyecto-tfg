"""Fixture-testable SMB audit and immutable manifest generation boundary."""

from __future__ import annotations

import argparse
import copy
import csv
import errno
import fcntl
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import date
from itertools import combinations
from pathlib import Path, PurePosixPath

import imagehash
import yaml
from PIL import Image, UnidentifiedImageError

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    assert_smb_purpose_allowed,
)
from score_super_resolution.contracts import (
    ContractValidationError,
    load_schema,
    recovery_bundle_id_v2,
    recovery_metadata_sha256,
    validate_instance,
)
from score_super_resolution.review_evidence import (
    ACCESS_STATUSES,
    DATASET_LICENCE_STATUSES,
    HUMAN_PAIR_DISPOSITIONS,
    ITEM_PROVENANCE_STATUSES,
    LEGACY_REUSE_STATUSES,
    LEGACY_SUITABILITY_DISPOSITIONS,
    V2_REUSE_STATUSES,
    VISUAL_SUITABILITY_DISPOSITIONS,
    ReviewEvidenceError,
    canonical_review_csv,
    read_review,
    review_quality_flags,
    save_review,
    validate_review_rows,
)
from score_super_resolution.review_evidence import (
    REVIEW_FIELDS as REVIEW_CSV_FIELDS,
)
from score_super_resolution.smb import _read_descriptor, load_smb

EXPECTED_ROW_COUNT = 685
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_MAX_ENCODED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PIXELS = 100_000_000
HASH_SIZE = 8
HIGHFREQ_FACTOR = 4
MAXIMUM_HAMMING_DISTANCE = 6
IMAGEHASH_VERSION = importlib.metadata.version("ImageHash")
PILLOW_VERSION = importlib.metadata.version("Pillow")
CANONICAL_PIXEL_DECODER_VERSION = "12.3.0"
CANONICAL_PIXEL_DOMAIN = b"smb-canonical-rgba-frame-v2\0"
CANONICAL_PIXEL_MODE = b"RGBA8\0"
GENERATION_DOMAIN = b"smb-manifest-generation-v1\0"
GENERATION_DOMAINS = {
    1: GENERATION_DOMAIN,
    2: b"smb-manifest-generation-v2\0",
}
AUDIT_SOURCE_SET_VERSION = 1
AUDIT_SOURCE_TREE_DOMAIN = b"smb-audit-source-tree-v1\0"
AUDIT_PATCH_DOMAIN = b"smb-audit-patch-state-v1\0"
AUDIT_LOCK_DOMAIN = b"smb-audit-uv-lock-v1\0"
RAW_METADATA_DOMAIN = b"smb-raw-metadata-v1\0"
RECOVERY_COMMAND = (
    "uv run python -m score_super_resolution.smb_audit recover-active "
    "--manifest-active data/manifests/smb-evaluation-v1.yaml "
    "--recovery-descriptor data/manifests/smb-evaluation-v1-recovery.yaml "
    "--recovery-records data/manifests/smb-evaluation-v1-recovery.jsonl.gz "
    "--manifest-generation-root artifacts/smb-manifests/generations"
)
MANIFEST_DESCRIPTOR_FILENAME = "manifest-descriptor.yaml"
MANIFEST_RECORDS_FILENAME = "manifest-records.jsonl"
RECOVERY_RECORDS_FILENAME = "manifest-records.jsonl.gz"
RECOVERY_READ_CHUNK_SIZE = 64 * 1024
_OS_OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", set())
_OS_STAT_SUPPORTS_DIR_FD = os.stat in getattr(os, "supports_dir_fd", set())
AUTHORITATIVE_MIGRATION_STAGE = ".migrate-authoritative-v2"
PUBLICATION_BOUNDARIES = (
    "generation_parent_anchored",
    "generation_records_written",
    "generation_records_fsynced",
    "generation_descriptor_written",
    "generation_descriptor_fsynced",
    "temporary_generation_directory_fsynced",
    "generation_renamed",
    "generations_parent_fsynced",
    "active_parent_anchored",
    "pointer_written",
    "pointer_fsynced",
    "pointer_replaced",
    "active_parent_fsynced",
)
INSTALL_BOUNDARIES = (
    "candidate_generation_installed",
    "recovery_bundle_installed",
    "install_lock_acquired",
    "install_cas_read",
    "install_pointer_replaced",
    "install_pointer_fsynced",
    "install_lock_released",
    "install_lock_closed",
)
INSTALL_LOCK_BASENAME = ".smb-evaluation-v1.install.lock"

_SAFE_METADATA_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SAFE_UPSTREAM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PAGE_SUFFIX_PATTERN = re.compile(r"_p[0-9]+\Z")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_IMAGE_PATH_FAILURE_CODES = frozenset(
    {
        "image_path_invalid",
        "image_path_not_regular",
        "image_path_outside_trusted_cache",
        "image_path_symlink",
        "image_path_unavailable",
    }
)


class ManifestPublicationError(RuntimeError):
    """Report a generation integrity/publication failure and pointer commit state."""

    def __init__(self, message: str, *, committed: bool) -> None:
        self.committed = committed
        super().__init__(message)


class ReviewFinalizationError(ValueError):
    """Report an invalid or incomplete human review without changing evidence."""


class _ImageIngestionError(ValueError):
    """Carry only a safe allowlisted image-ingestion failure code."""

    def __init__(self, code: str) -> None:
        if code not in _IMAGE_PATH_FAILURE_CODES:
            raise ValueError("invalid image-ingestion failure code")
        self.code = code
        super().__init__(code)


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
    version = identity_descriptor.get("schema_version")
    if isinstance(version, bool) or version not in GENERATION_DOMAINS:
        raise ValueError("unsupported manifest schema version")
    payload = (
        GENERATION_DOMAINS[int(version)]
        + _canonical_descriptor(identity_descriptor)
        + b"\0"
        + records_bytes
    )
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


def _normalized_trusted_cache_roots(trusted_cache_roots: Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(trusted_cache_roots, (str, bytes, bytearray)) or not trusted_cache_roots:
        raise ValueError("trusted_cache_roots must be a non-empty sequence")
    normalized: list[Path] = []
    for raw_root in trusted_cache_roots:
        root = Path(raw_root)
        if not root.is_absolute():
            raise ValueError("trusted_cache_roots must contain only absolute paths")
        try:
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("trusted_cache_roots must contain existing directories") from error
        if not resolved.is_dir():
            raise ValueError("trusted_cache_roots must contain only directories")
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ValueError("trusted_cache_roots must contain at least one directory")
    return tuple(normalized)


def _path_failure_from_os_error(error: OSError) -> _ImageIngestionError:
    if error.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
        return _ImageIngestionError("image_path_symlink")
    return _ImageIngestionError("image_path_unavailable")


def _read_trusted_regular_file(raw_path: str, trusted_cache_roots: Sequence[Path]) -> bytes:
    if "\0" in raw_path:
        raise _ImageIngestionError("image_path_invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise _ImageIngestionError("image_path_invalid")
    try:
        resolved_candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise _ImageIngestionError("image_path_unavailable") from error
    matching_roots = [
        root
        for root in trusted_cache_roots
        if resolved_candidate != root and resolved_candidate.is_relative_to(root)
    ]
    if len(matching_roots) != 1:
        raise _ImageIngestionError("image_path_outside_trusted_cache")
    trusted_root = matching_roots[0]
    try:
        relative = candidate.relative_to(trusted_root)
    except ValueError as error:
        raise _ImageIngestionError("image_path_outside_trusted_cache") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise _ImageIngestionError("image_path_invalid")

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_only = getattr(os, "O_DIRECTORY", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    non_blocking = getattr(os, "O_NONBLOCK", 0)
    if no_follow == 0 or directory_only == 0:
        raise RuntimeError("secure no-follow file access is unavailable")
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(
            trusted_root,
            os.O_RDONLY | directory_only | no_follow | close_on_exec,
        )
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory_only | no_follow | close_on_exec,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | no_follow | close_on_exec | non_blocking,
            dir_fd=directory_fd,
        )
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _ImageIngestionError("image_path_not_regular")
        with os.fdopen(file_fd, "rb", closefd=True) as handle:
            file_fd = None
            return handle.read()
    except _ImageIngestionError:
        raise
    except OSError as error:
        raise _path_failure_from_os_error(error) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _encoded_image(record: Mapping[str, object], *, trusted_cache_roots: Sequence[Path]) -> bytes:
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
        raw_path = image.get("path")
        if isinstance(raw_path, str) and raw_path:
            return _read_trusted_regular_file(raw_path, trusted_cache_roots)
    raise ValueError("image_bytes_unavailable")


def _regions_valid(regions: object, width: int, height: int) -> tuple[int, bool, list[str]]:
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes, bytearray)):
        return 0, False, ["regions_not_sequence"]
    failures: list[str] = []
    for index, region in enumerate(regions):
        bbox = region.get("bbox") if isinstance(region, Mapping) else None
        if isinstance(bbox, Mapping) and set(bbox) == {"x", "y", "width", "height"}:
            values = tuple(bbox[field] for field in ("x", "y", "width", "height"))
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                failures.append(f"region_{index}_invalid_bbox")
                continue
            left, top, box_width, box_height = values
            right = left + box_width
            bottom = top + box_height
        elif (
            isinstance(bbox, Sequence)
            and not isinstance(bbox, (str, bytes, bytearray))
            and len(bbox) == 4
            and all(
                not isinstance(value, bool) and isinstance(value, (int, float)) for value in bbox
            )
        ):
            left, top, right, bottom = bbox
        else:
            failures.append(f"region_{index}_invalid_bbox")
            continue
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
    row["expected_status"] = "unprocessable"
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


def _canonical_rgba_sha256(*, width: int, height: int, pixel_bytes: bytes) -> str:
    """Hash one geometry-framed stored-raster RGBA8 serialization."""

    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or not 0 < width < 2**64
        or not 0 < height < 2**64
    ):
        raise ValueError("canonical RGBA geometry is invalid")
    expected_length = width * height * 4
    if len(pixel_bytes) != expected_length:
        raise ValueError("canonical RGBA byte length disagrees with geometry")
    framed = (
        CANONICAL_PIXEL_DOMAIN
        + width.to_bytes(8, "big")
        + height.to_bytes(8, "big")
        + CANONICAL_PIXEL_MODE
        + pixel_bytes
    )
    return hashlib.sha256(framed).hexdigest()


def _pixel_limit_exceeded(width: object, height: object, *, max_pixels: int) -> bool:
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width < 1
        or height < 1
        or width >= 2**64
        or height >= 2**64
    ):
        return True
    return width * height > max_pixels


def _audit_after_guard(
    record: Mapping[str, object],
    *,
    upstream_index: int,
    source_revision: str,
    trusted_cache_roots: Sequence[Path],
    max_encoded_bytes: int,
    max_pixels: int,
    detailed_limit_failures: bool = False,
) -> tuple[dict[str, object], imagehash.ImageHash | None]:
    row = _base_row(record, upstream_index)
    row["source_revision"] = source_revision
    try:
        encoded = _encoded_image(record, trusted_cache_roots=trusted_cache_roots)
    except _ImageIngestionError as error:
        return _mark_failure(row, error.code), None
    except ValueError:
        return _mark_failure(row, "image_bytes_unavailable"), None
    row["byte_count"] = len(encoded)
    if len(encoded) > max_encoded_bytes:
        return _mark_failure(row, "encoded_image_too_large", quality_flag="oversized"), None
    row["encoded_sha256"] = hashlib.sha256(encoded).hexdigest()

    if (
        row["declared_width"] is not None
        and row["declared_height"] is not None
        and _pixel_limit_exceeded(
            row["declared_width"], row["declared_height"], max_pixels=max_pixels
        )
    ):
        reason = "declared_image_too_large" if detailed_limit_failures else "image_too_large"
        return _mark_failure(row, reason, quality_flag="oversized"), None
    if PILLOW_VERSION != CANONICAL_PIXEL_DECODER_VERSION:
        return _mark_failure(row, "unsupported_pillow_version"), None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(encoded)) as opened:
                width, height = opened.size
                row["decoded_width"] = width
                row["decoded_height"] = height
                row["image_mode"] = opened.mode
                row["image_format"] = opened.format
                if _pixel_limit_exceeded(width, height, max_pixels=max_pixels):
                    return (
                        _mark_failure(row, "image_too_large", quality_flag="oversized"),
                        None,
                    )
                opened.load()
                canonical = opened.convert("RGBA")
                pixel_bytes = canonical.tobytes()
                row["pixel_sha256"] = _canonical_rgba_sha256(
                    width=width,
                    height=height,
                    pixel_bytes=pixel_bytes,
                )
                perceptual_hash = imagehash.phash(
                    canonical, hash_size=HASH_SIZE, highfreq_factor=HIGHFREQ_FACTOR
                )
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        return _mark_failure(row, "decompression_bomb"), None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, MemoryError):
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
    trusted_cache_roots: Sequence[Path],
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, object]:
    """Audit one item after the locked-benchmark guard, returning one strict row."""

    normalized_roots = _normalized_trusted_cache_roots(trusted_cache_roots)
    result = assert_smb_purpose_allowed(
        source_descriptor=source_descriptor,
        purpose=BenchmarkPurpose.CONTENT_AUDIT,
        callback=lambda: _audit_after_guard(
            record,
            upstream_index=upstream_index,
            source_revision=str(source_descriptor.get("revision", "")),
            trusted_cache_roots=normalized_roots,
            max_encoded_bytes=max_encoded_bytes,
            max_pixels=max_pixels,
            detailed_limit_failures=True,
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


def _pair_id(candidate_type: str, first_item_id: str, second_item_id: str) -> str:
    """Return a stable, type-separated identifier for one canonical item pair."""

    if candidate_type not in {"exact", "perceptual"}:
        raise ValueError("candidate_type must be exact or perceptual")
    first, second = sorted((first_item_id, second_item_id))
    if first == second:
        raise ValueError("a duplicate pair requires two different items")
    digest = hashlib.sha256(
        f"smb-duplicate-pair-v2\0{candidate_type}\0{first}\0{second}".encode()
    ).hexdigest()
    return f"pair-{digest[:16]}"


def _duplicate_group_id(item_ids: Iterable[str]) -> str:
    members = sorted(set(item_ids))
    digest = hashlib.sha256(("smb-duplicate-group-v2\0" + "\0".join(members)).encode()).hexdigest()
    return f"duplicate-group-{digest[:16]}"


def _v2_group_ids_by_item(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    adjacency: dict[str, set[str]] = {str(row["item_id"]): set() for row in rows}
    for row in rows:
        item_id = str(row["item_id"])
        relations = row.get("duplicate_relations")
        if not isinstance(relations, Sequence):
            continue
        for relation in relations:
            if not isinstance(relation, Mapping) or relation.get("disposition") != "duplicate":
                continue
            counterpart = relation.get("counterpart_item_id")
            if isinstance(counterpart, str) and counterpart in adjacency:
                adjacency[item_id].add(counterpart)
                adjacency[counterpart].add(item_id)
    assigned: set[str] = set()
    result = {item_id: [] for item_id in adjacency}
    for item_id in sorted(adjacency):
        if item_id in assigned or not adjacency[item_id]:
            continue
        component: set[str] = set()
        pending = [item_id]
        while pending:
            member = pending.pop()
            if member in component:
                continue
            component.add(member)
            pending.extend(adjacency[member] - component)
        assigned.update(component)
        group_id = _duplicate_group_id(component)
        for member in component:
            result[member].append(group_id)
    return result


def _v2_summary(row: Mapping[str, object], *, group_ids: Sequence[str]) -> dict[str, object]:
    relations = row.get("duplicate_relations")
    if not isinstance(relations, Sequence):
        relations = ()
    relation_maps = [relation for relation in relations if isinstance(relation, Mapping)]
    dispositions = Counter(str(relation.get("disposition")) for relation in relation_maps)
    return {
        "exact_relation_count": sum(
            relation.get("candidate_type") == "exact" for relation in relation_maps
        ),
        "perceptual_relation_count": sum(
            relation.get("candidate_type") == "perceptual" for relation in relation_maps
        ),
        "pending_relation_count": dispositions["pending"],
        "duplicate_relation_count": dispositions["duplicate"],
        "related_relation_count": dispositions["related"],
        "distinct_relation_count": dispositions["distinct"],
        "unavailable_relation_count": dispositions["unavailable"],
        "group_ids": sorted(group_ids),
    }


def derive_v2_exact_relations(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Derive exact pair records and all item summaries from immutable audit facts."""

    updated = copy.deepcopy(list(rows))
    by_exact: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in updated:
        relations = row.get("duplicate_relations")
        if not isinstance(relations, list):
            raise _publication_error("v2 duplicate_relations must be an array")
        row["duplicate_relations"] = [
            relation
            for relation in relations
            if isinstance(relation, Mapping) and relation.get("candidate_type") != "exact"
        ]
        image = row.get("image")
        if row.get("processing_status") != "processed" or not isinstance(image, Mapping):
            continue
        pixel_sha256 = image.get("pixel_sha256")
        if isinstance(pixel_sha256, str):
            by_exact[pixel_sha256].append(row)
    for pixel_sha256, members in by_exact.items():
        for first, second in combinations(sorted(members, key=lambda row: str(row["item_id"])), 2):
            item_ids = sorted((str(first["item_id"]), str(second["item_id"])))
            pair_id = _pair_id("exact", *item_ids)
            first_image = first["image"]
            second_image = second["image"]
            assert isinstance(first_image, Mapping) and isinstance(second_image, Mapping)
            encoded_sha256 = first_image.get("encoded_sha256")
            encoded_equality = isinstance(
                encoded_sha256, str
            ) and encoded_sha256 == second_image.get("encoded_sha256")
            shared = {
                "pair_id": pair_id,
                "candidate_type": "exact",
                "item_ids": item_ids,
                "evidence_basis": "canonical_pixel_sha256",
                "evidence": {
                    "pixel_sha256": pixel_sha256,
                    "encoded_equality": encoded_equality,
                    "encoded_sha256": encoded_sha256 if encoded_equality else None,
                },
                "disposition": "duplicate",
                "reviewer": None,
                "reviewed_at": None,
                "rationale": "Derived from matching canonical framed RGBA pixel SHA-256 values.",
            }
            for row, counterpart in ((first, second), (second, first)):
                relation = {**shared, "counterpart_item_id": counterpart["item_id"]}
                row["duplicate_relations"].append(relation)  # type: ignore[union-attr]
    group_ids = _v2_group_ids_by_item(updated)
    for row in updated:
        relations = row["duplicate_relations"]
        assert isinstance(relations, list)
        relations.sort(key=lambda relation: str(relation["pair_id"]))
        row["near_duplicate_candidate_ids"] = sorted(
            str(relation["pair_id"])
            for relation in relations
            if relation["candidate_type"] == "perceptual"
        )
        row["duplicate_summary"] = _v2_summary(row, group_ids=group_ids[str(row["item_id"])])
    return updated


def _raw_metadata_sha256(metadata: object) -> str | None:
    """Digest one upstream scalar without retaining its potentially large raw value."""

    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("raw")
    if raw is None:
        return None
    payload = json.dumps(
        raw,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(RAW_METADATA_DOMAIN + payload).hexdigest()


def _canonical_source_group_id(original_score_normalized: object) -> str:
    """Derive one leakage-safe score identity from a normalized SMB page identity."""

    if (
        not isinstance(original_score_normalized, str)
        or _SAFE_METADATA_PATTERN.fullmatch(original_score_normalized) is None
    ):
        raise ValueError("source identity must contain a safe normalized original score")
    source_group_id, replacements = _PAGE_SUFFIX_PATTERN.subn(
        "", original_score_normalized, count=1
    )
    if replacements != 1 or _SAFE_METADATA_PATTERN.fullmatch(source_group_id) is None:
        raise ValueError("source identity must end in one canonical page suffix")
    return source_group_id


def _v2_row_from_automated_audit(row: Mapping[str, object]) -> dict[str, object]:
    """Compact one freshly audited v1-shaped row without adding human claims."""

    original_score = row["original_score"]
    page = row["page"]
    page_texture = row["page_texture"]
    quality = row["quality"]
    assert isinstance(original_score, Mapping)
    assert isinstance(page, Mapping)
    assert isinstance(page_texture, Mapping)
    assert isinstance(quality, Mapping)
    sampled = row["audit_sample_member"] is True
    source_group_id = _canonical_source_group_id(original_score.get("normalized"))
    return {
        "schema_version": 2,
        "record_type": "manifest-row",
        "manifest_version": 2,
        "source_key": "smb",
        "source_revision": row["source_revision"],
        "split": "test",
        "upstream_index": row["upstream_index"],
        "item_id": row["item_id"],
        "source_identity": {
            "original_score_normalized": original_score.get("normalized"),
            "original_score_raw_sha256": _raw_metadata_sha256(original_score),
            "page_normalized": page.get("normalized"),
            "page_raw_sha256": _raw_metadata_sha256(page),
            "page_texture_normalized": page_texture.get("normalized"),
            "page_texture_raw_sha256": _raw_metadata_sha256(page_texture),
        },
        "source_group_id": source_group_id,
        "image": {
            "encoded_sha256": row["encoded_sha256"],
            "pixel_sha256": row["pixel_sha256"],
            "declared_width": row["declared_width"],
            "declared_height": row["declared_height"],
            "decoded_width": row["decoded_width"],
            "decoded_height": row["decoded_height"],
            "mode": row["image_mode"],
            "format": row["image_format"],
            "byte_count": row["byte_count"],
        },
        "annotations": {
            "region_count": row["region_count"],
            "bbox_valid": row["bbox_valid"],
            "required_text_present": row["required_text_present"],
            "failures": list(row["annotation_failures"]),
        },
        "automated_audit": {
            "status": "automated",
            "algorithm_version": "smb-audit-v2",
            "quality_flags": list(quality["flags"]),
        },
        "visual_review": (
            {
                "status": "unavailable",
                "reason": "Frozen-sample human evidence has not yet been migrated.",
            }
            if sampled
            else {"status": "not_visually_reviewed"}
        ),
        "audit_sample_member": sampled,
        "duplicate_relations": [],
        "near_duplicate_candidate_ids": [],
        "duplicate_summary": {
            "exact_relation_count": 0,
            "perceptual_relation_count": 0,
            "pending_relation_count": 0,
            "duplicate_relation_count": 0,
            "related_relation_count": 0,
            "distinct_relation_count": 0,
            "unavailable_relation_count": 0,
            "group_ids": [],
        },
        "expected_status": row["expected_status"],
        "processing_status": row["processing_status"],
        "unprocessable_reason": row["unprocessable_reason"],
        "rights": {
            "dataset_licence": {
                "status": "confirmed",
                "identifier": "CC-BY-NC-4.0",
                "reference": "https://creativecommons.org/licenses/by-nc/4.0/",
            },
            "item_provenance": {
                "status": "pending",
                "rationale": "Per-item provenance has not been established.",
            },
            "access_status": "confirmed",
            "redistribution": {
                "status": "not_established",
                "reviewed_basis_ref": None,
            },
            "figure_reproduction": {
                "status": "prohibited",
                "reviewed_basis_ref": None,
            },
        },
        "paired_eligible": row["paired_eligible"],
        "paired_ineligibility_reason": row["paired_ineligibility_reason"],
    }


def audit_dataset_v2(
    records: Sequence[Mapping[str, object]],
    *,
    source_descriptor: Mapping[str, object],
    trusted_cache_roots: Sequence[Path],
    deterministic_seed: int = 20260818,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, object]]:
    """Run one authenticated audit and retain compact v2 duplicate evidence."""

    normalized_roots = _normalized_trusted_cache_roots(trusted_cache_roots)
    audited: list[tuple[dict[str, object], imagehash.ImageHash | None]] = []
    for upstream_index, record in enumerate(records):
        result = assert_smb_purpose_allowed(
            source_descriptor=source_descriptor,
            purpose=BenchmarkPurpose.CONTENT_AUDIT,
            callback=lambda record=record, upstream_index=upstream_index: _audit_after_guard(
                record,
                upstream_index=upstream_index,
                source_revision=str(source_descriptor.get("revision", "")),
                trusted_cache_roots=normalized_roots,
                max_encoded_bytes=max_encoded_bytes,
                max_pixels=max_pixels,
                detailed_limit_failures=True,
            ),
        )
        if not isinstance(result, tuple):
            raise RuntimeError("SMB audit guard returned no item result")
        audited.append(result)

    audited_rows = [row for row, _ in audited]
    selected = set(
        select_visual_sample(audited_rows, seed=deterministic_seed, sample_size=sample_size)
    )
    for row in audited_rows:
        row["audit_sample_member"] = row["item_id"] in selected
    rows = [_v2_row_from_automated_audit(row) for row in audited_rows]

    for first_index, (first_row, first_hash) in enumerate(audited):
        if first_hash is None:
            continue
        for second_index in range(first_index + 1, len(audited)):
            second_row, second_hash = audited[second_index]
            if second_hash is None or first_row["pixel_sha256"] == second_row["pixel_sha256"]:
                continue
            distance = int(first_hash - second_hash)
            if distance > MAXIMUM_HAMMING_DISTANCE:
                continue
            item_ids = sorted((str(first_row["item_id"]), str(second_row["item_id"])))
            pair_id = _pair_id("perceptual", *item_ids)
            shared: dict[str, object] = {
                "pair_id": pair_id,
                "candidate_type": "perceptual",
                "item_ids": item_ids,
                "evidence_basis": "perceptual_hash_candidate",
                "evidence": {"algorithm": "phash", "version": 1, "distance": distance},
                "disposition": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "rationale": "",
            }
            for row_index, counterpart in (
                (first_index, second_row),
                (second_index, first_row),
            ):
                rows[row_index]["duplicate_relations"].append(  # type: ignore[union-attr]
                    {**shared, "counterpart_item_id": counterpart["item_id"]}
                )

    rows = derive_v2_exact_relations(rows)
    for row in rows:
        validate_instance("manifest-row", row, version=2)
    return rows


def audit_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    source_descriptor: Mapping[str, object],
    trusted_cache_roots: Sequence[Path],
    deterministic_seed: int = 20260818,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_encoded_bytes: int = DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, object]]:
    """Audit fixtures safely, preserve every input, and label duplicate candidates."""

    normalized_roots = _normalized_trusted_cache_roots(trusted_cache_roots)
    audited: list[tuple[dict[str, object], imagehash.ImageHash | None]] = []
    for upstream_index, record in enumerate(records):
        result = assert_smb_purpose_allowed(
            source_descriptor=source_descriptor,
            purpose=BenchmarkPurpose.CONTENT_AUDIT,
            callback=lambda record=record, upstream_index=upstream_index: _audit_after_guard(
                record,
                upstream_index=upstream_index,
                source_revision=str(source_descriptor.get("revision", "")),
                trusted_cache_roots=normalized_roots,
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


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot establish authoritative SMB audit provenance") from error
    return completed.stdout


def _nul_paths(output: bytes) -> tuple[PurePosixPath, ...]:
    try:
        decoded = [part.decode("utf-8") for part in output.split(b"\0") if part]
    except UnicodeError as error:
        raise RuntimeError(
            "authoritative SMB audit provenance contains a non-UTF-8 path"
        ) from error
    return tuple(PurePosixPath(path) for path in decoded)


def _is_authoritative_audit_source_path(relative_path: PurePosixPath) -> bool:
    if relative_path.as_posix() in {"pyproject.toml", "uv.lock"}:
        return True
    parts = relative_path.parts
    return (
        len(parts) >= 3
        and parts[:2] == ("src", "score_super_resolution")
        and relative_path.suffix == ".py"
    ) or (len(parts) >= 3 and parts[:2] == ("data", "schemas") and relative_path.suffix == ".json")


def _authoritative_audit_source_paths(project_root: Path) -> tuple[Path, ...]:
    root = project_root.expanduser().resolve()
    lock_path = root / "uv.lock"
    project_path = root / "pyproject.toml"
    git_paths = _nul_paths(
        _git_bytes(
            root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src/score_super_resolution",
            "data/schemas",
            "pyproject.toml",
            "uv.lock",
        )
    )
    relevant = sorted(
        {path for path in git_paths if _is_authoritative_audit_source_path(path)},
        key=lambda path: path.as_posix(),
    )
    python_paths = [path for path in relevant if path.suffix == ".py"]
    schema_paths = [path for path in relevant if path.suffix == ".json"]
    if (
        not python_paths
        or not schema_paths
        or not lock_path.is_file()
        or not project_path.is_file()
    ):
        raise RuntimeError("authoritative SMB audit provenance source set is incomplete")
    paths = [root / path for path in relevant]
    relative_paths: list[Path] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("authoritative SMB audit provenance sources must be regular files")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "authoritative SMB audit provenance source escapes project"
            ) from error
        if not _is_authoritative_audit_source_path(PurePosixPath(relative.as_posix())):
            raise RuntimeError("authoritative SMB audit provenance source set is invalid")
        relative_paths.append(relative)
    return tuple(sorted(set(relative_paths), key=lambda path: path.as_posix()))


def _update_framed_digest(digest: object, label: bytes, payload: bytes) -> None:
    assert hasattr(digest, "update")
    digest.update(len(label).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(label)  # type: ignore[attr-defined]
    digest.update(len(payload).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(payload)  # type: ignore[attr-defined]


def _source_tree_sha256(project_root: Path, source_paths: Iterable[Path]) -> str:
    root = project_root.expanduser().resolve()
    normalized: dict[str, Path] = {}
    for supplied in source_paths:
        path = supplied if supplied.is_absolute() else root / supplied
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "authoritative SMB audit provenance source escapes project"
            ) from error
        normalized[relative.as_posix()] = path
    if not normalized:
        raise RuntimeError("authoritative SMB audit provenance source set is empty")
    digest = hashlib.sha256(AUDIT_SOURCE_TREE_DOMAIN)
    for relative, path in sorted(normalized.items()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("authoritative SMB audit provenance source is unavailable")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RuntimeError("cannot read authoritative SMB audit provenance source") from error
        _update_framed_digest(digest, relative.encode("utf-8"), content)
    return digest.hexdigest()


def audit_source_provenance(project_root: Path | None = None) -> dict[str, object]:
    """Identify exact authoritative audit sources, Git patch state, and uv lock bytes."""

    root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    discovered_root = Path(
        _git_bytes(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if discovered_root != root:
        raise RuntimeError("authoritative SMB audit provenance resolved the wrong Git repository")
    revision = _git_bytes(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("authoritative SMB audit provenance has no committed revision")

    source_paths = _authoritative_audit_source_paths(root)
    tracked_paths = _nul_paths(_git_bytes(root, "ls-files", "-z"))
    relevant_tracked = tuple(
        sorted(path for path in tracked_paths if _is_authoritative_audit_source_path(path))
    )
    changed_paths = _nul_paths(_git_bytes(root, "diff", "--name-only", "-z", "HEAD", "--"))
    untracked_paths = _nul_paths(
        _git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    )
    relevant_changed = tuple(
        sorted(path for path in changed_paths if _is_authoritative_audit_source_path(path))
    )
    relevant_untracked = tuple(
        sorted(path for path in untracked_paths if _is_authoritative_audit_source_path(path))
    )
    patch_paths = sorted(
        set((*relevant_tracked, *relevant_changed)), key=lambda path: path.as_posix()
    )
    patch_bytes = _git_bytes(
        root,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        *(path.as_posix() for path in patch_paths),
    )
    patch_digest = hashlib.sha256(AUDIT_PATCH_DOMAIN)
    _update_framed_digest(patch_digest, b"tracked-diff", patch_bytes)
    for relative in relevant_untracked:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("authoritative SMB audit provenance untracked source is unavailable")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise RuntimeError("cannot read authoritative SMB audit provenance source") from error
        _update_framed_digest(patch_digest, relative.as_posix().encode("utf-8"), content)

    try:
        lock_bytes = (root / "uv.lock").read_bytes()
    except OSError as error:
        raise RuntimeError("authoritative SMB audit provenance lock is unavailable") from error
    return {
        "source_set_version": AUDIT_SOURCE_SET_VERSION,
        "algorithm": "sha256",
        "revision": revision,
        "dirty": bool(relevant_changed or relevant_untracked),
        "source_tree_sha256": _source_tree_sha256(root, source_paths),
        "patch_sha256": patch_digest.hexdigest(),
        "lock_sha256": hashlib.sha256(AUDIT_LOCK_DOMAIN + lock_bytes).hexdigest(),
    }


def _audit_creation_command(
    *,
    source_path: Path,
    audit_descriptor_path: Path,
    audit_records_path: Path,
    sample_path: Path,
    review_path: Path,
    active_path: Path,
    generation_root: Path,
) -> str:
    arguments = (
        ("--source", source_path),
        ("--audit-descriptor", audit_descriptor_path),
        ("--audit-records", audit_records_path),
        ("--sample", sample_path),
        ("--review", review_path),
        ("--manifest-active", active_path),
        ("--manifest-generation-root", generation_root),
    )
    suffix = " ".join(f"{flag} {path.as_posix()}" for flag, path in arguments)
    return f"uv run python -m score_super_resolution.smb_audit audit {suffix}"


def _manifest_descriptor(
    *,
    source_descriptor: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    code_revision: str,
    creation_command: str,
    deterministic_seed: int,
) -> dict[str, object]:
    exclusions = [
        {
            "upstream_index": row["upstream_index"],
            "item_id": row["item_id"],
            "reason": row["paired_ineligibility_reason"],
        }
        for row in rows
        if row["paired_eligible"] is False
    ]
    return {
        "schema_version": 1,
        "record_type": "manifest-descriptor",
        "manifest_id": "smb-evaluation-v1",
        "generation_algorithm": {
            "algorithm": "sha256",
            "version": 1,
            "domain_separator": "smb-manifest-generation-v1",
            "descriptor_canonicalization": "yaml-safe-sort-keys-utf8-v1",
            "records_canonicalization": "jsonl-utf8-sorted-keys-v1",
        },
        "source_key": "smb",
        "source_revision": source_descriptor["revision"],
        "creation_command": creation_command,
        "code_revision": code_revision,
        "grouping_unit": "source_score",
        "upstream_split": "test",
        "project_split": "evaluation",
        "deterministic_seed": deterministic_seed,
        "exclusions": exclusions,
        "row_schema_id": "manifest-row",
        "row_schema_version": 1,
        "row_count": len(rows),
        "records_sha256": "0" * 64,
        "audit_version": "smb-audit-v1",
        "benchmark_state": "AUDITED_LOCKED",
        "hash_provenance": {
            "encoded": {
                "algorithm": "sha256",
                "version": 1,
                "canonicalization": "encoded-bytes-v1",
            },
            "pixels": {
                "algorithm": "sha256",
                "version": 1,
                "canonicalization": "rgba-uint8-row-major-v1",
            },
        },
        "duplicate_provenance": {
            "exact": {"algorithm": "encoded-and-pixel-sha256", "version": 1},
            "near": {
                "algorithm": "phash",
                "version": 1,
                "library": "ImageHash",
                "library_version": IMAGEHASH_VERSION,
                "hash_size": HASH_SIZE,
                "highfreq_factor": HIGHFREQ_FACTOR,
                "maximum_hamming_distance": MAXIMUM_HAMMING_DISTANCE,
            },
        },
        "sample_selection": {
            "algorithm": "sha256-rank",
            "version": 1,
            "seed": deterministic_seed,
            "population_size": EXPECTED_ROW_COUNT,
            "sample_size": DEFAULT_SAMPLE_SIZE,
            "identity_fields": ["upstream_index", "item_id"],
            "selection_state": "pre-review",
        },
    }


def _require_complete_audit_rows(rows: Sequence[Mapping[str, object]]) -> None:
    expected_indices = set(range(EXPECTED_ROW_COUNT))
    expected_ids = {f"smb-test-{index:06d}" for index in expected_indices}
    indices = [row.get("upstream_index") for row in rows]
    item_ids = [row.get("item_id") for row in rows]
    if (
        len(rows) != EXPECTED_ROW_COUNT
        or len(set(indices)) != EXPECTED_ROW_COUNT
        or set(indices) != expected_indices
        or len(set(item_ids)) != EXPECTED_ROW_COUNT
        or set(item_ids) != expected_ids
    ):
        raise ValueError("authenticated SMB audit must preserve exactly 685 unique identities")


def _without_automatic_image_decoding(dataset: object) -> object:
    cast_column = getattr(dataset, "cast_column", None)
    features = getattr(dataset, "features", None)
    if not callable(cast_column) or not isinstance(features, Mapping) or "image" not in features:
        return dataset
    from datasets import Image as DatasetImage

    return cast_column("image", DatasetImage(decode=False))


def _hugging_face_datasets_cache_roots() -> tuple[Path, ...]:
    from datasets import config as datasets_config

    return (Path(datasets_config.HF_DATASETS_CACHE),)


def run_authenticated_audit(
    *,
    source_path: Path,
    audit_descriptor_path: Path,
    audit_records_path: Path,
    sample_path: Path,
    review_path: Path,
    active_path: Path,
    generation_root: Path,
    dataset_loader: Callable[..., object] | None = None,
    deterministic_seed: int = 20260818,
) -> dict[str, int]:
    """Audit the exact gated SMB revision and publish only pointer-derived redacted evidence."""

    source_descriptor = _read_descriptor(source_path)
    dataset = load_smb(
        purpose=BenchmarkPurpose.CONTENT_AUDIT,
        loader=dataset_loader,
        descriptor_path=source_path,
    )
    dataset = _without_automatic_image_decoding(dataset)
    try:
        source_count = len(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("authenticated SMB source must expose an exact row count") from error
    if source_count != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"authenticated SMB source must contain exactly 685 rows, found {source_count}"
        )
    rows = audit_dataset(
        dataset,  # type: ignore[arg-type]
        source_descriptor=source_descriptor,
        trusted_cache_roots=_hugging_face_datasets_cache_roots(),
        deterministic_seed=deterministic_seed,
        sample_size=DEFAULT_SAMPLE_SIZE,
    )
    _require_complete_audit_rows(rows)
    implementation_provenance = audit_source_provenance()
    descriptor = _manifest_descriptor(
        source_descriptor=source_descriptor,
        rows=rows,
        code_revision=str(implementation_provenance["revision"]),
        creation_command=_audit_creation_command(
            source_path=source_path,
            audit_descriptor_path=audit_descriptor_path,
            audit_records_path=audit_records_path,
            sample_path=sample_path,
            review_path=review_path,
            active_path=active_path,
            generation_root=generation_root,
        ),
        deterministic_seed=deterministic_seed,
    )
    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=rows,
    )
    report = reconcile_manifest(active_path=active_path, generation_root=generation_root)
    emit_review_evidence_from_active_manifest(
        active_path=active_path,
        generation_root=generation_root,
        audit_descriptor_path=audit_descriptor_path,
        audit_records_path=audit_records_path,
        sample_path=sample_path,
        review_path=review_path,
        implementation_provenance=implementation_provenance,
    )
    return report


def _migration_project_paths(source_path: Path) -> dict[str, Path]:
    source = source_path.expanduser().resolve()
    project_root = source.parents[2]
    if source != project_root / "data" / "sources" / "smb.yaml":
        raise ValueError("authoritative migration requires the canonical SMB descriptor path")
    return {
        "project_root": project_root,
        "audit_descriptor": project_root / "data" / "audits" / "smb-audit-v1.yaml",
        "audit_records": project_root / "data" / "audits" / "smb-audit-v1.jsonl",
        "review": project_root / "data" / "audits" / "smb-review-v1.csv",
    }


def _migration_stage_paths(generation_root: Path) -> dict[str, Path]:
    stage_root = generation_root.resolve().parent / AUTHORITATIVE_MIGRATION_STAGE
    return {
        "root": stage_root,
        "descriptor": stage_root / "stage.yaml",
        "records": stage_root / "automated-records.jsonl",
        "legacy_active": stage_root / "legacy-active.yaml",
        "candidate_active": stage_root / "candidate-active.yaml",
        "candidate_review": stage_root / "candidate-review.csv",
    }


def _durable_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _review_file_identity(path: Path) -> tuple[str, tuple[int, int, int, int, int, int]]:
    opened = path.stat()
    metadata = (
        opened.st_ino,
        opened.st_mode,
        opened.st_uid,
        opened.st_gid,
        opened.st_size,
        opened.st_mtime_ns,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest(), metadata


def _sample_rows(path: Path) -> list[dict[str, str]]:
    expected = (
        "upstream_index",
        "item_id",
        "source_group_id",
        "processing_status",
        "audit_sample_member",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != expected:
                raise ValueError("frozen sample has an unexpected header")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("cannot read the frozen SMB sample") from error
    if (
        len(rows) != DEFAULT_SAMPLE_SIZE
        or len({row["item_id"] for row in rows}) != DEFAULT_SAMPLE_SIZE
        or any(row["audit_sample_member"] != "True" for row in rows)
    ):
        raise ValueError("frozen sample must contain exactly 64 unique selected identities")
    return rows


def _legacy_candidate_pairs(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> dict[str, tuple[str, str]]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        candidates = row.get("near_duplicate_candidate_ids")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise _review_error("legacy manifest candidate evidence is malformed")
        for candidate_id in candidates:
            occurrences[str(candidate_id)].append(str(row["item_id"]))
    result: dict[str, tuple[str, str]] = {}
    for candidate_id, item_ids in occurrences.items():
        canonical = tuple(sorted(item_ids))
        if (
            len(item_ids) != 2
            or len(set(item_ids)) != 2
            or candidate_id != _candidate_id(*canonical)
        ):
            raise _review_error("legacy candidate evidence is not canonical")
        result[candidate_id] = canonical
    reviewed = {
        str(row["review_key"]): tuple(sorted((row["item_id"], row["candidate_item_id"])))
        for row in review_rows
        if row["review_kind"] == "candidate"
    }
    if reviewed != result:
        raise _review_error("legacy review candidate keys disagree with the active generation")
    return result


def _v2_perceptual_pairs(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for row in rows:
        relations = row.get("duplicate_relations")
        if not isinstance(relations, Sequence):
            raise _review_error("v2 duplicate relations are malformed")
        for relation in relations:
            if not isinstance(relation, Mapping) or relation.get("candidate_type") != "perceptual":
                continue
            item_ids = relation.get("item_ids")
            if not isinstance(item_ids, Sequence) or isinstance(item_ids, (str, bytes)):
                raise _review_error("v2 perceptual pair identities are malformed")
            pair = tuple(str(item_id) for item_id in item_ids)
            if len(pair) != 2 or pair != tuple(sorted(pair)):
                raise _review_error("v2 perceptual pair identities are not canonical")
            pair_id = str(relation["pair_id"])
            previous = result.setdefault(pair_id, pair)
            if previous != pair:
                raise _review_error("v2 perceptual pair mirrors disagree")
    return result


def _manifest_descriptor_v2(
    *,
    source_descriptor: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    source_provenance: Mapping[str, object],
    creation_command: str,
    deterministic_seed: int,
) -> dict[str, object]:
    exclusions = [
        {
            "upstream_index": row["upstream_index"],
            "item_id": row["item_id"],
            "reason": row["paired_ineligibility_reason"],
        }
        for row in rows
        if row["paired_eligible"] is False
    ]
    return {
        "schema_version": 2,
        "record_type": "manifest-descriptor",
        "manifest_id": "smb-evaluation-v2",
        "generation_algorithm": {
            "algorithm": "sha256",
            "version": 2,
            "domain_separator": "smb-manifest-generation-v2",
            "descriptor_canonicalization": "yaml-safe-sort-keys-utf8-v1",
            "records_canonicalization": "jsonl-utf8-sorted-keys-v1",
        },
        "source_key": "smb",
        "source_revision": source_descriptor["revision"],
        "creation_command": creation_command,
        "source_provenance": dict(source_provenance),
        "grouping_unit": "source_score",
        "upstream_split": "test",
        "project_split": "evaluation",
        "deterministic_seed": deterministic_seed,
        "exclusions": exclusions,
        "row_schema_id": "manifest-row",
        "row_schema_version": 2,
        "row_count": len(rows),
        "records_sha256": "0" * 64,
        "audit_version": "smb-audit-v2",
        "benchmark_state": "AUDITED_LOCKED",
        "hash_provenance": {
            "encoded": {
                "algorithm": "sha256",
                "version": 1,
                "canonicalization": "encoded-bytes-v1",
            },
            "pixels": {
                "algorithm": "sha256",
                "version": 2,
                "canonicalization": "canonical-rgba-frame-v2",
                "domain_separator": "smb-canonical-rgba-frame-v2",
                "decoder_library": "Pillow",
                "decoder_version": CANONICAL_PIXEL_DECODER_VERSION,
                "output_mode": "RGBA8",
                "alpha_policy": "retain-alpha-and-underlying-rgb",
                "orientation_policy": "stored-raster-ignore-exif",
                "metadata_policy": "ignore-non-raster-metadata",
                "max_encoded_bytes": DEFAULT_MAX_ENCODED_BYTES,
                "max_pixels": DEFAULT_MAX_PIXELS,
                "failure_policy": "safe-explicit-failure-no-digest",
            },
        },
        "duplicate_provenance": {
            "exact": {"algorithm": "canonical-pixel-sha256", "version": 2},
            "near": {
                "algorithm": "phash",
                "version": 1,
                "library": "ImageHash",
                "library_version": IMAGEHASH_VERSION,
                "hash_size": HASH_SIZE,
                "highfreq_factor": HIGHFREQ_FACTOR,
                "maximum_hamming_distance": MAXIMUM_HAMMING_DISTANCE,
            },
        },
        "sample_selection": {
            "algorithm": "sha256-rank",
            "version": 1,
            "seed": deterministic_seed,
            "population_size": EXPECTED_ROW_COUNT,
            "sample_size": DEFAULT_SAMPLE_SIZE,
            "identity_fields": ["upstream_index", "item_id"],
            "selection_state": "pre-review",
        },
        "review_inference": _v2_review_inference(rows),
    }


def migrate_authoritative_audit(
    *,
    source_path: Path,
    sample_path: Path,
    active_path: Path,
    generation_root: Path,
    dataset_loader: Callable[..., object] | None = None,
    deterministic_seed: int = 20260818,
) -> dict[str, int]:
    """Stage a corrected audit while leaving human evidence and active state intact."""

    paths = _migration_project_paths(source_path)
    review_path = paths["review"]
    legacy_pointer_bytes = active_path.read_bytes()
    legacy_descriptor, legacy_rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    if legacy_descriptor.get("schema_version") != 1:
        raise ValueError("authoritative audit migration requires the legacy v1 active generation")
    legacy_review = _read_review_rows(review_path)
    validated_legacy_review = _validated_review_rows(legacy_rows, legacy_review)
    legacy_candidates = _legacy_candidate_pairs(legacy_rows, validated_legacy_review)
    frozen_sample = _sample_rows(sample_path)
    frozen_sample_ids = {row["item_id"] for row in frozen_sample}
    if frozen_sample_ids != {
        str(row["item_id"]) for row in legacy_rows if row["audit_sample_member"] is True
    }:
        raise ValueError("frozen sample disagrees with the legacy active generation")
    review_sha256, review_metadata = _review_file_identity(review_path)
    sample_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()

    source_descriptor = _read_descriptor(source_path)
    dataset = load_smb(
        purpose=BenchmarkPurpose.CONTENT_AUDIT,
        loader=dataset_loader,
        descriptor_path=source_path,
    )
    dataset = _without_automatic_image_decoding(dataset)
    try:
        source_count = len(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("authenticated SMB source must expose an exact row count") from error
    if source_count != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"authenticated SMB source must contain exactly 685 rows, found {source_count}"
        )
    rows = audit_dataset_v2(
        dataset,  # type: ignore[arg-type]
        source_descriptor=source_descriptor,
        trusted_cache_roots=_hugging_face_datasets_cache_roots(),
        deterministic_seed=deterministic_seed,
        sample_size=DEFAULT_SAMPLE_SIZE,
    )
    _require_complete_audit_rows(rows)
    new_sample_ids = {str(row["item_id"]) for row in rows if row["audit_sample_member"] is True}
    if new_sample_ids != frozen_sample_ids:
        raise ValueError("corrected audit changed the frozen visual sample identities")
    new_pairs = set(_v2_perceptual_pairs(rows).values())
    if new_pairs != set(legacy_candidates.values()):
        raise ValueError("corrected audit changed the perceptual candidate identity set")
    source_provenance = audit_source_provenance(paths["project_root"])
    records_bytes = _canonical_jsonl(rows)
    stage = {
        "schema_version": 1,
        "record_type": "smb-authoritative-migration-stage",
        "source_revision": source_descriptor["revision"],
        "deterministic_seed": deterministic_seed,
        "row_count": len(rows),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "legacy_active_sha256": hashlib.sha256(legacy_pointer_bytes).hexdigest(),
        "legacy_generation_id": legacy_descriptor["generation_id"],
        "legacy_records_sha256": legacy_descriptor["records_sha256"],
        "legacy_review_sha256": review_sha256,
        "legacy_sample_sha256": sample_sha256,
        "sample_count": len(new_sample_ids),
        "perceptual_pair_count": len(new_pairs),
        "source_provenance": source_provenance,
    }
    stage_paths = _migration_stage_paths(generation_root)
    _durable_replace(stage_paths["records"], records_bytes)
    _durable_replace(stage_paths["descriptor"], _canonical_descriptor(stage))
    _durable_replace(stage_paths["legacy_active"], legacy_pointer_bytes)

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
    audit_descriptor = {
        "audit_version": "smb-audit-v2",
        "benchmark_state": "AUDITED_LOCKED",
        "implementation_provenance": source_provenance,
        "legacy_generation_id": legacy_descriptor["generation_id"],
        "manifest_id": "smb-evaluation-v2",
        "record_type": "smb-audit-export",
        "records_sha256": stage["records_sha256"],
        "row_count": len(rows),
        "schema_version": 2,
        "source_key": "smb",
        "source_revision": source_descriptor["revision"],
    }
    _durable_replace(paths["audit_descriptor"], _canonical_descriptor(audit_descriptor))
    _durable_replace(paths["audit_records"], _canonical_jsonl(redacted_rows))
    if _review_file_identity(review_path) != (review_sha256, review_metadata):
        raise ValueError("authoritative automated audit changed the human review evidence")
    if hashlib.sha256(sample_path.read_bytes()).hexdigest() != sample_sha256:
        raise ValueError("authoritative automated audit changed the frozen sample evidence")
    if active_path.read_bytes() != legacy_pointer_bytes:
        raise ValueError("authoritative automated audit changed the active manifest pointer")
    return {
        "row_count": len(rows),
        "processed": sum(row["processing_status"] == "processed" for row in rows),
        "failed": sum(row["processing_status"] == "failed" for row in rows),
        "paired_eligible": sum(row["paired_eligible"] is True for row in rows),
        "sampled": len(new_sample_ids),
        "perceptual_pairs": len(new_pairs),
    }


def _load_authoritative_stage(
    generation_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, Path]]:
    paths = _migration_stage_paths(generation_root)
    stage = _load_yaml_mapping(paths["descriptor"], label="authoritative migration stage")
    try:
        records_bytes = paths["records"].read_bytes()
        if hashlib.sha256(records_bytes).hexdigest() != stage["records_sha256"]:
            raise ValueError("authoritative migration stage records checksum mismatch")
        rows = [json.loads(line) for line in records_bytes.decode("utf-8").splitlines()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("authoritative migration stage contains a non-object row")
        _require_complete_audit_rows(rows)
        for row in rows:
            validate_instance("manifest-row", row, version=2)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise ValueError("authoritative migration stage is invalid") from error
    return stage, rows, paths


def _migrated_v2_review_rows(
    *,
    staged_rows: Sequence[Mapping[str, object]],
    legacy_rows: Sequence[Mapping[str, object]],
    legacy_review_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    validated = _validated_review_rows(legacy_rows, legacy_review_rows)
    legacy_items = {row["item_id"]: row for row in validated if row["review_kind"] == "item"}
    legacy_candidates = {
        tuple(sorted((row["item_id"], row["candidate_item_id"]))): row
        for row in validated
        if row["review_kind"] == "candidate"
    }
    staged_pairs = _v2_perceptual_pairs(staged_rows)
    if set(staged_pairs.values()) != set(legacy_candidates):
        raise _review_error("perceptual candidate pairs drifted during decision migration")

    migrated: list[dict[str, str]] = []
    for row in staged_rows:
        item_id = str(row["item_id"])
        legacy = legacy_items[item_id]
        migrated.append(
            {
                **{field: "" for field in REVIEW_CSV_FIELDS},
                "review_kind": "item_policy",
                "review_key": f"policy:{item_id}",
                "item_id": item_id,
                "review_status": "reviewed",
                "reviewer": legacy["reviewer"],
                "reviewed_at": legacy["reviewed_at"],
                "rationale": (
                    "Stable-key migration of the reviewed source grouping and dataset-level "
                    "CC BY-NC 4.0 access policy; per-item provenance remains unavailable."
                ),
                "source_group_id": legacy["source_group_id"],
                "dataset_licence_status": "confirmed",
                "item_provenance_status": "unavailable",
                "access_status": "confirmed",
                "redistribution_status": "not_established",
                "figure_reproduction_status": "prohibited",
            }
        )
        if row["audit_sample_member"] is True:
            migrated.append(
                {
                    **{field: "" for field in REVIEW_CSV_FIELDS},
                    "review_kind": "visual_item",
                    "review_key": f"visual:{item_id}",
                    "item_id": item_id,
                    "review_status": "reviewed",
                    "reviewer": legacy["reviewer"],
                    "reviewed_at": legacy["reviewed_at"],
                    "rationale": legacy["rationale"],
                    "quality_disposition": legacy["quality_disposition"],
                    "suitability_disposition": legacy["suitability_disposition"],
                }
            )
    for pair_id, pair in sorted(staged_pairs.items()):
        legacy = legacy_candidates[pair]
        migrated.append(
            {
                **{field: "" for field in REVIEW_CSV_FIELDS},
                "review_kind": "duplicate_pair",
                "review_key": pair_id,
                "item_id": pair[0],
                "candidate_item_id": pair[1],
                "review_status": "reviewed",
                "reviewer": legacy["reviewer"],
                "reviewed_at": legacy["reviewed_at"],
                "rationale": legacy["rationale"],
                "duplicate_disposition": legacy["duplicate_disposition"],
            }
        )
    return migrated


def migrate_authoritative_decisions(
    *,
    legacy_review_path: Path,
    sample_path: Path,
    active_path: Path,
    generation_root: Path,
) -> dict[str, int]:
    """Migrate only checksum-bound human decisions, then activate one final v2 generation."""

    stage, staged_rows, stage_paths = _load_authoritative_stage(generation_root)
    legacy_pointer_bytes = stage_paths["legacy_active"].read_bytes()
    if hashlib.sha256(legacy_pointer_bytes).hexdigest() != stage["legacy_active_sha256"]:
        raise ValueError("legacy active-pointer migration input changed")
    legacy_descriptor, legacy_rows = resolve_active_manifest(
        active_path=stage_paths["legacy_active"], generation_root=generation_root
    )
    if (
        legacy_descriptor["generation_id"] != stage["legacy_generation_id"]
        or legacy_descriptor["records_sha256"] != stage["legacy_records_sha256"]
    ):
        raise ValueError("legacy active generation disagrees with the migration stage")
    review_document = read_review(legacy_review_path)
    if review_document.sha256 != stage["legacy_review_sha256"]:
        raise ValueError("legacy human review bytes changed after authoritative audit")
    if hashlib.sha256(sample_path.read_bytes()).hexdigest() != stage["legacy_sample_sha256"]:
        raise ValueError("frozen sample bytes changed after authoritative audit")
    frozen_sample_ids = {row["item_id"] for row in _sample_rows(sample_path)}
    if frozen_sample_ids != {
        str(row["item_id"]) for row in staged_rows if row["audit_sample_member"] is True
    }:
        raise ValueError("frozen sample identities drifted during decision migration")

    migrated_review = _migrated_v2_review_rows(
        staged_rows=staged_rows,
        legacy_rows=legacy_rows,
        legacy_review_rows=review_document.rows,
    )
    updated_rows = _apply_v2_review_dispositions(staged_rows, migrated_review)
    source_descriptor = _read_descriptor(
        legacy_review_path.resolve().parents[1] / "sources" / "smb.yaml"
    )
    descriptor = _manifest_descriptor_v2(
        source_descriptor=source_descriptor,
        rows=updated_rows,
        source_provenance=stage["source_provenance"],
        creation_command=(
            "uv run python -m score_super_resolution.smb_audit migrate-authoritative "
            "decisions --legacy-review data/audits/smb-review-v1.csv "
            "--sample data/audits/smb-visual-sample-v1.csv "
            "--manifest-active data/manifests/smb-evaluation-v1.yaml "
            "--manifest-generation-root artifacts/smb-manifests/generations"
        ),
        deterministic_seed=int(stage["deterministic_seed"]),
    )
    _require_complete_denominator(updated_rows)
    counts = {
        "row_count": len(updated_rows),
        "processed": sum(row["processing_status"] == "processed" for row in updated_rows),
        "failed": sum(row["processing_status"] == "failed" for row in updated_rows),
        "paired_eligible": sum(row["paired_eligible"] is True for row in updated_rows),
        "groups": len({row["source_group_id"] for row in updated_rows}),
        "sampled_human": sum(
            isinstance(row["visual_review"], Mapping)
            and row["visual_review"].get("status") == "sampled_human_reviewed"
            for row in updated_rows
        ),
        "not_visually_reviewed": sum(
            isinstance(row["visual_review"], Mapping)
            and row["visual_review"].get("status") == "not_visually_reviewed"
            for row in updated_rows
        ),
        "perceptual_pairs": len(_v2_perceptual_pairs(updated_rows)),
    }
    required_counts = {
        "row_count": 685,
        "processed": 685,
        "failed": 0,
        "paired_eligible": 681,
        "groups": 260,
        "sampled_human": 64,
        "not_visually_reviewed": 621,
        "perceptual_pairs": 14,
    }
    if counts != required_counts:
        raise ValueError(f"authoritative migration count reconciliation failed: {counts}")
    if any(
        row["rights"]["item_provenance"]["status"] == "unavailable"  # type: ignore[index]
        and (
            row["rights"]["redistribution"]["status"] == "permitted"  # type: ignore[index]
            or row["rights"]["figure_reproduction"]["status"] == "permitted"  # type: ignore[index]
        )
        for row in updated_rows
    ):
        raise ValueError("unavailable per-item provenance inferred reuse permission")

    _validate_generation_inputs(descriptor, updated_rows)
    stage_paths["candidate_review"].write_bytes(legacy_review_path.read_bytes())
    candidate_review_sha256 = save_review(
        stage_paths["candidate_review"],
        migrated_review,
        expected_sha256=review_document.sha256,
    )
    publish_manifest_generation(
        active_path=stage_paths["candidate_active"],
        generation_root=generation_root,
        descriptor=descriptor,
        rows=updated_rows,
    )
    candidate_report = reconcile_manifest(
        active_path=stage_paths["candidate_active"], generation_root=generation_root
    )
    if {
        field: candidate_report[field]
        for field in ("row_count", "processed", "failed", "paired_eligible")
    } != {
        "row_count": 685,
        "processed": 685,
        "failed": 0,
        "paired_eligible": 681,
    }:
        raise ValueError("candidate v2 generation reconciliation failed")
    candidate_review = read_review(stage_paths["candidate_review"])
    if candidate_review.sha256 != candidate_review_sha256:
        raise ValueError("candidate v2 review persistence changed its identity")
    candidate_pointer_bytes = stage_paths["candidate_active"].read_bytes()
    current_pointer_bytes = active_path.read_bytes()
    if current_pointer_bytes not in {legacy_pointer_bytes, candidate_pointer_bytes}:
        raise ValueError("active manifest changed outside the authoritative migration")

    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=updated_rows,
    )
    current_review = read_review(legacy_review_path)
    if current_review.sha256 == review_document.sha256:
        save_review(
            legacy_review_path,
            migrated_review,
            expected_sha256=review_document.sha256,
        )
    elif current_review.sha256 != candidate_review_sha256:
        raise ValueError("human review changed during final authoritative activation")
    final_report = reconcile_manifest(active_path=active_path, generation_root=generation_root)
    if {
        field: final_report[field]
        for field in ("row_count", "processed", "failed", "paired_eligible")
    } != {
        "row_count": 685,
        "processed": 685,
        "failed": 0,
        "paired_eligible": 681,
    }:
        raise ValueError("final v2 generation reconciliation failed")
    return counts


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


def _install_boundary(name: str, hook: Callable[[str], None] | None) -> None:
    if name not in INSTALL_BOUNDARIES:
        raise ValueError(f"unknown install boundary: {name}")
    if hook is not None:
        hook(name)
    failpoint = os.environ.get("SCORE_SR_SMB_INSTALL_FAILPOINT")
    if failpoint == f"{name}:raise":
        raise OSError(f"injected install failure after {name}")
    if failpoint == f"{name}:exit":
        os._exit(92)


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


def _write_fsynced_at(
    parent_fd: int,
    basename: str,
    content: bytes,
    *,
    written_boundary: str,
    fsynced_boundary: str,
    boundary_hook: Callable[[str], None] | None,
) -> None:
    no_follow, _, close_on_exec = _secure_dirfd_support()
    descriptor = os.open(
        basename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | close_on_exec,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short publication write")
            view = view[written:]
        _publication_boundary(written_boundary, boundary_hook)
        os.fsync(descriptor)
        _publication_boundary(fsynced_boundary, boundary_hook)
    finally:
        os.close(descriptor)


def _read_regular_at(parent_fd: int, basename: str, *, label: str) -> bytes:
    no_follow, _, close_on_exec = _secure_dirfd_support()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | no_follow | close_on_exec | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise _publication_error(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, RECOVERY_READ_CHUNK_SIZE):
            chunks.append(chunk)
        result = b"".join(chunks)
        if len(result) != opened.st_size:
            raise _publication_error(f"{label} changed while being read")
        return result
    except ManifestPublicationError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise _publication_error(f"{label} must be a regular file") from error
        raise _publication_error(f"cannot read {label}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _retained_directory_fd(path: Path, *, label: str, create: bool) -> Iterable[int]:
    no_follow, directory_only, close_on_exec = _secure_dirfd_support()
    expanded = path.expanduser()
    if expanded.is_absolute():
        anchor = expanded.anchor
        components = expanded.parts[1:]
    else:
        anchor = "."
        components = expanded.parts
    if any(component in {"", ".", ".."} for component in components):
        raise _publication_error(f"invalid {label}")
    directory_fd: int | None = None
    try:
        directory_fd = os.open(
            anchor,
            os.O_RDONLY | directory_only | no_follow | close_on_exec,
        )
        for component in components:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | directory_only | no_follow | close_on_exec,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                os.fsync(directory_fd)
                next_fd = os.open(
                    component,
                    os.O_RDONLY | directory_only | no_follow | close_on_exec,
                    dir_fd=directory_fd,
                )
            os.close(directory_fd)
            directory_fd = next_fd
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise _publication_error(f"{label} must be a directory")
    except ManifestPublicationError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    except OSError as error:
        if directory_fd is not None:
            os.close(directory_fd)
        raise _publication_error(f"cannot access {label}") from error
    try:
        yield directory_fd
    finally:
        os.close(directory_fd)


def _remove_temporary_generation(root_fd: int, basename: str) -> None:
    no_follow, directory_only, close_on_exec = _secure_dirfd_support()
    try:
        temporary_fd = os.open(
            basename,
            os.O_RDONLY | directory_only | no_follow | close_on_exec,
            dir_fd=root_fd,
        )
    except FileNotFoundError:
        return
    try:
        for filename in (MANIFEST_DESCRIPTOR_FILENAME, MANIFEST_RECORDS_FILENAME):
            with suppress(FileNotFoundError):
                os.unlink(filename, dir_fd=temporary_fd)
    finally:
        os.close(temporary_fd)
    with suppress(FileNotFoundError):
        os.rmdir(basename, dir_fd=root_fd)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _uses_canonical_pixel_v2(descriptor: Mapping[str, object]) -> bool:
    hash_provenance = descriptor.get("hash_provenance")
    duplicate_provenance = descriptor.get("duplicate_provenance")
    pixels = hash_provenance.get("pixels") if isinstance(hash_provenance, Mapping) else None
    exact = duplicate_provenance.get("exact") if isinstance(duplicate_provenance, Mapping) else None
    return (
        isinstance(pixels, Mapping)
        and isinstance(exact, Mapping)
        and dict(pixels)
        == {
            "algorithm": "sha256",
            "version": 2,
            "canonicalization": "canonical-rgba-frame-v2",
            "domain_separator": "smb-canonical-rgba-frame-v2",
            "decoder_library": "Pillow",
            "decoder_version": CANONICAL_PIXEL_DECODER_VERSION,
            "output_mode": "RGBA8",
            "alpha_policy": "retain-alpha-and-underlying-rgb",
            "orientation_policy": "stored-raster-ignore-exif",
            "metadata_policy": "ignore-non-raster-metadata",
            "max_encoded_bytes": DEFAULT_MAX_ENCODED_BYTES,
            "max_pixels": DEFAULT_MAX_PIXELS,
            "failure_policy": "safe-explicit-failure-no-digest",
        }
        and dict(exact) == {"algorithm": "canonical-pixel-sha256", "version": 2}
    )


def validate_v2_manifest_collection(
    descriptor: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    allow_legacy_hash_provenance: bool = False,
) -> None:
    """Enforce v2 invariants that cannot be expressed by one row's JSON Schema."""

    try:
        descriptor_contract = dict(descriptor)
        descriptor_contract.setdefault("generation_id", "0" * 64)
        validate_instance("manifest-descriptor", descriptor_contract, version=2)
        if not _uses_canonical_pixel_v2(descriptor) and not allow_legacy_hash_provenance:
            raise ValueError("new v2 evidence requires canonical-pixel-sha256 version 2")
        _require_complete_audit_rows(rows)
        by_id = {str(row["item_id"]): row for row in rows}
        if any(row.get("item_id") != _safe_item_id(int(row["upstream_index"])) for row in rows):
            raise ValueError("item identity does not match upstream index")

        selection = descriptor["sample_selection"]
        assert isinstance(selection, Mapping)
        if selection["seed"] != descriptor["deterministic_seed"]:
            raise ValueError("sample selection seed disagrees with deterministic seed")
        expected_selection = {
            "algorithm": "sha256-rank",
            "version": 1,
            "seed": descriptor["deterministic_seed"],
            "population_size": len(rows),
            "sample_size": DEFAULT_SAMPLE_SIZE,
            "identity_fields": ["upstream_index", "item_id"],
            "selection_state": "pre-review",
        }
        if dict(selection) != expected_selection:
            raise ValueError("sample selection contract is unsupported or inconsistent")
        selected_ids = set(
            select_visual_sample(
                rows,
                seed=int(selection["seed"]),
                sample_size=int(selection["sample_size"]),
            )
        )
        actual_selected_ids = {
            str(row["item_id"]) for row in rows if row["audit_sample_member"] is True
        }
        if actual_selected_ids != selected_ids:
            raise ValueError("sample membership disagrees with deterministic selection")

        source_contract = {
            "source_key": descriptor["source_key"],
            "source_revision": descriptor["source_revision"],
            "split": descriptor["upstream_split"],
        }
        for index, row in enumerate(rows):
            for field, expected in source_contract.items():
                if row.get(field) != expected:
                    raise ValueError(f"row {index}: source {field} disagrees with descriptor")
            source_identity = row["source_identity"]
            assert isinstance(source_identity, Mapping)
            expected_group = _canonical_source_group_id(
                source_identity.get("original_score_normalized")
            )
            if row.get("source_group_id") != expected_group:
                raise ValueError(f"row {index}: source group disagrees with audited identity")

        derived_exclusions = sorted(
            (
                {
                    "upstream_index": row["upstream_index"],
                    "item_id": row["item_id"],
                    "reason": row["paired_ineligibility_reason"],
                }
                for row in rows
                if row["paired_eligible"] is False
            ),
            key=lambda exclusion: (
                int(exclusion["upstream_index"]),
                str(exclusion["item_id"]),
                str(exclusion["reason"]),
            ),
        )
        descriptor_exclusions = descriptor["exclusions"]
        assert isinstance(descriptor_exclusions, Sequence)
        exclusion_identities = [
            (exclusion["upstream_index"], exclusion["item_id"])
            for exclusion in descriptor_exclusions
            if isinstance(exclusion, Mapping)
        ]
        derived_identities = [
            (exclusion["upstream_index"], exclusion["item_id"]) for exclusion in derived_exclusions
        ]
        if len(exclusion_identities) != len(set(exclusion_identities)) or len(
            derived_identities
        ) != len(set(derived_identities)):
            raise ValueError("exclusion ledger contains duplicate identities")
        if list(descriptor_exclusions) != derived_exclusions:
            raise ValueError("exclusion ledger disagrees with paired-ineligible rows")

        sampled_count = 0
        visual_counts: Counter[str] = Counter()
        automated_count = 0
        relation_occurrences: dict[str, list[tuple[str, Mapping[str, object]]]] = defaultdict(list)
        exact_pairs: set[str] = set()
        perceptual_pairs: dict[str, Mapping[str, object]] = {}
        for index, row in enumerate(rows):
            validate_instance("manifest-row", row, version=2)
            item_id = str(row["item_id"])
            sampled = row["audit_sample_member"] is True
            sampled_count += sampled
            visual = row["visual_review"]
            assert isinstance(visual, Mapping)
            visual_status = str(visual["status"])
            visual_counts[visual_status] += 1
            if sampled and visual_status != "sampled_human_reviewed":
                raise ValueError(f"row {index}: frozen sample lacks sampled human evidence")
            if not sampled and visual_status == "sampled_human_reviewed":
                raise ValueError(f"row {index}: unsampled item claims sampled review")
            automated = row["automated_audit"]
            assert isinstance(automated, Mapping)
            automated_count += automated.get("status") == "automated"

            relations = row["duplicate_relations"]
            assert isinstance(relations, Sequence)
            pair_ids: list[str] = []
            perceptual_ids: list[str] = []
            for relation in relations:
                assert isinstance(relation, Mapping)
                pair_id = str(relation["pair_id"])
                pair_ids.append(pair_id)
                candidate_type = str(relation["candidate_type"])
                item_ids = relation["item_ids"]
                if not isinstance(item_ids, Sequence) or isinstance(item_ids, (str, bytes)):
                    raise ValueError(f"{pair_id}: invalid pair identities")
                canonical_items = sorted(str(value) for value in item_ids)
                counterpart = str(relation["counterpart_item_id"])
                if (
                    canonical_items != list(item_ids)
                    or canonical_items != sorted((item_id, counterpart))
                    or item_id == counterpart
                    or counterpart not in by_id
                ):
                    raise ValueError(f"{pair_id}: swapped or unknown pair identity")
                if pair_id != _pair_id(candidate_type, *canonical_items):
                    raise ValueError(f"{pair_id}: non-canonical pair key")
                relation_occurrences[pair_id].append((item_id, relation))
                if candidate_type == "exact":
                    exact_pairs.add(pair_id)
                    evidence = relation["evidence"]
                    assert isinstance(evidence, Mapping)
                    member_images = []
                    for member_id in canonical_items:
                        image = by_id[member_id]["image"]
                        assert isinstance(image, Mapping)
                        member_images.append(image)
                        if image.get("pixel_sha256") != evidence.get("pixel_sha256"):
                            raise ValueError(
                                f"{pair_id}: canonical pixel evidence does not match both rows"
                            )
                    encoded_values = [image.get("encoded_sha256") for image in member_images]
                    encoded_equality = (
                        isinstance(encoded_values[0], str)
                        and encoded_values[0] == encoded_values[1]
                    )
                    expected_encoded = encoded_values[0] if encoded_equality else None
                    if (
                        evidence.get("encoded_equality") is not encoded_equality
                        or evidence.get("encoded_sha256") != expected_encoded
                    ):
                        raise ValueError(f"{pair_id}: encoded equality evidence is untruthful")
                else:
                    perceptual_ids.append(pair_id)
                    perceptual_pairs[pair_id] = relation
            if len(pair_ids) != len(set(pair_ids)):
                raise ValueError(f"{item_id}: duplicate pair relation")
            if sorted(row["near_duplicate_candidate_ids"]) != sorted(perceptual_ids):
                raise ValueError(f"{item_id}: perceptual candidate summary disagrees")

        for pair_id, occurrences in relation_occurrences.items():
            if len(occurrences) != 2 or len({item_id for item_id, _ in occurrences}) != 2:
                raise ValueError(f"{pair_id}: missing mirrored relation")
            normalized = []
            for _item_id, relation in occurrences:
                value = dict(relation)
                value.pop("counterpart_item_id", None)
                normalized.append(value)
            if normalized[0] != normalized[1]:
                raise ValueError(f"{pair_id}: mirrored relations disagree")

        expected_exact_pairs: set[str] = set()
        exact_groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if row["processing_status"] != "processed":
                continue
            image = row["image"]
            assert isinstance(image, Mapping)
            pixel_sha256 = image.get("pixel_sha256")
            if isinstance(pixel_sha256, str):
                exact_groups[pixel_sha256].append(str(row["item_id"]))
        for members in exact_groups.values():
            expected_exact_pairs.update(
                _pair_id("exact", first, second)
                for first, second in combinations(sorted(members), 2)
            )
        if exact_pairs != expected_exact_pairs:
            raise ValueError("exact cryptographic pairs are missing or spurious")

        group_ids = _v2_group_ids_by_item(rows)
        for row in rows:
            expected_summary = _v2_summary(row, group_ids=group_ids[str(row["item_id"])])
            if row["duplicate_summary"] != expected_summary:
                raise ValueError(f"{row['item_id']}: derived duplicate summary disagrees")

        inference = descriptor.get("review_inference")
        if not isinstance(inference, Mapping):
            raise ValueError("descriptor lacks review inference")
        expected_inference = {
            "automated_population_audit_count": automated_count,
            "sampled_human_review_count": visual_counts["sampled_human_reviewed"],
            "targeted_human_review_count": visual_counts["targeted_human_reviewed"],
            "not_visually_reviewed_count": visual_counts["not_visually_reviewed"],
            "unavailable_visual_review_count": visual_counts["unavailable"],
            "not_applicable_visual_review_count": visual_counts["not_applicable"],
            "exact_pair_automated_count": len(exact_pairs),
            "perceptual_pair_count": len(perceptual_pairs),
            "perceptual_pair_human_review_count": sum(
                relation["disposition"] in {"distinct", "duplicate", "related"}
                for relation in perceptual_pairs.values()
            ),
            "perceptual_pair_pending_count": sum(
                relation["disposition"] == "pending" for relation in perceptual_pairs.values()
            ),
        }
        if sampled_count != DEFAULT_SAMPLE_SIZE:
            raise ValueError("frozen visual sample must contain exactly 64 identities")
        for field, expected in expected_inference.items():
            if inference.get(field) != expected:
                raise ValueError(f"descriptor review inference disagrees for {field}")
    except ManifestPublicationError:
        raise
    except (AssertionError, ContractValidationError, KeyError, TypeError, ValueError) as error:
        raise _publication_error(f"v2 collection validation failed: {error}") from error


def _validate_generation_inputs(
    descriptor: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    allow_legacy_hash_provenance: bool = False,
) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    try:
        schema_version = descriptor.get("schema_version")
        if isinstance(schema_version, bool) or schema_version not in {1, 2}:
            raise _publication_error("unsupported manifest schema version")
        version = int(schema_version)
        for index, row in enumerate(rows):
            try:
                validate_instance("manifest-row", row, version=version)
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
        if version == 2:
            validate_v2_manifest_collection(
                completed_descriptor,
                rows,
                allow_legacy_hash_provenance=allow_legacy_hash_provenance,
            )
        generation_id = _generation_id(completed_descriptor, records_bytes)
        completed_descriptor["generation_id"] = generation_id
        validate_instance("manifest-descriptor", completed_descriptor, version=version)
        descriptor_bytes = _canonical_descriptor(completed_descriptor)
        pointer = {
            "schema_version": version,
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
        validate_instance("manifest-active", pointer, version=version)
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
    allow_legacy_hash_provenance: bool = False,
    recovery_binding: Mapping[str, object] | None = None,
) -> None:
    """Validate and publish one immutable content-addressed generation through one pointer."""

    _, pointer, descriptor_bytes, records_bytes = _validate_generation_inputs(
        descriptor,
        rows,
        allow_legacy_hash_provenance=allow_legacy_hash_provenance,
    )
    if recovery_binding is not None:
        pointer.update(dict(recovery_binding))
        validate_instance("manifest-active", pointer, version=int(pointer["schema_version"]))
    generation_id = str(pointer["generation_id"])
    if _SAFE_METADATA_PATTERN.fullmatch(generation_id) is None:
        raise _publication_error("generation path escapes generation root")
    temp_generation = f".tmp-{generation_id}-{uuid.uuid4().hex}"
    committed = False
    try:
        with _retained_directory_fd(
            generation_root, label="generation root", create=True
        ) as root_fd:
            _publication_boundary("generation_parent_anchored", boundary_hook)
            try:
                generation_fd = os.open(
                    generation_id,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                generation_fd = None
            if generation_fd is not None:
                try:
                    if (
                        _read_regular_at(
                            generation_fd,
                            MANIFEST_DESCRIPTOR_FILENAME,
                            label="existing generation descriptor",
                        )
                        != descriptor_bytes
                        or _read_regular_at(
                            generation_fd,
                            MANIFEST_RECORDS_FILENAME,
                            label="existing generation records",
                        )
                        != records_bytes
                    ):
                        raise _publication_error("existing generation is not byte-identical")
                finally:
                    os.close(generation_fd)
            else:
                os.mkdir(temp_generation, mode=0o700, dir_fd=root_fd)
                temporary_fd = os.open(
                    temp_generation,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=root_fd,
                )
                try:
                    _write_fsynced_at(
                        temporary_fd,
                        MANIFEST_RECORDS_FILENAME,
                        records_bytes,
                        written_boundary="generation_records_written",
                        fsynced_boundary="generation_records_fsynced",
                        boundary_hook=boundary_hook,
                    )
                    _write_fsynced_at(
                        temporary_fd,
                        MANIFEST_DESCRIPTOR_FILENAME,
                        descriptor_bytes,
                        written_boundary="generation_descriptor_written",
                        fsynced_boundary="generation_descriptor_fsynced",
                        boundary_hook=boundary_hook,
                    )
                    os.fsync(temporary_fd)
                    _publication_boundary("temporary_generation_directory_fsynced", boundary_hook)
                finally:
                    os.close(temporary_fd)
                os.rename(
                    temp_generation,
                    generation_id,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                )
                _publication_boundary("generation_renamed", boundary_hook)
                os.fsync(root_fd)
                _publication_boundary("generations_parent_fsynced", boundary_hook)

            pointer_bytes = _canonical_descriptor(pointer)
            with _retained_directory_fd(
                active_path.parent, label="active pointer parent", create=True
            ) as active_parent_fd:
                _publication_boundary("active_parent_anchored", boundary_hook)
                try:
                    existing_pointer = _read_regular_at(
                        active_parent_fd,
                        active_path.name,
                        label="active pointer",
                    )
                except ManifestPublicationError as error:
                    cause = error.__cause__
                    if not isinstance(cause, FileNotFoundError):
                        raise
                    existing_pointer = None
                if existing_pointer == pointer_bytes:
                    return
                pointer_temp = f".{active_path.name}.tmp-{uuid.uuid4().hex}"
                try:
                    _write_fsynced_at(
                        active_parent_fd,
                        pointer_temp,
                        pointer_bytes,
                        written_boundary="pointer_written",
                        fsynced_boundary="pointer_fsynced",
                        boundary_hook=boundary_hook,
                    )
                    os.replace(
                        pointer_temp,
                        active_path.name,
                        src_dir_fd=active_parent_fd,
                        dst_dir_fd=active_parent_fd,
                    )
                    committed = True
                    _publication_boundary("pointer_replaced", boundary_hook)
                    os.fsync(active_parent_fd)
                    _publication_boundary("active_parent_fsynced", boundary_hook)
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(pointer_temp, dir_fd=active_parent_fd)
    except ManifestPublicationError:
        raise
    except Exception as error:
        detail = error.strerror if isinstance(error, OSError) else type(error).__name__
        raise _publication_error(f"publication failed: {detail}", committed=committed) from error
    finally:
        try:
            with _retained_directory_fd(
                generation_root, label="generation root", create=False
            ) as root_fd:
                _remove_temporary_generation(root_fd, temp_generation)
        except ManifestPublicationError:
            pass


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
    pointer_version = pointer.get("schema_version")
    if isinstance(pointer_version, bool) or pointer_version not in {1, 2}:
        raise _publication_error(
            "active pointer names an unsupported schema version", committed=True
        )
    version = int(pointer_version)
    validate_instance("manifest-active", pointer, version=version)
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
    if descriptor.get("schema_version") != version:
        raise _publication_error("pointer and descriptor schema versions disagree", committed=True)
    validate_instance("manifest-descriptor", descriptor, version=version)
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
            validate_instance("manifest-row", loaded, version=version)
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
    if version == 2:
        try:
            validate_v2_manifest_collection(descriptor, rows, allow_legacy_hash_provenance=True)
        except ManifestPublicationError as error:
            raise _publication_error(str(error), committed=True) from error
    return descriptor, rows


def reconcile_manifest(
    *,
    active_path: Path,
    generation_root: Path,
    expected_indices: Iterable[int] = range(EXPECTED_ROW_COUNT),
) -> dict[str, object]:
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
    report: dict[str, object] = {
        "row_count": len(rows),
        "processed": sum(row["processing_status"] == "processed" for row in rows),
        "failed": sum(row["processing_status"] == "failed" for row in rows),
        "paired_eligible": sum(row["paired_eligible"] is True for row in rows),
    }
    if descriptor["schema_version"] == 2:
        report.update(
            {
                "generation_id": descriptor["generation_id"],
                "records_sha256": descriptor["records_sha256"],
                "benchmark_state": descriptor["benchmark_state"],
                "exclusion_count": len(descriptor["exclusions"]),
                "source_group_count": len(
                    {
                        row["source_group_id"]
                        for row in rows
                        if isinstance(row["source_group_id"], str) and row["source_group_id"]
                    }
                ),
            }
        )
    return report


def _manifest_recovery_limits(version: int = 1) -> tuple[int, int]:
    schema = load_schema("manifest-recovery", version=version)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):  # pragma: no cover - schema self-check owns this
        raise RuntimeError("manifest recovery schema has no properties")
    compressed = properties.get("compressed_size_bytes")
    uncompressed = properties.get("uncompressed_size_bytes")
    if not isinstance(compressed, Mapping) or not isinstance(uncompressed, Mapping):
        raise RuntimeError("manifest recovery schema has no byte limits")
    compressed_maximum = compressed.get("maximum")
    uncompressed_maximum = uncompressed.get("maximum")
    if (
        isinstance(compressed_maximum, bool)
        or not isinstance(compressed_maximum, int)
        or isinstance(uncompressed_maximum, bool)
        or not isinstance(uncompressed_maximum, int)
        or compressed_maximum < 1
        or uncompressed_maximum < 1
    ):
        raise RuntimeError("manifest recovery schema byte limits are invalid")
    return compressed_maximum, uncompressed_maximum


def _reject_symlink_components(path: Path, *, label: str, final_may_not_exist: bool) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            opened = current.lstat()
        except FileNotFoundError:
            if final_may_not_exist and index == len(absolute.parts) - 1:
                return absolute
            if final_may_not_exist:
                continue
            raise _publication_error(f"{label} must be a regular file") from None
        except OSError as error:
            raise _publication_error(f"cannot inspect {label}") from error
        if stat.S_ISLNK(opened.st_mode):
            raise _publication_error(f"{label} must be a regular file")
    return absolute


def _secure_dirfd_support() -> tuple[int, int, int]:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_only = getattr(os, "O_DIRECTORY", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    if (
        no_follow == 0
        or directory_only == 0
        or not _OS_OPEN_SUPPORTS_DIR_FD
        or not _OS_STAT_SUPPORTS_DIR_FD
    ):
        raise _publication_error("secure no-follow recovery reads are unavailable")
    return no_follow, directory_only, close_on_exec


@contextmanager
def _retained_parent_dirfd(path: Path, *, label: str) -> Iterable[tuple[int, str]]:
    """Retain the verified parent inode while a basename-only operation runs."""

    no_follow, directory_only, close_on_exec = _secure_dirfd_support()
    expanded = path.expanduser()
    if expanded.is_absolute():
        anchor = expanded.anchor
        components = expanded.parts[1:]
    else:
        anchor = "."
        components = expanded.parts
    if not components or any(component in {"", ".", ".."} for component in components):
        raise _publication_error(f"{label} must be a regular file")
    directory_fd: int | None = None
    try:
        directory_fd = os.open(
            anchor,
            os.O_RDONLY | directory_only | no_follow | close_on_exec,
        )
        for component in components[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory_only | no_follow | close_on_exec,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
    except ManifestPublicationError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    except OSError as error:
        if directory_fd is not None:
            os.close(directory_fd)
        raise _publication_error(f"cannot inspect {label}") from error
    try:
        yield directory_fd, components[-1]
    finally:
        os.close(directory_fd)


def _read_regular_nofollow(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    no_follow, _, close_on_exec = _secure_dirfd_support()
    non_blocking = getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        with _retained_parent_dirfd(path, label=label) as (parent_fd, basename):
            descriptor = os.open(
                basename,
                os.O_RDONLY | no_follow | close_on_exec | non_blocking,
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise _publication_error(f"{label} must be a regular file")
            if opened.st_size > maximum_bytes:
                raise _publication_error(f"{label} exceeds its schema-declared maximum")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(RECOVERY_READ_CHUNK_SIZE, maximum_bytes + 1 - consumed),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                consumed += len(chunk)
                if consumed > maximum_bytes:
                    raise _publication_error(f"{label} exceeds its schema-declared maximum")
            if consumed != opened.st_size:
                raise _publication_error(f"{label} changed while being read")
            return b"".join(chunks)
    except ManifestPublicationError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise _publication_error(f"{label} must be a regular file") from error
        raise _publication_error(f"cannot read {label}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _deterministic_gzip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(payload)
    return output.getvalue()


def _assert_safe_recovery_content(value: object, *, path: str = "$") -> None:
    forbidden_keys = {
        "authorization",
        "checkpoint",
        "checkpoints",
        "content",
        "credential",
        "credentials",
        "data",
        "image_bytes",
        "lr",
        "metric",
        "metrics",
        "model",
        "models",
        "outcome",
        "outcomes",
        "password",
        "prediction",
        "predictions",
        "raw",
        "secret",
        "sr",
        "token",
        "weight",
        "weights",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise _publication_error(f"recovery content has a non-string key at {path}")
            if key.casefold() in forbidden_keys:
                raise _publication_error(f"recovery content contains forbidden field {path}.{key}")
            _assert_safe_recovery_content(nested, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_safe_recovery_content(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, (bytes, bytearray)):
        raise _publication_error(f"recovery content contains binary payload at {path}")
    if isinstance(value, str):
        if value.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise _publication_error(
                f"recovery content contains an absolute machine path at {path}"
            )
        if re.search(r"(?:hf_|ghp_|sk-)[A-Za-z0-9_-]{12,}", value) or "PRIVATE KEY" in value:
            raise _publication_error(f"recovery content contains secret-like material at {path}")


def _build_manifest_recovery_v2(
    descriptor: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    bytes,
    bytes,
    bytes,
    bytes,
]:
    """Build non-cyclic recovery metadata and its corrected one-way active pointer."""

    compressed_maximum, uncompressed_maximum = _manifest_recovery_limits(2)
    completed, pointer, descriptor_bytes, records_bytes = _validate_generation_inputs(
        descriptor, rows
    )
    if len(records_bytes) > uncompressed_maximum:
        raise _publication_error("active records exceed the recovery-v2 uncompressed maximum")
    compressed_bytes = _deterministic_gzip(records_bytes)
    if len(compressed_bytes) > compressed_maximum:
        raise _publication_error("recovery-v2 gzip exceeds its compressed maximum")
    recovery: dict[str, object] = {
        "schema_version": 2,
        "record_type": "manifest-recovery",
        "bundle_id": "",
        "active_pointer_path": "data/manifests/smb-evaluation-v1.yaml",
        "manifest_id": completed["manifest_id"],
        "generation_id": completed["generation_id"],
        "descriptor_path": pointer["descriptor_path"],
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "descriptor_yaml": descriptor_bytes.decode("utf-8"),
        "records_path": pointer["records_path"],
        "row_schema_id": completed["row_schema_id"],
        "row_schema_version": completed["row_schema_version"],
        "row_count": completed["row_count"],
        "records_sha256": completed["records_sha256"],
        "source_revision": completed["source_revision"],
        "source_provenance": completed["source_provenance"],
        "audit_version": completed["audit_version"],
        "benchmark_state": completed["benchmark_state"],
        "recovery_descriptor_path": "",
        "recovery_records_path": "",
        "compression": {
            "algorithm": "gzip",
            "format_version": 1,
            "compresslevel": 9,
            "mtime": 0,
            "filename": "",
        },
        "compressed_sha256": hashlib.sha256(compressed_bytes).hexdigest(),
        "compressed_size_bytes": len(compressed_bytes),
        "uncompressed_size_bytes": len(records_bytes),
        "metadata_sha256": "",
    }
    recovery["bundle_id"] = recovery_bundle_id_v2(recovery)
    bundle_prefix = f"data/manifests/recovery/canonical-pixel-v2/{recovery['bundle_id']}"
    recovery["recovery_descriptor_path"] = f"{bundle_prefix}/manifest-recovery.yaml"
    recovery["recovery_records_path"] = f"{bundle_prefix}/{RECOVERY_RECORDS_FILENAME}"
    recovery["metadata_sha256"] = recovery_metadata_sha256(recovery, version=2)
    validate_instance("manifest-recovery", recovery, version=2)
    recovery_bytes = _canonical_descriptor(recovery)
    pointer.update(
        {
            "recovery_descriptor_path": recovery["recovery_descriptor_path"],
            "recovery_descriptor_sha256": hashlib.sha256(recovery_bytes).hexdigest(),
            "recovery_records_path": recovery["recovery_records_path"],
            "recovery_records_sha256": recovery["compressed_sha256"],
        }
    )
    validate_instance("manifest-active", pointer, version=2)
    return (
        completed,
        pointer,
        recovery,
        descriptor_bytes,
        records_bytes,
        compressed_bytes,
        recovery_bytes,
    )


def _candidate_benchmark_state(descriptor: Mapping[str, object]) -> str:
    return str(descriptor.get("benchmark_state", ""))


def _perceptual_relation_records(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        relations = row.get("duplicate_relations")
        if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes)):
            raise ValueError("protected perceptual relations are malformed")
        for relation in relations:
            if not isinstance(relation, Mapping) or relation.get("candidate_type") != "perceptual":
                continue
            normalized = dict(relation)
            normalized.pop("counterpart_item_id", None)
            pair_id = str(normalized.get("pair_id", ""))
            previous = records.setdefault(pair_id, normalized)
            if previous != normalized:
                raise ValueError("protected perceptual relation mirrors disagree")
    return records


def _join_protected_candidate_evidence(
    fresh_rows: Sequence[Mapping[str, object]],
    legacy_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join only reviewed/stable evidence; newly computed hashes remain authoritative."""

    legacy_by_id = {str(row.get("item_id")): row for row in legacy_rows}
    fresh_by_id = {str(row.get("item_id")): row for row in fresh_rows}
    if len(legacy_by_id) != len(legacy_rows) or set(fresh_by_id) != set(legacy_by_id):
        raise ValueError("protected stable item identities changed during canonical rehash")

    stable_fields = ("audit_sample_member", "source_group_id", "source_identity", "rights")
    for item_id, legacy in legacy_by_id.items():
        fresh = fresh_by_id[item_id]
        for field in stable_fields:
            if fresh.get(field) != legacy.get(field):
                raise ValueError(f"protected {field} evidence changed during canonical rehash")
        if fresh.get("paired_eligible") != legacy.get("paired_eligible") or fresh.get(
            "paired_ineligibility_reason"
        ) != legacy.get("paired_ineligibility_reason"):
            raise ValueError("protected eligibility evidence changed during canonical rehash")

        fresh_visual = fresh.get("visual_review")
        legacy_visual = legacy.get("visual_review")
        if fresh_visual != legacy_visual:
            fresh_status = fresh_visual.get("status") if isinstance(fresh_visual, Mapping) else None
            if fresh_status not in {"not_visually_reviewed", "unavailable"}:
                raise ValueError("protected visual_review evidence changed during canonical rehash")

    legacy_perceptual = _perceptual_relation_records(legacy_rows)
    fresh_perceptual = _perceptual_relation_records(fresh_rows)
    if set(fresh_perceptual) != set(legacy_perceptual):
        raise ValueError("protected perceptual pair identities changed during canonical rehash")
    human_fields = ("disposition", "reviewer", "reviewed_at", "rationale")
    identity_fields = ("pair_id", "candidate_type", "item_ids", "evidence_basis", "evidence")
    for pair_id, legacy_relation in legacy_perceptual.items():
        fresh_relation = fresh_perceptual[pair_id]
        if any(
            fresh_relation.get(field) != legacy_relation.get(field) for field in identity_fields
        ):
            raise ValueError("protected perceptual pair evidence changed during canonical rehash")
        if fresh_relation.get("disposition") != "pending" and any(
            fresh_relation.get(field) != legacy_relation.get(field) for field in human_fields
        ):
            raise ValueError("protected perceptual review changed during canonical rehash")

    migrated: list[dict[str, object]] = []
    for fresh in fresh_rows:
        item_id = str(fresh["item_id"])
        legacy = legacy_by_id[item_id]
        row = copy.deepcopy(dict(fresh))
        row["visual_review"] = copy.deepcopy(legacy["visual_review"])
        legacy_relations = legacy.get("duplicate_relations")
        if not isinstance(legacy_relations, Sequence):
            raise ValueError("protected perceptual relations are malformed")
        row["duplicate_relations"] = [
            copy.deepcopy(dict(relation))
            for relation in legacy_relations
            if isinstance(relation, Mapping) and relation.get("candidate_type") == "perceptual"
        ]
        migrated.append(row)
    return derive_v2_exact_relations(migrated)


def _candidate_tree_bytes(root: Path) -> dict[str, bytes]:
    active_relative = Path("data/manifests/smb-evaluation-v1.yaml")
    active_path = root / active_relative
    if not active_path.exists():
        synthetic_relative = Path("candidate.yaml")
        return {synthetic_relative.as_posix(): (root / synthetic_relative).read_bytes()}
    active_bytes = active_path.read_bytes()
    try:
        pointer = yaml.safe_load(active_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("candidate active pointer is invalid") from error
    if not isinstance(pointer, Mapping):
        raise ValueError("candidate active pointer is invalid")
    generation_prefix = Path("artifacts/smb-manifests/generations")
    relatives = (
        active_relative,
        generation_prefix / str(pointer["descriptor_path"]),
        generation_prefix / str(pointer["records_path"]),
        Path(str(pointer["recovery_descriptor_path"])),
        Path(str(pointer["recovery_records_path"])),
        Path("install-metadata.yaml"),
    )
    return {relative.as_posix(): (root / relative).read_bytes() for relative in relatives}


def build_canonical_pixel_rehash_candidate(
    *,
    source_path: Path,
    trusted_cache_roots: Sequence[Path],
    legacy_active_path: Path,
    legacy_recovery_descriptor_path: Path,
    legacy_recovery_records_path: Path,
    staging_root: Path,
    dataset_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Build and validate a corrected candidate entirely below ``staging_root``."""

    source_descriptor = _read_descriptor(source_path)
    original_active_bytes = legacy_active_path.read_bytes()
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smb-canonical-rehash-") as temporary:
        temporary_root = Path(temporary)
        local_active = temporary_root / "data/manifests/smb-evaluation-v1.yaml"
        legacy_generation_root = temporary_root / "generations"
        _durable_replace(local_active, original_active_bytes)
        recover_active_manifest(
            active_path=local_active,
            recovery_descriptor_path=legacy_recovery_descriptor_path,
            recovery_records_path=legacy_recovery_records_path,
            generation_root=legacy_generation_root,
        )
        legacy_descriptor, legacy_rows = resolve_active_manifest(
            active_path=local_active, generation_root=legacy_generation_root
        )

        if _candidate_benchmark_state(legacy_descriptor) != "AUDITED_LOCKED":
            raise ValueError("candidate migration requires protected AUDITED_LOCKED state")
        records = load_smb(
            purpose=BenchmarkPurpose.CONTENT_AUDIT,
            loader=dataset_loader,
            descriptor_path=source_path,
        )
        records = _without_automatic_image_decoding(records)
        if not isinstance(records, Sequence):
            records = list(records)  # type: ignore[arg-type]
        if len(records) != EXPECTED_ROW_COUNT:
            raise ValueError("authenticated canonical rehash must load exactly 685 rows")
        fresh_rows = audit_dataset_v2(
            records,
            source_descriptor=source_descriptor,
            trusted_cache_roots=trusted_cache_roots,
            deterministic_seed=int(legacy_descriptor["deterministic_seed"]),
            sample_size=DEFAULT_SAMPLE_SIZE,
        )
        _require_complete_audit_rows(fresh_rows)
        migrated_rows = _join_protected_candidate_evidence(fresh_rows, legacy_rows)
        _require_complete_audit_rows(migrated_rows)

        audited_revisions = {str(row["source_revision"]) for row in migrated_rows}
        if len(audited_revisions) != 1 or audited_revisions != {
            str(legacy_descriptor["source_revision"])
        }:
            raise ValueError("protected source revision changed during canonical rehash")
        candidate_source_descriptor = dict(source_descriptor)
        candidate_source_descriptor["revision"] = audited_revisions.pop()

        source_provenance = audit_source_provenance(source_path.resolve().parents[2])
        descriptor = _manifest_descriptor_v2(
            source_descriptor=candidate_source_descriptor,
            rows=migrated_rows,
            source_provenance=source_provenance,
            creation_command=(
                "uv run python -m score_super_resolution.smb_audit "
                "build-canonical-pixel-rehash-candidate"
            ),
            deterministic_seed=int(legacy_descriptor["deterministic_seed"]),
        )
        descriptor["benchmark_state"] = _candidate_benchmark_state(legacy_descriptor)
        (
            completed,
            pointer,
            recovery,
            descriptor_bytes,
            records_bytes,
            compressed_bytes,
            recovery_bytes,
        ) = _build_manifest_recovery_v2(descriptor, migrated_rows)
        _assert_safe_recovery_content(completed)
        _assert_safe_recovery_content(migrated_rows)
        _assert_safe_recovery_content(recovery)

        generation_root = staging_root / "artifacts/smb-manifests/generations"
        generation_path = generation_root / str(completed["generation_id"])
        active_path = staging_root / str(recovery["active_pointer_path"])
        recovery_descriptor_path = staging_root / str(recovery["recovery_descriptor_path"])
        recovery_records_path = staging_root / str(recovery["recovery_records_path"])
        _durable_replace(generation_path / MANIFEST_DESCRIPTOR_FILENAME, descriptor_bytes)
        _durable_replace(generation_path / MANIFEST_RECORDS_FILENAME, records_bytes)
        _durable_replace(recovery_records_path, compressed_bytes)
        _durable_replace(recovery_descriptor_path, recovery_bytes)
        _durable_replace(active_path, _canonical_descriptor(pointer))
        install_metadata = {
            "schema_version": 1,
            "record_type": "canonical-pixel-candidate-install",
            "expected_previous_active_sha256": hashlib.sha256(original_active_bytes).hexdigest(),
            "generation_id": completed["generation_id"],
            "bundle_id": recovery["bundle_id"],
            "active_pointer_path": recovery["active_pointer_path"],
            "recovery_descriptor_path": recovery["recovery_descriptor_path"],
            "recovery_records_path": recovery["recovery_records_path"],
        }
        _assert_safe_recovery_content(install_metadata)
        _durable_replace(
            staging_root / "install-metadata.yaml", _canonical_descriptor(install_metadata)
        )

        resolved_descriptor, resolved_rows = resolve_active_manifest(
            active_path=active_path, generation_root=generation_root
        )
        if resolved_descriptor != completed or resolved_rows != migrated_rows:
            raise ValueError("staged candidate does not resolve to its canonical generation")
        validation_active = temporary_root / "validation/data/manifests/smb-evaluation-v1.yaml"
        _durable_replace(validation_active, _canonical_descriptor(pointer))
        recover_active_manifest(
            active_path=validation_active,
            recovery_descriptor_path=recovery_descriptor_path,
            recovery_records_path=recovery_records_path,
            generation_root=temporary_root / "validation-generations",
        )
        if legacy_active_path.read_bytes() != original_active_bytes:
            raise ValueError("candidate construction changed the real active pointer")

    return {
        "generation_id": completed["generation_id"],
        "bundle_id": recovery["bundle_id"],
        "row_count": len(migrated_rows),
        "sampled_human_review_count": sum(
            row["audit_sample_member"] is True for row in migrated_rows
        ),
        "perceptual_pair_count": len(_perceptual_relation_records(migrated_rows)),
        "source_group_count": len({row["source_group_id"] for row in migrated_rows}),
        "benchmark_state": completed["benchmark_state"],
        "active_pointer_path": recovery["active_pointer_path"],
        "recovery_descriptor_path": recovery["recovery_descriptor_path"],
        "recovery_records_path": recovery["recovery_records_path"],
    }


def verify_authoritative_determinism(
    *,
    source_path: Path,
    trusted_cache_roots: Sequence[Path],
    legacy_active_path: Path,
    legacy_recovery_descriptor_path: Path,
    legacy_recovery_records_path: Path,
    stage_parent: Path,
    verified_stage: Path,
    dataset_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Require independent and same-root byte identity before materializing a candidate."""

    stage_parent.mkdir(parents=True, exist_ok=True)
    first = Path(tempfile.mkdtemp(prefix="candidate-a-", dir=stage_parent))
    second = Path(tempfile.mkdtemp(prefix="candidate-b-", dir=stage_parent))
    try:
        arguments = {
            "source_path": source_path,
            "trusted_cache_roots": trusted_cache_roots,
            "legacy_active_path": legacy_active_path,
            "legacy_recovery_descriptor_path": legacy_recovery_descriptor_path,
            "legacy_recovery_records_path": legacy_recovery_records_path,
            "dataset_loader": dataset_loader,
        }
        first_report = build_canonical_pixel_rehash_candidate(**arguments, staging_root=first)
        second_report = build_canonical_pixel_rehash_candidate(**arguments, staging_root=second)
        first_bytes = _candidate_tree_bytes(first)
        if first_report != second_report or first_bytes != _candidate_tree_bytes(second):
            raise ValueError("independent canonical rehash candidates are not byte-identical")
        retry_report = build_canonical_pixel_rehash_candidate(**arguments, staging_root=first)
        if retry_report != first_report or _candidate_tree_bytes(first) != first_bytes:
            raise ValueError("same-root canonical rehash retry is not byte-idempotent")
        if verified_stage.exists():
            if _candidate_tree_bytes(verified_stage) != first_bytes:
                raise ValueError("verified stage already exists with different bytes")
        else:
            shutil.copytree(first, verified_stage)
        return first_report
    finally:
        shutil.rmtree(first)
        shutil.rmtree(second)


def _load_canonical_yaml_nofollow(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[dict[str, object], bytes]:
    payload = _read_regular_nofollow(path, label=label, maximum_bytes=maximum_bytes)
    try:
        loaded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise _publication_error(f"{label} is not canonical YAML") from error
    if not isinstance(loaded, dict) or _canonical_descriptor(loaded) != payload:
        raise _publication_error(f"{label} is not canonical YAML")
    return loaded, payload


def _declared_project_root(path: Path, declared: str) -> Path:
    declared_parts = PurePosixPath(declared).parts
    absolute = path.expanduser().absolute()
    if not declared_parts or tuple(absolute.parts[-len(declared_parts) :]) != declared_parts:
        raise _publication_error("manifest active path disagrees with candidate declaration")
    root_parts = absolute.parts[: -len(declared_parts)]
    return Path(*root_parts) if root_parts else Path(absolute.anchor)


def _install_immutable_directory(
    *,
    destination: Path,
    files: Mapping[str, bytes],
    boundary: str,
    boundary_hook: Callable[[str], None] | None,
) -> None:
    if not destination.name or any(
        _SAFE_METADATA_PATTERN.fullmatch(name) is None for name in files
    ):
        raise _publication_error("immutable install names are invalid")
    temporary = f".tmp-{destination.name}-{uuid.uuid4().hex}"

    def verify_existing(parent_fd: int) -> None:
        installed_fd = os.open(
            destination.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            for name, expected in files.items():
                if (
                    _read_regular_at(installed_fd, name, label="immutable installed file")
                    != expected
                ):
                    raise _publication_error("existing immutable directory is not byte-identical")
        finally:
            os.close(installed_fd)

    def remove_temporary(parent_fd: int) -> None:
        try:
            temporary_fd = os.open(
                temporary,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        try:
            for name in files:
                with suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=temporary_fd)
        finally:
            os.close(temporary_fd)
        with suppress(FileNotFoundError):
            os.rmdir(temporary, dir_fd=parent_fd)

    with _retained_directory_fd(
        destination.parent, label="immutable install parent", create=True
    ) as parent_fd:
        try:
            installed_fd = os.open(
                destination.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            installed_fd = None
        if installed_fd is not None:
            os.close(installed_fd)
            verify_existing(parent_fd)
        else:
            try:
                os.mkdir(temporary, mode=0o700, dir_fd=parent_fd)
                temporary_fd = os.open(
                    temporary,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
                try:
                    for name, payload in files.items():
                        _write_fsynced_at(
                            temporary_fd,
                            name,
                            payload,
                            written_boundary="generation_records_written",
                            fsynced_boundary="generation_records_fsynced",
                            boundary_hook=None,
                        )
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                try:
                    os.rename(
                        temporary,
                        destination.name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                except OSError as error:
                    if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                    remove_temporary(parent_fd)
                    verify_existing(parent_fd)
                os.fsync(parent_fd)
            finally:
                remove_temporary(parent_fd)
        _install_boundary(boundary, boundary_hook)


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return "unavailable"
    return value if re.fullmatch(r"[0-9a-f-]{36}", value) else "unavailable"


def _write_lock_diagnostics(lock_fd: int, metadata: Mapping[str, object]) -> None:
    payload = _canonical_descriptor(metadata)
    os.ftruncate(lock_fd, 0)
    os.lseek(lock_fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(lock_fd, view)
        if written < 1:
            raise OSError("short lock diagnostic write")
        view = view[written:]
    os.fsync(lock_fd)


@contextmanager
def _permanent_install_lock(
    active_parent_fd: int,
    *,
    expected_sha256: str,
    candidate_sha256: str,
    boundary_hook: Callable[[str], None] | None,
) -> Iterable[int]:
    no_follow, _, close_on_exec = _secure_dirfd_support()
    try:
        os.stat(INSTALL_LOCK_BASENAME, dir_fd=active_parent_fd, follow_symlinks=False)
        newly_created = False
    except FileNotFoundError:
        newly_created = True
    lock_fd = os.open(
        INSTALL_LOCK_BASENAME,
        os.O_RDWR | os.O_CREAT | no_follow | close_on_exec,
        0o600,
        dir_fd=active_parent_fd,
    )
    locked = False
    released = False
    try:
        opened = os.fstat(lock_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _publication_error("permanent install lock is not a regular file")
        if newly_created:
            os.fsync(active_parent_fd)
        for attempt in range(200):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if attempt == 199:
                    raise _publication_error("timed out acquiring permanent install lock") from None
                time.sleep(0.01)
        named = os.stat(
            INSTALL_LOCK_BASENAME,
            dir_fd=active_parent_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise _publication_error("permanent install lock inode changed")
        diagnostics = {
            "schema_version": 1,
            "state": "held",
            "boot_id": _boot_id(),
            "pid": os.getpid(),
            "process_start": str(time.monotonic_ns()),
            "nonce": uuid.uuid4().hex,
            "expected_active_sha256": expected_sha256,
            "candidate_pointer_sha256": candidate_sha256,
        }
        _write_lock_diagnostics(lock_fd, diagnostics)
        _install_boundary("install_lock_acquired", boundary_hook)
        yield lock_fd
        _write_lock_diagnostics(lock_fd, {**diagnostics, "state": "released"})
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        released = True
        _install_boundary("install_lock_released", boundary_hook)
    finally:
        if locked and not released:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        _install_boundary("install_lock_closed", boundary_hook)


def install_candidate(
    *,
    stage_root: Path,
    generation_root: Path,
    active_path: Path,
    expected_active_sha256_from_stage: bool,
    boundary_hook: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Install one fully validated staged v2 tuple through permanent-flock CAS."""

    compressed_maximum, uncompressed_maximum = _manifest_recovery_limits(2)
    metadata, _ = _load_canonical_yaml_nofollow(
        stage_root / "install-metadata.yaml",
        label="candidate install metadata",
        maximum_bytes=131072,
    )
    pointer_relative = str(metadata.get("active_pointer_path", ""))
    if pointer_relative != "data/manifests/smb-evaluation-v1.yaml":
        raise _publication_error("candidate active pointer path is non-canonical")
    pointer, pointer_bytes = _load_canonical_yaml_nofollow(
        stage_root / pointer_relative,
        label="candidate active pointer",
        maximum_bytes=131072,
    )
    validate_instance("manifest-active", pointer, version=2)
    generation_id = str(pointer["generation_id"])
    generation_stage = stage_root / "artifacts/smb-manifests/generations" / generation_id
    descriptor_bytes = _read_regular_nofollow(
        generation_stage / MANIFEST_DESCRIPTOR_FILENAME,
        label="candidate generation descriptor",
        maximum_bytes=131072,
    )
    records_bytes = _read_regular_nofollow(
        generation_stage / MANIFEST_RECORDS_FILENAME,
        label="candidate generation records",
        maximum_bytes=uncompressed_maximum,
    )
    try:
        descriptor = yaml.safe_load(descriptor_bytes.decode("utf-8"))
        rows = [json.loads(line) for line in records_bytes.decode("utf-8").splitlines()]
    except (UnicodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise _publication_error("candidate generation is malformed") from error
    if not isinstance(descriptor, dict) or any(not isinstance(row, dict) for row in rows):
        raise _publication_error("candidate generation is malformed")
    completed, rebuilt_pointer, canonical_descriptor, canonical_records = (
        _validate_generation_inputs(descriptor, rows)
    )
    recovery_path = Path(str(pointer["recovery_descriptor_path"]))
    recovery_records_path = Path(str(pointer["recovery_records_path"]))
    recovery, recovery_bytes = _load_canonical_yaml_nofollow(
        stage_root / recovery_path,
        label="candidate recovery descriptor",
        maximum_bytes=131072,
    )
    recovery_records = _read_regular_nofollow(
        stage_root / recovery_records_path,
        label="candidate recovery records",
        maximum_bytes=compressed_maximum,
    )
    validate_instance("manifest-recovery", recovery, version=2)
    bundle_id = str(recovery["bundle_id"])
    expected_prefix = PurePosixPath("data/manifests/recovery/canonical-pixel-v2") / bundle_id
    if (
        recovery_bundle_id_v2(recovery) != bundle_id
        or recovery_metadata_sha256(recovery, version=2) != recovery["metadata_sha256"]
        or recovery_path.parent != expected_prefix
        or recovery_records_path.parent != expected_prefix
        or hashlib.sha256(recovery_bytes).hexdigest() != pointer["recovery_descriptor_sha256"]
        or hashlib.sha256(recovery_records).hexdigest() != pointer["recovery_records_sha256"]
        or canonical_descriptor != descriptor_bytes
        or canonical_records != records_bytes
    ):
        raise _publication_error("candidate tuple identity is inconsistent")
    if (
        metadata.get("generation_id") != generation_id
        or metadata.get("bundle_id") != bundle_id
        or metadata.get("recovery_descriptor_path") != recovery["recovery_descriptor_path"]
        or metadata.get("recovery_records_path") != recovery["recovery_records_path"]
    ):
        raise _publication_error("candidate install metadata disagrees with staged tuple")
    rebuilt_pointer.update(
        {
            "recovery_descriptor_path": recovery["recovery_descriptor_path"],
            "recovery_descriptor_sha256": hashlib.sha256(recovery_bytes).hexdigest(),
            "recovery_records_path": recovery["recovery_records_path"],
            "recovery_records_sha256": hashlib.sha256(recovery_records).hexdigest(),
        }
    )
    if _canonical_descriptor(rebuilt_pointer) != pointer_bytes:
        raise _publication_error("candidate pointer disagrees with staged tuple")
    project_root = _declared_project_root(active_path, pointer_relative)
    _install_immutable_directory(
        destination=generation_root / generation_id,
        files={
            MANIFEST_DESCRIPTOR_FILENAME: descriptor_bytes,
            MANIFEST_RECORDS_FILENAME: records_bytes,
        },
        boundary="candidate_generation_installed",
        boundary_hook=boundary_hook,
    )
    _install_immutable_directory(
        destination=project_root / expected_prefix,
        files={
            "manifest-recovery.yaml": recovery_bytes,
            RECOVERY_RECORDS_FILENAME: recovery_records,
        },
        boundary="recovery_bundle_installed",
        boundary_hook=boundary_hook,
    )
    expected_sha256 = str(metadata.get("expected_previous_active_sha256", ""))
    if not expected_active_sha256_from_stage or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise _publication_error("candidate expected active checksum is unavailable")
    candidate_sha256 = hashlib.sha256(pointer_bytes).hexdigest()
    with (
        _retained_directory_fd(
            active_path.parent, label="active pointer parent", create=True
        ) as active_parent_fd,
        _permanent_install_lock(
            active_parent_fd,
            expected_sha256=expected_sha256,
            candidate_sha256=candidate_sha256,
            boundary_hook=boundary_hook,
        ),
    ):
        current = _read_regular_at(active_parent_fd, active_path.name, label="active pointer")
        current_sha256 = hashlib.sha256(current).hexdigest()
        _install_boundary("install_cas_read", boundary_hook)
        if current_sha256 == candidate_sha256:
            outcome = "idempotent"
        elif current_sha256 != expected_sha256:
            raise _publication_error("active pointer compare-and-swap conflict")
        else:
            temporary = f".{active_path.name}.tmp-{uuid.uuid4().hex}"
            try:
                _write_fsynced_at(
                    active_parent_fd,
                    temporary,
                    pointer_bytes,
                    written_boundary="pointer_written",
                    fsynced_boundary="pointer_fsynced",
                    boundary_hook=None,
                )
                os.replace(
                    temporary,
                    active_path.name,
                    src_dir_fd=active_parent_fd,
                    dst_dir_fd=active_parent_fd,
                )
                _install_boundary("install_pointer_replaced", boundary_hook)
                os.fsync(active_parent_fd)
                _install_boundary("install_pointer_fsynced", boundary_hook)
                outcome = "installed"
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=active_parent_fd)
    return {
        "status": outcome,
        "generation_id": completed["generation_id"],
        "bundle_id": bundle_id,
        "candidate_pointer_sha256": candidate_sha256,
    }


def export_manifest_recovery(
    *,
    active_path: Path,
    generation_root: Path,
    recovery_descriptor_path: Path,
    recovery_records_path: Path,
) -> dict[str, object]:
    """Export the exact active compact generation as one bounded deterministic pair."""

    compressed_maximum, uncompressed_maximum = _manifest_recovery_limits()
    descriptor, rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    if descriptor.get("schema_version") != 2:
        raise _publication_error("manifest recovery requires the active compact v2 generation")
    records_bytes = _canonical_jsonl(rows)
    if len(records_bytes) > uncompressed_maximum:
        raise _publication_error("active records exceed the schema-declared uncompressed maximum")
    if hashlib.sha256(records_bytes).hexdigest() != descriptor["records_sha256"]:
        raise _publication_error("active records checksum changed during recovery export")
    _assert_safe_recovery_content(descriptor)
    for row in rows:
        _assert_safe_recovery_content(row)

    active_bytes = _read_regular_nofollow(
        active_path,
        label="active pointer",
        maximum_bytes=compressed_maximum,
    )
    try:
        active_pointer = yaml.safe_load(active_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise _publication_error("active pointer is not canonical YAML") from error
    if not isinstance(active_pointer, dict) or active_bytes != _canonical_descriptor(
        active_pointer
    ):
        raise _publication_error("active pointer is not canonical YAML")

    descriptor_bytes = _canonical_descriptor(descriptor)
    compressed_bytes = _deterministic_gzip(records_bytes)
    if len(compressed_bytes) > compressed_maximum:
        raise _publication_error("recovery gzip exceeds the schema-declared compressed maximum")
    recovery: dict[str, object] = {
        "schema_version": 1,
        "record_type": "manifest-recovery",
        "manifest_id": descriptor["manifest_id"],
        "generation_id": descriptor["generation_id"],
        "active_pointer_path": "data/manifests/smb-evaluation-v1.yaml",
        "active_pointer_sha256": hashlib.sha256(active_bytes).hexdigest(),
        "descriptor_path": active_pointer["descriptor_path"],
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "descriptor_yaml": descriptor_bytes.decode("utf-8"),
        "records_path": active_pointer["records_path"],
        "row_schema_id": descriptor["row_schema_id"],
        "row_schema_version": descriptor["row_schema_version"],
        "row_count": descriptor["row_count"],
        "records_sha256": descriptor["records_sha256"],
        "source_revision": descriptor["source_revision"],
        "source_provenance": descriptor["source_provenance"],
        "audit_version": descriptor["audit_version"],
        "benchmark_state": descriptor["benchmark_state"],
        "recovery_records_path": "data/manifests/smb-evaluation-v1-recovery.jsonl.gz",
        "compression": {
            "algorithm": "gzip",
            "format_version": 1,
            "compresslevel": 9,
            "mtime": 0,
            "filename": "",
        },
        "compressed_sha256": hashlib.sha256(compressed_bytes).hexdigest(),
        "compressed_size_bytes": len(compressed_bytes),
        "uncompressed_size_bytes": len(records_bytes),
        "recovery_command": RECOVERY_COMMAND,
        "metadata_sha256": "",
    }
    recovery["metadata_sha256"] = recovery_metadata_sha256(recovery)
    validate_instance("manifest-recovery", recovery, version=1)
    recovery_bytes = _canonical_descriptor(recovery)
    _durable_replace(recovery_records_path, compressed_bytes)
    _durable_replace(recovery_descriptor_path, recovery_bytes)
    return recovery


def _validated_recovery_mapping(
    recovery_descriptor_path: Path,
) -> tuple[dict[str, object], bytes, int]:
    maximum_bytes = max(_manifest_recovery_limits(version)[0] for version in (1, 2))
    descriptor_bytes = _read_regular_nofollow(
        recovery_descriptor_path,
        label="recovery descriptor",
        maximum_bytes=maximum_bytes,
    )
    try:
        recovery = yaml.safe_load(descriptor_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise _publication_error("recovery descriptor is not valid YAML") from error
    if not isinstance(recovery, dict):
        raise _publication_error("recovery descriptor must be a mapping")
    declared_version = recovery.get("schema_version")
    if isinstance(declared_version, bool) or declared_version not in {1, 2}:
        raise _publication_error("recovery descriptor names an unsupported schema version")
    version = int(declared_version)
    try:
        validate_instance("manifest-recovery", recovery, version=version)
    except ContractValidationError as error:
        raise _publication_error(f"recovery descriptor failed validation: {error}") from error
    if descriptor_bytes != _canonical_descriptor(recovery):
        raise _publication_error("recovery descriptor is not canonical YAML")
    return recovery, descriptor_bytes, version


def _stream_recovery_records(
    compressed_bytes: bytes,
    *,
    declared_size: int,
    hard_maximum: int,
) -> bytes:
    output = bytearray()
    compressed_stream = io.BytesIO(compressed_bytes)
    try:
        with gzip.GzipFile(fileobj=compressed_stream, mode="rb") as handle:
            while True:
                remaining_declared = declared_size - len(output)
                remaining_hard = hard_maximum - len(output)
                request_size = min(
                    RECOVERY_READ_CHUNK_SIZE,
                    remaining_declared + 1,
                    remaining_hard + 1,
                )
                chunk = handle.read(max(1, request_size))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > hard_maximum:
                    raise _publication_error(
                        "recovery records exceed the schema-declared uncompressed maximum"
                    )
                if len(output) > declared_size:
                    if declared_size == hard_maximum:
                        raise _publication_error(
                            "recovery records exceed the schema-declared uncompressed maximum"
                        )
                    raise _publication_error("recovery records exceed their declared size")
    except ManifestPublicationError:
        raise
    except (EOFError, gzip.BadGzipFile, OSError) as error:
        raise _publication_error("recovery gzip is malformed or truncated") from error
    if len(output) != declared_size:
        raise _publication_error("recovery records do not match their declared size")
    return bytes(output)


def recover_active_manifest(
    *,
    active_path: Path,
    recovery_descriptor_path: Path,
    recovery_records_path: Path,
    generation_root: Path,
) -> dict[str, object]:
    """Verify and durably restore exactly the generation named by the active pointer."""

    recovery, recovery_descriptor_bytes, recovery_version = _validated_recovery_mapping(
        recovery_descriptor_path
    )
    compressed_maximum, uncompressed_maximum = _manifest_recovery_limits(recovery_version)
    active_bytes = _read_regular_nofollow(
        active_path,
        label="active pointer",
        maximum_bytes=compressed_maximum,
    )
    if (
        recovery_version == 1
        and hashlib.sha256(active_bytes).hexdigest() != recovery["active_pointer_sha256"]
    ):
        raise _publication_error("active pointer checksum does not match recovery metadata")

    compressed_bytes = _read_regular_nofollow(
        recovery_records_path,
        label="compressed recovery records",
        maximum_bytes=compressed_maximum,
    )
    if len(compressed_bytes) != recovery["compressed_size_bytes"]:
        raise _publication_error("compressed recovery size does not match metadata")
    if hashlib.sha256(compressed_bytes).hexdigest() != recovery["compressed_sha256"]:
        raise _publication_error("compressed checksum does not match recovery metadata")
    records_bytes = _stream_recovery_records(
        compressed_bytes,
        declared_size=int(recovery["uncompressed_size_bytes"]),
        hard_maximum=uncompressed_maximum,
    )
    if hashlib.sha256(records_bytes).hexdigest() != recovery["records_sha256"]:
        raise _publication_error("uncompressed recovery records checksum mismatch")

    rows: list[dict[str, object]] = []
    try:
        text = records_bytes.decode("utf-8")
        if not text.endswith("\n"):
            raise ValueError("records JSONL lacks final newline")
        for index, line in enumerate(text.splitlines()):
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"row {index} is not an object")
            validate_instance("manifest-row", loaded, version=2)
            rows.append(loaded)
    except (UnicodeError, json.JSONDecodeError, ContractValidationError, ValueError) as error:
        raise _publication_error(f"recovery records failed validation: {error}") from error
    if len(rows) != recovery["row_count"] or _canonical_jsonl(rows) != records_bytes:
        raise _publication_error("recovery records are non-canonical or have the wrong row count")

    try:
        descriptor = yaml.safe_load(str(recovery["descriptor_yaml"]))
    except yaml.YAMLError as error:  # pragma: no cover - shared contract already proves this
        raise _publication_error("embedded recovery descriptor is invalid") from error
    if not isinstance(descriptor, dict):  # pragma: no cover - shared contract already proves this
        raise _publication_error("embedded recovery descriptor is invalid")
    completed, pointer, descriptor_bytes, validated_records = _validate_generation_inputs(
        descriptor,
        rows,
        allow_legacy_hash_provenance=recovery_version == 1,
    )
    if recovery_version == 2:
        if not recovery_descriptor_path.absolute().as_posix().endswith(
            str(recovery["recovery_descriptor_path"])
        ) or not recovery_records_path.absolute().as_posix().endswith(
            str(recovery["recovery_records_path"])
        ):
            raise _publication_error("supplied recovery files disagree with bundle paths")
        pointer.update(
            {
                "recovery_descriptor_path": recovery["recovery_descriptor_path"],
                "recovery_descriptor_sha256": hashlib.sha256(recovery_descriptor_bytes).hexdigest(),
                "recovery_records_path": recovery["recovery_records_path"],
                "recovery_records_sha256": recovery["compressed_sha256"],
            }
        )
        validate_instance("manifest-active", pointer, version=2)
    expected_pointer_bytes = _canonical_descriptor(pointer)
    identity_mismatch = (
        completed["generation_id"] != recovery["generation_id"]
        or hashlib.sha256(descriptor_bytes).hexdigest() != recovery["descriptor_sha256"]
        or validated_records != records_bytes
        or active_bytes != expected_pointer_bytes
    )
    if recovery_version == 1:
        identity_mismatch = (
            identity_mismatch
            or hashlib.sha256(expected_pointer_bytes).hexdigest()
            != recovery["active_pointer_sha256"]
        )
    if identity_mismatch:
        raise _publication_error("reconstructed recovery identity does not match metadata")

    root = _reject_symlink_components(
        generation_root, label="generation root", final_may_not_exist=True
    )
    if root.exists() and not root.is_dir():
        raise _publication_error("generation root must be a directory")
    publish_manifest_generation(
        active_path=active_path,
        generation_root=root,
        descriptor=completed,
        rows=rows,
        allow_legacy_hash_provenance=recovery_version == 1,
        recovery_binding=(
            {
                "recovery_descriptor_path": recovery["recovery_descriptor_path"],
                "recovery_descriptor_sha256": hashlib.sha256(recovery_descriptor_bytes).hexdigest(),
                "recovery_records_path": recovery["recovery_records_path"],
                "recovery_records_sha256": recovery["compressed_sha256"],
            }
            if recovery_version == 2
            else None
        ),
    )
    resolved_descriptor, resolved_rows = resolve_active_manifest(
        active_path=active_path, generation_root=root
    )
    return {
        "generation_id": resolved_descriptor["generation_id"],
        "row_count": len(resolved_rows),
        "records_sha256": resolved_descriptor["records_sha256"],
        "row_schema_id": resolved_descriptor["row_schema_id"],
        "row_schema_version": resolved_descriptor["row_schema_version"],
        "source_revision": resolved_descriptor["source_revision"],
        "source_provenance": resolved_descriptor["source_provenance"],
        "benchmark_state": resolved_descriptor["benchmark_state"],
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


def prepare_v2_review_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Emit disjoint policy, frozen-visual, and perceptual-pair evidence rows."""

    item_rows: list[dict[str, str]] = []
    visual_rows: list[dict[str, str]] = []
    pair_relations: dict[str, Mapping[str, object]] = {}
    for row in rows:
        validate_instance("manifest-row", row, version=2)
        item_id = str(row["item_id"])
        source_group_id = row.get("source_group_id")
        source_identity = row["source_identity"]
        assert isinstance(source_identity, Mapping)
        try:
            expected_group = _canonical_source_group_id(
                source_identity.get("original_score_normalized")
            )
        except ValueError as error:
            raise _review_error(f"{item_id}: invalid audited source group") from error
        if source_group_id != expected_group:
            raise _review_error(f"{item_id}: source_group_id disagrees with audited identity")
        policy = {
            **{field: "" for field in REVIEW_CSV_FIELDS},
            "review_kind": "item_policy",
            "review_key": f"policy:{item_id}",
            "item_id": item_id,
            "source_group_id": str(source_group_id or ""),
            "review_status": "pending",
            "dataset_licence_status": "pending",
            "item_provenance_status": "pending",
            "access_status": "pending",
            "redistribution_status": "pending",
            "figure_reproduction_status": "pending",
        }
        item_rows.append(policy)

        if row["audit_sample_member"] is True:
            visual = row["visual_review"]
            assert isinstance(visual, Mapping)
            reviewed = visual.get("status") == "sampled_human_reviewed"
            visual_rows.append(
                {
                    **{field: "" for field in REVIEW_CSV_FIELDS},
                    "review_kind": "visual_item",
                    "review_key": f"visual:{item_id}",
                    "item_id": item_id,
                    "review_status": "reviewed" if reviewed else "pending",
                    "reviewer": str(visual.get("reviewer", "")) if reviewed else "",
                    "reviewed_at": str(visual.get("reviewed_at", "")) if reviewed else "",
                    "rationale": str(visual.get("rationale", "")) if reviewed else "",
                    "quality_disposition": (
                        ";".join(str(flag) for flag in visual.get("quality_flags", ()))
                        or "acceptable"
                        if reviewed
                        else ""
                    ),
                    "suitability_disposition": (
                        str(visual.get("suitability", "")) if reviewed else ""
                    ),
                }
            )
        relations = row["duplicate_relations"]
        assert isinstance(relations, Sequence)
        for relation in relations:
            assert isinstance(relation, Mapping)
            if relation.get("candidate_type") == "perceptual":
                pair_relations[str(relation["pair_id"])] = relation

    if len(item_rows) != EXPECTED_ROW_COUNT or len(visual_rows) != DEFAULT_SAMPLE_SIZE:
        raise _review_error("v2 review preparation requires 685 policies and 64 sample rows")
    pair_rows: list[dict[str, str]] = []
    for pair_id, relation in sorted(pair_relations.items()):
        item_ids = relation["item_ids"]
        assert isinstance(item_ids, Sequence)
        reviewed = relation["disposition"] in {"distinct", "duplicate", "related"}
        unavailable = relation["disposition"] == "unavailable"
        pair_rows.append(
            {
                **{field: "" for field in REVIEW_CSV_FIELDS},
                "review_kind": "duplicate_pair",
                "review_key": pair_id,
                "item_id": str(item_ids[0]),
                "candidate_item_id": str(item_ids[1]),
                "review_status": (
                    "unavailable" if unavailable else "reviewed" if reviewed else "pending"
                ),
                "reviewer": str(relation.get("reviewer") or "") if reviewed else "",
                "reviewed_at": str(relation.get("reviewed_at") or "") if reviewed else "",
                "rationale": str(relation.get("rationale") or ""),
                "duplicate_disposition": str(relation["disposition"]),
            }
        )
    return [*item_rows, *visual_rows, *pair_rows]


def _expected_review_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    if rows and rows[0].get("schema_version") == 2:
        return prepare_v2_review_rows(rows)
    return [*(_review_row(row) for row in rows), *_candidate_review_rows(rows)]


def _write_review_rows(path: Path, review_rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_review_csv(review_rows))


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
    implementation_provenance: Mapping[str, object] | None = None,
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
    if implementation_provenance is not None:
        audit_descriptor["implementation_provenance"] = dict(implementation_provenance)
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
        return list(read_review(review_path).rows)
    except ReviewEvidenceError as error:
        raise _review_error(str(error)) from error


def _validated_review_rows(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    try:
        review_rows = validate_review_rows(review_rows)
    except ReviewEvidenceError as error:
        raise _review_error(str(error)) from error
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
            if row["duplicate_disposition"] not in HUMAN_PAIR_DISPOSITIONS:
                raise _review_error(f"{key}: invalid duplicate_disposition")
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
            review_quality_flags(row["quality_disposition"], review_key=key)
            shared_domains = (
                ("suitability_disposition", LEGACY_SUITABILITY_DISPOSITIONS),
                ("duplicate_disposition", HUMAN_PAIR_DISPOSITIONS),
                ("dataset_licence_status", DATASET_LICENCE_STATUSES),
                ("item_provenance_status", ITEM_PROVENANCE_STATUSES),
                ("access_status", ACCESS_STATUSES),
                ("redistribution_status", LEGACY_REUSE_STATUSES),
                ("figure_reproduction_status", LEGACY_REUSE_STATUSES),
            )
            for field, allowed in shared_domains:
                if row[field] not in allowed:
                    raise _review_error(f"{key}: invalid {field}")
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


def _validated_v2_review_rows(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    try:
        review_rows = validate_review_rows(review_rows)
    except ReviewEvidenceError as error:
        raise _review_error(str(error)) from error
    expected_rows = prepare_v2_review_rows(rows)
    expected = {row["review_key"]: row for row in expected_rows}
    seen: dict[str, dict[str, str]] = {}
    for untrusted in review_rows:
        if set(untrusted) != set(REVIEW_CSV_FIELDS):
            raise _review_error("review row does not match the exact header")
        row = {field: str(untrusted[field]) for field in REVIEW_CSV_FIELDS}
        key = row["review_key"]
        if key in seen:
            raise _review_error(f"duplicate review key: {key}")
        emitted = expected.get(key)
        if emitted is None:
            raise _review_error(f"unknown review key: {key}")
        for field in ("review_kind", "item_id", "candidate_item_id"):
            if row[field] != emitted[field]:
                raise _review_error(f"{key}: {field} does not match the emitted key")
        unavailable_pair = (
            row["review_kind"] == "duplicate_pair" and row["review_status"] == "unavailable"
        )
        if unavailable_pair:
            if row["duplicate_disposition"] != "unavailable":
                raise _review_error(f"{key}: unavailable pair must use unavailable disposition")
            if row["reviewer"] or row["reviewed_at"]:
                raise _review_error(f"{key}: unavailable pair cannot have reviewer or reviewed_at")
            if not row["rationale"].strip():
                raise _review_error(f"{key}: unavailable pair rationale is required")
            if row["rationale"] != emitted["rationale"]:
                raise _review_error(f"{key}: unavailable pair rationale changed")
        else:
            if row["review_status"] == "unavailable":
                raise _review_error(
                    f"{key}: unavailable review_status is allowed only for duplicate_pair rows"
                )
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

        if row["review_kind"] == "item_policy":
            if row["source_group_id"] != emitted["source_group_id"]:
                raise _review_error(f"{key}: source_group_id does not match emitted audit evidence")
            shared_domains = (
                ("dataset_licence_status", frozenset({"confirmed"})),
                ("item_provenance_status", ITEM_PROVENANCE_STATUSES),
                ("access_status", ACCESS_STATUSES),
                ("redistribution_status", V2_REUSE_STATUSES),
                ("figure_reproduction_status", V2_REUSE_STATUSES),
            )
            for field, allowed in shared_domains:
                if row[field] not in allowed:
                    raise _review_error(f"{key}: invalid {field}")
            if row["item_provenance_status"] == "unavailable" and (
                row["redistribution_status"] == "permitted"
                or row["figure_reproduction_status"] == "permitted"
            ):
                raise _review_error(f"{key}: unavailable item provenance cannot infer permission")
            if any(
                row[field]
                for field in (
                    "quality_disposition",
                    "suitability_disposition",
                    "duplicate_disposition",
                )
            ):
                raise _review_error(f"{key}: policy row contains visual or pair claims")
        elif row["review_kind"] == "visual_item":
            review_quality_flags(row["quality_disposition"], review_key=key)
            if row["suitability_disposition"] not in VISUAL_SUITABILITY_DISPOSITIONS:
                raise _review_error(f"{key}: invalid suitability_disposition")
            irrelevant = (
                "source_group_id",
                "duplicate_disposition",
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            )
            if any(row[field] for field in irrelevant):
                raise _review_error(f"{key}: visual row contains policy or pair claims")
        elif row["review_kind"] == "duplicate_pair":
            if unavailable_pair:
                assert row["duplicate_disposition"] == "unavailable"
            else:
                if row["duplicate_disposition"] == "unavailable":
                    raise _review_error(
                        f"{key}: human-reviewed pair cannot use unavailable disposition"
                    )
                if row["duplicate_disposition"] not in HUMAN_PAIR_DISPOSITIONS:
                    raise _review_error(f"{key}: invalid duplicate_disposition")
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
                raise _review_error(f"{key}: pair row contains item-only claims")
        else:  # pragma: no cover - expected-row join already prevents this
            raise _review_error(f"{key}: unsupported review kind")
        seen[key] = row
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise _review_error(f"missing review key: {missing[0]}")
    return [seen[row["review_key"]] for row in expected_rows]


def _apply_v2_review_dispositions(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    validated = _validated_v2_review_rows(rows, review_rows)
    updated = copy.deepcopy(list(rows))
    by_id = {str(row["item_id"]): row for row in updated}
    for review in validated:
        item = by_id[review["item_id"]]
        if review["review_kind"] == "item_policy":
            item_provenance = (
                {
                    "status": "confirmed",
                    "evidence_ref": f"review:{review['review_key']}",
                }
                if review["item_provenance_status"] == "confirmed"
                else {
                    "status": "unavailable",
                    "reason": "The dataset does not expose a per-item source chain.",
                }
            )
            item["rights"] = {
                "dataset_licence": {
                    "status": "confirmed",
                    "identifier": "CC-BY-NC-4.0",
                    "reference": "https://creativecommons.org/licenses/by-nc/4.0/",
                },
                "item_provenance": item_provenance,
                "access_status": review["access_status"],
                "redistribution": {
                    "status": review["redistribution_status"],
                    "reviewed_basis_ref": (
                        f"review:{review['review_key']}"
                        if review["redistribution_status"] == "permitted"
                        else None
                    ),
                },
                "figure_reproduction": {
                    "status": review["figure_reproduction_status"],
                    "reviewed_basis_ref": (
                        f"review:{review['review_key']}"
                        if review["figure_reproduction_status"] == "permitted"
                        else None
                    ),
                },
            }
        elif review["review_kind"] == "visual_item":
            item["visual_review"] = {
                "status": "sampled_human_reviewed",
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "rationale": review["rationale"],
                "quality_flags": list(
                    review_quality_flags(
                        review["quality_disposition"], review_key=review["review_key"]
                    )
                ),
                "suitability": review["suitability_disposition"],
            }
        else:
            for member_id in (review["item_id"], review["candidate_item_id"]):
                relations = by_id[member_id]["duplicate_relations"]
                assert isinstance(relations, list)
                relation = next(
                    relation
                    for relation in relations
                    if relation["pair_id"] == review["review_key"]
                )
                disposition = review["duplicate_disposition"]
                relation.update(
                    {
                        "evidence_basis": (
                            "perceptual_hash_plus_human_review"
                            if disposition != "unavailable"
                            else "perceptual_hash_candidate"
                        ),
                        "disposition": disposition,
                        "reviewer": review["reviewer"] if disposition != "unavailable" else None,
                        "reviewed_at": (
                            review["reviewed_at"] if disposition != "unavailable" else None
                        ),
                        "rationale": review["rationale"],
                    }
                )
    return derive_v2_exact_relations(updated)


def apply_review_dispositions(
    rows: Sequence[Mapping[str, object]], review_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    """Apply one complete stable-key review to copied manifest rows."""

    if rows and rows[0].get("schema_version") == 2:
        return _apply_v2_review_dispositions(rows, review_rows)
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
                "flags": list(
                    review_quality_flags(
                        review["quality_disposition"], review_key=review["review_key"]
                    )
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


def _v2_review_inference(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    visual_counts: Counter[str] = Counter()
    automated_count = 0
    pairs: dict[str, Mapping[str, object]] = {}
    for row in rows:
        visual = row["visual_review"]
        automated = row["automated_audit"]
        assert isinstance(visual, Mapping) and isinstance(automated, Mapping)
        visual_counts[str(visual["status"])] += 1
        automated_count += automated["status"] == "automated"
        relations = row["duplicate_relations"]
        assert isinstance(relations, Sequence)
        for relation in relations:
            assert isinstance(relation, Mapping)
            pairs[str(relation["pair_id"])] = relation
    exact = [relation for relation in pairs.values() if relation["candidate_type"] == "exact"]
    perceptual = [
        relation for relation in pairs.values() if relation["candidate_type"] == "perceptual"
    ]
    return {
        "automated_population_audit_count": automated_count,
        "sampled_human_review_count": visual_counts["sampled_human_reviewed"],
        "targeted_human_review_count": visual_counts["targeted_human_reviewed"],
        "not_visually_reviewed_count": visual_counts["not_visually_reviewed"],
        "unavailable_visual_review_count": visual_counts["unavailable"],
        "not_applicable_visual_review_count": visual_counts["not_applicable"],
        "exact_pair_automated_count": len(exact),
        "perceptual_pair_count": len(perceptual),
        "perceptual_pair_human_review_count": sum(
            relation["disposition"] in {"distinct", "duplicate", "related"}
            for relation in perceptual
        ),
        "perceptual_pair_pending_count": sum(
            relation["disposition"] == "pending" for relation in perceptual
        ),
        "inference_scope": "sample_observation_only",
        "population_prevalence_inference": "not_supported",
    }


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
    if descriptor.get("schema_version") == 2:
        descriptor = {**descriptor, "review_inference": _v2_review_inference(updated)}
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
    export_recovery = commands.add_parser("export-recovery")
    export_recovery.add_argument("--manifest-active", type=Path, required=True)
    export_recovery.add_argument("--manifest-generation-root", type=Path, required=True)
    export_recovery.add_argument("--recovery-descriptor", type=Path, required=True)
    export_recovery.add_argument("--recovery-records", type=Path, required=True)
    recover_active = commands.add_parser("recover-active")
    recover_active.add_argument("--manifest-active", type=Path, required=True)
    recover_active.add_argument("--recovery-descriptor", type=Path, required=True)
    recover_active.add_argument("--recovery-records", type=Path, required=True)
    recover_active.add_argument("--manifest-generation-root", type=Path, required=True)
    candidate = commands.add_parser("build-canonical-pixel-rehash-candidate")
    candidate.add_argument("--source", type=Path, required=True)
    candidate.add_argument("--trusted-cache-root", type=Path, action="append", required=True)
    candidate.add_argument("--legacy-manifest-active", type=Path, required=True)
    candidate.add_argument("--legacy-recovery-descriptor", type=Path, required=True)
    candidate.add_argument("--legacy-recovery-records", type=Path, required=True)
    candidate.add_argument("--staging-root", type=Path, required=True)
    determinism = commands.add_parser("verify-authoritative-determinism")
    determinism.add_argument("--source", type=Path, required=True)
    determinism.add_argument("--trusted-cache-root", type=Path, action="append", required=True)
    determinism.add_argument("--legacy-manifest-active", type=Path, required=True)
    determinism.add_argument("--legacy-recovery-descriptor", type=Path, required=True)
    determinism.add_argument("--legacy-recovery-records", type=Path, required=True)
    determinism.add_argument("--stage-parent", type=Path, required=True)
    determinism.add_argument("--verified-stage", type=Path, required=True)
    install = commands.add_parser("install-candidate")
    install.add_argument("--stage-root", type=Path, required=True)
    install.add_argument("--manifest-generation-root", type=Path, required=True)
    install.add_argument("--manifest-active", type=Path, required=True)
    install.add_argument("--expected-active-sha256-from-stage", action="store_true", required=True)
    migrate = commands.add_parser("migrate-authoritative")
    migration_commands = migrate.add_subparsers(dest="migration_command", required=True)
    migrate_audit = migration_commands.add_parser("audit")
    migrate_audit.add_argument("--source", type=Path, required=True)
    migrate_audit.add_argument("--sample", type=Path, required=True)
    migrate_audit.add_argument("--manifest-active", type=Path, required=True)
    migrate_audit.add_argument("--manifest-generation-root", type=Path, required=True)
    migrate_decisions = migration_commands.add_parser("decisions")
    migrate_decisions.add_argument("--legacy-review", type=Path, required=True)
    migrate_decisions.add_argument("--sample", type=Path, required=True)
    migrate_decisions.add_argument("--manifest-active", type=Path, required=True)
    migrate_decisions.add_argument("--manifest-generation-root", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--audit-descriptor", type=Path, required=True)
    audit.add_argument("--audit-records", type=Path, required=True)
    audit.add_argument("--sample", type=Path, required=True)
    audit.add_argument("--review", type=Path, required=True)
    audit.add_argument("--manifest-active", type=Path, required=True)
    audit.add_argument("--manifest-generation-root", type=Path, required=True)
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
    if arguments.command == "export-recovery":
        recovery = export_manifest_recovery(
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
            recovery_descriptor_path=arguments.recovery_descriptor,
            recovery_records_path=arguments.recovery_records,
        )
        print(
            json.dumps(
                {
                    "generation_id": recovery["generation_id"],
                    "row_count": recovery["row_count"],
                    "records_sha256": recovery["records_sha256"],
                    "compressed_size_bytes": recovery["compressed_size_bytes"],
                    "uncompressed_size_bytes": recovery["uncompressed_size_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "recover-active":
        report = recover_active_manifest(
            active_path=arguments.manifest_active,
            recovery_descriptor_path=arguments.recovery_descriptor,
            recovery_records_path=arguments.recovery_records,
            generation_root=arguments.manifest_generation_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "build-canonical-pixel-rehash-candidate":
        report = build_canonical_pixel_rehash_candidate(
            source_path=arguments.source,
            trusted_cache_roots=arguments.trusted_cache_root,
            legacy_active_path=arguments.legacy_manifest_active,
            legacy_recovery_descriptor_path=arguments.legacy_recovery_descriptor,
            legacy_recovery_records_path=arguments.legacy_recovery_records,
            staging_root=arguments.staging_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "verify-authoritative-determinism":
        report = verify_authoritative_determinism(
            source_path=arguments.source,
            trusted_cache_roots=arguments.trusted_cache_root,
            legacy_active_path=arguments.legacy_manifest_active,
            legacy_recovery_descriptor_path=arguments.legacy_recovery_descriptor,
            legacy_recovery_records_path=arguments.legacy_recovery_records,
            stage_parent=arguments.stage_parent,
            verified_stage=arguments.verified_stage,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "install-candidate":
        report = install_candidate(
            stage_root=arguments.stage_root,
            generation_root=arguments.manifest_generation_root,
            active_path=arguments.manifest_active,
            expected_active_sha256_from_stage=arguments.expected_active_sha256_from_stage,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "migrate-authoritative":
        if arguments.migration_command == "audit":
            report = migrate_authoritative_audit(
                source_path=arguments.source,
                sample_path=arguments.sample,
                active_path=arguments.manifest_active,
                generation_root=arguments.manifest_generation_root,
            )
        else:
            report = migrate_authoritative_decisions(
                legacy_review_path=arguments.legacy_review,
                sample_path=arguments.sample,
                active_path=arguments.manifest_active,
                generation_root=arguments.manifest_generation_root,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.command == "audit":
        report = run_authenticated_audit(
            source_path=arguments.source,
            audit_descriptor_path=arguments.audit_descriptor,
            audit_records_path=arguments.audit_records,
            sample_path=arguments.sample,
            review_path=arguments.review,
            active_path=arguments.manifest_active,
            generation_root=arguments.manifest_generation_root,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
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
