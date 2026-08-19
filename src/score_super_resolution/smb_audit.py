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
import stat
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath

import imagehash
import yaml
from PIL import Image, UnidentifiedImageError

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    assert_smb_purpose_allowed,
)
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.review_evidence import (
    REVIEW_FIELDS as REVIEW_CSV_FIELDS,
)
from score_super_resolution.review_evidence import (
    ReviewEvidenceError,
    canonical_review_csv,
    read_review,
    validate_review_rows,
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
GENERATION_DOMAIN = b"smb-manifest-generation-v1\0"
AUDIT_SOURCE_SET_VERSION = 1
AUDIT_SOURCE_TREE_DOMAIN = b"smb-audit-source-tree-v1\0"
AUDIT_PATCH_DOMAIN = b"smb-audit-patch-state-v1\0"
AUDIT_LOCK_DOMAIN = b"smb-audit-uv-lock-v1\0"
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


def _audit_after_guard(
    record: Mapping[str, object],
    *,
    upstream_index: int,
    source_revision: str,
    trusted_cache_roots: Sequence[Path],
    max_encoded_bytes: int,
    max_pixels: int,
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
