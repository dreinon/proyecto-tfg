"""Canonical, fail-closed I/O primitives for tracked human-review evidence."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REVIEW_FIELDS = (
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

LEGACY_REVIEW_KINDS = frozenset({"item", "candidate"})
V2_REVIEW_KINDS = frozenset({"item_policy", "visual_item", "duplicate_pair"})
REVIEW_KINDS = LEGACY_REVIEW_KINDS | V2_REVIEW_KINDS
REVIEW_STATUSES = frozenset({"pending", "reviewed", "unavailable"})
QUALITY_FLAGS = frozenset({"blurred", "low_contrast", "oversized", "skewed", "unprocessable"})
LEGACY_SUITABILITY_DISPOSITIONS = frozenset({"suitable", "unsuitable", "unavailable"})
VISUAL_SUITABILITY_DISPOSITIONS = frozenset({"suitable", "unsuitable", "uncertain", "not_assessed"})
HUMAN_PAIR_DISPOSITIONS = frozenset({"distinct", "duplicate", "related"})
DATASET_LICENCE_STATUSES = frozenset({"confirmed", "restricted"})
ITEM_PROVENANCE_STATUSES = frozenset({"confirmed", "unavailable"})
ACCESS_STATUSES = frozenset({"confirmed", "restricted"})
LEGACY_REUSE_STATUSES = frozenset({"permitted", "prohibited"})
V2_REUSE_STATUSES = frozenset({"not_established", "permitted", "prohibited"})

_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SAFE_REVIEW_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVIEW_SAVE_BOUNDARIES = (
    "review_written",
    "review_fsynced",
    "review_replaced",
    "review_parent_fsynced",
)


class ReviewEvidenceError(ValueError):
    """Report unsafe or structurally invalid review evidence without reflecting payloads."""


class ReviewPersistenceError(RuntimeError):
    """Report whether a failed durable save already committed replacement bytes."""

    def __init__(self, message: str, *, committed: bool, sha256: str | None = None) -> None:
        self.committed = committed
        self.sha256 = sha256
        super().__init__(message)


class StaleReviewError(ReviewPersistenceError):
    """Require an explicit reload when the review changed since the session read it."""

    def __init__(self) -> None:
        super().__init__(
            "stale review session; reload required before saving",
            committed=False,
        )


@dataclass(frozen=True)
class ReviewDocument:
    """One validated review snapshot and its exact on-disk identity."""

    rows: tuple[dict[str, str], ...]
    sha256: str
    canonical_bytes: bytes


def _safe_key(value: object, *, row_index: int | None = None) -> str:
    if isinstance(value, str) and _SAFE_REVIEW_KEY.fullmatch(value) is not None:
        return value
    return f"row-{row_index}" if row_index is not None else "unknown-review"


def _cell_error(*, field: str, review_key: str, reason: str) -> ReviewEvidenceError:
    return ReviewEvidenceError(f"{review_key}: {field} {reason}")


def validate_human_cell(value: str, *, field: str, review_key: str) -> str:
    """Reject spreadsheet formulas and control characters in canonical evidence."""

    if not isinstance(value, str):
        raise _cell_error(field=field, review_key=review_key, reason="must be text")
    if any(ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value):
        raise _cell_error(field=field, review_key=review_key, reason="contains a control character")
    if value.lstrip().startswith(_FORMULA_PREFIXES):
        raise _cell_error(field=field, review_key=review_key, reason="has a formula prefix")
    return value


def _require_domain_value(row: Mapping[str, str], field: str, allowed: frozenset[str]) -> str:
    value = row[field]
    if value not in allowed:
        raise _cell_error(field=field, review_key=row["review_key"], reason="is invalid")
    return value


def _require_empty_domains(row: Mapping[str, str], fields: Sequence[str]) -> None:
    for field in fields:
        if row[field]:
            raise _cell_error(
                field=field,
                review_key=row["review_key"],
                reason="does not apply to this review kind",
            )


def validate_quality_flags(flags: Sequence[str], *, review_key: str) -> tuple[str, ...]:
    """Validate a unique, canonically ordered quality-flag sequence."""

    if isinstance(flags, (str, bytes, bytearray)) or any(
        not isinstance(flag, str) or flag not in QUALITY_FLAGS for flag in flags
    ):
        raise _cell_error(field="quality_disposition", review_key=review_key, reason="is invalid")
    canonical = tuple(sorted(flags))
    if len(canonical) != len(set(canonical)) or tuple(flags) != canonical:
        raise _cell_error(
            field="quality_disposition",
            review_key=review_key,
            reason="must be unique and canonical",
        )
    return canonical


def review_quality_flags(value: str, *, review_key: str) -> tuple[str, ...]:
    """Parse one canonical quality disposition into its shared flag vocabulary."""

    if value == "acceptable":
        return ()
    if not value:
        raise _cell_error(field="quality_disposition", review_key=review_key, reason="is invalid")
    return validate_quality_flags(value.split(";"), review_key=review_key)


def validate_suitability_disposition(
    value: str, *, review_kind: str, review_status: str, review_key: str
) -> str:
    """Validate suitability against the state and kind where it is meaningful."""

    if review_status == "pending":
        allowed = frozenset({"pending"}) if review_kind == "item" else frozenset({""})
    elif review_kind == "item":
        allowed = LEGACY_SUITABILITY_DISPOSITIONS
    elif review_kind == "visual_item":
        allowed = VISUAL_SUITABILITY_DISPOSITIONS
    else:
        allowed = frozenset({""})
    if value not in allowed:
        raise _cell_error(
            field="suitability_disposition", review_key=review_key, reason="is invalid"
        )
    return value


def validate_duplicate_disposition(
    value: str, *, review_kind: str, review_status: str, review_key: str
) -> str:
    """Validate human pair decisions and the sole non-human unavailable branch."""

    if review_status == "pending":
        allowed = (
            frozenset({"pending"})
            if review_kind in {"candidate", "duplicate_pair", "item"}
            else frozenset({""})
        )
    elif review_status == "unavailable":
        allowed = frozenset({"unavailable"}) if review_kind == "duplicate_pair" else frozenset()
    elif review_kind in {"candidate", "duplicate_pair", "item"}:
        allowed = HUMAN_PAIR_DISPOSITIONS
    else:
        allowed = frozenset({""})
    if value not in allowed:
        reason = (
            "cannot use unavailable for a human-reviewed decision"
            if value == "unavailable" and review_status == "reviewed"
            else "is invalid"
        )
        raise _cell_error(field="duplicate_disposition", review_key=review_key, reason=reason)
    return value


def validate_review_row_domains(row: Mapping[str, str]) -> None:
    """Enforce the exact state-aware enum union for one canonical review row."""

    kind = _require_domain_value(row, "review_kind", REVIEW_KINDS)
    status = _require_domain_value(row, "review_status", REVIEW_STATUSES)
    key = row["review_key"]

    if status == "unavailable":
        if kind != "duplicate_pair":
            raise _cell_error(
                field="review_status",
                review_key=key,
                reason="unavailable is allowed only for duplicate_pair",
            )
        validate_duplicate_disposition(
            row["duplicate_disposition"],
            review_kind=kind,
            review_status=status,
            review_key=key,
        )
        if row["reviewer"] or row["reviewed_at"]:
            field = "reviewer" if row["reviewer"] else "reviewed_at"
            raise _cell_error(field=field, review_key=key, reason="must be empty when unavailable")
        if not row["rationale"].strip():
            raise _cell_error(
                field="rationale", review_key=key, reason="is required when unavailable"
            )
    elif status == "reviewed":
        for field in ("reviewer", "reviewed_at", "rationale"):
            if not row[field].strip():
                raise _cell_error(field=field, review_key=key, reason="is required")
    elif row["reviewed_at"]:
        raise _cell_error(
            field="review_status",
            review_key=key,
            reason="must be reviewed when reviewed_at is populated",
        )

    if kind == "item":
        if status == "pending":
            if row["quality_disposition"]:
                review_quality_flags(row["quality_disposition"], review_key=key)
            validate_suitability_disposition(
                row["suitability_disposition"],
                review_kind=kind,
                review_status=status,
                review_key=key,
            )
            validate_duplicate_disposition(
                row["duplicate_disposition"],
                review_kind=kind,
                review_status=status,
                review_key=key,
            )
            _require_domain_value(row, "dataset_licence_status", frozenset({"pending"}))
            _require_domain_value(
                row,
                "item_provenance_status",
                frozenset({"pending", "unavailable"}),
            )
            _require_domain_value(row, "access_status", frozenset({"pending", "confirmed"}))
            _require_domain_value(row, "redistribution_status", frozenset({"pending"}))
            _require_domain_value(row, "figure_reproduction_status", frozenset({"pending"}))
        else:
            review_quality_flags(row["quality_disposition"], review_key=key)
            validate_suitability_disposition(
                row["suitability_disposition"],
                review_kind=kind,
                review_status=status,
                review_key=key,
            )
            validate_duplicate_disposition(
                row["duplicate_disposition"],
                review_kind=kind,
                review_status=status,
                review_key=key,
            )
            _require_domain_value(row, "dataset_licence_status", DATASET_LICENCE_STATUSES)
            _require_domain_value(row, "item_provenance_status", ITEM_PROVENANCE_STATUSES)
            _require_domain_value(row, "access_status", ACCESS_STATUSES)
            _require_domain_value(row, "redistribution_status", LEGACY_REUSE_STATUSES)
            _require_domain_value(row, "figure_reproduction_status", LEGACY_REUSE_STATUSES)
        _require_empty_domains(row, ("candidate_item_id",))
        return

    if kind == "candidate":
        validate_duplicate_disposition(
            row["duplicate_disposition"],
            review_kind=kind,
            review_status=status,
            review_key=key,
        )
        _require_empty_domains(
            row,
            (
                "source_group_id",
                "quality_disposition",
                "suitability_disposition",
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            ),
        )
        return

    if kind == "item_policy":
        if status == "pending":
            for field in (
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            ):
                _require_domain_value(row, field, frozenset({"pending"}))
        else:
            _require_domain_value(row, "dataset_licence_status", frozenset({"confirmed"}))
            _require_domain_value(row, "item_provenance_status", ITEM_PROVENANCE_STATUSES)
            _require_domain_value(row, "access_status", ACCESS_STATUSES)
            _require_domain_value(row, "redistribution_status", V2_REUSE_STATUSES)
            _require_domain_value(row, "figure_reproduction_status", V2_REUSE_STATUSES)
            if row["item_provenance_status"] == "unavailable" and (
                row["redistribution_status"] == "permitted"
                or row["figure_reproduction_status"] == "permitted"
            ):
                raise _cell_error(
                    field="item_provenance_status",
                    review_key=key,
                    reason="cannot infer reuse permission",
                )
        if not row["source_group_id"]:
            raise _cell_error(field="source_group_id", review_key=key, reason="is required")
        _require_empty_domains(
            row,
            (
                "candidate_item_id",
                "quality_disposition",
                "suitability_disposition",
                "duplicate_disposition",
            ),
        )
        return

    if kind == "visual_item":
        if status == "pending":
            _require_empty_domains(row, ("quality_disposition", "suitability_disposition"))
        else:
            review_quality_flags(row["quality_disposition"], review_key=key)
            validate_suitability_disposition(
                row["suitability_disposition"],
                review_kind=kind,
                review_status=status,
                review_key=key,
            )
        _require_empty_domains(
            row,
            (
                "candidate_item_id",
                "source_group_id",
                "duplicate_disposition",
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            ),
        )
        return

    validate_duplicate_disposition(
        row["duplicate_disposition"],
        review_kind=kind,
        review_status=status,
        review_key=key,
    )
    _require_empty_domains(
        row,
        (
            "source_group_id",
            "quality_disposition",
            "suitability_disposition",
            "dataset_licence_status",
            "item_provenance_status",
            "access_status",
            "redistribution_status",
            "figure_reproduction_status",
        ),
    )


def validate_review_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Validate the exact review shape and every cell without normalizing its contents."""

    validated: list[dict[str, str]] = []
    expected_fields = set(REVIEW_FIELDS)
    for index, supplied in enumerate(rows):
        if set(supplied) != expected_fields:
            raise ReviewEvidenceError(f"row-{index}: review row does not match the exact header")
        key = _safe_key(supplied.get("review_key"), row_index=index)
        row: dict[str, str] = {}
        for field in REVIEW_FIELDS:
            value = supplied[field]
            if not isinstance(value, str):
                raise _cell_error(field=field, review_key=key, reason="must be text")
            row[field] = validate_human_cell(value, field=field, review_key=key)
        validate_review_row_domains(row)
        validated.append(row)
    return validated


def canonical_review_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize validated rows as deterministic UTF-8 CSV with LF line endings."""

    validated = validate_review_rows(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=REVIEW_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(validated)
    return output.getvalue().encode("utf-8")


def read_review(path: Path) -> ReviewDocument:
    """Read one exact-header review file and return validated rows plus its digest."""

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ReviewEvidenceError("review CSV header does not match the exact contract")
        rows = list(reader)
    except ReviewEvidenceError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReviewEvidenceError(f"cannot read review CSV: {type(error).__name__}") from error
    if any(None in row or set(row) != set(REVIEW_FIELDS) for row in rows):
        raise ReviewEvidenceError("review CSV row does not match the exact header")
    validated = validate_review_rows(rows)
    canonical = canonical_review_csv(validated)
    return ReviewDocument(
        rows=tuple(validated),
        sha256=hashlib.sha256(raw).hexdigest(),
        canonical_bytes=canonical,
    )


def _review_lock_path(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    identity = hashlib.sha256(os.fsencode(resolved)).hexdigest()
    user_id = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"score-sr-review-{user_id}-{identity}.lock"


@contextmanager
def _locked_review(path: Path) -> object:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0:
        raise RuntimeError("review locking requires no-follow file support")
    descriptor = os.open(_review_lock_path(path), flags | no_follow, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError("review lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _save_boundary(name: str, hook: Callable[[str], None] | None) -> None:
    if name not in REVIEW_SAVE_BOUNDARIES:
        raise ValueError(f"unknown review save boundary: {name}")
    if hook is not None:
        hook(name)
    failpoint = os.environ.get("SCORE_SR_REVIEW_SAVE_FAILPOINT")
    if failpoint == f"{name}:raise":
        raise OSError("injected review save failure")
    if failpoint == f"{name}:exit":
        os._exit(91)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_review_temp(path: Path) -> tuple[Path, int]:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    for _attempt in range(16):
        temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
        try:
            return temporary, os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique review temporary file")


def save_review(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    expected_sha256: str,
    boundary_hook: Callable[[str], None] | None = None,
) -> str:
    """Compare-and-swap one canonical review snapshot through a durable replacement."""

    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    content = canonical_review_csv(rows)
    new_sha256 = hashlib.sha256(content).hexdigest()
    temporary: Path | None = None
    committed = False
    try:
        with _locked_review(path):
            current = read_review(path)
            if current.sha256 != expected_sha256:
                raise StaleReviewError
            temporary, descriptor = _exclusive_review_temp(path)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(content)
                handle.flush()
                _save_boundary("review_written", boundary_hook)
                os.fsync(handle.fileno())
                _save_boundary("review_fsynced", boundary_hook)
            os.replace(temporary, path)
            committed = True
            _save_boundary("review_replaced", boundary_hook)
            _fsync_directory(path.parent)
            _save_boundary("review_parent_fsynced", boundary_hook)
    except (ReviewEvidenceError, ReviewPersistenceError):
        raise
    except Exception as error:
        state = "after replacement; reload required" if committed else "before replacement"
        raise ReviewPersistenceError(
            f"review save failed {state}: {type(error).__name__}",
            committed=committed,
            sha256=new_sha256 if committed else None,
        ) from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return new_sha256
