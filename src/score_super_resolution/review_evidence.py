"""Canonical, fail-closed I/O primitives for tracked human-review evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Mapping, Sequence
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

_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SAFE_REVIEW_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ReviewEvidenceError(ValueError):
    """Report unsafe or structurally invalid review evidence without reflecting payloads."""


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
