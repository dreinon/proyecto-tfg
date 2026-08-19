from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from score_super_resolution.review_evidence import (
    REVIEW_FIELDS,
    REVIEW_SAVE_BOUNDARIES,
    ReviewEvidenceError,
    ReviewPersistenceError,
    StaleReviewError,
    canonical_review_csv,
    read_review,
    save_review,
    validate_human_cell,
)
from score_super_resolution.smb_review_ui import SMBReviewSession

TRACKED_REVIEW = Path(__file__).parents[1] / "data" / "audits" / "smb-review-v1.csv"
CONTROL_CHARACTERS = tuple(chr(codepoint) for codepoint in (*range(0x20), 0x7F, *range(0x80, 0xA0)))
FORMULA_PREFIXES = ("=2+2", "+2+2", "-2+2", "@SUM(1,1)", "  =2+2", "\v@SUM(1,1)")


def _safe_rows() -> list[dict[str, str]]:
    with TRACKED_REVIEW.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REVIEW_FIELDS
        return list(reader)


def _session(review_path: Path) -> SMBReviewSession:
    session = object.__new__(SMBReviewSession)
    session.review_path = review_path
    session.reload()
    return session


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_bytes(canonical_review_csv(rows))


@pytest.mark.parametrize("payload", FORMULA_PREFIXES)
def test_formula_leading_human_cells_are_rejected_without_reflection(payload: str) -> None:
    with pytest.raises(ReviewEvidenceError) as caught:
        validate_human_cell(payload, field="rationale", review_key="smb-test-000000")

    message = str(caught.value)
    assert "rationale" in message
    assert "smb-test-000000" in message
    assert payload not in message


@pytest.mark.parametrize("control", CONTROL_CHARACTERS, ids=lambda value: f"U+{ord(value):04X}")
def test_complete_control_character_table_is_rejected(control: str) -> None:
    payload = f"texto{control}seguro"

    with pytest.raises(ReviewEvidenceError) as caught:
        validate_human_cell(payload, field="reviewer", review_key="smb-test-000000")

    message = str(caught.value)
    assert "reviewer" in message
    assert "smb-test-000000" in message
    assert payload not in message


def test_existing_accented_review_round_trips_to_identical_canonical_bytes() -> None:
    document = read_review(TRACKED_REVIEW)

    assert len(document.rows) == 699
    assert any(row["reviewer"] == "Daniel Reinón García" for row in document.rows)
    assert canonical_review_csv(document.rows) == TRACKED_REVIEW.read_bytes()
    assert document.canonical_bytes == TRACKED_REVIEW.read_bytes()


@pytest.mark.parametrize(
    ("reviewer", "rationale"),
    (("=2+2", "Justificación segura"), ("Daniel Reinón García", "@SUM(1,1)")),
)
def test_direct_ui_save_rejects_formula_examples(
    tmp_path: Path, reviewer: str, rationale: str
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    session = _session(review_path)

    with pytest.raises(ReviewEvidenceError):
        session.save_item(
            item_id="smb-test-000000",
            reviewer=reviewer,
            quality_flags=(),
            suitability="suitable",
            rationale=rationale,
        )

    assert review_path.read_bytes() == canonical_review_csv(_safe_rows())


@pytest.mark.parametrize("field", REVIEW_FIELDS)
def test_ui_write_boundary_rejects_unsafe_content_in_every_cell(tmp_path: Path, field: str) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    rows = _safe_rows()
    _write_rows(review_path, rows)
    mutated = copy.deepcopy(rows)
    mutated[0][field] = "  =2+2"
    session = _session(review_path)

    with pytest.raises(ReviewEvidenceError, match=field):
        session.write_rows(mutated)

    assert review_path.read_bytes() == canonical_review_csv(rows)


def test_display_neutralization_is_not_valid_canonical_input() -> None:
    assert validate_human_cell("'=2+2", field="rationale", review_key="smb-test-000000") == "'=2+2"
    with pytest.raises(ReviewEvidenceError):
        validate_human_cell("=2+2", field="rationale", review_key="smb-test-000000")


def _changed_rows(*, rationale: str) -> list[dict[str, str]]:
    rows = _safe_rows()
    rows[0]["rationale"] = rationale
    return rows


def test_two_synchronized_sessions_yield_one_commit_and_one_stale_error(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    sessions = (_session(review_path), _session(review_path))
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def save(session: SMBReviewSession, rationale: str) -> None:
        barrier.wait(timeout=10)
        try:
            session.save_item(
                item_id="smb-test-000000",
                reviewer="Daniel Reinón García",
                quality_flags=(),
                suitability="suitable",
                rationale=rationale,
            )
            outcomes.append(rationale)
        except BaseException as error:  # pragma: no cover - asserted below
            outcomes.append(error)

    writers = [
        threading.Thread(target=save, args=(sessions[0], "edición primera")),
        threading.Thread(target=save, args=(sessions[1], "edición segunda")),
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=10)

    stale = [outcome for outcome in outcomes if isinstance(outcome, StaleReviewError)]
    committed = [outcome for outcome in outcomes if isinstance(outcome, str)]
    assert len(stale) == 1
    assert len(committed) == 1
    assert "reload" in str(stale[0]).casefold()
    assert read_review(review_path).rows[0]["rationale"] == committed[0]


def test_stale_session_can_reload_then_save_without_repeating_review(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    first = _session(review_path)
    stale = _session(review_path)
    first.save_item(
        item_id="smb-test-000000",
        reviewer="Daniel Reinón García",
        quality_flags=(),
        suitability="suitable",
        rationale="edición confirmada",
    )
    assert first.expected_sha256 == read_review(review_path).sha256

    with pytest.raises(StaleReviewError, match="reload"):
        stale.save_item(
            item_id="smb-test-000001",
            reviewer="Daniel Reinón García",
            quality_flags=(),
            suitability="suitable",
            rationale="edición conservada en sesión",
        )

    stale.reload()
    stale.save_item(
        item_id="smb-test-000001",
        reviewer="Daniel Reinón García",
        quality_flags=(),
        suitability="suitable",
        rationale="edición conservada en sesión",
    )
    assert stale.expected_sha256 == read_review(review_path).sha256
    rows = read_review(review_path).rows
    assert rows[0]["rationale"] == "edición confirmada"
    assert rows[1]["rationale"] == "edición conservada en sesión"


def test_unique_same_directory_temps_do_not_reuse_fixed_collision(tmp_path: Path) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    fixed_collision = review_path.with_suffix(".csv.tmp")
    fixed_collision.write_bytes(b"do-not-touch")
    observed: list[Path] = []

    def observe(boundary: str) -> None:
        if boundary == "review_written":
            observed.extend(review_path.parent.glob(f".{review_path.name}.tmp-*"))

    for rationale in ("guardado uno", "guardado dos"):
        current = read_review(review_path)
        save_review(
            review_path,
            _changed_rows(rationale=rationale),
            expected_sha256=current.sha256,
            boundary_hook=observe,
        )

    assert len(observed) == 2
    assert len(set(observed)) == 2
    assert all(path.parent == review_path.parent for path in observed)
    assert fixed_collision.read_bytes() == b"do-not-touch"
    assert not any(path.exists() for path in observed)


@pytest.mark.parametrize("boundary", REVIEW_SAVE_BOUNDARIES)
def test_ordinary_failure_boundaries_leave_complete_old_or_new(
    tmp_path: Path, boundary: str
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    before = read_review(review_path)
    changed = _changed_rows(rationale="nueva edición completa")
    changed_bytes = canonical_review_csv(changed)

    def failpoint(observed: str) -> None:
        if observed == boundary:
            raise OSError("injected review save failure")

    with pytest.raises(ReviewPersistenceError) as caught:
        save_review(
            review_path,
            changed,
            expected_sha256=before.sha256,
            boundary_hook=failpoint,
        )

    committed = boundary in {"review_replaced", "review_parent_fsynced"}
    assert caught.value.committed is committed
    assert review_path.read_bytes() in {before.canonical_bytes, changed_bytes}
    assert read_review(review_path).canonical_bytes == review_path.read_bytes()
    assert not list(review_path.parent.glob(f".{review_path.name}.tmp-*"))


@pytest.mark.parametrize("boundary", REVIEW_SAVE_BOUNDARIES)
def test_abrupt_exit_boundaries_leave_parseable_complete_old_or_new(
    tmp_path: Path, boundary: str
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    before = read_review(review_path)
    changed = _changed_rows(rationale="edición tras salida abrupta")
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    code = """
import json
import sys
from pathlib import Path
from score_super_resolution.review_evidence import read_review, save_review

path = Path(sys.argv[1])
rows = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
save_review(path, rows, expected_sha256=read_review(path).sha256)
"""
    environment = os.environ.copy()
    environment["SCORE_SR_REVIEW_SAVE_FAILPOINT"] = f"{boundary}:exit"

    completed = subprocess.run(
        [sys.executable, "-c", code, str(review_path), str(changed_path)],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 91, completed.stderr
    assert review_path.read_bytes() in {before.canonical_bytes, canonical_review_csv(changed)}
    assert read_review(review_path).canonical_bytes == review_path.read_bytes()


def test_save_boundaries_prove_fsync_replace_and_parent_fsync_order(tmp_path: Path) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    current = read_review(review_path)
    events: list[str] = []

    new_sha256 = save_review(
        review_path,
        _changed_rows(rationale="orden durable"),
        expected_sha256=current.sha256,
        boundary_hook=events.append,
    )

    assert tuple(events) == REVIEW_SAVE_BOUNDARIES
    assert new_sha256 == read_review(review_path).sha256
