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

import score_super_resolution.review_evidence as review_evidence
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

    assert len(document.rows) == 763
    assert any(row["reviewer"] == "Daniel Reinón García" for row in document.rows)
    assert canonical_review_csv(document.rows) == TRACKED_REVIEW.read_bytes()
    assert document.canonical_bytes == TRACKED_REVIEW.read_bytes()


@pytest.mark.parametrize(
    ("review_kind", "field", "invalid_value"),
    (
        ("item_policy", "review_kind", "invented-kind"),
        ("item_policy", "review_status", "invented-status"),
        ("visual_item", "quality_disposition", "invented-quality"),
        ("visual_item", "suitability_disposition", "invented-suitability"),
        ("duplicate_pair", "duplicate_disposition", "invented-pair-state"),
        ("item_policy", "dataset_licence_status", "invented-licence"),
        ("item_policy", "item_provenance_status", "invented-provenance"),
        ("item_policy", "access_status", "invented-access"),
        ("item_policy", "redistribution_status", "invented-redistribution"),
        ("item_policy", "figure_reproduction_status", "invented-figure-policy"),
    ),
)
def test_canonical_review_rejects_every_invalid_domain_enum(
    review_kind: str,
    field: str,
    invalid_value: str,
) -> None:
    before = TRACKED_REVIEW.read_bytes()
    rows = _safe_rows()
    row = next(row for row in rows if row["review_kind"] == review_kind)
    row[field] = invalid_value

    with pytest.raises(ReviewEvidenceError, match=field):
        canonical_review_csv(rows)

    assert TRACKED_REVIEW.read_bytes() == before


def test_canonical_review_accepts_emitted_pending_and_unavailable_state_unions() -> None:
    rows = _safe_rows()
    policy = next(row for row in rows if row["review_kind"] == "item_policy")
    visual = next(row for row in rows if row["review_kind"] == "visual_item")
    pair = next(row for row in rows if row["review_kind"] == "duplicate_pair")

    for field in ("reviewer", "reviewed_at", "rationale"):
        policy[field] = ""
        visual[field] = ""
    policy["review_status"] = "pending"
    for field in (
        "dataset_licence_status",
        "item_provenance_status",
        "access_status",
        "redistribution_status",
        "figure_reproduction_status",
    ):
        policy[field] = "pending"
    visual["review_status"] = "pending"
    visual["quality_disposition"] = ""
    visual["suitability_disposition"] = ""
    pair.update(
        {
            "review_status": "unavailable",
            "reviewer": "",
            "reviewed_at": "",
            "rationale": "Perceptual comparison evidence is unavailable.",
            "duplicate_disposition": "unavailable",
        }
    )

    assert canonical_review_csv((policy, visual, pair))


@pytest.mark.parametrize("reviewed_at", ("not-a-date", "2026-02-30", "2026-8-20"))
def test_canonical_review_rejects_noncanonical_or_impossible_review_dates(
    reviewed_at: str,
) -> None:
    row = _safe_rows()[0]
    row["reviewed_at"] = reviewed_at

    with pytest.raises(ReviewEvidenceError, match="reviewed_at"):
        canonical_review_csv((row,))


def test_canonical_review_rejects_duplicate_stable_review_keys() -> None:
    first, second = (row.copy() for row in _safe_rows()[:2])
    second["review_key"] = first["review_key"]

    with pytest.raises(ReviewEvidenceError, match=r"review_key.*duplicate"):
        canonical_review_csv((first, second))


@pytest.mark.parametrize(
    ("review_kind", "updates", "match"),
    (
        (
            "visual_item",
            {"review_status": "unavailable", "suitability_disposition": "unavailable"},
            "review_status",
        ),
        ("duplicate_pair", {"quality_disposition": "acceptable"}, "quality_disposition"),
        (
            "duplicate_pair",
            {"duplicate_disposition": "unavailable"},
            "duplicate_disposition",
        ),
    ),
)
def test_canonical_review_rejects_state_or_kind_irrelevant_domains(
    review_kind: str,
    updates: dict[str, str],
    match: str,
) -> None:
    row = next(row for row in _safe_rows() if row["review_kind"] == review_kind)
    row.update(updates)

    with pytest.raises(ReviewEvidenceError, match=match):
        canonical_review_csv((row,))


def test_invalid_domain_save_fails_before_temporary_creation_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    before = read_review(review_path)
    rows = [row.copy() for row in before.rows]
    visual = next(row for row in rows if row["review_kind"] == "visual_item")
    visual["quality_disposition"] = "invented-quality"

    def unexpected_temp(_path: Path) -> tuple[Path, int]:
        raise AssertionError("invalid review reached temporary creation")

    monkeypatch.setattr(review_evidence, "_exclusive_review_temp", unexpected_temp)

    with pytest.raises(ReviewEvidenceError, match="quality_disposition"):
        save_review(review_path, rows, expected_sha256=before.sha256)

    after = read_review(review_path)
    assert review_path.read_bytes() == before.canonical_bytes
    assert after.sha256 == before.sha256
    assert after.rows == before.rows


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("quality", ("invented-quality",)),
        ("suitability", "invented-suitability"),
    ),
)
def test_invalid_item_enum_save_leaves_file_digest_and_session_rows_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: tuple[str, ...] | str,
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    session = _session(review_path)
    item_id = next(
        row["item_id"] for row in session.read_rows() if row["review_kind"] == "visual_item"
    )
    before_bytes = review_path.read_bytes()
    before_sha256 = session.expected_sha256
    before_rows = session.read_rows()

    def unexpected_write(_rows: list[dict[str, str]]) -> None:
        raise AssertionError("invalid item enum reached persistence")

    monkeypatch.setattr(session, "write_rows", unexpected_write)
    quality_flags = value if field == "quality" else ()
    suitability = value if field == "suitability" else "suitable"
    assert isinstance(quality_flags, tuple)
    assert isinstance(suitability, str)

    with pytest.raises(ReviewEvidenceError, match=f"{field}.*disposition"):
        session.save_item(
            item_id=item_id,
            reviewer="Daniel Reinón García",
            quality_flags=quality_flags,
            suitability=suitability,
            rationale="La revisión válida no debe cambiar.",
        )

    assert review_path.read_bytes() == before_bytes
    assert session.expected_sha256 == before_sha256
    assert session.read_rows() == before_rows


@pytest.mark.parametrize("disposition", ("invented-state", "unavailable"))
def test_invalid_pair_enum_save_leaves_file_digest_and_session_rows_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    session = _session(review_path)
    review_key = next(
        row["review_key"] for row in session.read_rows() if row["review_kind"] == "duplicate_pair"
    )
    before_bytes = review_path.read_bytes()
    before_sha256 = session.expected_sha256
    before_rows = session.read_rows()

    def unexpected_write(_rows: list[dict[str, str]]) -> None:
        raise AssertionError("invalid pair enum reached persistence")

    monkeypatch.setattr(session, "write_rows", unexpected_write)

    with pytest.raises(ReviewEvidenceError, match="duplicate_disposition"):
        session.save_candidate(
            review_key=review_key,
            reviewer="Daniel Reinón García",
            disposition=disposition,
            rationale="La revisión válida no debe cambiar.",
        )

    assert review_path.read_bytes() == before_bytes
    assert session.expected_sha256 == before_sha256
    assert session.read_rows() == before_rows


def test_candidate_save_rejects_non_pair_kind_without_mutating_or_persisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    session = _session(review_path)
    review_key = next(
        row["review_key"] for row in session.read_rows() if row["review_kind"] == "item_policy"
    )
    before_bytes = review_path.read_bytes()
    before_sha256 = session.expected_sha256
    before_rows = session.read_rows()

    def unexpected_write(_rows: list[dict[str, str]]) -> None:
        raise AssertionError("non-pair review kind reached persistence")

    monkeypatch.setattr(session, "write_rows", unexpected_write)

    with pytest.raises(ReviewEvidenceError, match="review_kind"):
        session.save_candidate(
            review_key=review_key,
            reviewer="Daniel Reinón García",
            disposition="distinct",
            rationale="La revisión válida no debe cambiar.",
        )

    assert review_path.read_bytes() == before_bytes
    assert session.expected_sha256 == before_sha256
    assert session.read_rows() == before_rows


def test_summary_counts_unavailable_as_terminal_not_pending(tmp_path: Path) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    rows = _safe_rows()
    pair = next(row for row in rows if row["review_kind"] == "duplicate_pair")
    pair.update(
        {
            "review_status": "unavailable",
            "reviewer": "",
            "reviewed_at": "",
            "rationale": "Perceptual comparison evidence is unavailable.",
            "duplicate_disposition": "unavailable",
        }
    )
    _write_rows(review_path, rows)
    session = _session(review_path)
    session.visual_item_ids = [
        row["item_id"] for row in rows if row["review_kind"] == "visual_item"
    ]
    session.sample_ids = session.visual_item_ids.copy()
    session.candidate_keys = [
        row["review_key"] for row in rows if row["review_kind"] == "duplicate_pair"
    ]

    summary = session.summary()

    assert summary["reviewed"] == len(rows) - 1
    assert summary["unavailable"] == 1
    assert summary["completed"] == len(rows)
    assert summary["pending"] == 0


def test_apply_policy_rejects_unsupported_legacy_item_ui_without_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = next(row.copy() for row in _safe_rows() if row["review_kind"] == "item_policy")
    row.update(
        {
            "review_kind": "item",
            "review_key": row["item_id"],
            "review_status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "rationale": "",
            "quality_disposition": "",
            "suitability_disposition": "pending",
            "duplicate_disposition": "pending",
            "dataset_licence_status": "pending",
            "item_provenance_status": "pending",
            "access_status": "pending",
            "redistribution_status": "pending",
            "figure_reproduction_status": "pending",
        }
    )
    review_path = tmp_path / "legacy-review.csv"
    _write_rows(review_path, [row])
    session = _session(review_path)

    def unexpected_write(_rows: list[dict[str, str]]) -> None:
        raise AssertionError("unsupported legacy policy reached persistence")

    monkeypatch.setattr(session, "write_rows", unexpected_write)

    with pytest.raises(ReviewEvidenceError, match=r"legacy.*unsupported|unsupported.*legacy"):
        session.apply_policy("Daniel Reinón García")

    assert review_path.read_bytes() == canonical_review_csv((row,))


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
                item_id="smb-test-000006",
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
    committed_row = next(
        row
        for row in read_review(review_path).rows
        if row["review_key"] == "visual:smb-test-000006"
    )
    assert committed_row["rationale"] == committed[0]


def test_stale_session_can_reload_then_save_without_repeating_review(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    first = _session(review_path)
    stale = _session(review_path)
    first.save_item(
        item_id="smb-test-000006",
        reviewer="Daniel Reinón García",
        quality_flags=(),
        suitability="suitable",
        rationale="edición confirmada",
    )
    assert first.expected_sha256 == read_review(review_path).sha256

    with pytest.raises(StaleReviewError, match="reload"):
        stale.save_item(
            item_id="smb-test-000007",
            reviewer="Daniel Reinón García",
            quality_flags=(),
            suitability="suitable",
            rationale="edición conservada en sesión",
        )

    stale.reload()
    stale.save_item(
        item_id="smb-test-000007",
        reviewer="Daniel Reinón García",
        quality_flags=(),
        suitability="suitable",
        rationale="edición conservada en sesión",
    )
    assert stale.expected_sha256 == read_review(review_path).sha256
    rows = read_review(review_path).rows
    by_key = {row["review_key"]: row for row in rows}
    assert by_key["visual:smb-test-000006"]["rationale"] == "edición confirmada"
    assert by_key["visual:smb-test-000007"]["rationale"] == "edición conservada en sesión"


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


@pytest.mark.parametrize(
    "boundary",
    ("review_before_cas_read", "review_before_temp_create", "review_before_replace"),
)
def test_review_save_parent_swap_stays_on_retained_directory(tmp_path: Path, boundary: str) -> None:
    review_parent = tmp_path / "review-parent"
    review_parent.mkdir()
    review_path = review_parent / "smb-review-v1.csv"
    _write_rows(review_path, _safe_rows())
    before = read_review(review_path)
    changed = _changed_rows(rationale=f"guardado confinado en {boundary}")
    changed_bytes = canonical_review_csv(changed)
    retained_parent = tmp_path / f"retained-{boundary}"
    outside_parent = tmp_path / f"outside-{boundary}"
    outside_parent.mkdir()
    outside_review = outside_parent / review_path.name
    outside_review.write_bytes(before.canonical_bytes)
    swapped = False

    def hook(observed: str) -> None:
        nonlocal swapped
        if observed == boundary:
            review_parent.rename(retained_parent)
            review_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True

    new_sha256 = save_review(
        review_path,
        changed,
        expected_sha256=before.sha256,
        boundary_hook=hook,
    )

    assert swapped is True
    assert outside_review.read_bytes() == before.canonical_bytes
    retained_review = retained_parent / review_path.name
    assert retained_review.read_bytes() == changed_bytes
    assert new_sha256 == read_review(retained_review).sha256


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
