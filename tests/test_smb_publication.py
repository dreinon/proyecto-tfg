from __future__ import annotations

import ast
import copy
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

import score_super_resolution.smb_audit as smb_audit

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "smb" / "records.json"
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
COMMITTED_BOUNDARIES = {"pointer_replaced", "active_parent_fsynced"}


def _fixtures() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _descriptor() -> dict[str, Any]:
    return copy.deepcopy(_fixtures()["manifest_descriptor"])


def _full_rows(*, with_candidate: bool = True) -> list[dict[str, Any]]:
    template = _fixtures()["normal_row"]
    rows: list[dict[str, Any]] = []
    for index in range(685):
        row = copy.deepcopy(template)
        row["upstream_index"] = index
        row["item_id"] = f"smb-test-{index:06d}"
        row["original_score"] = {
            "raw": f"score-{index:03d}",
            "normalized": f"score-{index:03d}",
        }
        row["source_group_id"] = f"score-{index:03d}"
        row["audit_sample_member"] = index < 64
        rows.append(row)
    if with_candidate:
        candidate_id = smb_audit._candidate_id(rows[0]["item_id"], rows[1]["item_id"])
        rows[0]["near_duplicate_candidate_ids"] = [candidate_id]
        rows[1]["near_duplicate_candidate_ids"] = [candidate_id]
    return rows


def _publish(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    active_path = tmp_path / "data" / "manifests" / "smb-evaluation-v1.yaml"
    generation_root = tmp_path / "artifacts" / "smb-manifests" / "generations"
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=rows or _full_rows(),
    )
    return active_path, generation_root


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=smb_audit.REVIEW_CSV_FIELDS, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(rows)


def _complete_review(path: Path) -> list[dict[str, str]]:
    rows = _read_review(path)
    for row in rows:
        row["review_status"] = "reviewed"
        row["reviewer"] = "reviewer-1"
        row["reviewed_at"] = "2026-08-18"
        row["rationale"] = "Reviewed from the frozen audit evidence."
        if row["review_kind"] == "item":
            row["quality_disposition"] = "acceptable"
            row["suitability_disposition"] = "suitable"
            row["duplicate_disposition"] = "distinct"
            row["dataset_licence_status"] = "confirmed"
            row["item_provenance_status"] = "confirmed"
            row["access_status"] = "confirmed"
            row["redistribution_status"] = "permitted"
            row["figure_reproduction_status"] = "permitted"
        else:
            row["duplicate_disposition"] = "distinct"
    _write_review(path, rows)
    return rows


def _emit_review_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    active_path, generation_root = _publish(tmp_path)
    audit_descriptor = tmp_path / "tracked" / "smb-audit-v1.yaml"
    audit_records = tmp_path / "tracked" / "smb-audit-v1.jsonl"
    sample = tmp_path / "tracked" / "smb-visual-sample-v1.csv"
    review = tmp_path / "tracked" / "smb-review-v1.csv"
    smb_audit.emit_review_evidence_from_active_manifest(
        active_path=active_path,
        generation_root=generation_root,
        audit_descriptor_path=audit_descriptor,
        audit_records_path=audit_records,
        sample_path=sample,
        review_path=review,
    )
    return active_path, generation_root, audit_descriptor, audit_records, sample, review


def test_review_evidence_is_pointer_resolved_reproducible_and_redacted(tmp_path: Path) -> None:
    paths = _emit_review_bundle(tmp_path)
    active_path, generation_root, *outputs = paths
    first_bytes = [path.read_bytes() for path in outputs]

    smb_audit.emit_review_evidence_from_active_manifest(
        active_path=active_path,
        generation_root=generation_root,
        audit_descriptor_path=outputs[0],
        audit_records_path=outputs[1],
        sample_path=outputs[2],
        review_path=outputs[3],
    )

    assert [path.read_bytes() for path in outputs] == first_bytes
    audit_rows = [json.loads(line) for line in outputs[1].read_text().splitlines()]
    assert len(audit_rows) == 685
    assert all(
        set(row)
        == {
            "audit_sample_member",
            "item_id",
            "processing_status",
            "source_group_id",
            "upstream_index",
        }
        for row in audit_rows
    )
    serialized = b"\n".join(first_bytes)
    assert b"image" not in serialized
    assert b"encoded_sha256" not in serialized
    assert b"pixel_sha256" not in serialized
    assert len(_read_review(outputs[3])) == 686


def test_validate_review_is_strictly_read_only(tmp_path: Path) -> None:
    active_path, generation_root, _, _, _, review = _emit_review_bundle(tmp_path)
    _complete_review(review)
    before_bytes = review.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()
    before_stat = review.stat()

    smb_audit.validate_review_from_active_manifest(
        review_path=review,
        active_path=active_path,
        generation_root=generation_root,
    )

    after_bytes = review.read_bytes()
    after_stat = review.stat()
    assert after_bytes == before_bytes
    assert hashlib.sha256(after_bytes).hexdigest() == before_hash
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "missing review key"),
        ("duplicate", "duplicate review key"),
        ("unknown", "unknown review key"),
        ("pending", "must be reviewed"),
        ("missing_reviewer", "reviewer"),
        ("missing_rationale", "rationale"),
        ("bad_date", "reviewed_at"),
        ("bad_enum", "suitability_disposition"),
        ("reversed_candidate", "canonical"),
        ("ambiguous", "ambiguous"),
    ),
)
def test_review_validation_rejects_incomplete_or_unstable_joins(
    tmp_path: Path, mutation: str, match: str
) -> None:
    active_path, generation_root, _, _, _, review = _emit_review_bundle(tmp_path)
    rows = _complete_review(review)
    candidate = next(row for row in rows if row["review_kind"] == "candidate")
    if mutation == "missing":
        rows.pop(5)
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[5]))
    elif mutation == "unknown":
        rows[5]["review_key"] = "smb-test-999999"
        rows[5]["item_id"] = "smb-test-999999"
    elif mutation == "pending":
        rows[5]["review_status"] = "pending"
    elif mutation == "missing_reviewer":
        rows[5]["reviewer"] = ""
    elif mutation == "missing_rationale":
        rows[5]["rationale"] = ""
    elif mutation == "bad_date":
        rows[5]["reviewed_at"] = "18/08/2026"
    elif mutation == "bad_enum":
        rows[5]["suitability_disposition"] = "maybe"
    elif mutation == "reversed_candidate":
        candidate["item_id"], candidate["candidate_item_id"] = (
            candidate["candidate_item_id"],
            candidate["item_id"],
        )
    else:
        rows[0]["duplicate_disposition"] = "duplicate"
        candidate["duplicate_disposition"] = "related"
    _write_review(review, rows)

    with pytest.raises(smb_audit.ReviewFinalizationError, match=match):
        smb_audit.validate_review_from_active_manifest(
            review_path=review,
            active_path=active_path,
            generation_root=generation_root,
        )


def test_apply_review_dispositions_preserves_denominator_and_updates_candidate_pair(
    tmp_path: Path,
) -> None:
    _, _, _, _, _, review = _emit_review_bundle(tmp_path)
    review_rows = _complete_review(review)
    original_rows = _full_rows()
    candidate = next(row for row in review_rows if row["review_kind"] == "candidate")

    updated = smb_audit.apply_review_dispositions(original_rows, review_rows)

    assert len(updated) == 685
    assert [row["item_id"] for row in updated] == [row["item_id"] for row in original_rows]
    assert updated is not original_rows
    assert original_rows[0]["quality"]["review_status"] == "pending"
    by_id = {row["item_id"]: row for row in updated}
    for item_id in (candidate["item_id"], candidate["candidate_item_id"]):
        assert by_id[item_id]["duplicate_review"] == {
            "review_status": "reviewed",
            "disposition": "distinct",
            "reviewer": "reviewer-1",
            "reviewed_at": "2026-08-18",
            "rationale": "Reviewed from the frozen audit evidence.",
        }
    assert all(row["quality"]["review_status"] == "reviewed" for row in updated)


def test_finalize_review_resolves_then_publishes_all_rows_idempotently(tmp_path: Path) -> None:
    active_path, generation_root, _, _, _, review = _emit_review_bundle(tmp_path)
    _complete_review(review)
    old_pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))

    smb_audit.finalize_reviewed_manifest(
        review_path=review,
        active_path=active_path,
        generation_root=generation_root,
    )
    first_pointer_bytes = active_path.read_bytes()
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )

    assert descriptor["benchmark_state"] == "AUDITED_LOCKED"
    assert descriptor["generation_id"] != old_pointer["generation_id"]
    assert len(rows) == 685
    assert all(row["quality"]["review_status"] == "reviewed" for row in rows)
    smb_audit.finalize_reviewed_manifest(
        review_path=review,
        active_path=active_path,
        generation_root=generation_root,
    )
    assert active_path.read_bytes() == first_pointer_bytes


def test_cli_review_commands_expose_only_pointer_based_manifest_inputs(tmp_path: Path) -> None:
    active_path, generation_root, audit_descriptor, audit_records, sample, review = (
        _emit_review_bundle(tmp_path)
    )
    _complete_review(review)
    assert (
        smb_audit.main(
            [
                "validate-review",
                "--review",
                str(review),
                "--manifest-active",
                str(active_path),
                "--manifest-generation-root",
                str(generation_root),
            ]
        )
        == 0
    )
    prepared = tmp_path / "prepared"
    assert (
        smb_audit.main(
            [
                "prepare-review",
                "--manifest-active",
                str(active_path),
                "--manifest-generation-root",
                str(generation_root),
                "--audit-descriptor",
                str(prepared / audit_descriptor.name),
                "--audit-records",
                str(prepared / audit_records.name),
                "--sample",
                str(prepared / sample.name),
                "--review",
                str(prepared / review.name),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit):
        smb_audit.main(
            [
                "validate-review",
                "--review",
                str(review),
                "--manifest-descriptor",
                "forbidden.yaml",
                "--manifest-records",
                "forbidden.jsonl",
            ]
        )


def test_repository_readers_are_confined_to_validated_active_resolution() -> None:
    module_path = Path(smb_audit.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_path_owners = {
        "_validate_generation_inputs",
        "publish_manifest_generation",
        "resolve_active_manifest",
    }
    forbidden_scans = {"glob", "rglob", "iterdir", "walk", "scandir"}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in allowed_path_owners:
            source = ast.get_source_segment(module_path.read_text(encoding="utf-8"), node) or ""
            assert "manifest-descriptor.yaml" not in source
            assert "manifest-records.jsonl" not in source
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in forbidden_scans:
                pytest.fail(f"{node.name} scans generation directories via {child.attr}")

    for name in (
        "emit_review_evidence_from_active_manifest",
        "validate_review_from_active_manifest",
        "finalize_reviewed_manifest",
        "reconcile_manifest",
    ):
        parameters = inspect.signature(getattr(smb_audit, name)).parameters
        assert not {
            "descriptor_path",
            "records_path",
            "manifest_path",
            "jsonl_path",
        }.intersection(parameters)


def _changed_rows() -> list[dict[str, Any]]:
    rows = _full_rows()
    rows[0]["quality"]["notes"] = "new-generation"
    return rows


def _generation_files(active_path: Path, generation_root: Path) -> dict[str, bytes]:
    pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    generation = generation_root / pointer["generation_id"]
    return {
        path.name: path.read_bytes()
        for path in (
            generation / "manifest-descriptor.yaml",
            generation / "manifest-records.jsonl",
        )
    }


def test_publication_boundaries_are_ordered_before_and_after_one_pointer_commit(
    tmp_path: Path,
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    events: list[str] = []

    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=_changed_rows(),
        boundary_hook=events.append,
    )

    assert tuple(events) == PUBLICATION_BOUNDARIES
    assert tuple(smb_audit.PUBLICATION_BOUNDARIES) == PUBLICATION_BOUNDARIES
    assert events.index("generation_descriptor_fsynced") < events.index("generation_renamed")
    assert events.index("generations_parent_fsynced") < events.index("pointer_written")
    assert events.index("pointer_fsynced") < events.index("pointer_replaced")
    assert events[-1] == "active_parent_fsynced"


@pytest.mark.parametrize("boundary", PUBLICATION_BOUNDARIES)
def test_ordinary_failure_at_every_boundary_preserves_complete_old_or_new(
    tmp_path: Path, boundary: str
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    old_files = _generation_files(active_path, generation_root)

    def failpoint(observed: str) -> None:
        if observed == boundary:
            raise OSError("injected ordinary failure")

    with pytest.raises(smb_audit.ManifestPublicationError, match="publication") as caught:
        smb_audit.publish_manifest_generation(
            active_path=active_path,
            generation_root=generation_root,
            descriptor=_descriptor(),
            rows=_changed_rows(),
            boundary_hook=failpoint,
        )

    assert caught.value.committed is (boundary in COMMITTED_BOUNDARIES)
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    if boundary in COMMITTED_BOUNDARIES:
        assert descriptor["generation_id"] != old_pointer["generation_id"]
        assert rows[0]["quality"]["notes"] == "new-generation"
    else:
        assert descriptor["generation_id"] == old_pointer["generation_id"]
        assert rows[0]["quality"]["notes"] == ""
    old_generation = generation_root / old_pointer["generation_id"]
    assert {
        path.name: path.read_bytes()
        for path in (
            old_generation / "manifest-descriptor.yaml",
            old_generation / "manifest-records.jsonl",
        )
    } == old_files


@pytest.mark.parametrize("boundary", PUBLICATION_BOUNDARIES)
def test_abrupt_subprocess_exit_at_every_boundary_leaves_complete_old_or_new(
    tmp_path: Path, boundary: str
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    descriptor_path = tmp_path / "descriptor.json"
    rows_path = tmp_path / "rows.json"
    descriptor_path.write_text(json.dumps(_descriptor()), encoding="utf-8")
    rows_path.write_text(json.dumps(_changed_rows()), encoding="utf-8")
    code = """
import json
import sys
from pathlib import Path
from score_super_resolution.smb_audit import publish_manifest_generation

publish_manifest_generation(
    active_path=Path(sys.argv[1]),
    generation_root=Path(sys.argv[2]),
    descriptor=json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')),
    rows=json.loads(Path(sys.argv[4]).read_text(encoding='utf-8')),
)
"""
    environment = os.environ.copy()
    environment["SCORE_SR_SMB_PUBLICATION_FAILPOINT"] = f"{boundary}:exit"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(active_path),
            str(generation_root),
            str(descriptor_path),
            str(rows_path),
        ],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 91, completed.stderr
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    if boundary in COMMITTED_BOUNDARIES:
        assert descriptor["generation_id"] != old_pointer["generation_id"]
        assert rows[0]["quality"]["notes"] == "new-generation"
    else:
        assert descriptor["generation_id"] == old_pointer["generation_id"]
        assert rows[0]["quality"]["notes"] == ""


def test_concurrent_readers_observe_only_validated_old_or_new_generations(
    tmp_path: Path,
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_id = yaml.safe_load(active_path.read_text(encoding="utf-8"))["generation_id"]
    pointer_ready = threading.Event()
    continue_publication = threading.Event()
    stop_readers = threading.Event()
    observations: list[tuple[str, str]] = []
    failures: list[BaseException] = []

    def hook(boundary: str) -> None:
        if boundary == "pointer_written":
            pointer_ready.set()
            assert continue_publication.wait(timeout=10)

    def publish() -> None:
        try:
            smb_audit.publish_manifest_generation(
                active_path=active_path,
                generation_root=generation_root,
                descriptor=_descriptor(),
                rows=_changed_rows(),
                boundary_hook=hook,
            )
        except BaseException as error:  # pragma: no cover - reported below
            failures.append(error)

    def read() -> None:
        try:
            while not stop_readers.is_set():
                descriptor, rows = smb_audit.resolve_active_manifest(
                    active_path=active_path, generation_root=generation_root
                )
                observations.append(
                    (str(descriptor["generation_id"]), str(rows[0]["quality"]["notes"]))
                )
        except BaseException as error:  # pragma: no cover - reported below
            failures.append(error)

    readers = [threading.Thread(target=read) for _ in range(4)]
    for reader in readers:
        reader.start()
    writer = threading.Thread(target=publish)
    writer.start()
    assert pointer_ready.wait(timeout=10)
    continue_publication.set()
    writer.join(timeout=10)
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    new_id = str(descriptor["generation_id"])
    observations.append((new_id, str(rows[0]["quality"]["notes"])))
    stop_readers.set()
    for reader in readers:
        reader.join(timeout=10)

    assert not failures
    assert old_id != new_id
    assert observations
    assert set(observations) <= {(old_id, ""), (new_id, "new-generation")}
    assert (old_id, "") in observations
    assert (new_id, "new-generation") in observations


def test_identical_retry_is_noop_and_mismatched_existing_generation_fails(
    tmp_path: Path,
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    before_bytes = active_path.read_bytes()
    before_stat = active_path.stat()
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=_full_rows(),
    )
    assert active_path.read_bytes() == before_bytes
    assert active_path.stat().st_mtime_ns == before_stat.st_mtime_ns

    _, pointer, _, _ = smb_audit._validate_generation_inputs(_descriptor(), _changed_rows())
    mismatched = generation_root / str(pointer["generation_id"])
    mismatched.mkdir()
    (mismatched / "manifest-descriptor.yaml").write_text("wrong: bytes\n", encoding="utf-8")
    (mismatched / "manifest-records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(smb_audit.ManifestPublicationError, match="not byte-identical") as caught:
        smb_audit.publish_manifest_generation(
            active_path=active_path,
            generation_root=generation_root,
            descriptor=_descriptor(),
            rows=_changed_rows(),
        )
    assert caught.value.committed is False
    assert active_path.read_bytes() == before_bytes


def test_partial_and_unreferenced_generations_are_never_inferred_as_active(tmp_path: Path) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    for name in (".tmp-newest", "f" * 64):
        unreachable = generation_root / name
        unreachable.mkdir()
        (unreachable / "manifest-descriptor.yaml").write_text("newer: false\n", encoding="utf-8")

    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )

    assert descriptor["generation_id"] == active["generation_id"]
    assert len(rows) == 685
