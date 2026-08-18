from __future__ import annotations

import base64
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import score_super_resolution.smb_audit as smb_audit
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.smb_audit import (
    ManifestPublicationError,
    audit_dataset,
    audit_item,
    publish_manifest_generation,
    reconcile_manifest,
    resolve_active_manifest,
    select_visual_sample,
    write_review_csv,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "smb" / "records.json"
SOURCE_PATH = Path(__file__).parents[1] / "data" / "sources" / "smb.yaml"


def _fixtures() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _source_descriptor() -> dict[str, Any]:
    return yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))


def _audit_records() -> list[dict[str, Any]]:
    records = copy.deepcopy(_fixtures()["audit_cases"])
    for record in records:
        record["image"] = base64.b64decode(record.pop("image_base64"))
    return records


def _full_rows() -> list[dict[str, Any]]:
    template = _fixtures()["normal_row"]
    rows = []
    for index in range(685):
        row = copy.deepcopy(template)
        row["upstream_index"] = index
        row["item_id"] = f"smb-test-{index:06d}"
        row["original_score"] = {"raw": f"score-{index:03d}", "normalized": f"score-{index:03d}"}
        row["source_group_id"] = f"score-{index:03d}"
        rows.append(row)
    return rows


def _descriptor() -> dict[str, Any]:
    return copy.deepcopy(_fixtures()["manifest_descriptor"])


def test_audit_guard_runs_before_decode_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def guard(**kwargs: object) -> object:
        events.append("guard")
        callback = kwargs["callback"]
        assert callable(callback)
        return callback()

    original_open = smb_audit.Image.open

    def open_after_guard(*args: object, **kwargs: object) -> object:
        assert events == ["guard"]
        events.append("decode")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(smb_audit, "assert_smb_purpose_allowed", guard)
    monkeypatch.setattr(smb_audit.Image, "open", open_after_guard)

    record = _audit_records()[0]
    row = audit_item(record, upstream_index=0, source_descriptor=_source_descriptor())

    assert events[:2] == ["guard", "decode"]
    assert row["processing_status"] == "processed"
    assert row["encoded_sha256"] == hashlib.sha256(record["image"]).hexdigest()


def test_fixture_audit_preserves_every_input_and_candidate_state() -> None:
    records = _audit_records()
    rows = audit_dataset(
        records, source_descriptor=_source_descriptor(), sample_size=3, max_pixels=2048
    )

    assert len(rows) == len(records)
    assert [row["upstream_index"] for row in rows] == list(range(len(records)))
    assert all(row["item_id"] == f"smb-test-{index:06d}" for index, row in enumerate(rows))
    assert rows[4]["processing_status"] == "failed"
    assert rows[4]["unprocessable_reason"] == "decode_failed"
    assert rows[5]["processing_status"] == "failed"
    assert rows[5]["unprocessable_reason"] == "image_too_large"
    assert rows[6]["processing_status"] == "processed"
    assert rows[6]["required_text_present"] is False
    assert rows[6]["paired_eligible"] is False
    assert rows[7]["item_id"] == "smb-test-000007"
    assert "unsafe_upstream_id" in rows[7]["annotation_failures"]
    assert rows[7]["bbox_valid"] is False
    assert rows[0]["source_group_id"] == rows[3]["source_group_id"]

    assert rows[0]["exact_duplicate_group"] == rows[1]["exact_duplicate_group"]
    assert rows[0]["exact_duplicate_group"] is not None
    shared_candidates = set(rows[0]["near_duplicate_candidate_ids"]) & set(
        rows[2]["near_duplicate_candidate_ids"]
    )
    assert len(shared_candidates) == 1
    assert rows[0]["duplicate_review"]["review_status"] == "pending"

    for row in rows:
        validate_instance("manifest-row", row)


def test_audit_accepts_smb_xywh_region_mapping() -> None:
    record = _audit_records()[0]
    record["regions"] = [{"bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}}]

    row = audit_item(record, upstream_index=0, source_descriptor=_source_descriptor())

    assert row["region_count"] == 1
    assert row["bbox_valid"] is True
    assert row["paired_eligible"] is True
    assert row["paired_ineligibility_reason"] is None


def test_visual_sample_is_exact_deterministic_and_outcome_independent() -> None:
    identities = [
        {"upstream_index": index, "item_id": f"smb-test-{index:06d}"} for index in range(685)
    ]
    selected = select_visual_sample(identities, seed=20260818)
    changed_outcomes = [dict(row, processing_status="failed", metric=999) for row in identities]

    assert len(selected) == 64
    assert len(set(selected)) == 64
    assert selected == select_visual_sample(identities, seed=20260818)
    assert selected == select_visual_sample(changed_outcomes, seed=20260818)
    assert selected != select_visual_sample(identities, seed=20260819)


def test_publication_is_content_addressed_and_resolves_only_active_generation(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "data" / "manifests" / "smb-evaluation-v1.yaml"
    generation_root = tmp_path / "artifacts" / "smb-manifests" / "generations"
    rows = _full_rows()

    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=rows,
    )
    pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    validate_instance("manifest-active", pointer)
    assert list(generation_root.iterdir()) == [generation_root / pointer["generation_id"]]

    descriptor, resolved_rows = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    assert descriptor["generation_id"] == pointer["generation_id"]
    assert descriptor["row_count"] == len(resolved_rows) == 685
    assert descriptor["records_sha256"] == pointer["records_sha256"]
    assert descriptor["benchmark_state"] == "AUDITED_LOCKED"

    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=rows,
    )
    assert yaml.safe_load(active_path.read_text(encoding="utf-8")) == pointer

    unreferenced = generation_root / ("f" * 64)
    unreferenced.mkdir()
    (unreferenced / "manifest-descriptor.yaml").write_text("not: active\n", encoding="utf-8")
    resolved_again, _ = resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    assert resolved_again["generation_id"] == pointer["generation_id"]


@pytest.mark.parametrize("mutation", ("missing", "unknown"))
def test_publication_validates_every_row_before_writing(tmp_path: Path, mutation: str) -> None:
    rows = _full_rows()
    if mutation == "missing":
        del rows[300]["processing_status"]
    else:
        rows[300]["unknown"] = True

    with pytest.raises(ManifestPublicationError, match="manifest-row") as caught:
        publish_manifest_generation(
            active_path=tmp_path / "active.yaml",
            generation_root=tmp_path / "generations",
            descriptor=_descriptor(),
            rows=rows,
        )

    assert caught.value.committed is False
    assert not (tmp_path / "active.yaml").exists()
    assert not (tmp_path / "generations").exists()


def test_manifest_checksum_and_generation_change_when_a_row_changes(tmp_path: Path) -> None:
    rows = _full_rows()
    first_active = tmp_path / "first.yaml"
    second_active = tmp_path / "second.yaml"
    generation_root = tmp_path / "generations"
    publish_manifest_generation(
        active_path=first_active,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=rows,
    )
    changed = copy.deepcopy(rows)
    changed[0]["quality"]["notes"] = "changed"
    publish_manifest_generation(
        active_path=second_active,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=changed,
    )

    first = yaml.safe_load(first_active.read_text(encoding="utf-8"))
    second = yaml.safe_load(second_active.read_text(encoding="utf-8"))
    assert first["records_sha256"] != second["records_sha256"]
    assert first["generation_id"] != second["generation_id"]


def test_publication_rejects_inaccurate_imagehash_provenance(tmp_path: Path) -> None:
    descriptor = _descriptor()
    descriptor["duplicate_provenance"]["near"]["library_version"] = "9.9.9"

    with pytest.raises(ManifestPublicationError, match="ImageHash provenance"):
        publish_manifest_generation(
            active_path=tmp_path / "active.yaml",
            generation_root=tmp_path / "generations",
            descriptor=descriptor,
            rows=_full_rows(),
        )


@pytest.mark.parametrize("target", ("pointer", "descriptor", "row"))
@pytest.mark.parametrize("mutation", ("missing", "unknown"))
def test_resolution_revalidates_every_contract(tmp_path: Path, target: str, mutation: str) -> None:
    active_path = tmp_path / "active.yaml"
    generation_root = tmp_path / "generations"
    publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=_full_rows(),
    )
    pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    generation = generation_root / pointer["generation_id"]
    if target == "pointer":
        document = pointer
        path = active_path
        field = "record_type"
    elif target == "descriptor":
        path = generation / "manifest-descriptor.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        field = "audit_version"
    else:
        path = generation / "manifest-records.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        document = json.loads(lines[0])
        field = "processing_status"
    if mutation == "missing":
        del document[field]
    else:
        document["unknown"] = True
    if target == "row":
        lines[0] = json.dumps(document, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

    with pytest.raises((ManifestPublicationError, ContractValidationError)):
        resolve_active_manifest(active_path=active_path, generation_root=generation_root)


def test_reconciliation_retains_failures_and_rejects_index_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _full_rows()
    rows[5] = copy.deepcopy(_fixtures()["failure_row"])
    rows[5]["upstream_index"] = 5
    rows[5]["item_id"] = "smb-test-000005"
    descriptor = _descriptor()
    monkeypatch.setattr(smb_audit, "resolve_active_manifest", lambda **_: (descriptor, rows))

    report = reconcile_manifest(active_path=Path("unused"), generation_root=Path("unused"))
    assert report == {"row_count": 685, "processed": 684, "failed": 1, "paired_eligible": 684}

    rows[-1]["upstream_index"] = 683
    with pytest.raises(ManifestPublicationError, match=r"duplicate.*missing"):
        reconcile_manifest(active_path=Path("unused"), generation_root=Path("unused"))


def test_review_csv_is_resolved_from_pointer_and_formula_neutralized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _full_rows()
    rows[0]["duplicate_review"]["rationale"] = "=untrusted display text"
    descriptor = _descriptor()
    calls = []

    def resolved(**kwargs: object) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        calls.append(kwargs)
        return descriptor, rows

    monkeypatch.setattr(smb_audit, "resolve_active_manifest", resolved)
    output = tmp_path / "smb-review-v1.csv"
    write_review_csv(
        active_path=tmp_path / "does-not-exist.yaml",
        generation_root=tmp_path / "does-not-exist",
        output_path=output,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert len(calls) == 1
    assert review_rows[0]["item_id"] == "smb-test-000000"
    assert review_rows[0]["review_key"] == "smb-test-000000"
    assert review_rows[0]["rationale"].startswith("'=untrusted")
    assert tuple(review_rows[0]) == smb_audit.REVIEW_CSV_FIELDS


def test_audit_module_has_no_outcome_producing_imports() -> None:
    source = Path(smb_audit.__file__).read_text(encoding="utf-8")
    forbidden = ("degradation", "inference", "model", "metric", "outcome_ranking")
    assert all(f"score_super_resolution.{name}" not in source for name in forbidden)


def test_authenticated_audit_orchestrates_exact_revision_and_redacted_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_descriptor = _source_descriptor()
    loaded: list[tuple[str, str, str]] = []
    records = [object()] * 685
    rows = _full_rows()
    for row in rows:
        row["source_revision"] = source_descriptor["revision"]
    for row in rows[:64]:
        row["audit_sample_member"] = True
    audited: list[tuple[object, dict[str, Any], int, int]] = []

    def loader(repository_id: str, *, split: str, revision: str) -> object:
        loaded.append((repository_id, split, revision))
        return records

    def audit(
        received: object,
        *,
        source_descriptor: dict[str, Any],
        deterministic_seed: int,
        sample_size: int,
    ) -> list[dict[str, Any]]:
        audited.append((received, source_descriptor, deterministic_seed, sample_size))
        return rows

    monkeypatch.setattr(smb_audit, "audit_dataset", audit)
    monkeypatch.setattr(smb_audit, "_current_code_revision", lambda: "a" * 40)
    audit_descriptor = tmp_path / "data" / "audits" / "smb-audit-v1.yaml"
    audit_records = tmp_path / "data" / "audits" / "smb-audit-v1.jsonl"
    sample = tmp_path / "data" / "audits" / "smb-visual-sample-v1.csv"
    review = tmp_path / "data" / "audits" / "smb-review-v1.csv"
    active = tmp_path / "data" / "manifests" / "smb-evaluation-v1.yaml"
    generations = tmp_path / "artifacts" / "smb-manifests" / "generations"

    report = smb_audit.run_authenticated_audit(
        source_path=SOURCE_PATH,
        audit_descriptor_path=audit_descriptor,
        audit_records_path=audit_records,
        sample_path=sample,
        review_path=review,
        active_path=active,
        generation_root=generations,
        dataset_loader=loader,
    )

    assert loaded == [("PRAIG/SMB", "test", source_descriptor["revision"])]
    assert audited == [(records, source_descriptor, 20260818, 64)]
    assert report == {"row_count": 685, "processed": 685, "failed": 0, "paired_eligible": 685}
    descriptor, resolved = resolve_active_manifest(active_path=active, generation_root=generations)
    assert descriptor["code_revision"] == "a" * 40
    assert descriptor["source_revision"] == source_descriptor["revision"]
    assert descriptor["benchmark_state"] == "AUDITED_LOCKED"
    assert len(resolved) == 685
    exported = [json.loads(line) for line in audit_records.read_text().splitlines()]
    assert len(exported) == 685
    assert set(exported[0]) == {
        "audit_sample_member",
        "item_id",
        "processing_status",
        "source_group_id",
        "upstream_index",
    }
    assert len(list(csv.DictReader(sample.open(encoding="utf-8", newline="")))) == 64
    assert len(list(csv.DictReader(review.open(encoding="utf-8", newline="")))) == 685


def test_authenticated_audit_rejects_wrong_upstream_count_before_auditing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        smb_audit,
        "audit_dataset",
        lambda *_args, **_kwargs: pytest.fail("audit must not run for a wrong source count"),
    )

    with pytest.raises(ValueError, match="exactly 685"):
        smb_audit.run_authenticated_audit(
            source_path=SOURCE_PATH,
            audit_descriptor_path=tmp_path / "audit.yaml",
            audit_records_path=tmp_path / "audit.jsonl",
            sample_path=tmp_path / "sample.csv",
            review_path=tmp_path / "review.csv",
            active_path=tmp_path / "active.yaml",
            generation_root=tmp_path / "generations",
            dataset_loader=lambda *_args, **_kwargs: [object()] * 684,
        )

    assert not (tmp_path / "active.yaml").exists()


def test_audit_cli_exposes_the_exact_controlled_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        smb_audit,
        "run_authenticated_audit",
        lambda **kwargs: (
            calls.append(kwargs)
            or {"row_count": 685, "processed": 685, "failed": 0, "paired_eligible": 685}
        ),
    )
    arguments = [
        "audit",
        "--source",
        str(SOURCE_PATH),
        "--audit-descriptor",
        str(tmp_path / "audit.yaml"),
        "--audit-records",
        str(tmp_path / "audit.jsonl"),
        "--sample",
        str(tmp_path / "sample.csv"),
        "--review",
        str(tmp_path / "review.csv"),
        "--manifest-active",
        str(tmp_path / "active.yaml"),
        "--manifest-generation-root",
        str(tmp_path / "generations"),
    ]

    assert smb_audit.main(arguments) == 0
    assert calls == [
        {
            "source_path": SOURCE_PATH,
            "audit_descriptor_path": tmp_path / "audit.yaml",
            "audit_records_path": tmp_path / "audit.jsonl",
            "sample_path": tmp_path / "sample.csv",
            "review_path": tmp_path / "review.csv",
            "active_path": tmp_path / "active.yaml",
            "generation_root": tmp_path / "generations",
        }
    ]
