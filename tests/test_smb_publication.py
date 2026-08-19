from __future__ import annotations

import ast
import copy
import csv
import fcntl
import gzip
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, PngImagePlugin

import score_super_resolution.smb_audit as smb_audit
from score_super_resolution.contracts import (
    load_schema,
    recovery_metadata_sha256,
    validate_instance,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "smb" / "records.json"
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
COMMITTED_BOUNDARIES = {"pointer_replaced", "active_parent_fsynced"}


def _fixtures() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _png_bytes(
    image: Image.Image,
    *,
    compress_level: int = 6,
    comment: str | None = None,
    exif_orientation: int | None = None,
) -> bytes:
    output = io.BytesIO()
    pnginfo = None
    if comment is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("Comment", comment)
    exif = None
    if exif_orientation is not None:
        exif = image.getexif()
        exif[274] = exif_orientation
    image.save(
        output,
        format="PNG",
        compress_level=compress_level,
        pnginfo=pnginfo,
        exif=exif,
    )
    return output.getvalue()


def _audit_record(encoded: bytes, *, index: int = 0) -> dict[str, Any]:
    return {
        "id": f"upstream-{index}",
        "image": encoded,
        "original_width": 2,
        "original_height": 2,
        "regions": [{"bbox": {"x": 0, "y": 0, "width": 1, "height": 1}}],
        "original_score": f"score-{index}_p0",
        "page": str(index),
        "page_texture": "clean",
    }


def _audit_v2_images(
    *encoded_images: bytes,
    max_encoded_bytes: int = smb_audit.DEFAULT_MAX_ENCODED_BYTES,
    max_pixels: int = smb_audit.DEFAULT_MAX_PIXELS,
) -> list[dict[str, Any]]:
    descriptor = yaml.safe_load(
        (Path(__file__).parents[1] / "data" / "sources" / "smb.yaml").read_text(encoding="utf-8")
    )
    return smb_audit.audit_dataset_v2(
        [_audit_record(encoded, index=index) for index, encoded in enumerate(encoded_images)],
        source_descriptor=descriptor,
        trusted_cache_roots=(Path.cwd(),),
        sample_size=0,
        max_encoded_bytes=max_encoded_bytes,
        max_pixels=max_pixels,
    )


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


def _compact_v2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(685):
        item_id = f"smb-test-{index:06d}"
        encoded = hashlib.sha256(f"encoded-{index}".encode()).hexdigest()
        pixels = hashlib.sha256(f"pixels-{index}".encode()).hexdigest()
        rows.append(
            {
                "schema_version": 2,
                "record_type": "manifest-row",
                "manifest_version": 2,
                "source_key": "smb",
                "source_revision": "a" * 40,
                "split": "test",
                "upstream_index": index,
                "item_id": item_id,
                "source_identity": {
                    "original_score_normalized": f"score-{index:03d}_p0",
                    "original_score_raw_sha256": hashlib.sha256(
                        f"score-{index:03d}".encode()
                    ).hexdigest(),
                    "page_normalized": str(index),
                    "page_raw_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                    "page_texture_normalized": "clean",
                    "page_texture_raw_sha256": hashlib.sha256(b"clean").hexdigest(),
                },
                "source_group_id": f"score-{index:03d}",
                "image": {
                    "encoded_sha256": encoded,
                    "pixel_sha256": pixels,
                    "declared_width": 32,
                    "declared_height": 24,
                    "decoded_width": 32,
                    "decoded_height": 24,
                    "mode": "RGB",
                    "format": "PNG",
                    "byte_count": 128,
                },
                "annotations": {
                    "region_count": 1,
                    "bbox_valid": True,
                    "required_text_present": True,
                    "failures": [],
                },
                "automated_audit": {
                    "status": "automated",
                    "algorithm_version": "smb-audit-v2",
                    "quality_flags": [],
                },
                "visual_review": {"status": "not_visually_reviewed"},
                "audit_sample_member": False,
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
                "expected_status": "processable",
                "processing_status": "processed",
                "unprocessable_reason": None,
                "rights": {
                    "dataset_licence": {
                        "status": "confirmed",
                        "identifier": "CC-BY-NC-4.0",
                        "reference": "https://creativecommons.org/licenses/by-nc/4.0/",
                    },
                    "item_provenance": {
                        "status": "unavailable",
                        "reason": "No per-item source chain is available.",
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
                "paired_eligible": True,
                "paired_ineligibility_reason": None,
            }
        )
    selected = set(smb_audit.select_visual_sample(rows, seed=5, sample_size=64))
    for row in rows:
        sampled = row["item_id"] in selected
        row["audit_sample_member"] = sampled
        if sampled:
            row["visual_review"] = {
                "status": "sampled_human_reviewed",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-08-18",
                "rationale": "Reviewed in the frozen sample.",
                "quality_flags": [],
                "suitability": "suitable",
            }
    return rows


def _compact_v2_descriptor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_records = {
        relation["pair_id"]: relation for row in rows for relation in row["duplicate_relations"]
    }
    exact_count = sum(r["candidate_type"] == "exact" for r in pair_records.values())
    perceptual = [r for r in pair_records.values() if r["candidate_type"] == "perceptual"]
    return {
        "schema_version": 2,
        "record_type": "manifest-descriptor",
        "manifest_id": "smb-evaluation-v2",
        "generation_id": "0" * 64,
        "generation_algorithm": {
            "algorithm": "sha256",
            "version": 2,
            "domain_separator": "smb-manifest-generation-v2",
            "descriptor_canonicalization": "yaml-safe-sort-keys-utf8-v1",
            "records_canonicalization": "jsonl-utf8-sorted-keys-v1",
        },
        "source_key": "smb",
        "source_revision": "a" * 40,
        "creation_command": "uv run python -m score_super_resolution.smb_audit audit",
        "source_provenance": {
            "source_set_version": 1,
            "algorithm": "sha256",
            "revision": "b" * 40,
            "dirty": False,
            "source_tree_sha256": "7" * 64,
            "patch_sha256": "8" * 64,
            "lock_sha256": "9" * 64,
        },
        "grouping_unit": "source_score",
        "upstream_split": "test",
        "project_split": "evaluation",
        "deterministic_seed": 5,
        "exclusions": [],
        "row_schema_id": "manifest-row",
        "row_schema_version": 2,
        "row_count": 685,
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
                "decoder_version": "12.3.0",
                "output_mode": "RGBA8",
                "alpha_policy": "retain-alpha-and-underlying-rgb",
                "orientation_policy": "stored-raster-ignore-exif",
                "metadata_policy": "ignore-non-raster-metadata",
                "max_encoded_bytes": 67_108_864,
                "max_pixels": 100_000_000,
                "failure_policy": "safe-explicit-failure-no-digest",
            },
        },
        "duplicate_provenance": {
            "exact": {"algorithm": "canonical-pixel-sha256", "version": 2},
            "near": {
                "algorithm": "phash",
                "version": 1,
                "library": "ImageHash",
                "library_version": smb_audit.IMAGEHASH_VERSION,
                "hash_size": 8,
                "highfreq_factor": 4,
                "maximum_hamming_distance": 6,
            },
        },
        "sample_selection": {
            "algorithm": "sha256-rank",
            "version": 1,
            "seed": 5,
            "population_size": 685,
            "sample_size": 64,
            "identity_fields": ["upstream_index", "item_id"],
            "selection_state": "pre-review",
        },
        "review_inference": {
            "automated_population_audit_count": 685,
            "sampled_human_review_count": 64,
            "targeted_human_review_count": 0,
            "not_visually_reviewed_count": 621,
            "unavailable_visual_review_count": 0,
            "not_applicable_visual_review_count": 0,
            "exact_pair_automated_count": exact_count,
            "perceptual_pair_count": len(perceptual),
            "perceptual_pair_human_review_count": sum(
                r["disposition"] not in {"pending", "unavailable"} for r in perceptual
            ),
            "perceptual_pair_pending_count": sum(r["disposition"] == "pending" for r in perceptual),
            "inference_scope": "sample_observation_only",
            "population_prevalence_inference": "not_supported",
        },
    }


def _perceptual_relation(
    first: str, second: str, *, disposition: str, distance: int = 3
) -> dict[str, Any]:
    pair_id = smb_audit._pair_id("perceptual", first, second)
    reviewed = disposition not in {"pending", "unavailable"}
    return {
        "pair_id": pair_id,
        "candidate_type": "perceptual",
        "item_ids": sorted((first, second)),
        "counterpart_item_id": second,
        "evidence_basis": (
            "perceptual_hash_plus_human_review" if reviewed else "perceptual_hash_candidate"
        ),
        "evidence": {"algorithm": "phash", "version": 1, "distance": distance},
        "disposition": disposition,
        "reviewer": "reviewer-1" if reviewed else None,
        "reviewed_at": "2026-08-18" if reviewed else None,
        "rationale": (
            "Independent pair adjudication."
            if reviewed
            else "Perceptual comparison evidence is unavailable."
            if disposition == "unavailable"
            else ""
        ),
    }


def _attach_relation(
    rows: list[dict[str, Any]], first_index: int, second_index: int, *, disposition: str
) -> None:
    first = rows[first_index]
    second = rows[second_index]
    relation = _perceptual_relation(
        str(first["item_id"]), str(second["item_id"]), disposition=disposition
    )
    mirrored = copy.deepcopy(relation)
    mirrored["counterpart_item_id"] = first["item_id"]
    first["duplicate_relations"].append(relation)
    second["duplicate_relations"].append(mirrored)
    for row in (first, second):
        row["near_duplicate_candidate_ids"].append(relation["pair_id"])
        row["near_duplicate_candidate_ids"].sort()
        row["duplicate_relations"].sort(key=lambda value: value["pair_id"])
        summary = row["duplicate_summary"]
        summary["perceptual_relation_count"] += 1
        summary[f"{disposition}_relation_count"] += 1


def _completed_v2_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    review_rows = smb_audit.prepare_v2_review_rows(rows)
    for row in review_rows:
        row["review_status"] = "reviewed"
        row["reviewer"] = "reviewer-1"
        row["reviewed_at"] = "2026-08-18"
        row["rationale"] = "Reviewed without changing automated identity evidence."
        if row["review_kind"] == "item_policy":
            row["dataset_licence_status"] = "confirmed"
            row["item_provenance_status"] = "unavailable"
            row["access_status"] = "confirmed"
            row["redistribution_status"] = "not_established"
            row["figure_reproduction_status"] = "prohibited"
        elif row["review_kind"] == "visual_item":
            row["quality_disposition"] = "acceptable"
            row["suitability_disposition"] = "suitable"
        else:
            row["duplicate_disposition"] = "distinct"
    return review_rows


def _unavailable_pair_review_fixture() -> tuple[
    list[dict[str, Any]], list[dict[str, str]], str, str
]:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 64, disposition="unavailable")
    pair_id = smb_audit._pair_id(
        "perceptual",
        str(rows[0]["item_id"]),
        str(rows[64]["item_id"]),
    )
    reason = "Perceptual comparison evidence is unavailable."
    review_rows = _completed_v2_review_rows(rows)
    pair_review = next(row for row in review_rows if row["review_key"] == pair_id)
    pair_review.update(
        {
            "review_status": "unavailable",
            "reviewer": "",
            "reviewed_at": "",
            "rationale": reason,
            "duplicate_disposition": "unavailable",
        }
    )
    return rows, review_rows, pair_id, reason


def _unavailable_pair_relations(rows: list[dict[str, Any]], pair_id: str) -> list[dict[str, Any]]:
    return [
        relation
        for row in rows
        for relation in row["duplicate_relations"]
        if relation["pair_id"] == pair_id
    ]


def test_v2_review_preparation_separates_population_sample_and_pair_evidence() -> None:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 64, disposition="pending")

    prepared = smb_audit.prepare_v2_review_rows(rows)

    kinds = Counter(row["review_kind"] for row in prepared)
    assert kinds == {"item_policy": 685, "visual_item": 64, "duplicate_pair": 1}
    expected_sample = set(smb_audit.select_visual_sample(rows, seed=5, sample_size=64))
    assert {
        row["item_id"] for row in prepared if row["review_kind"] == "visual_item"
    } == expected_sample
    unsampled = next(row["item_id"] for row in rows if row["item_id"] not in expected_sample)
    assert not any(
        row["review_kind"] == "visual_item" and row["item_id"] == unsampled for row in prepared
    )


def test_v2_unavailable_pair_round_trips_through_publication_and_empty_root_recovery(
    tmp_path: Path,
) -> None:
    rows, review_rows, pair_id, reason = _unavailable_pair_review_fixture()
    original_relations = _unavailable_pair_relations(rows, pair_id)
    assert len(original_relations) == 2
    assert original_relations[0]["evidence"] == original_relations[1]["evidence"]

    prepared = smb_audit.prepare_v2_review_rows(rows)
    prepared_pair = next(row for row in prepared if row["review_key"] == pair_id)
    assert prepared_pair == {
        **{field: "" for field in smb_audit.REVIEW_CSV_FIELDS},
        "review_kind": "duplicate_pair",
        "review_key": pair_id,
        "item_id": "smb-test-000000",
        "candidate_item_id": "smb-test-000064",
        "review_status": "unavailable",
        "rationale": reason,
        "duplicate_disposition": "unavailable",
    }
    smb_audit.canonical_review_csv(review_rows)

    finalized = smb_audit.apply_review_dispositions(rows, review_rows)
    finalized_relations = _unavailable_pair_relations(finalized, pair_id)
    assert finalized_relations == original_relations
    for relation in finalized_relations:
        assert relation["disposition"] == "unavailable"
        assert relation["reviewer"] is None
        assert relation["reviewed_at"] is None
        assert relation["rationale"] == reason
    descriptor = _compact_v2_descriptor(finalized)
    smb_audit.validate_v2_manifest_collection(descriptor, finalized)

    source_active = tmp_path / "source" / "data/manifests/smb-evaluation-v1.yaml"
    source_generations = tmp_path / "source-generations"
    smb_audit.publish_manifest_generation(
        active_path=source_active,
        generation_root=source_generations,
        descriptor=descriptor,
        rows=finalized,
    )
    published_descriptor, published_rows = smb_audit.resolve_active_manifest(
        active_path=source_active,
        generation_root=source_generations,
    )
    assert _unavailable_pair_relations(published_rows, pair_id) == original_relations

    recovery_descriptor = tmp_path / "recovery" / "manifest-recovery.yaml"
    recovery_records = tmp_path / "recovery" / "manifest-records.jsonl.gz"
    smb_audit.export_manifest_recovery(
        active_path=source_active,
        generation_root=source_generations,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
    )
    recovered_active = tmp_path / "recovered" / "data/manifests/smb-evaluation-v1.yaml"
    recovered_active.parent.mkdir(parents=True)
    recovered_active.write_bytes(source_active.read_bytes())
    recovered_generations = tmp_path / "recovered-generations"
    assert not recovered_generations.exists()
    smb_audit.recover_active_manifest(
        active_path=recovered_active,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
        generation_root=recovered_generations,
    )
    recovered_descriptor, recovered_rows = smb_audit.resolve_active_manifest(
        active_path=recovered_active,
        generation_root=recovered_generations,
    )

    assert recovered_descriptor == published_descriptor
    assert _unavailable_pair_relations(recovered_rows, pair_id) == original_relations
    for row in recovered_rows:
        validate_instance("manifest-row", row, version=2)
    smb_audit.validate_v2_manifest_collection(recovered_descriptor, recovered_rows)


@pytest.mark.parametrize(
    "mutation",
    (
        "reviewer",
        "reviewed_at",
        "missing_reason",
        "human_reviewed_unavailable",
        "visual_unavailable",
        "policy_unavailable",
    ),
)
def test_v2_unavailable_pair_review_state_rejects_attribution_or_state_confusion(
    mutation: str,
) -> None:
    rows, review_rows, pair_id, reason = _unavailable_pair_review_fixture()
    pair_review = next(row for row in review_rows if row["review_key"] == pair_id)
    if mutation == "reviewer":
        pair_review["reviewer"] = "fabricated-reviewer"
    elif mutation == "reviewed_at":
        pair_review["reviewed_at"] = "2026-08-19"
    elif mutation == "missing_reason":
        pair_review["rationale"] = ""
    elif mutation == "human_reviewed_unavailable":
        pair_review.update(
            {
                "review_status": "reviewed",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-08-19",
                "rationale": reason,
            }
        )
    else:
        review_kind = "visual_item" if mutation == "visual_unavailable" else "item_policy"
        wrong_kind = next(row for row in review_rows if row["review_kind"] == review_kind)
        wrong_kind.update(
            {
                "review_status": "unavailable",
                "reviewer": "",
                "reviewed_at": "",
                "rationale": reason,
            }
        )

    with pytest.raises(smb_audit.ReviewFinalizationError, match=r"unavailable|rationale"):
        smb_audit.apply_review_dispositions(rows, review_rows)


def test_v2_human_reviewed_pair_semantics_remain_attributed() -> None:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 64, disposition="pending")
    review_rows = _completed_v2_review_rows(rows)
    pair_review = next(row for row in review_rows if row["review_kind"] == "duplicate_pair")

    finalized = smb_audit.apply_review_dispositions(rows, review_rows)
    relations = _unavailable_pair_relations(finalized, pair_review["review_key"])

    assert len(relations) == 2
    assert all(relation["disposition"] == "distinct" for relation in relations)
    assert all(relation["reviewer"] == "reviewer-1" for relation in relations)
    assert all(relation["reviewed_at"] == "2026-08-18" for relation in relations)


def test_v2_exact_equality_generates_automatic_mirrored_duplicate_evidence() -> None:
    rows = _compact_v2_rows()
    rows[1]["image"]["encoded_sha256"] = rows[0]["image"]["encoded_sha256"]
    rows[1]["image"]["pixel_sha256"] = rows[0]["image"]["pixel_sha256"]

    derived = smb_audit.derive_v2_exact_relations(rows)

    pair = derived[0]["duplicate_relations"][0]
    assert pair["candidate_type"] == "exact"
    assert pair["evidence_basis"] == "canonical_pixel_sha256"
    assert pair["evidence"]["encoded_equality"] is True
    assert pair["disposition"] == "duplicate"
    assert pair["reviewer"] is None
    assert derived[1]["duplicate_relations"][0]["pair_id"] == pair["pair_id"]
    smb_audit.validate_v2_manifest_collection(_compact_v2_descriptor(derived), derived)


def test_v2_exact_relation_uses_canonical_pixels_across_png_reencoding() -> None:
    image = Image.new("RGBA", (2, 2), (12, 34, 56, 255))
    first = _png_bytes(image, compress_level=0, comment="first encoding")
    second = _png_bytes(image, compress_level=9, comment="second encoding")
    assert first != second
    assert hashlib.sha256(first).hexdigest() != hashlib.sha256(second).hexdigest()

    rows = _audit_v2_images(first, second)

    assert rows[0]["image"]["pixel_sha256"] == rows[1]["image"]["pixel_sha256"]
    for row, counterpart in ((rows[0], rows[1]), (rows[1], rows[0])):
        assert len(row["duplicate_relations"]) == 1
        relation = row["duplicate_relations"][0]
        assert relation["candidate_type"] == "exact"
        assert relation["counterpart_item_id"] == counterpart["item_id"]
        assert relation["evidence_basis"] == "canonical_pixel_sha256"
        assert relation["evidence"] == {
            "pixel_sha256": row["image"]["pixel_sha256"],
            "encoded_equality": False,
            "encoded_sha256": None,
        }
        assert row["near_duplicate_candidate_ids"] == []


def test_canonical_frame_binds_geometry_and_normalizes_declared_rgba_modes() -> None:
    rgba_stream = bytes(
        (
            10,
            20,
            30,
            255,
            40,
            50,
            60,
            255,
            70,
            80,
            90,
            255,
            100,
            110,
            120,
            255,
        )
    )
    one_by_four = _png_bytes(Image.frombytes("RGBA", (1, 4), rgba_stream))
    two_by_two = _png_bytes(Image.frombytes("RGBA", (2, 2), rgba_stream))
    geometry_rows = _audit_v2_images(one_by_four, two_by_two)
    assert geometry_rows[0]["image"]["pixel_sha256"] != geometry_rows[1]["image"]["pixel_sha256"]

    rgb = Image.new("RGB", (2, 2), (10, 20, 30))
    opaque_rgba = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    palette = Image.new("P", (2, 2), 0)
    palette.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    palette.info["transparency"] = bytes([255] + [0] * 255)
    normalized = _audit_v2_images(
        _png_bytes(rgb),
        _png_bytes(opaque_rgba),
        _png_bytes(palette),
    )
    assert len({row["image"]["pixel_sha256"] for row in normalized}) == 1

    alpha_changed = _audit_v2_images(
        _png_bytes(opaque_rgba),
        _png_bytes(Image.new("RGBA", (2, 2), (10, 20, 30, 254))),
    )
    assert alpha_changed[0]["image"]["pixel_sha256"] != alpha_changed[1]["image"]["pixel_sha256"]


def test_canonical_frame_ignores_orientation_metadata_but_not_raster_rotation() -> None:
    image = Image.new("RGB", (2, 1))
    image.putdata([(1, 2, 3), (200, 201, 202)])
    plain = _png_bytes(image)
    orientation_tagged = _png_bytes(image, exif_orientation=6)
    physically_rotated = _png_bytes(image.transpose(Image.Transpose.ROTATE_270))
    assert plain != orientation_tagged

    rows = _audit_v2_images(plain, orientation_tagged, physically_rotated)

    assert rows[0]["image"]["pixel_sha256"] == rows[1]["image"]["pixel_sha256"]
    assert rows[0]["image"]["pixel_sha256"] != rows[2]["image"]["pixel_sha256"]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("encoded_limit", "encoded_image_too_large"),
        ("declared_limit", "declared_image_too_large"),
        ("decoded_limit", "image_too_large"),
        ("bomb_warning", "decompression_bomb"),
        ("bomb_error", "decompression_bomb"),
        ("truncated", "decode_failed"),
        ("unidentified", "decode_failed"),
        ("allocation", "decode_failed"),
        ("pillow_version", "unsupported_pillow_version"),
    ),
)
def test_canonical_decoder_failures_are_safe_and_deterministic(
    case: str,
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _png_bytes(Image.new("RGBA", (2, 2), (1, 2, 3, 4)))
    record = _audit_record(encoded)
    max_encoded_bytes = smb_audit.DEFAULT_MAX_ENCODED_BYTES
    max_pixels = smb_audit.DEFAULT_MAX_PIXELS
    if case == "encoded_limit":
        max_encoded_bytes = len(encoded) - 1
    elif case == "declared_limit":
        record["original_width"] = 10_000
        record["original_height"] = 10_000
        max_pixels = 4
    elif case == "decoded_limit":
        record["original_width"] = 1
        record["original_height"] = 1
        max_pixels = 3
    elif case == "bomb_warning":
        monkeypatch.setattr(smb_audit.Image, "MAX_IMAGE_PIXELS", 3)
    elif case == "bomb_error":
        monkeypatch.setattr(smb_audit.Image, "MAX_IMAGE_PIXELS", 1)
    elif case == "truncated":
        record["image"] = encoded[: len(encoded) // 2]
    elif case == "unidentified":
        record["image"] = b"not-an-image-and-not-a-secret"
    elif case == "allocation":
        monkeypatch.setattr(
            smb_audit.Image,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("private payload")),
        )
    else:
        monkeypatch.setattr(smb_audit, "PILLOW_VERSION", "0.0.0", raising=False)

    descriptor = yaml.safe_load(
        (Path(__file__).parents[1] / "data" / "sources" / "smb.yaml").read_text(encoding="utf-8")
    )
    row = smb_audit.audit_item(
        record,
        upstream_index=0,
        source_descriptor=descriptor,
        trusted_cache_roots=(Path.cwd(),),
        max_encoded_bytes=max_encoded_bytes,
        max_pixels=max_pixels,
    )

    assert row["processing_status"] == "failed"
    assert row["unprocessable_reason"] == expected_reason
    assert row["pixel_sha256"] is None
    assert "private payload" not in json.dumps(row, sort_keys=True)


@pytest.mark.parametrize(
    "mutation",
    ("missing_mirror", "pixel_digest", "encoded_equality", "encoded_digest"),
)
def test_v2_collection_rejects_canonical_exact_evidence_mutations(mutation: str) -> None:
    rows = _compact_v2_rows()
    rows[1]["image"]["pixel_sha256"] = rows[0]["image"]["pixel_sha256"]
    derived = smb_audit.derive_v2_exact_relations(rows)
    descriptor = _compact_v2_descriptor(derived)
    descriptor["hash_provenance"]["pixels"] = {
        "algorithm": "sha256",
        "version": 2,
        "canonicalization": "canonical-rgba-frame-v2",
        "domain_separator": "smb-canonical-rgba-frame-v2",
        "decoder_library": "Pillow",
        "decoder_version": "12.3.0",
        "output_mode": "RGBA8",
        "alpha_policy": "retain-alpha-and-underlying-rgb",
        "orientation_policy": "stored-raster-ignore-exif",
        "metadata_policy": "ignore-non-raster-metadata",
        "max_encoded_bytes": 67_108_864,
        "max_pixels": 100_000_000,
        "failure_policy": "safe-explicit-failure-no-digest",
    }
    descriptor["duplicate_provenance"]["exact"] = {
        "algorithm": "canonical-pixel-sha256",
        "version": 2,
    }
    if mutation == "missing_mirror":
        derived[1]["duplicate_relations"] = []
        derived[1]["duplicate_summary"]["exact_relation_count"] = 0
        derived[1]["duplicate_summary"]["duplicate_relation_count"] = 0
        derived[1]["duplicate_summary"]["group_ids"] = []
    elif mutation == "pixel_digest":
        derived[0]["duplicate_relations"][0]["evidence"]["pixel_sha256"] = "f" * 64
    elif mutation == "encoded_equality":
        derived[0]["duplicate_relations"][0]["evidence"]["encoded_equality"] = True
    else:
        derived[0]["duplicate_relations"][0]["evidence"]["encoded_sha256"] = "e" * 64

    with pytest.raises(smb_audit.ManifestPublicationError):
        smb_audit.validate_v2_manifest_collection(descriptor, derived)


def test_new_v2_publication_rejects_legacy_unframed_pixel_provenance() -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    descriptor["hash_provenance"]["pixels"] = {
        "algorithm": "sha256",
        "version": 1,
        "canonicalization": "rgba-uint8-row-major-v1",
    }
    descriptor["duplicate_provenance"]["exact"] = {
        "algorithm": "encoded-and-pixel-sha256",
        "version": 1,
    }

    with pytest.raises(smb_audit.ManifestPublicationError, match="canonical-pixel"):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


@pytest.mark.parametrize("mutation", ("missing_mirror", "mismatched_mirror", "bad_summary"))
def test_v2_collection_rejects_pair_or_summary_contradictions(mutation: str) -> None:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 1, disposition="related")
    descriptor = _compact_v2_descriptor(rows)
    if mutation == "missing_mirror":
        rows[1]["duplicate_relations"] = []
        rows[1]["near_duplicate_candidate_ids"] = []
        rows[1]["duplicate_summary"]["perceptual_relation_count"] = 0
        rows[1]["duplicate_summary"]["related_relation_count"] = 0
    elif mutation == "mismatched_mirror":
        rows[1]["duplicate_relations"][0]["disposition"] = "distinct"
        rows[1]["duplicate_summary"]["related_relation_count"] = 0
        rows[1]["duplicate_summary"]["distinct_relation_count"] = 1
    else:
        rows[0]["duplicate_summary"]["related_relation_count"] = 0

    with pytest.raises(smb_audit.ManifestPublicationError):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


def test_v2_collection_retains_different_relations_sharing_one_item() -> None:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 1, disposition="distinct")
    _attach_relation(rows, 0, 2, disposition="related")

    smb_audit.validate_v2_manifest_collection(_compact_v2_descriptor(rows), rows)

    relations = {r["counterpart_item_id"]: r["disposition"] for r in rows[0]["duplicate_relations"]}
    assert relations == {"smb-test-000001": "distinct", "smb-test-000002": "related"}


def test_v2_collection_rejects_count_preserving_sample_identity_swap() -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    sampled = next(row for row in rows if row["audit_sample_member"] is True)
    unsampled = next(row for row in rows if row["audit_sample_member"] is False)
    sampled["audit_sample_member"] = False
    sampled["visual_review"] = {"status": "not_visually_reviewed"}
    unsampled["audit_sample_member"] = True
    unsampled["visual_review"] = {
        "status": "sampled_human_reviewed",
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-08-18",
        "rationale": "Substituted while preserving all declared counts.",
        "quality_flags": [],
        "suitability": "suitable",
    }

    with pytest.raises(smb_audit.ManifestPublicationError, match="sample"):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("seed", "seed"),
        ("identity_fields", "validation"),
        ("algorithm", "validation"),
        ("version", "validation"),
    ),
)
def test_v2_collection_rejects_sample_selector_contract_drift(mutation: str, match: str) -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    if mutation == "seed":
        descriptor["sample_selection"]["seed"] += 1
    elif mutation == "identity_fields":
        descriptor["sample_selection"]["identity_fields"] = ["item_id", "upstream_index"]
    elif mutation == "algorithm":
        descriptor["sample_selection"]["algorithm"] = "other-rank"
    else:
        descriptor["sample_selection"]["version"] = 2

    with pytest.raises(smb_audit.ManifestPublicationError, match=match):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


@pytest.mark.parametrize("field", ("source_key", "source_revision", "split"))
def test_v2_collection_rejects_row_source_provenance_drift(field: str) -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    rows[0][field] = {
        "source_key": "other",
        "source_revision": "b" * 40,
        "split": "validation",
    }[field]

    with pytest.raises(smb_audit.ManifestPublicationError, match="source"):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


@pytest.mark.parametrize(
    "mutation", ("deleted", "added", "duplicated", "reordered", "identity", "reason")
)
def test_v2_collection_rejects_exclusion_ledger_drift(mutation: str) -> None:
    rows = _compact_v2_rows()
    for index in (4, 8):
        rows[index]["paired_eligible"] = False
        rows[index]["paired_ineligibility_reason"] = "invalid_region_annotation"
    descriptor = _compact_v2_descriptor(rows)
    descriptor["exclusions"] = [
        {
            "upstream_index": row["upstream_index"],
            "item_id": row["item_id"],
            "reason": row["paired_ineligibility_reason"],
        }
        for row in rows
        if row["paired_eligible"] is False
    ]
    if mutation == "deleted":
        descriptor["exclusions"].pop()
    elif mutation == "added":
        descriptor["exclusions"].append(
            {"upstream_index": 12, "item_id": "smb-test-000012", "reason": "extra"}
        )
    elif mutation == "duplicated":
        descriptor["exclusions"].append(copy.deepcopy(descriptor["exclusions"][0]))
    elif mutation == "reordered":
        descriptor["exclusions"].reverse()
    elif mutation == "identity":
        descriptor["exclusions"][0]["item_id"] = "smb-test-000005"
    else:
        descriptor["exclusions"][0]["reason"] = "different_reason"

    with pytest.raises(smb_audit.ManifestPublicationError, match="exclusion"):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


def test_v2_reconciliation_reports_validated_generation_facts(tmp_path: Path) -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    active_path = tmp_path / "active.yaml"
    generation_root = tmp_path / "generations"
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=rows,
    )
    resolved, _ = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )

    report = smb_audit.reconcile_manifest(active_path=active_path, generation_root=generation_root)

    assert report == {
        "row_count": 685,
        "processed": 685,
        "failed": 0,
        "paired_eligible": 685,
        "generation_id": resolved["generation_id"],
        "records_sha256": resolved["records_sha256"],
        "benchmark_state": "AUDITED_LOCKED",
        "exclusion_count": 0,
        "source_group_count": 685,
    }


@pytest.mark.parametrize("mutation", ("all_one", "all_null", "one_null"))
def test_v2_collection_rejects_destroyed_source_group_mapping(mutation: str) -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    if mutation == "all_one":
        for row in rows:
            row["source_group_id"] = "collapsed-group"
    elif mutation == "all_null":
        for row in rows:
            row["source_group_id"] = None
    else:
        rows[0]["source_group_id"] = None

    with pytest.raises(smb_audit.ManifestPublicationError, match="group"):
        smb_audit.validate_v2_manifest_collection(descriptor, rows)


def test_v2_policy_group_replacement_is_rejected_without_mutating_audit_rows() -> None:
    rows = _compact_v2_rows()
    review_rows = _completed_v2_review_rows(rows)
    original = copy.deepcopy(rows)
    policy = next(row for row in review_rows if row["review_kind"] == "item_policy")
    policy["source_group_id"] = "replacement-group"

    with pytest.raises(smb_audit.ReviewFinalizationError, match="source_group_id"):
        smb_audit.apply_review_dispositions(rows, review_rows)

    assert rows == original


def test_v2_review_application_preserves_audited_source_groups() -> None:
    rows = _compact_v2_rows()
    original_groups = [row["source_group_id"] for row in rows]

    updated = smb_audit.apply_review_dispositions(rows, _completed_v2_review_rows(rows))

    assert [row["source_group_id"] for row in updated] == original_groups


def test_tracked_active_recovery_retains_260_canonical_source_groups(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_root = project_root / "data" / "manifests"
    recovery_descriptor = manifest_root / "smb-evaluation-v1-recovery.yaml"
    recovery_records = manifest_root / "smb-evaluation-v1-recovery.jsonl.gz"
    active_path = tmp_path / "smb-evaluation-v1.yaml"
    active_path.write_bytes((manifest_root / "smb-evaluation-v1.yaml").read_bytes())
    generation_root = tmp_path / "generations"

    smb_audit.recover_active_manifest(
        active_path=active_path,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
        generation_root=generation_root,
    )
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )
    report = smb_audit.reconcile_manifest(active_path=active_path, generation_root=generation_root)

    assert descriptor["benchmark_state"] == "AUDITED_LOCKED"
    assert report["source_group_count"] == 260
    assert len({row["source_group_id"] for row in rows}) == 260


def test_v2_policy_and_candidate_saves_do_not_create_visual_or_item_pair_claims(
    tmp_path: Path,
) -> None:
    from score_super_resolution.smb_review_ui import SMBReviewSession

    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 1, disposition="pending")
    _attach_relation(rows, 0, 2, disposition="pending")
    review_rows = smb_audit.prepare_v2_review_rows(rows)
    review_path = tmp_path / "review.csv"
    review_path.write_bytes(smb_audit.canonical_review_csv(review_rows))
    session = object.__new__(SMBReviewSession)
    session.review_path = review_path
    session.sample_ids = [f"smb-test-{index:06d}" for index in range(64)]
    session.visual_item_ids = list(session.sample_ids)
    session.candidate_keys = [
        row["review_key"] for row in review_rows if row["review_kind"] == "duplicate_pair"
    ]
    session.reload()

    session.apply_policy("reviewer-1")
    after_policy = session.read_rows()
    policies = [row for row in after_policy if row["review_kind"] == "item_policy"]
    assert all(row["item_provenance_status"] == "unavailable" for row in policies)
    assert all(row["redistribution_status"] == "not_established" for row in policies)
    assert all(row["figure_reproduction_status"] == "prohibited" for row in policies)
    assert all(not row["quality_disposition"] for row in policies)
    assert all(not row["suitability_disposition"] for row in policies)

    first, second = session.candidate_keys
    session.save_candidate(
        review_key=first,
        reviewer="reviewer-1",
        disposition="distinct",
        rationale="The first pair is distinct.",
    )
    session.save_candidate(
        review_key=second,
        reviewer="reviewer-1",
        disposition="related",
        rationale="The second pair is related.",
    )
    by_key = {row["review_key"]: row for row in session.read_rows()}
    assert by_key[first]["duplicate_disposition"] == "distinct"
    assert by_key[second]["duplicate_disposition"] == "related"
    assert all(not row["duplicate_disposition"] for row in policies)


def test_v2_publication_round_trip_dispatches_contract_version(tmp_path: Path) -> None:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    active_path = tmp_path / "data" / "manifests" / "smb-evaluation-v2.yaml"
    generation_root = tmp_path / "artifacts" / "smb-manifests" / "generations"

    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=rows,
    )
    resolved_descriptor, resolved_rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )

    assert resolved_descriptor["schema_version"] == 2
    assert resolved_descriptor["row_schema_version"] == 2
    assert len(resolved_rows) == 685


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


@pytest.mark.parametrize(
    ("field", "payload"),
    (("reviewer", "=2+2"), ("rationale", "@SUM(1,1)")),
)
def test_finalizer_rejects_formula_prefixed_human_evidence(
    tmp_path: Path, field: str, payload: str
) -> None:
    active_path, generation_root, _, _, _, review = _emit_review_bundle(tmp_path)
    rows = _complete_review(review)
    rows[0][field] = payload
    _write_review(review, rows)

    with pytest.raises(smb_audit.ReviewFinalizationError) as caught:
        smb_audit.finalize_reviewed_manifest(
            review_path=review,
            active_path=active_path,
            generation_root=generation_root,
        )

    message = str(caught.value)
    assert field in message
    assert rows[0]["review_key"] in message
    assert payload not in message


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


def test_publication_generation_parent_swap_stays_on_retained_inode(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    generation_root = storage / "generations"
    active_path = tmp_path / "manifests" / "active.yaml"
    outside = tmp_path / "outside"
    storage.mkdir()
    active_path.parent.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"outside-generation-sentinel")
    retained = tmp_path / "retained-storage"
    swapped = False

    def hook(boundary: str) -> None:
        nonlocal swapped
        if boundary == "generation_parent_anchored":
            storage.rename(retained)
            storage.symlink_to(outside, target_is_directory=True)
            swapped = True

    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=_full_rows(),
        boundary_hook=hook,
    )

    assert swapped is True
    assert sentinel.read_bytes() == b"outside-generation-sentinel"
    assert list(outside.iterdir()) == [sentinel]
    assert any((retained / "generations").iterdir())


def test_publication_active_parent_swap_stays_on_retained_inode(tmp_path: Path) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_generation_id = yaml.safe_load(active_path.read_text(encoding="utf-8"))["generation_id"]
    active_parent = active_path.parent
    retained = tmp_path / "retained-manifests"
    outside = tmp_path / "outside-active"
    outside.mkdir()
    outside_active = outside / active_path.name
    outside_active.write_bytes(b"outside-active-sentinel")
    swapped = False

    def hook(boundary: str) -> None:
        nonlocal swapped
        if boundary == "active_parent_anchored":
            active_parent.rename(retained)
            active_parent.symlink_to(outside, target_is_directory=True)
            swapped = True

    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=_descriptor(),
        rows=_changed_rows(),
        boundary_hook=hook,
    )

    assert swapped is True
    assert outside_active.read_bytes() == b"outside-active-sentinel"
    retained_pointer = yaml.safe_load((retained / active_path.name).read_text(encoding="utf-8"))
    assert retained_pointer["generation_id"] != old_generation_id


def test_install_candidate_surface_and_permanent_lock_ignore_are_declared() -> None:
    project_root = Path(__file__).parents[1]
    lock_path = project_root / "data/manifests/.smb-evaluation-v1.install.lock"
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(lock_path.relative_to(project_root))],
        cwd=project_root,
        check=False,
    )

    assert ignored.returncode == 0
    assert hasattr(smb_audit, "install_candidate")
    parser = smb_audit._parser()
    command = parser.parse_args(
        [
            "install-candidate",
            "--stage-root",
            "stage",
            "--manifest-generation-root",
            "generations",
            "--manifest-active",
            "active.yaml",
            "--expected-active-sha256-from-stage",
        ]
    )
    assert command.expected_active_sha256_from_stage is True
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(lock_path.relative_to(project_root))],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    assert tracked.returncode != 0


def _write_install_candidate_stage(
    stage_root: Path,
    *,
    expected_active_bytes: bytes,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    descriptor = _compact_v2_descriptor(rows)
    (
        completed,
        pointer,
        recovery,
        descriptor_bytes,
        records_bytes,
        compressed_bytes,
        recovery_bytes,
    ) = smb_audit._build_manifest_recovery_v2(descriptor, rows)
    generation = stage_root / "artifacts/smb-manifests/generations" / completed["generation_id"]
    generation.mkdir(parents=True)
    (generation / "manifest-descriptor.yaml").write_bytes(descriptor_bytes)
    (generation / "manifest-records.jsonl").write_bytes(records_bytes)
    recovery_descriptor = stage_root / recovery["recovery_descriptor_path"]
    recovery_descriptor.parent.mkdir(parents=True)
    recovery_descriptor.write_bytes(recovery_bytes)
    (stage_root / recovery["recovery_records_path"]).write_bytes(compressed_bytes)
    active = stage_root / recovery["active_pointer_path"]
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(smb_audit._canonical_descriptor(pointer))
    (stage_root / "install-metadata.yaml").write_bytes(
        smb_audit._canonical_descriptor(
            {
                "schema_version": 1,
                "record_type": "canonical-pixel-candidate-install",
                "expected_previous_active_sha256": hashlib.sha256(
                    expected_active_bytes
                ).hexdigest(),
                "generation_id": completed["generation_id"],
                "bundle_id": recovery["bundle_id"],
                "active_pointer_path": recovery["active_pointer_path"],
                "recovery_descriptor_path": recovery["recovery_descriptor_path"],
                "recovery_records_path": recovery["recovery_records_path"],
            }
        )
    )
    return pointer


def test_install_candidate_is_idempotent_and_rejects_third_digest_cas(tmp_path: Path) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_active = active_path.read_bytes()
    first_stage = tmp_path / "stage-a"
    first_pointer = _write_install_candidate_stage(
        first_stage,
        expected_active_bytes=old_active,
        rows=_compact_v2_rows(),
    )

    installed = smb_audit.install_candidate(
        stage_root=first_stage,
        generation_root=generation_root,
        active_path=active_path,
        expected_active_sha256_from_stage=True,
    )
    lock_path = active_path.parent / smb_audit.INSTALL_LOCK_BASENAME
    lock_inode = lock_path.stat().st_ino
    retry = smb_audit.install_candidate(
        stage_root=first_stage,
        generation_root=generation_root,
        active_path=active_path,
        expected_active_sha256_from_stage=True,
    )

    assert installed["status"] == "installed"
    assert retry["status"] == "idempotent"
    assert lock_path.stat().st_ino == lock_inode
    assert yaml.safe_load(active_path.read_text(encoding="utf-8")) == first_pointer

    third_digest_bytes = b"third-party-active-pointer\n"
    active_path.write_bytes(third_digest_bytes)

    with pytest.raises(smb_audit.ManifestPublicationError, match="compare-and-swap conflict"):
        smb_audit.install_candidate(
            stage_root=first_stage,
            generation_root=generation_root,
            active_path=active_path,
            expected_active_sha256_from_stage=True,
        )

    assert active_path.read_bytes() == third_digest_bytes
    assert lock_path.stat().st_ino == lock_inode


def test_permanent_flock_kernel_release_and_live_waiter_cannot_be_displaced(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / smb_audit.INSTALL_LOCK_BASENAME
    code = """
import fcntl
import os
import sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print(os.fstat(fd).st_ino, flush=True)
os._exit(0)
"""
    owner_a = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert owner_a.stdout is not None
    inode_a = int(owner_a.stdout.readline().strip())
    assert owner_a.wait(timeout=10) == 0

    waiter_b = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    fcntl.flock(waiter_b, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contender_c = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        assert os.fstat(waiter_b).st_ino == inode_a == os.fstat(contender_c).st_ino
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender_c, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert lock_path.stat().st_ino == inode_a
        fcntl.flock(waiter_b, fcntl.LOCK_UN)
        fcntl.flock(contender_c, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        with suppress(OSError):
            fcntl.flock(waiter_b, fcntl.LOCK_UN)
        with suppress(OSError):
            fcntl.flock(contender_c, fcntl.LOCK_UN)
        os.close(waiter_b)
        os.close(contender_c)


@pytest.mark.parametrize("same_candidate", (True, False))
def test_concurrent_candidate_installers_serialize_with_old_or_winner_reads(
    tmp_path: Path,
    same_candidate: bool,
) -> None:
    active_path, generation_root = _publish(tmp_path, _full_rows())
    old_pointer = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    old_active = active_path.read_bytes()
    stage_a = tmp_path / "concurrent-stage-a"
    _write_install_candidate_stage(
        stage_a,
        expected_active_bytes=old_active,
        rows=_compact_v2_rows(),
    )
    stage_b = stage_a
    if not same_candidate:
        changed_rows = _compact_v2_rows()
        changed_rows[0]["image"]["encoded_sha256"] = "f" * 64
        stage_b = tmp_path / "concurrent-stage-b"
        _write_install_candidate_stage(
            stage_b,
            expected_active_bytes=old_active,
            rows=changed_rows,
        )
    ready = threading.Barrier(2)
    stop_reader = threading.Event()
    outcomes: list[str] = []
    failures: list[BaseException] = []
    observations: list[str] = []

    def install(stage: Path) -> None:
        def hook(boundary: str) -> None:
            if boundary == "recovery_bundle_installed":
                ready.wait(timeout=10)

        try:
            result = smb_audit.install_candidate(
                stage_root=stage,
                generation_root=generation_root,
                active_path=active_path,
                expected_active_sha256_from_stage=True,
                boundary_hook=hook,
            )
            outcomes.append(str(result["status"]))
        except smb_audit.ManifestPublicationError as error:
            if "compare-and-swap conflict" in str(error):
                outcomes.append("conflict")
            else:  # pragma: no cover - reported below
                failures.append(error)

    def read() -> None:
        try:
            while not stop_reader.is_set():
                descriptor, _ = smb_audit.resolve_active_manifest(
                    active_path=active_path,
                    generation_root=generation_root,
                )
                observations.append(str(descriptor["generation_id"]))
        except BaseException as error:  # pragma: no cover - reported below
            failures.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    installers = [threading.Thread(target=install, args=(stage,)) for stage in (stage_a, stage_b)]
    for installer in installers:
        installer.start()
    for installer in installers:
        installer.join(timeout=30)
    stop_reader.set()
    reader.join(timeout=10)

    assert all(not installer.is_alive() for installer in installers)
    assert not failures
    assert sorted(outcomes) == (
        ["idempotent", "installed"] if same_candidate else ["conflict", "installed"]
    )
    winner = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    assert observations
    assert set(observations) <= {old_pointer["generation_id"], winner["generation_id"]}


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


def _export_recovery_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rows = _compact_v2_rows()
    descriptor = _compact_v2_descriptor(rows)
    active_path = tmp_path / "data" / "manifests" / "smb-evaluation-v1.yaml"
    source_root = tmp_path / "source-generations"
    recovery_descriptor = tmp_path / "data" / "manifests" / "smb-evaluation-v1-recovery.yaml"
    recovery_records = tmp_path / "data" / "manifests" / "smb-evaluation-v1-recovery.jsonl.gz"
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=source_root,
        descriptor=descriptor,
        rows=rows,
    )
    smb_audit.export_manifest_recovery(
        active_path=active_path,
        generation_root=source_root,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
    )
    return active_path, source_root, recovery_descriptor, recovery_records


def _legacy_recovery_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = _compact_v2_rows()
    _attach_relation(rows, 0, 64, disposition="distinct")
    descriptor = _compact_v2_descriptor(rows)
    descriptor["hash_provenance"]["pixels"] = {
        "algorithm": "sha256",
        "version": 1,
        "canonicalization": "rgba-uint8-row-major-v1",
    }
    descriptor["duplicate_provenance"]["exact"] = {
        "algorithm": "encoded-and-pixel-sha256",
        "version": 1,
    }
    active_path = tmp_path / "legacy" / "data" / "manifests" / "smb-evaluation-v1.yaml"
    generation_root = tmp_path / "legacy-generations"
    recovery_descriptor = tmp_path / "legacy-recovery.yaml"
    recovery_records = tmp_path / "legacy-recovery.jsonl.gz"
    smb_audit.publish_manifest_generation(
        active_path=active_path,
        generation_root=generation_root,
        descriptor=descriptor,
        rows=rows,
        allow_legacy_hash_provenance=True,
    )
    smb_audit.export_manifest_recovery(
        active_path=active_path,
        generation_root=generation_root,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
    )
    return active_path, recovery_descriptor, recovery_records


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_canonical_rehash_candidate_is_deterministic_and_never_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_path, recovery_descriptor, recovery_records = _legacy_recovery_bundle(tmp_path)
    original_active = active_path.read_bytes()
    legacy_root = tmp_path / "legacy-resolve"
    _, legacy_rows = smb_audit.resolve_active_manifest(
        active_path=active_path,
        generation_root=tmp_path / "legacy-generations",
    )
    corrected = copy.deepcopy(legacy_rows)
    for index, row in enumerate(corrected):
        row["image"]["pixel_sha256"] = hashlib.sha256(
            f"canonical-frame-{index}".encode()
        ).hexdigest()
    corrected = smb_audit.derive_v2_exact_relations(corrected)
    monkeypatch.setattr(smb_audit, "load_smb", lambda **_kwargs: [object()] * 685)
    monkeypatch.setattr(smb_audit, "_without_automatic_image_decoding", lambda value: value)
    monkeypatch.setattr(smb_audit, "audit_dataset_v2", lambda *_args, **_kwargs: corrected)
    monkeypatch.setattr(
        smb_audit,
        "audit_source_provenance",
        lambda _root: copy.deepcopy(_compact_v2_descriptor(corrected)["source_provenance"]),
    )
    source_path = Path(__file__).parents[1] / "data" / "sources" / "smb.yaml"

    first = smb_audit.build_canonical_pixel_rehash_candidate(
        source_path=source_path,
        trusted_cache_roots=(tmp_path,),
        legacy_active_path=active_path,
        legacy_recovery_descriptor_path=recovery_descriptor,
        legacy_recovery_records_path=recovery_records,
        staging_root=tmp_path / "candidate-a",
    )
    second = smb_audit.build_canonical_pixel_rehash_candidate(
        source_path=source_path,
        trusted_cache_roots=(tmp_path,),
        legacy_active_path=active_path,
        legacy_recovery_descriptor_path=recovery_descriptor,
        legacy_recovery_records_path=recovery_records,
        staging_root=tmp_path / "candidate-b",
    )

    assert first == second
    assert _tree_bytes(tmp_path / "candidate-a") == _tree_bytes(tmp_path / "candidate-b")
    assert active_path.read_bytes() == original_active
    assert first["row_count"] == 685
    assert first["sampled_human_review_count"] == 64
    assert first["source_group_count"] == 685
    assert first["benchmark_state"] == "AUDITED_LOCKED"
    assert first["bundle_id"] in first["recovery_descriptor_path"]
    assert legacy_root.exists() is False


@pytest.mark.parametrize(
    "protected_family",
    (
        "sample",
        "visual_review",
        "perceptual_review",
        "source_group",
        "rights",
        "source_identity",
        "eligibility",
        "benchmark_state",
    ),
)
def test_canonical_rehash_candidate_rejects_protected_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_family: str,
) -> None:
    active_path, recovery_descriptor, recovery_records = _legacy_recovery_bundle(tmp_path)
    _, legacy_rows = smb_audit.resolve_active_manifest(
        active_path=active_path,
        generation_root=tmp_path / "legacy-generations",
    )
    corrected = copy.deepcopy(legacy_rows)
    target = corrected[0]
    if protected_family == "sample":
        target["audit_sample_member"] = not target["audit_sample_member"]
    elif protected_family == "visual_review":
        sampled = next(row for row in corrected if row["audit_sample_member"] is True)
        sampled["visual_review"]["rationale"] = "changed"
    elif protected_family == "perceptual_review":
        target["duplicate_relations"][0]["disposition"] = "related"
    elif protected_family == "source_group":
        target["source_group_id"] = "changed-group"
    elif protected_family == "rights":
        target["rights"]["figure_reproduction"]["status"] = "permitted"
    elif protected_family == "source_identity":
        target["source_identity"]["page_normalized"] = "changed"
    elif protected_family == "eligibility":
        target["paired_eligible"] = False
        target["paired_ineligibility_reason"] = "changed"
    else:
        monkeypatch.setattr(
            smb_audit,
            "_candidate_benchmark_state",
            lambda _descriptor: "EVALUATION_UNLOCKED",
            raising=False,
        )
    monkeypatch.setattr(smb_audit, "load_smb", lambda **_kwargs: [object()] * 685)
    monkeypatch.setattr(smb_audit, "_without_automatic_image_decoding", lambda value: value)
    monkeypatch.setattr(smb_audit, "audit_dataset_v2", lambda *_args, **_kwargs: corrected)
    monkeypatch.setattr(
        smb_audit,
        "audit_source_provenance",
        lambda _root: copy.deepcopy(_compact_v2_descriptor(corrected)["source_provenance"]),
    )

    with pytest.raises(ValueError, match=r"protected|AUDITED_LOCKED"):
        smb_audit.build_canonical_pixel_rehash_candidate(
            source_path=Path(__file__).parents[1] / "data" / "sources" / "smb.yaml",
            trusted_cache_roots=(tmp_path,),
            legacy_active_path=active_path,
            legacy_recovery_descriptor_path=recovery_descriptor,
            legacy_recovery_records_path=recovery_records,
            staging_root=tmp_path / "candidate",
        )


def test_verify_authoritative_determinism_materializes_only_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_builder(*, staging_root: Path, **_kwargs: object) -> dict[str, object]:
        calls.append(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        (staging_root / "candidate.yaml").write_bytes(b"candidate: stable\n")
        return {"generation_id": "a" * 64, "row_count": 685}

    monkeypatch.setattr(smb_audit, "build_canonical_pixel_rehash_candidate", fake_builder)
    verified = tmp_path / "verified"
    report = smb_audit.verify_authoritative_determinism(
        source_path=tmp_path / "source.yaml",
        trusted_cache_roots=(tmp_path,),
        legacy_active_path=tmp_path / "active.yaml",
        legacy_recovery_descriptor_path=tmp_path / "recovery.yaml",
        legacy_recovery_records_path=tmp_path / "recovery.jsonl.gz",
        stage_parent=tmp_path / "stages",
        verified_stage=verified,
    )

    assert report == {"generation_id": "a" * 64, "row_count": 685}
    assert (verified / "candidate.yaml").read_bytes() == b"candidate: stable\n"
    assert len(calls) == 3
    assert calls[0] == calls[2]
    assert calls[0] != calls[1]


def test_verify_authoritative_determinism_rejects_independent_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = 0

    def fake_builder(*, staging_root: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        staging_root.mkdir(parents=True, exist_ok=True)
        (staging_root / "candidate.yaml").write_bytes(f"call: {call_count}\n".encode())
        return {"generation_id": "a" * 64}

    monkeypatch.setattr(smb_audit, "build_canonical_pixel_rehash_candidate", fake_builder)
    verified = tmp_path / "verified"
    with pytest.raises(ValueError, match="byte-identical"):
        smb_audit.verify_authoritative_determinism(
            source_path=tmp_path / "source.yaml",
            trusted_cache_roots=(tmp_path,),
            legacy_active_path=tmp_path / "active.yaml",
            legacy_recovery_descriptor_path=tmp_path / "recovery.yaml",
            legacy_recovery_records_path=tmp_path / "recovery.jsonl.gz",
            stage_parent=tmp_path / "stages",
            verified_stage=verified,
        )
    assert not verified.exists()


def _rewrite_recovery_descriptor(path: Path, recovery: dict[str, Any]) -> None:
    recovery["metadata_sha256"] = recovery_metadata_sha256(recovery)
    path.write_bytes(smb_audit._canonical_descriptor(recovery))


def test_recovery_export_is_byte_deterministic(tmp_path: Path) -> None:
    active_path, source_root, recovery_descriptor, recovery_records = _export_recovery_bundle(
        tmp_path
    )
    first = (recovery_descriptor.read_bytes(), recovery_records.read_bytes())

    smb_audit.export_manifest_recovery(
        active_path=active_path,
        generation_root=source_root,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
    )

    assert (recovery_descriptor.read_bytes(), recovery_records.read_bytes()) == first
    recovery = yaml.safe_load(first[0])
    assert recovery["compression"] == {
        "algorithm": "gzip",
        "format_version": 1,
        "compresslevel": 9,
        "mtime": 0,
        "filename": "",
    }


def test_recover_active_from_empty_generation_root_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    generation_root = tmp_path / "restored-generations"

    first = smb_audit.recover_active_manifest(
        active_path=active_path,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
        generation_root=generation_root,
    )
    generation = generation_root / first["generation_id"]
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in generation.iterdir()
    }
    second = smb_audit.recover_active_manifest(
        active_path=active_path,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
        generation_root=generation_root,
    )

    assert first == second
    assert first["row_count"] == 685
    assert first["benchmark_state"] == "AUDITED_LOCKED"
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in generation.iterdir()
    } == before


@pytest.mark.parametrize("target", ("descriptor", "records"))
def test_recover_active_rejects_symlink_inputs_before_materialization(
    tmp_path: Path, target: str
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    original = recovery_descriptor if target == "descriptor" else recovery_records
    symlink = tmp_path / f"linked-{original.name}"
    symlink.symlink_to(original)
    generation_root = tmp_path / "restored-generations"

    with pytest.raises(smb_audit.ManifestPublicationError, match="regular"):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=(symlink if target == "descriptor" else recovery_descriptor),
            recovery_records_path=(symlink if target == "records" else recovery_records),
            generation_root=generation_root,
        )

    assert not generation_root.exists()


@pytest.mark.parametrize(
    ("name", "label"),
    (
        ("manifest-recovery.yaml", "recovery descriptor"),
        ("manifest-records.jsonl.gz", "compressed recovery records"),
    ),
)
def test_recovery_parent_swap_after_anchor_never_reads_outside_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    label: str,
) -> None:
    safe_parent = tmp_path / "checked" / "bundle"
    safe_parent.mkdir(parents=True)
    safe_bytes = b"safe recovery bytes"
    target = safe_parent / name
    target.write_bytes(safe_bytes)
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir()
    outside_bytes = b"OUTSIDE-SENTINEL-MUST-NOT-BE-READ"
    (outside_parent / name).write_bytes(outside_bytes)
    parked_parent = safe_parent.with_name("retained-bundle")
    original_open = os.open
    swapped = False

    def swap_before_final_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        whole_path_open = dir_fd is None and Path(path) == target.absolute()
        anchored_final_open = dir_fd is not None and os.fsdecode(path) == name
        if not swapped and (whole_path_open or anchored_final_open):
            safe_parent.rename(parked_parent)
            safe_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_final_open)

    result = smb_audit._read_regular_nofollow(
        target,
        label=label,
        maximum_bytes=1024,
    )

    assert swapped is True
    assert result == safe_bytes
    assert outside_bytes not in result


def test_recovery_confinement_fails_closed_without_directory_nofollow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery = tmp_path / "recovery.yaml"
    recovery.write_bytes(b"safe")
    monkeypatch.setattr(os, "O_DIRECTORY", 0)

    with pytest.raises(smb_audit.ManifestPublicationError, match="unavailable") as caught:
        smb_audit._read_regular_nofollow(
            recovery,
            label="recovery descriptor",
            maximum_bytes=1024,
        )

    assert str(tmp_path) not in str(caught.value)


def test_recover_active_rejects_one_bit_compressed_corruption_before_visibility(
    tmp_path: Path,
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    corrupted = bytearray(recovery_records.read_bytes())
    corrupted[-1] ^= 1
    recovery_records.write_bytes(corrupted)
    generation_root = tmp_path / "restored-generations"

    with pytest.raises(smb_audit.ManifestPublicationError, match="compressed checksum"):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_descriptor,
            recovery_records_path=recovery_records,
            generation_root=generation_root,
        )

    assert not generation_root.exists()


def test_recover_active_rejects_recovery_yaml_bit_change_before_decompression(
    tmp_path: Path,
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    recovery = yaml.safe_load(recovery_descriptor.read_text(encoding="utf-8"))
    recovery["recovery_command"] = recovery["recovery_command"].replace(
        "recover-active", "recover-activa"
    )
    recovery_descriptor.write_bytes(smb_audit._canonical_descriptor(recovery))
    generation_root = tmp_path / "restored-generations"

    with pytest.raises(smb_audit.ManifestPublicationError, match="failed validation"):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_descriptor,
            recovery_records_path=recovery_records,
            generation_root=generation_root,
        )

    assert not generation_root.exists()


def test_recover_active_rejects_changed_uncompressed_rows_before_visibility(
    tmp_path: Path,
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    recovery = yaml.safe_load(recovery_descriptor.read_text(encoding="utf-8"))
    chunks: list[bytes] = []
    with gzip.GzipFile(fileobj=io.BytesIO(recovery_records.read_bytes()), mode="rb") as handle:
        while chunk := handle.read(64 * 1024):
            chunks.append(chunk)
    records_bytes = b"".join(chunks)
    changed = records_bytes.replace(b"smb-test-000000", b"smb-test-000009", 1)
    assert len(changed) == len(records_bytes) and changed != records_bytes
    compressed = smb_audit._deterministic_gzip(changed)
    recovery_records.write_bytes(compressed)
    recovery["compressed_sha256"] = hashlib.sha256(compressed).hexdigest()
    recovery["compressed_size_bytes"] = len(compressed)
    _rewrite_recovery_descriptor(recovery_descriptor, recovery)
    generation_root = tmp_path / "restored-generations"

    with pytest.raises(smb_audit.ManifestPublicationError, match="records checksum"):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_descriptor,
            recovery_records_path=recovery_records,
            generation_root=generation_root,
        )

    assert not generation_root.exists()


def test_high_ratio_gzip_aborts_at_schema_uncompressed_limit_before_materialization(
    tmp_path: Path,
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    recovery = yaml.safe_load(recovery_descriptor.read_text(encoding="utf-8"))
    schema = load_schema("manifest-recovery", version=1)
    hard_limit = schema["properties"]["uncompressed_size_bytes"]["maximum"]
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", compresslevel=9, mtime=0) as handle:
        handle.write(b"x" * (hard_limit + 65_536))
    compressed = buffer.getvalue()
    assert len(compressed) < schema["properties"]["compressed_size_bytes"]["maximum"]
    recovery_records.write_bytes(compressed)
    recovery["compressed_sha256"] = hashlib.sha256(compressed).hexdigest()
    recovery["compressed_size_bytes"] = len(compressed)
    recovery["uncompressed_size_bytes"] = hard_limit
    _rewrite_recovery_descriptor(recovery_descriptor, recovery)
    generation_root = tmp_path / "restored-generations"

    with pytest.raises(smb_audit.ManifestPublicationError, match="uncompressed maximum"):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_descriptor,
            recovery_records_path=recovery_records,
            generation_root=generation_root,
        )

    assert not generation_root.exists()


def test_recovery_publication_failure_never_exposes_partial_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_path, _, recovery_descriptor, recovery_records = _export_recovery_bundle(tmp_path)
    recovery = yaml.safe_load(recovery_descriptor.read_text(encoding="utf-8"))
    generation_root = tmp_path / "restored-generations"
    monkeypatch.setenv("SCORE_SR_SMB_PUBLICATION_FAILPOINT", "generation_records_fsynced:raise")

    with pytest.raises(smb_audit.ManifestPublicationError):
        smb_audit.recover_active_manifest(
            active_path=active_path,
            recovery_descriptor_path=recovery_descriptor,
            recovery_records_path=recovery_records,
            generation_root=generation_root,
        )

    assert not (generation_root / recovery["generation_id"]).exists()


def test_recover_active_from_source_controlled_tree_with_empty_generation_root(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    active_path = project_root / "data" / "manifests" / "smb-evaluation-v1.yaml"
    recovery_descriptor = project_root / "data" / "manifests" / "smb-evaluation-v1-recovery.yaml"
    recovery_records = project_root / "data" / "manifests" / "smb-evaluation-v1-recovery.jsonl.gz"
    if not recovery_descriptor.is_file() or not recovery_records.is_file():
        pytest.skip("tracked recovery pair is produced by Plan 01-20 Task 2")
    recovery = yaml.safe_load(recovery_descriptor.read_text(encoding="utf-8"))
    generation_root = tmp_path / "empty-generation-root"

    report = smb_audit.recover_active_manifest(
        active_path=active_path,
        recovery_descriptor_path=recovery_descriptor,
        recovery_records_path=recovery_records,
        generation_root=generation_root,
    )
    descriptor, rows = smb_audit.resolve_active_manifest(
        active_path=active_path, generation_root=generation_root
    )

    assert report["generation_id"] == recovery["generation_id"]
    assert report["row_count"] == recovery["row_count"] == 685
    assert report["records_sha256"] == recovery["records_sha256"]
    assert report["row_schema_id"] == recovery["row_schema_id"]
    assert report["row_schema_version"] == recovery["row_schema_version"] == 2
    assert report["source_revision"] == recovery["source_revision"]
    assert report["source_provenance"] == recovery["source_provenance"]
    assert report["benchmark_state"] == recovery["benchmark_state"] == "AUDITED_LOCKED"
    assert descriptor["generation_id"] == recovery["generation_id"]
    assert len(rows) == 685
