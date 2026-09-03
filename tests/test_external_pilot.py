from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import score_super_resolution.external_pilot as external_pilot
from score_super_resolution.external_pilot import (
    METADATA_FIELDS,
    ExternalPilotError,
    analyze_external_pilot,
    evaluate_external_pilot,
    freeze_external_pilot_manifest,
    load_external_pilot_pages,
    qualitative_assignment,
)
from score_super_resolution.staff_scale import StaffSpacingEstimate


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "external"
    source_root.mkdir()
    rows: list[dict[str, str]] = []
    roles = ["engineering"] * 3 + ["test"] * 12
    for index, role in enumerate(roles, start=1):
        work_id = f"work-{index:03d}"
        file_name = f"{work_id}.png"
        pixels = np.full((140, 220, 3), 255, dtype=np.uint8)
        pixels[30:110:8, 10:210] = 0
        Image.fromarray(pixels, mode="RGB").save(source_root / file_name)
        rows.append(
            {
                "role": role,
                "work_id": work_id,
                "file_name": file_name,
                "genre": f"genre-{index % 4}",
                "instrument": f"instrument-{index}",
                "orientation": "horizontal" if index % 3 == 0 else "vertical",
                "source_type": "scan" if index % 2 else "born-digital",
                "rights_basis": "private-study-only",
                "source_reference": "test-fixture",
                "notation_density": ("sparse", "medium", "dense")[index % 3],
                "document_condition": ("clean", "aged", "mixed")[index % 3],
                "text_present": "yes" if index % 2 else "no",
                "notes": "",
            }
        )
    with (source_root / "source-metadata.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return source_root


def _spacing() -> StaffSpacingEstimate:
    return StaffSpacingEstimate(
        spacing_px=8.0,
        estimator_id="full-page-hybrid-horizontal-v2",
        sequence_count=4,
        contributing_regions=2,
        median_deskew_degrees=0.0,
    )


def test_external_pilot_freezes_exact_work_disjoint_input_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(
        external_pilot, "estimate_staff_spacing_full_page", lambda _pixels: _spacing()
    )

    pages = load_external_pilot_pages(tmp_path, source_root=source_root)
    manifest_path, digest = freeze_external_pilot_manifest(pages, tmp_path / "artifacts")

    assert len(pages) == 15
    assert sum(page.role == "test" for page in pages) == 12
    assert len({page.work_id for page in pages}) == 15
    assert len(digest) == 64
    assert manifest_path.is_file()
    assert all(page.file_sha256 and page.pixel_sha256 for page in pages)


def test_external_qualitative_assignment_uses_each_test_work_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(
        external_pilot, "estimate_staff_spacing_full_page", lambda _pixels: _spacing()
    )
    pages = load_external_pilot_pages(tmp_path, source_root=source_root)

    assignment = qualitative_assignment(pages)

    assert len(assignment) == 12
    assert set(assignment.values()) == set(external_pilot.CONDITIONS)
    assert all(
        list(assignment.values()).count(condition) == 2 for condition in set(assignment.values())
    )


def test_external_pilot_rejects_incomplete_test_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    metadata_path = source_root / "source-metadata.csv"
    rows = list(csv.DictReader(metadata_path.read_text(encoding="utf-8").splitlines()))[:-1]
    with metadata_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        external_pilot, "estimate_staff_spacing_full_page", lambda _pixels: _spacing()
    )

    with pytest.raises(ExternalPilotError, match="exactly 12 test"):
        load_external_pilot_pages(tmp_path, source_root=source_root)


def test_external_pilot_executes_reconciles_and_manifests_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    monkeypatch.setattr(
        external_pilot, "estimate_staff_spacing_full_page", lambda _pixels: _spacing()
    )
    pages = load_external_pilot_pages(tmp_path, source_root=source_root)
    output_root = tmp_path / "artifacts"
    freeze_external_pilot_manifest(pages, output_root)

    class FakePretrainedRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(
            self,
            method_id: str,
            pixels: np.ndarray,
            *,
            target_shape: tuple[int, int, int],
            condition_id: str,
        ) -> SimpleNamespace:
            scale = int(condition_id[1])
            output = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
            assert output.shape == target_shape
            return SimpleNamespace(
                pixels=output,
                elapsed_ns=1,
                evidence={
                    "checkpoint_sha256": (
                        "f" * 64 if method_id == external_pilot.PRETRAINED_METHOD_ID else np.nan
                    )
                },
            )

    class FakeAdaptedService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def model_identity(self, scale: int) -> str:
            return str(scale) * 64

        def enhance(self, pixels: np.ndarray, *, scale: int) -> SimpleNamespace:
            output = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
            return SimpleNamespace(
                pixels=output,
                elapsed_seconds=0.000_001,
                checkpoint_sha256=str(scale) * 64,
            )

    monkeypatch.setattr(external_pilot, "PretrainedSRRunner", FakePretrainedRunner)
    monkeypatch.setattr(external_pilot, "ProfessionalInferenceService", FakeAdaptedService)
    monkeypatch.setattr(
        external_pilot,
        "load_scale_normalized_control",
        lambda _root: SimpleNamespace(control_id="test-control", sha256="c" * 64),
    )
    monkeypatch.setattr(
        external_pilot,
        "apply_scale_normalized_degradation",
        lambda pixels, **kwargs: SimpleNamespace(
            pixels=pixels[:: int(kwargs["condition_id"][1]), :: int(kwargs["condition_id"][1])]
        ),
    )
    monkeypatch.setattr(external_pilot, "_git_revision", lambda _root: "d" * 40)
    monkeypatch.setattr(external_pilot, "_git_dirty", lambda _root: False)
    monkeypatch.setattr(
        external_pilot,
        "_source_identity",
        lambda _root: {"files": {"fixture.py": "e" * 64}, "sha256": "e" * 64},
    )

    results = evaluate_external_pilot(tmp_path, pages, output_root=output_root, device="cpu")
    aggregate, paired = analyze_external_pilot(results, output_root)

    assert len(results) == 216
    assert results.source_group_id.nunique() == 12
    assert len(aggregate) == 18
    assert len(paired) == 24
    assert len(list((output_root / "qualitative").glob("*/*"))) == 12
    assert (output_root / "evaluation-identity.json").is_file()
    assert (output_root / "runtime-evidence.json").is_file()
    assert (output_root / "artifact-manifest.json").is_file()

    root = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(root / "scripts/build_professional_pilot_review.py"))
    review_html = output_root / "professional-pilot-v1-review.html"
    builder["build_review"](output_root, review_html)
    rendered = review_html.read_text(encoding="utf-8")
    assert rendered.count('class="case"') == 12
    assert "Aceptable para consulta" in rendered
    assert "Defecto introducido o amplificado" in rendered

    index_rows = list(
        csv.DictReader(
            (output_root / "qualitative-index.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    fixed_cases = sorted(
        {(row["item_id"], row["source_group_id"], row["condition_id"]) for row in index_rows}
    )
    review_payload = {
        "review_id": builder["REVIEW_ID"],
        "case_count": 12,
        "input_manifest_sha256": json.loads(
            (output_root / "evaluation-identity.json").read_text(encoding="utf-8")
        )["input_manifest_sha256"],
        "reviewed_at": "2026-09-03T12:00:00+02:00",
        "assessments": [
            {
                "item_id": item_id,
                "source_group_id": source_group_id,
                "condition_id": condition_id,
                "acceptance": "acceptable",
                "attribution": "no-clear-defect",
                "notes": "",
            }
            for item_id, source_group_id, condition_id in fixed_cases
        ],
    }
    review_json = output_root / "professional-pilot-v1-qualitative-review.json"
    review_json.write_text(json.dumps(review_payload), encoding="utf-8")
    monkeypatch.syspath_prepend(str(root / "scripts"))
    validator = runpy.run_path(str(root / "scripts/validate_professional_pilot_review.py"))
    validation = validator["validate"](
        review_json,
        output_root / "qualitative-index.csv",
        output_root / "evaluation-identity.json",
    )
    assert validation["status"] == "passed"
    assert validation["cases"] == 12


def test_professional_pilot_notebook_is_generated_and_outcome_blind() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "scripts/build_professional_pilot_notebook.py"), run_name="__main__")
    notebook = json.loads(
        (root / "notebooks/05-professional-demonstrator-validation.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert source.index("manifest_path, manifest_sha256 = freeze_external_pilot_manifest") < (
        source.index("results = evaluate_external_pilot")
    )
    assert "RUN_EXTERNAL_PILOT" in source
    assert "Sin resultados ejecutados" in source
    assert "Las conclusiones se redactarán únicamente después" in source
