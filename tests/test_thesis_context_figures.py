from __future__ import annotations

import gzip
import json
import runpy
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/generate_thesis_context_figures.py"
PROJECT_ROOT = SCRIPT.parents[1]


def test_manifest_reader_extracts_only_non_image_metadata(tmp_path: Path) -> None:
    module = runpy.run_path(SCRIPT)
    path = tmp_path / "manifest.jsonl.gz"
    records = [
        {
            "source_group_id": "work-a",
            "image": {"decoded_width": 1000, "decoded_height": 1500},
            "annotations": {"region_count": 6},
            "paired_eligible": True,
            "paired_ineligibility_reason": None,
        },
        {
            "source_group_id": "work-a",
            "image": {"decoded_width": 1100, "decoded_height": 1600},
            "annotations": {"region_count": 8},
            "paired_eligible": False,
            "paired_ineligibility_reason": "missing_required_region_text",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")

    result = module["read_manifest"](path)

    assert result[["source_group_id", "width", "height", "region_count"]].to_dict(
        orient="records"
    ) == [
        {"source_group_id": "work-a", "width": 1000, "height": 1500, "region_count": 6},
        {"source_group_id": "work-a", "width": 1100, "height": 1600, "region_count": 8},
    ]
    assert result["paired_eligible"].tolist() == [True, False]
    assert pd.isna(result.loc[0, "ineligibility_reason"])
    assert result.loc[1, "ineligibility_reason"] == "missing_required_region_text"


def test_context_figures_export_vector_and_preview_formats(tmp_path: Path) -> None:
    module = runpy.run_path(SCRIPT)
    records = pd.DataFrame(
        {
            "source_group_id": ["a", "a", "b"],
            "width": [900, 1000, 1200],
            "height": [1400, 1500, 1700],
            "region_count": [4, 6, 8],
        }
    )
    sample = pd.DataFrame({"staff_spacing_px": [7.0, 10.0, 18.0]})
    effort = pd.DataFrame(
        {
            "entry_id": [
                "EFF-P1",
                "EFF-P2",
                "EFF-P3",
                "EFF-P4",
                "EFF-P5-CURRENT",
                "EFF-P5-REMAINING",
            ],
            "estimate_hours": [64, 73, 51, 82, 56, 22],
            "low_hours": [58, 66, 46, 74, 50, 18],
            "high_hours": [70, 80, 56, 90, 62, 26],
        }
    )

    paths = [
        *module["dataset_profile_figure"](records, tmp_path),
        *module["staff_scale_figure"](sample, tmp_path),
        *module["effort_figure"](effort, tmp_path),
    ]

    assert len(paths) == 9
    assert {path.suffix for path in paths} == {".pdf", ".png", ".svg"}
    assert all(path.stat().st_size > 0 for path in paths)


def test_default_sources_match_reported_thesis_denominators() -> None:
    module = runpy.run_path(SCRIPT)
    records = module["read_manifest"](PROJECT_ROOT / module["DEFAULT_MANIFEST"])
    sample = pd.read_csv(PROJECT_ROOT / "data/audits/smb-evaluation-sample-v2.csv")

    assert len(records) == 685
    assert records["source_group_id"].nunique() == 260
    assert int(records["paired_eligible"].sum()) == 618
    assert len(sample) == 64
    assert sample["source_group_id"].nunique() == 64
    assert sample["staff_spacing_px"].between(6, 20).all()
