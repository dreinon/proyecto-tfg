from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pandas as pd


def _build_artifact(analysis_root: Path) -> dict[str, Any]:
    script = Path(__file__).parents[1] / "scripts/build_smb_v2_report.py"
    return runpy.run_path(script)["build_artifact"](analysis_root)


def test_report_artifact_has_auditable_technical_reading_path(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "condition_id": "x2-clean",
                "scale": 2,
                "profile": "clean",
                "method_id": method,
                "sources": 64,
                "psnr_y_mean": 20.0 + index,
                "ssim_y_mean": 0.8 + 0.01 * index,
                "psnr_rgb_mean": 19.0 + index,
                "ssim_rgb_mean": 0.79 + 0.01 * index,
                "runtime_seconds_median": 0.1 + index,
            }
            for index, method in enumerate(
                (
                    "bicubic-opencv-v1",
                    "edsr-baseline-official-v1",
                    "swinir-lightweight-official-v1",
                )
            )
        ]
    ).to_csv(tmp_path / "aggregate-metrics.csv", index=False)
    paired = pd.DataFrame(
        [
            {
                "condition_id": "x2-clean",
                "scale": 2,
                "profile": "clean",
                "metric": "psnr_y",
                "method_id": method,
                "comparator_id": comparator,
                "sources": 64,
                "mean_delta": delta,
                "ci95_low": delta - 0.1,
                "ci95_high": delta + 0.1,
                "sources_improved": 64 if delta > 0 else 0,
                "sources_worsened": 0 if delta > 0 else 64,
                "sources_tied": 0,
                "bootstrap_seed": 20260831,
                "bootstrap_repetitions": 2000,
                "interval_excludes_zero": True,
            }
            for method, comparator, delta in (
                ("edsr-baseline-official-v1", "bicubic-opencv-v1", 1.0),
                ("swinir-lightweight-official-v1", "bicubic-opencv-v1", 1.2),
                ("swinir-lightweight-official-v1", "edsr-baseline-official-v1", 0.2),
            )
        ]
    )
    paired.to_csv(tmp_path / "paired-bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_id": "x2-clean",
                "scale": 2,
                "profile": "clean",
                "method_id": "edsr-baseline-official-v1",
                "sources": 64,
                "runtime_seconds_median": 0.5,
                "runtime_vs_bicubic": 100.0,
                "runtime_vs_edsr": 1.0,
            }
        ]
    ).to_csv(tmp_path / "runtime-summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "method_id": "edsr-baseline-official-v1",
                "status": "issues-observed",
                "fixed_case_count": 4,
            }
        ]
    ).to_csv(tmp_path / "qualitative-status.csv", index=False)
    pd.DataFrame(
        [
            {
                "method_id": "edsr-baseline-official-v1",
                "flag_id": "altered-or-missing-symbol",
                "fixed_case_count": 4,
            }
        ]
    ).to_csv(tmp_path / "qualitative-flags.csv", index=False)
    (tmp_path / "integrated-summary.json").write_text(
        json.dumps({"experiment_id": "fixture"}), encoding="utf-8"
    )

    artifact = _build_artifact(tmp_path)

    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# ")
    assert len(artifact["manifest"]["charts"]) == 2
    assert all(chart["sourceId"] for chart in artifact["manifest"]["charts"])
    assert artifact["snapshot"]["status"] == "ready"
    assert artifact["analysis_context"]["omissions"]
