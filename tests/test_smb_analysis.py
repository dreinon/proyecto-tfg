from __future__ import annotations

import pandas as pd
import pytest

from score_super_resolution.smb_analysis import (
    AnalysisInputs,
    SmbAnalysisError,
    aggregate_metrics,
    paired_bootstrap,
)


def _inputs() -> AnalysisInputs:
    methods = [
        "bicubic-opencv-v1",
        "edsr-baseline-official-v1",
        "swinir-lightweight-official-v1",
    ]
    rows = []
    offsets = dict(zip(methods, (0.0, 1.0, 2.0), strict=True))
    for condition, scale, profile in (
        ("x2-clean", 2, "clean"),
        ("x2-moderate", 2, "moderate"),
        ("x2-strong", 2, "strong"),
        ("x4-clean", 4, "clean"),
        ("x4-moderate", 4, "moderate"),
        ("x4-strong", 4, "strong"),
    ):
        for source_index in range(4):
            for method in methods:
                rows.append(
                    {
                        "item_id": f"item-{source_index}",
                        "source_group_id": f"source-{source_index}",
                        "condition_id": condition,
                        "scale": scale,
                        "profile": profile,
                        "method_id": method,
                        "psnr_y": 20 + source_index + offsets[method],
                        "ssim_y": 0.7 + 0.01 * source_index + 0.01 * offsets[method],
                        "psnr_rgb": 19 + source_index + offsets[method],
                        "ssim_rgb": 0.69 + 0.01 * source_index + 0.01 * offsets[method],
                        "runtime_seconds": 0.1 + offsets[method],
                    }
                )
    experiment = {
        "experiment_id": "fixture",
        "conditions": [
            "x2-clean",
            "x2-moderate",
            "x2-strong",
            "x4-clean",
            "x4-moderate",
            "x4-strong",
        ],
        "methods": methods,
        "sample": {"item_count": 4, "independent_source_group_count": 4},
        "metrics": {"bootstrap_seed": 17, "bootstrap_repetitions": 200},
    }
    return AnalysisInputs(experiment, pd.DataFrame.from_records(rows), {"assessments": []})


def test_paired_bootstrap_uses_source_level_differences_and_is_reproducible() -> None:
    first = paired_bootstrap(_inputs())
    second = paired_bootstrap(_inputs())

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 36
    edsr_psnr = first[
        (first["condition_id"] == "x2-clean")
        & (first["metric"] == "psnr_y")
        & (first["method_id"] == "edsr-baseline-official-v1")
        & (first["comparator_id"] == "bicubic-opencv-v1")
    ].iloc[0]
    assert edsr_psnr["mean_delta"] == pytest.approx(1.0)
    assert edsr_psnr["ci95_low"] == pytest.approx(1.0)
    assert edsr_psnr["ci95_high"] == pytest.approx(1.0)
    assert edsr_psnr["sources_improved"] == 4


def test_aggregate_keeps_primary_metrics_and_runtime_separate() -> None:
    aggregate = aggregate_metrics(_inputs())

    assert len(aggregate) == 18
    assert set(aggregate["sources"]) == {4}
    assert "runtime_seconds_median" in aggregate


def test_bootstrap_rejects_too_few_repetitions() -> None:
    with pytest.raises(SmbAnalysisError, match="too small"):
        paired_bootstrap(_inputs(), repetitions=10)
