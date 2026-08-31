"""Final paired analysis for the frozen SMB v2 evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class SmbAnalysisError(ValueError):
    """Reject incomplete, mismatched, or non-frozen analysis inputs."""


PRIMARY_METRICS = ("psnr_y", "ssim_y")
METHOD_PAIRS = (
    ("edsr-baseline-official-v1", "bicubic-opencv-v1"),
    ("swinir-lightweight-official-v1", "bicubic-opencv-v1"),
    ("swinir-lightweight-official-v1", "edsr-baseline-official-v1"),
)


@dataclass(frozen=True)
class AnalysisInputs:
    experiment: dict[str, Any]
    metrics: pd.DataFrame
    qualitative_review: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    if not path.is_file() or path.is_symlink():
        raise SmbAnalysisError(f"Input is not a regular file: {path.name}")
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmbAnalysisError(f"Cannot read {path.name}") from error
    if not isinstance(payload, dict):
        raise SmbAnalysisError(f"{path.name} must contain an object")
    return payload


def load_analysis_inputs(
    project_root: Path,
    evaluation_root: Path,
    review_path: Path,
) -> AnalysisInputs:
    """Load and reconcile the frozen experiment, metrics, and human review."""

    experiment_path = project_root / "configs/experiments/smb-pretrained-evaluation-v2.yaml"
    try:
        experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(evaluation_root / "raw-metrics.csv")
    except (OSError, UnicodeError, yaml.YAMLError, pd.errors.ParserError) as error:
        raise SmbAnalysisError("Cannot load frozen SMB v2 inputs") from error
    if not isinstance(experiment, dict):
        raise SmbAnalysisError("Frozen experiment must contain an object")
    review = _read_json(review_path)

    required_columns = {
        "item_id",
        "source_group_id",
        "condition_id",
        "scale",
        "profile",
        "method_id",
        "psnr_y",
        "ssim_y",
        "psnr_rgb",
        "ssim_rgb",
        "runtime_seconds",
    }
    if not required_columns <= set(metrics.columns):
        raise SmbAnalysisError("Raw metrics columns differ from the analysis contract")
    expected_conditions = list(experiment["conditions"])
    expected_methods = list(experiment["methods"])
    if set(metrics["condition_id"]) != set(expected_conditions):
        raise SmbAnalysisError("Raw metric conditions differ from the frozen experiment")
    if set(metrics["method_id"]) != set(expected_methods):
        raise SmbAnalysisError("Raw metric methods differ from the frozen experiment")
    expected_rows = (
        int(experiment["sample"]["item_count"]) * len(expected_conditions) * len(expected_methods)
    )
    if len(metrics) != expected_rows:
        raise SmbAnalysisError("Raw metric denominator differs from the frozen experiment")
    key = ["source_group_id", "condition_id", "method_id"]
    if metrics.duplicated(key).any():
        raise SmbAnalysisError("Raw metrics repeat a paired source-condition-method tuple")
    counts = metrics.groupby(["source_group_id", "condition_id"])["method_id"].nunique()
    if not (counts == len(expected_methods)).all():
        raise SmbAnalysisError("A paired source-condition panel is incomplete")
    numeric = metrics[[*PRIMARY_METRICS, "psnr_rgb", "ssim_rgb", "runtime_seconds"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise SmbAnalysisError("Raw metrics contain non-finite values")
    if not metrics["ssim_y"].between(0, 1).all() or not metrics["ssim_rgb"].between(0, 1).all():
        raise SmbAnalysisError("SSIM values are outside [0, 1]")
    if (metrics["runtime_seconds"] < 0).any():
        raise SmbAnalysisError("Runtime values must be non-negative")
    if review.get("complete") is not True or len(review.get("assessments", [])) != 18:
        raise SmbAnalysisError("Qualitative review is incomplete")
    if review.get("assignment_sha256") != experiment["qualitative"]["assignment_sha256"]:
        raise SmbAnalysisError("Qualitative review assignment differs from the frozen experiment")
    return AnalysisInputs(experiment=experiment, metrics=metrics, qualitative_review=review)


def aggregate_metrics(inputs: AnalysisInputs) -> pd.DataFrame:
    """Aggregate page metrics at the declared one-page-per-source analysis grain."""

    result = (
        inputs.metrics.groupby(["condition_id", "scale", "profile", "method_id"], as_index=False)
        .agg(
            sources=("source_group_id", "nunique"),
            psnr_y_mean=("psnr_y", "mean"),
            ssim_y_mean=("ssim_y", "mean"),
            psnr_rgb_mean=("psnr_rgb", "mean"),
            ssim_rgb_mean=("ssim_rgb", "mean"),
            runtime_seconds_median=("runtime_seconds", "median"),
        )
        .sort_values(["scale", "profile", "method_id"])
        .reset_index(drop=True)
    )
    condition_order = {value: index for index, value in enumerate(inputs.experiment["conditions"])}
    method_order = {value: index for index, value in enumerate(inputs.experiment["methods"])}
    result["_condition_order"] = result["condition_id"].map(condition_order)
    result["_method_order"] = result["method_id"].map(method_order)
    return (
        result.sort_values(["_condition_order", "_method_order"])
        .drop(columns=["_condition_order", "_method_order"])
        .reset_index(drop=True)
    )


def paired_bootstrap(
    inputs: AnalysisInputs,
    *,
    seed: int | None = None,
    repetitions: int | None = None,
) -> pd.DataFrame:
    """Estimate paired mean differences and percentile intervals by source group."""

    seed = int(seed if seed is not None else inputs.experiment["metrics"]["bootstrap_seed"])
    repetitions = int(
        repetitions
        if repetitions is not None
        else inputs.experiment["metrics"]["bootstrap_repetitions"]
    )
    if repetitions < 100:
        raise SmbAnalysisError("Bootstrap repetitions are too small for interval estimation")
    records: list[dict[str, Any]] = []
    for condition_id in inputs.experiment["conditions"]:
        condition = inputs.metrics[inputs.metrics["condition_id"] == condition_id]
        sources = sorted(condition["source_group_id"].unique())
        if len(sources) != inputs.experiment["sample"]["independent_source_group_count"]:
            raise SmbAnalysisError("Condition has the wrong independent-source denominator")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(sources), size=(repetitions, len(sources)))
        scale = int(condition["scale"].iloc[0])
        profile = str(condition["profile"].iloc[0])
        for metric in PRIMARY_METRICS:
            pivot = condition.pivot(
                index="source_group_id", columns="method_id", values=metric
            ).loc[sources]
            for method, comparator in METHOD_PAIRS:
                differences = (pivot[method] - pivot[comparator]).to_numpy(dtype=float)
                bootstrap_means = differences[indices].mean(axis=1)
                low, high = np.quantile(bootstrap_means, [0.025, 0.975])
                records.append(
                    {
                        "condition_id": condition_id,
                        "scale": scale,
                        "profile": profile,
                        "metric": metric,
                        "method_id": method,
                        "comparator_id": comparator,
                        "sources": len(sources),
                        "mean_delta": float(differences.mean()),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "sources_improved": int((differences > 0).sum()),
                        "sources_worsened": int((differences < 0).sum()),
                        "sources_tied": int((differences == 0).sum()),
                        "bootstrap_seed": seed,
                        "bootstrap_repetitions": repetitions,
                        "interval_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
    return pd.DataFrame.from_records(records)


def runtime_summary(inputs: AnalysisInputs) -> pd.DataFrame:
    """Summarize measured inference time without treating it as portable hardware performance."""

    aggregate = aggregate_metrics(inputs)[
        ["condition_id", "scale", "profile", "method_id", "sources", "runtime_seconds_median"]
    ].copy()
    baseline = aggregate[aggregate["method_id"] == "bicubic-opencv-v1"].set_index("condition_id")[
        "runtime_seconds_median"
    ]
    edsr = aggregate[aggregate["method_id"] == "edsr-baseline-official-v1"].set_index(
        "condition_id"
    )["runtime_seconds_median"]
    aggregate["runtime_vs_bicubic"] = aggregate.apply(
        lambda row: row["runtime_seconds_median"] / baseline[row["condition_id"]], axis=1
    )
    aggregate["runtime_vs_edsr"] = aggregate.apply(
        lambda row: row["runtime_seconds_median"] / edsr[row["condition_id"]], axis=1
    )
    return aggregate


def qualitative_summary(inputs: AnalysisInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fixed-case review decisions without estimating population prevalence."""

    assessments = pd.DataFrame.from_records(inputs.qualitative_review["assessments"])
    statuses = (
        assessments.groupby(["method_id", "status"], as_index=False)
        .size()
        .rename(columns={"size": "fixed_case_count"})
    )
    flag_rows = [
        {"method_id": row.method_id, "flag_id": flag}
        for row in assessments.itertuples()
        for flag in row.flags
    ]
    flags = pd.DataFrame.from_records(flag_rows)
    if flags.empty:
        flags = pd.DataFrame(columns=["method_id", "flag_id", "fixed_case_count"])
    else:
        flags = (
            flags.groupby(["method_id", "flag_id"], as_index=False)
            .size()
            .rename(columns={"size": "fixed_case_count"})
        )
    return statuses, flags


def analysis_summary(
    inputs: AnalysisInputs,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    runtime: pd.DataFrame,
) -> dict[str, Any]:
    """Return decision-oriented, machine-readable findings for report generation."""

    learned_vs_bicubic = paired[paired["comparator_id"] == "bicubic-opencv-v1"]
    learned_evidence = learned_vs_bicubic.groupby(["condition_id", "metric"])[
        "interval_excludes_zero"
    ].all()
    direct = paired[paired["comparator_id"] == "edsr-baseline-official-v1"]
    review = inputs.qualitative_review["assessments"]
    status_by_method = {
        method: dict(Counter(row["status"] for row in review if row["method_id"] == method))
        for method in inputs.experiment["methods"]
    }
    return {
        "schema_version": 1,
        "record_type": "smb-v2-integrated-analysis",
        "experiment_id": inputs.experiment["experiment_id"],
        "independent_sources_per_condition": int(aggregate["sources"].min()),
        "bootstrap": {
            "unit": "source_group_id",
            "seed": int(inputs.experiment["metrics"]["bootstrap_seed"]),
            "repetitions": int(inputs.experiment["metrics"]["bootstrap_repetitions"]),
            "interval": "percentile-95",
        },
        "learned_methods_both_primary_intervals_above_bicubic": {
            condition: bool(
                learned_evidence.get((condition, "psnr_y"), False)
                and learned_evidence.get((condition, "ssim_y"), False)
            )
            for condition in inputs.experiment["conditions"]
        },
        "swinir_minus_edsr": direct.to_dict(orient="records"),
        "runtime_median_seconds_range": {
            method: {
                "min": float(rows["runtime_seconds_median"].min()),
                "max": float(rows["runtime_seconds_median"].max()),
            }
            for method, rows in runtime.groupby("method_id")
        },
        "qualitative_fixed_case_status_by_method": status_by_method,
        "qualitative_scope": (
            "Six outcome-independent fixed cases, one per condition; counts describe reviewed "
            "cases and are not prevalence estimates for SMB."
        ),
    }
