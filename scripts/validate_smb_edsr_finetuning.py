"""Validate the retained SMB EDSR adaptation bundle and write a local QA report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from score_super_resolution.adaptation_split import load_frozen_adaptation_split
from score_super_resolution.baselines import pixel_sha256
from score_super_resolution.edsr_finetuning import (
    ADAPTED_METHOD_ID,
    BICUBIC_METHOD_ID,
    PRETRAINED_METHOD_ID,
    analyze_adaptation_results,
    load_finetuned_edsr,
)

EXPECTED_CONDITIONS = {
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
}
EXPECTED_METHODS = {BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID}
FROZEN_INPUTS = {
    "smb-edsr-finetuning-v1.yaml": Path("configs/experiments/smb-edsr-finetuning-v1.yaml"),
    "staff-scale-score-v2.yaml": Path("configs/degradations/staff-scale-score-v2.yaml"),
    "smb-edsr-finetuning-v1-split.csv": Path("data/adaptation/smb-edsr-finetuning-v1-split.csv"),
    "smb-edsr-finetuning-v1-exclusions.csv": Path(
        "data/adaptation/smb-edsr-finetuning-v1-exclusions.csv"
    ),
    "smb.yaml": Path("data/sources/smb.yaml"),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_bundle(
    project_root: Path, artifact_root: Path, archive_path: Path
) -> dict[str, object]:
    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    archive_path = archive_path.resolve()
    manifest = json.loads((artifact_root / "artifact-manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads((artifact_root / "runtime-evidence.json").read_text(encoding="utf-8"))
    split = load_frozen_adaptation_split(project_root)

    listed = set(manifest["files"])
    actual = {
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if listed != actual:
        raise ValueError("artifact manifest file set differs from the retained bundle")
    for relative, identity in manifest["files"].items():
        path = artifact_root / relative
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"artifact identity differs: {relative}")

    for frozen_name, local_relative in FROZEN_INPUTS.items():
        if _sha256(artifact_root / "frozen-inputs" / frozen_name) != _sha256(
            project_root / local_relative
        ):
            raise ValueError(f"frozen input differs from the local protocol: {frozen_name}")

    reconciliation = runtime.get("revision_reconciliation")
    if (
        manifest["run"].get("git_revision") != runtime.get("git_revision")
        or not isinstance(reconciliation, dict)
        or reconciliation.get("verified_executed_git_revision") != runtime.get("git_revision")
    ):
        raise ValueError("runtime Git revision is not transparently reconciled")
    if runtime.get("split_sha256") != split.split_sha256:
        raise ValueError("runtime split identity differs from the frozen split")

    split_frame = pd.read_csv(artifact_root / "frozen-inputs/smb-edsr-finetuning-v1-split.csv")
    if (
        len(split_frame) != 302
        or split_frame["item_id"].nunique() != 302
        or split_frame.groupby("source_group_id")["partition"].nunique().max() != 1
    ):
        raise ValueError("adaptation split grain or source isolation differs")
    partition_pages = split_frame.groupby("partition").size().to_dict()
    partition_sources = split_frame.groupby("partition")["source_group_id"].nunique().to_dict()
    if partition_pages != {"test": 55, "train": 212, "validation": 35} or partition_sources != {
        "test": 20,
        "train": 45,
        "validation": 13,
    }:
        raise ValueError("adaptation partition denominators differ")

    evaluation_root = artifact_root / "evaluation"
    raw = pd.read_csv(evaluation_root / "raw-metrics.csv")
    aggregate = pd.read_csv(evaluation_root / "aggregate-metrics.csv")
    paired = pd.read_csv(evaluation_root / "paired-bootstrap.csv")
    qualitative = pd.read_csv(evaluation_root / "qualitative-index.csv")
    numeric = raw[["psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb", "runtime_seconds"]]
    if (
        len(raw) != 360
        or len(raw.drop_duplicates(["source_group_id", "condition_id", "method_id"])) != 360
        or raw["source_group_id"].nunique() != 20
        or set(raw["condition_id"]) != EXPECTED_CONDITIONS
        or set(raw["method_id"]) != EXPECTED_METHODS
        or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        or (raw["runtime_seconds"] < 0).any()
    ):
        raise ValueError("raw adaptation metrics violate their frozen contract")

    recomputed_aggregate, recomputed_paired = analyze_adaptation_results(
        raw,
        seed=20260903,
        repetitions=2000,
    )
    assert_frame_equal(
        recomputed_aggregate,
        aggregate,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )
    assert_frame_equal(
        recomputed_paired,
        paired,
        check_exact=False,
        rtol=1e-13,
        atol=1e-13,
    )

    if (
        len(qualitative) != 30
        or qualitative[["item_id", "condition_id"]].drop_duplicates().shape[0] != 6
    ):
        raise ValueError("qualitative evidence denominators differ")
    for row in qualitative.itertuples(index=False):
        bgr = cv2.imread(str(evaluation_root / row.path), cv2.IMREAD_COLOR)
        if bgr is None or pixel_sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) != row.pixel_sha256:
            raise ValueError(f"qualitative pixel identity differs: {row.path}")

    checkpoint_hashes: dict[str, str] = {}
    checkpoint_selection: dict[str, dict[str, object]] = {}
    for scale in (2, 4):
        checkpoint_path = artifact_root / "training" / f"edsr-smb-finetuned-v1-x{scale}.pt"
        _, digest = load_finetuned_edsr(
            checkpoint_path,
            split,
            scale=scale,
            device="cpu",
        )
        checkpoint_hashes[str(scale)] = digest
        history = pd.read_csv(artifact_root / "training" / f"training-history-x{scale}.csv")
        best = history.loc[history["validation_l1"].idxmin()]
        checkpoint_selection[str(scale)] = {
            "selected_step": int(best["step"]),
            "validation_l1": float(best["validation_l1"]),
            "completed_step": int(history["step"].max()),
        }

    adaptation = paired[paired["comparator_id"] == PRETRAINED_METHOD_ID]
    psnr = adaptation[adaptation["metric"] == "psnr_y"]
    ssim = adaptation[adaptation["metric"] == "ssim_y"]
    report: dict[str, object] = {
        "schema_version": 1,
        "record_type": "smb-edsr-finetuning-validation",
        "status": "passed",
        "assessment": "ready-with-declared-scope",
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "identities": {
            "git_revision": runtime["git_revision"],
            "revision_reconciliation": reconciliation,
            "source_revision": split.source_revision,
            "split_sha256": split.split_sha256,
            "checkpoint_sha256": checkpoint_hashes,
        },
        "denominators": {
            "manifest_files": len(actual),
            "partition_pages": partition_pages,
            "partition_sources": partition_sources,
            "raw_rows": len(raw),
            "paired_rows": len(paired),
            "qualitative_pngs": len(qualitative),
        },
        "checkpoint_selection": checkpoint_selection,
        "adapted_minus_pretrained": {
            "psnr_y_mean_delta_range_db": [
                float(psnr["mean_delta"].min()),
                float(psnr["mean_delta"].max()),
            ],
            "ssim_y_mean_delta_range": [
                float(ssim["mean_delta"].min()),
                float(ssim["mean_delta"].max()),
            ],
            "all_12_intervals_exclude_zero": bool(adaptation["interval_excludes_zero"].all()),
        },
        "checks": {
            "archive_readable": True,
            "manifest_identities": True,
            "frozen_inputs_match": True,
            "source_disjoint_partitions": True,
            "raw_metric_grain": True,
            "aggregate_recomputed": True,
            "bootstrap_recomputed": True,
            "qualitative_pixels_verified": True,
            "checkpoint_identities_verified": True,
        },
        "required_caveats": [
            (
                "The official SMB split is test; train, validation, and test are "
                "project-defined roles for this secondary study."
            ),
            (
                "The primary 64-work pretrained benchmark remains separate and excluded "
                "from adaptation."
            ),
            (
                "The result supports within-SMB matched-degradation adaptation only, not real "
                "scans, other corpora, OMR, archival replacement, or deployment."
            ),
            (
                "Six fixed qualitative cases detect failure modes but do not estimate "
                "their prevalence."
            ),
            "The x4 validation loss was still improving at the frozen 2500-step boundary.",
            (
                "Redistribution or commercial use of SMB-derived checkpoints requires "
                "separate rights review."
            ),
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1-validation.json"),
    )
    args = parser.parse_args()
    report = validate_bundle(args.project_root, args.artifact_root, args.archive)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
