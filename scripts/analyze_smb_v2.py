"""Generate the final reproducible SMB v2 quantitative and qualitative analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from score_super_resolution.qualitative_review import validate_review
from score_super_resolution.smb_analysis import (
    aggregate_metrics,
    analysis_summary,
    load_analysis_inputs,
    paired_bootstrap,
    qualitative_summary,
    runtime_summary,
    sha256_file,
)


def _write_text_atomic(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-evaluation-v2"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/smb-v2-qualitative-review.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/final"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    review_path = args.review.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    review_validation = validate_review(project_root, evaluation_root, review_path)
    inputs = load_analysis_inputs(project_root, evaluation_root, review_path)
    aggregate = aggregate_metrics(inputs)
    paired = paired_bootstrap(inputs)
    runtime = runtime_summary(inputs)
    qualitative_status, qualitative_flags = qualitative_summary(inputs)
    summary = analysis_summary(inputs, aggregate, paired, runtime)

    outputs = {
        "aggregate-metrics.csv": aggregate.to_csv(index=False),
        "paired-bootstrap.csv": paired.to_csv(index=False),
        "runtime-summary.csv": runtime.to_csv(index=False),
        "qualitative-status.csv": qualitative_status.to_csv(index=False),
        "qualitative-flags.csv": qualitative_flags.to_csv(index=False),
        "integrated-summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        "qualitative-review-validation.json": (
            json.dumps(review_validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ),
    }
    for name, payload in outputs.items():
        _write_text_atomic(output_root / name, payload)

    manifest = {
        "schema_version": 1,
        "record_type": "smb-v2-analysis-manifest",
        "experiment_id": inputs.experiment["experiment_id"],
        "inputs": {
            "raw_metrics_sha256": sha256_file(evaluation_root / "raw-metrics.csv"),
            "artifact_manifest_sha256": sha256_file(evaluation_root / "artifact-manifest.json"),
            "qualitative_review_sha256": sha256_file(review_path),
        },
        "outputs": {
            name: {
                "bytes": (output_root / name).stat().st_size,
                "sha256": sha256_file(output_root / name),
            }
            for name in outputs
        },
        "denominators": {
            "independent_sources_per_condition": 64,
            "conditions": 6,
            "methods": 3,
            "paired_metric_comparisons": len(paired),
            "fixed_qualitative_assessments": 18,
        },
    }
    manifest_path = output_root / "analysis-manifest.json"
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"output_root": str(output_root), **manifest["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
