"""Freeze the fresh work-disjoint SMB v2 sample before any v2 model output exists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from score_super_resolution.benchmark_policy import BenchmarkPurpose, BenchmarkState
from score_super_resolution.comparison import as_rgb8, ensure_manifest_generation
from score_super_resolution.smb import load_smb
from score_super_resolution.smb_audit import resolve_active_manifest
from score_super_resolution.staff_scale import ESTIMATOR_ID, StaffScaleError, estimate_staff_spacing

SAMPLE_SEED = 20260830
SAMPLE_SIZE = 64


def _rank(domain: str, *values: object) -> str:
    payload = "\0".join((domain, str(SAMPLE_SEED), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_csv_once(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze(project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    generation_root = ensure_manifest_generation(project_root)
    _, manifest_rows = resolve_active_manifest(
        active_path=project_root / "data/manifests/smb-evaluation-v1.yaml",
        generation_root=generation_root,
    )
    manifest_by_item = {str(row["item_id"]): row for row in manifest_rows}
    with (project_root / "data/audits/smb-visual-sample-v1.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        v1_rows = list(csv.DictReader(source))
    v1_counts = Counter(str(manifest_by_item[row["item_id"]]["source_group_id"]) for row in v1_rows)
    v1_group_rows = [
        {"source_group_id": source_group_id, "v1_page_count": page_count}
        for source_group_id, page_count in sorted(v1_counts.items())
    ]

    candidate_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        source_group_id = str(row["source_group_id"])
        duplicate_summary = row["duplicate_summary"]
        if (
            row["processing_status"] == "processed"
            and source_group_id not in v1_counts
            and duplicate_summary["duplicate_relation_count"] == 0
            and duplicate_summary["pending_relation_count"] == 0
        ):
            candidate_groups[source_group_id].append(row)

    unlock = json.loads(
        (project_root / "configs/smb-evaluation-v1/unlock.json").read_text(encoding="utf-8")
    )
    dataset = load_smb(
        purpose=BenchmarkPurpose.INFERENCE,
        state=BenchmarkState.EVALUATION_UNLOCKED,
        unlock_record=unlock,
        project_root=project_root,
        manifest_generation_root=generation_root,
    )
    selected: list[dict[str, object]] = []
    rejected_estimates = 0
    ranked_groups = sorted(
        candidate_groups, key=lambda value: (_rank("smb-v2-source-group-rank-v1", value), value)
    )
    for source_group_id in ranked_groups:
        pages = sorted(
            candidate_groups[source_group_id],
            key=lambda row: (
                _rank(
                    "smb-v2-page-rank-v1", source_group_id, row["upstream_index"], row["item_id"]
                ),
                row["item_id"],
            ),
        )
        accepted: dict[str, object] | None = None
        for page in pages:
            upstream_index = int(page["upstream_index"])
            dataset_row = dataset[upstream_index]
            try:
                estimate = estimate_staff_spacing(
                    as_rgb8(dataset_row["image"]), dataset_row["regions"]
                )
            except StaffScaleError:
                rejected_estimates += 1
                continue
            accepted = {
                "upstream_index": upstream_index,
                "item_id": str(page["item_id"]),
                "source_group_id": source_group_id,
                "staff_spacing_px": f"{estimate.spacing_px:.6f}",
                "estimator_id": ESTIMATOR_ID,
                "staff_sequence_count": estimate.sequence_count,
                "selection_rank": len(selected) + 1,
            }
            break
        if accepted is not None:
            selected.append(accepted)
        if len(selected) == SAMPLE_SIZE:
            break
    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError("fewer than 64 fresh source groups passed the frozen staff estimator")

    v1_groups_path = project_root / "data/audits/smb-evaluation-v1-source-groups.csv"
    sample_path = project_root / "data/audits/smb-evaluation-sample-v2.csv"
    _write_csv_once(
        v1_groups_path,
        ["source_group_id", "v1_page_count"],
        v1_group_rows,
    )
    _write_csv_once(
        sample_path,
        [
            "upstream_index",
            "item_id",
            "source_group_id",
            "staff_spacing_px",
            "estimator_id",
            "staff_sequence_count",
            "selection_rank",
        ],
        selected,
    )
    return {
        "sample_path": str(sample_path.relative_to(project_root)),
        "sample_sha256": _sha256(sample_path),
        "sample_pages": len(selected),
        "sample_source_groups": len({row["source_group_id"] for row in selected}),
        "v1_true_source_groups": len(v1_counts),
        "v1_source_groups_sha256": _sha256(v1_groups_path),
        "work_disjoint": not ({row["source_group_id"] for row in selected} & set(v1_counts)),
        "estimator_rejections_before_freeze": rejected_estimates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    print(json.dumps(freeze(arguments.project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
