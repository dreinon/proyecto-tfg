"""Validate the student's six-case EDSR adaptation review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_smb_adaptation_review import DECISIONS

REVIEW_ID = "smb-edsr-finetuning-v1-fixed-qualitative-review"


def validate(review_path: Path, index_path: Path) -> dict[str, object]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("review_id") != REVIEW_ID or review.get("case_count") != 6:
        raise ValueError("review identity or declared case count differs")
    if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"]:
        raise ValueError("review timestamp is missing")

    with index_path.open(encoding="utf-8", newline="") as source:
        expected = {
            (row["item_id"], row["source_group_id"], row["condition_id"])
            for row in csv.DictReader(source)
        }
    assessments = review.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(expected):
        raise ValueError("review must contain exactly the six fixed cases")

    allowed = {value for value, _ in DECISIONS}
    observed: set[tuple[str, str, str]] = set()
    counts = {decision: 0 for decision in sorted(allowed)}
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("every assessment must be an object")
        identity = (
            assessment.get("item_id"),
            assessment.get("source_group_id"),
            assessment.get("condition_id"),
        )
        if identity not in expected or identity in observed:
            raise ValueError("assessment identity is unexpected or duplicated")
        observed.add(identity)
        decision = assessment.get("decision")
        if decision not in allowed:
            raise ValueError("every case requires one allowed decision")
        if not isinstance(assessment.get("notes"), str):
            raise ValueError("assessment notes must be text")
        counts[decision] += 1

    if observed != expected:
        raise ValueError("review does not cover the fixed assignment exactly")
    return {
        "status": "passed",
        "review_id": REVIEW_ID,
        "cases": len(observed),
        "decision_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1/evaluation/qualitative-index.csv"),
    )
    args = parser.parse_args()
    print(
        json.dumps(validate(args.review.resolve(), args.index.resolve()), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
