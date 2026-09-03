"""Validate the student's twelve-case external professional-pilot review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_professional_pilot_review import ACCEPTANCE, ATTRIBUTION, REVIEW_ID

from score_super_resolution.edsr_finetuning import write_run_manifest


def validate(review_path: Path, index_path: Path, identity_path: Path) -> dict[str, object]:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if review.get("review_id") != REVIEW_ID or review.get("case_count") != 12:
        raise ValueError("review identity or declared case count differs")
    if review.get("input_manifest_sha256") != identity.get("input_manifest_sha256"):
        raise ValueError("review and external input manifest identities differ")
    if not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"]:
        raise ValueError("review timestamp is missing")
    with index_path.open(encoding="utf-8", newline="") as source:
        expected = {
            (row["item_id"], row["source_group_id"], row["condition_id"])
            for row in csv.DictReader(source)
        }
    assessments = review.get("assessments")
    if (
        not isinstance(assessments, list)
        or len(assessments) != len(expected)
        or len(expected) != 12
    ):
        raise ValueError("review must contain exactly the twelve fixed cases")
    acceptance_allowed = {value for value, _label in ACCEPTANCE}
    attribution_allowed = {value for value, _label in ATTRIBUTION}
    observed: set[tuple[str, str, str]] = set()
    acceptance_counts = {decision: 0 for decision in sorted(acceptance_allowed)}
    attribution_counts = {decision: 0 for decision in sorted(attribution_allowed)}
    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("every assessment must be an object")
        case_identity = (
            assessment.get("item_id"),
            assessment.get("source_group_id"),
            assessment.get("condition_id"),
        )
        if case_identity not in expected or case_identity in observed:
            raise ValueError("assessment identity is unexpected or duplicated")
        observed.add(case_identity)
        acceptance = assessment.get("acceptance")
        attribution = assessment.get("attribution")
        notes = assessment.get("notes")
        if acceptance not in acceptance_allowed or attribution not in attribution_allowed:
            raise ValueError("every case requires allowed acceptance and attribution decisions")
        if not isinstance(notes, str):
            raise ValueError("review notes must be text")
        if (acceptance != "acceptable" or attribution != "no-clear-defect") and not notes.strip():
            raise ValueError("reservations, rejection, or damage require a concrete note")
        acceptance_counts[acceptance] += 1
        attribution_counts[attribution] += 1
    if observed != expected:
        raise ValueError("review does not cover the fixed assignment exactly")
    return {
        "status": "passed",
        "review_id": REVIEW_ID,
        "cases": len(observed),
        "acceptance_counts": acceptance_counts,
        "attribution_counts": attribution_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/professional-pilot-v1/qualitative-index.csv")
    )
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path("artifacts/professional-pilot-v1/evaluation-identity.json"),
    )
    args = parser.parse_args()
    review_path = args.review.resolve()
    index_path = args.index.resolve()
    identity_path = args.identity.resolve()
    artifact_root = index_path.parent
    if review_path.parent != artifact_root or identity_path.parent != artifact_root:
        raise ValueError("review, index, and identity must share the external artifact root")
    validation = validate(review_path, index_path, identity_path)
    validation_path = artifact_root / "qualitative-review-validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = artifact_root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_run_manifest(artifact_root, manifest["run"])
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
