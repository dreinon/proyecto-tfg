"""Validate the completed fixed SMB v2 qualitative review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_super_resolution.qualitative_review import validate_review


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
    args = parser.parse_args()
    report = validate_review(args.project_root, args.evaluation_root, args.review)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
