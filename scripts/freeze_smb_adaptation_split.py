"""Freeze the input-only, source-disjoint split for the EDSR adaptation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_super_resolution.adaptation_split import (
    EXCLUSIONS_RELATIVE_PATH,
    SPLIT_RELATIVE_PATH,
    derive_adaptation_split,
)
from score_super_resolution.benchmark_policy import BenchmarkPurpose
from score_super_resolution.smb import load_smb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    dataset = load_smb(purpose=BenchmarkPurpose.CONTENT_AUDIT)
    split_bytes, exclusion_bytes, summary = derive_adaptation_split(project_root, dataset)
    for relative_path, payload in (
        (SPLIT_RELATIVE_PATH, split_bytes),
        (EXCLUSIONS_RELATIVE_PATH, exclusion_bytes),
    ):
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
