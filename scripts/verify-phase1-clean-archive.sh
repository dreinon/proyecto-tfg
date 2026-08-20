#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

required_head_paths=(
  data/manifests/smb-evaluation-v1.yaml
  data/sources/smb.yaml
  scripts/verify-phase1-clean-archive.sh
  tests/test_phase1_gap_closure.py
  tests/test_thesis_evidence_promotion.py
)
for path in "${required_head_paths[@]}"; do
  git cat-file -e "HEAD:$path"
  git ls-files --error-unmatch -- "$path" >/dev/null
done

mapfile -t recovery_paths < <(
  git show HEAD:data/manifests/smb-evaluation-v1.yaml |
    UV_OFFLINE=1 uv run --offline --frozen python -c '
import re
import sys
import yaml

pointer = yaml.safe_load(sys.stdin.read())
if not isinstance(pointer, dict) or pointer.get("schema_version") != 2:
    raise SystemExit("HEAD active pointer is not schema v2")
pattern = re.compile(
    r"data/manifests/recovery/canonical-pixel-v2/([0-9a-f]{64})/"
    r"(manifest-recovery\.yaml|manifest-records\.jsonl\.gz)\Z"
)
paths = [str(pointer.get(field, "")) for field in (
    "recovery_descriptor_path", "recovery_records_path"
)]
matches = [pattern.fullmatch(path) for path in paths]
if any(match is None for match in matches):
    raise SystemExit("HEAD active pointer selects a non-canonical recovery path")
if matches[0].group(1) != matches[1].group(1):
    raise SystemExit("HEAD active pointer selects two recovery bundles")
if [match.group(2) for match in matches] != [
    "manifest-recovery.yaml", "manifest-records.jsonl.gz"
]:
    raise SystemExit("HEAD active pointer selects the wrong recovery files")
print(*paths, sep="\n")
'
)
if [[ ${#recovery_paths[@]} -ne 2 ]]; then
  echo "HEAD active pointer did not resolve exactly two recovery files" >&2
  exit 1
fi
for path in "${recovery_paths[@]}"; do
  git cat-file -e "HEAD:$path"
  git ls-files --error-unmatch -- "$path" >/dev/null
done

forbidden_path_pattern='^(data/(raw|interim|processed)|artifacts/smb-manifests/generations|runs|checkpoints|models|outputs|metrics|rankings)/|\.(ckpt|pt|pth|safetensors|png|jpe?g|tiff?|bmp|webp)$|(^|/)(\.env($|\.)|kaggle\.json|credentials?\.json|secrets?\.json|tokens?\.json)$'
head_paths=$(git ls-tree -r --name-only HEAD)
if printf '%s\n' "$head_paths" | rg -i "$forbidden_path_pattern"; then
  echo "forbidden raw, cache, credential, model, checkpoint, outcome, or run path is committed in HEAD" >&2
  exit 1
fi

# The index and worktree are diagnostics only; the archive gate above is rooted in immutable HEAD.
tracked_paths=$(git ls-files)
if printf '%s\n' "$tracked_paths" | rg -i "$forbidden_path_pattern"; then
  echo "forbidden raw, cache, credential, model, checkpoint, outcome, or run path is present in the index" >&2
  exit 1
fi
if git grep -nEI 'hf_[A-Za-z0-9]{20,}|-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}' HEAD -- .; then
  echo "tracked content matches a credential signature" >&2
  exit 1
fi

lock_path=data/manifests/.smb-evaluation-v1.install.lock
git check-ignore -q -- "$lock_path"
if git ls-files --error-unmatch -- "$lock_path" >/dev/null 2>&1; then
  echo "runtime install lock is tracked" >&2
  exit 1
fi

temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/phase1-clean-archive.XXXXXX")
if [[ ! -d "$temporary_root" || "$temporary_root" == / || "$temporary_root" == "$PROJECT_ROOT" ]]; then
  echo "mktemp returned an unsafe verification root" >&2
  exit 1
fi
cleanup() {
  rm -rf -- "$temporary_root"
}
trap cleanup EXIT INT TERM

archive_path="$temporary_root/proyecto-head.tar"
archive_root="$temporary_root/proyecto"
mkdir -p "$archive_root"
git archive --format=tar --output="$archive_path" HEAD
if tar -tf "$archive_path" | rg -i "$forbidden_path_pattern"; then
  echo "forbidden raw, cache, credential, model, checkpoint, outcome, or run path entered the committed archive" >&2
  exit 1
fi
if tar -tf "$archive_path" | rg -qx "$lock_path"; then
  echo "runtime install lock entered the committed archive" >&2
  exit 1
fi
tar -xf "$archive_path" -C "$archive_root"

for path in "${required_head_paths[@]}" "${recovery_paths[@]}"; do
  [[ -f "$archive_root/$path" ]]
done
[[ -x "$archive_root/scripts/verify-phase1-clean-archive.sh" ]]

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN HUGGINGFACEHUB_API_TOKEN
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

cd "$archive_root"
uv run --offline --frozen pytest \
  tests/test_phase1_gap_closure.py::test_active_phase1_gap_closure_reconciles -q

generation_root="$temporary_root/empty-generations"
[[ ! -e "$generation_root" ]]
uv run --offline --frozen python -m score_super_resolution.smb_audit recover-active \
  --manifest-active data/manifests/smb-evaluation-v1.yaml \
  --recovery-descriptor "${recovery_paths[0]}" \
  --recovery-records "${recovery_paths[1]}" \
  --manifest-generation-root "$generation_root"
uv run --offline --frozen python -m score_super_resolution.smb_audit reconcile \
  --manifest-active data/manifests/smb-evaluation-v1.yaml \
  --manifest-generation-root "$generation_root"
uv run --offline --frozen python -c '
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from score_super_resolution import smb_audit

active_path = Path(sys.argv[1])
generation_root = Path(sys.argv[2])
descriptor, rows = smb_audit.resolve_active_manifest(
    active_path=active_path, generation_root=generation_root
)
by_hash = defaultdict(list)
occurrences = Counter()
for row in rows:
    if row["processing_status"] == "processed":
        by_hash[row["image"]["pixel_sha256"]].append(row["item_id"])
    for relation in row["duplicate_relations"]:
        if relation["candidate_type"] == "exact":
            occurrences[tuple(relation["item_ids"])] += 1
derived = {
    pair for members in by_hash.values() for pair in combinations(sorted(members), 2)
}
if set(occurrences) != derived or set(occurrences.values()) - {2}:
    raise SystemExit("active exact-pair evidence does not match canonical framed hashes")
if descriptor["review_inference"]["exact_pair_automated_count"] != len(derived):
    raise SystemExit("active exact-pair descriptor summary disagrees")
report = smb_audit.reconcile_manifest(
    active_path=active_path, generation_root=generation_root
)
expected = {
    "row_count": 685,
    "processed": 685,
    "failed": 0,
    "paired_eligible": 681,
    "exclusion_count": 4,
    "source_group_count": 260,
    "benchmark_state": "AUDITED_LOCKED",
}
if any(report.get(key) != value for key, value in expected.items()):
    raise SystemExit("active reconciliation facts disagree")
print(json.dumps({**expected, "exact_pair_count": len(derived)}, sort_keys=True))
' data/manifests/smb-evaluation-v1.yaml "$generation_root"

echo "Phase 1 clean committed archive verification passed"
