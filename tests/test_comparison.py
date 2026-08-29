from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    BenchmarkState,
    assert_smb_purpose_allowed,
)
from score_super_resolution.comparison import (
    CONDITIONS,
    QUALITATIVE_ASSIGNMENT,
    SAMPLE_SHA256,
    ensure_manifest_generation,
    fidelity_metrics,
    load_evaluation_sample,
    load_frozen_degradation,
    technical_smoke_image,
)
from score_super_resolution.contracts import validate_instance
from score_super_resolution.identities import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
FREEZE_CONTROL_PATHS = tuple(
    path
    for path in sorted((ROOT / "configs/smb-evaluation-v1").glob("*.json"))
    if path.name not in {"human-review.json", "unlock.json"}
)


def test_evaluation_sample_is_fixed_outcome_blind_and_source_independent() -> None:
    sample = load_evaluation_sample(ROOT)
    assert len(sample) == 64
    assert len({row.item_id for row in sample}) == 64
    assert len({row.source_group_id for row in sample}) == 64
    assert len(SAMPLE_SHA256) == 64
    assert {item_id for item_id, _ in QUALITATIVE_ASSIGNMENT} <= {row.item_id for row in sample}
    assert tuple(condition for _, condition in QUALITATIVE_ASSIGNMENT) == CONDITIONS
    assert canonical_sha256([list(value) for value in QUALITATIVE_ASSIGNMENT]) == (
        "d5a60c080d69bbfdf955f3a7cce1a29067595119bc0bb8b65404c2906efb1e02"
    )


def test_frozen_degradation_is_the_accepted_six_cell_control() -> None:
    control = load_frozen_degradation(ROOT)
    assert control.status == "frozen"
    assert control.condition_ids == CONDITIONS
    assert control.master_seed == 20260821
    assert control.sha256 == "6a61d9a28d2524c9b4e8b2138a23d93bb6fe10e37b8cd1d439cb35dcbdcd9949"


def test_fidelity_metrics_are_exact_for_identical_rgb8_images() -> None:
    pixels = technical_smoke_image()
    metrics = fidelity_metrics(pixels, pixels.copy())
    assert np.isinf(metrics["psnr_y"])
    assert np.isinf(metrics["psnr_rgb"])
    assert metrics["ssim_y"] == pytest.approx(1.0)
    assert metrics["ssim_rgb"] == pytest.approx(1.0)


def test_experiment_freezes_the_compact_comparison_without_tuning() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs/experiments/smb-pretrained-evaluation-v1.yaml").read_text(encoding="utf-8")
    )
    assert payload["status"] == "frozen"
    assert payload["claim_boundary"] == "fixed-outcome-blind-64-page-smb-sample"
    assert payload["dataset"]["role"] == "evaluation-only"
    assert payload["sample"]["item_count"] == 64
    assert payload["methods"] == [
        "bicubic-opencv-v1",
        "edsr-baseline-official-v1",
        "swinir-lightweight-official-v1",
    ]
    assert tuple(payload["conditions"]) == CONDITIONS
    assert "tuning" not in json.dumps(payload).casefold()


def test_ignored_manifest_generation_recovers_from_tracked_bundle(tmp_path: Path) -> None:
    active_relative = Path("data/manifests/smb-evaluation-v1.yaml")
    destination_active = tmp_path / active_relative
    destination_active.parent.mkdir(parents=True)
    shutil.copy2(ROOT / active_relative, destination_active)
    pointer = yaml.safe_load(destination_active.read_text(encoding="utf-8"))
    for field in ("recovery_descriptor_path", "recovery_records_path"):
        relative = Path(pointer[field])
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    generation_root = ensure_manifest_generation(tmp_path)
    generation_path = generation_root / pointer["generation_id"]

    assert (generation_path / "manifest-descriptor.yaml").is_file()
    assert (generation_path / "manifest-records.jsonl").is_file()


def test_human_approved_unlock_resolves_every_real_prerequisite() -> None:
    unlock = json.loads(
        (ROOT / "configs/smb-evaluation-v1/unlock.json").read_text(encoding="utf-8")
    )
    validate_instance("smb-evaluation-unlock", unlock, version=1)
    source = yaml.safe_load((ROOT / "data/sources/smb.yaml").read_text(encoding="utf-8"))
    calls: list[str] = []

    generation_root = ensure_manifest_generation(ROOT)
    result = assert_smb_purpose_allowed(
        source_descriptor=source,
        purpose=BenchmarkPurpose.INFERENCE,
        state=BenchmarkState.EVALUATION_UNLOCKED,
        unlock_record=unlock,
        project_root=ROOT,
        manifest_generation_root=generation_root,
        callback=lambda: calls.append("allowed") or "ready",
    )

    assert result == "ready"
    assert calls == ["allowed"]
    assert unlock["reviewer"] == "Daniel Reinón García"
    assert unlock["reviewed_at"] == "2026-08-30"


@pytest.mark.parametrize(
    "path",
    FREEZE_CONTROL_PATHS,
)
def test_freeze_controls_satisfy_the_policy_schema(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_instance("smb-freeze-control", payload, version=1)
    assert payload["status"] == "frozen"
    assert payload["control_id"] == path.stem
