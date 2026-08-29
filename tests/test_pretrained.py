from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from score_super_resolution.contracts import validate_instance
from score_super_resolution.pretrained import (
    CHECKPOINTS,
    MODEL_METHODS,
    CheckpointSpec,
    EDSRBaseline,
    PretrainedModelError,
    PretrainedSRRunner,
    ensure_checkpoint,
    tiled_forward,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_registry_covers_two_models_and_two_scales() -> None:
    assert MODEL_METHODS == (
        "bicubic-opencv-v1",
        "edsr-baseline-official-v1",
        "swinir-lightweight-official-v1",
    )
    assert set(CHECKPOINTS) == {
        ("edsr-baseline-official-v1", 2),
        ("edsr-baseline-official-v1", 4),
        ("swinir-lightweight-official-v1", 2),
        ("swinir-lightweight-official-v1", 4),
    }
    assert all(len(spec.sha256) == 64 for spec in CHECKPOINTS.values())
    assert all(spec.url.startswith("https://") for spec in CHECKPOINTS.values())


@pytest.mark.parametrize(
    ("scale", "parameter_count"),
    [(2, 1_369_883), (4, 1_517_595)],
)
def test_edsr_baseline_has_frozen_shape_and_parameter_count(
    scale: int, parameter_count: int
) -> None:
    model = EDSRBaseline(scale).eval()
    inputs = torch.zeros((1, 3, 8, 12), dtype=torch.float32)
    with torch.inference_mode():
        output = model(inputs)
    assert output.shape == (1, 3, 8 * scale, 12 * scale)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


class _NearestUpscale(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(inputs, scale_factor=self.scale, mode="nearest")


def test_tiled_forward_matches_full_forward_across_overlap() -> None:
    inputs = torch.arange(3 * 21 * 29, dtype=torch.float32).reshape(1, 3, 21, 29)
    model = _NearestUpscale(2)
    expected = model(inputs)
    actual = tiled_forward(
        model,
        inputs,
        scale=2,
        tile_size=16,
        tile_overlap=4,
    )
    torch.testing.assert_close(actual, expected)


def test_checkpoint_verification_fails_closed_without_network(tmp_path: Path) -> None:
    payload = b"official-like-checkpoint-fixture"
    path = tmp_path / "family" / "fixture.pt"
    path.parent.mkdir()
    path.write_bytes(payload)
    spec = CheckpointSpec(
        method_id="fixture-v1",
        scale=2,
        filename=path.name,
        relative_path="family/fixture.pt",
        url="https://example.invalid/fixture.pt",
        sha256=hashlib.sha256(payload).hexdigest(),
        architecture="fixture",
        source_revision="fixture-revision",
        license_id="MIT",
    )
    assert ensure_checkpoint(spec, tmp_path) == path
    path.write_bytes(b"mutated")
    with pytest.raises(PretrainedModelError, match="digest mismatch"):
        ensure_checkpoint(spec, tmp_path)


def test_bicubic_runner_uses_common_aligned_rgb8_contract() -> None:
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    result = PretrainedSRRunner(ROOT, device="cpu").run(
        "bicubic-opencv-v1",
        pixels,
        target_shape=(24, 32, 3),
        condition_id="x2-clean",
    )
    assert result.pixels.shape == (24, 32, 3)
    assert result.pixels.dtype == np.uint8
    assert result.evidence["method_id"] == "bicubic-opencv-v1"


@pytest.mark.parametrize(
    "descriptor",
    sorted((ROOT / "configs/models").glob("*.yaml")),
)
def test_model_descriptors_follow_the_provenance_schema(descriptor: Path) -> None:
    payload = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    validate_instance("model-descriptor", payload, version=1)
    scale = int(payload["output_conventions"][0].removeprefix("scale: "))
    registry_spec = CHECKPOINTS[(payload["source"]["version"], scale)]
    assert payload["checkpoint"]["sha256"] == registry_spec.sha256
