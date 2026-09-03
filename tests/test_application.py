from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import torch
from torch import nn

from score_super_resolution.application import DemonstratorError, ProfessionalInferenceService
from score_super_resolution.web_app import APP_TITLE, build_demo, create_handler

ROOT = Path(__file__).resolve().parents[1]


class _NearestModel(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(inputs, scale_factor=self.scale, mode="nearest")


def _service(*, maximum_output_pixels: int = 100_000) -> ProfessionalInferenceService:
    return ProfessionalInferenceService(
        ROOT,
        device="cpu",
        tile_size=16,
        tile_overlap=2,
        maximum_output_pixels=maximum_output_pixels,
        model_loader=lambda scale, _device: (_NearestModel(scale), f"{scale}" * 64),
    )


def test_professional_inference_returns_identified_derivative() -> None:
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    service = _service()

    result = service.enhance(pixels, scale=2)

    assert result.pixels.shape == (16, 20, 3)
    assert result.pixels.dtype == np.uint8
    assert result.method_id == "edsr-smb-finetuned-v1"
    assert result.scale == 2
    assert result.elapsed_seconds > 0
    assert len(result.input_sha256) == 64
    assert len(result.output_sha256) == 64
    assert result.checkpoint_sha256 == "2" * 64
    assert np.array_equal(result.pixels[::2, ::2], pixels)


def test_professional_inference_caches_one_model_per_scale() -> None:
    loaded: list[int] = []

    def loader(scale: int, _device: torch.device) -> tuple[nn.Module, str]:
        loaded.append(scale)
        return _NearestModel(scale), f"{scale}" * 64

    service = ProfessionalInferenceService(
        ROOT,
        device="cpu",
        tile_size=16,
        tile_overlap=2,
        maximum_output_pixels=100_000,
        model_loader=loader,
    )
    pixels = np.zeros((8, 10, 3), dtype=np.uint8)

    service.enhance(pixels, scale=2)
    service.enhance(pixels, scale=2)
    service.enhance(pixels, scale=4)

    assert loaded == [2, 4]


def test_professional_inference_exposes_validated_model_identity() -> None:
    service = _service()

    assert service.model_identity(2) == "2" * 64
    with pytest.raises(DemonstratorError, match="only x2 and x4"):
        service.model_identity(3)


@pytest.mark.parametrize("scale", [1, 3, 8, True])
def test_professional_inference_rejects_unvalidated_scales(scale: int) -> None:
    with pytest.raises(DemonstratorError, match="only x2 and x4"):
        _service().enhance(np.zeros((8, 8, 3), dtype=np.uint8), scale=scale)


def test_professional_inference_rejects_unsafe_output_size() -> None:
    with pytest.raises(DemonstratorError, match="output-size limit"):
        _service(maximum_output_pixels=100).enhance(np.zeros((8, 8, 3), dtype=np.uint8), scale=2)


def test_web_handler_returns_comparison_download_and_evidence() -> None:
    pixels = np.zeros((8, 10, 3), dtype=np.uint8)

    comparison, output, evidence = create_handler(_service())(pixels, "x2 · ampliación moderada")

    assert comparison[0] is pixels
    assert comparison[1].shape == (16, 20, 3)
    assert output.shape == (16, 20, 3)
    assert "EDSR" not in evidence or "edsr-smb-finetuned-v1" in evidence
    assert "x2" in evidence
    assert "derivado" in evidence.casefold()


def test_web_demo_builds_without_loading_private_weights() -> None:
    demo = build_demo(ROOT, service=_service())

    assert APP_TITLE in demo.title
    assert demo.config["enable_queue"] is True


def test_web_handler_rejects_an_unknown_scale_label() -> None:
    with pytest.raises(gr.Error, match="escala x2 o x4"):
        create_handler(_service())(np.zeros((8, 10, 3), dtype=np.uint8), "x8")
