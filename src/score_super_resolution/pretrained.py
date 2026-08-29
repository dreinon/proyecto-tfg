"""Official pretrained SR baselines with verified checkpoints and tiled inference.

The EDSR modules are a small, API-local adaptation of the MIT-licensed official
``sanghyun-son/EDSR-PyTorch`` baseline architecture. SwinIR is loaded through
Spandrel from the official Apache-2.0 checkpoint. Both paths expose the same
RGB8, aligned-output contract used by the transparent interpolation baseline.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn

from score_super_resolution.baselines import (
    BaselineContractError,
    BaselineResult,
    pixel_sha256,
    run_baseline,
    validate_rgb8,
)

MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MODEL_METHODS = (
    "bicubic-opencv-v1",
    "edsr-baseline-official-v1",
    "swinir-lightweight-official-v1",
)


class PretrainedModelError(ValueError):
    """A model identity, checkpoint, input, or inference result is invalid."""


@dataclass(frozen=True)
class CheckpointSpec:
    """Immutable identity and acquisition metadata for one scale-specific model."""

    method_id: str
    scale: int
    filename: str
    relative_path: str
    url: str
    sha256: str
    architecture: str
    source_revision: str
    license_id: str


@dataclass(frozen=True)
class ModelResult:
    """Aligned RGB8 reconstruction and reproducibility evidence."""

    pixels: np.ndarray
    elapsed_ns: int
    evidence: dict[str, Any]


_EDSR_REVISION = "8dba5581a7502b92de9641eb431130d6c8ca5d7f"
_SWINIR_REVISION = "6545850fbf8df298df73d81f3e8cba638787c8bd"

CHECKPOINTS = MappingProxyType(
    {
        ("edsr-baseline-official-v1", 2): CheckpointSpec(
            method_id="edsr-baseline-official-v1",
            scale=2,
            filename="edsr_baseline_x2-1bc95232.pt",
            relative_path="edsr/edsr_baseline_x2-1bc95232.pt",
            url=("https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt"),
            sha256="1bc9523228fdc3b0a5ddb0d7062001e22b31bbb398ae09cfdefb97e8ee06e171",
            architecture="EDSR-baseline-r16f64",
            source_revision=_EDSR_REVISION,
            license_id="MIT",
        ),
        ("edsr-baseline-official-v1", 4): CheckpointSpec(
            method_id="edsr-baseline-official-v1",
            scale=4,
            filename="edsr_baseline_x4-6b446fab.pt",
            relative_path="edsr/edsr_baseline_x4-6b446fab.pt",
            url=("https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt"),
            sha256="6b446fab734f4de74448d2fd1f3f990f5bae726e49dc7eff2ae9cefe444a1723",
            architecture="EDSR-baseline-r16f64",
            source_revision=_EDSR_REVISION,
            license_id="MIT",
        ),
        ("swinir-lightweight-official-v1", 2): CheckpointSpec(
            method_id="swinir-lightweight-official-v1",
            scale=2,
            filename="002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
            relative_path="swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
            url=(
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
            ),
            sha256="193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7",
            architecture="SwinIR-lightweight-s64w8",
            source_revision=_SWINIR_REVISION,
            license_id="Apache-2.0",
        ),
        ("swinir-lightweight-official-v1", 4): CheckpointSpec(
            method_id="swinir-lightweight-official-v1",
            scale=4,
            filename="002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
            relative_path="swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
            url=(
                "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
                "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
            ),
            sha256="09fad24e32ae62722e1a055efde9921328f4137981bab0a42a4a3a806306c58e",
            architecture="SwinIR-lightweight-s64w8",
            source_revision=_SWINIR_REVISION,
            license_id="Apache-2.0",
        ),
    }
)


def checkpoint_sha256(path: Path) -> str:
    """Hash a bounded regular checkpoint without following a symlink."""

    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PretrainedModelError("checkpoint is unavailable") from error
    if path.is_symlink() or not path.is_file() or metadata.st_size > MAX_CHECKPOINT_BYTES:
        raise PretrainedModelError("checkpoint must be a bounded regular non-symlink file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PretrainedModelError("checkpoint cannot be read") from error
    return digest.hexdigest()


def ensure_checkpoint(spec: CheckpointSpec, checkpoint_root: Path) -> Path:
    """Return a verified local checkpoint, downloading its allowlisted URL if absent."""

    checkpoint_root = Path(checkpoint_root).resolve()
    path = checkpoint_root / spec.relative_path
    if not path.is_relative_to(checkpoint_root):
        raise PretrainedModelError("checkpoint path escapes its root")
    if path.exists():
        if checkpoint_sha256(path) != spec.sha256:
            raise PretrainedModelError(f"checkpoint digest mismatch for {spec.filename}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.download-{os.getpid()}"
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "score-sr-tfg/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as out:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_CHECKPOINT_BYTES:
                    raise PretrainedModelError("checkpoint download exceeds the size bound")
                out.write(chunk)
        if checkpoint_sha256(temporary) != spec.sha256:
            raise PretrainedModelError(f"downloaded checkpoint digest mismatch for {spec.filename}")
        os.replace(temporary, path)
    except PretrainedModelError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise PretrainedModelError(f"checkpoint download failed for {spec.filename}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def ensure_all_checkpoints(checkpoint_root: Path) -> dict[str, str]:
    """Acquire and verify the exact four checkpoints required by the experiment."""

    return {
        f"{spec.method_id}-x{spec.scale}": str(ensure_checkpoint(spec, checkpoint_root))
        for spec in CHECKPOINTS.values()
    }


def _default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class _MeanShift(nn.Conv2d):
    def __init__(self, rgb_range: float, *, sign: int) -> None:
        super().__init__(3, 3, kernel_size=1)
        rgb_mean = torch.tensor((0.4488, 0.4371, 0.4040))
        self.weight.data.copy_(torch.eye(3).view(3, 3, 1, 1))
        self.bias.data.copy_(sign * rgb_range * rgb_mean)
        self.requires_grad_(False)


class _ResidualBlock(nn.Module):
    def __init__(self, features: int, *, residual_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _default_conv(features, features, 3),
            nn.ReLU(True),
            _default_conv(features, features, 3),
        )
        self.res_scale = residual_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.body(inputs).mul(self.res_scale)


class _Upsampler(nn.Sequential):
    def __init__(self, scale: int, features: int) -> None:
        modules: list[nn.Module] = []
        for _ in range(int(math.log2(scale))):
            modules.extend((_default_conv(features, 4 * features, 3), nn.PixelShuffle(2)))
        super().__init__(*modules)


class EDSRBaseline(nn.Module):
    """Official 16-block, 64-feature EDSR baseline architecture for x2 or x4."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        if scale not in {2, 4}:
            raise PretrainedModelError("EDSR baseline supports only x2 and x4")
        features = 64
        self.sub_mean = _MeanShift(255.0, sign=-1)
        self.add_mean = _MeanShift(255.0, sign=1)
        self.head = nn.Sequential(_default_conv(3, features, 3))
        body: list[nn.Module] = [_ResidualBlock(features, residual_scale=1.0) for _ in range(16)]
        body.append(_default_conv(features, features, 3))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(_Upsampler(scale, features), _default_conv(features, 3, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.head(self.sub_mean(inputs))
        residual = self.body(features) + features
        return self.add_mean(self.tail(residual))


class _NormalizedEDSR(nn.Module):
    def __init__(self, model: EDSRBaseline) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs * 255.0) / 255.0


def _load_edsr(path: Path, scale: int) -> nn.Module:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PretrainedModelError("EDSR checkpoint cannot be loaded safely") from error
    if not isinstance(state, dict):
        raise PretrainedModelError("EDSR checkpoint is not a state dictionary")
    model = EDSRBaseline(scale)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise PretrainedModelError(
            "EDSR checkpoint does not match the frozen architecture"
        ) from error
    return _NormalizedEDSR(model)


def _load_swinir(path: Path, scale: int) -> nn.Module:
    try:
        from spandrel import ImageModelDescriptor, ModelLoader

        descriptor = ModelLoader().load_from_file(path)
    except Exception as error:
        raise PretrainedModelError("SwinIR checkpoint cannot be loaded safely") from error
    if not isinstance(descriptor, ImageModelDescriptor) or descriptor.scale != scale:
        raise PretrainedModelError("SwinIR checkpoint identity or scale differs")
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise PretrainedModelError("SwinIR checkpoint does not satisfy the RGB contract")
    return descriptor


def _pad_to_window(inputs: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    height, width = inputs.shape[-2:]
    pad_height = (-height) % window_size
    pad_width = (-width) % window_size
    if not pad_height and not pad_width:
        return inputs, 0, 0
    mode = "reflect" if height > pad_height and width > pad_width else "replicate"
    padded = torch.nn.functional.pad(inputs, (0, pad_width, 0, pad_height), mode=mode)
    return padded, pad_height, pad_width


def _tile_starts(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return tuple(starts)


def tiled_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    window_size: int = 8,
) -> torch.Tensor:
    """Run bounded overlapping inference and average overlap pixels deterministically."""

    if inputs.ndim != 4 or inputs.shape[0] != 1 or inputs.shape[1] != 3:
        raise PretrainedModelError("model input must be one NCHW RGB tensor")
    if tile_size < window_size or tile_size % window_size != 0:
        raise PretrainedModelError("tile size must be a positive multiple of the model window")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise PretrainedModelError("tile overlap must be smaller than the tile size")

    height, width = inputs.shape[-2:]
    output = torch.zeros(
        (1, 3, height * scale, width * scale), dtype=torch.float32, device=inputs.device
    )
    weights = torch.zeros_like(output)
    for top in _tile_starts(height, tile_size, tile_overlap):
        for left in _tile_starts(width, tile_size, tile_overlap):
            patch = inputs[..., top : top + tile_size, left : left + tile_size]
            patch_height, patch_width = patch.shape[-2:]
            padded, _, _ = _pad_to_window(patch, window_size)
            reconstructed = model(padded)
            reconstructed = reconstructed[..., : patch_height * scale, : patch_width * scale]
            if reconstructed.shape != (1, 3, patch_height * scale, patch_width * scale):
                raise PretrainedModelError("model returned an unexpected reconstruction shape")
            output_top = top * scale
            output_left = left * scale
            output[
                ...,
                output_top : output_top + patch_height * scale,
                output_left : output_left + patch_width * scale,
            ] += reconstructed.float()
            weights[
                ...,
                output_top : output_top + patch_height * scale,
                output_left : output_left + patch_width * scale,
            ] += 1.0
    if torch.any(weights == 0):
        raise PretrainedModelError("tiled inference left uncovered output pixels")
    return output / weights


def _dimensions(pixels: np.ndarray) -> dict[str, int]:
    return {"width": pixels.shape[1], "height": pixels.shape[0], "channels": pixels.shape[2]}


class PretrainedSRRunner:
    """Cache official models per scale and execute one common RGB8 inference boundary."""

    def __init__(
        self,
        project_root: Path,
        *,
        device: str | torch.device | None = None,
        tile_size: int = 256,
        tile_overlap: int = 32,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.checkpoint_root = self.project_root / "checkpoints"
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self._models: dict[tuple[str, int], nn.Module] = {}

    def prepare(self) -> dict[str, str]:
        """Ensure every frozen checkpoint is present before any benchmark outcome is opened."""

        return ensure_all_checkpoints(self.checkpoint_root)

    def _load(self, method_id: str, scale: int) -> tuple[nn.Module, CheckpointSpec]:
        key = (method_id, scale)
        spec = CHECKPOINTS.get(key)
        if spec is None:
            raise PretrainedModelError("unknown learned method or scale")
        cached = self._models.get(key)
        if cached is not None:
            return cached, spec
        path = ensure_checkpoint(spec, self.checkpoint_root)
        model = (
            _load_edsr(path, scale) if method_id.startswith("edsr-") else _load_swinir(path, scale)
        )
        model = model.to(self.device).eval()
        self._models[key] = model
        return model, spec

    def run(
        self,
        method_id: str,
        lr_rgb: np.ndarray,
        *,
        target_shape: tuple[int, int, int],
        condition_id: str,
    ) -> ModelResult:
        """Reconstruct one LR image through bicubic, EDSR-baseline, or SwinIR-lightweight."""

        validate_rgb8(lr_rgb)
        if condition_id not in {
            "x2-clean",
            "x2-moderate",
            "x2-strong",
            "x4-clean",
            "x4-moderate",
            "x4-strong",
        }:
            raise PretrainedModelError("condition is outside the frozen six-cell design")
        scale = int(condition_id[1])
        expected_shape = (lr_rgb.shape[0] * scale, lr_rgb.shape[1] * scale, 3)
        if target_shape != expected_shape:
            raise PretrainedModelError("target shape is not aligned to the condition scale")
        if method_id == "bicubic-opencv-v1":
            try:
                baseline: BaselineResult = run_baseline(
                    method_id,
                    lr_rgb,
                    target_shape=target_shape,
                    condition_id=condition_id,
                )
            except BaselineContractError as error:
                raise PretrainedModelError("bicubic baseline rejected the input") from error
            return ModelResult(baseline.pixels, baseline.elapsed_ns, baseline.evidence)

        model, spec = self._load(method_id, scale)
        tensor = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32).div_(255.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            reconstructed = tiled_forward(
                model,
                tensor,
                scale=scale,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ns = time.perf_counter_ns() - started
        pixels = (
            reconstructed.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        pixels = np.ascontiguousarray(pixels)
        if pixels.shape != target_shape:
            raise PretrainedModelError("model output is not aligned with the HR target")
        parameter_owner = getattr(model, "model", model)
        if not isinstance(parameter_owner, nn.Module):
            raise PretrainedModelError("model does not expose countable parameters")
        evidence = {
            "method_id": method_id,
            "role": "pretrained-fidelity-baseline",
            "condition_id": condition_id,
            "architecture": spec.architecture,
            "scale": scale,
            "checkpoint_filename": spec.filename,
            "checkpoint_sha256": spec.sha256,
            "source_revision": spec.source_revision,
            "license_id": spec.license_id,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "parameter_count": sum(parameter.numel() for parameter in parameter_owner.parameters()),
            "input_pixel_sha256": pixel_sha256(lr_rgb),
            "output_pixel_sha256": pixel_sha256(pixels),
            "input_dimensions": _dimensions(lr_rgb),
            "output_dimensions": _dimensions(pixels),
        }
        return ModelResult(pixels=pixels, elapsed_ns=elapsed_ns, evidence=evidence)
