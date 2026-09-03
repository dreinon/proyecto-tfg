"""Safe inference boundary for the professional score-enhancement demonstrator."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from score_super_resolution.adaptation_split import load_frozen_adaptation_split
from score_super_resolution.baselines import BaselineContractError, pixel_sha256, validate_rgb8
from score_super_resolution.edsr_finetuning import (
    ADAPTED_METHOD_ID,
    FineTuningError,
    _model_pixels,
    load_finetuned_edsr,
)

DEFAULT_MAX_OUTPUT_PIXELS = 32_000_000
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("artifacts/kaggle/smb-edsr-finetuning-v1/training")
CHECKPOINT_DIRECTORY_ENVIRONMENT_VARIABLE = "SCORE_SR_CHECKPOINT_DIR"


class DemonstratorError(ValueError):
    """The requested inference would violate the demonstrator contract."""


@dataclass(frozen=True)
class EnhancementResult:
    """One reconstructed image and the evidence needed to identify the derivative."""

    pixels: np.ndarray
    scale: int
    elapsed_seconds: float
    checkpoint_sha256: str
    input_sha256: str
    output_sha256: str
    device: str
    method_id: str = ADAPTED_METHOD_ID


ModelLoader = Callable[[int, torch.device], tuple[nn.Module, str]]


class ProfessionalInferenceService:
    """Load the validated EDSR adaptation lazily and reconstruct bounded RGB pages."""

    def __init__(
        self,
        project_root: Path,
        *,
        checkpoint_directory: Path | None = None,
        device: str | torch.device | None = None,
        tile_size: int = 256,
        tile_overlap: int = 32,
        maximum_output_pixels: int = DEFAULT_MAX_OUTPUT_PIXELS,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        configured_directory = checkpoint_directory
        if configured_directory is None:
            configured = os.environ.get(CHECKPOINT_DIRECTORY_ENVIRONMENT_VARIABLE)
            configured_directory = (
                Path(configured) if configured else DEFAULT_CHECKPOINT_RELATIVE_PATH
            )
        if not configured_directory.is_absolute():
            configured_directory = self.project_root / configured_directory
        self.checkpoint_directory = configured_directory.resolve()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if tile_size < 8 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
            raise DemonstratorError("the tiling configuration is invalid")
        if maximum_output_pixels < 1:
            raise DemonstratorError("the output pixel limit must be positive")
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.maximum_output_pixels = maximum_output_pixels
        self._model_loader = model_loader
        self._models: dict[int, tuple[nn.Module, str]] = {}

    def _load(self, scale: int) -> tuple[nn.Module, str]:
        cached = self._models.get(scale)
        if cached is not None:
            return cached
        if self._model_loader is not None:
            loaded = self._model_loader(scale, self.device)
        else:
            split = load_frozen_adaptation_split(self.project_root)
            checkpoint_path = self.checkpoint_directory / f"edsr-smb-finetuned-v1-x{scale}.pt"
            loaded = load_finetuned_edsr(
                checkpoint_path,
                split,
                scale=scale,
                device=self.device,
            )
        self._models[scale] = loaded
        return loaded

    def model_identity(self, scale: int) -> str:
        """Return the validated checkpoint digest used for one supported scale."""

        if isinstance(scale, bool) or scale not in {2, 4}:
            raise DemonstratorError("the demonstrator supports only x2 and x4")
        return self._load(scale)[1]

    def enhance(self, pixels: np.ndarray, *, scale: int) -> EnhancementResult:
        """Create a reversible derivative without retaining or modifying the input image."""

        if isinstance(scale, bool) or scale not in {2, 4}:
            raise DemonstratorError("the demonstrator supports only x2 and x4")
        if not isinstance(pixels, np.ndarray):
            raise DemonstratorError("the input must be an RGB NumPy image")
        owned_pixels = np.ascontiguousarray(pixels)
        try:
            validate_rgb8(owned_pixels)
        except BaselineContractError as error:
            raise DemonstratorError("the input must be a bounded RGB8 image") from error
        output_pixels = owned_pixels.shape[0] * owned_pixels.shape[1] * scale * scale
        if output_pixels > self.maximum_output_pixels:
            raise DemonstratorError(
                "the selected scale would exceed the demonstrator output-size limit"
            )

        try:
            model, checkpoint_digest = self._load(scale)
            reconstructed, elapsed_seconds = _model_pixels(
                model,
                owned_pixels,
                scale=scale,
                target_shape=(owned_pixels.shape[0] * scale, owned_pixels.shape[1] * scale, 3),
                device=self.device,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        except (FineTuningError, OSError, RuntimeError) as error:
            raise DemonstratorError("the validated EDSR adaptation could not run") from error

        return EnhancementResult(
            pixels=reconstructed,
            scale=scale,
            elapsed_seconds=elapsed_seconds,
            checkpoint_sha256=checkpoint_digest,
            input_sha256=pixel_sha256(owned_pixels),
            output_sha256=pixel_sha256(reconstructed),
            device=str(self.device),
        )
