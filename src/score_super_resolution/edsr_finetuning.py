"""Bounded, source-disjoint EDSR fine-tuning and evaluation on derived SMB partitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from score_super_resolution.adaptation_split import (
    AdaptationSplitError,
    AdaptationSplitRow,
    FrozenAdaptationSplit,
)
from score_super_resolution.baselines import pixel_sha256
from score_super_resolution.comparison import as_rgb8, ensure_manifest_generation, fidelity_metrics
from score_super_resolution.degradation import align_reference
from score_super_resolution.pretrained import (
    CHECKPOINTS,
    EDSRBaseline,
    PretrainedSRRunner,
    checkpoint_sha256,
    ensure_checkpoint,
    tiled_forward,
)
from score_super_resolution.smb_audit import resolve_active_manifest
from score_super_resolution.staff_scale import (
    apply_scale_normalized_degradation,
    canonical_smb_pixel_sha256,
    load_scale_normalized_control,
)

ADAPTED_METHOD_ID = "edsr-smb-finetuned-v1"
PRETRAINED_METHOD_ID = "edsr-baseline-official-v1"
BICUBIC_METHOD_ID = "bicubic-opencv-v1"
RESULT_FIELDS = (
    "upstream_index",
    "item_id",
    "source_group_id",
    "condition_id",
    "scale",
    "profile",
    "method_id",
    "psnr_y",
    "ssim_y",
    "psnr_rgb",
    "ssim_rgb",
    "runtime_seconds",
    "output_sha256",
    "checkpoint_sha256",
)


class FineTuningError(ValueError):
    """The adaptation run violates its frozen data, optimization, or evidence contract."""


@dataclass(frozen=True)
class FineTuningRun:
    """Selected checkpoint and bounded optimization evidence for one scale."""

    scale: int
    checkpoint_path: Path
    checkpoint_sha256: str
    best_step: int
    best_validation_l1: float
    completed_steps: int
    stopped_early: bool
    history: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class AdaptationDataPreflight:
    """Evidence that exact immutable SMB bytes were checked for named partitions."""

    split_sha256: str
    partitions: tuple[str, ...]
    pages: int
    groups: int
    group_counts: tuple[tuple[str, int], ...]

    def authorizes(self, split: FrozenAdaptationSplit, required: Sequence[str]) -> bool:
        return self.split_sha256 == split.split_sha256 and set(required) <= set(self.partitions)


class NormalizedEDSR(nn.Module):
    """Expose the official 0--255 EDSR baseline through a normalized 0--1 boundary."""

    def __init__(self, model: EDSRBaseline) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs * 255.0) / 255.0


class _ImageCache:
    def __init__(self, dataset: Sequence[Mapping[str, object]], maximum_items: int = 12) -> None:
        self.dataset = dataset
        self.maximum_items = maximum_items
        self._values: OrderedDict[int, tuple[np.ndarray, object]] = OrderedDict()

    def get(self, index: int) -> tuple[np.ndarray, object]:
        cached = self._values.pop(index, None)
        if cached is not None:
            self._values[index] = cached
            return cached
        source = self.dataset[index]
        value = (as_rgb8(source["image"]), source["regions"])
        self._values[index] = value
        while len(self._values) > self.maximum_items:
            self._values.popitem(last=False)
        return value


def load_adaptation_dataset(split: FrozenAdaptationSplit) -> Sequence[Mapping[str, object]]:
    """Load the pinned upstream bytes only after the derived split has validated completely."""

    if not isinstance(split, FrozenAdaptationSplit):
        raise FineTuningError("a validated adaptation split is required before data access")
    dataset_config = split.config.get("dataset")
    if not isinstance(dataset_config, Mapping):
        raise FineTuningError("adaptation dataset config is missing")
    if (
        dataset_config.get("repository_id") != "PRAIG/SMB"
        or dataset_config.get("revision") != split.source_revision
        or dataset_config.get("split") != "test"
    ):
        raise FineTuningError("adaptation dataset identity differs from the frozen split")
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_config["repository_id"],
            split=dataset_config["split"],
            revision=dataset_config["revision"],
        )
    except Exception as error:
        raise FineTuningError("pinned SMB adaptation source cannot be loaded") from error
    if len(dataset) != 685:
        raise FineTuningError("pinned SMB adaptation source has an unexpected length")
    return dataset


def preflight_adaptation_dataset(
    project_root: Path,
    dataset: Sequence[Mapping[str, object]],
    split: FrozenAdaptationSplit,
    *,
    partitions: Sequence[str] = ("train", "validation", "test"),
) -> AdaptationDataPreflight:
    """Verify every accessed SMB page against the active audited pixel manifest."""

    project_root = Path(project_root).resolve()
    if len(dataset) != 685:
        raise FineTuningError("runtime SMB length differs from the frozen source")
    generation_root = ensure_manifest_generation(project_root)
    _, manifest_rows = resolve_active_manifest(
        active_path=project_root / "data/manifests/smb-evaluation-v1.yaml",
        generation_root=generation_root,
    )
    manifest_by_index = {int(row["upstream_index"]): row for row in manifest_rows}
    selected_rows = tuple(row for partition in partitions for row in split.rows_for(partition))
    if len({row.upstream_index for row in selected_rows}) != len(selected_rows):
        raise FineTuningError("adaptation preflight received repeated pages")
    group_counts: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        manifest_row = manifest_by_index.get(row.upstream_index)
        if manifest_row is None or any(
            manifest_row.get(field) != expected
            for field, expected in (
                ("item_id", row.item_id),
                ("source_group_id", row.source_group_id),
            )
        ):
            raise FineTuningError("adaptation row identity differs from the audited manifest")
        source = dataset[row.upstream_index]
        if canonical_smb_pixel_sha256(source["image"]) != manifest_row["image"]["pixel_sha256"]:
            raise FineTuningError("runtime SMB pixels differ from the audited manifest")
        group_counts[row.partition].add(row.source_group_id)
    return AdaptationDataPreflight(
        split_sha256=split.split_sha256,
        partitions=tuple(partitions),
        pages=len(selected_rows),
        groups=len({row.source_group_id for row in selected_rows}),
        group_counts=tuple((partition, len(group_counts[partition])) for partition in partitions),
    )


def _load_pretrained_edsr(
    project_root: Path, scale: int, device: torch.device
) -> tuple[nn.Module, str]:
    spec = CHECKPOINTS.get((PRETRAINED_METHOD_ID, scale))
    if spec is None:
        raise FineTuningError("EDSR adaptation supports only the frozen x2 and x4 checkpoints")
    path = ensure_checkpoint(spec, Path(project_root).resolve() / "checkpoints")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError
        base = EDSRBaseline(scale)
        base.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise FineTuningError("official EDSR checkpoint cannot initialize adaptation") from error
    return NormalizedEDSR(base).to(device), spec.sha256


def _seed(base_seed: int, *parts: object) -> int:
    payload = "\0".join(("edsr-smb-finetuning-v1", str(base_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _valid_region_boxes(
    regions: object, width: int, height: int
) -> list[tuple[int, int, int, int]]:
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes, bytearray)):
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        if not isinstance(region, Mapping) or not isinstance(region.get("bbox"), Mapping):
            continue
        bbox = region["bbox"]
        try:
            x, y, box_width, box_height = (
                float(bbox[field]) for field in ("x", "y", "width", "height")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all(math.isfinite(value) for value in (x, y, box_width, box_height))
            or x < 0
            or y < 0
            or box_width <= 0
            or box_height <= 0
            or x + box_width > 100.5
            or y + box_height > 100.5
        ):
            continue
        left = max(0, round(x / 100.0 * width))
        top = max(0, round(y / 100.0 * height))
        right = min(width, round((x + box_width) / 100.0 * width))
        bottom = min(height, round((y + box_height) / 100.0 * height))
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def sample_notation_patch(
    pixels: np.ndarray,
    regions: object,
    *,
    patch_size: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic notation-centred HR patch without consulting any SR outcome."""

    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise FineTuningError("patch source must be RGB uint8")
    height, width = pixels.shape[:2]
    if patch_size < 32 or patch_size > min(height, width):
        raise FineTuningError("HR patch size is outside the source page")
    boxes = _valid_region_boxes(regions, width, height)
    if not boxes:
        raise FineTuningError("training page has no valid notation region")
    rng = np.random.Generator(np.random.PCG64(seed))
    best_patch: np.ndarray | None = None
    best_ink = -1.0
    for _ in range(8):
        left, top, right, bottom = boxes[int(rng.integers(0, len(boxes)))]
        center_x = int(rng.integers(left, max(left + 1, right)))
        center_y = int(rng.integers(top, max(top + 1, bottom)))
        crop_left = min(max(0, center_x - patch_size // 2), width - patch_size)
        crop_top = min(max(0, center_y - patch_size // 2), height - patch_size)
        patch = np.ascontiguousarray(
            pixels[crop_top : crop_top + patch_size, crop_left : crop_left + patch_size]
        )
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        ink = float(np.mean(gray < 235))
        if ink > best_ink:
            best_patch = patch
            best_ink = ink
        if ink >= 0.015:
            break
    if best_patch is None:
        raise FineTuningError("notation patch selection failed")
    return best_patch


class PatchBatchFactory:
    """Produce deterministic source-balanced LR/HR patch batches for one frozen split."""

    def __init__(
        self,
        project_root: Path,
        dataset: Sequence[Mapping[str, object]],
        rows: Sequence[AdaptationSplitRow],
        *,
        seed: int,
        lr_patch_size: int,
    ) -> None:
        if not rows or any(row.partition == "test" for row in rows):
            raise FineTuningError("patch batches may use train or validation rows only")
        self.cache = _ImageCache(dataset)
        self.rows = tuple(rows)
        self.seed = seed
        self.lr_patch_size = lr_patch_size
        self.control = load_scale_normalized_control(Path(project_root).resolve())
        by_group: dict[str, list[AdaptationSplitRow]] = defaultdict(list)
        for row in self.rows:
            by_group[row.source_group_id].append(row)
        self.by_group = {group: tuple(values) for group, values in by_group.items()}
        self.groups = tuple(sorted(self.by_group))

    def _example_from_row(
        self,
        row: AdaptationSplitRow,
        *,
        scale: int,
        token: object,
        profile_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels, regions = self.cache.get(row.upstream_index)
        hr_patch_size = self.lr_patch_size * scale
        reference = sample_notation_patch(
            pixels,
            regions,
            patch_size=hr_patch_size,
            seed=_seed(self.seed, scale, token, "patch"),
        )
        profile = ("clean", "moderate", "strong")[profile_index % 3]
        condition_id = f"x{scale}-{profile}"
        degraded = apply_scale_normalized_degradation(
            reference,
            control=self.control,
            condition_id=condition_id,
            item_id=f"{row.item_id}-adapt-{token}",
            source_group_id=row.source_group_id,
            staff_spacing_px=row.staff_spacing_px,
        )
        aligned = align_reference(reference, scale).pixels
        return degraded.pixels, aligned

    def _example(
        self,
        *,
        scale: int,
        token: object,
        profile_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.Generator(np.random.PCG64(_seed(self.seed, scale, token, "row")))
        group = self.groups[int(rng.integers(0, len(self.groups)))]
        group_rows = self.by_group[group]
        row = group_rows[int(rng.integers(0, len(group_rows)))]
        return self._example_from_row(
            row,
            scale=scale,
            token=token,
            profile_index=profile_index,
        )

    def training_batch(
        self,
        *,
        scale: int,
        step: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        examples = [
            self._example(
                scale=scale,
                token=f"train-{step}-{slot}",
                profile_index=(step * batch_size + slot) % 3,
            )
            for slot in range(batch_size)
        ]
        return _stack_examples(examples, device)

    def validation_examples(self, *, scale: int) -> list[tuple[np.ndarray, np.ndarray]]:
        representatives = sorted(
            (row for row in self.rows if row.representative_page),
            key=lambda row: row.source_group_id,
        )
        examples: list[tuple[np.ndarray, np.ndarray]] = []
        for row in representatives:
            for profile_index in range(3):
                examples.append(
                    self._example_from_row(
                        row,
                        scale=scale,
                        token=f"validation-{row.item_id}-{profile_index}",
                        profile_index=profile_index,
                    )
                )
        return examples


def _stack_examples(
    examples: Sequence[tuple[np.ndarray, np.ndarray]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lr = torch.from_numpy(np.stack([value[0] for value in examples])).permute(0, 3, 1, 2)
    hr = torch.from_numpy(np.stack([value[1] for value in examples])).permute(0, 3, 1, 2)
    return (
        lr.to(device=device, dtype=torch.float32).div_(255.0),
        hr.to(device=device, dtype=torch.float32).div_(255.0),
    )


def _mean_validation_l1(
    model: nn.Module,
    examples: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            lr, hr = _stack_examples(examples[start : start + batch_size], device)
            prediction = model(lr).clamp(0.0, 1.0)
            losses = torch.mean(torch.abs(prediction - hr), dim=(1, 2, 3))
            total += float(losses.sum().item())
            count += int(losses.numel())
    model.train()
    if count == 0:
        raise FineTuningError("validation split produced no fixed examples")
    return total / count


def _training_config(split: FrozenAdaptationSplit) -> Mapping[str, object]:
    training = split.config.get("training")
    if not isinstance(training, Mapping):
        raise FineTuningError("training controls are absent")
    return training


def fine_tune_edsr_scale(
    project_root: Path,
    dataset: Sequence[Mapping[str, object]],
    split: FrozenAdaptationSplit,
    *,
    scale: int,
    output_root: Path,
    data_preflight: AdaptationDataPreflight,
    device: str | torch.device | None = None,
    steps_override: int | None = None,
    batch_size_override: int | None = None,
) -> FineTuningRun:
    """Fine-tune one official EDSR scale using train/validation only and save the best state."""

    if scale not in {2, 4}:
        raise FineTuningError("fine-tuning scale must be x2 or x4")
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    training = _training_config(split)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        steps = steps_override or int(training["steps_per_scale"])
        batch_size = batch_size_override or int(training["batch_size"])
        lr_patch_size = int(training["lr_patch_size"])
        validation_every = int(training["validation_every_steps"])
        patience = int(training["validation_patience"])
        min_delta = float(training["validation_min_delta"])
        learning_rate = float(training["learning_rate"])
        minimum_learning_rate = float(training["minimum_learning_rate"])
        gradient_clip = float(training["gradient_clip_norm"])
        training_seed = int(training["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise FineTuningError("training controls are malformed") from error
    if steps < 1 or batch_size < 1 or lr_patch_size < 16 or validation_every < 1:
        raise FineTuningError("training controls must be positive")
    if any(row.partition != "train" for row in split.rows_for("train")):
        raise AdaptationSplitError("training rows contain another partition")
    if any(row.partition != "validation" for row in split.rows_for("validation")):
        raise AdaptationSplitError("validation rows contain another partition")
    if not data_preflight.authorizes(split, ("train", "validation")):
        raise FineTuningError("training requires a train/validation pixel-identity preflight")

    torch.manual_seed(_seed(training_seed, scale, "torch"))
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(_seed(training_seed, scale, "cuda"))
    model, pretrained_sha256 = _load_pretrained_edsr(project_root, scale, selected_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=minimum_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=selected_device.type == "cuda")
    train_factory = PatchBatchFactory(
        project_root,
        dataset,
        split.rows_for("train"),
        seed=training_seed,
        lr_patch_size=lr_patch_size,
    )
    validation_factory = PatchBatchFactory(
        project_root,
        dataset,
        split.rows_for("validation"),
        seed=training_seed,
        lr_patch_size=lr_patch_size,
    )
    validation_examples = validation_factory.validation_examples(scale=scale)

    history: list[dict[str, float | int]] = []
    best_step = 0
    best_validation = _mean_validation_l1(
        model, validation_examples, device=selected_device, batch_size=batch_size
    )
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history.append(
        {
            "step": 0,
            "training_l1": math.nan,
            "validation_l1": best_validation,
            "learning_rate": learning_rate,
        }
    )
    stale_evaluations = 0
    stopped_early = False
    completed_steps = 0
    for step in range(1, steps + 1):
        lr, hr = train_factory.training_batch(
            scale=scale,
            step=step,
            batch_size=batch_size,
            device=selected_device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=selected_device.type,
            dtype=torch.float16,
            enabled=selected_device.type == "cuda",
        ):
            prediction = model(lr)
            loss = torch.nn.functional.l1_loss(prediction, hr)
        if not torch.isfinite(loss):
            raise FineTuningError("training loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        completed_steps = step

        if step % validation_every == 0 or step == steps:
            validation = _mean_validation_l1(
                model, validation_examples, device=selected_device, batch_size=batch_size
            )
            history.append(
                {
                    "step": step,
                    "training_l1": float(loss.detach().item()),
                    "validation_l1": validation,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if validation < best_validation - min_delta:
                best_validation = validation
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if stale_evaluations >= patience:
                stopped_early = True
                break

    model.load_state_dict(best_state, strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / f"edsr-smb-finetuned-v1-x{scale}.pt"
    checkpoint_payload = {
        "schema_version": 1,
        "method_id": ADAPTED_METHOD_ID,
        "scale": scale,
        "split_sha256": split.split_sha256,
        "source_revision": split.source_revision,
        "pretrained_checkpoint_sha256": pretrained_sha256,
        "best_step": best_step,
        "best_validation_l1": best_validation,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "state_dict": best_state,
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, checkpoint_path)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_root / f"training-history-x{scale}.csv", index=False)
    return FineTuningRun(
        scale=scale,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        best_step=best_step,
        best_validation_l1=best_validation,
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        history=tuple(history),
    )


def load_finetuned_edsr(
    checkpoint_path: Path,
    split: FrozenAdaptationSplit,
    *,
    scale: int,
    device: str | torch.device,
) -> tuple[nn.Module, str]:
    """Load a selected checkpoint only when its scale, source, and split identities agree."""

    checkpoint_path = Path(checkpoint_path)
    digest = checkpoint_sha256(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise FineTuningError("fine-tuned EDSR checkpoint cannot be loaded safely") from error
    if not isinstance(payload, Mapping) or any(
        payload.get(field) != expected
        for field, expected in (
            ("schema_version", 1),
            ("method_id", ADAPTED_METHOD_ID),
            ("scale", scale),
            ("split_sha256", split.split_sha256),
            ("source_revision", split.source_revision),
        )
    ):
        raise FineTuningError("fine-tuned EDSR checkpoint identity differs")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise FineTuningError("fine-tuned EDSR checkpoint state is absent")
    model = NormalizedEDSR(EDSRBaseline(scale))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise FineTuningError("fine-tuned EDSR state does not match the architecture") from error
    return model.to(torch.device(device)).eval(), digest


def load_completed_finetuning_run(
    checkpoint_path: Path,
    history_path: Path,
    split: FrozenAdaptationSplit,
    *,
    scale: int,
) -> FineTuningRun:
    """Recover a completed scale run without opening the test partition."""

    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)
    digest = checkpoint_sha256(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        history_frame = pd.read_csv(history_path)
    except Exception as error:
        raise FineTuningError("completed fine-tuning evidence cannot be recovered") from error
    expected_identity = {
        "schema_version": 1,
        "method_id": ADAPTED_METHOD_ID,
        "scale": scale,
        "split_sha256": split.split_sha256,
        "source_revision": split.source_revision,
    }
    if not isinstance(payload, Mapping) or any(
        payload.get(field) != value for field, value in expected_identity.items()
    ):
        raise FineTuningError("completed fine-tuning evidence belongs to another run")
    expected_columns = {"step", "training_l1", "validation_l1", "learning_rate"}
    if set(history_frame.columns) != expected_columns or history_frame.empty:
        raise FineTuningError("completed fine-tuning history is malformed")
    history = tuple(
        {
            "step": int(row.step),
            "training_l1": float(row.training_l1),
            "validation_l1": float(row.validation_l1),
            "learning_rate": float(row.learning_rate),
        }
        for row in history_frame.itertuples(index=False)
    )
    return FineTuningRun(
        scale=scale,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_sha256=digest,
        best_step=int(payload["best_step"]),
        best_validation_l1=float(payload["best_validation_l1"]),
        completed_steps=int(payload["completed_steps"]),
        stopped_early=bool(payload["stopped_early"]),
        history=history,
    )


def _model_pixels(
    model: nn.Module,
    lr_rgb: np.ndarray,
    *,
    scale: int,
    target_shape: tuple[int, int, int],
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(lr_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = tiled_forward(
            model,
            tensor,
            scale=scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    pixels = (
        output.squeeze(0)
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
        raise FineTuningError("fine-tuned model output is not aligned to its reference")
    return pixels, elapsed


def evaluate_adaptation(
    project_root: Path,
    dataset: Sequence[Mapping[str, object]],
    split: FrozenAdaptationSplit,
    *,
    checkpoint_paths: Mapping[int, Path],
    output_root: Path,
    data_preflight: AdaptationDataPreflight,
    device: str | torch.device | None = None,
    tile_size: int = 256,
    tile_overlap: int = 32,
) -> pd.DataFrame:
    """Evaluate bicubic, pretrained EDSR, and adapted EDSR on the fresh test representatives."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_rows = split.rows_for("test", representative_only=True)
    if not test_rows or any(row.prior_role != "fresh-holdout" for row in test_rows):
        raise FineTuningError("adaptation test must contain only fresh representative sources")
    fine_models: dict[int, nn.Module] = {}
    fine_hashes: dict[int, str] = {}
    for scale in (2, 4):
        model, digest = load_finetuned_edsr(
            checkpoint_paths[scale], split, scale=scale, device=selected_device
        )
        fine_models[scale] = model
        fine_hashes[scale] = digest
    if not data_preflight.authorizes(split, ("test",)):
        raise FineTuningError("evaluation requires a test pixel-identity preflight")
    pretrained = PretrainedSRRunner(
        project_root,
        device=selected_device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    control = load_scale_normalized_control(project_root)
    conditions = tuple(split.config["evaluation"]["conditions"])
    results_path = output_root / "raw-metrics.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    identity_path = output_root / "evaluation-identity.json"
    identity = {
        "schema_version": 1,
        "experiment_id": split.config.get("experiment_id"),
        "split_sha256": split.split_sha256,
        "source_revision": split.source_revision,
        "conditions": list(conditions),
        "checkpoint_sha256": {str(scale): fine_hashes[scale] for scale in (2, 4)},
    }
    if identity_path.is_file():
        try:
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FineTuningError("partial evaluation identity cannot be read") from error
        if existing_identity != identity:
            raise FineTuningError("partial evaluation belongs to another frozen run")
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if results_path.is_file():
        results = pd.read_csv(results_path)
        if tuple(results.columns) != RESULT_FIELDS:
            raise FineTuningError("partial adaptation metrics have an unexpected schema")
        records = results.to_dict("records")
    else:
        records = []
    completed = {
        (str(row["item_id"]), str(row["condition_id"]), str(row["method_id"])) for row in records
    }

    for row in test_rows:
        source = dataset[row.upstream_index]
        hr_original = as_rgb8(source["image"])
        for condition_id in conditions:
            scale = int(condition_id[1])
            degraded = apply_scale_normalized_degradation(
                hr_original,
                control=control,
                condition_id=condition_id,
                item_id=row.item_id,
                source_group_id=row.source_group_id,
                staff_spacing_px=row.staff_spacing_px,
            )
            reference = align_reference(hr_original, scale).pixels
            for method_id in (BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID):
                key = (row.item_id, condition_id, method_id)
                if key in completed:
                    continue
                if method_id == ADAPTED_METHOD_ID:
                    pixels, runtime = _model_pixels(
                        fine_models[scale],
                        degraded.pixels,
                        scale=scale,
                        target_shape=reference.shape,
                        device=selected_device,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                    )
                    checkpoint_digest: str | float = fine_hashes[scale]
                else:
                    result = pretrained.run(
                        method_id,
                        degraded.pixels,
                        target_shape=reference.shape,
                        condition_id=condition_id,
                    )
                    pixels = result.pixels
                    runtime = result.elapsed_ns / 1e9
                    checkpoint_digest = result.evidence.get("checkpoint_sha256", math.nan)
                records.append(
                    {
                        "upstream_index": row.upstream_index,
                        "item_id": row.item_id,
                        "source_group_id": row.source_group_id,
                        "condition_id": condition_id,
                        "scale": scale,
                        "profile": condition_id.split("-", 1)[1],
                        "method_id": method_id,
                        **fidelity_metrics(reference, pixels),
                        "runtime_seconds": runtime,
                        "output_sha256": pixel_sha256(pixels),
                        "checkpoint_sha256": checkpoint_digest,
                    }
                )
                completed.add(key)
            frame = pd.DataFrame(records, columns=RESULT_FIELDS)
            temporary = results_path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, results_path)
    result = pd.DataFrame(records, columns=RESULT_FIELDS)
    expected_rows = len(test_rows) * len(conditions) * 3
    if (
        len(result) != expected_rows
        or len(result.drop_duplicates(["item_id", "condition_id", "method_id"])) != expected_rows
    ):
        raise FineTuningError("adaptation evaluation did not reconcile every expected tuple")
    return result.sort_values(["item_id", "condition_id", "method_id"]).reset_index(drop=True)


def export_qualitative_cases(
    project_root: Path,
    dataset: Sequence[Mapping[str, object]],
    split: FrozenAdaptationSplit,
    results: pd.DataFrame,
    *,
    checkpoint_paths: Mapping[int, Path],
    output_root: Path,
    data_preflight: AdaptationDataPreflight,
    device: str | torch.device | None = None,
    tile_size: int = 256,
    tile_overlap: int = 32,
) -> pd.DataFrame:
    """Persist the six outcome-independent HR/LR/method comparisons and verify their hashes."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if not data_preflight.authorizes(split, ("test",)):
        raise FineTuningError("qualitative export requires a test pixel-identity preflight")
    evaluation = split.config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise FineTuningError("qualitative evaluation config is absent")
    assignments = evaluation.get("qualitative_assignment")
    conditions = tuple(evaluation.get("conditions", ()))
    if not isinstance(assignments, Sequence) or len(assignments) != len(conditions):
        raise FineTuningError("qualitative assignment is incomplete")
    test_rows = {row.item_id: row for row in split.rows_for("test", representative_only=True)}
    assigned_conditions: set[str] = set()
    assigned_groups: set[str] = set()
    parsed: list[tuple[AdaptationSplitRow, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise FineTuningError("qualitative assignment row is malformed")
        item_id = str(assignment.get("item_id"))
        condition_id = str(assignment.get("condition_id"))
        row = test_rows.get(item_id)
        if row is None or any(
            assignment.get(field) != expected
            for field, expected in (
                ("source_group_id", row.source_group_id),
                ("upstream_index", row.upstream_index),
            )
        ):
            raise FineTuningError("qualitative assignment differs from the fresh test split")
        if condition_id not in conditions:
            raise FineTuningError("qualitative assignment has an unknown condition")
        assigned_conditions.add(condition_id)
        assigned_groups.add(row.source_group_id)
        parsed.append((row, condition_id))
    if len(assigned_conditions) != len(conditions) or len(assigned_groups) != len(parsed):
        raise FineTuningError("qualitative cases must cover conditions with distinct sources")

    fine_models: dict[int, nn.Module] = {}
    for scale in (2, 4):
        fine_models[scale], _ = load_finetuned_edsr(
            checkpoint_paths[scale], split, scale=scale, device=selected_device
        )
    pretrained = PretrainedSRRunner(
        project_root,
        device=selected_device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    control = load_scale_normalized_control(project_root)
    expected_hashes = {
        (str(row.item_id), str(row.condition_id), str(row.method_id)): str(row.output_sha256)
        for row in results.itertuples(index=False)
    }
    index_records: list[dict[str, object]] = []
    for row, condition_id in parsed:
        scale = int(condition_id[1])
        source = dataset[row.upstream_index]
        original = as_rgb8(source["image"])
        degradation = apply_scale_normalized_degradation(
            original,
            control=control,
            condition_id=condition_id,
            item_id=row.item_id,
            source_group_id=row.source_group_id,
            staff_spacing_px=row.staff_spacing_px,
        )
        reference = align_reference(original, scale).pixels
        nearest = cv2.resize(
            degradation.pixels,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        case_root = output_root / "qualitative" / row.item_id / condition_id
        case_root.mkdir(parents=True, exist_ok=True)
        images: dict[str, np.ndarray] = {
            "reference-hr": reference,
            "input-lr-nearest": nearest,
        }
        for method_id in (BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID):
            output = pretrained.run(
                method_id,
                degradation.pixels,
                target_shape=reference.shape,
                condition_id=condition_id,
            ).pixels
            if pixel_sha256(output) != expected_hashes.get((row.item_id, condition_id, method_id)):
                raise FineTuningError("qualitative pretrained output differs from raw metrics")
            images[method_id] = output
        adapted, _ = _model_pixels(
            fine_models[scale],
            degradation.pixels,
            scale=scale,
            target_shape=reference.shape,
            device=selected_device,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if pixel_sha256(adapted) != expected_hashes.get(
            (row.item_id, condition_id, ADAPTED_METHOD_ID)
        ):
            raise FineTuningError("qualitative adapted output differs from raw metrics")
        images[ADAPTED_METHOD_ID] = adapted
        for label, pixels in images.items():
            path = case_root / f"{label}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
                raise FineTuningError("qualitative PNG could not be written")
            index_records.append(
                {
                    "item_id": row.item_id,
                    "source_group_id": row.source_group_id,
                    "condition_id": condition_id,
                    "image_role": label,
                    "path": str(path.relative_to(output_root)),
                    "pixel_sha256": pixel_sha256(pixels),
                }
            )
    index = pd.DataFrame(index_records)
    index.to_csv(output_root / "qualitative-index.csv", index=False)
    return index


def analyze_adaptation_results(
    results: pd.DataFrame,
    *,
    seed: int,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exact test tuples and bootstrap paired source-level adaptation gains."""

    required = set(RESULT_FIELDS)
    if set(results.columns) != required or repetitions < 100:
        raise FineTuningError("adaptation results or bootstrap controls are invalid")
    methods = {BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID}
    if set(results["method_id"]) != methods:
        raise FineTuningError("adaptation results do not contain the exact three methods")
    numeric = results[["psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb", "runtime_seconds"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or (results["runtime_seconds"] < 0).any():
        raise FineTuningError("adaptation results contain invalid numeric evidence")
    if not results["ssim_y"].between(0, 1).all() or not results["ssim_rgb"].between(0, 1).all():
        raise FineTuningError("adaptation SSIM evidence is outside [0, 1]")
    if results.duplicated(["source_group_id", "condition_id", "method_id"]).any():
        raise FineTuningError("adaptation results repeat an inference-unit tuple")
    metrics = ("psnr_y", "ssim_y")
    aggregate = (
        results.groupby(["condition_id", "method_id"], as_index=False)
        .agg(
            sources=("source_group_id", "nunique"),
            psnr_y_mean=("psnr_y", "mean"),
            ssim_y_mean=("ssim_y", "mean"),
            psnr_rgb_mean=("psnr_rgb", "mean"),
            ssim_rgb_mean=("ssim_rgb", "mean"),
            runtime_seconds_median=("runtime_seconds", "median"),
        )
        .sort_values(["condition_id", "method_id"])
        .reset_index(drop=True)
    )
    paired_rows: list[dict[str, object]] = []
    for condition_id in sorted(results["condition_id"].unique()):
        condition = results[results["condition_id"] == condition_id]
        sources = sorted(condition["source_group_id"].unique())
        rng = np.random.Generator(np.random.PCG64(_seed(seed, condition_id, "bootstrap")))
        draws = rng.integers(0, len(sources), size=(repetitions, len(sources)))
        for reference_method in (PRETRAINED_METHOD_ID, BICUBIC_METHOD_ID):
            for metric in metrics:
                pivot = condition.pivot(
                    index="source_group_id", columns="method_id", values=metric
                ).loc[sources]
                differences = (pivot[ADAPTED_METHOD_ID] - pivot[reference_method]).to_numpy(
                    dtype=np.float64
                )
                if len(differences) < 2:
                    raise FineTuningError("paired adaptation inference needs at least two sources")
                bootstrap = differences[draws].mean(axis=1)
                low, high = np.quantile(bootstrap, (0.025, 0.975))
                paired_rows.append(
                    {
                        "condition_id": condition_id,
                        "scale": int(condition_id[1]),
                        "profile": condition_id.split("-", 1)[1],
                        "metric": metric,
                        "method_id": ADAPTED_METHOD_ID,
                        "comparator_id": reference_method,
                        "sources": len(differences),
                        "mean_delta": float(differences.mean()),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "sources_improved": int((differences > 0).sum()),
                        "sources_worsened": int((differences < 0).sum()),
                        "sources_tied": int((differences == 0).sum()),
                        "bootstrap_seed": seed,
                        "bootstrap_repetitions": repetitions,
                        "interval_excludes_zero": bool(low > 0 or high < 0),
                    }
                )
    return aggregate, pd.DataFrame(paired_rows)


def write_run_manifest(output_root: Path, payload: Mapping[str, object]) -> Path:
    """Write a checksummed evidence manifest after training and evaluation complete."""

    output_root = Path(output_root).resolve()
    files = {
        str(path.relative_to(output_root)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    manifest = {"run": dict(payload), "files": files}
    path = output_root / "artifact-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
