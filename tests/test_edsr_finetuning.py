from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from score_super_resolution.adaptation_split import (
    AdaptationSplitRow,
    load_frozen_adaptation_split,
)
from score_super_resolution.edsr_finetuning import (
    ADAPTED_METHOD_ID,
    BICUBIC_METHOD_ID,
    PRETRAINED_METHOD_ID,
    RESULT_FIELDS,
    AdaptationDataPreflight,
    FineTuningError,
    PatchBatchFactory,
    analyze_adaptation_results,
    fine_tune_edsr_scale,
    sample_notation_patch,
)

ROOT = Path(__file__).resolve().parents[1]


def _page(value: int = 255) -> tuple[Image.Image, list[dict[str, object]]]:
    pixels = np.full((256, 320, 3), value, dtype=np.uint8)
    pixels[80:176:8, 20:300] = 0
    return Image.fromarray(pixels), [
        {"bbox": {"x": 0.0, "y": 20.0, "width": 100.0, "height": 60.0}}
    ]


def _row(
    partition: str,
    *,
    index: int = 0,
    group: str = "work-a",
    representative: bool = True,
) -> AdaptationSplitRow:
    return AdaptationSplitRow(
        partition=partition,
        prior_role="v1-development" if partition == "train" else "fresh-holdout",
        upstream_index=index,
        item_id=f"smb-test-{index:06d}",
        source_group_id=group,
        staff_spacing_px=8.0,
        estimator_id="staff-spacing-projection-v1",
        staff_sequence_count=2,
        representative_page=representative,
    )


def test_frozen_adaptation_split_is_source_disjoint_and_excludes_final_v2() -> None:
    split = load_frozen_adaptation_split(ROOT)
    groups = {
        partition: {row.source_group_id for row in split.rows_for(partition)}
        for partition in ("train", "validation", "test")
    }
    pages = {partition: len(split.rows_for(partition)) for partition in groups}
    with (ROOT / "data/audits/smb-evaluation-sample-v2.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        final_v2_groups = {row["source_group_id"] for row in csv.DictReader(source)}

    assert split.split_sha256 == (
        "ee3e2834679a184168e9fe689eb3e9575d450dbd21be36090eebdb544477075f"
    )
    assert {partition: len(values) for partition, values in groups.items()} == {
        "train": 45,
        "validation": 13,
        "test": 20,
    }
    assert pages == {"train": 212, "validation": 35, "test": 55}
    assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert not (set.union(*groups.values()) & final_v2_groups)
    assert all(row.prior_role == "v1-development" for row in split.rows_for("train"))
    assert all(
        row.prior_role == "fresh-holdout"
        for partition in ("validation", "test")
        for row in split.rows_for(partition)
    )


def test_qualitative_assignment_was_frozen_without_reusing_sources() -> None:
    split = load_frozen_adaptation_split(ROOT)
    test_rows = {
        (row.item_id, row.source_group_id, row.upstream_index)
        for row in split.rows_for("test", representative_only=True)
    }
    assignment = split.config["evaluation"]["qualitative_assignment"]

    assert len(assignment) == 6
    assert len({row["condition_id"] for row in assignment}) == 6
    assert len({row["source_group_id"] for row in assignment}) == 6
    assert all(
        (row["item_id"], row["source_group_id"], row["upstream_index"]) in test_rows
        for row in assignment
    )


def test_notation_patch_is_deterministic_and_contains_ink() -> None:
    image, regions = _page()
    pixels = np.asarray(image)
    first = sample_notation_patch(pixels, regions, patch_size=96, seed=31)
    second = sample_notation_patch(pixels, regions, patch_size=96, seed=31)

    assert np.array_equal(first, second)
    assert first.shape == (96, 96, 3)
    assert np.mean(first < 235) > 0.01


def test_validation_factory_uses_only_the_frozen_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, regions = _page()
    dataset = [{"image": image, "regions": regions}, {"image": image, "regions": regions}]
    rows = (
        _row("validation", index=0, representative=False),
        _row("validation", index=1, representative=True),
    )
    observed: list[int] = []

    def fake_example(
        self: PatchBatchFactory,
        row: AdaptationSplitRow,
        *,
        scale: int,
        token: object,
        profile_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed.append(row.upstream_index)
        return np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(PatchBatchFactory, "_example_from_row", fake_example)
    factory = PatchBatchFactory(ROOT, dataset, rows, seed=7, lr_patch_size=32)

    assert len(factory.validation_examples(scale=2)) == 3
    assert observed == [1, 1, 1]


def test_patch_factory_never_accepts_test_pages() -> None:
    image, regions = _page()
    with pytest.raises(FineTuningError, match="train or validation"):
        PatchBatchFactory(
            ROOT,
            [{"image": image, "regions": regions}],
            (_row("test"),),
            seed=7,
            lr_patch_size=32,
        )


def test_paired_analysis_keeps_source_as_the_inference_unit() -> None:
    records: list[dict[str, object]] = []
    methods = (BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID)
    for source_index in range(3):
        for method_index, method in enumerate(methods):
            records.append(
                {
                    "upstream_index": source_index,
                    "item_id": f"item-{source_index}",
                    "source_group_id": f"work-{source_index}",
                    "condition_id": "x2-clean",
                    "scale": 2,
                    "profile": "clean",
                    "method_id": method,
                    "psnr_y": 20.0 + method_index,
                    "ssim_y": 0.80 + 0.01 * method_index,
                    "psnr_rgb": 19.0 + method_index,
                    "ssim_rgb": 0.79 + 0.01 * method_index,
                    "runtime_seconds": 0.1 + method_index,
                    "output_sha256": f"output-{source_index}-{method}",
                    "checkpoint_sha256": f"checkpoint-{method}",
                }
            )
    results = pd.DataFrame(records, columns=RESULT_FIELDS)

    aggregate, paired = analyze_adaptation_results(results, seed=9, repetitions=100)

    assert set(aggregate["sources"]) == {3}
    assert set(paired["sources"]) == {3}
    versus_pretrained = paired[paired["comparator_id"] == PRETRAINED_METHOD_ID]
    assert set(versus_pretrained["mean_delta"].round(6)) == {0.01, 1.0}


def test_training_contract_keeps_test_closed_until_both_models_are_selected() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/smb-edsr-finetuning-v1.yaml").read_text(encoding="utf-8")
    )

    assert config["status"] == "frozen-before-training"
    assert config["selection"]["test_access"] == "after-both-scale-checkpoints-selected"
    assert config["training"]["objective"] == "rgb-l1"
    assert config["methods"]["scales"] == [2, 4]
    assert config["evaluation"]["aggregation_unit"] == "source_group_id"
    assert config["evaluation"]["bootstrap_repetitions"] == 2000


def test_adaptation_notebook_is_generated_and_keeps_the_test_gate_explicit() -> None:
    notebook_path = ROOT / "notebooks/04-smb-edsr-finetuning.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert source.index('for scale in config["methods"]["scales"]') < source.index(
        'partitions=("test",)'
    )
    assert "if set(runs) != {2, 4}" in source
    assert '"raw_rows_360": len(results) == 360' in source
    assert '"qualitative_pngs_30": len(qualitative_index) == 30' in source
    assert '["uv", "pip", "install", "--system", "--no-deps", "-e"' in source
    assert "torch_after != torch_before" in source


def test_adaptation_notebook_matches_its_builder() -> None:
    builder = runpy.run_path(str(ROOT / "scripts/build_edsr_finetuning_notebook.py"))
    expected = json.loads(nbformat.writes(builder["build_notebook"]()))
    observed = json.loads(
        (ROOT / "notebooks/04-smb-edsr-finetuning.ipynb").read_text(encoding="utf-8")
    )

    assert observed["cells"] == expected["cells"]
    assert observed["metadata"] == expected["metadata"]


class _TinyUpscaler(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.gain = nn.Parameter(torch.tensor(0.9))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.interpolate(
            inputs, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        return output * self.gain


def test_training_smoke_saves_a_split_bound_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = load_frozen_adaptation_split(ROOT)

    class FakeFactory:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def validation_examples(self, *, scale: int) -> list[tuple[np.ndarray, np.ndarray]]:
            low = np.full((8, 8, 3), 160, dtype=np.uint8)
            high = np.full((8 * scale, 8 * scale, 3), 180, dtype=np.uint8)
            return [(low, high)]

        def training_batch(
            self, *, scale: int, step: int, batch_size: int, device: torch.device
        ) -> tuple[torch.Tensor, torch.Tensor]:
            low = torch.full((batch_size, 3, 8, 8), 0.6, device=device)
            high = torch.full((batch_size, 3, 8 * scale, 8 * scale), 0.7, device=device)
            return low, high

    monkeypatch.setattr(
        "score_super_resolution.edsr_finetuning._load_pretrained_edsr",
        lambda project_root, scale, device: (_TinyUpscaler(scale).to(device), "a" * 64),
    )
    monkeypatch.setattr("score_super_resolution.edsr_finetuning.PatchBatchFactory", FakeFactory)

    run = fine_tune_edsr_scale(
        ROOT,
        [],
        split,
        scale=2,
        output_root=tmp_path,
        data_preflight=AdaptationDataPreflight(
            split_sha256=split.split_sha256,
            partitions=("train", "validation"),
            pages=247,
            groups=58,
            group_counts=(("train", 45), ("validation", 13)),
        ),
        device="cpu",
        steps_override=2,
        batch_size_override=1,
    )

    assert run.completed_steps == 2
    assert run.checkpoint_path.is_file()
    assert len(run.checkpoint_sha256) == 64
    payload = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["split_sha256"] == split.split_sha256
    assert payload["source_revision"] == split.source_revision
    assert payload["method_id"] == ADAPTED_METHOD_ID
