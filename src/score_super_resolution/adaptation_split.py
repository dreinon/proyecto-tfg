"""Outcome-independent SMB partitions for the bounded EDSR adaptation study."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from score_super_resolution.comparison import as_rgb8, ensure_manifest_generation
from score_super_resolution.smb_audit import resolve_active_manifest
from score_super_resolution.staff_scale import (
    ESTIMATOR_ID,
    StaffScaleError,
    canonical_smb_pixel_sha256,
    estimate_staff_spacing,
)

CONFIG_RELATIVE_PATH = Path("configs/experiments/smb-edsr-finetuning-v1.yaml")
SPLIT_RELATIVE_PATH = Path("data/adaptation/smb-edsr-finetuning-v1-split.csv")
EXCLUSIONS_RELATIVE_PATH = Path("data/adaptation/smb-edsr-finetuning-v1-exclusions.csv")
V1_GROUPS_RELATIVE_PATH = Path("data/audits/smb-evaluation-v1-source-groups.csv")
V2_SAMPLE_RELATIVE_PATH = Path("data/audits/smb-evaluation-sample-v2.csv")

SPLIT_FIELDS = (
    "partition",
    "prior_role",
    "upstream_index",
    "item_id",
    "source_group_id",
    "staff_spacing_px",
    "estimator_id",
    "staff_sequence_count",
    "representative_page",
)
EXCLUSION_FIELDS = ("upstream_index", "item_id", "source_group_id", "reason")
PARTITIONS = ("train", "validation", "test")


class AdaptationSplitError(ValueError):
    """The derived SMB adaptation partition is unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True)
class AdaptationSplitRow:
    """One staff-measurable SMB page assigned to exactly one adaptation partition."""

    partition: str
    prior_role: str
    upstream_index: int
    item_id: str
    source_group_id: str
    staff_spacing_px: float
    estimator_id: str
    staff_sequence_count: int
    representative_page: bool


@dataclass(frozen=True)
class FrozenAdaptationSplit:
    """Validated adaptation rows plus the exact identities that authorize their use."""

    rows: tuple[AdaptationSplitRow, ...]
    split_sha256: str
    source_revision: str
    split_seed: int
    config: dict[str, Any]

    def rows_for(
        self, partition: str, *, representative_only: bool = False
    ) -> tuple[AdaptationSplitRow, ...]:
        if partition not in PARTITIONS:
            raise AdaptationSplitError("partition must be train, validation, or test")
        return tuple(
            row
            for row in self.rows
            if row.partition == partition and (row.representative_page or not representative_only)
        )


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        if path.is_symlink() or not path.is_file():
            raise AdaptationSplitError("adaptation CSV must be a regular non-symlink file")
        payload = path.read_bytes()
        reader = csv.DictReader(payload.decode("utf-8").splitlines())
        rows = list(reader)
    except AdaptationSplitError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise AdaptationSplitError("adaptation CSV cannot be read safely") from error
    return tuple(reader.fieldnames or ()), rows


def _read_config(project_root: Path) -> dict[str, Any]:
    path = project_root / CONFIG_RELATIVE_PATH
    try:
        if path.is_symlink() or not path.is_file():
            raise AdaptationSplitError("adaptation config must be a regular non-symlink file")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except AdaptationSplitError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise AdaptationSplitError("adaptation config cannot be read safely") from error
    if not isinstance(loaded, dict):
        raise AdaptationSplitError("adaptation config root must be a mapping")
    return loaded


def _csv_group_set(path: Path, field: str) -> set[str]:
    fields, rows = _read_csv(path)
    if field not in fields:
        raise AdaptationSplitError("source-group CSV does not expose the expected field")
    values = {row[field] for row in rows}
    if not values or "" in values:
        raise AdaptationSplitError("source-group CSV contains an empty identity")
    return values


def _rank(seed: int, domain: str, identity: str) -> str:
    framed = f"smb-edsr-finetuning-v1\0{seed}\0{domain}\0{identity}".encode()
    return hashlib.sha256(framed).hexdigest()


def _canonical_csv(fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return target.getvalue().encode("utf-8")


def derive_adaptation_split(
    project_root: Path,
    dataset: Sequence[Mapping[str, object]],
) -> tuple[bytes, bytes, dict[str, object]]:
    """Derive the frozen split using only source identities and input-side staff estimates.

    The prior v1 development sources become training data. Every final v2 source remains excluded.
    Validation and test are selected from source groups that appeared in neither prior experiment.
    No SR output, metric, checkpoint choice, or qualitative outcome is consulted.
    """

    project_root = Path(project_root).resolve()
    config = _read_config(project_root)
    split = config.get("split")
    dataset_config = config.get("dataset")
    if not isinstance(split, Mapping) or not isinstance(dataset_config, Mapping):
        raise AdaptationSplitError("adaptation config is missing split or dataset controls")
    try:
        split_seed = int(split["seed"])
        expected_fresh_groups = int(split["expected_fresh_group_count"])
        test_group_count = int(split["test_group_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise AdaptationSplitError("adaptation split controls are incomplete") from error
    if dataset_config.get("repository_id") != "PRAIG/SMB" or dataset_config.get("split") != "test":
        raise AdaptationSplitError("adaptation config identifies an unexpected upstream source")

    v1_groups = _csv_group_set(project_root / V1_GROUPS_RELATIVE_PATH, "source_group_id")
    v2_groups = _csv_group_set(project_root / V2_SAMPLE_RELATIVE_PATH, "source_group_id")
    if v1_groups & v2_groups:
        raise AdaptationSplitError("prior development and final source groups unexpectedly overlap")

    generation_root = ensure_manifest_generation(project_root)
    _, manifest_rows = resolve_active_manifest(
        active_path=project_root / "data/manifests/smb-evaluation-v1.yaml",
        generation_root=generation_root,
    )
    candidates = [
        row
        for row in manifest_rows
        if row["processing_status"] == "processed"
        and row["paired_eligible"] is True
        and row["source_group_id"] not in v2_groups
    ]
    if len(dataset) != 685:
        raise AdaptationSplitError("runtime SMB length differs from the audited manifest")

    usable: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for manifest_row in candidates:
        upstream_index = int(manifest_row["upstream_index"])
        source = dataset[upstream_index]
        source_image = source["image"]
        if canonical_smb_pixel_sha256(source_image) != manifest_row["image"]["pixel_sha256"]:
            raise AdaptationSplitError("runtime SMB pixels differ from the audited manifest")
        try:
            estimate = estimate_staff_spacing(as_rgb8(source_image), source["regions"])
        except StaffScaleError as error:
            exclusions.append(
                {
                    "upstream_index": upstream_index,
                    "item_id": manifest_row["item_id"],
                    "source_group_id": manifest_row["source_group_id"],
                    "reason": str(error),
                }
            )
            continue
        usable.append(
            {
                "upstream_index": upstream_index,
                "item_id": manifest_row["item_id"],
                "source_group_id": manifest_row["source_group_id"],
                "staff_spacing_px": f"{estimate.spacing_px:.6f}",
                "estimator_id": estimate.estimator_id,
                "staff_sequence_count": estimate.sequence_count,
            }
        )

    usable_groups = {str(row["source_group_id"]) for row in usable}
    train_groups = usable_groups & v1_groups
    fresh_groups = usable_groups - v1_groups - v2_groups
    if len(fresh_groups) != expected_fresh_groups:
        raise AdaptationSplitError("fresh measurable group count differs from the frozen design")
    if not 1 <= test_group_count < len(fresh_groups):
        raise AdaptationSplitError("test group count is outside the fresh-group boundary")
    ranked_fresh = sorted(
        fresh_groups, key=lambda value: (_rank(split_seed, "group", value), value)
    )
    test_groups = set(ranked_fresh[:test_group_count])
    validation_groups = set(ranked_fresh[test_group_count:])
    if not train_groups or not validation_groups or not test_groups:
        raise AdaptationSplitError("every adaptation partition must contain source groups")

    partition_by_group = {
        **{group: "train" for group in train_groups},
        **{group: "validation" for group in validation_groups},
        **{group: "test" for group in test_groups},
    }
    rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in usable:
        rows_by_group[str(row["source_group_id"])].append(row)
    representatives = {
        group: min(
            group_rows,
            key=lambda row: (
                _rank(split_seed, "representative", str(row["item_id"])),
                str(row["item_id"]),
            ),
        )["item_id"]
        for group, group_rows in rows_by_group.items()
    }

    output_rows: list[dict[str, object]] = []
    partition_order = {name: index for index, name in enumerate(PARTITIONS)}
    for row in usable:
        group = str(row["source_group_id"])
        partition = partition_by_group[group]
        output_rows.append(
            {
                "partition": partition,
                "prior_role": "v1-development" if group in v1_groups else "fresh-holdout",
                **row,
                "representative_page": str(row["item_id"] == representatives[group]),
            }
        )
    output_rows.sort(
        key=lambda row: (
            partition_order[str(row["partition"])],
            str(row["source_group_id"]),
            int(row["upstream_index"]),
        )
    )
    exclusions.sort(key=lambda row: int(row["upstream_index"]))
    split_bytes = _canonical_csv(SPLIT_FIELDS, output_rows)
    exclusion_bytes = _canonical_csv(EXCLUSION_FIELDS, exclusions)
    group_counts = Counter(partition_by_group.values())
    page_counts = Counter(str(row["partition"]) for row in output_rows)
    summary: dict[str, object] = {
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "exclusions_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
        "page_counts": dict(page_counts),
        "group_counts": dict(group_counts),
        "excluded_v2_group_count": len(v2_groups),
        "v1_training_group_count": len(train_groups),
        "fresh_group_count": len(fresh_groups),
        "input_only_exclusion_count": len(exclusions),
    }
    return split_bytes, exclusion_bytes, summary


def load_frozen_adaptation_split(project_root: Path) -> FrozenAdaptationSplit:
    """Load and verify the exact tracked split before any fitting, selection, or test use."""

    project_root = Path(project_root).resolve()
    config = _read_config(project_root)
    split_config = config.get("split")
    dataset_config = config.get("dataset")
    if not isinstance(split_config, Mapping) or not isinstance(dataset_config, Mapping):
        raise AdaptationSplitError("adaptation config is incomplete")
    split_path = project_root / SPLIT_RELATIVE_PATH
    try:
        split_bytes = split_path.read_bytes()
    except OSError as error:
        raise AdaptationSplitError("frozen adaptation split is unavailable") from error
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 != split_config.get("sha256"):
        raise AdaptationSplitError("frozen adaptation split digest differs")
    exclusion_path = project_root / EXCLUSIONS_RELATIVE_PATH
    try:
        exclusion_bytes = exclusion_path.read_bytes()
    except OSError as error:
        raise AdaptationSplitError("frozen adaptation exclusions are unavailable") from error
    if hashlib.sha256(exclusion_bytes).hexdigest() != split_config.get("exclusions_sha256"):
        raise AdaptationSplitError("frozen adaptation exclusion digest differs")
    exclusion_fields, exclusions = _read_csv(exclusion_path)
    if exclusion_fields != EXCLUSION_FIELDS or len(exclusions) != split_config.get(
        "input_only_exclusion_count"
    ):
        raise AdaptationSplitError("frozen adaptation exclusion shape differs")
    try:
        excluded_indices = {int(row["upstream_index"]) for row in exclusions}
    except (KeyError, TypeError, ValueError) as error:
        raise AdaptationSplitError("frozen adaptation exclusion values are malformed") from error
    if len(excluded_indices) != len(exclusions) or any(
        not row["item_id"] or not row["source_group_id"] or not row["reason"] for row in exclusions
    ):
        raise AdaptationSplitError("frozen adaptation exclusions are incomplete")
    fields, raw_rows = _read_csv(split_path)
    if fields != SPLIT_FIELDS or not raw_rows:
        raise AdaptationSplitError("frozen adaptation split columns differ")
    try:
        rows = tuple(
            AdaptationSplitRow(
                partition=row["partition"],
                prior_role=row["prior_role"],
                upstream_index=int(row["upstream_index"]),
                item_id=row["item_id"],
                source_group_id=row["source_group_id"],
                staff_spacing_px=float(row["staff_spacing_px"]),
                estimator_id=row["estimator_id"],
                staff_sequence_count=int(row["staff_sequence_count"]),
                representative_page=row["representative_page"] == "True",
            )
            for row in raw_rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AdaptationSplitError("frozen adaptation split values are malformed") from error

    if any(
        row.partition not in PARTITIONS
        or row.prior_role not in {"v1-development", "fresh-holdout"}
        or row.estimator_id != ESTIMATOR_ID
        or row.upstream_index < 0
        or not 4.0 <= row.staff_spacing_px <= 32.0
        or row.staff_sequence_count < 2
        for row in rows
    ):
        raise AdaptationSplitError("frozen adaptation split contains an unknown value")
    if (
        len({row.item_id for row in rows}) != len(rows)
        or len({row.upstream_index for row in rows}) != len(rows)
        or {row.upstream_index for row in rows} & excluded_indices
    ):
        raise AdaptationSplitError("frozen adaptation split repeats an item")
    group_partitions: dict[str, set[str]] = defaultdict(set)
    representatives: Counter[str] = Counter()
    for row in rows:
        group_partitions[row.source_group_id].add(row.partition)
        if row.representative_page:
            representatives[row.source_group_id] += 1
    if any(len(values) != 1 for values in group_partitions.values()):
        raise AdaptationSplitError("a source group crosses adaptation partitions")
    if set(representatives.values()) != {1} or set(representatives) != set(group_partitions):
        raise AdaptationSplitError("each source group must expose one representative page")

    v2_groups = _csv_group_set(project_root / V2_SAMPLE_RELATIVE_PATH, "source_group_id")
    v1_groups = _csv_group_set(project_root / V1_GROUPS_RELATIVE_PATH, "source_group_id")
    if set(group_partitions) & v2_groups:
        raise AdaptationSplitError("final v2 sources entered the adaptation split")
    if any(row.prior_role == "v1-development" and row.partition != "train" for row in rows):
        raise AdaptationSplitError("prior development sources may only enter training")
    if any(row.prior_role == "fresh-holdout" and row.partition == "train" for row in rows):
        raise AdaptationSplitError("fresh holdout sources may not enter training")
    train_groups = {row.source_group_id for row in rows if row.partition == "train"}
    holdout_groups = set(group_partitions) - train_groups
    if not train_groups <= v1_groups or holdout_groups & v1_groups:
        raise AdaptationSplitError("adaptation prior roles differ from the v1 development sources")

    configured_groups = split_config.get("group_counts")
    configured_pages = split_config.get("page_counts")
    actual_groups = Counter(next(iter(values)) for values in group_partitions.values())
    actual_pages = Counter(row.partition for row in rows)
    if configured_groups != dict(actual_groups) or configured_pages != dict(actual_pages):
        raise AdaptationSplitError("frozen adaptation denominators differ from the config")
    revision = dataset_config.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise AdaptationSplitError("adaptation source revision is malformed")
    return FrozenAdaptationSplit(
        rows=rows,
        split_sha256=split_sha256,
        source_revision=revision,
        split_seed=int(split_config["seed"]),
        config=config,
    )
