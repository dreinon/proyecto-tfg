"""Secret-safe, exact-revision access to the quarantined SMB benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from score_super_resolution.benchmark_policy import (
    BenchmarkPurpose,
    assert_smb_purpose_allowed,
)

SMB_DESCRIPTOR_PATH = Path(__file__).resolve().parents[2] / "data" / "sources" / "smb.yaml"

_EXPECTED_REPOSITORY_ID = "PRAIG/SMB"
_EXPECTED_SPLIT = "test"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_METADATA_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")

DatasetLoader = Callable[..., object]
BuilderLoader = Callable[..., object]


class SmbDescriptorError(ValueError):
    """Report an invalid local SMB source descriptor."""


class SmbMetadataError(ValueError):
    """Report metadata that cannot be represented by the safe diagnostic."""


def _read_descriptor(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SmbDescriptorError("SMB source descriptor must be a mapping")

    if loaded.get("key") != "smb" or loaded.get("role") != "evaluation_benchmark":
        raise SmbDescriptorError("SMB source descriptor has an invalid key or role")
    if loaded.get("provider") != "hugging_face":
        raise SmbDescriptorError("SMB source descriptor has an invalid provider")
    if loaded.get("repository_id") != _EXPECTED_REPOSITORY_ID:
        raise SmbDescriptorError("SMB source descriptor identifies an unexpected repository")

    revision = loaded.get("revision")
    if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
        raise SmbDescriptorError("SMB source descriptor revision must be a 40-character commit")

    upstream_splits = loaded.get("upstream_splits")
    if not isinstance(upstream_splits, Mapping) or set(upstream_splits) != {_EXPECTED_SPLIT}:
        raise SmbDescriptorError("SMB source descriptor must declare only the test split")
    split_metadata = upstream_splits[_EXPECTED_SPLIT]
    if not isinstance(split_metadata, Mapping):
        raise SmbDescriptorError("SMB test split metadata must be a mapping")
    expected_examples = split_metadata.get("examples")
    if (
        isinstance(expected_examples, bool)
        or not isinstance(expected_examples, int)
        or expected_examples < 1
    ):
        raise SmbDescriptorError("SMB test split must declare a positive example count")

    features = loaded.get("features")
    if (
        not isinstance(features, Sequence)
        or isinstance(features, (str, bytes, bytearray))
        or not features
        or any(not isinstance(feature, str) for feature in features)
        or len(set(features)) != len(features)
    ):
        raise SmbDescriptorError("SMB source descriptor features must be unique names")

    loading = loaded.get("loading")
    if (
        not isinstance(loading, Mapping)
        or loading.get("library") != "datasets"
        or loading.get("call") != "load_dataset"
    ):
        raise SmbDescriptorError("SMB source descriptor has an invalid loading contract")

    access = loaded.get("access")
    if (
        not isinstance(access, Mapping)
        or access.get("store_credentials_in_repository") is not False
    ):
        raise SmbDescriptorError("SMB source descriptor must prohibit stored credentials")

    return loaded


def _default_dataset_loader(repository_id: str, *, split: str, revision: str) -> object:
    from datasets import load_dataset

    return load_dataset(repository_id, split=split, revision=revision)


def _default_builder_loader(repository_id: str, *, revision: str) -> object:
    from datasets import load_dataset_builder

    return load_dataset_builder(repository_id, revision=revision)


def load_smb(
    *,
    purpose: BenchmarkPurpose | str,
    loader: DatasetLoader | None = None,
    descriptor_path: Path = SMB_DESCRIPTOR_PATH,
) -> object:
    """Load the sole SMB split at its immutable revision after the policy guard passes.

    Authentication is intentionally absent from this API. Hugging Face resolves approved access
    through its standard cached-login or runtime-secret mechanisms.
    """

    descriptor = _read_descriptor(descriptor_path)
    repository_id = descriptor["repository_id"]
    revision = descriptor["revision"]
    split = next(iter(descriptor["upstream_splits"]))
    selected_loader = loader or _default_dataset_loader

    return assert_smb_purpose_allowed(
        source_descriptor=descriptor,
        purpose=purpose,
        callback=lambda: selected_loader(repository_id, split=split, revision=revision),
    )


def _safe_metadata_name(value: object) -> str:
    if not isinstance(value, str) or _METADATA_NAME_PATTERN.fullmatch(value) is None:
        raise SmbMetadataError("remote metadata contains an unsafe name")
    return value


def _metadata_summary(builder: object, descriptor: Mapping[str, Any]) -> dict[str, object]:
    info = getattr(builder, "info", None)
    features = getattr(info, "features", None)
    splits = getattr(info, "splits", None)
    if not isinstance(features, Mapping) or not isinstance(splits, Mapping):
        raise SmbMetadataError("remote metadata is incomplete")

    feature_names = sorted(_safe_metadata_name(name) for name in features)
    split_counts: dict[str, dict[str, int]] = {}
    for raw_name, split_info in splits.items():
        name = _safe_metadata_name(raw_name)
        count = getattr(split_info, "num_examples", None)
        if count is None and isinstance(split_info, Mapping):
            count = split_info.get("num_examples")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SmbMetadataError("remote split metadata has an invalid example count")
        split_counts[name] = {"examples": count}

    expected_features = sorted(descriptor["features"])
    expected_splits = {
        name: {"examples": values["examples"]}
        for name, values in descriptor["upstream_splits"].items()
    }
    return {
        "status": "accessible",
        "repository_id": descriptor["repository_id"],
        "resolved_revision": descriptor["revision"],
        "features": feature_names,
        "splits": split_counts,
        "descriptor_match": feature_names == expected_features and split_counts == expected_splits,
    }


def check_access(
    *,
    builder_loader: BuilderLoader | None = None,
    descriptor_path: Path = SMB_DESCRIPTOR_PATH,
) -> dict[str, object]:
    """Return allowlisted builder metadata or a credential-safe access blocker.

    This function never requests the full dataset, decodes images, or reproduces exception text.
    """

    try:
        descriptor = _read_descriptor(descriptor_path)
        selected_loader = builder_loader or _default_builder_loader
        builder = assert_smb_purpose_allowed(
            source_descriptor=descriptor,
            purpose=BenchmarkPurpose.METADATA_INSPECTION,
            callback=lambda: selected_loader(
                descriptor["repository_id"], revision=descriptor["revision"]
            ),
        )
        return _metadata_summary(builder, descriptor)
    except Exception:
        return {
            "status": "blocked",
            "reason": "SMB metadata access unavailable",
            "action": "Confirm approved Hugging Face access and cached authentication, then retry.",
        }


class _SafeArgumentParser(argparse.ArgumentParser):
    """Avoid reflecting arbitrary command arguments into diagnostics."""

    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command arguments\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Secret-safe SMB access diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-access", help="check exact-revision metadata access")
    check.add_argument(
        "--metadata-only",
        action="store_true",
        required=True,
        help="inspect builder metadata without loading examples",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit metadata-only access diagnostic."""

    arguments = _parser().parse_args(argv)
    if arguments.command != "check-access" or not arguments.metadata_only:
        return 2

    diagnostic = check_access()
    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    return 0 if diagnostic["status"] == "accessible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
