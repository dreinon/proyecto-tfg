"""Runtime metadata helpers for reproducible local and Kaggle experiments."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "datasets",
    "kaggle",
    "kagglehub",
    "matplotlib",
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "safetensors",
    "scikit-image",
    "seaborn",
    "tensorboard",
    "torch",
    "torchvision",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git(repository: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state(repository: Path, role: str) -> dict[str, Any]:
    requested_root = repository.expanduser().resolve()
    discovered_root = _git(requested_root, "rev-parse", "--show-toplevel")
    if discovered_root is None:
        return {
            "role": role,
            "root": str(requested_root),
            "revision_state": "unavailable",
            "revision": None,
            "dirty": None,
        }

    root = Path(discovered_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    if status is None:
        return {
            "role": role,
            "root": str(root),
            "revision_state": "unavailable",
            "revision": None,
            "dirty": None,
        }

    revision = _git(root, "rev-parse", "--verify", "HEAD")
    return {
        "role": role,
        "root": str(root),
        "revision_state": "committed" if revision is not None else "unborn",
        "revision": revision,
        "dirty": bool(status),
    }


def _torch_state() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }


def environment_snapshot(
    repository: Path | None = None,
    *,
    workspace_root: Path | None = None,
    memoria_repository: Path | None = None,
) -> dict[str, Any]:
    """Return secret-safe runtime metadata with three independent repository identities."""

    proyecto_root = (repository or Path(__file__).resolve().parents[2]).expanduser().resolve()
    planning_root = (workspace_root or proyecto_root.parent).expanduser().resolve()
    memoria_root = (memoria_repository or planning_root / "memoria").expanduser().resolve()
    repositories = {
        "workspace-planning": _git_state(planning_root, "workspace-planning"),
        "proyecto": _git_state(proyecto_root, "proyecto"),
        "memoria": _git_state(memoria_root, "memoria"),
    }
    proyecto_state = repositories["proyecto"]
    kaggle_keys = (
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_KERNEL_INTEGRATIONS",
        "KAGGLE_URL_BASE",
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "repository": proyecto_state["root"],
            "revision": proyecto_state["revision"],
            "dirty": proyecto_state["dirty"],
        },
        "repositories": repositories,
        "torch": _torch_state(),
        "packages": _package_versions(),
        "kaggle": {key: os.environ.get(key) for key in kaggle_keys},
    }


def main() -> None:
    """Print the current environment snapshot as stable, readable JSON."""

    print(json.dumps(environment_snapshot(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
