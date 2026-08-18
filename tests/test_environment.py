from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from score_super_resolution.environment import environment_snapshot


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repository(repository: Path, filename: str, content: str) -> str:
    repository.mkdir(parents=True)
    _git(repository, "init", "--quiet")
    (repository / filename).write_text(content, encoding="utf-8")
    _git(repository, "add", filename)
    _git(
        repository,
        "-c",
        "user.name=Identity Fixture",
        "-c",
        "user.email=identity-fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        f"Add {filename}",
    )
    return _git(repository, "rev-parse", "HEAD")


def test_environment_snapshot_has_reproducibility_fields(tmp_path: Path) -> None:
    snapshot = environment_snapshot(
        tmp_path / "proyecto",
        workspace_root=tmp_path,
        memoria_repository=tmp_path / "memoria",
    )

    assert snapshot.keys() == {
        "git",
        "kaggle",
        "packages",
        "platform",
        "python",
        "repositories",
        "torch",
    }
    assert snapshot["python"]["version"]
    assert snapshot["platform"]["system"]
    assert "numpy" in snapshot["packages"]
    assert "cuda_available" in snapshot["torch"] or snapshot["torch"] == {"installed": False}


def test_environment_snapshot_probes_three_repository_roles_independently(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proyecto = workspace / "proyecto"
    memoria = workspace / "memoria"

    workspace.mkdir()
    (workspace / ".gitignore").write_text("proyecto/\nmemoria/\n", encoding="utf-8")
    workspace_revision = _init_repository(workspace, "planning.txt", "planning root\n")
    proyecto_revision = _init_repository(proyecto, "implementation.py", "VALUE = 1\n")
    (proyecto / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    memoria.mkdir()
    _git(memoria, "init", "--quiet")
    (memoria / "main.tex").write_text("uncommitted thesis\n", encoding="utf-8")

    snapshot = environment_snapshot(
        proyecto,
        workspace_root=workspace,
        memoria_repository=memoria,
    )
    repositories = snapshot["repositories"]

    assert tuple(repositories) == ("workspace-planning", "proyecto", "memoria")
    assert {record["role"] for record in repositories.values()} == {
        "workspace-planning",
        "proyecto",
        "memoria",
    }
    assert repositories["workspace-planning"] == {
        "role": "workspace-planning",
        "root": str(workspace.resolve()),
        "revision_state": "committed",
        "revision": workspace_revision,
        "dirty": False,
    }
    assert repositories["proyecto"] == {
        "role": "proyecto",
        "root": str(proyecto.resolve()),
        "revision_state": "committed",
        "revision": proyecto_revision,
        "dirty": True,
    }
    assert repositories["memoria"] == {
        "role": "memoria",
        "root": str(memoria.resolve()),
        "revision_state": "unborn",
        "revision": None,
        "dirty": True,
    }
    assert workspace_revision != proyecto_revision
    assert snapshot["git"] == {
        "repository": str(proyecto.resolve()),
        "revision": proyecto_revision,
        "dirty": True,
    }


def test_environment_snapshot_represents_unavailable_repository_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    proyecto = workspace / "proyecto"
    missing_memoria = workspace / "missing-memoria"
    _init_repository(workspace, "planning.txt", "planning root\n")
    _init_repository(proyecto, "implementation.py", "VALUE = 1\n")

    snapshot = environment_snapshot(
        proyecto,
        workspace_root=workspace,
        memoria_repository=missing_memoria,
    )

    assert snapshot["repositories"]["memoria"] == {
        "role": "memoria",
        "root": str(missing_memoria.resolve()),
        "revision_state": "unavailable",
        "revision": None,
        "dirty": None,
    }


def test_environment_snapshot_does_not_capture_secret_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-be-captured")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-captured")

    encoded = json.dumps(environment_snapshot(tmp_path))

    assert "must-not-be-captured" not in encoded
    assert "HF_TOKEN" not in encoded
    assert "AWS_SECRET_ACCESS_KEY" not in encoded
