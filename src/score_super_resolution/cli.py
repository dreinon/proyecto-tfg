"""Safe fixture-only command line for the Phase 2 execution lifecycle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from score_super_resolution.execution import (
    ExecutionBusyError,
    ExecutionContractError,
    ReconciliationError,
    audit_no_smb,
    execute_run,
    export_reconciled_run,
    load_experiment_config,
    reconcile_run,
    resume_run,
    snapshot_run,
)


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ExecutionContractError("invalid command arguments")


def _relative_path(raw: str, *, must_exist: bool) -> Path:
    value = Path(raw)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ExecutionContractError("CLI paths must be canonical and relative")
    resolved = (Path.cwd() / value).resolve()
    if not resolved.is_relative_to(Path.cwd().resolve()):
        raise ExecutionContractError("CLI path escapes the current project")
    current = Path.cwd().resolve()
    for part in value.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ExecutionContractError("CLI path contains a symlink")
    if must_exist and not resolved.exists():
        raise ExecutionContractError("CLI input does not exist")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = _SafeParser(description="Run the closed Phase 2 fixture experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    status = commands.add_parser("status")
    status.add_argument("--artifact-root", required=True)
    for name in ("run", "resume", "reconcile", "export"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--artifact-root", required=True)
        if name in {"run", "resume"}:
            command.add_argument("--max-tuples", type=int)
    audit = commands.add_parser("audit-no-smb")
    audit.add_argument("--artifact-root", required=True)
    return parser


def _json(value: Any) -> None:
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "validate":
        config = load_experiment_config(_relative_path(arguments.config, must_exist=True))
        return {
            "status": "valid",
            "experiment_id": config.experiment_id,
            "experiment_sha256": config.experiment_sha256,
            "expected_tuple_count": len(config.fixture_items)
            * len(config.condition_ids)
            * len(config.method_ids),
        }
    root = _relative_path(arguments.artifact_root, must_exist=arguments.command == "status")
    if arguments.command == "status":
        snapshot = asdict(snapshot_run(root))
        return {"status": "ok", **snapshot}
    if arguments.command == "audit-no-smb":
        return audit_no_smb(root)
    config = _relative_path(arguments.config, must_exist=True)
    if arguments.command == "run":
        return {
            "status": "ok",
            **asdict(execute_run(config, root, max_tuples=arguments.max_tuples)),
        }
    if arguments.command == "resume":
        return {"status": "ok", **asdict(resume_run(config, root, max_tuples=arguments.max_tuples))}
    if arguments.command == "reconcile":
        report = reconcile_run(config, root)
        return {"status": "ok", **report.payload}
    bundle = export_reconciled_run(config, root)
    return {
        "status": "ok",
        "experiment_id": bundle.experiment_id,
        "reconciliation_id": bundle.reconciliation_id,
        "manifest_relative_path": bundle.manifest_path.relative_to(root).as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        _json(_dispatch(arguments))
        return 0
    except ExecutionBusyError:
        _json({"status": "error", "code": "busy"})
        return 3
    except ReconciliationError:
        _json({"status": "error", "code": "reconciliation_failed"})
        return 4
    except (ExecutionContractError, OSError, ValueError):
        _json({"status": "error", "code": "invalid_request"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
