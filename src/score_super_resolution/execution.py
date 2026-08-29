"""Recoverable fixture-only execution with a closed SQLite tuple ledger."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.identities import canonical_sha256, experiment_identity

EXPECTED_METHODS = (
    "nearest-opencv-exact-v1",
    "bilinear-opencv-exact-v1",
    "bicubic-opencv-v1",
)
EXPECTED_CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
TERMINAL_STATES = frozenset({"succeeded", "failed", "excluded"})
ALL_STATES = frozenset({"expected", "running", "retry_pending", *TERMINAL_STATES})
_MAX_CONFIG_BYTES = 1_048_576
_LOCK_NAME = ".phase2-writer.lock"
_LEDGER_NAME = "phase2-ledger.sqlite3"


class ExecutionContractError(ValueError):
    """A config, artefact path, or ledger record violates the execution contract."""


class ExecutionBusyError(ExecutionContractError):
    """Another coordinator already owns the artifact-root writer authority."""


class ReconciliationError(ExecutionContractError):
    """The closed expected tuple set cannot yet be reconciled."""


@dataclass(frozen=True, order=True)
class ExpectedTuple:
    """Complete scientific identity for one fixture computation."""

    item_id: str
    source_group_id: str
    condition_id: str
    method_id: str

    @property
    def tuple_id(self) -> str:
        return f"tuple-{canonical_sha256(self.__dict__)}"


@dataclass(frozen=True)
class ExperimentConfig:
    payload: dict[str, Any]
    project_root: Path
    experiment_id: str
    experiment_sha256: str
    method_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    fixture_items: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    experiment_sha256: str
    expected_tuple_count: int
    ledger_relative_path: str


@dataclass(frozen=True)
class RunSnapshot:
    experiment_id: str
    total: int
    counts: dict[str, int]
    attempt_count: int
    integrity_incident_count: int


@dataclass(frozen=True)
class ReconciliationReport:
    payload: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class ExecutionReport:
    experiment_id: str
    claimed: int
    succeeded: int
    failed: int
    skipped: int
    interrupted: bool


@dataclass(frozen=True)
class ExportBundle:
    experiment_id: str
    reconciliation_id: str
    manifest_path: Path


@dataclass(frozen=True)
class _WriterAuthority:
    root: Path
    descriptor: int


def _secret_like_key(value: Any, path: tuple[str, ...] = ()) -> str | None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            parts = set(normalized.split("_"))
            if normalized in {"api_key", "apikey", "access_key", "private_key"} or parts & {
                "authorization",
                "credential",
                "credentials",
                "password",
                "secret",
                "token",
            }:
                return ".".join((*path, key))
            found = _secret_like_key(child, (*path, key))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _secret_like_key(child, (*path, str(index)))
            if found is not None:
                return found
    return None


def _read_regular(path: Path, *, maximum_bytes: int, kind: str) -> bytes:
    path = Path(path)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ExecutionContractError(f"{kind} must be a regular non-symlink file")
        if metadata.st_size > maximum_bytes:
            raise ExecutionContractError(f"{kind} exceeds the byte bound")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            data = bytearray()
            while chunk := os.read(descriptor, min(65_536, maximum_bytes + 1 - len(data))):
                data.extend(chunk)
                if len(data) > maximum_bytes:
                    raise ExecutionContractError(f"{kind} exceeds the byte bound")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except ExecutionContractError:
        raise
    except OSError as error:
        raise ExecutionContractError(f"{kind} cannot be read safely") from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if identity(before) != identity(after):
        raise ExecutionContractError(f"{kind} changed while being read")
    return bytes(data)


def _project_relative(root: Path, raw: str, *, kind: str) -> Path:
    if not isinstance(raw, str) or not raw or raw.startswith("/") or "\\" in raw:
        raise ExecutionContractError(f"{kind} must be a canonical project-relative path")
    parts = Path(raw).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ExecutionContractError(f"{kind} contains traversal")
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ExecutionContractError(f"{kind} escapes the project root")
    return path


def _load_mapping(
    path: Path, *, kind: str, maximum_bytes: int = _MAX_CONFIG_BYTES
) -> dict[str, Any]:
    raw = _read_regular(path, maximum_bytes=maximum_bytes, kind=kind)
    try:
        loaded = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError, yaml.YAMLError) as error:
        raise ExecutionContractError(f"{kind} is malformed") from error
    if not isinstance(loaded, dict):
        raise ExecutionContractError(f"{kind} root must be a mapping")
    secret = _secret_like_key(loaded)
    if secret is not None:
        raise ExecutionContractError(f"{kind} contains secret-like key: {secret}")
    serialized = json.dumps(loaded, sort_keys=True).casefold()
    if any(marker in serialized for marker in ("praig/smb", "evaluation_benchmark", "hf_token")):
        raise ExecutionContractError(f"{kind} crosses the fixture-only SMB boundary")
    return loaded


def _config_project_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise ExecutionContractError("experiment config is not inside the project repository")


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Validate all frozen scientific inputs before returning an immutable config view."""

    path = Path(path)
    project_root = _config_project_root(path)
    payload = _load_mapping(path, kind="experiment config")
    try:
        validate_instance("experiment-config", payload, version=2)
    except ContractValidationError as error:
        raise ExecutionContractError(str(error)) from error
    identity = experiment_identity(payload, schema_version=2)

    fixture_path = _project_relative(
        project_root, payload["fixture"]["manifest_path"], kind="fixture manifest path"
    )
    fixture = _load_mapping(fixture_path, kind="fixture manifest")
    try:
        validate_instance("fixture-manifest", fixture, version=2)
    except ContractValidationError as error:
        raise ExecutionContractError(str(error)) from error
    if (
        fixture.get("manifest_id") != payload["fixture"]["manifest_id"]
        or canonical_sha256(fixture) != payload["fixture"]["manifest_sha256"]
        or len(fixture.get("items", ())) != payload["fixture"]["item_count"]
    ):
        raise ExecutionContractError("fixture manifest identity differs from the experiment")

    degradation = _load_mapping(
        _project_relative(
            project_root, payload["controls"]["degradation_path"], kind="degradation path"
        ),
        kind="frozen degradation control",
    )
    evaluation = _load_mapping(
        _project_relative(
            project_root, payload["controls"]["evaluation_path"], kind="evaluation path"
        ),
        kind="frozen evaluation control",
    )
    try:
        validate_instance("degradation-control", degradation, version=2)
    except ContractValidationError as error:
        raise ExecutionContractError(str(error)) from error
    if (
        degradation.get("status") != "frozen"
        or degradation.get("control_id") != "controlled-score-v1"
    ):
        raise ExecutionContractError("degradation control is not the accepted frozen control")
    if canonical_sha256(degradation) != payload["controls"]["degradation_sha256"]:
        raise ExecutionContractError("degradation control digest differs")
    if canonical_sha256(evaluation) != payload["controls"]["evaluation_sha256"]:
        raise ExecutionContractError("evaluation control digest differs")

    membership = _load_mapping(
        _project_relative(
            project_root,
            payload["qualitative_core"]["membership_path"],
            kind="qualitative core path",
        ),
        kind="qualitative core membership",
    )
    try:
        validate_instance("qualitative-sample", membership, version=2)
    except ContractValidationError as error:
        raise ExecutionContractError(str(error)) from error
    expected_core = payload["qualitative_core"]
    if (
        membership.get("core_membership_id") != expected_core["core_membership_id"]
        or membership.get("core_sha256") != expected_core["core_sha256"]
        or canonical_sha256(membership) != expected_core["membership_sha256"]
    ):
        raise ExecutionContractError("qualitative core identity differs")

    methods = tuple(payload["methods"])
    conditions = tuple(payload["conditions"])
    if methods != EXPECTED_METHODS or conditions != EXPECTED_CONDITIONS:
        raise ExecutionContractError("experiment method or condition order differs")
    items = tuple(dict(item) for item in fixture["items"])
    expected_count = len(items) * len(methods) * len(conditions)
    limits = payload["limits"]
    if len(items) > limits["max_items"] or expected_count > limits["max_expected_tuples"]:
        raise ExecutionContractError("experiment tuple denominator exceeds its bound")
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ExecutionContractError("fixture item IDs are not unique")
    return ExperimentConfig(
        payload=payload,
        project_root=project_root,
        experiment_id=identity["experiment_id"],
        experiment_sha256=identity["sha256"],
        method_ids=methods,
        condition_ids=conditions,
        fixture_items=items,
    )


def _ensure_root(root: Path) -> Path:
    root = Path(root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ExecutionContractError("artifact root must be a non-symlink directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = root.resolve()
    if root.is_symlink():
        raise ExecutionContractError("artifact root must not be a symlink")
    return resolved


@contextmanager
def artifact_writer_lock(artifact_root: Path) -> Iterator[_WriterAuthority]:
    """Acquire the one retained non-blocking writer authority for an artifact root."""

    root = _ensure_root(artifact_root)
    path = root / _LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExecutionContractError("writer lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExecutionBusyError("artifact root is busy") from error
        yield _WriterAuthority(root=root, descriptor=descriptor)
    finally:
        if "descriptor" in locals():
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _connect(root: Path, *, writable: bool) -> sqlite3.Connection:
    ledger = root / _LEDGER_NAME
    if not writable and not ledger.is_file():
        raise ExecutionContractError("run ledger does not exist")
    if ledger.is_symlink():
        raise ExecutionContractError("run ledger must not be a symlink")
    uri = f"file:{ledger}?mode={'rwc' if writable else 'ro'}"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=0.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    if writable:
        journal = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        connection.execute("PRAGMA synchronous=FULL")
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        if str(journal).casefold() != "delete" or int(synchronous) != 2:
            connection.close()
            raise ExecutionContractError("SQLite durability pragmas could not be enforced")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_identity (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          experiment_id TEXT NOT NULL,
          experiment_sha256 TEXT NOT NULL,
          config_sha256 TEXT NOT NULL,
          expected_tuple_count INTEGER NOT NULL CHECK (expected_tuple_count > 0)
        );
        CREATE TABLE IF NOT EXISTS expected_tuples (
          tuple_id TEXT PRIMARY KEY,
          item_id TEXT NOT NULL,
          source_group_id TEXT NOT NULL,
          condition_id TEXT NOT NULL,
          method_id TEXT NOT NULL,
          state TEXT NOT NULL CHECK (state IN (
            'expected','running','retry_pending','succeeded','failed','excluded'
          )),
          attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
          current_attempt_id TEXT,
          scientific_relative_path TEXT,
          output_relative_path TEXT,
          scientific_sha256 TEXT,
          output_encoded_sha256 TEXT,
          output_pixel_sha256 TEXT,
          exclusion_reason TEXT,
          UNIQUE (item_id, condition_id, method_id)
        );
        CREATE TABLE IF NOT EXISTS attempts (
          attempt_id TEXT PRIMARY KEY,
          tuple_id TEXT NOT NULL REFERENCES expected_tuples(tuple_id),
          ordinal INTEGER NOT NULL CHECK (ordinal > 0),
          outcome TEXT NOT NULL CHECK (outcome IN (
            'running','succeeded','retry_pending','failed','excluded','interrupted'
          )),
          failure_code TEXT,
          scientific_sha256 TEXT,
          output_encoded_sha256 TEXT,
          UNIQUE (tuple_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS integrity_incidents (
          incident_id TEXT PRIMARY KEY,
          tuple_id TEXT NOT NULL REFERENCES expected_tuples(tuple_id),
          attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
          payload_json TEXT NOT NULL,
          UNIQUE (tuple_id, attempt_id, payload_json)
        );
        """
    )


def _expected_tuples(config: ExperimentConfig) -> tuple[ExpectedTuple, ...]:
    return tuple(
        ExpectedTuple(
            item_id=item["item_id"],
            source_group_id=item["source_group_id"],
            condition_id=condition,
            method_id=method,
        )
        for item in config.fixture_items
        for condition in config.condition_ids
        for method in config.method_ids
    )


def _initialize_locked(config: ExperimentConfig, authority: _WriterAuthority) -> RunIdentity:
    connection = _connect(authority.root, writable=True)
    try:
        _create_schema(connection)
        expected = _expected_tuples(config)
        if len({item.tuple_id for item in expected}) != len(expected):
            raise ExecutionContractError("expected tuple identities are not unique")
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute("SELECT * FROM run_identity WHERE singleton=1").fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO run_identity VALUES (1, ?, ?, ?, ?)",
                (
                    config.experiment_id,
                    config.experiment_sha256,
                    canonical_sha256(config.payload),
                    len(expected),
                ),
            )
            connection.executemany(
                """INSERT INTO expected_tuples
                   (tuple_id,item_id,source_group_id,condition_id,method_id,state)
                   VALUES (?,?,?,?,?,'expected')""",
                [
                    (
                        item.tuple_id,
                        item.item_id,
                        item.source_group_id,
                        item.condition_id,
                        item.method_id,
                    )
                    for item in expected
                ],
            )
        elif (
            existing["experiment_id"] != config.experiment_id
            or existing["experiment_sha256"] != config.experiment_sha256
            or existing["expected_tuple_count"] != len(expected)
        ):
            raise ExecutionContractError("artifact root belongs to a different experiment")
        rows = connection.execute(
            "SELECT tuple_id,item_id,source_group_id,condition_id,method_id FROM expected_tuples"
        ).fetchall()
        observed = {
            (
                row["tuple_id"],
                row["item_id"],
                row["source_group_id"],
                row["condition_id"],
                row["method_id"],
            )
            for row in rows
        }
        required = {
            (item.tuple_id, item.item_id, item.source_group_id, item.condition_id, item.method_id)
            for item in expected
        }
        if observed != required:
            raise ExecutionContractError("ledger expected tuple denominator differs")
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()
    return RunIdentity(
        experiment_id=config.experiment_id,
        experiment_sha256=config.experiment_sha256,
        expected_tuple_count=len(expected),
        ledger_relative_path=_LEDGER_NAME,
    )


def initialize_run(config_path: Path, artifact_root: Path) -> RunIdentity:
    config = load_experiment_config(config_path)
    with artifact_writer_lock(artifact_root) as authority:
        return _initialize_locked(config, authority)


def snapshot_run(artifact_root: Path) -> RunSnapshot:
    """Read only a pure committed SQLite snapshot without writer authority."""

    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ExecutionContractError("artifact root is unavailable")
    connection = _connect(root.resolve(), writable=False)
    try:
        identity = connection.execute("SELECT * FROM run_identity WHERE singleton=1").fetchone()
        if identity is None:
            raise ExecutionContractError("run identity is absent")
        counts = Counter(
            {
                row["state"]: row["count"]
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM expected_tuples GROUP BY state"
                )
            }
        )
        unknown = set(counts) - ALL_STATES
        if unknown:
            raise ExecutionContractError("ledger contains an unknown tuple state")
        attempts = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        incidents = connection.execute("SELECT COUNT(*) FROM integrity_incidents").fetchone()[0]
        total = sum(counts.values())
        if total != identity["expected_tuple_count"]:
            raise ExecutionContractError("ledger tuple count differs from run identity")
        return RunSnapshot(
            experiment_id=identity["experiment_id"],
            total=total,
            counts=dict(sorted(counts.items())),
            attempt_count=int(attempts),
            integrity_incident_count=int(incidents),
        )
    finally:
        connection.close()


def _reconcile_locked(
    config: ExperimentConfig, authority: _WriterAuthority
) -> ReconciliationReport:
    snapshot = snapshot_run(authority.root)
    nonterminal = {
        state: count for state, count in snapshot.counts.items() if state not in TERMINAL_STATES
    }
    if nonterminal:
        raise ReconciliationError(f"run contains nonterminal tuples: {nonterminal}")
    raise ReconciliationError("reconciliation publication is implemented by Task 3")


def reconcile_run(config_path: Path, artifact_root: Path) -> ReconciliationReport:
    config = load_experiment_config(config_path)
    with artifact_writer_lock(artifact_root) as authority:
        _initialize_locked(config, authority)
        return _reconcile_locked(config, authority)


def execute_run(*_args: Any, **_kwargs: Any) -> ExecutionReport:
    raise NotImplementedError("tuple execution is implemented by Task 2")


def resume_run(*args: Any, **kwargs: Any) -> ExecutionReport:
    return execute_run(*args, **kwargs)


def export_reconciled_run(*_args: Any, **_kwargs: Any) -> ExportBundle:
    raise NotImplementedError("portable export is implemented by Task 3")
