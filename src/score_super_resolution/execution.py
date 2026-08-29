"""Recoverable fixture-only execution with a closed SQLite tuple ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import sqlite3
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import yaml

from score_super_resolution.baselines import pixel_sha256, run_baseline
from score_super_resolution.contracts import ContractValidationError, validate_instance
from score_super_resolution.degradation import (
    DegradationControl,
    DegradationResult,
    align_reference,
    apply_degradation,
    generate_fixture_bundle,
)
from score_super_resolution.environment import environment_snapshot
from score_super_resolution.evaluation import (
    AggregateControl,
    FidelityControl,
    QualitativeControl,
    aggregate_paired,
    compute_fidelity,
    load_evaluation_control,
    select_qualitative_panels,
)
from score_super_resolution.identities import canonical_sha256, experiment_identity
from score_super_resolution.resources import measure_baseline_resources

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


class ExecutionInterruptedError(RuntimeError):
    """A deliberate acceptance failpoint stopped execution after a durable boundary."""


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


def _contains_absolute_string(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_string(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_string(child) for child in value)
    if isinstance(value, str):
        return value.startswith(("/", "\\")) or (
            len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}
        )
    return False


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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _boundary(
    name: str,
    tuple_id: str,
    hook: Callable[[str, str], None] | None,
) -> None:
    if hook is not None:
        hook(name, tuple_id)
    failpoint = os.environ.get("SCORE_SR_PHASE2_FAILPOINT")
    if failpoint == f"{name}:raise":
        raise OSError(f"injected execution failure at {name}")
    if failpoint == f"{name}:exit":
        os._exit(91)


def _durable_replace(
    path: Path,
    payload: bytes,
    *,
    tuple_id: str,
    prefix: str,
    boundary_hook: Callable[[str, str], None] | None,
) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ExecutionContractError("publication target must not be a symlink")
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    replaced = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short publication write")
            view = view[written:]
        _boundary(f"before_{prefix}_fsync", tuple_id, boundary_hook)
        os.fsync(descriptor)
        _boundary(f"after_{prefix}_fsync", tuple_id, boundary_hook)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        replaced = True
        _boundary(f"after_{prefix}_replace", tuple_id, boundary_hook)
        _fsync_directory(path.parent)
        _boundary(f"after_{prefix}_parent_fsync", tuple_id, boundary_hook)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not replaced:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _degradation_control(config: ExperimentConfig) -> DegradationControl:
    payload = _load_mapping(
        _project_relative(
            config.project_root,
            config.payload["controls"]["degradation_path"],
            kind="degradation path",
        ),
        kind="frozen degradation control",
    )
    return DegradationControl(
        version=int(payload["candidate_version"]),
        candidate_id=str(payload["control_id"]),
        status=str(payload["status"]),
        claim_boundary=str(payload["claim_boundary"]),
        master_seed=int(payload["master_seed"]),
        image_contract=dict(payload["image_contract"]),
        alignment=dict(payload["alignment"]),
        runtime=dict(payload["runtime"]),
        condition_ids=tuple(payload["condition_order"]),
        conditions=tuple(dict(condition) for condition in payload["conditions"]),
        sha256=canonical_sha256(payload),
    )


def _resource_environment() -> dict[str, Any]:
    build = cv2.getBuildInformation()
    cpu_model = platform.processor() or "unknown-cpu"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.casefold().startswith("model name"):
                cpu_model = line.split(":", maxsplit=1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count() or 1,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(build.encode("utf-8")).hexdigest(),
    }


def _write_attempt_environment(config: ExperimentConfig, authority: _WriterAuthority) -> None:
    destination = authority.root / "attempts" / "local-attempt-environment.json"
    snapshot = environment_snapshot(
        config.project_root,
        workspace_root=config.project_root.parent,
        memoria_repository=config.project_root.parent / "memoria",
    )
    payload = {
        "record_type": "local-attempt-environment",
        "experiment_id": config.experiment_id,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "artifact_root": str(authority.root),
        "python_executable": sys.executable,
        "environment": snapshot,
    }
    if _secret_like_key(payload) is not None:
        raise ExecutionContractError("local attempt environment contains a secret-like key")
    _durable_replace(
        destination,
        _canonical_json(payload),
        tuple_id="run-environment",
        prefix="attempt_environment",
        boundary_hook=None,
    )


def _prepare_fixture_inputs(
    config: ExperimentConfig, authority: _WriterAuthority
) -> dict[str, Path]:
    manifest_path = _project_relative(
        config.project_root,
        config.payload["fixture"]["manifest_path"],
        kind="fixture manifest path",
    )
    bundle = generate_fixture_bundle(manifest_path, authority.root / "fixture-input")
    if (
        bundle["manifest_id"] != config.payload["fixture"]["manifest_id"]
        or bundle["manifest_sha256"] != config.payload["fixture"]["manifest_sha256"]
    ):
        raise ExecutionContractError("materialized fixture identity differs")
    return {
        item["item_id"]: authority.root / "fixture-input" / item["relative_path"]
        for item in bundle["items"]
    }


def _load_rgb(path: Path, *, maximum_bytes: int, maximum_pixels: int) -> np.ndarray:
    raw = _read_regular(path, maximum_bytes=maximum_bytes, kind="fixture image")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ExecutionContractError("fixture image cannot be decoded")
    if decoded.shape[0] * decoded.shape[1] > maximum_pixels:
        raise ExecutionContractError("fixture image exceeds the pixel bound")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _claim_next(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
) -> tuple[sqlite3.Row, str] | None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """SELECT * FROM expected_tuples
               WHERE state IN ('retry_pending','expected')
               ORDER BY CASE state WHEN 'retry_pending' THEN 0 ELSE 1 END,
                        condition_id,item_id,method_id
               LIMIT 1"""
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        ordinal = int(row["attempt_count"]) + 1
        if ordinal > int(config.payload["limits"]["max_attempts_per_tuple"]):
            connection.execute(
                "UPDATE expected_tuples SET state='failed' WHERE tuple_id=?",
                (row["tuple_id"],),
            )
            connection.commit()
            return _claim_next(connection, config)
        attempt_id = (
            f"attempt-{canonical_sha256({'tuple_id': row['tuple_id'], 'ordinal': ordinal})}"
        )
        connection.execute(
            "INSERT INTO attempts (attempt_id,tuple_id,ordinal,outcome) VALUES (?,?,?,'running')",
            (attempt_id, row["tuple_id"], ordinal),
        )
        changed = connection.execute(
            """UPDATE expected_tuples
               SET state='running',attempt_count=?,current_attempt_id=?
               WHERE tuple_id=? AND state IN ('retry_pending','expected')""",
            (ordinal, attempt_id, row["tuple_id"]),
        ).rowcount
        if changed != 1:
            raise ExecutionContractError("tuple claim lost its transactional precondition")
        connection.commit()
        claimed = connection.execute(
            "SELECT * FROM expected_tuples WHERE tuple_id=?", (row["tuple_id"],)
        ).fetchone()
        return claimed, attempt_id
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise


def _recover_running(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        running = connection.execute(
            "SELECT tuple_id,current_attempt_id FROM expected_tuples WHERE state='running'"
        ).fetchall()
        for row in running:
            connection.execute(
                """UPDATE attempts SET outcome='interrupted',failure_code='INTERRUPTED'
                   WHERE attempt_id=? AND outcome='running'""",
                (row["current_attempt_id"],),
            )
            connection.execute(
                "UPDATE expected_tuples SET state='retry_pending' WHERE tuple_id=?",
                (row["tuple_id"],),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _relative_result_paths(row: Mapping[str, Any]) -> tuple[str, str]:
    stem = f"{row['condition_id']}/{row['method_id']}/{row['item_id']}"
    return f"outputs/{stem}.png", f"scientific/{stem}.json"


def _compute_tuple(
    config: ExperimentConfig,
    row: Mapping[str, Any],
    fixture_paths: Mapping[str, Path],
    degradation_control: DegradationControl,
) -> tuple[bytes, dict[str, Any]]:
    limits = config.payload["limits"]
    reference = _load_rgb(
        fixture_paths[str(row["item_id"])],
        maximum_bytes=int(limits["max_input_bytes"]),
        maximum_pixels=int(limits["max_decoded_pixels"]),
    )
    degraded = apply_degradation(
        reference,
        control=degradation_control,
        condition_id=str(row["condition_id"]),
        item_id=str(row["item_id"]),
        source_group_id=str(row["source_group_id"]),
        fixture_manifest_id=config.payload["fixture"]["manifest_id"],
        purpose="benchmark",
    )
    aligned_dimensions = degraded.trace["aligned_dimensions"]
    target_shape = (
        int(aligned_dimensions["height"]),
        int(aligned_dimensions["width"]),
        int(aligned_dimensions["channels"]),
    )
    baseline = run_baseline(
        str(row["method_id"]),
        degraded.pixels,
        target_shape=target_shape,
        condition_id=str(row["condition_id"]),
    )
    output_relative, _ = _relative_result_paths(row)
    success, encoded = cv2.imencode(
        ".png", cv2.cvtColor(baseline.pixels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not success:
        raise ExecutionContractError("baseline output PNG encoding failed")
    output_bytes = encoded.tobytes()
    if len(output_bytes) > int(limits["max_output_bytes"]):
        raise ExecutionContractError("baseline output exceeds the byte bound")
    evaluation = load_evaluation_control(
        _project_relative(
            config.project_root,
            config.payload["controls"]["evaluation_path"],
            kind="evaluation path",
        )
    )
    fidelity = FidelityControl(
        evaluation=evaluation,
        experiment_id=config.experiment_id,
        item_id=str(row["item_id"]),
        source_group_id=str(row["source_group_id"]),
        condition_id=str(row["condition_id"]),
        method_id=str(row["method_id"]),
        reconstruction_id=str(row["tuple_id"]),
        reference_id=f"reference-{row['item_id']}-{row['condition_id']}",
    )
    scale = int(str(row["condition_id"])[1])
    aligned_reference = align_reference(reference, scale).pixels
    metrics = list(
        compute_fidelity(aligned_reference, baseline.pixels, scale=scale, control=fidelity)
    )
    resource_input = DegradationResult(
        pixels=degraded.pixels,
        encoded_bytes=degraded.encoded_bytes,
        trace={**degraded.trace, "output_pixel_sha256": pixel_sha256(degraded.pixels)},
    )
    resource = measure_baseline_resources(
        str(row["method_id"]),
        resource_input,
        control=evaluation.payload,
        environment=_resource_environment(),
    )
    core: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "scientific-result",
        "experiment_id": config.experiment_id,
        "tuple_id": str(row["tuple_id"]),
        "item_id": str(row["item_id"]),
        "source_group_id": str(row["source_group_id"]),
        "condition_id": str(row["condition_id"]),
        "method_id": str(row["method_id"]),
        "output_relative_path": output_relative,
        "output_encoded_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "output_pixel_sha256": pixel_sha256(baseline.pixels),
        "degradation_trace": degraded.trace,
        "baseline_evidence": baseline.evidence,
        "metrics": metrics,
        "resource": resource,
    }
    digest = canonical_sha256(core)
    scientific = {
        **core,
        "scientific_result_id": f"scientific-{digest}",
        "scientific_sha256": digest,
    }
    if _contains_absolute_string(scientific):
        raise ExecutionContractError("scientific result contains an absolute local path")
    validate_instance("scientific-result", scientific, version=2)
    return output_bytes, scientific


def _mark_attempt_failure(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    tuple_id: str,
    attempt_id: str,
    *,
    code: str,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT attempt_count,state FROM expected_tuples WHERE tuple_id=?", (tuple_id,)
        ).fetchone()
        if row is None or row["state"] != "running":
            raise ExecutionContractError("failed attempt no longer owns its running tuple")
        retryable = code in set(config.payload["retry_policy"]["retryable_codes"])
        retry = retryable and int(row["attempt_count"]) < int(
            config.payload["limits"]["max_attempts_per_tuple"]
        )
        state = "retry_pending" if retry else "failed"
        connection.execute(
            "UPDATE attempts SET outcome=?,failure_code=? WHERE attempt_id=? AND outcome='running'",
            (state, code, attempt_id),
        )
        connection.execute(
            "UPDATE expected_tuples SET state=? WHERE tuple_id=?",
            (state, tuple_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _commit_success(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    attempt_id: str,
    scientific: Mapping[str, Any],
    scientific_relative: str,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        changed_attempt = connection.execute(
            """UPDATE attempts SET outcome='succeeded',scientific_sha256=?,output_encoded_sha256=?
               WHERE attempt_id=? AND outcome='running'""",
            (
                scientific["scientific_sha256"],
                scientific["output_encoded_sha256"],
                attempt_id,
            ),
        ).rowcount
        changed_tuple = connection.execute(
            """UPDATE expected_tuples
               SET state='succeeded',scientific_relative_path=?,output_relative_path=?,
                   scientific_sha256=?,output_encoded_sha256=?,output_pixel_sha256=?
               WHERE tuple_id=? AND state='running' AND current_attempt_id=?""",
            (
                scientific_relative,
                scientific["output_relative_path"],
                scientific["scientific_sha256"],
                scientific["output_encoded_sha256"],
                scientific["output_pixel_sha256"],
                row["tuple_id"],
                attempt_id,
            ),
        ).rowcount
        if changed_attempt != 1 or changed_tuple != 1:
            raise ExecutionContractError("success commit lost its claimed tuple")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _confined_runtime_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or relative.startswith("/") or "\\" in relative:
        raise ExecutionContractError("runtime evidence path must be relative")
    parts = Path(relative).parts
    if any(part in {"", ".", ".."} for part in parts) or len(parts) > 12:
        raise ExecutionContractError("runtime evidence path is noncanonical or too deep")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ExecutionContractError("runtime evidence path escapes the artifact root")
    return path


def _scientific_digest(record: Mapping[str, Any]) -> str:
    core = {
        key: value
        for key, value in record.items()
        if key not in {"scientific_result_id", "scientific_sha256"}
    }
    return canonical_sha256(core)


def _success_validation_error(
    root: Path,
    row: Mapping[str, Any],
    limits: Mapping[str, Any],
) -> tuple[str, str | None]:
    try:
        output_path = _confined_runtime_path(root, str(row["output_relative_path"]))
        scientific_path = _confined_runtime_path(root, str(row["scientific_relative_path"]))
    except ExecutionContractError:
        return "PATH_OR_FILE_TYPE_INVALID", None
    if not output_path.exists():
        return "MISSING_OUTPUT", None
    if output_path.is_symlink() or not output_path.is_file():
        return "PATH_OR_FILE_TYPE_INVALID", None
    try:
        output = _read_regular(
            output_path,
            maximum_bytes=int(limits["max_output_bytes"]),
            kind="committed tuple output",
        )
    except ExecutionContractError:
        return "PATH_OR_FILE_TYPE_INVALID", None
    observed = hashlib.sha256(output).hexdigest()
    if observed != row["output_encoded_sha256"]:
        return "ENCODED_DIGEST_MISMATCH", observed
    decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return "PIXEL_DIGEST_MISMATCH", observed
    rgb = np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    if pixel_sha256(rgb) != row["output_pixel_sha256"]:
        return "PIXEL_DIGEST_MISMATCH", observed
    if (
        not scientific_path.exists()
        or scientific_path.is_symlink()
        or not scientific_path.is_file()
    ):
        return "SCIENTIFIC_PAYLOAD_MISMATCH", observed
    try:
        raw = _read_regular(
            scientific_path,
            maximum_bytes=int(limits["max_output_bytes"]),
            kind="committed scientific payload",
        )
        scientific = json.loads(raw)
        if not isinstance(scientific, dict):
            raise ValueError
        validate_instance("scientific-result", scientific, version=2)
        if _contains_absolute_string(scientific):
            raise ValueError
    except (ExecutionContractError, ContractValidationError, json.JSONDecodeError, ValueError):
        return "SCIENTIFIC_PAYLOAD_MISMATCH", observed
    digest = _scientific_digest(scientific)
    if (
        scientific["tuple_id"] != row["tuple_id"]
        or scientific["scientific_sha256"] != digest
        or scientific["scientific_result_id"] != f"scientific-{digest}"
        or digest != row["scientific_sha256"]
        or scientific["output_encoded_sha256"] != observed
        or scientific["output_pixel_sha256"] != row["output_pixel_sha256"]
        or scientific["output_relative_path"] != row["output_relative_path"]
    ):
        return "SCIENTIFIC_PAYLOAD_MISMATCH", observed
    return "", observed


def _quarantine_invalid_success(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    authority: _WriterAuthority,
    row: Mapping[str, Any],
    *,
    reason_code: str,
    observed_sha256: str | None,
) -> None:
    paths: list[tuple[str, Path]] = []
    for label, field in (
        ("output", "output_relative_path"),
        ("scientific", "scientific_relative_path"),
    ):
        value = row[field]
        if isinstance(value, str):
            try:
                path = _confined_runtime_path(authority.root, value)
            except ExecutionContractError:
                continue
            if path.exists() or path.is_symlink():
                paths.append((label, path))
    address = canonical_sha256(
        {
            "tuple_id": row["tuple_id"],
            "attempt_id": row["current_attempt_id"],
            "reason_code": reason_code,
            "prior_scientific_sha256": row["scientific_sha256"],
            "observed_sha256": observed_sha256,
        }
    )
    quarantine_root = authority.root / "quarantine" / str(row["tuple_id"]) / address
    quarantine_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    moved: list[tuple[str, Path, str]] = []
    for label, source in paths:
        destination = quarantine_root / f"{label}{source.suffix}"
        if destination.exists():
            existing = _read_regular(
                destination,
                maximum_bytes=int(config.payload["limits"]["max_output_bytes"]),
                kind="quarantine evidence",
            )
            source_bytes = _read_regular(
                source,
                maximum_bytes=int(config.payload["limits"]["max_output_bytes"]),
                kind="invalidated evidence",
            )
            if existing != source_bytes:
                raise ExecutionContractError("quarantine address collision")
            source.unlink()
        else:
            os.replace(source, destination)
        _fsync_directory(quarantine_root)
        relative = destination.relative_to(authority.root).as_posix()
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        moved.append((relative, destination, digest))
    quarantine_relative = moved[0][0] if moved else None
    quarantine_sha = moved[0][2] if moved else None
    core = {
        "schema_version": 2,
        "record_type": "integrity-incident",
        "experiment_id": config.experiment_id,
        "tuple_id": str(row["tuple_id"]),
        "attempt_id": str(row["current_attempt_id"]),
        "reason_code": reason_code,
        "prior_scientific_sha256": str(row["scientific_sha256"]),
        "observed_sha256": observed_sha256,
        "quarantine_relative_path": quarantine_relative,
        "quarantine_sha256": quarantine_sha,
    }
    incident = {**core, "incident_id": f"incident-{canonical_sha256(core)}"}
    validate_instance("integrity-incident", incident, version=2)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "INSERT OR IGNORE INTO integrity_incidents VALUES (?,?,?,?)",
            (
                incident["incident_id"],
                row["tuple_id"],
                row["current_attempt_id"],
                json.dumps(incident, separators=(",", ":"), sort_keys=True),
            ),
        )
        retry = int(row["attempt_count"]) < int(config.payload["limits"]["max_attempts_per_tuple"])
        state = "retry_pending" if retry else "failed"
        connection.execute(
            "UPDATE expected_tuples SET state=? WHERE tuple_id=? AND state='succeeded'",
            (state, row["tuple_id"]),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _validate_and_repair_successes(
    connection: sqlite3.Connection,
    config: ExperimentConfig,
    authority: _WriterAuthority,
) -> None:
    rows = connection.execute(
        "SELECT * FROM expected_tuples WHERE state='succeeded' ORDER BY tuple_id"
    ).fetchall()
    for row in rows:
        reason, observed = _success_validation_error(authority.root, row, config.payload["limits"])
        if reason:
            _quarantine_invalid_success(
                connection,
                config,
                authority,
                row,
                reason_code=reason,
                observed_sha256=observed,
            )


def _execute_locked(
    config: ExperimentConfig,
    authority: _WriterAuthority,
    *,
    max_tuples: int | None,
    boundary_hook: Callable[[str, str], None] | None,
) -> ExecutionReport:
    if max_tuples is not None and (
        isinstance(max_tuples, bool) or not isinstance(max_tuples, int) or max_tuples < 0
    ):
        raise ExecutionContractError("max_tuples must be a nonnegative integer or null")
    _initialize_locked(config, authority)
    _write_attempt_environment(config, authority)
    fixture_paths = _prepare_fixture_inputs(config, authority)
    degradation_control = _degradation_control(config)
    connection = _connect(authority.root, writable=True)
    claimed = succeeded = failed = skipped = 0
    try:
        _recover_running(connection)
        before_success = snapshot_run(authority.root).counts.get("succeeded", 0)
        _validate_and_repair_successes(connection, config, authority)
        after_success = snapshot_run(authority.root).counts.get("succeeded", 0)
        skipped = min(before_success, after_success)
        while max_tuples is None or claimed < max_tuples:
            claim = _claim_next(connection, config)
            if claim is None:
                break
            row, attempt_id = claim
            claimed += 1
            try:
                output_bytes, scientific = _compute_tuple(
                    config, row, fixture_paths, degradation_control
                )
                output_relative, scientific_relative = _relative_result_paths(row)
                _durable_replace(
                    authority.root / output_relative,
                    output_bytes,
                    tuple_id=str(row["tuple_id"]),
                    prefix="output",
                    boundary_hook=boundary_hook,
                )
                _boundary(
                    "after_output_replace_before_ledger",
                    str(row["tuple_id"]),
                    boundary_hook,
                )
                _durable_replace(
                    authority.root / scientific_relative,
                    _canonical_json(scientific),
                    tuple_id=str(row["tuple_id"]),
                    prefix="scientific",
                    boundary_hook=boundary_hook,
                )
                _commit_success(connection, row, attempt_id, scientific, scientific_relative)
                succeeded += 1
                _boundary("after_tuple_commit", str(row["tuple_id"]), boundary_hook)
            except ExecutionInterruptedError:
                raise
            except BaseException as error:
                # Keyboard/SystemExit remain interruptions; ordinary tuple failures are durable.
                if not isinstance(error, Exception):
                    raise
                _mark_attempt_failure(
                    connection,
                    config,
                    str(row["tuple_id"]),
                    attempt_id,
                    code="TRANSIENT_COMPUTATION",
                )
                failed += 1
    finally:
        connection.close()
    return ExecutionReport(
        experiment_id=config.experiment_id,
        claimed=claimed,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        interrupted=False,
    )


def _reconcile_locked(
    config: ExperimentConfig, authority: _WriterAuthority
) -> ReconciliationReport:
    connection = _connect(authority.root, writable=True)
    try:
        _recover_running(connection)
        _validate_and_repair_successes(connection, config, authority)
        rows = connection.execute("SELECT * FROM expected_tuples ORDER BY tuple_id").fetchall()
        attempts = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        incidents = int(
            connection.execute("SELECT COUNT(*) FROM integrity_incidents").fetchone()[0]
        )
    finally:
        connection.close()
    counts = Counter(str(row["state"]) for row in rows)
    nonterminal = {state: count for state, count in counts.items() if state not in TERMINAL_STATES}
    if nonterminal:
        raise ReconciliationError(f"run contains nonterminal tuples: {nonterminal}")
    if len(rows) != len(_expected_tuples(config)):
        raise ReconciliationError("run tuple denominator differs from the experiment")
    states = _tuple_states(rows, include_failure_stage=False)
    records = _scientific_records(authority.root, rows, config)
    identity_core = {
        "experiment_id": config.experiment_id,
        "experiment_sha256": config.experiment_sha256,
        "tuple_state_sha256": canonical_sha256(states),
        "scientific_records_sha256": canonical_sha256(
            [_scientific_projection(record) for record in records]
        ),
        "attempt_count": attempts,
        "integrity_incident_count": incidents,
    }
    reconciliation_id = f"reconciliation-{canonical_sha256(identity_core)}"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "reconciliation-report",
        "reconciliation_id": reconciliation_id,
        "experiment_id": config.experiment_id,
        "experiment_sha256": config.experiment_sha256,
        "degradation_control_sha256": config.payload["controls"]["degradation_sha256"],
        "evaluation_control_sha256": config.payload["controls"]["evaluation_sha256"],
        "fixture_manifest_sha256": config.payload["fixture"]["manifest_sha256"],
        "core_membership_id": config.payload["qualitative_core"]["core_membership_id"],
        "core_sha256": config.payload["qualitative_core"]["core_sha256"],
        "expected_tuple_count": len(rows),
        "terminal_tuple_count": len(rows),
        "counts": {
            "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0),
            "excluded": counts.get("excluded", 0),
        },
        "attempt_count": attempts,
        "integrity_incident_count": incidents,
        "tuple_state_sha256": identity_core["tuple_state_sha256"],
        "scientific_records_sha256": identity_core["scientific_records_sha256"],
    }
    payload["report_sha256"] = canonical_sha256(payload)
    validate_instance("reconciliation-report", payload, version=2)
    path = authority.root / "reconciliation-report.json"
    _durable_replace(
        path,
        _canonical_json(payload),
        tuple_id="reconciliation",
        prefix="reconciliation",
        boundary_hook=None,
    )
    return ReconciliationReport(payload=payload, path=path)


def reconcile_run(config_path: Path, artifact_root: Path) -> ReconciliationReport:
    config = load_experiment_config(config_path)
    with artifact_writer_lock(artifact_root) as authority:
        _initialize_locked(config, authority)
        return _reconcile_locked(config, authority)


def execute_run(
    config_path: Path,
    artifact_root: Path,
    *,
    max_tuples: int | None = None,
    boundary_hook: Callable[[str, str], None] | None = None,
) -> ExecutionReport:
    config = load_experiment_config(config_path)
    with artifact_writer_lock(artifact_root) as authority:
        return _execute_locked(
            config,
            authority,
            max_tuples=max_tuples,
            boundary_hook=boundary_hook,
        )


def resume_run(
    config_path: Path,
    artifact_root: Path,
    *,
    max_tuples: int | None = None,
    boundary_hook: Callable[[str, str], None] | None = None,
) -> ExecutionReport:
    return execute_run(
        config_path,
        artifact_root,
        max_tuples=max_tuples,
        boundary_hook=boundary_hook,
    )


def _tuple_states(rows: list[sqlite3.Row], *, include_failure_stage: bool) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row in rows:
        state = "success" if row["state"] == "succeeded" else str(row["state"])
        item: dict[str, Any] = {
            "item_id": str(row["item_id"]),
            "source_group_id": str(row["source_group_id"]),
            "condition_id": str(row["condition_id"]),
            "method_id": str(row["method_id"]),
            "state": state,
            "attempt_count": int(row["attempt_count"]),
            "exclusion_reason": row["exclusion_reason"],
        }
        if include_failure_stage and state == "failed":
            item["failure_stage"] = "execution"
        states.append(item)
    return sorted(
        states,
        key=lambda row: (
            row["condition_id"],
            row["source_group_id"],
            row["item_id"],
            row["method_id"],
        ),
    )


def _scientific_records(
    root: Path, rows: list[sqlite3.Row], config: ExperimentConfig
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row["state"] != "succeeded":
            continue
        reason, _observed = _success_validation_error(root, row, config.payload["limits"])
        if reason:
            raise ReconciliationError("a reconciled success no longer validates")
        path = _confined_runtime_path(root, str(row["scientific_relative_path"]))
        record = json.loads(
            _read_regular(
                path,
                maximum_bytes=int(config.payload["limits"]["max_output_bytes"]),
                kind="scientific record",
            )
        )
        if not isinstance(record, dict):
            raise ReconciliationError("scientific record root is invalid")
        records.append(record)
    return sorted(records, key=lambda record: str(record["tuple_id"]))


def _scientific_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return only deterministic scientific fields for reconciliation and replay."""

    return {
        key: value
        for key, value in record.items()
        if key not in {"scientific_result_id", "scientific_sha256", "resource"}
    }


def _read_ledger_rows(root: Path) -> list[sqlite3.Row]:
    connection = _connect(root, writable=False)
    try:
        return connection.execute("SELECT * FROM expected_tuples ORDER BY tuple_id").fetchall()
    finally:
        connection.close()


def _portable_environment(config: ExperimentConfig, authority: _WriterAuthority) -> dict[str, Any]:
    raw_path = authority.root / "attempts/local-attempt-environment.json"
    raw = json.loads(
        _read_regular(
            raw_path,
            maximum_bytes=int(config.payload["limits"]["max_output_bytes"]),
            kind="local attempt environment",
        )
    )
    environment = raw["environment"]
    runtime = _resource_environment()
    repositories = []
    for role in ("workspace-planning", "proyecto", "memoria"):
        repository = environment["repositories"][role]
        repositories.append(
            {
                "role": role,
                "revision_state": repository["revision_state"],
                "revision": repository["revision"],
                "dirty": repository["dirty"],
            }
        )
    return {
        "runtime": {
            "python_version": environment["python"]["version"],
            "implementation": environment["python"]["implementation"],
            "platform": environment["platform"]["system"],
            "machine": environment["platform"]["machine"],
        },
        "dependencies": dict(sorted(environment["packages"].items())),
        "repositories": repositories,
        "opencv": {
            "version": runtime["opencv_version"],
            "build_sha256": runtime["opencv_build_sha256"],
        },
        "hardware": {
            "cpu_model": runtime["cpu_model"],
            "logical_cpu_count": runtime["logical_cpu_count"],
        },
        "applicability": {"gpu": "not_used", "learned_model": "not_applicable"},
    }


def _evidence_file(path: Path, root: Path, *, record_count: int) -> dict[str, Any]:
    raw = _read_regular(path, maximum_bytes=64 * 1024 * 1024, kind="portable evidence")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": record_count,
    }


def _export_locked(
    config: ExperimentConfig, authority: _WriterAuthority, reconciliation: ReconciliationReport
) -> ExportBundle:
    rows = _read_ledger_rows(authority.root)
    records = _scientific_records(authority.root, rows, config)
    metrics = sorted(
        (dict(metric) for record in records for metric in record["metrics"]),
        key=lambda metric: str(metric["metric_result_id"]),
    )
    aggregate_states = _tuple_states(rows, include_failure_stage=False)
    evaluation = load_evaluation_control(
        _project_relative(
            config.project_root,
            config.payload["controls"]["evaluation_path"],
            kind="evaluation path",
        )
    )
    aggregate = aggregate_paired(
        metrics,
        aggregate_states,
        control=AggregateControl(
            evaluation=evaluation,
            experiment_id=config.experiment_id,
            reconciliation_id=reconciliation.payload["reconciliation_id"],
            reconciliation_sha256=reconciliation.payload["report_sha256"],
            raw_metric_input_sha256=canonical_sha256(metrics),
            tuple_state_input_sha256=canonical_sha256(aggregate_states),
        ),
    )
    aggregate_path = authority.root / "evidence/aggregate-six-cell.json"
    _durable_replace(
        aggregate_path,
        _canonical_json(aggregate),
        tuple_id="aggregate",
        prefix="aggregate",
        boundary_hook=None,
    )

    fixture = _load_mapping(
        _project_relative(
            config.project_root, config.payload["fixture"]["manifest_path"], kind="fixture path"
        ),
        kind="fixture manifest",
    )
    core = _load_mapping(
        _project_relative(
            config.project_root,
            config.payload["qualitative_core"]["membership_path"],
            kind="qualitative core path",
        ),
        kind="qualitative core membership",
    )
    membership = select_qualitative_panels(
        metrics,
        _tuple_states(rows, include_failure_stage=True),
        fixture,
        control=QualitativeControl(
            evaluation=evaluation,
            experiment_id=config.experiment_id,
            core_membership=core,
        ),
    )
    membership_path = authority.root / "evidence/qualitative-membership.json"
    _durable_replace(
        membership_path,
        _canonical_json(membership),
        tuple_id="qualitative-membership",
        prefix="qualitative_membership",
        boundary_hook=None,
    )

    jsonl_path = authority.root / "export/scientific-records.jsonl"
    jsonl = b"".join(_canonical_json(record) for record in records)
    _durable_replace(
        jsonl_path,
        jsonl,
        tuple_id="scientific-jsonl",
        prefix="scientific_jsonl",
        boundary_hook=None,
    )
    reconciliation_path = authority.root / "reconciliation-report.json"
    manifest_core: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "portable-export",
        "experiment_id": config.experiment_id,
        "experiment_sha256": config.experiment_sha256,
        "reconciliation_id": reconciliation.payload["reconciliation_id"],
        "reconciliation_sha256": reconciliation.payload["report_sha256"],
        "dataset_role": config.payload["dataset_role"],
        "controls": {
            "degradation_sha256": config.payload["controls"]["degradation_sha256"],
            "evaluation_sha256": config.payload["controls"]["evaluation_sha256"],
            "fixture_manifest_sha256": config.payload["fixture"]["manifest_sha256"],
            "core_membership_id": config.payload["qualitative_core"]["core_membership_id"],
            "core_sha256": config.payload["qualitative_core"]["core_sha256"],
        },
        "scientific_records": _evidence_file(jsonl_path, authority.root, record_count=len(records)),
        "evidence": [
            _evidence_file(reconciliation_path, authority.root, record_count=len(rows)),
            _evidence_file(aggregate_path, authority.root, record_count=len(aggregate["cells"])),
            _evidence_file(
                membership_path,
                authority.root,
                record_count=len(membership["core_panels"]) + len(membership["additional_panels"]),
            ),
        ],
        "environment": _portable_environment(config, authority),
    }
    export_id = f"export-{canonical_sha256(manifest_core)}"
    manifest = {**manifest_core, "portable_export_id": export_id}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if _secret_like_key(manifest) is not None or _contains_absolute_string(manifest):
        raise ExecutionContractError("portable export contains unsafe local metadata")
    validate_instance("portable-export", manifest, version=2)
    manifest_path = authority.root / "export/portable-export-manifest.json"
    _durable_replace(
        manifest_path,
        _canonical_json(manifest),
        tuple_id="portable-export",
        prefix="portable_export",
        boundary_hook=None,
    )
    return ExportBundle(
        experiment_id=config.experiment_id,
        reconciliation_id=reconciliation.payload["reconciliation_id"],
        manifest_path=manifest_path,
    )


def export_reconciled_run(config_path: Path, artifact_root: Path) -> ExportBundle:
    config = load_experiment_config(config_path)
    with artifact_writer_lock(artifact_root) as authority:
        _initialize_locked(config, authority)
        reconciliation = _reconcile_locked(config, authority)
        return _export_locked(config, authority, reconciliation)


def replay_run(config_path: Path, primary_root: Path, replay_root: Path) -> dict[str, Any]:
    """Compare two already-complete roots and publish an equivalence report in the primary."""

    config = load_experiment_config(config_path)
    replay_reconciliation = reconcile_run(config_path, replay_root)
    replay_rows = _read_ledger_rows(Path(replay_root).resolve())
    replay_records = _scientific_records(Path(replay_root).resolve(), replay_rows, config)
    with artifact_writer_lock(primary_root) as authority:
        _initialize_locked(config, authority)
        primary_reconciliation = _reconcile_locked(config, authority)
        primary_rows = _read_ledger_rows(authority.root)
        primary_records = _scientific_records(authority.root, primary_rows, config)
        primary_projection = canonical_sha256(
            [_scientific_projection(record) for record in primary_records]
        )
        replay_projection = canonical_sha256(
            [_scientific_projection(record) for record in replay_records]
        )
        primary_pixels = canonical_sha256([str(row["output_pixel_sha256"]) for row in primary_rows])
        replay_pixels = canonical_sha256([str(row["output_pixel_sha256"]) for row in replay_rows])
        controls_sha = canonical_sha256(
            {
                "degradation": config.payload["controls"]["degradation_sha256"],
                "evaluation": config.payload["controls"]["evaluation_sha256"],
                "fixture": config.payload["fixture"]["manifest_sha256"],
                "core": config.payload["qualitative_core"]["core_sha256"],
            }
        )
        if primary_projection != replay_projection or primary_pixels != replay_pixels:
            raise ReconciliationError("clean-root replay differs from the interrupted run")
        core = {
            "schema_version": 2,
            "record_type": "replay-report",
            "experiment_id": config.experiment_id,
            "experiment_sha256": config.experiment_sha256,
            "primary_reconciliation_id": primary_reconciliation.payload["reconciliation_id"],
            "primary_reconciliation_sha256": primary_reconciliation.payload["report_sha256"],
            "replay_reconciliation_id": replay_reconciliation.payload["reconciliation_id"],
            "replay_reconciliation_sha256": replay_reconciliation.payload["report_sha256"],
            "expected_tuple_count": len(primary_rows),
            "resolved_controls_sha256": controls_sha,
            "primary_scientific_projection_sha256": primary_projection,
            "replay_scientific_projection_sha256": replay_projection,
            "primary_output_pixels_sha256": primary_pixels,
            "replay_output_pixels_sha256": replay_pixels,
            "status": "equivalent",
        }
        replay_id = f"replay-{canonical_sha256(core)}"
        report = {**core, "replay_id": replay_id}
        report["report_sha256"] = canonical_sha256(report)
        validate_instance("replay-report", report, version=2)
        _durable_replace(
            authority.root / "replay-report.json",
            _canonical_json(report),
            tuple_id="replay-report",
            prefix="replay_report",
            boundary_hook=None,
        )
        return report


def audit_no_smb(artifact_root: Path) -> dict[str, Any]:
    """Boundedly verify that a stable fixture artifact tree contains no SMB/secret marker."""

    with artifact_writer_lock(artifact_root) as authority:
        checked = 0
        for path in sorted(authority.root.rglob("*")):
            if path.is_symlink():
                raise ExecutionContractError("artifact root contains a symlink")
            if not path.is_file() or path.name in {_LOCK_NAME, _LEDGER_NAME}:
                continue
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ExecutionContractError("artifact root contains an oversized file")
            checked += 1
            if path.suffix.casefold() not in {".json", ".jsonl", ".yaml", ".yml", ".txt"}:
                continue
            text = _read_regular(path, maximum_bytes=64 * 1024 * 1024, kind="audit evidence")
            lowered = text.lower()
            if any(
                marker in lowered
                for marker in (b"praig/smb", b"hf_token", b"authorization: bearer", b'"token":')
            ):
                raise ExecutionContractError("artifact root crosses the fixture-only boundary")
        return {"status": "clean", "checked_file_count": checked}
