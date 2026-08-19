"""Fail-closed purpose and unlock policy for the SMB evaluation benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml

from score_super_resolution.contracts import ContractValidationError, validate_instance


class BenchmarkPolicyError(PermissionError):
    """Report a rejected or malformed SMB policy request."""


class BenchmarkPurpose(StrEnum):
    """Every supported reason for crossing the SMB policy boundary."""

    METADATA_INSPECTION = "metadata_inspection"
    SCHEMA_AUDIT = "schema_audit"
    CONTENT_AUDIT = "content_audit"
    GROUPING_AUDIT = "grouping_audit"
    PROVENANCE_AUDIT = "provenance_audit"
    RIGHTS_AUDIT = "rights_audit"
    SUITABILITY_AUDIT = "suitability_audit"
    QUALITY_AUDIT = "quality_audit"
    DUPLICATE_AUDIT = "duplicate_audit"
    FIXED_VISUAL_AUDIT = "fixed_visual_audit"

    DEGRADATION = "degradation"
    INFERENCE = "inference"
    SR_OUTPUT = "sr_output"
    METRIC = "metric"

    TUNING = "tuning"
    FITTING = "fitting"
    SELECTION = "selection"
    OUTCOME_SAMPLING = "outcome_sampling"


class BenchmarkState(StrEnum):
    """States relevant to the SMB outcome quarantine."""

    AUDIT_LOCKED = "AUDIT_LOCKED"
    EVALUATION_UNLOCKED = "EVALUATION_UNLOCKED"


AUDIT_PURPOSES = frozenset(
    {
        BenchmarkPurpose.METADATA_INSPECTION,
        BenchmarkPurpose.SCHEMA_AUDIT,
        BenchmarkPurpose.CONTENT_AUDIT,
        BenchmarkPurpose.GROUPING_AUDIT,
        BenchmarkPurpose.PROVENANCE_AUDIT,
        BenchmarkPurpose.RIGHTS_AUDIT,
        BenchmarkPurpose.SUITABILITY_AUDIT,
        BenchmarkPurpose.QUALITY_AUDIT,
        BenchmarkPurpose.DUPLICATE_AUDIT,
        BenchmarkPurpose.FIXED_VISUAL_AUDIT,
    }
)

FINAL_EVALUATION_PURPOSES = frozenset(
    {
        BenchmarkPurpose.DEGRADATION,
        BenchmarkPurpose.INFERENCE,
        BenchmarkPurpose.SR_OUTPUT,
        BenchmarkPurpose.METRIC,
    }
)

NEVER_ALLOWED_PURPOSES = frozenset(
    {
        BenchmarkPurpose.TUNING,
        BenchmarkPurpose.FITTING,
        BenchmarkPurpose.SELECTION,
        BenchmarkPurpose.OUTCOME_SAMPLING,
    }
)

_SMB_KEY = "smb"
_SMB_ROLE = "evaluation_benchmark"
_PRECEDING_PREREQUISITES = (
    "smb_audit_complete",
    "evaluation_manifest_frozen",
    "methods_frozen",
    "checkpoints_frozen",
    "controlled_conditions_frozen",
    "metrics_frozen",
    "independent_units_frozen",
    "exclusions_frozen",
    "seeds_frozen",
    "qualitative_samples_frozen",
    "interpretation_rules_frozen",
)


def _policy_error(detail: str) -> BenchmarkPolicyError:
    return BenchmarkPolicyError(f"SMB benchmark policy rejected: {detail}")


def _validate_source_descriptor(source_descriptor: object) -> None:
    if not isinstance(source_descriptor, Mapping):
        raise _policy_error("source_descriptor must be a parsed mapping")
    if source_descriptor.get("key") != _SMB_KEY:
        raise _policy_error("source_descriptor.key must identify smb")
    if source_descriptor.get("role") != _SMB_ROLE:
        raise _policy_error("source_descriptor.role must be evaluation_benchmark")


def _parse_purpose(purpose: object) -> BenchmarkPurpose:
    if isinstance(purpose, BenchmarkPurpose):
        return purpose
    if isinstance(purpose, str):
        try:
            return BenchmarkPurpose(purpose)
        except ValueError:
            pass
    raise _policy_error("purpose is missing or unknown")


def _parse_state(state: object) -> BenchmarkState:
    if isinstance(state, BenchmarkState):
        return state
    if isinstance(state, str):
        try:
            return BenchmarkState(state)
        except ValueError:
            pass
    raise _policy_error("state is missing or unknown")


class _ArtifactVerificationError(ValueError):
    """Keep untrusted artifact details behind one non-disclosing policy error."""


@dataclass
class _VerificationContext:
    project_root: Path
    manifest_generation_root: Path
    unlock_record: Mapping[str, object]
    artifacts: dict[str, Mapping[str, object]] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


type _Parser = Callable[[bytes], Mapping[str, object]]
type _SemanticVerifier = Callable[
    [str, Mapping[str, object], Mapping[str, object], _VerificationContext], None
]


@dataclass(frozen=True)
class _PrerequisiteContract:
    artifact_kind: str
    schema_id: str
    schema_versions: tuple[int, ...]
    parser: _Parser
    semantic_verifier: _SemanticVerifier


def _parse_json_mapping(payload: bytes) -> Mapping[str, object]:
    try:
        loaded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _ArtifactVerificationError("artifact is not valid JSON") from error
    if not isinstance(loaded, dict):
        raise _ArtifactVerificationError("artifact root is not a mapping")
    return loaded


def _parse_yaml_mapping(payload: bytes) -> Mapping[str, object]:
    try:
        loaded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise _ArtifactVerificationError("artifact is not valid YAML") from error
    if not isinstance(loaded, dict):
        raise _ArtifactVerificationError("artifact root is not a mapping")
    return loaded


def _read_root_confined_regular(root: Path, relative: object) -> tuple[bytes, Path]:
    if not isinstance(relative, str):
        raise _ArtifactVerificationError("artifact path is not text")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise _ArtifactVerificationError("artifact path is not canonical")

    try:
        if root.is_symlink() or not root.is_dir():
            raise _ArtifactVerificationError("artifact root is unavailable")
        root = root.resolve(strict=True)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        root_fd = os.open(root, directory_flags)
        directory_fd = root_fd
        try:
            for part in relative_path.parts[:-1]:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                if directory_fd != root_fd:
                    os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(relative_path.parts[-1], file_flags, dir_fd=directory_fd)
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise _ArtifactVerificationError("artifact is not a regular file")
                chunks: list[bytes] = []
                while chunk := os.read(file_fd, 1024 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(file_fd)
        finally:
            if directory_fd != root_fd:
                os.close(directory_fd)
            os.close(root_fd)
    except _ArtifactVerificationError:
        raise
    except OSError as error:
        raise _ArtifactVerificationError("artifact cannot be opened safely") from error
    return b"".join(chunks), root / relative_path


def _verify_control_identity(
    prerequisite_id: str,
    reference: Mapping[str, object],
    artifact: Mapping[str, object],
    context: _VerificationContext,
) -> None:
    del reference, context
    if artifact.get("control_id") != prerequisite_id or artifact.get("status") != "frozen":
        raise _ArtifactVerificationError("control identity does not satisfy prerequisite")


def _verify_manifest_identity(
    prerequisite_id: str,
    reference: Mapping[str, object],
    artifact: Mapping[str, object],
    context: _VerificationContext,
) -> None:
    del prerequisite_id
    generation_id = artifact.get("generation_id")
    if not isinstance(generation_id, str):
        raise _ArtifactVerificationError("manifest generation identity is absent")
    _read_root_confined_regular(
        context.manifest_generation_root,
        f"{generation_id}/manifest-descriptor.yaml",
    )
    _read_root_confined_regular(
        context.manifest_generation_root,
        f"{generation_id}/manifest-records.jsonl",
    )
    try:
        from score_super_resolution.smb_audit import resolve_active_manifest

        descriptor, rows = resolve_active_manifest(
            active_path=context.artifact_paths["evaluation_manifest_frozen"],
            generation_root=context.manifest_generation_root,
        )
    except Exception as error:
        raise _ArtifactVerificationError("active manifest generation cannot be resolved") from error

    expected = {
        "generation_id": reference["expected_generation_id"],
        "records_sha256": reference["expected_records_sha256"],
        "row_count": reference["expected_row_count"],
    }
    if any(artifact.get(field) != value for field, value in expected.items()):
        raise _ArtifactVerificationError("active pointer does not match frozen identity")
    if any(descriptor.get(field) != value for field, value in expected.items()):
        raise _ArtifactVerificationError("manifest descriptor does not match frozen identity")
    if len(rows) != reference["expected_row_count"]:
        raise _ArtifactVerificationError("manifest row count does not match frozen identity")
    if descriptor.get("benchmark_state") != reference["expected_benchmark_state"]:
        raise _ArtifactVerificationError("manifest state is not the frozen audited state")


def _verify_human_approval(
    prerequisite_id: str,
    reference: Mapping[str, object],
    artifact: Mapping[str, object],
    context: _VerificationContext,
) -> None:
    _verify_control_identity(prerequisite_id, reference, artifact, context)
    content = artifact["content"]
    if not isinstance(content, Mapping):
        raise _ArtifactVerificationError("human approval content is absent")
    if (
        content.get("reviewer") != context.unlock_record["reviewer"]
        or content.get("reviewed_at") != context.unlock_record["reviewed_at"]
    ):
        raise _ArtifactVerificationError("human approval does not match unlock reviewer")
    digests = content.get("prerequisite_digests")
    if not isinstance(digests, Mapping) or any(
        digests.get(control_id)
        != context.unlock_record["prerequisites"][control_id]["artifact_sha256"]
        for control_id in _PRECEDING_PREREQUISITES
    ):
        raise _ArtifactVerificationError("human approval does not digest every prerequisite")

    evidence_reference = content.get("approval_evidence")
    if not isinstance(evidence_reference, Mapping):
        raise _ArtifactVerificationError("human review evidence is absent")
    evidence_bytes, _ = _read_root_confined_regular(
        context.project_root, evidence_reference.get("artifact_path")
    )
    if hashlib.sha256(evidence_bytes).hexdigest() != evidence_reference.get("artifact_sha256"):
        raise _ArtifactVerificationError("human review evidence digest does not match")
    evidence = _parse_json_mapping(evidence_bytes)
    if set(evidence) != {"evidence_type", "evidence_id", "reviewer", "reviewed_at", "decision"}:
        raise _ArtifactVerificationError("human review evidence fields are not exact")
    for field_name in ("evidence_type", "evidence_id", "reviewer", "reviewed_at", "decision"):
        expected_value = (
            evidence_reference[field_name]
            if field_name in evidence_reference
            else content[field_name]
        )
        if evidence.get(field_name) != expected_value:
            raise _ArtifactVerificationError("human review evidence does not match approval")


_PREREQUISITE_CONTRACTS = {
    "smb_audit_complete": _PrerequisiteContract(
        "smb_audit_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "evaluation_manifest_frozen": _PrerequisiteContract(
        "manifest_active", "manifest-active", (1, 2), _parse_yaml_mapping, _verify_manifest_identity
    ),
    "methods_frozen": _PrerequisiteContract(
        "methods_control", "smb-freeze-control", (1,), _parse_json_mapping, _verify_control_identity
    ),
    "checkpoints_frozen": _PrerequisiteContract(
        "checkpoints_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "controlled_conditions_frozen": _PrerequisiteContract(
        "controlled_conditions_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "metrics_frozen": _PrerequisiteContract(
        "metrics_control", "smb-freeze-control", (1,), _parse_json_mapping, _verify_control_identity
    ),
    "independent_units_frozen": _PrerequisiteContract(
        "independent_units_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "exclusions_frozen": _PrerequisiteContract(
        "exclusions_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "seeds_frozen": _PrerequisiteContract(
        "seeds_control", "smb-freeze-control", (1,), _parse_json_mapping, _verify_control_identity
    ),
    "qualitative_samples_frozen": _PrerequisiteContract(
        "qualitative_samples_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "interpretation_rules_frozen": _PrerequisiteContract(
        "interpretation_rules_control",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_control_identity,
    ),
    "human_unlock_recorded": _PrerequisiteContract(
        "human_unlock_approval",
        "smb-freeze-control",
        (1,),
        _parse_json_mapping,
        _verify_human_approval,
    ),
}


def _validate_unlock_artifacts(
    unlock_record: object,
    *,
    project_root: Path | None,
    manifest_generation_root: Path | None,
) -> None:
    try:
        if not isinstance(unlock_record, Mapping):
            raise _ArtifactVerificationError("unlock record is not a mapping")
        validate_instance("smb-evaluation-unlock", unlock_record)
        date.fromisoformat(unlock_record["reviewed_at"])
        if project_root is None or manifest_generation_root is None:
            raise _ArtifactVerificationError("artifact roots are required")
        project_root = Path(project_root)
        manifest_generation_root = Path(manifest_generation_root)
        resolved_project_root = project_root.resolve(strict=True)
        resolved_generation_root = manifest_generation_root.resolve(strict=True)
        if (
            project_root.is_symlink()
            or manifest_generation_root.is_symlink()
            or not resolved_generation_root.is_relative_to(resolved_project_root)
        ):
            raise _ArtifactVerificationError("artifact roots are not confined")

        context = _VerificationContext(
            project_root=resolved_project_root,
            manifest_generation_root=resolved_generation_root,
            unlock_record=unlock_record,
        )
        prerequisites = unlock_record["prerequisites"]
        if not isinstance(prerequisites, Mapping):
            raise _ArtifactVerificationError("prerequisites are not a mapping")
        original_bytes: dict[str, bytes] = {}
        for prerequisite_id, contract in _PREREQUISITE_CONTRACTS.items():
            reference = prerequisites[prerequisite_id]
            if not isinstance(reference, Mapping):
                raise _ArtifactVerificationError("prerequisite reference is not a mapping")
            if (
                reference.get("artifact_kind") != contract.artifact_kind
                or reference.get("schema_id") != contract.schema_id
                or reference.get("schema_version") not in contract.schema_versions
            ):
                raise _ArtifactVerificationError("prerequisite contract is not allowlisted")
            payload, artifact_path = _read_root_confined_regular(
                context.project_root, reference.get("artifact_path")
            )
            if hashlib.sha256(payload).hexdigest() != reference.get("artifact_sha256"):
                raise _ArtifactVerificationError("artifact digest does not match")
            artifact = contract.parser(payload)
            validate_instance(contract.schema_id, artifact, int(reference["schema_version"]))
            context.artifact_paths[prerequisite_id] = artifact_path
            context.artifacts[prerequisite_id] = artifact
            contract.semantic_verifier(prerequisite_id, reference, artifact, context)
            original_bytes[prerequisite_id] = payload

        audit_content = context.artifacts["smb_audit_complete"]["content"]
        manifest_reference = prerequisites["evaluation_manifest_frozen"]
        if not isinstance(audit_content, Mapping) or any(
            audit_content.get(audit_field) != manifest_reference[reference_field]
            for audit_field, reference_field in (
                ("active_generation_id", "expected_generation_id"),
                ("row_count", "expected_row_count"),
                ("records_sha256", "expected_records_sha256"),
                ("benchmark_state", "expected_benchmark_state"),
            )
        ):
            raise _ArtifactVerificationError("audit and manifest controls disagree")

        for prerequisite_id, reference in prerequisites.items():
            payload, _ = _read_root_confined_regular(
                context.project_root, reference["artifact_path"]
            )
            if payload != original_bytes[prerequisite_id]:
                raise _ArtifactVerificationError("artifact changed during verification")
    except (
        _ArtifactVerificationError,
        ContractValidationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise _policy_error("unlock_record or referenced artifacts are invalid") from error


def assert_smb_purpose_allowed[T](
    *,
    source_descriptor: Mapping[str, object],
    purpose: BenchmarkPurpose | str,
    state: BenchmarkState | str = BenchmarkState.AUDIT_LOCKED,
    unlock_record: Mapping[str, object] | None = None,
    project_root: Path | None = None,
    manifest_generation_root: Path | None = None,
    callback: Callable[[], T] | None = None,
) -> T | None:
    """Validate one SMB operation before invoking its optional callback.

    The source identity comes from the already parsed descriptor. Omitted state is locked. A
    caller may name ``EVALUATION_UNLOCKED`` only with an exact, complete, reviewed record for all
    frozen controls. Adaptive purposes remain forbidden because SMB is evaluation-only.
    """

    _validate_source_descriptor(source_descriptor)
    parsed_purpose = _parse_purpose(purpose)
    parsed_state = _parse_state(state)

    if parsed_purpose in NEVER_ALLOWED_PURPOSES:
        raise _policy_error(
            f"purpose={parsed_purpose.value} is forbidden for the evaluation benchmark"
        )

    if parsed_state is BenchmarkState.EVALUATION_UNLOCKED:
        _validate_unlock_artifacts(
            unlock_record,
            project_root=project_root,
            manifest_generation_root=manifest_generation_root,
        )

    if (
        parsed_purpose in FINAL_EVALUATION_PURPOSES
        and parsed_state is not BenchmarkState.EVALUATION_UNLOCKED
    ):
        raise _policy_error(
            f"purpose={parsed_purpose.value} requires a complete reviewed evaluation unlock"
        )

    if parsed_purpose not in AUDIT_PURPOSES | FINAL_EVALUATION_PURPOSES:
        raise _policy_error(f"purpose={parsed_purpose.value} is not allowed")

    if callback is None:
        return None
    if not callable(callback):
        raise _policy_error("callback must be callable")
    return callback()
