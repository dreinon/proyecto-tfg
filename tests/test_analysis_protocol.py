from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "protocols" / "analysis-v1.yaml"

REQUIRED_INTERFACE_KEYS = {
    "protocol_version",
    "status",
    "professional_problem",
    "audiences",
    "objectives",
    "evaluation_questions",
    "dataset_roles",
    "unit_hierarchy",
    "comparator_roles",
    "outcome_definitions",
    "success_rule",
    "claim_boundaries",
    "forbidden_adaptations",
    "smb_unlock_prerequisites",
    "internal_target",
    "hard_deadline",
    "contingency_window",
}

REQUIRED_HYPOTHESIS_FIELDS = {
    "comparison",
    "estimand",
    "direction",
    "unit",
    "testable_outcome",
}

REQUIRED_UNLOCK_PREREQUISITES = {
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
    "human_unlock_recorded",
}


def _load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _validate_protocol(protocol: dict[str, Any]) -> None:
    assert protocol.keys() >= REQUIRED_INTERFACE_KEYS
    assert protocol["protocol_version"] == "1.0.0"

    status = protocol["status"]
    assert status["contract"] == "locally_frozen"
    assert status["smb_outcomes"] == "locked"
    assert status["human_compatibility"] == "pending_academic_closeout"

    smb = protocol["dataset_roles"]["smb"]
    assert smb["role"] == "evaluation_benchmark"
    assert smb["outcomes_status"] == "locked"

    hypotheses = protocol.get("hypotheses")
    if hypotheses is not None:
        assert isinstance(hypotheses, list)
        for hypothesis in hypotheses:
            assert hypothesis.keys() >= REQUIRED_HYPOTHESIS_FIELDS
            assert all(hypothesis[field] for field in REQUIRED_HYPOTHESIS_FIELDS)


def test_analysis_protocol_has_complete_locked_interface() -> None:
    protocol = _load_protocol()

    _validate_protocol(protocol)
    assert {question["id"] for question in protocol["evaluation_questions"]} == {
        "EQ1",
        "EQ2",
        "EQ3",
        "EQ4",
        "EQ5",
    }
    assert len(protocol["objectives"]["specific"]) == 6
    assert protocol["unit_hierarchy"][0]["id"] == "UNIT-SOURCE"
    assert protocol["unit_hierarchy"][0]["audit_status"] == "pending_confirmation"
    assert {item["id"] for item in protocol["smb_unlock_prerequisites"]} == (
        REQUIRED_UNLOCK_PREREQUISITES
    )
    assert protocol["internal_target"] == "2026-08-31"
    assert protocol["hard_deadline"] == "2026-09-07"
    assert protocol["contingency_window"] == {
        "start": "2026-09-01",
        "end": "2026-09-06",
        "purpose": "corrections, clean replay, compliance review, and deposit contingency",
    }


@pytest.mark.parametrize("missing_key", sorted(REQUIRED_INTERFACE_KEYS))
def test_protocol_check_rejects_each_missing_interface_key(missing_key: str) -> None:
    protocol = _load_protocol()
    del protocol[missing_key]

    with pytest.raises(AssertionError):
        _validate_protocol(protocol)


@pytest.mark.parametrize(
    ("path", "unlocked_value"),
    (
        (("status", "smb_outcomes"), "unlocked"),
        (("dataset_roles", "smb", "outcomes_status"), "unlocked"),
    ),
)
def test_protocol_check_rejects_any_unlocked_smb_outcome_state(
    path: tuple[str, ...], unlocked_value: str
) -> None:
    protocol = deepcopy(_load_protocol())
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unlocked_value

    with pytest.raises(AssertionError):
        _validate_protocol(protocol)


def test_hypotheses_are_optional_but_must_be_testable_when_present() -> None:
    protocol = _load_protocol()
    protocol_without_hypotheses = deepcopy(protocol)
    protocol_without_hypotheses.pop("hypotheses", None)
    _validate_protocol(protocol_without_hypotheses)

    malformed = deepcopy(protocol)
    malformed["hypotheses"] = [
        {
            "comparison": "CMP-BICUBIC versus a later frozen comparator",
            "estimand": "paired difference",
            "direction": "greater fidelity",
            "unit": "UNIT-SOURCE",
        }
    ]
    with pytest.raises(AssertionError):
        _validate_protocol(malformed)


def test_protocol_selects_no_learned_method_or_checkpoint() -> None:
    protocol = _load_protocol()
    learned = protocol["comparator_roles"]["learned_methods"]

    assert learned["selection_status"] == "deferred_to_phase_3"
    assert learned["allowed_now"] is False
    assert "selected_method" not in protocol
    assert "selected_model" not in protocol
    assert "selected_checkpoint" not in protocol
    assert "checkpoint_id" not in protocol
