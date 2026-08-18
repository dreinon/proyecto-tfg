from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "docs" / "study-contract.md"
PROTOCOL_PATH = ROOT / "configs" / "protocols" / "analysis-v1.yaml"

REQUIRED_SECTIONS = (
    "## Control status",
    "## Professional decision problem",
    "## Intended audiences",
    "## Objectives",
    "## Evaluation questions",
    "## Unit hierarchy",
    "## Comparator roles",
    "## Outcome definitions",
    "## Success and stop rules",
    "## Claim boundaries",
    "## Hypotheses policy",
    "## Schedule guardrails",
)


def _assert_contract_fields(text: str) -> None:
    for section in REQUIRED_SECTIONS:
        assert section in text, f"missing contract section: {section}"


def test_study_contract_contains_all_required_controls() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")

    _assert_contract_fields(contract)
    assert "method-by-degradation-by-use" in contract
    assert "digitization" in contract
    assert "library" in contract
    assert "archive" in contract
    assert "conservation" in contract
    assert "heritage" in contract
    assert "Musicians, researchers, and publishers" in contract
    assert contract.count("**OBJ-S") == 6
    assert contract.count("**EQ") == 5
    assert "source score/work" in contract
    assert "Pages are paired items" in contract
    assert "Regions and crops are nested observations" in contract
    assert "nearest-neighbour" in contract
    assert "bilinear" in contract
    assert "bicubic" in contract
    assert "no method recommended" in contract.lower()
    assert "31 August 2026" in contract
    assert "1-6 September 2026" in contract
    assert "7 September 2026" in contract


@pytest.mark.parametrize("missing_section", REQUIRED_SECTIONS)
def test_contract_check_rejects_each_missing_section(missing_section: str) -> None:
    complete = "\n".join(REQUIRED_SECTIONS)

    with pytest.raises(AssertionError, match="missing contract section"):
        _assert_contract_fields(complete.replace(missing_section, ""))


def test_markdown_and_yaml_use_matching_control_identifiers() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))

    identifiers = {
        protocol["professional_problem"]["id"],
        protocol["objectives"]["general"]["id"],
        *(objective["id"] for objective in protocol["objectives"]["specific"]),
        *(question["id"] for question in protocol["evaluation_questions"]),
        *(unit["id"] for unit in protocol["unit_hierarchy"]),
        *(comparator["id"] for comparator in protocol["comparator_roles"]["interpolation"]),
        *(outcome["id"] for outcome in protocol["outcome_definitions"]),
        *(boundary["id"] for boundary in protocol["claim_boundaries"]),
    }

    assert identifiers
    assert all(identifier in contract for identifier in identifiers)


def test_contract_keeps_human_approval_and_results_pending() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8").lower()

    assert "academic-closeout" in contract
    assert "human compatibility and approval are pending" in contract
    assert "no smb outcome" in contract
    assert "no learned method or checkpoint is selected" in contract
