from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORK_PLAN_PATH = ROOT / "docs" / "work-plan.md"
EFFORT_LOG_PATH = ROOT / "docs" / "effort-log.csv"

PHASE_COLUMNS = (
    "Phase",
    "Target dates",
    "Planned hours",
    "Human resources",
    "Material resources",
    "Approximate cost",
    "Key risks",
    "Checkpoint owner",
)
EFFORT_COLUMNS = (
    "entry_id",
    "period_start",
    "period_end",
    "phase",
    "work_item",
    "estimate_hours",
    "low_hours",
    "high_hours",
    "evidence_reference",
    "estimate_basis",
    "verification_status",
    "notes",
)
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _markdown_table(text: str, heading: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    section = text.split(heading, maxsplit=1)[1]
    lines = []
    for line in section.splitlines():
        if line.startswith("|"):
            lines.append(line)
        elif lines:
            break
    header = tuple(cell.strip() for cell in lines[0].strip("|").split("|"))
    rows = []
    for line in lines[2:]:
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        if len(cells) != len(header):
            break
        rows.append(dict(zip(header, cells, strict=True)))
    return header, rows


def test_work_plan_allocates_a_complete_12_ects_workload() -> None:
    text = WORK_PLAN_PATH.read_text(encoding="utf-8")
    columns, rows = _markdown_table(text, "## Phase allocation")

    assert columns == PHASE_COLUMNS
    assert [row["Phase"] for row in rows] == ["P1", "P2", "P3", "P4", "P5"]
    assert 300 <= sum(int(row["Planned hours"]) for row in rows) <= 360
    for row in rows:
        assert all(row[column] for column in PHASE_COLUMNS)
        assert "pending" not in row["Planned hours"].lower()
        assert "actual" not in row["Planned hours"].lower()


def test_work_plan_preserves_schedule_and_d16_lanes() -> None:
    text = WORK_PLAN_PATH.read_text(encoding="utf-8").lower()

    assert "31 august 2026" in text
    assert "1-6 september 2026" in text
    assert "3 september 2026" in text
    assert "7 september 2026" in text
    assert "phase 2 blocking scientific core" in text
    assert "non-blocking sota and thesis enrichment" in text
    assert "academic-closeout" in text
    assert "defence window" in text
    assert "provisional" in text


def test_work_plan_has_cost_resource_risk_and_checkpoint_controls() -> None:
    text = WORK_PLAN_PATH.read_text(encoding="utf-8")
    risk_columns, risk_rows = _markdown_table(text, "## Risk register")

    assert risk_columns == (
        "Risk ID",
        "Risk",
        "Likelihood",
        "Impact",
        "Trigger",
        "Mitigation or contingency",
        "Owner",
        "Checkpoint",
    )
    assert len(risk_rows) >= 6
    assert len({row["Risk ID"] for row in risk_rows}) == len(risk_rows)
    for row in risk_rows:
        assert all(row.values())

    lowered = text.lower()
    assert "estimated, not incurred" in lowered
    assert "no actual cost" in lowered
    assert "human resources" in lowered
    assert "material resources" in lowered


def test_effort_log_separates_reconstruction_from_forecast() -> None:
    with EFFORT_LOG_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == EFFORT_COLUMNS
    assert [row["phase"] for row in rows] == ["P1", "P2", "P3", "P4", "P5", "P5"]
    reconstructed = [row for row in rows if row["verification_status"] == "reconstructed_estimate"]
    forecast = [row for row in rows if row["verification_status"] == "forecast"]
    assert sum(int(row["estimate_hours"]) for row in reconstructed) == 346
    assert sum(int(row["low_hours"]) for row in reconstructed) == 312
    assert sum(int(row["high_hours"]) for row in reconstructed) == 380
    assert [int(row["estimate_hours"]) for row in forecast] == [22]
    assert all(
        int(row["low_hours"]) <= int(row["estimate_hours"]) <= int(row["high_hours"])
        for row in rows
    )
    assert all(
        row["estimate_basis"] in {"task_based_reconstruction", "remaining_forecast"} for row in rows
    )
    for row in rows:
        for value in row.values():
            if value:
                assert not value.lstrip().startswith(FORMULA_PREFIXES), value
