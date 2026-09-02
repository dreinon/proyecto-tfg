from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "docs" / "experimental-claim-evidence.csv"
COLUMNS = (
    "claim_id",
    "chapter_section",
    "claim",
    "evidence_path",
    "evidence_sha256",
    "scope_limitations",
    "review_status",
    "reviewer",
    "review_date",
)


def test_experimental_claims_are_bounded_and_resolve_when_evidence_is_retained() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == COLUMNS
    assert {row["claim_id"] for row in rows} == {
        "RESULT-V2-RECONCILIATION",
        "RESULT-LEARNED-VS-BICUBIC",
        "RESULT-EDSR-SWINIR",
        "RESULT-QUALITATIVE",
        "PROF-EDSR-DEFAULT",
        "ADAPT-V1-RECONCILIATION",
        "ADAPT-V1-FIDELITY",
        "ADAPT-V1-QUALITATIVE",
    }
    for row in rows:
        assert row["review_status"] == "reviewed"
        assert row["scope_limitations"] and row["reviewer"]
        assert row["review_date"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"])
        evidence = ROOT / row["evidence_path"]
        if evidence.exists():
            assert hashlib.sha256(evidence.read_bytes()).hexdigest() == row["evidence_sha256"]
