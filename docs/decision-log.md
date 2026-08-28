# Decision log

This append-only ledger records material decisions that affect scope, comparability,
interpretation, or delivery. Stable local scientific decisions already frozen in
`study-contract.md` and `analysis-v1.yaml` remain authoritative there; this initial register
prepares the human/external decisions without inventing their outcomes.

Evidence references are either tracked project paths/IDs or opaque `external-ref:` labels. The
opaque label points to evidence held outside Git and contains no private payload. Silence or a
pending state never means acceptance.

## Decision register

| Record ID | Date | Owner | Status | Subject | Rationale | Evidence reference | Affected controls/runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEC-ACAD-01 | 2026-08-18 | Student and tutor | pending_human | Compatibility of the SMB-centred core with the official offer's plural datasets wording | This is the one material scope interpretation that requires an attributable tutor answer; reversible technical details do not | external-ref:ACAD-01-TUTOR-COMPATIBILITY | CTRL-05; no experiment run is authorized or invalidated by this pending row |
| DEC-ACAD-02 | 2026-08-18 | Student | pending_private_check | Current seminar, enrolment, assignment, CAT, article 11, Ebrón record, and live-access state | These are personal or live administrative facts and cannot be inferred from public guidance | external-ref:ACAD-02-PRIVATE-ELIGIBILITY | DELV-01; no scientific run affected; Phase 5 deposit closeout remains open |
| DEC-ACAD-03 | 2026-08-18 | Student | pending_authoritative_review | Authoritative Overleaf synchronization, both renders, and promoted-claim review | Local preflight cannot establish the state of the user-controlled authoritative project | external-ref:ACAD-03-OVERLEAF-REVIEW | THES-04; THES-16; no experiment run affected; Phase 5 deposit closeout remains open |
| DEC-SCI-01 | 2026-08-25 | Student | decided | Retain x2/x4 as the comparable controlled core | x2 remains the reference condition and x4 the principal challenge; x6/x8 are not core because narrowing native model compatibility would weaken fair comparison; candidate 4 instead strengthens fine-detail loss and uses denser authored ROIs | external-ref:PHASE2-CANDIDATE3-REVIEW | DEGR-01..04; controlled-score-v4-candidate; no SMB or benchmark run affected |
| DEC-SCI-02 | 2026-08-28 | Student | decided | Final direct candidate after candidate 4 remained insufficiently challenging | Preserve the x2 reference and x4 principal challenge roles plus the unchanged moderate tier; candidate 5 is the final direct increment and strengthens only blur/compression-oriented strong degradation; rejection routes to a predeclared three-level calibration bracket | external-ref:PHASE2-CANDIDATE4-REVIEW | DEGR-01..04; controlled-score-v5-candidate; no SMB or benchmark run affected |

## Update rules

1. Append a new stable record; do not rewrite a prior outcome to hide history.
2. Record the decision date, accountable role, rationale, evidence reference, and affected
   controls/runs before dependent work proceeds.
3. A material requested change creates or links to a deviation record before execution.
4. Keep private correspondence, personal records, screenshots, account details, and signatures
   outside Git. Only a safe status/date/source category/opaque reference may be tracked here.
5. A rejected, silent, inaccessible, stale, or pending human outcome remains open.
