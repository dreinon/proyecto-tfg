# Professional study contract

This document freezes the local, review-ready control for the TFG *Study, Adaptation, and
Evaluation of Super-Resolution Techniques for Enhancing Digitized Music Scores*. It operationalizes
the approved offer and the project research protocol before any reportable SMB experimentation.
The matching machine-readable control is
[`configs/protocols/analysis-v1.yaml`](../configs/protocols/analysis-v1.yaml).

## Control status

- Protocol version: `1.0.0`.
- Contract state: locally frozen for review.
- SMB outcome state: locked. No SMB outcome may be generated or inspected under this contract.
- Human compatibility and approval are pending in `academic-closeout`; this document does not claim
  tutor approval.
- No learned method or checkpoint is selected or executed in Phase 1.

Any material change to the scientific comparison, interpretation, or delivery must be recorded
before affected work continues. Later human approval does not retroactively authorize an
unrecorded deviation.

## Professional decision problem

**D-01:** Determine when and how digitization, library, archive, conservation, and heritage
professionals can apply super-resolution to digitized music scores under evaluated conditions
without introducing visually plausible but musically meaningful changes.

The principal contribution is a **method-by-degradation-by-use** decision framework. The technical
comparison supplies evidence for that framework; it is not an isolated model leaderboard. A valid
conclusion may be that no tested method is suitable for a given condition or use.

## Intended audiences

The primary audiences are professionals responsible for digitization, libraries, archives,
conservation, and musical heritage collections. They need evidence about fidelity, musical risk,
cost, licensing, and operational constraints before applying super-resolution to score images.

Musicians, researchers, and publishers are secondary users. Any guidance for them remains subject
to the same evaluated population, degradation, rights, and evidence boundaries.

## Objectives

### General objective

- **OBJ-G:** Evaluate a small, scientifically contrasting set of super-resolution techniques under
  controlled, reproducible degradations of digitized music scores; quantify image fidelity,
  resource cost, and music-notation failures; and translate the evidence into a
  method-by-degradation-by-use decision framework with explicit limits and no-use cases. Completion
  is evidenced by answered evaluation questions and claim-to-evidence links, not by requiring one
  method to win.

### Specific objectives

- **OBJ-S1:** Establish a focused primary-source account of super-resolution foundations,
  document/music-score evidence, degradation assumptions, evaluation practice, and
  semantic-alteration risk. Measure: a traceable review matrix satisfying the Phase 1 search and
  coverage protocol.
- **OBJ-S2:** Authenticate, audit, describe, and immutably inventory SMB while preserving its sole
  official split as evaluation-only. Measure: a reconciled, checksummed manifest with one explicit
  status for every upstream item and reviewed provenance, grouping, rights, and suitability.
- **OBJ-S3:** Predeclare controlled degradation roles, comparator roles, independent units, outcome
  families, comparison rules, and claim boundaries before final SMB outcomes. Measure: versioned
  controls that fail closed when a required field or unlock prerequisite is absent.
- **OBJ-S4:** Compare later-frozen methods on identical paired inputs using complementary fidelity,
  resource, and music-specific evidence while retaining failures and negative cases. Measure:
  reconciled paired denominators and evidence for each declared outcome family.
- **OBJ-S5:** Make an evidence-based GO/NO-GO decision on bounded domain adaptation without using
  SMB for selection. Measure: a recorded gate addressing the hypothesis, independent licensed
  train/validation data, grouping, selection rule, time, compute, and tutor approval; a justified
  NO-GO satisfies the objective.
- **OBJ-S6:** Produce reproducible professional guidance and a traceable thesis whose material
  claims resolve to reviewed evidence. Measure: a reviewed claim index linking recommendations,
  limitations, tables, figures, and examples to recoverable evidence and rights status.

## Evaluation questions

- **EQ1:** Under each later-frozen controlled degradation and scale, how do interpolation and the
  later-frozen learned comparator roles differ in paired image fidelity and music-notation
  failures?
- **EQ2:** How do those differences vary by degradation severity and audited score characteristics
  when source score/work is treated as the independent grouping unit?
- **EQ3:** Which fidelity, notation-failure, runtime, memory, licence, and operational trade-offs
  make a method suitable or unsuitable for each named professional scenario?
- **EQ4:** Do any later-frozen learned comparators add consistent value over bicubic interpolation
  without increasing safety-critical musical alterations?
- **EQ5:** Conditional on validation-only baselines exposing a specific domain gap, does one bounded
  adaptation improve the declared balance using independent licensed train/validation data?

EQ5 does not authorize adaptation. Its comparison, estimand, direction, unit, outcome, selection
rule, data source, and stop rule must all be frozen before it can become a formal hypothesis.

## Unit hierarchy

- **UNIT-SOURCE:** The audited source score/work is the primary independent unit, subject to audit
  confirmation. The upstream `original_score` field may represent it only if the audit proves the
  field stable; otherwise a reviewed `source_group_id` mapping must replace it.
- **UNIT-PAGE:** Pages are paired items evaluated under identical methods and conditions. Page
  counts are reported but are not treated automatically as independent sample size.
- **UNIT-REGION:** Regions and crops are nested observations within pages and source groups. They
  may localize failures but must never be counted as independent replicates.

Grouping and aggregation must prevent source-related material from crossing adaptive splits or
inflating precision.

## Comparator roles

- **CMP-NEAREST:** nearest-neighbour interpolation is a transparent low-complexity reference.
- **CMP-BILINEAR:** bilinear interpolation is a transparent smooth interpolation reference.
- **CMP-BICUBIC:** bicubic interpolation is the principal simple interpolation baseline.
- Learned comparators have only a future scientific role in this contract. Their selection is
  deferred to Phase 3 after the state-of-the-art refresh and validation-only evidence. No model or
  weights identity is frozen here.
- Domain adaptation is conditional on the post-baseline gate and independent licensed
  train/validation evidence. It is not a mandatory comparator.

Every comparable method must later receive identical paired inputs, conditions, aggregation, and
outcome definitions.

## Outcome definitions

- **OUT-FIDELITY:** paired reference-fidelity evidence under fixed colour, range, alignment,
  border, scale, and aggregation conventions. Exact metric controls are frozen in Phase 2.
- **OUT-NOTATION:** systematic evidence of broken, removed, thickened, joined, separated,
  deformed, or hallucinated staff lines, symbols, text, digits, and other musically meaningful
  changes, using a predeclared taxonomy and sampling rule.
- **OUT-RESOURCE:** runtime, memory, model size/parameters, hardware, warm-up, repeats, batch, and
  tile evidence where relevant to a professional decision.
- **OUT-PROFESSIONAL:** a bounded suitability or no-use decision derived by triangulating fidelity,
  notation failures, resources, licences, rights, and operational constraints. It is never inferred
  from one aggregate metric.

Perceptual evidence may be added only when its domain validity and interpretation are declared; it
cannot substitute for reference fidelity or musical correctness.

## Success and stop rules

A method is professionally supportable only for a named audience, use, SMB subgroup, and frozen
controlled condition when all of the following hold:

1. its paired evidence improves a predeclared benefit relative to the relevant comparator;
2. notation-failure evidence does not contradict or make the use unsafe;
3. resource, licence, rights, and operating constraints fit the stated scenario; and
4. every claim resolves to reconciled evidence with its denominator and limitations.

"No method recommended" is a valid and reportable result. Stop or withhold a recommendation when
evidence is incomplete, denominators do not reconcile, musical failures contradict the intended
use, rights are unresolved, or the measured benefit does not justify the cost or risk. Stop all SMB
outcome work while any unlock prerequisite below is incomplete.

### SMB unlock prerequisites

Phase 4 SMB execution requires a recorded human unlock only after all of these are complete and
identified by version or hash where applicable:

- `smb_audit_complete`: authenticated audit and reviewed grouping, suitability, provenance, and
  rights dispositions are complete;
- `evaluation_manifest_frozen`: the immutable evaluation inventory and denominator reconcile;
- `methods_frozen`: every comparison method and implementation identity is frozen;
- `checkpoints_frozen`: every learned weights/checkpoint identity and checksum is frozen;
- `controlled_conditions_frozen`: degradation conditions and execution conventions are frozen;
- `metrics_frozen`: quantitative and qualitative outcome definitions are frozen;
- `independent_units_frozen`: grouping, pairing, aggregation, and uncertainty units are frozen;
- `exclusions_frozen`: all exclusion and failure-handling rules are frozen;
- `seeds_frozen`: generation, execution, and sampling seed policies are frozen;
- `qualitative_samples_frozen`: the outcome-independent qualitative sampling rule and IDs are
  frozen;
- `interpretation_rules_frozen`: success, stop, contradiction, and claim rules are frozen; and
- `human_unlock_recorded`: the accountable approval record names every preceding frozen control.

An absent, incomplete, or malformed record means locked.

## Claim boundaries

- **CLAIM-SMB-CONTROLLED:** findings may describe only the audited SMB population and subgroups
  under the exact controlled degradations, methods, and evaluation rules later executed.
- **CLAIM-EXCLUDED:** uncontrolled real scans, PDF ingestion or reconstruction, OMR benefit,
  user-facing applications, deployment, and populations outside audited SMB are unvalidated future
  work unless a later gated protocol adds direct evidence.
- **CLAIM-PENDING-RESULTS:** this contract contains no experimental result, method ranking,
  professional recommendation, audit finding, tutor statement, or claim that SMB is already
  suitable. Descriptor and upstream statements remain reported facts until the authenticated audit
  reconciles them.

No visual plausibility, isolated metric, or selected example may be presented as evidence of
musical correctness.

## Hypotheses policy

Formal hypotheses are optional. The current contract predeclares evaluation questions and contains
no hypothesis. If a later comparison benefits from one, the hypothesis must name its comparison,
estimand, expected direction, independent unit, testable outcome, analysis rule, and stop rule
before the relevant final evidence is inspected. A hypothesis may not be added after observing SMB
outcomes to explain or select a favourable result.

## Schedule guardrails

- Internal target: a deposit-ready candidate by **31 August 2026**.
- Contingency window: **1-6 September 2026**, reserved for tutor feedback, invalidating bug fixes,
  clean replay, final compliance, and deposit operations rather than scope expansion.
- Hard deposit deadline: **7 September 2026**.

Missing a phase guardrail triggers explicit replanning. Optional models, adaptation, applicability
extensions, or presentation polish stop before they threaten the controlled core, thesis validity,
or deposit margin.

## Sources of authority

This contract derives from the approved multilingual offer, Phase 1 decisions D-01 through D-04,
[`docs/research-protocol.md`](research-protocol.md), the workspace `AGENTS.md`, and the routed
UPV/ETSINF/GCD academic guidance. Binding regulations and later amendments prevail over derived
guidance. This local freeze remains review-ready rather than human-approved until the separate
`academic-closeout` evidence says otherwise.
