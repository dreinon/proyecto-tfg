# Work, resource, cost, risk, and deadline plan

This is the initial 12-ECTS workload allocation for the complete TFG. It is a planning control,
not a claim about work already performed. Because no contemporaneous timesheet was maintained, the
student authorized a transparent retrospective reconstruction on 1 September 2026. The initial
330-hour plan and the reconstructed estimate are kept separate; `effort-log.csv` records central
estimates, uncertainty bounds, and evidence rather than presenting reconstructed values as clocked
hours. Both the initial plan and the 348-hour final forecast are within the GCD guidance range of
approximately 300-360 hours.

The target windows below are evidence-gate dates, not statements of actual attendance or completed
effort. The compressed calendar is a material schedule risk: it is viable only if the student-entered
actual log later shows that sufficient eligible work preceded this freeze and the remaining critical
path still fits. Otherwise the schedule must be replanned instead of inventing hours.

## Phase allocation

| Phase | Target dates | Planned hours | Human resources | Material resources | Approximate cost | Key risks | Checkpoint owner |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| P1 | 14-18 August 2026 | 60 | Student for controls and audit; tutor only through ACAD-01 | Versioned workspace; local CPU; gated SMB source | EUR 0-10 planned incremental | R-01; R-02; R-03 | Student for scientific core; academic-closeout for human gates |
| P2 | 18-21 August 2026 | 65 | Student | Local CPU; deterministic fixtures; project storage | EUR 5-30 planned incremental | R-01; R-05; R-06 | Student at degradation and evaluation freeze |
| P3 | 21-24 August 2026 | 75 | Student; tutor only for a material scientific gate | Local CPU; Kaggle only through the shared package path | EUR 5-40 planned incremental | R-01; R-04; R-05 | Student; tutor for any scope-changing choice |
| P4 | 24-27 August 2026 | 75 | Student; accountable human for SMB unlock | Frozen SMB manifest; Kaggle runtime if required; external artefact storage | EUR 5-50 planned incremental | R-01; R-02; R-05; R-06 | Student plus recorded human unlock owner |
| P5 | 27-31 August 2026 | 55 | Student; tutor and tribunal-facing reviewers at named gates | Overleaf; local TeX preflight; Ebrón; defence materials | EUR 5-30 planned incremental | R-01; R-07; R-08 | Student and tutor; academic-closeout must be closed |

**Planned total: 330 hours.** This is the full-TFG allocation, not a daily forecast for the target
windows and not an actual-hours statement.

## Retrospective reconstruction and final forecast

The evidence-backed reconstruction is maintained in `effort-log.csv`. Its central estimate is
**326 hours completed by 1 September 2026** (plausible interval: 294-358 hours), plus **22 hours
remaining** for two review rounds, deposit closeout, and defence preparation. The resulting final
forecast is **348 hours**, 18 hours or 5.5% above the initial plan. The principal offsetting
deviations are:

- fewer hours than planned in Phase 3 because fine-tuning and extra model families were explicit
  deadline-driven NO-GO decisions;
- more hours in Phase 2 for degradation calibration and in Phase 4 for diagnosing the non-
  transferable v1 degradation, implementing staff-scale normalization, rerunning SMB, and
  repeating qualitative and quantitative analysis;
- a bounded Phase 5 reserve for sequential review: Jorge Calvo Zaragoza first at the technical-
  scientific level, then Elena Vázquez Barrachina at the structural and academic level.

This is a task-based retrospective estimate, not a recovered stopwatch log. The interval is
reported because commit timestamps, compute runtime, and automated-agent duration do not equal
student dedication. No exact daily attendance or unobserved cost is inferred.

## Execution lanes under D-16

1. **Phase 2 blocking scientific core:** the locally frozen objectives and questions,
   authenticated SMB audit, complete immutable manifest, quarantine enforcement, and minimal
   validated evidence formats. Main execution advances only when the relevant scientific gate is
   satisfied.
2. **Non-blocking SOTA and thesis enrichment:** literature refresh, critical synthesis, and stable
   thesis prose may continue across phases without idling the scientific core.
3. **Parallel `academic-closeout`:** ACAD-01 tutor compatibility, ACAD-02 private DELV-01 checks,
   and ACAD-03 authoritative Overleaf review remain human-owned. They do not block main Phases 1-4
   but must be closed before the Phase 5 deposit gate.

No lane may infer reviewer answers, private eligibility, contemporaneously measured hours, or
authoritative render status. A retrospective effort estimate must remain labelled as such.

## Schedule controls

- Deposit-ready internal target: **31 August 2026**.
- Contingency: **1-6 September 2026**, reserved for tutor feedback, invalidating defects, clean
  replay, final compliance, and deposit operations.
- Scope-expansion freeze: **3 September 2026**. Optional work cannot enter after this point.
- Hard deposit deadline: **7 September 2026**.
- Missing any critical-path checkpoint triggers a recorded replan and removal of optional scope
  before the contingency is consumed.
- **Provisional defence window:** 8-30 September 2026, or the first eligible slot after the deposit
  workflow. This is an internal planning window only; the current GCD/ETSINF call and Ebrón record
  must supply the authoritative date, presentation allocation, and logistics before use.

## Human resources

- **Student:** owns implementation, evidence capture, time entry, scientific reasoning, thesis
  drafting, and explicit acceptance of personal or signed claims.
- **Elena Vázquez Barrachina, tutor:** owns academic and structural guidance and the final formal
  review gates; no approval is inferred before her attributable response.
- **Jorge Calvo Zaragoza, experimental director:** proposed the subject and SMB, provided dataset
  access, and performs the first technical-scientific review; this is guidance and review, not
  authorship of the student's implementation or conclusions.
- **Independent reader or operator:** used only where later clean replay, claim review, or access
  conditions permit and the participation is actually evidenced.
- **Tribunal:** intended audience for the final thesis and defence; no assessment outcome is
  predicted here.

## Material resources and cost basis

| Resource | Planned use | Cost estimate | Evidence boundary |
| --- | --- | --- | --- |
| Existing local CPU workspace | Development, fixtures, audit, analysis, and local verification | EUR 0 acquisition; EUR 20-80 electricity/network planning allowance | Availability is evidenced by local execution; consumption remains unmeasured until logged |
| Kaggle accelerator environment | Bounded learned inference or training only when a frozen plan requires it | EUR 0 base plan; any paid compute requires a prior replan | Runtime, GPU, dependencies, seed, and exported artefacts must be captured per run |
| Existing Overleaf and TeX workflow | Thesis source, authorized local preflight, and later authoritative render | EUR 0 incremental in the base plan | Subscription or private account facts are not asserted |
| External storage and backup | Gated data cache and generated evidence outside Git | EUR 0-30 contingency | No raw data, weights, secrets, or large outputs enter Git |
| Printing and administrative incidentals | Only if the current live process requires them | EUR 0-50 provisional allowance | Current call and Ebrón instructions determine need |

The direct external planning range is **EUR 20-160**. It is estimated, not incurred. Student labour
is represented by the initial 330-hour plan and revised 348-hour forecast and is not assigned a
fabricated monetary rate. No actual cost is recorded in this file; actual expenditure requires
student evidence and a dated entry.

## Risk register

| Risk ID | Risk | Likelihood | Impact | Trigger | Mitigation or contingency | Owner | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-01 | Deadline compression exceeds truthful available effort | High | Critical | Actual log or critical-path forecast no longer fits 31 August | Replan immediately; remove optional work; preserve 1-6 September for invalidating fixes and deposit only | Student | Daily critical-path review and every phase boundary |
| R-02 | SMB access, content, grouping, suitability, or rights remain unresolved | Medium | Critical | Audit cannot reconcile every expected item or a rights/grouping disposition | Keep SMB locked; retain every failure; escalate only the material unresolved question | Student; human reviewer where required | P1 audit and Phase 4 unlock |
| R-03 | Tutor response changes plural-dataset interpretation | Medium | High | ACAD-01 returns a material change rather than compatibility | Keep ACAD-01 open; record decision; route change to main planning before affected experiments | Student and tutor | ACAD-01 before Phase 5 closeout |
| R-04 | Learned method, checkpoint, licence, or Kaggle compatibility is unsuitable | Medium | High | Phase 3 live validation fails | Prefer the smallest justified comparison; record exclusions; allow adaptation NO-GO | Student | Phase 3 method freeze |
| R-05 | Compute, storage, or runtime exceeds the bounded budget | Medium | High | Pilot exceeds frozen resource or time threshold | Reduce optional comparisons; use tiling only under a validated contract; do not consume deposit buffer | Student | Pre-run resource gate and each exported run |
| R-06 | Aggregate metrics conflict with notation-failure evidence | Medium | Critical | Systematic musical alteration appears despite favourable image scores | Withhold recommendation; retain negative cases; apply predeclared contradiction rules | Student; accountable interpretation reviewer | Phase 4 reconciliation and claim review |
| R-07 | Private eligibility, Ebrón access, or authoritative Overleaf state remains pending | Medium | Critical | Any ACAD-02 or ACAD-03 item is unresolved at Phase 5 closeout | Keep private proof outside Git; resolve through academic-closeout; fail deposit readiness while open | Student | Academic-closeout and Phase 5 pre-deposit gate |
| R-08 | Thesis claims, citations, figures, or defence logistics lack current evidence | Medium | High | Claim ledger is not reviewed or live call differs from planning assumptions | Reject unreviewed promotion; recheck current sources; keep a static defence fallback | Student and tutor at named reviews | Phase 5 evidence and defence gates |

## Checkpoint cadence

- At each phase boundary, compare planned hours and dates with the explicitly labelled
  retrospective estimate; never relabel reconstructed values as contemporaneously clocked time.
- Before any material protocol change, create a deviation record naming old and new versions,
  reason, affected controls and runs, evidence, and re-execution decision.
- Before every reported run, verify frozen identities, resources, expected denominator, and output
  location.
- Before promoting a thesis claim, require a schema-valid reviewed claim row with limitations,
  reviewer, and date.
- Before deposit, require all main scientific gates and the complete `academic-closeout` assertion.
