# Protocol deviation log

This append-only control records enacted departures from a versioned analysis protocol. It does
not treat a possible external outcome as a deviation before that outcome exists.

## Enacted deviations

| Record ID | Date | Owner | Status | Subject | Old rule | Replacement rule | Evidence-backed reason | Affected controls/runs | Re-execution and closure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV-SCI-01 | 2026-08-30 | Student | enacted_pending_rerun | Correct non-transferable absolute-pixel SMB degradation | `controlled-score-v1` applied fixed blur values calibrated on a large synthetic fixture; the initial CSV also treated page-specific labels as independent source groups | `staff-scale-score-v2` uses input-only staff-relative blur and a fresh 64-page/64-work sample excluding every v1 work | The v1 full-page review showed moderate/strong severities did not reproduce their intended roles; authoritative-manifest reconciliation found 53 true works among the 64 v1 pages | Preserve `smb-pretrained-evaluation-v1` as development/stress evidence; final claims move to `smb-pretrained-evaluation-v2`; methods, x2/x4 scales, and metric definitions remain unchanged | Full v2 Kaggle re-execution required; closes only after its 1152 rows, 384 traces, 64-work preflight, and archive reconcile |

The rows below are pending external triggers, not claims that those academic events have occurred.

## Pending deviation triggers

| Record ID | Date | Owner | Status | Subject | Rationale | Evidence reference | Affected controls/runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEV-ACAD-01 | 2026-08-18 | Student and tutor | pending_external_trigger | Tutor response may require a material dataset-scope change | A non-compatible or conditional response would change scope and must be evaluated before it becomes a protocol deviation | external-ref:ACAD-01-TUTOR-COMPATIBILITY | analysis-v1.0.0; CTRL-05; affected runs none because no change is enacted |
| DEV-ACAD-02 | 2026-08-18 | Student | pending_external_trigger | Live academic checks may invalidate a delivery assumption | A failed or inaccessible private check may require a schedule or delivery replan but is not currently evidence of one | external-ref:ACAD-02-PRIVATE-ELIGIBILITY | work-plan-v1; DELV-01; affected runs none |

## Required enacted record

When a trigger becomes an actual deviation, append a separate record containing all of the
following before affected interpretation or execution continues:

- stable deviation ID and date;
- accountable owner and review status;
- old protocol/control version and proposed new version;
- exact old rule and exact replacement rule;
- evidence-backed reason and safe evidence reference;
- affected requirements, configs, manifests, claims, and run/execution IDs;
- whether prior evidence remains comparable;
- explicit re-execution decision and its owner;
- closure evidence and date.

An empty affected-run set must be written as `none`, not omitted. A change that affects an already
reported run cannot close until re-execution or a justified invalidation is recorded.
