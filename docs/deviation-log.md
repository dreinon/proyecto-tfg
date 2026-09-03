# Protocol deviation log

This append-only control records enacted departures from a versioned analysis protocol. It does
not treat a possible external outcome as a deviation before that outcome exists.

## Enacted deviations

| Record ID | Date | Owner | Status | Subject | Old rule | Replacement rule | Evidence-backed reason | Affected controls/runs | Re-execution and closure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV-SCI-01 | 2026-08-30 | Student | closed_reexecuted | Correct non-transferable absolute-pixel SMB degradation | `controlled-score-v1` applied fixed blur values calibrated on a large synthetic fixture; the initial CSV also treated page-specific labels as independent source groups | `staff-scale-score-v2` uses input-only staff-relative blur and a fresh 64-page/64-work sample excluding every v1 work | The v1 full-page review showed moderate/strong severities did not reproduce their intended roles; authoritative-manifest reconciliation found 53 true works among the 64 v1 pages | Preserve `smb-pretrained-evaluation-v1` as development/stress evidence; final claims move to `smb-pretrained-evaluation-v2`; methods, x2/x4 scales, and metric definitions remain unchanged | Closed 2026-08-31: v2 reconciled 1152/1152 unique rows, 384 traces, 64/64-work identity preflight, 30 qualitative PNGs, finite metrics, output hashes, and archive SHA-256 `18d904eda110c37cffb29674d04b11947bc49c5c37710fd9d74728cd185e499c` |
| DEV-PROF-01 | 2026-09-03 | Student | closed_reconciled | Admit one applicability extension at the scope-freeze boundary | Optional application and external-transfer work remained deferred until the core and complete thesis candidate were secure | Implement an image-only local demonstrator and a predeclared 12-work external pilot in parallel with joint review; do not create visible thesis placeholders or delay deposit | The complete SMB study, bounded EDSR adaptation, compiled thesis candidate, and pushed review revision already existed; external pages were available under SJMA processing authorization and the extension remained independently removable | Existing SMB and adaptation runs remain unchanged; new scope is `professional-pilot-v1`; PDF, deployment, OMR, model retuning, public page reproduction, and new families remain NO-GO | Closed 2026-09-03: 216/216 outputs, 12/12 reviewed cases, 70-file final manifest, reviewed claim rows, aggregate thesis figure, and coverless 62-page local render reconcile; authoritative Overleaf review remains an academic closeout gate, not part of this scientific deviation |

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
