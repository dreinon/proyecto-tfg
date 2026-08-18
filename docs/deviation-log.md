# Protocol deviation log

This append-only control records enacted departures from a versioned analysis protocol. It does
not treat a possible external outcome as a deviation before that outcome exists.

**No enacted deviations are recorded as of 2026-08-18.** The rows below are pending external
triggers, not claims that the protocol, scope, schedule, or runs have changed.

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
