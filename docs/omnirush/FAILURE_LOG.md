# ASTRA Failure Log

## Entry format

Each failure gets a stable ID, severity, subsystem, observed behavior, minimal reproduction, root cause, regression test, fix, verification, and hardware status. Do not close an entry from a documentation claim alone.

## Imported audit findings awaiting live reproduction

| ID | Severity | Subsystem | Observed / reported | Current status |
|---|---:|---|---|---|
| FAIL-H1 | P0 | Firmware safety | Audit reports dead PZEM at boot can leave relay energized because loss watchdog is gated by `pzemEverValid` while ON checks only lock state | OPEN; reproduce against current dirty tree |
| FAIL-H2 | P0 | Firmware concurrency | Audit reports safety trip opens on Core 0, then Core 1 applies lockout; an ON between them can re-close | OPEN; reproduce interleaving |
| FAIL-H3 | P0 | Firmware/PZEM | Audit reports getter-count watchdog and library timeout can make claimed 3 s cutoff about 15 s | OPEN; trace installed library/runtime path |
| FAIL-H4 | P0 | Hardware/UI claims | Prototype active-power threshold/dP/dt described too close to commercial overcurrent/AFCI/electrical safety | OPEN; audit claims and update decision |
| FAIL-H5 | P1 | Hardware/config/docs | 250 W laptop/phone-only authoritative spec conflicts with projector class/config/plan | OPEN; human hardware decision required |
| FAIL-B1 | P1 | MQTT/ACL | Audit reports API identity cannot read required UI-event topic under shipped ACL | OPEN; real broker ACL test |
| FAIL-B2 | P1 | API MQTT bridge | Audit reports schema/type/UTF-8 errors escape frame handler and kill bridge while health says connected | OPEN; minimal malformed-frame reproducer |
| FAIL-PARSER | P1 | Parser | Audit reports empty/{} convert to 0 W and negative values survive live parser | OPEN; parser/data-flow trace |
| FAIL-M1 | P1 | ML deployment | Hardware profile has no enrolled physical registry/envelopes; partial checkpoint path may enable incomplete model | OPEN/BLOCKED on physical capture |
| FAIL-M2 | P1 | ML open set | Audit reports survivor renormalization can make in-band unknowns confidence 1.0 | OPEN; current classifier path trace |
| FAIL-M3 | P1 | ML disaggregation | Audit reports one label per aggregate node rather than maintained active set | OPEN; detector-path e2e |
| FAIL-U1 | P1 | Frontend truth | Audit reports random history/hardcoded savings and stale/inferred state can appear live | OPEN; component/data provenance audit |

## Working-tree safety note

The current uncommitted changes add firmware/simulator PZEM watchdog and overcurrent-latch logic plus extensive tests. The added simulator tests explicitly preserve a dead-at-boot `ON` path when `_pzem_ever_valid` is false; that is a candidate H1 violation, not evidence that H1 is resolved. No uncommitted file is to be discarded or reset without user authorization.

## Baseline reproducer evidence

`tests/test_audit_reproductions.py` was added before remediation and run with the matching Python 3.10 environment. It fails 4/4 as expected: FAIL-H1 relay remains ON after dead-at-boot PZEM; FAIL-H2 relay re-closes in the Core 0/Core 1 window; FAIL-PARSER direct handler accepts empty/object/negative input; FAIL-B2 Pydantic `ValidationError` escapes the MQTT bridge. These failures remain OPEN.

## Reproduction rule

On each failure: capture the command and smallest failing test first; then assign one root-cause hypothesis; write/confirm a failing regression; apply the smallest safe fix; run focused, neighboring, and full regression; adversarially review; checkpoint.
