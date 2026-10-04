# ASTRA Failure Log

## Entry format

Each failure gets a stable ID, severity, subsystem, observed behavior, minimal reproduction, root cause, regression test, fix, verification, and hardware status. Do not close an entry from a documentation claim alone.

## Imported audit findings awaiting live reproduction

| ID | Severity | Subsystem | Observed / reported | Current status |
|---|---:|---|---|---|
| FAIL-H1 | P0 | Firmware safety | Audit reported dead PZEM at boot could leave relay energized because loss watchdog was gated by `pzemEverValid` while ON checked only lock state | SOFTWARE RESOLVED / PHYSICAL NOT RUN; CP-001 gate rejects ON |
| FAIL-H2 | P0 | Firmware concurrency | Audit reported safety trip opened on Core 0, then Core 1 applied lockout; an ON between them could re-close | SOFTWARE RESOLVED / PHYSICAL NOT RUN; CP-001 queue/inhibit interleaving passes |
| FAIL-H3 | P0 | Firmware/PZEM | Audit reports getter-count watchdog and library timeout can make claimed 3 s cutoff about 15 s | OPEN; trace installed library/runtime path |
| FAIL-H4 | P0 | Hardware/UI claims | Prototype active-power threshold/dP/dt described too close to commercial overcurrent/AFCI/electrical safety | OPEN; audit claims and update decision |
| FAIL-H5 | P1 | Hardware/config/docs | 250 W laptop/phone-only authoritative spec conflicts with projector class/config/plan | OPEN; human hardware decision required |
| FAIL-B1 | P1 | MQTT/ACL | Audit reports API identity cannot read required UI-event topic under shipped ACL | OPEN; real broker ACL test |
| FAIL-B2 | P1 | API MQTT bridge | Audit reported schema/type/UTF-8 errors escaped frame handler and killed bridge while health said connected | SOFTWARE RESOLVED / REAL BROKER NOT RUN; per-frame isolation passes |
| FAIL-PARSER | P1 | Parser | Audit reported empty/{} converted to 0 W and negative values survived live parser | SOFTWARE RESOLVED / REAL BROKER NOT RUN; state-integrity regression passes |
| FAIL-M1 | P1 | ML deployment | Hardware profile has no enrolled physical registry/envelopes; partial checkpoint path may enable incomplete model | OPEN/BLOCKED on physical capture |
| FAIL-M2 | P1 | ML open set | Audit reports survivor renormalization can make in-band unknowns confidence 1.0 | OPEN; current classifier path trace |
| FAIL-M3 | P1 | ML disaggregation | Audit reports one label per aggregate node rather than maintained active set | OPEN; detector-path e2e |
| FAIL-U1 | P1 | Frontend truth | Audit reports random history/hardcoded savings and stale/inferred state can appear live | OPEN; component/data provenance audit |

## Working-tree safety note

The current uncommitted CP-001 changes add the Core-0 command queue/health gate, parser/API validation fixes, and regression assertions. No uncommitted file is to be discarded or reset without user authorization.

## Baseline reproducer evidence

`tests/test_audit_reproductions.py` first failed 4/4 before remediation. After CP-001 it passes 4/4, and the focused neighboring gate passes 163 tests. These are software results only; physical validation remains NOT RUN.

## Reproduction rule

On each failure: capture the command and smallest failing test first; then assign one root-cause hypothesis; write/confirm a failing regression; apply the smallest safe fix; run focused, neighboring, and full regression; adversarially review; checkpoint.
