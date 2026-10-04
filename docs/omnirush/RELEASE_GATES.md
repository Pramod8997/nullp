# ASTRA Release Gates

| Gate | Required evidence | Current status | Blocking reason |
|---|---|---|---|
| 1. Reproducible baseline | versions, config, model/registry, exact test counts | PARTIAL | CP-001: 646 Python + 20 frontend pass; default venv command is broken |
| 2. H1 boot safety | dead/timeout/malformed/stale PZEM + ON/retained/reconnect tests | SIMULATED ONLY | CP-001 software gate passes; firmware compile and physical boot/relay test not run |
| 3. H2 atomic actuation | single relay owner and race tests | SIMULATED ONLY | CP-001 queue/inhibit/static owner gates pass; physical dual-core/contact test not run |
| 4. H3 freshness | successful-read timestamps and bounded measured blind interval | OPEN | current library/runtime timing unresolved |
| 5. H4 claim boundary | explicit prototype-only claims; no unsupported certification | OPEN | contradictory docs/UI wording |
| 6. Hardware consistency | one approved BOM/wiring/config/firmware/profile | BLOCKED | projector versus 250 W spec conflict |
| 7. Real broker/API | authenticated Mosquitto, ACL, malformed/replay/duplicate/stale/reconnect tests | OPEN | parser/bridge software gate passes; real deployment/ACL not run |
| 8. Power/EMI bench | rails, Wi-Fi, relay, PZEM UART, sag, inrush, thermal, calibration | NOT RUN | equipment/qualified procedure not evidenced |
| 9. ML physical validation | separate enrollment/validation, held-out metrics, unknown rejection | BLOCKED | no physical captures/registry |
| 10. End-to-end truth | live/sim/stale/offline/inferred/unknown/rejected/tripped UI states | OPEN | dashboard provenance audit incomplete |
| 11. Soak/capacity | bounded memory/storage, reconnect/rollover evidence | NOT STARTED | tests not run |
| 12. Final audit | P0=0, P1 disposition, deployment/rollback, adversarial review | NOT STARTED | dependent gates open |

## Status policy

- Software tests may prove `PASS` for a software invariant only.
- Simulator/HIL harness results are `SIMULATED ONLY`.
- Physical result requires raw physical evidence and is never inferred from a passing test.
- Missing equipment or unsafe/ambiguous wiring is `BLOCKED — HARDWARE/EQUIPMENT REQUIRED`.
- Current release recommendation: `NOT READY` / `SOFTWARE-READY / PHYSICAL-VALIDATION-PENDING` is not yet earned because P0 software gates remain open.
