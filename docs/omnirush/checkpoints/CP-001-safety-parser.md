# CHECKPOINT CP-001 — Safety and Parser Software Gate

**Timestamp:** 2026-10-04  
**Git HEAD:** `302e1b214c2a23e9d4061d847eb4abfc19baebf7`  
**Phase:** VERIFY → CHECKPOINT → REASSESS  
**Status:** PARTIAL — software-only gate closed for reproduced defects; release remains NOT READY.

## Completed

- H1 dead-at-boot ON reproducer fixed at the actuation boundary.
- H2 safety-trip plus ON interleaving fixed: MQTT callback queues requests; Core 0 is the sole runtime relay writer and safety inhibit is evaluated before ON.
- Live parser rejects empty/missing/negative/non-finite/nonnumeric values before mutating device state.
- API MQTT bridge isolates malformed UTF-8/schema/type frames without terminating the listener.
- Added firmware-structure regression for callback ownership and Core-0 health/inhibit/lockout gating.
- Reclassified old dry-relay/never-valid PZEM tests to the production fail-closed contract; isolated maintenance continuity is not reachable through production ON.

## Tests

| Command | Result |
|---|---|
| Focused safety/protocol/API gate | **163 passed** |
| `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short` | **646 passed, 3 warnings** |
| `cd frontend && npm test -- --run` | **20 passed** |
| `git diff --check` | PASS |
| PlatformIO/`pio` compile | **NOT RUN — tool unavailable** |

## Evidence boundary

H1, H2, parser, and B2 are **SOFTWARE REGRESSION PASS / SIMULATED ONLY**. No mains, relay contact, PZEM UART, brownout, EMI, calibration, or physical ML evidence was produced. Firmware source/static checks and the simulator are not hardware proof.

## Remaining blockers

- H3 transaction freshness/liveness and measured blind-time bound.
- H4 prototype-versus-certified safety claim boundary.
- H5 authoritative hardware/projector contradiction and actual inventory.
- B1 authenticated real broker/ACL/replay behavior.
- M1/M2/M3 physical enrollment, held-out, unknown, and overlap evidence.
- U1/U2 dashboard provenance/stale/live truth.
- G11 soak/capacity evidence.

## Assumptions and next action

Thresholds were not changed. Physical validation remains `NOT RUN`. Next action is human review of the CP-001 firmware safety diff and qualified low-voltage bench preparation; no mains claim is made from this checkpoint.
