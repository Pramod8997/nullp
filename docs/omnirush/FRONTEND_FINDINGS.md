# Frontend Truthfulness Findings — Agent 6 Consolidation

**Date:** 2026-10-04  
**Physical validation:** **NOT RUN**  
**Evidence:** source audit and 22 component tests; all software evidence is simulated/mock-only.

## Blocking findings

| ID | Severity | Finding | Evidence |
|---|---:|---|---|
| U1 | P1 | Energy history must come from measured backend history | CP-002 removed `Math.random()` and shows unavailable until `analytics.energy_history` is supplied; backend history contract remains absent |
| U2 | P1 | Summary energy/cost/savings require backend analytics | CP-002 removed fixed durations, fallback ₹8/kWh, savings, and comparison percentages; unavailable state is shown until backend summary fields exist |
| U3 | P1 | Appliance energy is fabricated from one sample | `ApplianceTable.jsx:70-75` computes `(power / 1000) * 0.5` rather than persisted measurements |
| U4 | P1 | Live state has no measurement-age/provenance contract | `App.jsx:67-130` uses browser receipt times, ignores source/batch timestamp, and has no stale expiry |
| U5 | P1 | WebSocket connection is presented as live hardware | `App.jsx:420-445` shows `LIVE 1Hz Stream` while old device values survive disconnect |
| U6 | P1 | Inferred/unknown/rejected states are not distinct in rendered device state | `App.jsx`, `DeviceCards.jsx`, and pipeline/API schemas omit source, freshness, inferred/rejected, and relay confirmation fields |
| U7 | P1 | Safety trip is inferred from alert text/timer, not physical relay state | `AlertsPage.jsx:13-31` derives breaker status from five-minute alert timing; no edge relay state/contact feedback is consumed |
| U8 | P1 | ACK is not correlated to command or physical state | Firmware/pipeline emit `ON_CONFIRMED`/`OFF_CONFIRMED` and `HARDWARE_ACK`, but `App.jsx` does not map ACK to requested action, command ID, or relay state |
| U9 | P1 | Reconnect does not resynchronize truth | `App.jsx:260-313` preserves old devices/charts/alerts; API init state omits freshness, relay/ACK state, and provenance |
| U10 | P1 | Frontend tests do not exercise the full truth boundary | The 22 tests cover the new empty states but do not render `App` WebSocket state transitions |

## Required software invariants

- Every displayed measurement must carry source timestamp/age and provenance (`LIVE HARDWARE`, `SIMULATION`, `STALE`, `OFFLINE`, `INFERRED`, `UNKNOWN`, or `REJECTED`).
- A disconnected or expired device must not remain visually active/live.
- Safety-originated status must be distinguishable from a pipeline inference; a command ACK is not contact confirmation.
- Rejected commands and parser events must not appear as successful actuation.
- Energy, cost, and savings must come from a timestamped backend analytics source or be visibly labeled illustrative and excluded from production metrics.
- Reconnect must request/receive a state snapshot that includes freshness, provenance, relay state, safety lockout, and latest ACK correlation.

## Missing tests

Add a focused `App` WebSocket test with fixtures for live, simulated, stale, offline, inferred, unknown, rejected, and safety-tripped state; assert rendered labels and transitions. Add reconnect and ACK-correlation tests. These tests remain software-only until a live physical telemetry trace is captured.

## Disposition

Frontend truth gate: **OPEN/NOT READY**; CP-002 removed fabricated energy/cost/savings fallbacks.
Physical live-dashboard validation: **NOT RUN**.
