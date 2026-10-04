# ASTRA Physical Validation Ledger

**Overall:** `NOT RUN`  
**Never mark this file PASS from simulator, HIL, unit, or static-analysis results.**

| Gate | Procedure/evidence required | Status | Evidence artifact |
|---|---|---|---|
| Board identity/pinout | board marking/photo, continuity map | NOT RUN | none |
| Relay safe boot | qualified low-voltage/open-contact verification before mains | NOT RUN | none |
| PZEM UART/levels | measured TX level, valid Modbus reads, timeout trace | NOT RUN | none |
| Dead sensor boot | sensor disconnected/dead with safe isolated setup; relay remains open | NOT RUN | none |
| Wi-Fi/broker loss | edge cutoff/lockout behavior with network unavailable | NOT RUN | none |
| Brownout/recovery | instrumented rail and reset waveform | NOT RUN | none |
| EMI/relay/inrush | safe representative loads, scope/current instrumentation | NOT RUN | none |
| Calibration | reference meter versus PZEM across loads/voltage | NOT RUN | none |
| Phone enrollment | separate capture and held-out validation | NOT RUN | none |
| Laptop enrollment | separate capture and held-out validation | NOT RUN | none |
| Projector enrollment | only if approved hardware ceiling/nameplate supports it | BLOCKED | hardware decision required |
| Unknown rejection | in-band and out-of-envelope unknown load rehearsal | NOT RUN | none |
| Overlap/delta | ordered pair event capture and ≥90% added-load gate if shipped | NOT RUN | none |
| Dashboard live truth | physical trip/offline/stale/ack observed end-to-end | NOT RUN | none |

## Safety boundary

No mains procedure is issued from this ledger while the board, relay, PZEM variant, power supply, fuse/RCBO/PE path, enclosure, and qualified operator are unknown or contradictory. Use the existing wiring/runbook only after its assumptions match the actual build.
