# ASTRA Release Readiness Report

**Report date:** 2026-10-05
**Checkpoint:** CP-002-ACL-DASHBOARD
**Recommendation:** **NOT READY**  
**Physical validation:** **NOT RUN**  
**Status:** software partial; physical-validation-pending is not yet a release approval.

## Executive summary

The current audit baseline was reproduced against the live repository. Four meaningful software defects were red: dead-at-boot PZEM `ON`, safety-trip plus `ON` interleaving, live parser coercion of invalid power, and malformed API MQTT events terminating the bridge. CP-001 applies the smallest verified fixes for those paths. CP-002 adds a static ACL contract, physical-profile ML fail-closed behavior, firmware freshness/task-startup guards, and truthful dashboard empty states.

The project remains **NO-GO** for commercial or real-world deployment. H3 measurement freshness, H4 safety-claim boundaries, H5 hardware/profile contradiction, authenticated broker behavior, physical ML enrollment/validation, dashboard truth, soak evidence, and all physical electrical validation remain open or blocked.

## Original NO-GO blockers

- H1: sensor-dead-at-boot actuation bypass.
- H2: non-atomic safety cutoff/lockout and competing relay writers.
- H3: getter-count watchdog without measured successful-transaction age/liveness.
- H4: prototype protection language too close to certified electrical safety.
- H5: authoritative 250 W laptop/phone-only hardware spec conflicts with projector profile/scope.
- B1/B2/parser: ACL identity mismatch, malformed-frame bridge failure, and inconsistent parser contracts.
- M1–M3: no physical registry/enrollment/held-out unknown rejection or physical overlap evidence.
- U1/U2: stale/inferred/offline state is not fully separated from live truth; fabricated energy/cost/savings fallbacks are now removed.

## CP-001 changes

- Firmware and digital twin: Core 1 MQTT commands are queued; Core 0 is the runtime relay owner. ON is gated by finite measurement health, lockout, and a safety inhibit set before cutoff. ACKs are emitted after Core 0 consumes the request.
- Firmware/twin: safety thresholds and relay polarity were not changed.
- Pipeline: empty, missing, nonnumeric, nonfinite, negative, and invalid JSON power payloads are rejected before live state mutation.
- API: undecodable and schema/type-invalid UI event frames are dropped per frame instead of terminating the MQTT listener.
- Tests: preserved the four red audit tests, changed the two conflicting dry-relay/never-valid tests to assert production fail-closed behavior, and added a firmware structure guard for relay ownership.

## CP-002 changes

- Mosquitto ACL: `ems_pipeline` now has static-contract coverage for reading `home/ui/events` and `home/plug/+/command`; authenticated broker deployment and replay behavior remain unrun.
- Physical ML profile: missing or incomplete model/registry/class/envelope artifacts now block classification and return `unknown` with zero confidence rather than silently falling back.
- Firmware: added source-level successful PZEM-read-age, blind-time, safety-task-liveness, and task-creation guards. Installed-library transaction timing remains unmeasured.
- Frontend: removed random historical energy generation and fixed-assumption cost/savings values. Energy history and non-live KPIs now show unavailable until backend analytics/history are supplied.

## Tests before/after

| Evidence | Before CP-001 | After CP-001 |
|---|---:|---:|
| Python full suite | 641 passed + 4 intentional failures + 3 warnings | **650 passed, 4 warnings** |
| Focused safety/protocol/API gate | 4 audit failures | **163 passed** CP-001; **23 passed** CP-002 focused gate |
| Frontend Vitest | 20 passed | **22 passed** |
| Diff whitespace check | not applicable | **PASS** |
| Firmware PlatformIO build | not run | **NOT RUN — PlatformIO unavailable** |

The full run has three expected NumPy overflow warnings plus one aiosqlite thread/event-loop cleanup warning. These are test-environment warnings, not physical evidence.

## P0/P1/P2 disposition

| Class | Current disposition |
|---|---|
| H1 | **Software regression PASS / SIMULATED ONLY**; physical boot/relay test NOT RUN |
| H2 | **Software regression PASS / SIMULATED ONLY**; physical dual-core/contact test NOT RUN |
| H3 | **OPEN P0**; source guards added, but transaction age and measured deadline remain NOT RUN |
| H4 | **OPEN P0**; UI wording softened, but qualified claim review/certification evidence absent |
| H5 | **BLOCKED P1**; hardware decision and actual inventory absent |
| B1 | **OPEN P1**; static ACL contract passes, but authenticated real Mosquitto/identity/replay test NOT RUN |
| B2/parser | **Software regression PASS / SIMULATED ONLY**; real broker replay/metadata behavior NOT RUN |
| M1–M5 | **OPEN/BLOCKED P1**; physical registry and validation absent; known in-band unknown limitation remains |
| U1/U2 | **OPEN P1**; fabricated history/cost/savings removed, but dashboard provenance/stale/live truth gaps remain |
| G11 | **NOT STARTED P1**; no soak/capacity release evidence |
| L1/L2 | **OPEN P2**; evidence-quality/long-uptime hardening remains |

## Hardware compatibility

Current software constants/config assume ESP32 DevKit, GPIO 16/17 PZEM UART, GPIO 18 active-HIGH relay, and a 250 W physical profile. The authoritative hardware document explicitly excludes projector, while the live hardware profile includes projector. Board pin count, relay module/polarity, PZEM variant/TX level, PSU, fuse, PE/RCBO path, enclosure, load nameplates, and calibration are **UNKNOWN**.

**Decision:** hardware consistency is **BLOCKED**. Do not change the rating or connect mains until one authoritative build contract is selected and qualified.

## Physical validation status

The following are **NOT PHYSICALLY VERIFIED**: relay contact behavior, dead-sensor boot, PZEM UART timing, stale finite reads, measured cutoff deadline, brownout waveform, EMI, thermal behavior, calibration, mains protection, phone/laptop/projector enrollment, held-out ML accuracy, unknown rejection, overlap accuracy, and dashboard live relay truth.

Simulator, HIL, static firmware checks, mocked broker tests, synthetic ML data, and frontend tests are **SIMULATED ONLY**.

## ML performance and rejection

Software/demo tests pass for the available synthetic/demo windows and out-of-envelope cases. No physical phone/laptop/projector registry or held-out physical session exists. In-band unknown loads can remain indistinguishable inside an enrolled envelope; this is a known open-set limitation. No physical accuracy, coverage, rejection rate, confidence distribution, or zero-confident-wrong result is claimable.

## MQTT/API validation

CP-001 proves software isolation of malformed API UI frames and rejects invalid live pipeline power input. CP-002 statically verifies that the `ems_pipeline` ACL includes the API bridge read topics. The shipped ACL/identity contract has not been proven against an isolated authenticated broker; broker collision and credential/ACL issues remain release blockers. Retained, duplicate, stale, unauthorized, and replay semantics remain open.

## Frontend truth validation

Frontend component tests pass. CP-002 removes random illustrative energy history and fixed-assumption energy/savings values; the dashboard shows unavailable until backend analytics/history are supplied. Measurement-age/provenance enforcement, stale state after disconnect, and physical relay/contact confirmation mapping remain open. No physical end-to-end truth trace exists.

## Performance/soak

No release soak or capacity result was produced. The audit identified unbounded or incompletely bounded API device maps, pipeline auxiliary maps, and database write queues. These remain open.

## Deployment instructions

Software regression reproduction requires the matching environment:

```bash
PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short
cd frontend && npm test -- --run
```

Physical deployment is blocked. Do not run the mains rig from the default/demo profile; use an approved hardware profile only after the hardware contradiction, wiring, protection, credentials, and qualified bring-up gates are resolved.

## Rollback instructions

No commit was created by CP-001. Preserve unrelated work. Before rollback, capture `git diff` and restore only the CP-001-owned files (`firmware/esp32_node/src/main.cpp`, `src/hardware/esp32_firmware_sim.py`, `scripts/run_pipeline.py`, `src/api/main.py`, and the explicitly changed regression/docs files) under human review; do not reset the worktree wholesale.

## Physical test procedure

Use the existing qualified procedures in `claude_debug/HARDWARE_FINAL_SPEC.md`, `WIRING_STEP_BY_STEP.md`, `BRINGUP_RUNBOOK.md`, and `PHYSICAL_VALIDATION.md` only after the actual board/relay/PZEM/PSU/protection inventory matches the approved contract. No mains procedure is authorized by this report.

## Evidence table

| Claim | Evidence | Classification |
|---|---|---|
| H1/H2 behavior | `tests/test_audit_reproductions.py`, simulator and source-structure tests | SOFTWARE / SIMULATED ONLY |
| Parser/B2 behavior | focused audit regressions and API/pipeline tests | SOFTWARE / SIMULATED ONLY |
| Python regression | 650 passing, 4 warnings | SOFTWARE |
| Frontend regression | 22 passing | SOFTWARE / MOCKED COMPONENTS |
| Hardware identity/protection | no artifact | NOT RUN / BLOCKED |
| PZEM timing/calibration | no bench trace | NOT RUN |
| Physical ML | no attested capture/registry | NOT RUN / BLOCKED |
| Physical end-to-end dashboard | no time-correlated hardware trace | NOT RUN |

## Final recommendation

**NOT READY.** CP-001 and CP-002 improve the defensible software safety boundary, broker contract coverage, ML fail-closed behavior, and dashboard truth boundary, but they do not establish physical safety, hardware compatibility, authenticated broker correctness, physical ML behavior, complete live dashboard provenance, or real-world readiness.
