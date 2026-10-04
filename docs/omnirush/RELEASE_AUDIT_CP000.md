# ASTRA Release Audit — CP-000

**Audit date:** 2026-10-04  
**Auditor:** Agent 9 — Release Auditor  
**Checkpoint:** CP-000-BASELINE  
**Verdict:** **NO-GO / RELEASE NOT APPROVED**  
**Physical validation:** **NOT RUN**

This is a release-audit checkpoint, not a release approval. The live source and
live test results take precedence over status prose, historical close-out text,
and the supplied `Untitled Document 1`.

## 1. Checkpoint approval verdict

Release is **not approved**. H1, H2, and H4 remain open P0 safety/release
blockers. H3 is also an open P0 gate, but its often-repeated approximately
15-second blind-time figure is only a code-derived estimate, not a measured
result. P1 software, protocol, ML, dashboard, deployment, and hardware-contract
gates remain open or blocked. No physical gate can be promoted from `NOT RUN`.

The supplied audit's commercial-deployment `NO-GO` conclusion is therefore
reconciled as correct, with the evidence qualifications below.

## 2. Live baseline and metadata reconciliation

### Exact commands and results

| Check | Exact command | Live result |
|---|---|---|
| Default Python invocation | `venv/bin/python -m pytest tests/ -q` | **BLOCKED**: venv Python is 3.12.3 and cannot import the populated pytest installation |
| Matching Python baseline | `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short` | **641 passed, 4 failed, 3 warnings, 21.58 s** |
| Frontend baseline | `npm test -- --run` from `frontend/` | **20 passed, 1 file, 2.05 s** |
| Safety/static/HIL neighboring tests | `... -m pytest tests/test_relay_safety_boot_brownout.py tests/test_hardware_alignment.py tests/test_hil_uart_corruption.py -q --tb=short` | **100 passed, 0 failed, 0.66 s**; simulator/static evidence only |
| API/MQTT neighboring tests | `... -m pytest tests/test_mqtt.py tests/test_api.py tests/test_api_extended.py -q --tb=short` | **40 passed, 0 failed, 4.89 s**; mock/API evidence only |
| ML/detector neighboring tests | `... -m pytest tests/test_ml_pipeline_recognition.py tests/test_detector_path_e2e.py tests/test_overlap_delta.py tests/test_real_data_and_ml_fallback.py tests/test_e2e_five_class_recognition.py -q --tb=short` | **128 passed, 0 failed, 5.79 s**; demo/synthetic software evidence only |
| Database/security neighboring tests | `... -m pytest tests/test_database.py tests/test_security_penetration.py -q --tb=short` | **55 passed, 0 failed, 1.77 s**; mock/security software evidence only |
| Compose syntax | `docker compose config --quiet` | Exit **0**; Docker warns that the `version` attribute is obsolete |
| Graph query | `graphify query "release safety firmware PZEM MQTT parser recognition dashboard evidence"` | **BLOCKED**: installed launcher raises `ModuleNotFoundError: No module named 'graphify'` |

Runtime used for Python verification: Python 3.10.12, pytest 9.0.3, torch
2.11.0+cu130, numpy 2.2.6, FastAPI 0.136.0. Frontend runtime: Node
24.14.0, npm 11.9.0, Vitest 4.1.10.

### Repository-state corrections

- Live `HEAD` is `302e1b21` (`Updated Docs`), not the `4e07b658` recorded in
  `CURRENT_STATE.md` and `MASTER_CHECKLIST.md`. The current HEAD contains the
  four domain finding documents; the audit added no source, test, or
  configuration changes.
- At final verification, `git status --short` showed only this new audit
  artifact. No tracked source, test, or configuration diff was present.
- `Untitled Document 1` and `__agent__/` are in the live `HEAD`; they are not
  untracked in the current checkout. The corresponding claims in
  `CURRENT_STATE.md` are stale.
- The phrase “641 Python tests passed” is incomplete when used as the baseline:
  the live full run has 641 passing tests **and four failing audit tests**.
  The four failures are intentional red release reproductions, not a clean
  641-test pass.

## 3. Exact red reproducer results

Command prefix for the focused tests:

```text
PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest
```

| ID | Severity | Exact live result | Reconciled disposition |
|---|---:|---|---|
| H1 | P0 | `test_dead_pzem_at_boot_cannot_accept_on_command` **FAIL**. After `ON` and 60 invalid Core 0 samples, `gpio18_relay_state` is `True`; `_pzem_ever_valid` remains false and no PZEM fault latch is raised. | **OPEN / CONFIRMED in simulator and source.** `main.cpp` gates the watchdog on `pzemEverValid` while the command path checks only `relayLocked` (`main.cpp:263-283,396-405`). No physical confirmation. |
| H2 | P0 | `test_safety_cutoff_wins_over_on_before_core1_latch_tick` **FAIL**. Core 0 opens the relay and raises the latch; an intervening `ON` closes it; Core 1 then locks it while final relay state remains `True`. | **OPEN / CONFIRMED in simulator and source.** `setRelay()` remains callable from both cores; latch consumption is later than the vulnerable command window. No physical dual-core or relay-contact trace. |
| H3 | P0 | No current failing test measures the production PZEM elapsed blind interval. The installed library statically shows `UPDATE_TIME=200 ms`, `READ_TIMEOUT=100 ms`, and `_lastRead` updated only after a successful block read. | **OPEN / UNMEASURED.** The approximately 15-second value in the supplied audit is an estimate, not a bench result. The firmware's 30-cycle/3-second comment is also not an elapsed-time proof. |
| Parser | P1 | Direct `_handle_mqtt_message` probe stores `""` as `0.0`, `{}` as `0.0`, `-3` as `-3.0`, and `{"watts":5}` as `5.0`; `NaN` and `Infinity` are rejected. The stricter `process_raw_mqtt()` helper has a different contract. | **OPEN / CONFIRMED.** Production broker delivery uses `_handle_mqtt_message`, not the stricter helper. |
| B2 | P1 | `test_malformed_ui_event_does_not_crash_mqtt_bridge` **FAIL**. A schema/type-invalid `DEVICE_STATUS` frame raises Pydantic `ValidationError` out of `mqtt_listener_task`. | **OPEN / CONFIRMED in API source and test.** `UnicodeDecodeError`, wrong top-level JSON shapes, and other validation/runtime errors are outside the current per-frame catch. `/ready` checks object presence, not listener liveness. |

The full-suite result is consequently **641 passed, 4 failed, 3 warnings**;
the four failures are H1, H2, Parser, and B2.

## 4. P0 disposition

| Finding | Disposition | Evidence boundary |
|---|---|---|
| H1 dead-at-boot sensor gate | **P0 OPEN; blocks release** | Deterministic digital-twin failure plus firmware source trace; physical boot/relay test not run |
| H2 atomic safety actuation | **P0 OPEN; blocks release** | Deterministic digital-twin interleaving plus dual-caller source trace; physical race/contact test not run |
| H3 freshness/liveness | **P0 OPEN; blocks release** | Static library/source analysis and simulator cycle-count tests; elapsed timing, stale-cache, rollover, and task-liveness evidence not run |
| H4 prototype safety-claim boundary | **P0 OPEN; blocks release** | Live UI/source wording includes “HARDWARE RELAY CUTOFF AUTOMATICALLY TRIPPED”, “High-frequency arc signature isolated”, and “Breakers open” (`frontend/src/App.jsx:383-391`); no qualified certification, contact feedback, AFCI proof, or isolation evidence |

H4 is a claim/evidence blocker. The software overcurrent threshold and low-rate
dP/dt logic do not establish certified overcurrent protection, AFCI behavior,
galvanic isolation, or relay-contact opening.

## 5. P1 disposition

| Findings | Reconciled live disposition |
|---|---|
| H5 hardware/profile/calibration contract | **OPEN/BLOCKED.** `HARDWARE_FINAL_SPEC.md` declares a 250 W laptop/phone-only build and excludes projector; `config/config.hardware.yaml` declares phone/laptop/projector at 250 W. Board pin count, relay module/polarity, PZEM variant, nameplates, calibration, and BOM are not verified. |
| B1 MQTT identity/ACL | **OPEN; static mismatch confirmed, real deployment unverified.** API and pipeline use `ems_pipeline`; API subscribes to UI events and command topics, while the ACL grants that identity no read access to either. `ems_api` is present in `passwd` without an ACL block; `esp32` has an ACL block but no shipped password entry. The repository broker was not successfully brought up for an authenticated ACL test. |
| B2 malformed UI events | **OPEN/CONFIRMED.** See red reproducer above. Passing API tests do not exercise the live listener failure boundary. |
| P1-PARSER / D1 temporal input contract | **OPEN/CONFIRMED for empty/object/negative coercion.** Live power is a plain float without timestamp, sequence, boot ID, quality, duplicate, or replay semantics; invalid input paths are split. |
| S1 transport/identity exposure | **OPEN.** Live deployment uses plaintext MQTT/HTTP/WS, shared `ems_pipeline` identity, broad command write permission, and a browser-exposed API key fallback. No authenticated TLS/HTTPS/WSS or per-device authorization evidence exists. |
| S2 image/artifact boundary | **OPEN.** `Dockerfile` uses `COPY . .`; `.dockerignore` does not exclude the MQTT password file, local environment files, runtime data, or firmware build artifacts. Runtime model loading uses `weights_only=False` in the pipeline. No release-image scan was run. |
| M1 physical ML enrollment/checkpoint integrity | **BLOCKED/OPEN.** Hardware profile has no `registry_path` and resolves to `backend/models/weights_demo/prototype_registry.pt`, which live-loads seven classes (`phone_charger`, not `phone`) and zero envelopes. Partial ProtoNet state dicts are accepted with `strict=False`; absent/incomplete artifacts do not fail closed. No physical registry exists. |
| M2 unknown/open-set behavior | **PARTIAL/OPEN.** Out-of-envelope simulator cases reject; a novel load inside an enrolled envelope can be named with survivor confidence near 1.0. No physical in-band unknown rehearsal or calibrated deployed decision evidence exists. |
| M3 aggregate disaggregation | **PARTIAL/OPEN.** Synthetic delta tests pass, but live state stores one replaceable classification per aggregate device and has no persistent active appliance set. Physical overlap and ordered-pair gates were not run. |
| M4 enrollment propagation | **OPEN.** `/api/submit-label` publishes to `home/ml/label`; `/api/v1/appliances/label-unrecognized` creates a separate API-local pipeline and persists locally without publishing to the running pipeline. Cross-process acknowledgement was not proven. |
| M5 unknown stability isolation | **OPEN.** `run_pipeline.py` constructs one `self.delta_analyzer`; per-device raw buffers exist, but the stability analyzer is not partitioned by device. The required interleaving test was not run in this audit. |
| B3/D2 boundedness and recovery | **OPEN.** Adversarial traces recorded API device-map growth, omitted pipeline-map eviction, and an unbounded database write queue. No bounded real-deployment soak/capacity result exists. |
| D3 energy accounting | **OPEN.** Recognition and persistence paths are coupled, while dashboard totals derive from sample/fixed-duration assumptions rather than a single timestamped analytics source. |
| U1/U2 dashboard truth and degraded UX | **OPEN.** The frontend preserves state across disconnect, labels a WebSocket connection “LIVE 1Hz Stream”, lacks a freshness/provenance contract, and does not consume a physical relay/contact confirmation. `EnergyChart` generates random illustrative history; summary/appliance energy uses fixed assumptions. No App-level live/stale/safety provenance test passed. |
| T1 training-data/cache contract | **OPEN.** Historical order, gap, cache provenance, and held-out physical data requirements remain unverified; no physical training/validation run occurred. |
| G11 soak/capacity | **NOT STARTED / BLOCKING.** No release soak, rollover, storage, reconnect, or declared-capacity evidence was produced. |

## 6. P2 disposition

| Finding | Disposition |
|---|---|
| L1 allocation/reconnect hardening | **OPEN, non-closing hardening.** Static allocation and fixed retry concerns are not release approval evidence and were not physically or soak tested. |
| L2 verification/performance semantics | **OPEN, evidence-quality concern.** Host tests reimplement some firmware behavior and the reported latency metric is handler duration rather than end-to-end ESP32-to-browser latency. |

P2 status does not reduce the P0/P1 release blockers.

## 7. Reconciled contradictions and duplicated claims

1. **Baseline status duplication:** `CURRENT_STATE.md`, `MASTER_CHECKLIST.md`,
   `RELEASE_GATES.md`, `FAILURE_LOG.md`, and `TEST_MATRIX.md` repeat the
   matching-runtime baseline and the four red tests. They should be read as one
   evidence set, not separate passes. The clean-sounding “641 passed” wording
   must always be paired with “4 intentional failures”.
2. **H1 safety-pass contradiction:** simulator tests such as
   `test_pzem_watchdog_stays_disarmed_until_first_valid_read` and
   `test_bringup_gate7_dry_relay_close_stays_closed` encode the current
   never-valid bring-up exception. They are not H1 safety passes and do not
   disprove the failing H1 test.
3. **H2 latch-versus-owner contradiction:** the overcurrent latch improvement
   closes the missed-lockout path after Core 1 consumes it, but it does not make
   relay actuation single-owner or atomic. Existing tests that issue `ON` after
   latch consumption do not cover the failing cutoff → `ON` → latch sequence.
4. **H3 timing inflation:** “3 seconds” is a cycle-count assumption; “about 15
   seconds” is a source/library estimate. Neither is a measured cutoff deadline.
   Simulator 100 ms stepping and direct register injection cannot establish the
   production UART transaction time, cache age, `millis()` rollover behavior, or
   task liveness.
5. **Parser duplication:** the live callback, `process_raw_mqtt()`, API raw
   power path, and `FleetDiagnosticsMonitor` apply different validation rules.
   Passing helper tests cannot be counted as proof of the production callback
   contract.
6. **Broker-test inflation:** MQTT/API tests use in-memory clients, mocks, or
   ASGI transport. They do not prove Mosquitto authentication, ACL grants,
   retained-message metadata, reconnect/resubscription, or duplicate handling.
7. **ML evidence inflation:** the 128 focused ML/detector tests prove shipped
   demo artifacts and synthetic windows. They do not prove physical enrollment,
   physical held-out accuracy, physical unknown rejection, projector hardware
   suitability, or the three-class physical release claim.
8. **Recognition-path duplication:** `OverlapAwareNILMDetector` is exported but
   not instantiated by the live orchestrator; the live overlap behavior is the
   `_delta_overlap` path in `run_pipeline.py`. OpenMax descriptions in older
   documentation also overstate the normal live registry gate.
9. **Hardware-scope contradiction:** the authoritative hardware document
   excludes projector while current scope/config requires it. Simulator class
   names and demo registry envelopes cannot resolve a BOM, rating, nameplate, or
   wiring decision.
10. **Dashboard truth contradiction:** `EnergyChart` is labelled
    “Illustrative”, but it remains in the energy view; the header presents a
    connected WebSocket as “LIVE 1Hz Stream”, while stale values survive
    disconnect. A disclaimer is not provenance enforcement.
11. **Checkpoint metadata contradiction:** current docs describe a dirty tree and
    HEAD `4e07b658`; the live checkout is at `302e1b21` with no tracked source,
    test, or config diff. This does not close the findings, but it invalidates
    those metadata claims.

## 8. Simulation-only evidence

The following evidence is software-only and cannot be promoted to physical
verification:

- `src/hardware/esp32_firmware_sim.py` relay/PZEM state transitions and HIL
  tests; the twin is not an ESP32, PZEM, relay module, or mains circuit.
- Firmware static inspection, host C tests, direct Python traces, and source
  structure checks.
- Pytest safety, parser, API, MQTT, ML, persistence, chaos, and e2e tests when
  they use mocks, direct callbacks, register injection, synthetic windows, or
  host reimplementations.
- Demo/UK-DALE/REDD model and registry artifacts, including
  `prototype_registry_enrolled.pt`; its observed envelopes are synthetic/demo
  evidence and carry no physical operator, instrument, nameplate, or capture
  attestation.
- Frontend Vitest component tests and mocked browser/WebSocket/API behavior.
- `docker compose config --quiet`; this proves YAML interpolation/syntax only,
  not a running, authenticated, healthy deployment.

## 9. Physical work not run

The following remain `NOT RUN` or blocked, consistent with
`PHYSICAL_VALIDATION.md` and `HARDWARE_STATE.md`:

- Board identity, pin count, UART wiring/levels, relay part, trigger polarity,
  relay contacts, power supply, fuse/RCBO/PE path, enclosure, and load
  nameplates.
- Any mains test, relay-contact test, dead-sensor boot test, brownout waveform,
  EMI/inrush/thermal test, or qualified electrical review.
- PZEM transaction timing, refresh/cache behavior, successful-read age, stale
  finite values, `millis()` rollover, calibration, and measured fault cutoff
  deadline.
- Physical phone, laptop, or projector captures; physical registry creation;
  held-out validation; in-band/out-of-band unknown rehearsal; and physical
  overlap/delta acceptance.
- Physical end-to-end dashboard truth for trip, offline, stale, relay ACK, or
  contact state.

Mains authorization is blocked until one authoritative hardware contract is
selected. In particular, no software test establishes that the current 250 W
build supports projector operation.

## 10. Single recommended next software action

**Fix H1 and H2 at the production actuation boundary as one safety change:**
make the existing safety task the sole relay-closing authority, require a fresh
valid measurement-health state before accepting any production `ON`, and reject
an `ON` whenever a safety inhibit is raised—even before Core 1 consumes the
latch. Preserve the existing protection thresholds. Keep the current H1/H2 red
tests as the acceptance gate, add the retained/reconnect variant, and rerun the
neighboring safety suite before touching the next release domain.

This is the single recommended next software action; it is not implemented by
this audit.

## 11. Audit change boundary

No source, test, or configuration file was modified for this audit. This
checkpoint artifact is the only requested output from Agent 9.
