# ASTRA Baseline and Defect Reproduction Matrix

**Status:** Reproduction commands are defined before remediation. Results are filled only from current live runs.

## Baseline commands

| ID | Command | Purpose | Result |
|---|---|---|---|
| BASE-PY | `python -m pytest tests/ -q` | Current Python regression count | BLOCKED by broken venv interpreter |
| BASE-PY-MATCH | `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/ -q --tb=short` | Matching-runtime Python regression | **641 passed, 3 warnings, 22.70 s** |
| BASE-FE | `cd frontend && npm test -- --run` | Current frontend count | **20 passed, 1 file, 1.45 s** |
| BASE-VER | `python3 --version; node --version; npm --version` | Runtime versions | Python 3.12.3 system / Python 3.10.12 test / Node 24.14.0 / npm 11.9.0 |
| BASE-E2E | `python scripts/test_firmware_and_ai_e2e.py` | Closed-loop simulator evidence | NOT RUN; simulated only |
| BASE-HIL | `python scripts/hil_hardware_test.py` | HIL simulator/harness | NOT RUN; not physical validation |

## Focused red reproducer result

Command: `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py -q --tb=short`  
Result: **4 failed, 0 passed, 1.44 s** — expected before remediation.

- `test_dead_pzem_at_boot_cannot_accept_on_command`: relay remained `True` after 60 invalid samples.
- `test_safety_cutoff_wins_over_on_before_core1_latch_tick`: relay was `True` after Core 0 cutoff, ON, and Core 1 latch consumption.
- `test_direct_pipeline_handler_rejects_empty_object_and_negative_power`: `""` became `0.0`, and the subsequent invalid cases are accepted by the direct handler.
- `test_malformed_ui_event_does_not_crash_mqtt_bridge`: Pydantic `ValidationError` escaped `mqtt_listener_task`.

## Required defect matrix

| ID | Severity | Reproduction target | Minimal evidence to capture | Regression invariant | Hardware status |
|---|---:|---|---|---|---|
| H1 | P0 | Construct/boot node with dead PZEM; issue ON and retained/reconnect ON | relay state after boot and command; sensor health state | no valid fresh measurement means relay cannot energize | physical NOT RUN |
| H2 | P0 | Safety trip and ON request in same scheduling window | ordered events and final relay/lock state | safety inhibit/open wins every interleaving; one actuation owner | physical NOT RUN |
| H3 | P0 | Freeze/timeout PZEM after valid reading; measure elapsed blind interval | successful transaction timestamp, timeout behavior, cutoff time | age is based on successful transaction, bounded to stated limit | physical NOT RUN |
| H4 | P0 | Audit docs/UI terminology for overcurrent/AFCI/isolation/certification claims | exact claim locations and evidence source | prototype limitations are visible and unambiguous | physical/certification NOT RUN |
| H5 | P1 | Compare authoritative spec, config, firmware constants, wiring, BOM, ML class set | contradiction table | one resolved hardware contract | actual hardware unknown |
| B1 | P1 | Start real Mosquitto with shipped identities/ACL; test API subscribe/read/write | auth result, topics, ACL decision | each identity has least required grants and no hidden shared control | real broker pending |
| B2 | P1 | Inject invalid JSON, wrong type, UTF-8, schema-invalid event into API MQTT callback | bridge task lifecycle/readiness after each frame | malformed frame is isolated and status reflects listener health | no |
| P1-PARSER | P1 | Feed empty, `{}`, negative, NaN, Inf, nonnumeric, oversized power payloads | parser result and downstream state | reject invalid values; never coerce invalid input to safe-looking 0 W | no |
| M1 | P1 | Load hardware profile with no registry and partial checkpoint | inference enablement and failure mode | not enrolled/invalid weights fail closed | physical enrollment blocked |
| M2 | P1 | Classify in-band unknown, out-of-envelope, low confidence, ambiguous signature | label/confidence/rejection event | UNKNOWN/label loop preferred over confident wrong class | physical rehearsal blocked |
| M3 | P1 | Ordered aggregate load additions/removals and simultaneous changes through actual detector path | active set and event method/labels | maintain active appliance set; unplug does not re-emit stale class | physical overlap pending |
| U1 | P1 | Feed stale/offline/safety-originated/inferred/synthetic events to frontend | rendered badges, measurement age, provenance | UI never represents inferred/stale/random data as live hardware truth | physical live feed NOT RUN |
| G11 | P1 | Long bounded sample/event/reconnect/storage sequences | RSS, queue/dict sizes, DB/file growth, rollover | declared bounds hold deterministically | no |

## Test quality checks

- Tests that only assert execution are insufficient; assertions must cover relay state, lock state, parser outcome, task liveness, label provenance, or bounded resource invariant.
- Static firmware tests must parse code structure without counting comments/prose as safety behavior.
- Simulator/HIL tests prove software behavior only and are tagged `SIMULATED ONLY`.
- Physical evidence must identify hardware serial/config, measurement instrument, operator, timestamp, and raw artifact; absent evidence remains `NOT RUN`.
