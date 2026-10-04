# ASTRA Adversarial Findings

**Reviewer:** Agent 8 — Adversarial Reviewer  
**Date:** 2026-10-04  
**Scope:** software-only attacks against the current working tree.  No source,
test, firmware, or configuration files were changed.  Simulator and inline
Python traces are software evidence only; none is physical validation.

## Summary

| ID | Severity | Attack surface | Result | Physical status |
|---|---:|---|---|---|
| ADV-H1 | P0 | Dead PZEM at boot + ON | **Confirmed fail-open**: relay remains ON after 60 invalid samples | NOT RUN |
| ADV-H2 | P0 | Safety trip + ON race | **Confirmed**: ON re-closes relay before the Core 1 latch lockout | NOT RUN |
| ADV-H3 | P0 | Finite frozen PZEM value | **Confirmed software gap**: stale finite reading is treated as healthy indefinitely | NOT RUN |
| ADV-B2 | P1 | Malformed UI MQTT frame | **Confirmed**: Pydantic and non-UTF-8 errors escape and kill the bridge task | NOT RUN |
| ADV-B3 | P1 | Nonfinite UI event fields | **Confirmed**: `NaN`/`Infinity` enter API device state | NOT RUN |
| ADV-PARSER | P1 | Live power parser | **Confirmed**: empty/object become `0 W`; negative power is stored | NOT RUN |
| ADV-CMD | P1 | Duplicate/stale/retained `ON` | **Confirmed**: repeated and boot-time retained-equivalent `ON` are accepted | NOT RUN |
| ADV-PZEM-ZERO | P1 | Zero PZEM reading | **Confirmed software gap**: finite zero arms health and leaves an energized relay ON | NOT RUN |
| ADV-M1 | P1 | Missing/partial ML registry/checkpoint | **Confirmed fail-open loading**: incomplete artifacts are accepted or degraded into heuristic mode | NOT RUN |
| ADV-M2 | P1 | ML unknown/open set | Out-of-envelope rejection passes; an in-envelope unknown remains confidently indistinguishable | PHYSICAL REHEARSAL NOT RUN |
| ADV-U1 | P1 | Stale frontend/API state | **Confirmed**: old device state remains observable with no age-based eviction | NOT RUN |
| ADV-R1 | P1 | API device-state flood | **Confirmed**: 1,000 unique valid events produce 1,000 retained device entries | NOT RUN |
| ADV-R2 | P1 | Pipeline per-device map eviction | **Confirmed leak**: unknown-window and phantom maps survive device eviction | NOT RUN |
| ADV-R3 | P1 | Database outage/backlog | **Confirmed**: write queue has no bound; 10,000 writes remain queued | NOT RUN |

## Findings

### ADV-H1 — dead-at-boot PZEM does not block `ON`

- **Severity:** P0
- **Reproduction:**
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_dead_pzem_at_boot_cannot_accept_on_command -q --tb=short`
- **Result:** Failed as intended.  With `voltage = NaN`, `ON` was accepted;
  after `PZEM_FAIL_TRIP_COUNT * 2` Core 0 samples the simulator reported
  `relay=True`, `ever_valid=False`, `fail_count=30`, and
  `shared_pzem_fault=False`.
- **Root-cause hypothesis:** The loss watchdog is gated by
  `_pzem_ever_valid`/`pzemEverValid`, while the command handler gates `ON`
  only on `relay_locked`.  A sensor that is dead from boot never arms the
  watchdog, so the command path can energize an unmeasured circuit.
- **Physical status:** NOT RUN.  The result is simulator/firmware-code
  evidence only; no PZEM, mains, relay, or boot test was performed.

### ADV-H2 — Core 1 can re-close after a Core 0 safety cutoff

- **Severity:** P0
- **Reproduction:**
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_safety_cutoff_wins_over_on_before_core1_latch_tick -q --tb=short`
- **Result:** Failed as intended.  Core 0 first produced
  `relay=False, shared_overcurrent_latch=True`; an intervening `ON` followed
  by the Core 1 tick produced `relay=True, relay_locked=True`.
- **Root-cause hypothesis:** Core 0 opens the relay and raises a latch, but
  Core 1 does not consume that latch until later.  `handle_mqtt_command("ON")`
  observes the old unlocked state and writes the relay closed.  Lockout is
  applied after the unsafe actuation and does not itself force the relay open.
- **Physical status:** NOT RUN.  This is a deterministic simulator/inter-core
  scheduling trace, not a physical dual-core or relay test.

### ADV-H3 — finite frozen PZEM values are accepted indefinitely

- **Severity:** P0
- **Reproduction:** Software-only isolated trace: close the simulated relay,
  take one valid `100 W` sample, then repeat the same finite PZEM register
  values for 100 Core 0 samples without a new measurement sequence.
- **Result:** `relay=True`, `fail_count=0`, `ever_valid=True`, and
  `last_watts=100.0`.  No freshness fault or cutoff was generated.
- **Root-cause hypothesis:** Firmware/simulator validity is only a finiteness
  check.  There is no PZEM transaction sequence, sample freshness timestamp,
  or independent age check proving that a finite register value is new.  A
  frozen value therefore continually resets/avoids the loss watchdog and
  leaves both overcurrent and dP/dt decisions based on stale data.
- **Physical status:** NOT RUN.  The trace proves the missing software
  invariant; whether the deployed PZEM/library can physically return stale
  finite values was not tested.

### ADV-B2 — malformed UI MQTT frames terminate the API bridge

- **Severity:** P1
- **Reproduction 1:**
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_malformed_ui_event_does_not_crash_mqtt_bridge -q --tb=short`
- **Result 1:** Failed as intended.  A schema-valid JSON envelope containing
  `power: "bad"` raised a Pydantic `ValidationError` out of
  `mqtt_listener_task`.
- **Reproduction 2:** Feed `home/ui/events` the bytes `b"\xff\xfe\xfd"`.
- **Result 2:** `UnicodeDecodeError` escaped at `src/api/main.py:277`; the
  task ended while `system_state["pipeline_status"]` remained `"connected"`.
- **Root-cause hypothesis:** The UI-event branch catches only
  `json.JSONDecodeError`.  Pydantic validation errors, non-UTF-8 decoding
  errors, and other event-shape exceptions are outside that catch and there is
  no outer generic task-health handler.  Readiness/status can consequently
  report a connected bridge after the consumer has died.
- **Physical status:** NOT RUN.  No broker or hardware is required for this
  software failure.

### ADV-B3 — nonfinite UI event values bypass API validation

- **Severity:** P1
- **Reproduction:** Send one UI event with Python-JSON-compatible constants:
  `{"type":"DEVICE_STATUS","device_id":"nan_ui","power":NaN,"confidence":Infinity,"pmv":-Infinity}`.
- **Result:** The listener completed without an exception and stored
  `power=nan`, `confidence=inf`, and `pmv=-inf` in
  `system_state["devices"]["nan_ui"]`.  The raw `/power` and `/telemetry`
  paths do reject nonfinite values, but this structured event path does not.
- **Root-cause hypothesis:** `DeviceStatusEvent` declares plain Pydantic
  `float` fields without finite-value validators, and Python's default JSON
  decoder accepts `NaN`/`Infinity`.  The invalid values can then reach REST or
  WebSocket serialization and contaminate dashboard state.
- **Physical status:** NOT RUN.  Software/API trace only.

### ADV-PARSER — live callback parser coerces or stores invalid power

- **Severity:** P1
- **Reproduction:**
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_direct_pipeline_handler_rejects_empty_object_and_negative_power -q --tb=short`
- **Result:** Failed as intended.  Direct `_handle_mqtt_message` processing
  stored `""` as `0.0`, stored `{}` as `0.0`, and stored `-3` as `-3.0`.
  The public `process_raw_mqtt` wrapper separately returned `PARSE_ERROR` for
  empty/object and `SENSOR_ERROR` for negative/NaN/Infinity, proving that the
  two live entry paths have different contracts.
- **Root-cause hypothesis:** The callback initializes `power_watts = 0.0`,
  uses a missing JSON field's default `0.0`, and checks only NaN/Infinity in
  the direct handler.  It does not reject empty/object/negative values before
  mutating `last_device_power`.
- **Physical status:** NOT RUN.  No sensor or broker was used.

### ADV-CMD — duplicate, stale, and retained-equivalent `ON` replay

- **Severity:** P1
- **Reproduction:** Invoke `handle_mqtt_command("ON")` twice on a node, then
  invoke `"ON"` once on a freshly constructed node as a retained-message
  replay equivalent.
- **Result:** Both duplicate commands leave the relay ON with
  `relay_locked=False`; a newly booted node also accepts the replay-equivalent
  `ON` and leaves the relay ON.  There is no command ID, boot/session epoch,
  expiry, or retained-message distinction in the software path.
- **Root-cause hypothesis:** The firmware command contract is an exact string
  match and the only actuation gate is the in-memory `relayLocked` flag.  MQTT
  reconnect/reboot does not establish a fresh command epoch, and the relay
  lockout is not persistent across reboot.  A stale retained desired-state
  message is therefore indistinguishable from a current operator command.
- **Physical status:** NOT RUN.  This is a simulator command-path result; no
  real broker retained delivery or physical reboot was performed.

### ADV-PZEM-ZERO — finite zero is treated as a healthy measurement

- **Severity:** P1
- **Reproduction:** Close the relay, set the simulator PZEM to a finite zero
  load, and run one Core 0 sample.
- **Result:** `relay=True`, `ever_valid=True`, `fail_count=0`, and
  `shared_power=0.0`.  A zero-valued sensor frame is not rejected or marked
  stale.  Negative simulator input is also silently clamped by
  `VirtualPZEM004T.set_load(-100)` to `0.0`; negative raw MQTT input is stored
  by the direct pipeline handler as covered by ADV-PARSER.
- **Root-cause hypothesis:** PZEM validity requires only finite numeric fields;
  there is no closed-relay plausibility rule for an all-zero measurement and
  no distinction between a legitimate unloaded socket and a failed sensor that
  reports zeros.
- **Physical status:** NOT RUN.  Whether zero is a valid no-load frame for the
  exact installed PZEM and wiring was not established on hardware.

### ADV-M1 — missing/partial ML artifacts do not fail closed

- **Severity:** P1
- **Reproduction 1:** Instantiate `config/config.hardware.yaml` with the
  current artifacts.  The profile has no explicit `registry_path`; loader
  resolution nevertheless loaded the default shipped registry with classes
  `laptop, tv, desktop_computer, monitor, phone_charger, router, projector`
  and **no power envelopes**.
- **Reproduction 2:** Provide a temporary `protonet.pt` containing one state
  tensor plus a temporary registry containing only `laptop`.  Startup reported
  `encoder_loaded=True`, `registry_loaded=True`, one class, and classified a
  100 W window as `laptop` with confidence `0.999999999999`.
- **Result:** Partial checkpoints are accepted through
  `load_state_dict(..., strict=False)` and a partial registry is accepted with
  no required-class, envelope, provenance, or completeness check.  When the
  registry is absent, the code degrades to heuristic classification rather
  than disabling recognition.
- **Root-cause hypothesis:** Artifact loading treats missing/unexpected model
  keys and incomplete registry contents as warnings, not deployment-fatal
  validation errors.  The one-class survivor is then renormalized to full
  confidence, making artifact incompleteness look like certainty.
- **Physical status:** NOT RUN.  No physical enrollment captures or registry
  provenance artifact exists in this execution loop.

### ADV-M2 — open-set behavior is good outside envelopes but not inside them

- **Severity:** P1 residual limitation
- **Reproduction:** The focused ML suite passed:
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_ml_pipeline_recognition.py -q --tb=short`
  (**52 passed**).  Its out-of-family 800–7,000 W cases and the direct
  2,000 W probe return `unknown` with zero confidence.  The same suite's
  `test_a_novel_load_inside_an_enrolled_envelope_is_the_known_limit` returns
  `laptop` for a novel 120 W trace inside the enrolled laptop envelope.
- **Result:** The requested out-of-envelope abstention control held in
  software.  An unknown device whose power and waveform fall inside an
  enrolled class's measured envelope remains confidently named as that class.
- **Root-cause hypothesis:** The learned embedding is power-scale-blind and
  the physical envelope is the independent gate.  Once only one enrolled
  envelope survives, probability renormalization makes the answer effectively
  certain; no independent per-appliance measurement remains to distinguish an
  in-band mimic.
- **Physical status:** Physical unknown-device rehearsal NOT RUN.  This is a
  documented software limitation, not evidence of physical recognition.

### ADV-U1 — stale device state remains visible as live fleet state

- **Severity:** P1
- **Reproduction:** Seed API state with a device whose `last_seen` is two hours
  old, then call `get_devices()`.
- **Result:** The endpoint returned `stale_node` unchanged with an observed
  `last_seen_age_s=7200`.  In the frontend, WebSocket `onclose` changes only
  `connectionStatus`; it does not clear or age `devices`.  `DeviceCards` counts
  every entry as Active and renders its last power/state without checking
  `last_seen`.  Existing frontend tests still passed (**20 passed**) because
  the stale/reconnect sequence is not covered.
- **Root-cause hypothesis:** Freshness is retained as metadata but is not an
  API eviction contract or a frontend display gate.  Reconnect status is shown
  separately while old state remains visually authoritative.
- **Physical status:** NOT RUN.  No live dashboard connected to physical
  telemetry.

### ADV-R1 — API device-state map has no capacity or TTL bound

- **Severity:** P1
- **Reproduction:** Feed 1,000 valid `DEVICE_STATUS` UI events with unique
  device IDs into the isolated API listener.
- **Result:** The listener task ended cleanly and
  `len(system_state["devices"])` was **1,000**.  There is no corresponding
  `max_tracked_devices`/TTL eviction in `src/api/main.py`; those controls only
  exist in the pipeline orchestrator's separate state.
- **Root-cause hypothesis:** The API bridge trusts every unique UI event and
  mutates a process-global dictionary without a cap, authentication-to-device
  binding, or stale-entry cleanup.  A permitted MQTT publisher can exhaust
  memory with unique IDs even if the pipeline itself evicts its own maps.
- **Physical status:** NOT RUN.  Software/API flood only.

### ADV-R2 — pipeline eviction omits unknown-window and phantom maps

- **Severity:** P1
- **Reproduction:** Configure `max_tracked_devices=2`, seed five old device
  IDs in `_device_last_seen`, `_unknown_windows`, and
  `phantom_tracker.phantom_loads`, then call `_evict_stale_devices()`.
- **Result:** `_device_last_seen` fell to **0**, but
  `_unknown_windows` remained **5** and `phantom_loads` remained **5**.
  Repeated unique unknown/phantom IDs can therefore grow these maps after the
  nominal tracked-device eviction has run.
- **Root-cause hypothesis:** The eviction loop removes a fixed list of
  orchestrator dictionaries but does not include `_unknown_windows` or the
  `PhantomTracker` store.  Per-entry deques are bounded, but their owning
  device-key dictionaries are not.
- **Physical status:** NOT RUN.  Software memory-bound trace only.

### ADV-R3 — database write queue is unbounded during outage/backlog

- **Severity:** P1
- **Reproduction:** Set a `DatabaseSession` to its running state without a
  flushing connection and enqueue 10,000 measurements through the public
  `insert_measurement()` method.
- **Result:** The internal `asyncio.Queue` contained **10,000** records.  Its
  constructor has no `maxsize`, and `insert_measurement()` always awaits
  `put()` without backpressure, rejection, or a spill bound.
- **Root-cause hypothesis:** Database availability is decoupled from ingest
  admission, so a prolonged DB lock, outage, or slow disk converts incoming
  telemetry into unbounded process memory growth.  CSV fallback protects some
  write failures but does not bound the in-memory queue before flush failure.
- **Physical status:** NOT RUN.  Software resource trace only.

## Controls that held during this review

These are recorded to distinguish attempted attacks that passed from the open
findings above:

- `tests/test_relay_safety_boot_brownout.py`: **52 passed**.  This covers the
  normal simulator cutoff, nonfinite-read handling after a valid sample, latch
  consumption, and lockout paths; it does not close ADV-H1 or ADV-H2 because
  the dedicated audit interleavings remain red.
- `tests/test_security_penetration.py tests/test_mqtt.py`: **54 passed**.
  Raw non-UTF-8 power payloads, simulator command length limits, ordinary mock
  reconnect, and normal MQTT callback paths held.  The API UI-event decoder is
  a separate path and remains vulnerable under ADV-B2/B3.
- `tests/test_detector_path_e2e.py tests/test_overlap_delta.py`: **20 passed**.
  The software detector/delta paths did not produce an additional finding in
  this review.
- `tests/test_hil_uart_corruption.py`: **30 passed**.  The simulator's
  corrupted/nonfinite UART-oriented cases held for the covered scenarios; the
  finite-freeze and dead-at-boot cases above are outside those assertions.
- Nonfinite PZEM after a previously valid sample: the simulator stayed ON for
  samples 1–29 and opened on sample 30, matching `PZEM_FAIL_TRIP_COUNT`.
  This is a pass for the armed watchdog case only; it does not mitigate the
  dead-at-boot or finite-freeze findings.
- `tests/test_ml_pipeline_recognition.py`: **52 passed**.  Out-of-envelope
  rejection, label payload validation, and the documented in-envelope unknown
  limitation behaved as asserted.
- `frontend`: `npm test -- --run` — **20 passed**.  No test exercised a
  disconnect followed by stale device rendering.

## Remaining verification blockers

- Physical validation is still **NOT RUN**: no mains, relay contact,
  brownout, PZEM timing/freeze, retained-broker replay, calibration, or
  physical ML enrollment test was performed.
- A real Mosquitto identity/ACL test and a physical unknown-device rehearsal
  remain outside this software-only review.
- The four baseline audit reproducers remain open and red: H1, H2, direct
  parser validation, and B2 bridge resilience.
