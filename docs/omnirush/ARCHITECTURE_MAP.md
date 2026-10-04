# ASTRA Architecture Map

**Evidence date:** 2026-10-04 · **Source of truth:** live source plus targeted tests; historical documents are referenced only when marked.

## End-to-end data flow

```text
PZEM-004T / VirtualPZEM
        ↓ UART / simulated registers
ESP32 Core 0 SafetySamplingTask ── owns cutoff decision ── GPIO 18 relay
        ↓ Core 1 telemetry/status/ack MQTT
Mosquitto ACL/topics
        ↓
scripts/run_pipeline.py EMSOrchestrator
  ├─ telemetry validation + fleet safety monitor
  ├─ transient/delta detector → window or delta signature
  ├─ ProtoNet/PrototypeRegistry + heuristic fallback + unknown gate
  ├─ PhantomTracker for sub-threshold standby loads
  ├─ DB/CSV persistence and analytics
  └─ UI/status/command events
        ↓
FastAPI src/api/main.py ── MQTT bridge / REST / WebSocket
        ↓
React frontend App.jsx and dashboard components
```

## Boundary inventory

| Boundary | Implementation | Input | Output / side effect | Safety/truth concern |
|---|---|---|---|---|
| Edge measurement | `main.cpp` / `ESP32FirmwareNode` | PZEM voltage/current/PF/power | shared reading, cutoff, relay GPIO | boot state, non-finite data, freshness, relay ownership |
| Edge command | firmware MQTT callback / twin `handle_mqtt_command` | command payload | relay/ACK/lockout | stale/retained/duplicate/unauthorized ON |
| MQTT transport | Mosquitto ACL + `src/hardware/mqtt.py` | topics/payloads | delivery/subscription | identity, QoS, malformed message isolation |
| Pipeline ingest | `EMSOrchestrator._handle_mqtt_message` | power/telemetry/status | per-device state and safety events | timestamp/sequence/freshness, duplicate/replay |
| Safety analytics | `src/pipeline/safety.py` | validated readings | warnings/cutoff events | server safety must not override edge truth |
| NILM | `NILMTransientDetector`, delta path | aggregate watts | candidate window/event | pre-event window and overlap semantics |
| ML decision | `PrototypeRegistry`, ProtoNet, heuristic | candidate signature + profile | class/confidence/unknown | envelope, incomplete registry, confident-wrong abstention |
| Persistence | `src/database/session.py`, pipeline maps/CSV | readings/events | DB/CSV/in-memory state | bounded memory, duplicate/loss safety |
| API bridge | `src/api/main.py` | MQTT frames and HTTP/WS | browser events/acks | validation exceptions, readiness truth |
| Dashboard | `frontend/src/*` | WS event stream/API | displayed live state/energy | random/hardcoded values and provenance labels |

## Critical callers/callees

- `main.cpp::setup` initializes relay/safety task before network; `SafetySamplingTask` reads PZEM and can call `setRelay(false)`; `loop` handles MQTT/lockout/latches.
- `scripts/run_pipeline.py::main` constructs `EMSOrchestrator`; `_handle_mqtt_message` routes power, telemetry, status, and command-related events through safety/NILM/ML/persistence paths.
- `src/api/main.py` starts MQTT listener/bridge and broadcasts normalized events over WebSockets; label endpoints publish/enroll.
- `frontend/src/App.jsx` consumes `DEVICE_STATUS`, `TELEMETRY`, safety, label, and analytics events; child components render device, alert, energy, and twin state.

## Config and artifact flow

- `config/config.yaml` is the general/default profile.
- `config/config.demo.yaml` is simulator/demo scope.
- `config/config.hardware.yaml` is intended physical profile, currently [phone, laptop, projector] with no `registry_path` and 250 W ceiling.
- `backend/models/weights*` and registry files are model inputs; absence/partial validity must fail closed for physical inference.
- `docker-compose.yml` supplies broker/API/pipeline/frontend services and environment overrides; `mosquitto/config/acl` defines topic permissions.

## Architecture risks to preserve in later work

1. Firmware safety authority must remain edge-local and fail-safe.
2. Server/ML decisions must not be treated as physical actuation confirmation without edge ACK/state.
3. Simulator/HIL evidence is `SIMULATED ONLY` unless accompanied by physical measurements.
4. The three classifiable physical classes from the current scope lock are phone, laptop, projector; LED bulb is phantom-tracked and fan is demo-only/drop scope. This remains subordinate to the unresolved authoritative hardware contradiction.

## Live verification addendum — 2026-10-04

The map above is retained as the baseline artifact. Findings below are **LIVE** when
they cite executable code; statements labelled **HISTORICAL/DOCUMENTED** are target
state, prior documentation, or comments and are not runtime evidence. **Physical
validation: NOT RUN.**

### Runtime entrypoints and ownership

| Layer | Live entrypoint | Boundary and authority |
|---|---|---|
| Firmware | `firmware/esp32_node/src/main.cpp`, built by `firmware/esp32_node/platformio.ini` | `setup()` drives the relay open, starts `SafetySamplingTask` on Core 0 before Wi-Fi, then Core 1 runs MQTT/telemetry. Core 0 performs the physical cutoff; Core 1 handles commands, lockout consumption, ACKs, and status publication. |
| Bench twin | `src/hardware/esp32_firmware_sim.py` | Software model of the Core 0/Core 1 safety state machine and the same five MQTT topic families. It is simulation evidence only. |
| MQTT profile simulator | `backend/scripts/simulate_esp32.py` | Independent 1 Hz publisher/command ACK simulator. It does not instantiate `ESP32FirmwareNode`, PZEM watchdog logic, or relay cutoff logic. |
| Pipeline | `scripts/run_pipeline.py:main()` → `EMSOrchestrator.run()` | Starts two independent broker consumers: a safety monitor consumer and an ML/persistence consumer. The server safety callback publishes UI alerts; it is not the edge cutoff authority. |
| API | `uvicorn src.api.main:app`, `src/api/main.py:lifespan()` | Starts the MQTT→WebSocket bridge, heartbeat, and power-batch task. REST readiness checks only this API's MQTT client and DB connection, not pipeline health. |
| Frontend | `frontend` Vite `dev` / built Nginx image; `frontend/src/App.jsx` | Consumes `ws://<host>:8000/ws`; it does not connect to MQTT directly. |
| Deployment | `make run`/`make demo` or `docker-compose.yml` | `make run` uses the default broad profile and the independent all-device simulator; `make demo` uses `scripts/demo_full_system.py --demo`; Compose starts broker, pipeline, API, and frontend but no simulator or firmware. |

### Verified topic/data flow

1. The firmware publishes `home/sensor/{id}/power` as a plain float at roughly 1 Hz,
   `home/sensor/{id}/telemetry` as `{v,i,w,pf}` roughly every 10 s, status on
   `home/sensor/{id}/status`, and relay ACKs on `home/plug/{id}/ack`. It subscribes
   to `home/plug/{id}/command` and accepts exact `ON`, `OFF`, and `WARNING` strings.
2. `EMSOrchestrator.run()` subscribes its ML client to power, ACK, status, telemetry,
   and `home/ml/label`. A second client subscribes to power for
   `FleetDiagnosticsMonitor`. Power is parsed into device state, the per-device
   `NILMTransientDetector`/delta path, classification, phantom tracking, DB/CSV,
   analytics, digital-twin, and RL paths. `home/ml/label` updates and persists the
   pipeline's registry.
3. Pipeline events are published to `home/ui/events`. The API also independently
   subscribes to raw power and command topics: raw power updates API-local state and
   `_ws_power_buffer`; command messages are rendered as `safety_alert` messages
   regardless of whether the payload is `ON` or `OFF`. Pipeline event JSON is
   validated, copied into API-local state, and broadcast to WebSocket clients.
4. The frontend consumes `init_state`, `power_batch`, `DEVICE_STATUS`, `TELEMETRY`,
   safety, label, analytics, PMV, and latency events. The `power_reading` frontend
   branch has no producer in the live API path; the live raw-power broadcast is
   `power_batch`.

### Safety authority and verified safety gap

- **LIVE authority:** edge Core 0 opens GPIO 18 without network dependency for
  overcurrent (`RATED_WATTS * 1.25`), arc-fault proxy (`dP/dt > 1000 W/s` outside
  inrush), and post-valid-read PZEM-loss watchdog. Core 1 consumes the latches,
  applies the five-minute lockout, and reports status. The server's
  `FleetDiagnosticsMonitor` is diagnostic/UI monitoring; its normal critical/arc
  callback does not command the relay. RL `SHED` is a separate server control path
  and can publish `OFF` only after policy promotion; the legacy `_relay_callback`
  `OFF` branch can also publish an OFF request.
- **LIVE unresolved safety hole:** `main.cpp` explicitly gates the PZEM watchdog on
  `pzemEverValid`. A sensor dead from boot can therefore accept Core 1 `ON` without
  a fresh measurement. The twin carries the same residual hole. This matches the
  current red H1/H2 reproduction status in `TEST_MATRIX.md`; it is not a physical
  pass. `setRelay()` has callers on both cores, so edge-local authority does not
  mean there is only one concurrent relay caller; the Core 0/Core 1 interleaving is
  the H2 contract under review.
- **LIVE parser split:** broker delivery calls `_handle_mqtt_message()` directly,
  where empty payloads become `0.0`, `{}` can become `0.0`, and negative finite
  values continue into the pipeline. The separate `process_raw_mqtt()` test/helper
  path rejects missing JSON power and negative values. `FleetDiagnosticsMonitor`
  has a third behavior: it applies `abs()` to negative values. These are distinct
  ingestion contracts, not one shared parser.

### Duplicated or conflicting implementations

- `main.cpp`, `src/hardware/esp32_firmware_sim.py`, and
  `backend/scripts/simulate_esp32.py` are three different edge representations.
  Only the first two implement the safety state machine; the MQTT profile simulator
  can publish plausible telemetry and ACKs without proving cutoff behavior.
- `src/pipeline/aggregate_nilm.py` exports `OverlapAwareNILMDetector`, but the live
  orchestrator instantiates plain `NILMTransientDetector` per device. The live
  overlap behavior is the custom `_delta_overlap` state in `run_pipeline.py`, not
  the exported overlap class. The **HISTORICAL/DOCUMENTED** `CLAUDE.md` warning
  that `OverlapAwareNILMDetector` is not live remains accurate.
- **HISTORICAL/DOCUMENTED** README/module descriptions advertise OpenMax and a
  `>=0.90` confidence gate. The live `_classify_device()` path uses registry
  distances, a physical power-envelope gate, optional heuristic agreement, and
  `recognition_threshold` (0.45 in the profiles); OpenMax is only reported active
  when named Weibull tails are loaded. The 0.90 `confidence_threshold` is not the
  final gate for the normal registry path.
- The existing map's line stating that missing/partial physical weights must fail
  closed is **TARGET STATE**, not the observed fallback: live code falls back to
  the heuristic classifier when the encoder/registry is unavailable. In the
  hardware profile, omitted `registry_path` resolves to the shipped
  `weights_demo/prototype_registry.pt`, not a physical enrollment artifact.
- The API and pipeline each construct a `FullPipeline` for labeling. The running
  pipeline consumes `home/ml/label`; `/api/v1/appliances/label-unrecognized` lazily
  constructs an API-local pipeline from `EMS_CONFIG` or defaults to
  `config/config.demo.yaml` and writes its own registry. Without an explicit
  `EMS_CONFIG`/shared artifact contract, this endpoint is not the same in-process
  registry as the running pipeline. The older `/api/submit-label` route publishes
  to `home/ml/label` and is the connected path.
- API state, pipeline state, and frontend state are separate in-memory stores. The
  API's raw-power updates and pipeline `DEVICE_STATUS` events can both update the
  same browser device, while `/ready` can report API readiness even when the
  pipeline consumer is unhealthy.
- `EnergyChart` still generates random illustrative series; it is visibly labelled
  `Illustrative`. `SummaryCards` and `ApplianceTable` also derive energy/cost from
  sample-count or fixed-hour/fixed-tariff assumptions, so those values are not the
  same persisted analytics source as the pipeline's `ANALYTICS_UPDATE` event.

### Configuration, identity, and artifact contradictions

- `config/config.yaml` is the default pipeline profile (broad household classes,
  3.5 kW aggregate limit). `config/config.demo.yaml` is a 600 W simulated
  consumer-electronics profile. `config/config.hardware.yaml` is the 250 W,
  one-node profile. These ceilings and device lists are intentionally not
  interchangeable, but `docker-compose.yml` starts `run_pipeline.py` without
  `--config`, so Compose uses the default profile unless externally overridden.
- The demo config allows `phone`, `laptop`, `bulb`, and `projector`; its simulator
  also retains `fan` as an out-of-family device. The physical config allows
  `phone`, `laptop`, and `projector`, while its comments say only two classes and
  omit `registry_path`. The live `prototype_registry_enrolled.pt` artifact contains
  `fan` plus other demo classes, and `_eligible_classes()` exempts enrolled classes
  from the profile allow-list. This can reintroduce fan/demo classes despite the
  current scope lock.
- **Authoritative hardware contradiction:** `claude_debug/HARDWARE_FINAL_SPEC.md`
  is marked authoritative and locks laptop + phone charger only, explicitly not
  projector, while the live hardware YAML and root/current scope lock require
  projector. The board, relay, PZEM, nameplate, and intended build have not resolved
  whether the spec, config, or scope claim wins.
- Compose uses the shared `ems_pipeline` MQTT identity for both pipeline and API;
  `mosquitto/config/acl` documents a dedicated `esp32` identity, but `esp32` is
  absent from the shipped `passwd`. The `ems_api` user exists but is unused.
  Firmware bring-up therefore still requires deployment-time credentials and ACL
  provisioning. QoS claims also diverge: firmware `PubSubClient` publishes use its
  default QoS behavior, pipeline manager subscribe/command calls request QoS 1,
  and the API's direct label publish does not specify QoS; **HISTORICAL/DOCUMENTED**
  QoS tables must not be treated as a single live transport guarantee.

### Evidence gaps

- Physical mains, relay contact, PZEM timing/freshness, brownout/EMI, calibration,
  physical enrollment, and unknown-device rehearsal remain **NOT RUN**.
- Current documented software gates remain the evidence boundary: H1, H2, malformed
  parser, and API malformed-event reproductions are red by design; real-broker ACL
  verification, physical registry enrollment, physical open-set/overlap validation,
  dashboard live-truth validation, and soak/resource bounds remain pending or
  blocked as recorded in `TEST_MATRIX.md` and `MASTER_CHECKLIST.md`.
- `claude_debug/HARDWARE_ALIGNMENT_CONTRACT.md` G1's old wording that telemetry
  subscription is a gap is **HISTORICAL**; live `run_pipeline.py` now subscribes,
  validates, and emits `TELEMETRY`. Its physical recognition and hardware identity
  entries remain evidence/configuration prerequisites, not physical validation.
