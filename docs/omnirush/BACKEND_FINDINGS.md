# Backend / Protocol Findings

**Agent:** 5 — Backend/Protocol Engineer
**Evidence date:** 2026-10-04
**Scope:** MQTT identity/ACL, payload parsing and validation, replay/freshness,
bridge lifecycle, readiness, event-loop behavior, queues, and memory bounds.
**Physical validation:** `NOT RUN`
**Change policy followed:** source, tests, and configuration were not modified.

## Executive result

The backend/protocol gate is **NO-GO / OPEN**. The live system has confirmed
parser and MQTT-bridge failure paths, while the shipped ACL does not grant the
API bridge the subscriptions it requests. Existing tests mostly exercise
in-memory mocks or helper entry points and do not prove authenticated
least-privilege behavior against the broker that is deployed.

## Root-cause findings

### B1 — Identity and ACL contract is internally inconsistent — P1, OPEN

**Root cause:** the API and pipeline are both configured as `ems_pipeline`, but
the ACL grants that identity write access to `home/ui/events` without read
access, and does not grant read access to `home/plug/+/command` even though the
API subscribes to it.

Evidence:

- `src/api/main.py:271-273` subscribes to
  `home/sensor/+/power`, `home/plug/+/command`, and `home/ui/events`.
- `mosquitto/config/acl:21-28` grants `ems_pipeline` read access to
  `home/sensor/#`, `home/plug/+/ack`, `home/ml/label`, and `$SYS/#`, but not
  `home/ui/events` or `home/plug/+/command`. Its write grants include all
  `home/plug/+/command`, `home/ui/events`, and `home/ml/label`.
- `docker-compose.yml:45-48` gives both `ems-pipeline` and `ems-api` the same
  `MQTT_USERNAME`/`MQTT_PASSWORD`, defaulting to the `ems_pipeline` identity.
- The ACL contains an `esp32` block, but the current passwd file has no
  corresponding `esp32` entry. The passwd file does contain `ems_api`, but the
  ACL has no `user ems_api` block.
- `src/hardware/mqtt.py:60-85` accepts an arbitrary command topic and payload;
  it performs no device/topic allowlist check before publishing.

Impact:

- The API bridge cannot receive the pipeline's UI events under the shipped
  ACL, so the intended MQTT-to-WebSocket path is not proven and is expected to
  fail with an authorization error.
- The API's command subscription is denied, so command safety alerts are not
  delivered through that path.
- Any holder of the shared `ems_pipeline` credential can publish a command for
  any `home/plug/+/command` topic and spoof UI or ML-label messages. The broker
  cannot distinguish the API from the pipeline.
- `esp32` is not deployable from the checked-in credential state, and the
  historical `ems_api` identity is ambiguous rather than least privilege.

### B1.1 — The currently listening broker is not the repo broker — P1, OPEN

The repository host configuration was not successfully brought up:

```text
mosquitto -c mosquitto/config/mosquitto-host.conf -v
  Error: Unable to decode password salt for user ems_pipeline, removing entry.
  Error: Unable to decode password salt for user ems_api, removing entry.
  Error: Address already in use
```

The existing process is `/usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf`.
An unauthenticated `aiomqtt` connection to `127.0.0.1:1883` was accepted. This
is evidence of broker collision/security downgrade in the current environment,
not evidence that the repository ACL works. The repository config itself sets
`allow_anonymous false` (`mosquitto/config/mosquitto.conf:7-12`), so it requires
an isolated, single-owner broker test before any ACL result can be trusted.

The host broker also warned that the repository passwd and ACL files are world
readable. File permissions and valid Mosquitto password hashes are deployment
prerequisites.

### B2 — MQTT bridge dies on valid UTF-8/schema/type failures — P1, OPEN

**Root cause:** frame decoding and event dispatch are not isolated by a
per-message exception boundary.

- `src/api/main.py:275-277` decodes bytes before the UI-event `try` block. A
  malformed UTF-8 frame raises `UnicodeDecodeError` out of the listener.
- `src/api/main.py:281-396` catches only `json.JSONDecodeError`.
  `pydantic.ValidationError`, `AttributeError` for a JSON list/scalar, and
  unexpected model/runtime errors escape the `async for` loop.
- `src/api/main.py:107-170` defines permissive event models: no finite-value,
  range, non-empty-ID, payload-size, or strict type constraints are present.
  `timestamp: Optional[Any]` accepts untrusted values.
- There is no inbound MQTT payload-size limit in the API bridge. The only
  256-byte limit found is in the firmware command simulator/firmware contract,
  not in `src/api/main.py`.
- `src/api/main.py:433-443` handles only `aiomqtt.MqttError` and cancellation.
  An unexpected per-frame exception therefore ends the task without clearing
  `_shared_mqtt_client` or changing `system_state["pipeline_status"]`.

Live reproducer evidence:

- `tests/test_audit_reproductions.py -q --tb=short`: **4 failed as intended**;
  `test_malformed_ui_event_does_not_crash_mqtt_bridge` reports the escaped
  Pydantic `ValidationError` for `power="bad"`.
- A direct listener probe produced task-level exceptions for each of:
  `b"\xff\xfe"` (`UnicodeDecodeError`), `b"[]"` (`AttributeError`), and a
  schema/type-invalid `DEVICE_STATUS` frame (`ValidationError`).

Readiness consequence: `/health` is unconditional 200
(`src/api/main.py:562-565`). `/ready` only checks whether the global client and
DB references are non-`None` (`src/api/main.py:568-578`); it does not verify
that the listener task is alive, authenticated, subscribed, or processing
frames. A listener that dies from one malformed frame can therefore leave
readiness falsely healthy.

### P1-PARSER — Live pipeline handler accepts invalid power — P1, OPEN

**Root cause:** production MQTT ingest uses `_handle_mqtt_message` directly,
not the stricter `process_raw_mqtt` helper.

- `scripts/run_pipeline.py:1747-1757` registers
  `self._handle_mqtt_message` as the live MQTT callback.
- `scripts/run_pipeline.py:938-939` maps falsey payloads, including `""` and
  numeric zero, to an empty string.
- `scripts/run_pipeline.py:1010-1025` defaults an empty plain payload to
  `0.0`; an empty JSON object or an object with none of the fallback keys also
  becomes `0.0`.
- `scripts/run_pipeline.py:1016-1017` accepts alternate keys (`watts`, `W`,
  `value`) and coercible strings/booleans rather than enforcing one schema.
- `scripts/run_pipeline.py:1033-1035` rejects NaN/Inf, but there is no
  negative-value rejection in this live path. Negative values reach
  `last_device_power` at `:1040` and downstream processing.
- The test helper at `scripts/run_pipeline.py:1845-1908` rejects empty,
  negative, NaN, and Inf values, so helper tests do not cover the production
  callback behavior. It also has a different JSON contract: it requires a
  `power` key, while the live handler accepts `watts`, `W`, and `value`.

Direct probe results:

| Payload | Live `_handle_mqtt_message` state | `process_raw_mqtt` helper |
|---|---:|---|
| `""` | `0.0` | `PARSE_ERROR`, rejected |
| `{}` | `0.0` | `PARSE_ERROR`, rejected |
| `-3` | `-3.0` | `SENSOR_ERROR`, rejected |
| `NaN` / `Infinity` | rejected | `SENSOR_ERROR`, rejected |
| `{"watts": 5}` | `5.0` | `PARSE_ERROR`, rejected |
| `{"power": "6"}` | `6.0` | accepted/coerced |
| `{"power": true}` | `1.0` | accepted/coerced |

The four-test audit reproducer confirms the first invalid live mutation before
checking the later cases. The server safety monitor separately applies
`abs()` to negative power (`src/pipeline/safety.py:224-230`), which makes a
negative reading a positive safety input rather than a rejected sensor frame.

### B3 — Retained, duplicate, stale, and forged message semantics are absent — P1, OPEN

There is no message freshness/replay contract at the backend boundary.

- `src/api/main.py:275-277` reads only `message.topic` and `message.payload`;
  retain/duplicate/message-id metadata is not inspected.
- `src/hardware/mqtt.py:43-47` forwards only topic and payload to callbacks.
  The pipeline callback has no timestamp, sequence, boot epoch, or message
  metadata with which to reject replay.
- Plain power payloads carry no timestamp or sequence. A retained old reading
  delivered after reconnect is treated as current; a QoS-1 duplicate is
  processed as a new reading and can repeat persistence/analytics/event work.
- `src/api/main.py:423-431` broadcasts every command payload as a critical
  safety alert without command schema, age, source, or correlation validation.
- `scripts/run_pipeline.py:940-949` clears a device action cooldown for any
  payload on an ACK topic. It does not validate `ON_CONFIRMED`, `OFF_CONFIRMED`,
  command identity, freshness, or an outstanding command.
- `DeviceStatusEvent.timestamp` is accepted and converted at
  `src/api/main.py:288-293`, but no finite/range/monotonicity check is applied.
  A missing timestamp is replaced with receipt time, so receipt time does not
  distinguish retained/replayed data.
- The firmware/simulator command path accepts exact `ON`/`OFF` payloads and
  has lockout handling, but MQTT retain/duplicate metadata is not available to
  the backend contract. The retained `ON`/reconnect/lockout matrix therefore
  remains an edge integration test, not a backend proof.

Unauthorized command exposure is amplified by the broad ACL write grant and
the shared identity described in B1. The server must not infer authorization
from the payload string alone.

### B4 — Reconnect and readiness state are not authoritative — P1, OPEN

- API reconnect handling (`src/api/main.py:433-440`) updates status only for
  `aiomqtt.MqttError`. Decode, validation, callback, subscribe, and other
  runtime errors can kill the listener without setting `mqtt_reconnecting`.
- On successful context entry, status becomes `connected` before all
  subscriptions have been proven (`:265-273`). There is no explicit
  subscription-ready state or task-health registry.
- `_mqtt_connect_kwargs()` parses `MQTT_PORT` before the client call. A bad
  environment value raises a non-MQTT exception and is not handled by the
  reconnect loop.
- `MQTTClientManager` starts with `_connected = True` and
  `is_connected()` returns true when `client is not None` without checking the
  actual broker session (`src/hardware/mqtt.py:16-26`). This is optimistic
  state, not a successful connection proof.
- `run()` retries after `aiomqtt.MqttError`, but a normal messages-iterator
  termination has no explicit state transition/backoff (`:28-58`). The API and
  pipeline therefore have different reconnect/liveness semantics.
- `scripts/run_pipeline.py:1783-1792` cancels child tasks in `finally` but does
  not explicitly await each cancelled safety/ML task before closing the DB.

The current API test suite verifies `/ready` is 503 when globals are absent,
but does not test a dead listener with stale connected globals or a successful
authentication followed by subscription failure.

### B5 — Serial callback processing can block transport progress — P1, OPEN

`MQTTClientManager.run()` awaits `_read_callback` inline for every received
message (`src/hardware/mqtt.py:43-47`). There is no bounded ingress queue,
worker pool, overload policy, or per-topic priority. The live callback performs
CPU-heavy and synchronous work in the event-loop task, including NumPy/Torch
classification and registry save paths (`scripts/run_pipeline.py:1165-1225`,
`:1659-1693`). Database writes are queued asynchronously, but classification
and event handling remain on the MQTT consumer.

Additional blocking/backpressure points:

- `publish_command()` sleeps and retries inside the caller (`:60-99`), so a
  pipeline callback can wait through multiple retry delays.
- API bridge event processing waits for WebSocket fan-out and label capture
  before consuming the next MQTT message (`src/api/main.py:313-340`,
  `:396-400`).
- Safety monitoring is on a separate MQTT connection, which protects it from
  some ML work, but its callback still awaits alert publication and threaded
  log I/O serially (`src/pipeline/safety.py:255-281`).
- State locks protect selected bounded lists only. Device maps and the power
  aggregation buffer are mutated/read without a common snapshot lock.

No latency, queue-depth, broker backlog, or event-loop-lag acceptance test was
run against a real broker.

### B6 — Memory bounds are partial and do not cover the ingress path — P1, OPEN

Confirmed bounds:

- API pending labels/low-confidence/safety/mitigation lists are capped at
  50/100/50/50 entries; signature count is capped at 200.
- Pipeline deques for NILM and unknown signatures have fixed local lengths;
  latency samples are capped at 100.

Unbounded or incompletely evicted state:

- `src/api/main.py:81-92` has no cap/TTL for `system_state["devices"]`,
  `phantom_loads`, or arbitrary analytics content.
- `_ws_power_buffer` (`src/api/main.py:104`) can grow once per unique device
  between one-second flushes; no device-ID or payload bound is enforced.
- MQTT UI event payloads have no byte cap and `LABEL_REQUEST` segments have no
  inbound count/dimension cap before being copied into state/signature buffers
  (`src/api/main.py:135-146`, `:313-340`). A count cap on signatures does not
  cap the size of each entry.
- `manager.active_connections` and the test `AsyncMQTTClient`'s
  `published_messages` list have no hard capacity.
- `DatabaseSession._write_queue` is an unbounded `asyncio.Queue` (observed
  `maxsize == 0`). A database outage can therefore accumulate all accepted
  readings in memory.
- `_evict_stale_devices()` removes many per-device maps but omits
  `_unknown_windows` (`scripts/run_pipeline.py:1814-1822`). A probe with 201
  stale IDs left 201 unknown-window entries after the device index was reduced
  to the configured 200.
- The telemetry branch updates `_device_last_seen` and `device_telemetry`
  (`scripts/run_pipeline.py:976-985`) but does not invoke eviction. A
  telemetry-only topic flood is therefore not bounded by the configured
  device capacity.
- Database `unmapped_clusters` has no retention/maximum policy
  (`src/database/session.py:51-69`, `:240-302`).

The existing chaos test named `test_mqtt_published_messages_list_growth`
asserts that an unbounded list reaches length 100, rather than asserting a
bound, so it is not a capacity regression test.

## Focused verification performed

All commands used the matching runtime documented in `CURRENT_STATE.md`:

| Command | Result |
|---|---|
| `pytest tests/test_audit_reproductions.py -q --tb=short` | **4 failed**, as intended: H1, H2, parser, B2 red reproductions |
| `pytest tests/test_mqtt.py -q --tb=short` | **5 passed** |
| `pytest tests/test_api.py tests/test_api_extended.py -q --tb=short` | **35 passed** |
| Focused security MQTT/parser selection | **9 passed**, 40 deselected |
| `pytest tests/test_database.py -q --tb=short` | **6 passed**, plus one `PytestUnhandledThreadExceptionWarning` from an aiosqlite worker after loop close |
| Focused chaos MQTT/lifecycle selection | **6 passed**, 39 deselected |
| `docker compose config --quiet` | Exit 0; Docker warned that compose `version` is obsolete |
| Repository host Mosquitto startup probe | Blocked by invalid passwd salts and port collision |
| Anonymous connection to the already-listening `127.0.0.1:1883` broker | **Accepted**; not the repository broker |

The normal passing tests are not contradictory: they use `AsyncMQTTClient`,
`MockMQTTBroker`, direct helper methods, or ASGI transport. They do not prove
Mosquitto authentication/ACL behavior, retain/duplicate metadata, reconnect
resubscription, or bounded overload behavior.

## Test gaps and real-broker requirements

### Broker bring-up prerequisites

1. Establish exactly one owner of the test port. Stop/disable the system
   broker or use an isolated listener/configuration; do not run the repo
   broker, Docker broker, and demo launcher simultaneously.
2. Generate a valid hashed Mosquitto password file on the deployment host.
   Define the actual `ems_pipeline` and dedicated per-device `esp32` account(s)
   and decide whether `ems_api` is retained as a separate identity.
3. Require `allow_anonymous false`, verify password/ACL file permissions, and
   use non-default deployment secrets. Do not treat the anonymous system broker
   observed above as a test oracle.
4. Resolve identity separation before release: API needs the bridge reads and
   label publish; pipeline needs sensor/ACK/label reads and command/UI/label
   writes; an API identity must not inherit command-write permission. Device
   credentials should be limited to their own command and telemetry topics.

### Required authenticated integration matrix

Run with `mosquitto_pub`/`mosquitto_sub` or equivalent authenticated clients,
recording CONNACK/ACL results and broker logs:

| Area | Required assertions |
|---|---|
| Authentication | Anonymous, wrong password, missing user, and valid-user attempts; only valid identities connect. |
| ACL least privilege | Each identity can perform every required read/write and every unlisted topic operation is denied. Explicitly test UI-event read, command read, command write, ML-label read/write, `$SYS` health read, and cross-device topics. |
| API bridge | Pipeline UI event reaches API WebSocket; power/command subscriptions behave according to the final identity contract; `/ready` is not green before authenticated subscribe completion. |
| Malformed input | Invalid UTF-8, invalid JSON, JSON list/scalar, missing fields, wrong types, empty/object/negative/nonfinite/oversized power, oversized UI event, and unknown event type. After each frame the listener remains alive, readiness/status is truthful, and no invalid state/broadcast is produced. Send a valid frame afterward to prove recovery. |
| Replay/freshness | Retained old `ON`, retained telemetry, QoS-1 duplicate, stale timestamp, repeated ACK, wrong ACK payload, and out-of-order sequence. Assert no stale ON, no duplicate persistence/action, and no stale ACK cooldown clear. |
| Reconnect | Kill/restart the broker; assert explicit disconnected/reconnecting/ready transitions, client reference cleanup, resubscription to every required topic, bounded retry/backlog, and one delivery per accepted message after recovery. |
| Command authorization | A device cannot receive another device's command; API cannot publish relay commands; pipeline command publication is allowlisted and correlated to an authorized action; retained/duplicate commands remain safe. |
| Concurrency | Flood telemetry and UI events while introducing a slow WebSocket, slow classifier, DB outage, and publish failure. Measure event-loop lag, callback latency, broker backlog, task count, and safety-monitor independence. |
| Capacity | Set explicit limits and assert RSS, ingress queue, DB queue, per-device maps, WebSocket clients, payload bytes, database size, fallback CSV size, and cluster/signature storage remain bounded over a long deterministic run. |
| Health | `/health` remains liveness-only; `/ready` is 503 for auth failure, dead listener, incomplete subscriptions, unavailable DB, and stale task state, and returns 200 only after all declared dependencies are genuinely usable. |

Required artifacts are broker version/config checksum, non-secret identity
matrix, ACL decision log, exact test payloads with secrets redacted, reconnect
timestamps, queue/RSS samples, and the final readiness timeline.

## Final disposition

**Fixed:** no source fix was authorized or applied; this agent produced the
findings document only.
**Verified:** the focused reproductions and protocol probes above were run; the
four audit defects remain reproducibly open.
**Remaining blocker:** authenticated real-broker validation is blocked by the
existing broker collision and invalid repository passwd entries, and physical
validation was not run.
