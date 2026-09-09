# Hardware Alignment Contract — single source of truth

> **Date:** 2026-09-10 · **Purpose:** every hardware-binding value, its source of truth
> (`firmware/esp32_node/src/main.cpp`), and every other place it must appear. If any site
> disagrees with the firmware constant, that is a defect — not a tuning opportunity.
> Enforced by `tests/test_hardware_alignment.py` (Run 0.1) where a test can reach.
> Companion evidence: [`VERIFICATION_LEDGER_2026-09-10.md`](./VERIFICATION_LEDGER_2026-09-10.md).
>
> **Line numbers below are current as of 2026-09-10 and WILL drift.** The symbol name is the
> contract; the line number is a finding aid. Docs cite symbols, not lines (Run 0.5 rule).

---

## 1. The matrix

### 1.1 Power ladder

| Value | Firmware (source of truth) | Must also appear at | Verified state |
| :-- | :-- | :-- | :-- |
| `RATED_WATTS = 250.0` | main.cpp:74 | `config/config.hardware.yaml` `system_safety.max_aggregate_wattage: 250.0` **and** `device_wattage_limits` (`default` + `node_bench_agg` = 250.0) **and** `devices.node_bench_agg.rated: 250`; twin ctor default (`src/hardware/esp32_firmware_sim.py:56` — was 200.0, re-anchored Run 0.2); HIL (`scripts/hil_hardware_test.py:163-166` — parse, never hardcode); parity test asserts `max_aggregate_wattage == RATED_WATTS` | aligned in config+firmware; twin/HIL re-anchor in Wave 1 |
| `CRITICAL_PCT = 1.25` | main.cpp:100 | config `system_safety.critical_pct: 1.25`; twin (`critical_watts = rated_watts * 1.25`, esp32_firmware_sim.py:178) | aligned; trip = 312.5 W |
| `WARNING_PCT = 1.10` | config-side (`warning_pct`) | config `system_safety.warning_pct: 1.10`; WARNING branch in the twin | aligned; WARNING = 275 W |
| Coordination ladder | main.cpp:74 comment + spec §D8 | 312 W trip (1.36 A) → 5 A fuse (1150 W, 3.7×) → 10 A relay/PZEM → 13 A wire | monotonic, verified |
| **WS-A ceiling decision (pending, bench-day)** | `RATED_WATTS` is the one constant that may rise | If it moves to 400/500: firmware constant, config `system_safety` (both keys), twin, HIL, spec D8 — **same commit** (parity-test-enforced). Flash-day rule: projector nameplate ≤ 320 W → 400 W; > 320 W → 500 W | decision deferred to flash day |

### 1.2 Pins & polarity

| Value | Firmware | Must also appear at | Notes |
| :-- | :-- | :-- | :-- |
| `RELAY_PIN = 18` | main.cpp:80 | [`WIRING_STEP_BY_STEP.md`](./WIRING_STEP_BY_STEP.md) §2 (GPIO 18 → relay IN1); [`HARDWARE_FINAL_SPEC.md`](./HARDWARE_FINAL_SPEC.md) §6 checklist; root [`Hardware.md`](../Hardware.md) | Not a strapping pin — no boot-time pull fights the drive |
| `PZEM_RX_PIN = 16` | main.cpp:97 | same three docs (GPIO 16 ← PZEM TX, crossed on purpose) | WROVER uses 16/17 for PSRAM — board must be WROOM-32D |
| `PZEM_TX_PIN = 17` | main.cpp:98 | same three docs (GPIO 17 → PZEM RX) | B-4 logic-level check before first connect |
| `RELAY_ACTIVE_LOW = false` | main.cpp:96 | twin ctor default `relay_active_low=False` (esp32_firmware_sim.py:57); spec D4/D5; Hardware.md; wiring guide §6; `tests/test_relay_safety_boot_brownout.py` (parses it — the drift-guard precedent) | Active-HIGH net: HIGH = closed, LOW = open, Hi-Z = open (fail-safe via the **mandatory 100 kΩ IN→GND pull-down**). Flipping it is the D5 purchase-contingency only, with bring-up Stage 2 re-run |

### 1.3 Arc-fault / inrush / lockout constants

| Value | Firmware | Twin counterpart | Notes |
| :-- | :-- | :-- | :-- |
| `EDGE_ROC_THRESHOLD = 1000.0` W/s | main.cpp:106 | `roc > 1000.0` (esp32_firmware_sim.py:158) | **Never tuned to make a demo pass** (ledger C3) |
| `BASELINE_WINDOW = 5` | main.cpp:107 | `_baseline_ring` length 5 (:89) | sliding baseline sample count |
| `BASELINE_INRUSH_CEIL = 50.0` W | main.cpp:108 | `baseline_avg < 50.0` (:153) | inrush suppression active only below this baseline — this is why projector-onto-≥50 W-baseline trips (expected, C3) |
| `INRUSH_HEADROOM = 100.0` W | main.cpp:109 | `self._last_watts < (baseline_avg + 100.0)` (:153) | tolerated overshoot above baseline during inrush |
| `SAFETY_LOCKOUT_MS = 300000` (5 min) | main.cpp:112 | `safety_lockout_seconds = 300.0` (:74) | lockout expiry honored every tick (Run 0.3) |
| `POWER_FACTOR = 1.0` (reference) | main.cpp:75 | twin PF init 1.0 | PZEM reports measured PF |

### 1.4 Identity & transport

| Value | Source of truth | Must also appear at | Notes |
| :-- | :-- | :-- | :-- |
| `DEVICE_ID = "node_bench_agg"` | `firmware/esp32_node/include/secrets.h` (`EMS_DEVICE_ID`) → main.cpp:58 | `config/config.hardware.yaml` `devices.node_bench_agg`; runbook Gate 1b | Any mismatch = backend **silently ignores every reading** |
| MQTT topics | main.cpp `snprintf` :339-343 — `home/sensor/%s/power`, `home/sensor/%s/telemetry`, `home/plug/%s/command`, `home/sensor/%s/status`, `home/plug/%s/ack` | pipeline subscriptions (`scripts/run_pipeline.py` :1521-1526: config `reads` = `home/sensor/+/power`, plus `home/plug/+/ack`, `home/ml/label`, `home/sensor/+/status`); config `mqtt.topics` (`reads`/`writes`/`events`) | `home/sensor/+/telemetry` subscription is being wired (WS-E.2/Run 4) — see gap G1 |
| MQTT auth | `secrets.h` `EMS_MQTT_USER`/`EMS_MQTT_PASSWORD` | broker passwd (`ems_pipeline`, dedicated `esp32` user + ACL — runbook Gate 0); pipeline env `MQTT_USERNAME`/`MQTT_PASSWORD` | repo `mosquitto/config/acl` carries the grant matrix |
| Board | ESP32-**WROOM-32D** DevKit (spec D10) | spec §2 BOM; wiring guide §1 | **Not WROVER** — PSRAM steals GPIO 16/17 and silently kills metering |

### 1.5 PZEM cadence — the ROC dt note (ledger C4)

Firmware loop period ≈ **0.134–0.16 s effective** (PZEM-004T v3 register cache ≈ 200 ms; the
100 ms poll reads stale registers between refreshes). The arc-fault division therefore uses an
effective dt in [0.134, 0.16] s, not the nominal 0.1 s. The twin must model exactly this
(Run 0.3 pin (a): **ROC dt-rescale clamp to [0.134, 0.16] s; PZEM reads stay instantaneous**).
Consequences that follow from the number: 300 W projector step → ≈1875–2239 W/s (trips);
120 W laptop step → 750–896 W/s (never trips); phone +336 W/s max (never trips). A register-lag
implementation is wrong (breaks every first-sample overcurrent test); any dt ≥ 0.2 s is wrong
(`trigger_arc_fault` 200/0.2 = 1000 is not > 1000).

---

## 2. Known-gap register (what is NOT yet true — do not claim it)

| # | Gap | State |
| :-: | :-- | :-- |
| G1 | **Telemetry subscription** | ~~Backend does not subscribe `home/sensor/+/telemetry`~~ **WIRED (2026-09-10, wave 1/2)**: subscribed, parsed, WS-broadcast, and pinned by a HARD assertion in `tests/test_hardware_alignment.py` (no xfail remains). |
| G2 | **Envelopes pending physical capture** | `registry_path` is deliberately absent from `config.hardware.yaml`; the shipped UK-DALE fallback cannot represent this rig (its `laptop` prototype is a 21 W netbook; `phone_charger` has zero windows above the 20 W on-threshold). Recognition on this rig is **not physically verified** until Run 2 (state-scoped capture per ledger C7). |
| G3 | **QoS-0 reality** | Power/telemetry/label transports are QoS 0 — delivery is best-effort by design. The 503-on-publish-failure fix (Run 0.10) makes failures *visible*, not deliveries guaranteed. |
| G4 | **30- vs 38-pin procurement note** | Spec D6″/BOM specify a **30-pin** DevKit + 30-pin shield; the wiring guide documents the **38-pin** DevKit actually on the bench (its §2 "Finding the pins on a 38-pin DevKit"; shield must match the board you have — a 30-pin shield will not seat a 38-pin board and vice versa). Pins 16/17/18, silkscreen labels, and all firmware constants are identical either way; **trust the silkscreen over any description**. |
| G5 | **secrets.h placeholders** | Current file holds placeholders (`PLACEHOLDER_SSID`, `192.168.1.100`) — firmware builds green and WiFi silently never associates. Real values go in at Gate 1b. |

---

## 3. Twin instantiation classification — 32 sites

32 call sites of `ESP32FirmwareNode(` exist across tests/scripts (counted 2026-09-10). They
split into two classes with different obligations:

| Class | Obligation | Sites |
| :-- | :-- | :-- |
| **Bench-parity** | Must track the shipped firmware constants (parse from `main.cpp`, never hardcode). A constant change (e.g. WS-A ceiling) must flow to these in the same commit. | `tests/test_relay_safety_boot_brownout.py` (12 sites — the parsing precedent; a few sites deliberately use non-bench ratings like 1200 W/0 W as *test conditions*, which is fine — the parsed constants are the anchors); `scripts/real_world_physical_stress.py` scenario-4 `oc_node` (:190); `scripts/test_firmware_and_ai_e2e.py` stage-8 aggregate assert (:318-335); plus the non-instantiation assertion sites: twin ctor default (esp32_firmware_sim.py:56) and `scripts/hil_hardware_test.py:163-166`. |
| **Generic-fleet** | Intentionally arbitrary ratings exercising behavior, not the bench rig. Forcing 250 W here would destroy the test's meaning — **do not "align" these**. | `tests/test_security_penetration.py` (7), `tests/test_chaos_engineering.py` (2), `scripts/test_firmware_and_ai_e2e.py` demo fleet (:67-71 — node_fridge 200 W, node_kettle 2500 W, node_hvac 2000 W, node_microwave 1200 W, node_vacuum 750 W), `scripts/real_world_physical_stress.py` stress loads (:73, :157, :213, :278), `tests/test_hil_uart_corruption.py` (1). |

The distinction is the rule the parity test encodes: **bench-parity sites parse the firmware;
generic-fleet sites are fixtures.** Misclassifying a generic-fleet site as bench-parity (or vice
versa) is how the 250 W divergence at four sites happened (ledger §2a).
