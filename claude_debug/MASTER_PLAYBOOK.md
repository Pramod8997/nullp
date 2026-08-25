# CLAUDE.md: Master Playbook & Technical Guidelines

> **Project:** Digital Twin Smart Energy Monitoring & Disaggregation System (EMS)  
> **Target Folder:** `claude_debug/`  
> **Role:** Principal Embedded Systems QA Architect, Lead ML Test Engineer & Senior Backend Engineer  
> **Target Models:** Claude 3.5 Sonnet / Claude 3.7 Sonnet / Claude Opus 4.5 & 5  
> **Current Health:** 511/511 Tests Passing (100%) | Real UK-DALE & REDD Data Integrated | Physical Stress Verified

---

## 1. Token Economy & Development Rules (STRICT)

1. **Zero Fluff & Concise Output:** Output only direct code diffs, command executions, and 1–2 sentence operational summaries. Do NOT write conversational filler, restate prompt requirements, or provide unrequested lengthy explanations.
2. **Do Not Overcomplicate:** Fix root causes cleanly and directly in place. Never introduce speculative abstractions, wrapper classes, or unnecessary architectural refactors that trigger secondary cascading bugs.
3. **Graph-First Architecture Navigation:** NEVER dump or recursively traverse the directory tree. Use the pre-built knowledge graph at `graphify-out/` via `graphify query "<topic>"` or `graphify path "<A>" "<B>"` to retrieve scoped subgraphs in $<500$ tokens.
4. **Zero API Hallucinations:** Never invent class names or method signatures. Consult [`claude_debug/ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md).
5. **AST Synchronization:** After modifying any code file, execute `graphify update .` (AST-only, zero API cost).
6. **No Regressions:** Verify with `python -m pytest tests/ -q` (baseline is **511 passing tests**).

---

## 2. Essential CLI Commands

```bash
# Activate environment
source .venv/bin/activate

# 1. Query Codebase Knowledge Graph (Fast, token-efficient)
graphify query "<question>"
graphify path "<nodeA>" "<nodeB>"

# 2. Run All 511 Regression Tests
python -m pytest tests/ -q

# 3. Run Real Data Provenance & Fallback Suite (36 tests)
python -m pytest tests/test_real_data_and_ml_fallback.py -v

# 3b. Run ML Recognition & Label-Loop Suite (35 tests — M-5…M-8 + OpenMax deadness)
python -m pytest tests/test_ml_pipeline_recognition.py -v

# 4. Run Real-World Physical Stress & HIL Harness
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py

# 5. Run Core 5 Stress & Chaos Suites (216 tests)
python -m pytest tests/test_hil_uart_corruption.py \
                 tests/test_relay_safety_boot_brownout.py \
                 tests/test_ml_nilm_math_stress.py \
                 tests/test_security_penetration.py \
                 tests/test_chaos_engineering.py -v --tb=short

# 6. Keep Graph Synchronized
graphify update .
```

---

## 3. High-Level Architecture & Component Map

```mermaid
graph TD
    subgraph Edge [ESP32 Firmware Node (Dual-Core FreeRTOS)]
        PZEM["PZEM-004T v3.0 (UART Modbus RTU)"] -->|GPIO 16 RX / 17 TX| Core0["Core 0: SafetySamplingTask (100ms)"]
        Core0 -->|Overcurrent > 125% or dP/dt > 1000W/s| Relay["GPIO 18 Relay — ACTIVE-HIGH (RELAY_ACTIVE_LOW=false)"]
        Core0 -->|portMUX_TYPE sharedMux spinlock| Core1["Core 1: Arduino Loop + MQTT Task"]
    end

    subgraph Transport [MQTT Message Bus]
        Core1 -->|home/sensor/{id}/power (1Hz plain float)| Mosquitto["Mosquitto MQTT Broker (Port 1883)"]
        Core1 -->|home/sensor/{id}/telemetry (JSON)| Mosquitto
        Mosquitto -->|home/plug/{id}/command (ON/OFF/WARNING)| Core1
    end

    subgraph Backend_Pipeline [Server-Side Pipeline]
        Mosquitto --> Safety["FleetDiagnosticsMonitor (Agg > 3500W / Demo 600W)"]
        Mosquitto --> NILM["NILMTransientDetector (Savitzky-Golay + diff)"]
        NILM --> ProtoNet["ProtoNet Embedding Network (General / Demo Weights)"]
        ProtoNet --> Fallback["HeuristicApplianceClassifier (Centroid Fallback)"]
        ProtoNet --> Calib["TemperatureScaler (T >= 0.05) + confidence_gate(0.90)"]
        Calib --> Watchdog["SoftAnomalyWatchdog (Rolling Z-Score)"]
        Calib --> RL["RL Load Shedding Agent (PPO / DQN)"]
        RL -->|home/plug/{id}/command| Mosquitto
    end
```

> ⚠️ **`OverlapAwareNILMDetector` is deliberately NOT shown as a pipeline stage.**
> It is exported from `src/pipeline/__init__.py` and exercised by
> `scripts/stress_test_hardware_sim.py` and the tests, but it is **never instantiated
> in the running pipeline** — `scripts/run_pipeline.py` uses plain
> `NILMTransientDetector`. It is also unreachable at its own defaults: the base
> detector enforces a 5 s post-detection cooldown, so two detections can never fall
> inside `overlap_window_s = 3.0`. Open item **M-2**: wire it in with
> `overlap_window_s > 5.0`, or delete it. Do not treat it as live until then.

---

## 4. Anti-Hallucination API Quick Reference

| Class | Correct Methods & Attributes | Forbidden Hallucinations (DO NOT USE) |
| :--- | :--- | :--- |
| **`ESP32FirmwareNode`** | `set_relay(bool)`, `core0_safety_step(sim_dt)`, `handle_mqtt_command(str)`, `core1_telemetry_tick()`, `.gpio18_relay_state`, `.relay_locked`, `.lock_start_time`, `.pzem`, `.shared_power_watts` | ❌ `.relay_state`, ❌ `.sensor`, ❌ `.wifi_connected`, ❌ `.relay_pin`, ❌ `.process_reading()` |
| **`VirtualPZEM004T`** | `set_load(target_watts, pf)`, `.voltage`, `.current`, `.active_power`, `.power_factor`, `.energy_kwh` | ❌ `.parse_modbus_frame()`, ❌ `.read_power()`, ❌ `.set_voltage()` |
| **`AsyncMQTTClient`** | `subscribe(topic)`, `publish(topic, payload)`, `disconnect()`, `reconnect()`, `is_connected()`, `get_published()`, `.published_messages`, `._connected` | ❌ `.send()`, ❌ `.connected`, ❌ `.connect()` |
| **`MockMQTTBroker`** | `register(client)`, `unregister(client)`, `await disconnect_all()`, `await restart()` | ❌ Sync `disconnect_all()` (must `await`), ❌ `.kill()` |
| **`FleetDiagnosticsMonitor`** | `check_aggregate(power_map)`, `check_roc(device, prev, curr, dt)`, `check_device(device, power)`, `_log_event_sync()`, `_log_event_async()` | ❌ `.update_reading()`, ❌ `.log_event()`, ❌ `.trigger_safety_event()`, ❌ `.is_heartbeat_lost()` |
| **`NILMTransientDetector`** | `push(power_w) -> (bool, np.ndarray)`, `reset()`, `._buffer`, `._cooldown` | ❌ `.detect()`, ❌ `.add_sample()`, ❌ `.process()` |
| **`SoftAnomalyWatchdog`** | `update(device_id, reading) -> (bool, float)`, `.window_size`, `.threshold` | ❌ `.check()`, ❌ `.is_anomaly()`, ❌ `.add_reading()` |
| **`HeuristicApplianceClassifier`** | `classify(window_128) -> HeuristicResult`, `extract_features(window)`, `feature_vector(f)`, `classify_batch(windows)`, ctor `allowed_classes=`, `reject_radius=`, `on_threshold_w=`, `.rules`, `.centroids`, `._extra_centroids` | ❌ `.predict()`, ❌ `.infer()` |
| **`plausible_classes`** (module fn, `heuristic_fallback`) | `plausible_classes(features, rules=None, slack=ENVELOPE_SLACK) -> set` — the physical gate; empty set means UNKNOWN | ❌ passing a raw window (it takes `extract_features` output) |
| **`PrototypeRegistry`** | `add_class(name, (K,128) watts)`, `classify(segment) -> (name, d2, dists)`, `power_envelope(name) -> Optional[(lo,hi)]`, `class_names()`, `save/load(path)`, `.prototypes`, `.envelopes`, `.ENVELOPE_KEY` | ❌ enrolling embeddings as `support_segments`, ❌ thresholding `d2` to detect novelty (scale-blind), ❌ `.predict()` |
| **`EMSOrchestrator`** (`FullPipeline`) | ctor `config=<dict>`, `_classify_device(device_id, power_watts, filtered_segment=None) -> (name, conf, dists)`, `handle_label_submitted(name, segments_list)`, `_eligible_classes()`, `_registry_heuristic(names)`, `_classify_heuristic(window)`, `.recognition_threshold`, `.heuristic_min_confidence`, `.prototype_registry` | ❌ `config_path=` kwarg, ❌ `EMSPipeline`, ❌ classifying via `.support_manager` (empty on demo/hardware profiles) |
| **`TemperatureScaler`** | `forward(logits) -> Tensor`, `calibrate(logits, labels)`, `temperature_scale(logits, T)`, `confidence_gate(prob, threshold=0.90)` | ❌ `.predict()`, ❌ `.scale()` |

---

## 5. Master Documentation Index (`claude_debug/`)

* 📄 [`claude_debug/INDEX.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/INDEX.md) — Master Navigation Index
* 📄 [`claude_debug/PROMPT.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PROMPT.md) — Ultra-Dense Master Prompt for Claude Opus 5
* 📄 [`claude_debug/PRD.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PRD.md) — Product Requirements Document
* 📄 [`claude_debug/TECHNICAL_REVIEW.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/TECHNICAL_REVIEW.md) — Technical Review & FreeRTOS Diagrams
* 📄 [`claude_debug/ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md) — Complete API Cheatsheet
* 📄 [`claude_debug/HARDWARE_FINAL_SPEC.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_FINAL_SPEC.md) — 🔒 **AUTHORITATIVE Physical Hardware Spec.** Locked build: single aggregate node, ~600 W consumer electronics, India 230 V. PZEM 10 A direct-connect, SRD 10 A relay, BSS138 + 100 kΩ pull-down, 5 A load fuse. Relay net polarity is **ACTIVE-HIGH** (`RELAY_ACTIVE_LOW = false`).
* 📄 [`claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md) — Real-World Hardware Hazards & Schematics (BOM superseded by the spec)
* 📄 [`claude_debug/HARDWARE_READINESS_CHECKLIST.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_READINESS_CHECKLIST.md) — Pre-Procurement Review & Bring-Up Gate (order tables superseded by the spec)
* 📄 [`claude_debug/REAL_WORLD_TESTING_PLAN.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/REAL_WORLD_TESTING_PLAN.md) — 8 Physical Bench Tests & Protocols
