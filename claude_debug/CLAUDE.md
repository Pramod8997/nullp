# CLAUDE.md: Master Playbook & Technical Guidelines

> **Project:** Digital Twin Smart Energy Monitoring & Disaggregation System (EMS)  
> **Target Folder:** `claude_debug/`  
> **Role:** Principal Embedded Systems QA Architect, Lead ML Test Engineer & Senior Backend Engineer  
> **Target Models:** Claude 3.5 Sonnet / Claude 3.7 Sonnet / Claude Opus 4.5 & 5  
> **Current Health:** 467/467 Tests Passing (100%) | Real UK-DALE & REDD Data Integrated | Physical Stress Verified

---

## 1. Token Economy & Development Rules (STRICT)

1. **Zero Fluff & Concise Output:** Output only direct code diffs, command executions, and 1–2 sentence operational summaries. Do NOT write conversational filler, restate prompt requirements, or provide unrequested lengthy explanations.
2. **Do Not Overcomplicate:** Fix root causes cleanly and directly in place. Never introduce speculative abstractions, wrapper classes, or unnecessary architectural refactors that trigger secondary cascading bugs.
3. **Graph-First Architecture Navigation:** NEVER dump or recursively traverse the directory tree. Use the pre-built knowledge graph at `graphify-out/` via `graphify query "<topic>"` or `graphify path "<A>" "<B>"` to retrieve scoped subgraphs in $<500$ tokens.
4. **Zero API Hallucinations:** Never invent class names or method signatures. Consult [`claude_debug/ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md).
5. **AST Synchronization:** After modifying any code file, execute `graphify update .` (AST-only, zero API cost).
6. **No Regressions:** Verify with `python -m pytest tests/ -q` (baseline is **467 passing tests**).

---

## 2. Essential CLI Commands

```bash
# Activate environment
source .venv/bin/activate

# 1. Query Codebase Knowledge Graph (Fast, token-efficient)
graphify query "<question>"
graphify path "<nodeA>" "<nodeB>"

# 2. Run All 467 Regression Tests
python -m pytest tests/ -q

# 3. Run Real Data Provenance & Fallback Suite (34 tests)
python -m pytest tests/test_real_data_and_ml_fallback.py -v

# 4. Run Real-World Physical Stress & HIL Harness
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py

# 5. Run Core 5 Stress & Chaos Suites (209 tests)
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
        Core0 -->|Overcurrent > 125% or dP/dt > 1000W/s| Relay["Active-LOW Relay (GPIO 18)"]
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
        NILM --> Overlap["OverlapAwareNILMDetector (Power Subtraction)"]
        Overlap --> ProtoNet["ProtoNet Embedding Network (General / Demo Weights)"]
        ProtoNet --> Fallback["HeuristicApplianceClassifier (Centroid Fallback)"]
        ProtoNet --> Calib["TemperatureScaler (T >= 0.05) + confidence_gate(0.90)"]
        Calib --> Watchdog["SoftAnomalyWatchdog (Rolling Z-Score)"]
        Calib --> RL["RL Load Shedding Agent (PPO / DQN)"]
        RL -->|home/plug/{id}/command| Mosquitto
    end
```

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
| **`HeuristicApplianceClassifier`** | `classify(window_128) -> HeuristicResult`, `extract_features(window)`, `feature_vector(f)` | ❌ `.predict()`, ❌ `.infer()` |
| **`SoftAnomalyWatchdog`** | `update(device_id, reading) -> (bool, float)`, `.window_size`, `.threshold` | ❌ `.check()`, ❌ `.is_anomaly()`, ❌ `.add_reading()` |
| **`TemperatureScaler`** | `forward(logits) -> Tensor`, `calibrate(logits, labels)`, `temperature_scale(logits, T)`, `confidence_gate(prob, threshold=0.90)` | ❌ `.predict()`, ❌ `.scale()` |

---

## 5. Master Documentation Index (`claude_debug/`)

* 📄 [`claude_debug/INDEX.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/INDEX.md) — Master Navigation Index
* 📄 [`claude_debug/PROMPT.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PROMPT.md) — Ultra-Dense Master Prompt for Claude Opus 5
* 📄 [`claude_debug/PRD.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/PRD.md) — Product Requirements Document
* 📄 [`claude_debug/TECHNICAL_REVIEW.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/TECHNICAL_REVIEW.md) — Technical Review & FreeRTOS Diagrams
* 📄 [`claude_debug/ARCHITECTURE_AND_APIS.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/ARCHITECTURE_AND_APIS.md) — Complete API Cheatsheet
* 📄 [`claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/HARDWARE_DEPLOYMENT_GUIDE.md) — Real-World Hardware Hazards, Schematics & BOM
* 📄 [`claude_debug/REAL_WORLD_TESTING_PLAN.md`](file:///home/pramodsb/Downloads/mjr/claude_debug/REAL_WORLD_TESTING_PLAN.md) — 8 Physical Bench Tests & Protocols
