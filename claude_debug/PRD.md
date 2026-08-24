# Product Requirements Document (PRD)

## Project: Digital Twin Smart Home Energy Monitoring & Disaggregation System (EMS)

> **Document Version:** 2.0.0  
> **Status:** Phase 1 Complete (Simulation & Verification) | Phase 2 Active (Hardware Deployment)  
> **Target Models:** Claude (Opus 4.5 / 5 / Sonnet) & Technical Lead Agents

---

## 1. System Vision & Value Proposition

The Smart Home EMS is an industrial-grade, edge-hybrid smart energy monitoring and safety management platform. It pairs edge-local microcontrollers (ESP32) running dual-core FreeRTOS firmware with a cloud/edge Python backend executing Non-Intrusive Load Monitoring (NILM), rolling z-score anomaly detection, temperature-calibrated few-shot classification, and Reinforcement Learning (RL) load shedding.

### Core Value Pillars
1. **Zero-Cloud Edge Safety (Tier-0):** Safety-critical disconnects (overcurrent $>125\%$, arc-fault proxy $dP/dt > 1000\text{W/s}$) execute locally on the ESP32 in $<100\text{ms}$ with zero network or broker dependencies.
2. **High-Accuracy Sub-Meter Disaggregation:** Identifies individual appliance energy consumption from an aggregate mains line without deploying expensive smart plugs on every socket.
3. **Reinforcement Learning Demand-Response:** Dynamically balances household wattage beneath grid limits ($3,500\text{W}$) without compromising critical appliances (e.g., medical devices, refrigerators).
4. **Resilience & Fault-Tolerance:** Withstands sustained network drops, broker crashes, frame corruption, and clock skew without service interruption.

---

## 2. System Personas & Use Cases

* **Primary Persona — Homeowner / Energy Auditor:** Requires real-time visibility into appliance energy usage, instantaneous hazard isolation, and automated peak-tariff cost savings.
* **Secondary Persona — Microgrid / Grid Operator:** Demands guaranteed aggregate household power clamping to prevent transformer brownouts.
* **Tertiary Persona — Embedded Systems QA Engineer:** Requires automated HIL regression suites, fault injection, and chaos engineering verification.

---

## 3. Functional Requirements

### 3.1 Edge Firmware & Safety (ESP32 + FreeRTOS)

| ID | Requirement | Acceptance Criteria |
| :--- | :--- | :--- |
| **FR-FW-01** | **Dual-Core FreeRTOS Partitioning** | Core 0 executes `SafetySamplingTask` at 100ms interval (Priority 2). Core 1 executes Arduino `loop()` for WiFi, MQTT, and telemetry. |
| **FR-FW-02** | **Active-LOW Relay Safety Default** | GPIO 18 defaults to `HIGH` (Relay OFF) during boot, brownouts, and resets prior to WiFi and MQTT handshakes. |
| **FR-FW-03** | **Overcurrent Cutoff** | Immediate relay cutoff if power exceeds $125\%$ of rated appliance wattage (`CRITICAL_PCT = 1.25`). |
| **FR-FW-04** | **Edge Arc-Fault Proxy ($dP/dt$)** | Immediate relay cutoff if power rise rate exceeds $1,000\text{W/s}$ (`EDGE_ROC_THRESHOLD = 1000.0`). |
| **FR-FW-05** | **Inrush Current Suppression** | A 5-sample sliding baseline buffer (`_baseline_ring`) tolerates cold-start motor inrush ($<50\text{W}$ baseline avg) without tripping. |
| **FR-FW-06** | **Anti-Thrashing Cooldown Lockout** | Upon safety trip, relay enters a 300-second (`SAFETY_LOCKOUT_MS = 300000`) hardware lockout. Rejects all remote `ON` commands with `LOCKOUT_NACK`. |
| **FR-FW-07** | **Atomic Shared State** | Core 0 $\leftrightarrow$ Core 1 float data exchange is protected via `portMUX_TYPE sharedMux` spinlocks with zero scheduler overhead. |
| **FR-FW-08** | **PZEM-004T Metering Interface** | Modbus RTU at 9600 baud over `Serial2` (`GPIO 16 RX`, `GPIO 17 TX`), measuring Voltage (V), Current (A), Power (W), Energy (kWh), and Power Factor (PF). |

---

### 3.2 MQTT Communication & Protocol Contracts

| ID | Requirement | Acceptance Criteria |
| :--- | :--- | :--- |
| **FR-MQ-01** | **Telemetry Ingestion Topic** | Publishes 1Hz power data to `home/sensor/{device_id}/power` as a plain float string (e.g., `"145.8"`). |
| **FR-MQ-02** | **Diagnostics Telemetry Topic** | Publishes 10s diagnostic bundle to `home/sensor/{device_id}/telemetry` formatted as `{"v": 230.0, "i": 1.2, "w": 276.0, "pf": 0.95}`. |
| **FR-MQ-03** | **Command Actuation Topic** | Subscribes to `home/plug/{device_id}/command` accepting only exact `ON`, `OFF`, and `WARNING` payloads. |
| **FR-MQ-04** | **Payload Bounds Enforcement** | Payloads exceeding `MAX_MQTT_PAYLOAD = 256` bytes are dropped at the network buffer layer. |
| **FR-MQ-05** | **Safety QoS Contract** | Safety commands use QoS 1 with 3-retry exponential backoff; telemetry uses QoS 0. |

---

### 3.3 Machine Learning, NILM & Optimization Pipeline

| ID | Requirement | Acceptance Criteria |
| :--- | :--- | :--- |
| **FR-ML-01** | **Savitzky-Golay Transient Detection** | `NILMTransientDetector` applies an SG filter ($w=7, p=2$) and first derivative with $20\text{W}$ threshold and $5\text{s}$ cooldown to capture transient onset signatures. |
| **FR-ML-02** | **Multi-Appliance Overlap Handling** | `OverlapAwareNILMDetector` subtracts known appliance baselines from aggregate transients to isolate concurrent device activations. |
| **FR-ML-03** | **ProtoNet Few-Shot Classification** | `ProtoNet` maps 128-sample transient embeddings into Euclidean metric space across 10 UK-DALE appliance classes. |
| **FR-ML-04** | **Temperature Scaling & Confidence Gating** | `TemperatureScaler` applies post-hoc logit scaling ($T \ge 0.05$ clamp); `confidence_gate` passes actions to RL agent only if confidence $\ge 0.90$. |
| **FR-ML-05** | **Rolling Z-Score Anomaly Watchdog** | `SoftAnomalyWatchdog` tracks rolling 30-sample mean/std; flags deviations $\ge 3.0\sigma$ while guarding against zero std floor ($10^{-6}$). |
| **FR-ML-06** | **RL Demand-Response Actor** | RL agent sheds non-critical loads (e.g., EV charger, dishwasher) when aggregate demand approaches $3,500\text{W}$ while preserving Tier-0 critical loads (e.g., fridge). |
| **FR-ML-07** | **Deterministic Heuristic Fallback** | `HeuristicApplianceClassifier` provides zero-torch nearest-centroid classification (confidence capped at 0.75, `degraded=True`) if ProtoNet weights or runtime are unavailable. |

---

## 4. Supported Appliance Classes

### 4.1 Standard Full-House Set (10 Classes)
1. `fridge` (Rated: 200W, Max: 300W, Tier-0: True)
2. `microwave` (Rated: 1200W, Max: 1500W, Tier-0: False)
3. `kettle` (Rated: 2500W, Max: 2500W, Tier-0: False)
4. `hvac` (Rated: 2000W, Max: 2500W, Tier-0: False)
5. `washing_machine` (Rated: 1800W, Max: 2200W, Tier-0: False)
6. `dishwasher` (Rated: 1500W, Max: 2200W, Tier-0: False)
7. `oven` (Rated: 3000W, Max: 2500W, Tier-0: False)
8. `tv` (Rated: 150W, Max: 300W, Tier-0: False)
9. `ev_charger` (Rated: 3500W, Max: 3800W, Tier-0: False)
10. `laptop` (Rated: 65W, Max: 150W, Tier-0: False)

### 4.2 Benchtop Electronics Demo Set (7 Classes — `config.demo.yaml`)
1. `laptop` (Rated: 30–200W; ultrabooks 30–65W, workstation/gaming 70–200W)
2. `desktop_computer` (Rated: 80–350W)
3. `monitor` (Rated: 20–60W)
4. `projector` (Rated: 30–400W; LED 30–50W, lamp 200–400W)
5. `tv` (Rated: 50–250W)
6. `router` (Rated: 5–25W, Tier-0 Critical Network Node)
7. `phone_charger` (Rated: 5–120W; 3–10W trickle/standby tracked via `PhantomTracker`, 18–120W USB-PD / powerbanks classified via NILM + ProtoNet / heuristic fallback)

---

## 5. Non-Functional Requirements

* **Safety Latency:** Edge overcurrent cutoff must execute in $<100\text{ms}$ (single FreeRTOS cycle).
* **Memory Bounds:** Python buffer queues (`deque`) and Ring Buffers must maintain hard `maxlen` bounds (`3 * embed_window = 384` samples) to permit 24/7/365 uninterrupted execution.
* **Test Coverage:** 100% pass rate across 467 unit, integration, HIL, security, chaos, real-data NILM, and heuristic fallback tests.
* **Mains Isolation:** 230V AC mains traces must maintain $\ge 6.3\text{mm}$ creepage clearance from 3.3V/5V DC planes.
* **Disaggregation Transparency:** Advisory classification outputs confidence and degraded tags, with unseen-house cross-validation reported in `training_results/training_report.json`.
