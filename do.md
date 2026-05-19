Here is the complete, unified master document combining the Phase 1 Codebase Audit, the Hardware Bridging Strategy, the Phase 2 Deployment Plan, and the Journal Evaluation Protocol.

You can copy the entire block below in one go and save it as your `Phase2_Master_Plan.md`.

```markdown
# Comprehensive Phase 1 Audit & Phase 2 Execution Master Plan
**Project:** Confidence-Aware Digital Twin Energy Management System
**Target:** Phase 2 Hardware Deployment, Cyber-Physical Integration, and Academic Publication
**Status:** 🔴 **Action Required** (Remediate Phase 1 bugs prior to hardware deployment)

---

## PART 1: Phase 1 Codebase Audit & Required Remediation
Before physical relay actuation is permitted, the following critical bugs and architectural disconnects in the Phase 1 Python simulation must be patched.

### 1.1 CRITICAL: The "Label Submission" Architectural Break
**Location:** `src/api/main.py` and `scripts/run_pipeline.py`
**The Bug:** When a user labels an unknown device via the REST API (`/api/submit-label`), it only triggers a WebSocket broadcast. The ML orchestrator (`run_pipeline.py`) strictly listens to MQTT and cannot hear WebSocket events. The `handle_label_submitted()` function is dead code, meaning the ProtoNet never actually learns new devices.
**The Fix:** Bridge the API back to the Orchestrator using MQTT.
* **In `main.py` (`label_device`):** Publish the segments to a new MQTT topic:
    ```python
    async with aiomqtt.Client("localhost", port=1883) as client:
        await client.publish("home/ml/label", json.dumps({"class_name": req.class_name, "segments": req.segments}))
    ```
* **In `run_pipeline.py` (`_handle_mqtt_message`):** Subscribe to `home/ml/label` and invoke the handler:
    ```python
    elif "/label" in topic:
        payload_dict = json.loads(payload)
        self.handle_label_submitted(payload_dict["class_name"], payload_dict["segments"])
    ```

### 1.2 HIGH: Digital Twin Physics Reset Bug (State Obliteration)
**Location:** `scripts/run_pipeline.py` -> `_handle_mqtt_message()`
**The Bug:** When updating the room temperature, the orchestrator passes the active power state to the Digital Twin but zeros out every other device because it only looks at the *current* MQTT tick. Thermal accumulation is drastically underestimated, breaking the PMV empathy gate.
**The Fix:** Pass the actual stored state of all devices, applying the live tick only to the active device.
```python
appliance_watts = {k: (self.power_windows[k][-1] if k in self.power_windows and len(self.power_windows[k])>0 else 0) for k in self.device_states}
appliance_watts[device_id] = power_watts  # Apply the live tick

```

### 1.3 HIGH: Conflicting Unknown Device API Endpoints

**Location:** `src/api/main.py`
**The Bug:** There are two disjointed APIs. `/api/unknown_devices` relies on a global list `pending_unknowns_store` that is *never* populated, returning empty arrays 100% of the time.
**The Fix:** Delete `/api/unknown_devices` and `pending_unknowns_store`. Standardize the REST UI to strictly use `/api/pending-labels` (to fetch requests populated by the pipeline) and `/api/submit-label` (to classify them).

### 1.4 MEDIUM: RL Hyper-Exploration Decay (Epsilon Burnout)

**Location:** `src/rl/agent.py` -> `update()`
**The Bug:** Epsilon decays after *every single TD update*. At a 1Hz publish rate across 10 devices, a decay factor of `0.999` drops Epsilon from `0.3` to `0.01` in exactly 340 seconds. The RL agent permanently stops exploring before the Digital Twin experiences even a fraction of a simulated day.
**The Fix:** Decay epsilon on a time-scheduled interval (e.g., once an hour simulated time), or change the decay rate to a microscopic value (e.g., `0.999995`).

### 1.5 MEDIUM: Database Data Loss on Shutdown & CSV Path Crash

**Location:** `src/database/session.py` and `scripts/run_pipeline.py`

* **Bug A (CSV Path):** `os.makedirs(os.path.dirname("fallback.csv"))` evaluates to an empty string and crashes the fallback writer. **Fix:** Add a check `if directory: os.makedirs(...)`.
* **Bug B (DB Shutdown):** Items sitting in `self._write_queue` that haven't been popped into the batch yet are permanently lost when the system shuts down. **Fix:** Add a `while not self._write_queue.empty():` drain loop inside the `except asyncio.CancelledError:` block of `_flush_loop()`.

---

## PART 2: Hardware-to-Model Bridging Strategy (Domain Mismatch)

The ML models were trained on 230V UK-DALE data (loads of 50W to 3000W). If the Phase 2 ESP32 testbed uses safe, low-voltage 5V/12V components (drawing 1W to 5W), the $\pm50$W transient detector and the ProtoNet will fail instantly. Choose ONE of the following strategies:

### Option A: Virtual Scaling Layer (Software Fix - Recommended)

Intercept the low-power physical readings and scale them up to virtual high-power equivalents *before* they hit the ML pipeline.

* **Config (`config.yaml`):** Define a scaling multiplier (e.g., `1 physical Watt = 500 virtual Watts`).
* **Pipeline (`run_pipeline.py`):** Multiply incoming `power_watts` by the multiplier right after parsing the MQTT payload.
* **Firmware (`main.cpp`):** Ensure the ESP32 calculates its 125% Tier-0 cutoff threshold based on the *physical* limit (e.g., 2.5W), not the virtual 3000W limit.

### Option B: High-Draw 12V/24V Proxy (Hardware Fix)

Use safe, low-voltage DC hardware that naturally pulls massive current to trigger the $\pm50$W ML thresholds without scaling.

* **Hardware:** Power the testbed with a PC ATX power supply. Use 12V Halogen car bulbs (55W) or 3D printer heated beds (120W) as proxy appliances.

---

## PART 3: Phase 2 Execution & Deployment Plan

Transitioning from synthetic UK-DALE simulation to the cyber-physical edge system.

### 3.1 Hardware Layer (ESP32 Tier-0 Safety)

* **Assembly:** ESP32 DevKit V1 + SCT-013-030 CT Sensor + 33Ω Burden Resistor + 5V/10A Relay Module.
* **Firmware (`main.cpp`):** * Sample ADC1_CH6 for RMS current.
* Hold local `critical_pct` threshold (125%). If breached, toggle GPIO 5 to `LOW` in $<50$ms independently of MQTT.
* Publish $P = V \times I \times PF$ to `home/sensor/{device_id}/power` at 1Hz.
* Subscribe to `home/plug/+/command` to receive RL shedding commands.



### 3.2 Environment Setup

* Remove `simulate_esp32.py` from the Makefile execution stack.
* Use `esptool.py` and CP210x drivers to flash the edge nodes.
* Utilize the **NILMTK toolkit** to stream real-world UK-DALE/REDD traces through Mosquitto to benchmark alongside live ESP32 nodes.

---
# Master Codebase Audit & Bug Report
**Project:** Confidence-Aware Digital Twin EMS
**Target:** Phase 2 Real-World Deployment
**Status:** 🔴 **CRITICAL ACTION REQUIRED** (Contains 20+ execution-blocking bugs)

---

## PART 1: Reinforcement Learning & ML Pipeline Bugs
*These bugs destroy the Markov Decision Process (MDP), prevent the agent from learning, or block data from reaching the model.*

### 1.1 RL Agent "State Blindness" (MDP Rupture)
* **Location:** `scripts/run_pipeline.py`
* **Bug:** When constructing the RL state, only the *currently ticking* device is passed to the agent (`"devices": {class_name: pct_of_rated}`).
* **Impact:** The agent's `total_pct` and `active_count` only ever reflect 1 device. The agent is completely blind to aggregate house load and will never learn grid-level shedding.
* **Fix:** Iterate through `self.device_states` and pass the total known house state into `state_dict["devices"]`.

### 1.2 RL Agent "Blind" to Energy Savings (Reward Logic Flaw)
* **Location:** `src/rl/agent.py` (`compute_reward`)
* **Bug:** The cost penalty is calculated using `current_watts` (pre-action wattage) regardless of whether the agent chose to `SHED` or `DEFER`.
* **Impact:** The agent receives the exact same financial penalty even if it turns a device off. It will never learn energy-saving behaviors.
* **Fix:** Calculate reward using `projected_watts = 0.0 if action == "SHED" else current_watts`.

### 1.3 RL Global Policy Overwrite (State Space Flaw)
* **Location:** `src/rl/agent.py` (`_discretize`)
* **Bug:** The state string (`load:X::active:Y::price:Z::pmv:W`) does not include the target appliance class.
* **Impact:** If the agent learns that `SHED` is good in State A for an HVAC, it will blindly apply `SHED` to the Fridge the next time State A occurs.
* **Fix:** Pass `classified_device` to `_discretize` and append it to the state hash.

### 1.4 RL Hyper-Exploration Decay (Epsilon Burnout)
* **Location:** `src/rl/agent.py` (`update`)
* **Bug:** Epsilon decays by `0.999` on every MQTT tick. 
* **Impact:** At 10 ticks per second, Epsilon hits rock bottom (`0.01`) in under 6 minutes. The agent permanently stops exploring before the simulation completes a single day.
* **Fix:** Apply epsilon decay on a time-scheduled interval (e.g., hourly), or change the decay factor to `0.999995`.

### 1.5 "Shadow Mode" Delusion
* **Location:** `scripts/run_pipeline.py` (Policy Promotion Gate)
* **Bug:** If the agent is in Shadow Mode, physical relays aren't triggered, but the code still tells the RL agent (`next_state`) that the device dropped to `0.0W`. 
* **Impact:** The agent learns from a delusion. The physical sensor will immediately tick back at full wattage, breaking the Q-table convergence.
* **Fix:** `next_state` must use actual wattage unless `action == "SHED"` AND `is_promoted == True`.

### 1.6 The Transient Gating Bug (Total Data Starvation)
* **Location:** `scripts/run_pipeline.py` (NILM Preprocessing)
* **Bug:** If no transient is detected, `confidence` is forced to `0.0`. The pipeline then hits the `< 0.90` gate and skips the entire block (no DB write, no Analytics, no RL).
* **Impact:** Devices in a steady state are completely ignored by the system.
* **Fix:** Carry over `self.last_known_confidences.get(device_id)` during steady-state ticks.

---

## PART 2: Digital Twin & Physics Engine Bugs
*These bugs corrupt the physical math of the simulation, causing runaway temperatures and false analytics.*

### 2.1 Digital Twin Physics Reset (State Obliteration)
* **Location:** `scripts/run_pipeline.py` (`simulate_step`)
* **Bug:** The script builds the `appliance_watts` dictionary by applying the live tick and assigning `0W` to every other device.
* **Impact:** The twin assumes only 1 device is running at any given millisecond. Total house heat generation is drastically underestimated.
* **Fix:** Map the last known running wattages for all devices from `self.power_windows`, then apply the live tick.

### 2.2 Time-Dilation in Digital Twin
* **Location:** `scripts/run_pipeline.py` (`simulate_step`)
* **Bug:** `dt_minutes` is hardcoded to `1.0/60.0` (1 second) for every tick.
* **Impact:** 10 devices ticking at 1Hz advance the simulation by 10 seconds every 1 real second. The virtual house heats up 10x faster than reality.
* **Fix:** Calculate actual `dt = time.time() - self.last_sim_time`.

### 2.3 The "Empathy Spam" Logic Bomb
* **Location:** `src/rl/agent.py` (`act`)
* **Bug:** The PMV empathy gate forces a `SHED_HVAC` or `SCHEDULE_HVAC` action if the house is hot/cold, regardless of what device triggered the tick.
* **Impact:** A ticking TV or Kettle will output HVAC commands, spamming the UI with 10 empathy alerts a second.
* **Fix:** The empathy gate must check `if classified_device == "hvac"` before taking action.

### 2.4 Analytics Clock Drift
* **Location:** `scripts/run_pipeline.py` (`analytics.record_usage`)
* **Bug:** Usage duration is hardcoded to `1.0 / 3600.0` hours per tick.
* **Impact:** Network jitter or dropped ESP32 packets will cause massive under/over-estimation of total kWh.
* **Fix:** Use real timestamp deltas (`dt = current_time - last_seen_time`).

---

## PART 3: API, WebSocket & UI Integration Bugs
*These bugs break communication between the frontend React dashboard and the backend ML pipeline.*

### 3.1 The "Label Submission" Architectural Break
* **Location:** `src/api/main.py` & `scripts/run_pipeline.py`
* **Bug:** Submitting a label via the REST API only triggers a WebSocket broadcast. The ML orchestrator only listens to MQTT. The registry is never updated.
* **Fix:** The REST endpoint must publish the label data to an MQTT topic (e.g., `home/ml/label`), and the orchestrator must subscribe to it.

### 3.2 Dropped Embeddings in API
* **Location:** `src/api/main.py` (MQTT Listener)
* **Bug:** When caching the `LABEL_REQUEST` event to `system_state["pending_labels"]`, the 128D `embedding` array is completely omitted.
* **Impact:** The UI cannot send the embedding back. The ProtoNet cannot learn.
* **Fix:** Add `"embedding": event_data.get("embedding", [])` to the `label_entry` dict.

### 3.3 The Unknown Device UI Spam Loop
* **Location:** `scripts/run_pipeline.py` (Unknown Device Flow)
* **Bug:** After broadcasting a `LABEL_REQUEST`, the device class is never actually updated in memory.
* **Impact:** The system will re-trigger the delta-analyzer and broadcast a new `LABEL_REQUEST` every single second for the same device.
* **Fix:** Set `self.device_classifications[device_id] = pseudo_class` immediately after the first broadcast.

### 3.4 Conflicting & Broken Unknown Device Endpoints
* **Location:** `src/api/main.py`
* **Bug:** `/api/unknown_devices` reads from `pending_unknowns_store`, which is never populated.
* **Fix:** Delete `/api/unknown_devices`. Use `/api/pending-labels` exclusively.

### 3.5 Pydantic Schema Validation Crash
* **Location:** `src/api/main.py` (`LabelSubmission` class)
* **Bug:** The Pydantic model for `/api/submit-label` only accepts `device_id` and `label`. If the UI sends the required 128D embeddings, FastAPI throws a 422 Error.
* **Fix:** Add `segments: List[List[float]]` to the `LabelSubmission` model.

### 3.6 Dead WebSocket Connection Leak & Backpressure
* **Location:** `src/api/main.py` (`ConnectionManager`)
* **Bug:** Broadcasting is done via a sequential `for` loop. Dead (half-open) TCP connections will stall the `await ws.send_json()` call.
* **Impact:** The MQTT listener task will freeze, halting all incoming data processing.
* **Fix:** Use `asyncio.gather` with a strict `asyncio.wait_for(timeout=0.5)` on sends.

---

## PART 4: Database, File I/O & System Stability
*These bugs cause silent data loss or fatal crashes on boot.*

### 4.1 The "Fake" Database Fallback (Silent Data Loss)
* **Location:** `scripts/run_pipeline.py` & `src/database/session.py`
* **Bug:** The `run_pipeline.py` `try/except` block assumes `db.insert_measurement` throws an error on failure. However, that function merely puts data in an `asyncio.Queue` (which never fails). When the background DB writer actually fails, the exception is swallowed.
* **Fix:** Move the CSV fallback logic entirely inside `session.py`'s `_flush_loop`.

### 4.2 Database Data Loss on Shutdown
* **Location:** `src/database/session.py` (`_flush_loop` cancellation)
* **Bug:** On shutdown (`CancelledError`), the flusher writes the active batch but discards all remaining items in `_write_queue`.
* **Fix:** Add a `while not self._write_queue.empty(): batch.append(self._write_queue.get_nowait())` drain loop during cancellation.

### 4.3 Async Event-Loop Blocking (CSV Corruptor)
* **Location:** `scripts/run_pipeline.py` (`_csv_fallback_write`)
* **Bug:** The CSV writer uses synchronous `with open(...)`. This blocks the async event loop. Furthermore, concurrent MQTT ticks can cause race conditions/file corruption.
* **Fix:** Use an `asyncio.Lock()` and the `aiofiles` library.

### 4.4 CSV Fallback Path Crash
* **Location:** `scripts/run_pipeline.py` (`_csv_fallback_write`)
* **Bug:** `os.makedirs(os.path.dirname("fallback.csv"))` resolves to an empty string and throws a fatal `FileNotFoundError`.
* **Fix:** Add `if directory:` before calling `os.makedirs`.

### 4.5 Hardcoded `localhost` Traps
* **Location:** `src/api/main.py` & `scripts/run_pipeline.py`
* **Bug:** The MQTT bridge and parallel safety tasks hardcode `"localhost"`, ignoring the `MQTT_BROKER` environment variables.
* **Impact:** The system will instantly crash if deployed via Docker or a multi-node network.
* **Fix:** Inject `os.environ.get("MQTT_BROKER")` into all `aiomqtt.Client` instantiations.

### 4.6 `asyncio` task_done() Without join()
* **Location:** `src/database/session.py` (`_flush_loop`)
* **Bug:** Calling `task_done()` without an accompanying `join()` can trigger a `ValueError` if calls get desynchronized.
* **Fix:** Delete all `self._write_queue.task_done()` lines.





## PART 4: Journal-Grade Evaluation & Testing Protocol

To target high-impact journals (e.g., *IEEE Transactions on Smart Grid*), execute these tests to populate the "Results and Evaluation" section.

### 4.1 Phase A: Few-Shot NILM & Feature Extraction

* **N-Way K-Shot Benchmarking:** Prove the ProtoNet handles the "Cold Start" problem better than standard CNNs/LSTMs using 5-way 5-shot episodic tests.
* **Cross-Domain Generalization:** Train exclusively on UK-DALE (230V) and evaluate on REDD (120V) using only 5 support samples per device to prove domain-invariant feature extraction.

### 4.2 Phase B: Open-Set Calibration & Safety

* **OpenMax AUROC:** Introduce 3 novel device classes not in the registry. Compare OpenMax (Weibull EVT) rejection rates against standard Softmax thresholding to prove safer unknown handling.
* **Expected Calibration Error (ECE):** Plot a reliability diagram to prove Temperature Scaling aligns the network's stated confidence with empirical accuracy, justifying the $0.90$ RL gate.

### 4.3 Phase C: Digital Twin Meta-RL Control

* **PMV Comfort Compliance:** Run a 30-day simulation. Prove the environment stays within ISO 7730 Category A bounds (-0.5 to +0.5) for $\ge 95\%$ of time steps.
* **Economic Optimization:** Compare total kWh and utility costs against (1) Unmanaged schedules and (2) Rule-Based Time-of-Use logic. Target 15-20% savings.

### 4.4 Phase D: Cyber-Physical Hardware Validation

* **Tier-0 Latency (Oscilloscope Test):** Inject a physical 150% overcurrent fault. Measure the time delta between the CT sensor read and the relay drop (Must be $< 50$ms).
* **Fault-Injection & Graceful Degradation:** Physically kill the MQTT broker and force a Python out-of-memory crash. Verify that local ESP32 hardware thresholds still trigger safely.

### 4.5 Ablation Studies

Provide a performance matrix showing system degradation when removing:

1. SG Preprocessing (Shows drop in Transient F1-Score).
2. OpenMax Weibull (Shows spike in False Acceptance Rates).
3. Temperature Scaling (Shows high-confidence misclassifications).
4. PMV Digital Twin (Shows RL maximizing profit but freezing occupants).

```

```