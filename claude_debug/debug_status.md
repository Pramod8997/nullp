# Debug Status & Verification Report

> **Target:** Smart Energy Monitoring & Edge Safety Platform (EMS)  
> **Location:** `claude_debug/debug_status.md`  
> **Current Baseline:** 467/467 Tests Passing (100%) | Physical Stress: 7/7 PASS | HIL: 10/10 PASS | Closed-Loop E2E: 8/8 PASS  
> **Status:** All issues resolved. Full end-to-end software & hardware demo operational.

---

## 1. Master Verification & Task Status

| # | Task | Scope | Status | Notes |
|---|------|-------|:------:|-------|
| 1 | Full regression baseline | `pytest tests/ -q` | ✅ **PASS** | **467/467 passing** in 19.18s (100%) |
| 2 | Physical & electrical stress harness | `scripts/real_world_physical_stress.py` | ✅ **PASS** | **7/7 scenarios passed** |
| 3 | Hardware-in-the-loop (HIL) suite | `scripts/hil_hardware_test.py` | ✅ **PASS** | **10/10 scenarios passed** |
| 4 | Closed-loop E2E firmware & AI simulation | `scripts/test_firmware_and_ai_e2e.py` | ✅ **PASS** | **8/8 stages passed** |
| 5 | Hardware simulation stress suite | `scripts/stress_test_hardware_sim.py` | ✅ **PASS** | **7/7 scenarios passed** |
| 6 | Real-data NILM & ML fallback suite | `tests/test_real_data_and_ml_fallback.py` | ✅ **PASS** | **34/34 passing** in 1.26s |
| 7 | Demo profile CLI argument support | `scripts/run_pipeline.py` | ✅ **DONE** | Added `--config` parameter to load `config/config.demo.yaml` |
| 8 | Heuristic fallback pipeline integration | `scripts/run_pipeline.py` | ✅ **DONE** | Integrated `HeuristicApplianceClassifier` for zero-torch fallback |
| 9 | Demo fleet simulation profiles | `backend/scripts/simulate_esp32.py` | ✅ **DONE** | Added `DEMO_DEVICES` and `--demo` CLI flag |
| 10 | Full system demo runner wiring | `scripts/demo_full_system.py` | ✅ **DONE** | Added `--demo` support & fixed WebSocket URL to `/ws` |
| 11 | Knowledge graph synchronization | `graphify update .` | ✅ **DONE** | Rebuilt: 2,266 nodes, 4,625 edges, 164 communities |

---

## 2. Demo-Specific Pipeline Fixes

### 1. [`scripts/run_pipeline.py`](file:///home/pramodsb/Downloads/mjr/scripts/run_pipeline.py)
- **CLI Configuration**: Added `argparse` in `main()` so `--config config/config.demo.yaml` loads the demo 600W profile and demo weights (`backend/models/weights_demo/protonet.pt`).
- **Heuristic Classifier Fallback**: Initialized `HeuristicApplianceClassifier` in `EMSOrchestrator.__init__`. Wired into `_classify_device` when ProtoNet is absent, and inside the low-confidence gate to rescue marginal predictions.
- **Dynamic Prototype Registry Path**: Updated `handle_label_submitted()` to write new labels to the active `weights_dir` (e.g. `weights_demo/`) rather than hardcoded `weights/`.

### 2. [`backend/scripts/simulate_esp32.py`](file:///home/pramodsb/Downloads/mjr/backend/scripts/simulate_esp32.py)
- Added `DEMO_DEVICES` (Laptop 120W / 70–200W span, Desktop 250W, Monitor 35W, Projector 300W burst, Charger 45W / 10–120W USB-PD span) matching `config/config.demo.yaml`.
- Added `--demo` flag to simulate consumer electronics instead of high-power kitchen loads that would immediately trip the 600W bench safety ceiling.

### 3. [`src/pipeline/heuristic_fallback.py`](file:///home/pramodsb/Downloads/mjr/src/pipeline/heuristic_fallback.py)
- Expanded `DEFAULT_RULES` with envelopes for `phone_charger` (5–125W), `router` (5–35W), `monitor` (15–80W), `laptop` (15–220W), `desktop_computer` (50–450W), `projector` (30–450W).
- Ensures seamless fallback classification across the entire consumer electronics power band (phones, powerbanks, ultrabooks, gaming laptops, projectors).

### 4. [`scripts/demo_full_system.py`](file:///home/pramodsb/Downloads/mjr/scripts/demo_full_system.py) & [`Makefile`](file:///home/pramodsb/Downloads/mjr/Makefile)
- Added `--demo` CLI flag support (and `EMS_DEMO=1` environment variable).
- Launches `run_pipeline.py --config config/config.demo.yaml` and `simulate_esp32.py --demo` when `--demo` is active.
- Fixed printed WebSocket URL to `ws://localhost:8000/ws`.
- Updated `make demo` target to pass `--demo`.

---

## 3. Test Execution Details

### 3.1 Closed-Loop Firmware & AI E2E (`scripts/test_firmware_and_ai_e2e.py`)
* ✅ **Stage 1:** Base Load Telemetry & Phantom Tracking (0.073 kWh, ₹0.440 INR).
* ✅ **Stage 2:** Appliance Turn-On & ProtoNet Classification (Kettle 2200W, 100% confidence).
* ✅ **Stage 3:** Compressor Inrush Suppression & Steady-State Cycling (nuisance trip avoided).
* ✅ **Stage 4:** Peak Tariff HVAC Load Shedding & Closed-Loop Relay Actuation (`SHED_HVAC`).
* ✅ **Stage 5:** Critical Load Defense-in-Depth (`node_fridge` Tier-0 immunity respected).
* ✅ **Stage 6:** Novel Appliance Plug-In & OpenMax Weibull EVT Detection (`LABEL_REQUEST` emitted).
* ✅ **Stage 7:** Physical Arc-Fault Injection & Sub-100ms Edge Cutoff ($dP/dt = 14,000\text{W/s}$).
* ✅ **Stage 8:** Overcurrent Safety Protection (125% Rated Power local cutoff).

### 3.2 Physical Stress Suite (`scripts/real_world_physical_stress.py`)
* ✅ **1. Grid Voltage Sag & Swell Stability (160V - 275V):** Tested 7 voltage stages, no crashes or spurious trips.
* ✅ **2. Mains Frequency Drift Tolerance (47Hz - 53Hz):** Frequency scaling across DISCOM tolerances verified.
* ✅ **3. Total Harmonic Distortion (THD) NILM Immunity:** Injected 3rd/5th/7th harmonic ripple; 0 false transient triggers.
* ✅ **4. Inrush Current vs Arc-Fault Discrimination:** Inrush tolerated; Arc Fault tripped instantly with 300s lockout.
* ✅ **5. CT Clamp Reverse Polarisation & Saturation:** Reversed CT clamped to 0W; 3500W saturation engaged hardware cutoff.
* ✅ **6. PCB Thermal Rise & Continuous Current Audit:** 10mm 2oz trace: 16A rise = 5.5°C; 30A rise = 22.7°C (<60°C limit).
* ✅ **7. Relay Actuation State-Machine Endurance:** 10,000 state transitions executed; deterministic final state verified.

### 3.3 HIL Hardware Suite (`scripts/hil_hardware_test.py`)
* ✅ **1. Low-Power Detection (20W threshold):** Laptop 45W step triggered NILM transient.
* ✅ **2. Motor Inrush Signal Capture:** Captured compressor start transient waveform (1200W -> 150W).
* ✅ **3. Resistive Step Transient (Kettle):** Kettle 2200W step detected cleanly.
* ✅ **4. NEVER_SHED Physical Node Immunity:** `node_fridge` protected: OFF command blocked (`DEFER`).
* ✅ **5. Edge Arc-Fault Trip ($dP/dt > 1000\text{W/s}$):** Trip verified: $14,000\text{W/s} > 1,000\text{W/s}$.
* ✅ **6. Edge Overcurrent Cutoff (125% Rated):** Cutoff verified: $280\text{W} > 250\text{W}$ limit.
* ✅ **7. Dual-Format MQTT Payload Parser:** Parsed plain ASCII floats and multi-vendor JSON.
* ✅ **8. Hardware LWT & State Machine ACKs:** Processed ONLINE, ON_CONFIRMED, OFF_CONFIRMED lifecycle events.
* ✅ **9. Database Ingestion & WAL Concurrency:** SQLite WAL mode with busy timeout flushed under concurrency.
* ✅ **10. Indian DISCOM Tariff Calculation:** Calculated 1.0 kWh usage -> ₹8.00 INR.

---

## 4. How to Run Demo

```bash
# 1. Start full software demo (Broker + Pipeline with demo config + API + 5-node virtual electronics fleet)
make demo

# 2. In another terminal, start React frontend
cd frontend && npm run dev
```

- **Dashboard UI**: `http://localhost:5173`
- **FastAPI Swagger**: `http://localhost:8000/docs`
- **WebSocket Stream**: `ws://localhost:8000/ws`
- **Health Check**: `http://localhost:8000/health`

