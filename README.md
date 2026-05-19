# ⚡ Confidence-Aware Digital Twin Energy Management System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-79%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Node](https://img.shields.io/badge/node-18%2B-green)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)
![License](https://img.shields.io/badge/license-Academic-orange)

A production-grade Smart Home Energy Management System integrating **CNN/ProtoNet open-set device classification**, **Reinforcement Learning (Q-Learning)**, **real-time safety monitoring**, and a **premium React dashboard** — all orchestrated over MQTT with support for both **physical ESP32 hardware** and **simulated devices** in hybrid mode.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              MQTT Broker (Eclipse Mosquitto :1883)               │
│              Docker container · QoS 1 · Persistence             │
└──────┬───────────┬──────────────┬──────────────┬────────────────┘
       │           │              │              │
  ┌────▼────┐ ┌────▼─────┐  ┌────▼────┐   ┌────▼──────────┐
  │  ESP32  │ │ Pipeline │  │ FastAPI │   │   React       │
  │  Nodes  │ │Orchestr. │  │  API    │   │  Dashboard    │
  │ (4 HW)  │ │          │  │ :8000   │   │  :5173        │
  └─────────┘ └────┬─────┘  └────┬────┘   └───────────────┘
  ┌─────────┐      │              │
  │Simulator│──────┘              │
  │ (6 SW)  │                     │
  └─────────┘      ┌──────────────▼──────────────┐
                   │    Processing Pipeline       │
                   │                              │
                   │  1. Safety Layer (‖ cutoff)  │
                   │  2. RoC Arc-Fault Proxy      │
                   │  3. Soft Anomaly Watchdog    │
                   │  4. CNN → ProtoNet (open-set)│
                   │  5. Phantom Load Tracker     │
                   │  6. Database Persistence     │
                   │  7. Analytics Engine (kWh)   │
                   │  8. Digital Twin PMV (ISO)   │
                   │  9. RL Agent (SHED/empathy)  │
                   │ 10. Latency Monitor          │
                   │ 11. Dashboard Broadcast      │
                   └──────────────────────────────┘
```

### Pipeline Flow

| Step | Module | Description |
|------|--------|-------------|
| 0 | **Safety Monitor** | **Parallel asyncio.Task** — independent MQTT subscription, 110% warning / 125% cutoff |
| 0b | **RoC Arc-Fault Proxy** | Rate-of-change detection (> 1000 W/s → immediate relay OFF) |
| 1 | **Watchdog** | Rolling z-score anomaly detection for sensor drift |
| 2 | **ProtoNet CNN** | 5-layer 1D CNN + Temporal Attention → 128D embedding → prototypical distance |
| 2b | **OpenMax** | Weibull EVT on tail distances → open-set unknown rejection |
| 2c | **Temp Scaling** | Learned temperature T for calibrated softmax confidence |
| 3 | **Confidence Gate** | If confidence < 0.90 → skip RL, emit `LOW_CONFIDENCE` event |
| 4 | **Delta Stability** | Buffer unknown embeddings; stable → `LABEL_REQUEST`, transient → discard |
| 5 | **Phantom Tracker** | Exponential moving average of vampire loads when devices are "OFF" |
| 6 | **Database** | aiosqlite with WAL mode, batched writes every 10s, CSV fallback on lock-up |
| 7 | **Analytics** | Per-device kWh accumulation and ToU cost estimation (peak/mid/off-peak) |
| 8 | **Digital Twin** | Full ISO 7730 Fanger PMV (6-input) thermal comfort model |
| 9 | **RL Agent** | Tabular Q-Learning with confidence gate + PMV empathy gate + ToU reward shaping |
| 10 | **Latency Monitor** | `time.perf_counter()` per message; broadcasts avg/max/p95 every 30s |
| 11 | **Broadcast** | Structured JSON events to dashboard via MQTT → WebSocket bridge |

### Failure Handling

| Failure | Mitigation |
|---------|------------|
| Server crash | ESP32 relay still operates locally (heartbeat watchdog, 30s timeout) |
| MQTT disconnect | Local edge execution mode, automatic reconnection with backoff |
| Database lock | CSV fallback captures writes; replayed on restart |
| CT clamp drift | Calibration script (`calibrate_ct.py`) corrects CT_RATIO in-situ |

---

## 📊 Implementation Status

### Phase 1 — Simulation ✅
| Feature | Status |
|---------|--------|
| 5-layer CNN ProtoNet (128D embeddings) | ✅ |
| Temporal Attention layer | ✅ |
| Episodic N-way K-shot meta-learning | ✅ |
| OpenMax + Weibull EVT unknown rejection | ✅ |
| Temperature Scaling calibration | ✅ |
| Full ISO 7730 Fanger PMV (6-input) | ✅ |
| Q-Learning with ToU pricing & confidence gate | ✅ |
| Policy Promotion Gate (RL twin validation) | ✅ |
| Safety monitor (parallel asyncio task) | ✅ |
| Delta stability for unknown device routing | ✅ |
| 10-device ESP32 simulator (1Hz) | ✅ |
| React dashboard with Digital Twin label prompts | ✅ |
| API handling for LABEL_REQUEST / LOW_CONFIDENCE | ✅ |
| Real UK-DALE dataset training (via Colab HDF5 loader) | ✅ |
| Real REDD dataset training (via Colab) | ✅ |

### Phase 2 — Production Hardware ✅
| Feature | Status |
|---------|--------|
| Firmware rewrite (true RMS, non-blocking cutoff, heartbeat watchdog) | ✅ |
| Infrastructure hardening (Docker Mosquitto, docker-compose) | ✅ |
| RL agent fix (state-space 3.9M → 576, epsilon decay, NEVER_SHED) | ✅ |
| Real data integration (NILMTK replay from HDF5 traces) | ✅ |
| Hardware ACK protocol (OFF_CONFIRMED relay feedback) | ✅ |
| Rate-of-Change arc-fault safety proxy (> 1000 W/s) | ✅ |
| Hybrid mode (4 physical + 6 simulated devices) | ✅ |
| Pipeline latency instrumentation (avg/max/p95 < 200ms target) | ✅ |
| CT-clamp calibration script | ✅ |
| Firmware flash script (esptool.py) | ✅ |
| 79-test integration suite (100% pass) | ✅ |
| Dashboard latency panel | ✅ |

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend pipeline, ML, API |
| Node.js | 18+ | React frontend |
| Docker | 20+ | Mosquitto MQTT broker |
| npm | 9+ | Frontend dependency management |
| Git | any | Version control |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Pramod8997/nullp.git
cd mjr

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies (Python + Node.js)
make install
```

---

## 🏃 Running the System

### Start Everything (Recommended)

```bash
make run
```

This single command starts **5 services** with proper process management:

| # | Service | Port | Description |
|---|---------|------|-------------|
| 1 | Mosquitto MQTT Broker | `:1883` | Docker container (eclipse-mosquitto:2) |
| 2 | Pipeline Orchestrator | — | `scripts/run_pipeline.py` |
| 3 | FastAPI Backend | `:8000` | `uvicorn src.api.main:app` |
| 4 | React Dashboard | `:5173` | `frontend/` via Vite |
| 5 | ESP32 Simulator | — | `backend/scripts/simulate_esp32.py` (6 virtual devices) |

**Dashboard URL**: [http://localhost:5173/](http://localhost:5173/)

Press `Ctrl+C` to cleanly shut down all services (trap-based process cleanup).

### What You'll See

When the pipeline starts, you'll see:
```
═══════════════════════════════════════════
  🏠 EMS Pipeline Orchestrator ONLINE
  Safety Layer: ✅  |  ProtoNet: ✅
  Watchdog: ✅  |  Phantom Tracker: ✅
  RL Agent: ✅  |  Empathy Gate: ✅
═══════════════════════════════════════════
```

The dashboard displays 6 panels:
1. **Device Fleet** — Live power, ON/OFF state, ProtoNet classification per device
2. **Real-Time Chart** — Per-device power traces with safety threshold line
3. **Safety Alerts** — Critical cutoff and warning events with severity levels
4. **Digital Twin** — PMV comfort gauge, RL action log, unknown device prompts
5. **Phantom Tracker** — Vampire load detection with per-device breakdown
6. **System Status** — WebSocket/pipeline health, daily kWh/cost, **pipeline latency (avg/p95/max)**

### Hybrid Mode (Physical + Simulated)

For mixed deployment with physical ESP32 nodes alongside simulated devices:

```bash
# 1. Start infrastructure
make run

# 2. Flash physical nodes (adjust /dev/ttyUSB* ports)
./scripts/flash_firmware.sh

# 3. Calibrate CT clamps on each node
python scripts/calibrate_ct.py node_fridge
python scripts/calibrate_ct.py node_microwave
python scripts/calibrate_ct.py node_kettle
python scripts/calibrate_ct.py node_hvac
```

Physical nodes publish to `home/sensor/{DEVICE_ID}/power` and accept commands on `home/plug/{DEVICE_ID}/command`. The pipeline automatically detects physical vs. simulated devices via `config/config.yaml`.

---

## 🧪 Testing

Run the full test suite (79 tests):

```bash
make test
```

### Test Coverage

| Category | Tests | Scope |
|----------|-------|-------|
| API Endpoints | 5 | health, devices, analytics, phantom, status |
| Phase 1 Core | 34 | watchdog, phantom, analytics, failure matrix, mode classifier, thermodynamics, confidence gate, openmax, temperature scaling, delta stability, ToU reward, safety parallel, episodic training, full pipeline |
| Phase 1 Bug Fixes | 20 | NILM transient, OpenMax predict, calibration, delta stability, PMV thermodynamics, policy promotion, confidence gate, temporal validator, CSV fallback, NILM preprocessing, unknown device RL routing |
| Phase 2 Integration | 20 | relay ACK, hybrid mode, RL action chain, RoC safety, data retention, CSV replay, epsilon decay, state space, latency, NEVER_SHED, MQTT topics, docker compose |

---

## 🛠 Advanced Usage

### Model Training (Local Mock)

Train the CNN/ProtoNet and RL agent from scratch locally using synthetic data:

```bash
make train_all
```

This runs:
1. **CNN Feature Extractor** — trains on UK-DALE 1Hz data (mock)
2. **ProtoNet Support Sets** — generates prototype anchors per device class
3. **RL Q-Table** — 2000 episodes of tabular Q-learning

Weights are saved to `backend/models/weights/`.

### Model Training (Google Colab with Real Datasets)

We provide a Colab notebook to train the ProtoNet on real UK-DALE and REDD datasets using an NVIDIA T4 GPU:

1. Open `notebooks/colab_train.py` in Google Colab.
2. Select **T4 GPU** runtime and run the script.
3. The script will download datasets, train the model, and package the weights into `ems_weights.zip`.
4. Run the local import script to load the weights into your project:

```bash
python3 scripts/import_colab_weights.py path/to/ems_weights.zip
```

### NILMTK Real-Data Replay

Replay real household power traces from UK-DALE / REDD HDF5 files:

```bash
python3 scripts/nilmtk_replay.py --hdf5 /path/to/ukdale.h5 --building 1 --speed 10
```

The script reads appliance-level traces from the HDF5 file, interleaves them by timestamp, and publishes each reading to MQTT at the specified playback speed multiplier.

### CT-Clamp Calibration

Calibrate a physical ESP32 node's CT-clamp sensor against a known load:

```bash
python3 scripts/calibrate_ct.py node_fridge
```

The script:
1. Publishes a `CALIBRATE_REQUEST` to the device.
2. Waits for a power reading while a 100 W reference load is attached.
3. Computes a correction factor and updates `CT_RATIO` in `firmware/esp32_node/src/main.cpp`.

### Safety Stress Test

Verify cutoff behavior under dangerous power spikes:

```bash
make test_safety
```

### Clean Database & Caches

```bash
make clean
```

### Manual Component Execution

Run each service individually (useful for debugging):

```bash
# Terminal 1: MQTT Broker (Docker)
docker-compose up -d mosquitto

# Terminal 2: Pipeline
export PYTHONPATH=$(pwd)
source venv/bin/activate
python3 scripts/run_pipeline.py

# Terminal 3: FastAPI
export PYTHONPATH=$(pwd)
source venv/bin/activate
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 4: Simulator
export PYTHONPATH=$(pwd)
source venv/bin/activate
python3 backend/scripts/simulate_esp32.py

# Terminal 5: Frontend
cd frontend && npm run dev
```

---

## 📁 Project Structure

```
mjr/
├── config/
│   └── config.yaml                 # Centralized config (devices, safety, ProtoNet, RL, analytics)
├── scripts/
│   ├── run_pipeline.py             # Main pipeline orchestrator (11-step architecture)
│   ├── start_broker.py             # Legacy local amqtt MQTT broker (replaced by Docker)
│   ├── train_models.py             # Offline CNN/ProtoNet/RL training
│   ├── generate_mock_ukdale.py     # Synthetic UK-DALE HDF5 generator
│   ├── import_colab_weights.py     # Import trained weights from Colab
│   ├── nilmtk_replay.py           # Real-data HDF5 trace replayer
│   └── calibrate_ct.py            # CT-clamp calibration utility
├── firmware/
│   └── esp32_node/
│       └── src/
│           └── main.cpp            # ESP32 firmware (true RMS, relay ACK, heartbeat watchdog)
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI + WebSocket bridge (CORS, REST, heartbeat)
│   ├── database/
│   │   └── session.py              # aiosqlite WAL-mode persistence with CSV fallback
│   ├── hardware/
│   │   └── mqtt.py                 # Async MQTT client manager (aiomqtt)
│   ├── models/
│   │   ├── protonet.py             # CNN encoder + SupportSetManager (open-set)
│   │   └── thermodynamics.py       # PMV thermal comfort model (ISO 7730)
│   ├── pipeline/
│   │   ├── safety.py               # Real-time safety threshold monitor
│   │   ├── watchdog.py             # Soft anomaly z-score watchdog
│   │   ├── phantom_tracker.py      # Vampire load EMA tracker
│   │   ├── analytics.py            # kWh usage + ToU cost estimation engine
│   │   ├── classifier.py           # Single vs. multi-device mode classifier
│   │   ├── aggregate_nilm.py       # NILM step-change event detector
│   │   └── failure_matrix.py       # Failure → mitigation mapping
│   └── rl/
│       └── agent.py                # Tabular Q-Learning agent (ε-greedy, NEVER_SHED)
├── backend/
│   ├── scripts/
│   │   └── simulate_esp32.py       # 10-device simulator (4 physical profiles, 6 virtual)
│   ├── data/
│   │   └── mock_ukdale.h5          # Generated synthetic training data
│   └── models/
│       └── weights/                # Trained model weights (CNN, ProtoNet anchors, Q-table)
├── frontend/
│   └── src/
│       ├── App.jsx                 # Main dashboard (WebSocket auto-reconnect, latency state)
│       ├── App.css                 # 6-panel responsive grid layout
│       ├── index.css               # Dark-mode design system (Inter, animations)
│       └── components/
│           ├── DeviceCards.jsx      # Per-device status cards with glow effects
│           ├── RealTimeChart.jsx    # Multi-line Recharts with safety threshold
│           ├── SafetyAlerts.jsx     # Severity-based alert feed
│           ├── DigitalTwin.jsx      # PMV gauge + RL log + unknown device prompts
│           ├── PhantomTracker.jsx   # Vampire load visualization
│           └── SystemStatus.jsx     # Pipeline health + analytics + latency display
├── mosquitto/
│   └── config/
│       └── mosquitto.conf          # Production Mosquitto config (persistence, logging)
├── tests/
│   ├── test_pipeline.py            # 74 pipeline + integration tests
│   └── test_api.py                 # 5 API endpoint tests
├── data/
│   ├── uk_dale.py                  # UK-DALE data loader
│   ├── redd.py                     # REDD data loader
│   └── synd.py                     # Synthetic data loader with realistic transients
├── notebooks/
│   └── colab_train.py              # Google Colab training notebook (T4 GPU)
├── docker-compose.yml              # Mosquitto + Pipeline + API orchestration
├── Dockerfile                      # Python container image
├── Makefile                        # install, train_all, test, run, clean
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

All system parameters are defined in [`config/config.yaml`](config/config.yaml):

| Section | Key Parameters |
|---------|---------------|
| `mqtt` | broker host, port, topic patterns (`home/sensor/+/power`, `home/plug/+/command`) |
| `devices` | per-device config: `simulated` flag, `rated` wattage, `tier0` (NEVER_SHED) |
| `system_safety` | per-device wattage limits, max aggregate (3500 W), warning/critical percentages |
| `protonet` | CNN weights path, distance threshold (15.0), confidence threshold (0.90) |
| `analytics` | ToU pricing: peak ($0.28), mid ($0.18), off-peak ($0.09) |
| `rl` | cooldown (15s), PMV empathy bounds (±0.5), epsilon decay (0.999), promotion episodes (50) |
| `preprocessing` | Savitzky-Golay filter, transient detection threshold (50 W) |
| `delta_stability` | buffer size (10), stability threshold (3.0), min occurrences (3) |
| `database` | SQLite path, CSV fallback path, retention days (30) |

### Device Configuration (Hybrid Mode)

```yaml
devices:
  node_fridge:      { simulated: false, rated: 200,  tier0: true  }   # Physical — never shed
  node_microwave:   { simulated: false, rated: 1200, tier0: false }   # Physical
  node_kettle:      { simulated: false, rated: 2500, tier0: false }   # Physical
  node_hvac:        { simulated: false, rated: 2000, tier0: false }   # Physical
  esp32_tv:         { simulated: true,  rated: 150,  tier0: false }   # Simulated
  esp32_washer:     { simulated: true,  rated: 1800, tier0: false }   # Simulated
  esp32_dryer:      { simulated: true,  rated: 2000, tier0: false }   # Simulated
  esp32_dishwasher: { simulated: true,  rated: 1500, tier0: false }   # Simulated
  esp32_oven:       { simulated: true,  rated: 3000, tier0: false }   # Simulated
  esp32_lighting:   { simulated: true,  rated: 100,  tier0: false }   # Simulated
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health check |
| `GET` | `/api/devices` | Current power state of all devices |
| `GET` | `/api/analytics` | Daily usage summary and cost |
| `GET` | `/api/phantom` | Phantom (vampire) load report |
| `GET` | `/api/status` | Full system status snapshot |
| `WS` | `/ws` | Real-time event stream (WebSocket) |

### WebSocket Event Types

| Event | Description |
|-------|-------------|
| `POWER_UPDATE` | Per-device power reading (1Hz) |
| `SAFETY_ALERT` | Over-current warning or critical cutoff |
| `ARC_FAULT` | Rate-of-change spike detection (> 1000 W/s) |
| `RL_ACTION` / `EMPATHY_ACTION` | RL shed/unshed with PMV context |
| `ANALYTICS_UPDATE` | Periodic kWh and cost summary |
| `PMV_UPDATE` | Thermal comfort index update |
| `LATENCY_STATS` | Pipeline latency metrics (avg/max/p95 ms) |
| `LABEL_REQUEST` | Unknown device requiring user classification |
| `LOW_CONFIDENCE` | ProtoNet confidence below threshold |
| `PHANTOM_UPDATE` | Vampire load detection update |

---

## 🔧 ESP32 Firmware

The firmware in `firmware/esp32_node/src/main.cpp` implements:

| Feature | Details |
|---------|---------|
| **True RMS Measurement** | 200 samples over 100ms window using `analogRead()` |
| **Non-blocking Relay Control** | `millis()`-based cooldown instead of `delay()` |
| **MQTT Topics** | Publishes `home/sensor/{ID}/power`, subscribes `home/plug/{ID}/command` |
| **Hardware ACK** | Sends `OFF_CONFIRMED` after relay state change |
| **Heartbeat Watchdog** | If no server heartbeat for 30s → safe mode (relay OFF) |
| **WiFi Auto-reconnect** | Exponential backoff on disconnect |

### Flashing

```bash
# Using esptool.py (requires pip install esptool)
esptool.py --chip esp32 --port /dev/ttyUSB0 --baud 460800 \
  write_flash -z 0x1000 firmware/esp32_node/build/esp32_node.bin

# Or use the convenience script
./scripts/flash_firmware.sh
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 1883 in use | Stop local Mosquitto: `sudo service mosquitto stop` |
| `Connection refused` on simulator | Ensure Mosquitto is running: `docker-compose up -d mosquitto` |
| `ModuleNotFoundError: No module named 'src'` | Run from project root with `PYTHONPATH=$(pwd)` or use `make run` |
| Frontend shows "Disconnected" | Ensure the FastAPI backend is running on port 8000 |
| ProtoNet shows ⚠️ disabled | Run `make train_all` to generate model weights |
| Kettle constantly triggering safety | Normal — kettle draws ~2200W; the safety limit is set to 2500W |
| Stale database | Run `make clean` to reset |
| Latency panel shows > 200ms (red) | Check system load; reduce logging verbosity in `config.yaml` |
| ESP32 enters safe mode | Server heartbeat lost — verify pipeline and Mosquitto connectivity |
| CT clamp readings off by > 5% | Re-run `python scripts/calibrate_ct.py <DEVICE_ID>` with known 100W load |

---

## 📜 License

This project is developed as part of an academic research initiative.
