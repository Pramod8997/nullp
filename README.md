# ⚡ Confidence-Aware Digital Twin Energy Management System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-27%2F27-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Node](https://img.shields.io/badge/node-18%2B-green)
![License](https://img.shields.io/badge/license-Academic-orange)

A production-grade Smart Home Energy Management System integrating **CNN/ProtoNet open-set device classification**, **Reinforcement Learning (Q-Learning)**, **real-time safety monitoring**, and a **premium React dashboard** — all orchestrated over MQTT.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MQTT Broker (amqtt :1883)                    │
└──────┬───────────┬──────────────┬──────────────┬────────────────┘
       │           │              │              │
  ┌────▼────┐ ┌────▼─────┐  ┌────▼────┐   ┌────▼──────────┐
  │  ESP32  │ │ Pipeline │  │ FastAPI │   │   React       │
  │Simulator│ │Orchestr. │  │  API    │   │  Dashboard    │
  │ (1Hz)   │ │          │  │ :8000   │   │  :5173        │
  └─────────┘ └────┬─────┘  └────┬────┘   └───────────────┘
                   │              │
        ┌──────────▼──────────────▼──────────┐
        │         Processing Pipeline         │
        │                                     │
        │  1. Safety Layer (parallel cutoff)  │
        │  2. Soft Anomaly Watchdog (z-score) │
        │  3. CNN → ProtoNet (open-set)       │
        │  4. Phantom Load Tracker            │
        │  5. Database Persistence (WAL)      │
        │  6. Analytics Engine (kWh/cost)     │
        │  7. Digital Twin PMV (comfort)      │
        │  8. RL Agent (cooldown + empathy)   │
        │  9. Dashboard Broadcast             │
        └─────────────────────────────────────┘
```

### Pipeline Flow

| Step | Module | Description |
|------|--------|-------------|
| 1 | **Safety Monitor** | Parallel threshold check → hardware relay cutoff (highest priority) |
| 2 | **Watchdog** | Rolling z-score anomaly detection for sensor drift |
| 3 | **ProtoNet CNN** | 1D CNN feature extraction → prototypical distance → open-set known/unknown classification |
| 4 | **Phantom Tracker** | Exponential moving average of vampire loads when devices are "OFF" |
| 5 | **Database** | aiosqlite with WAL mode, batched writes every 10s |
| 6 | **Analytics** | Per-device kWh accumulation and cost estimation |
| 7 | **Digital Twin** | Simplified ISO 7730 PMV thermal comfort index |
| 8 | **RL Agent** | Tabular Q-Learning with 15s cooldown + PMV empathy gate |
| 9 | **Broadcast** | Structured JSON events to dashboard via MQTT → WebSocket bridge |

### Failure Handling

| Failure | Mitigation |
|---------|------------|
| Server crash | ESP32 relay still operates locally |
| MQTT disconnect | Local edge execution mode, queue for sync |
| DB failure | Log to fallback, continue safety monitoring |
| Model drift | Fall back to rule-based thresholds |

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend pipeline, ML, API |
| Node.js | 18+ | React frontend |
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

| # | Service | Port | Script |
|---|---------|------|--------|
| 1 | MQTT Broker | `:1883` | `scripts/start_broker.py` |
| 2 | Pipeline Orchestrator | — | `scripts/run_pipeline.py` |
| 3 | FastAPI Backend | `:8000` | `uvicorn src.api.main:app` |
| 4 | React Dashboard | `:5173` | `frontend/` via Vite |
| 5 | ESP32 Simulator | — | `backend/scripts/simulate_esp32.py` |

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
6. **System Status** — WebSocket/pipeline health, daily kWh usage, estimated cost

---

## 🧪 Testing

Run the full test suite (27 tests):

```bash
make test
```

Tests cover:
- **SafetyMonitor** — threshold breach → cutoff trigger
- **SoftAnomalyWatchdog** — z-score spike detection
- **PhantomTracker** — vampire load tracking and ranking
- **AnalyticsEngine** — kWh recording and cost calculation
- **FailureMatrix** — failure type → mitigation mapping
- **ModeClassifier** — single vs. multi-device aggregate detection
- **ThermodynamicsModel** — PMV comfort zone, hot zone, clamped bounds
- **RL Agent** — state padding, action range, Q-table updates, cooldown logic
- **FastAPI endpoints** — health, devices, analytics, phantom, status

---

## 🛠 Advanced Usage

### Model Training

Train the CNN/ProtoNet and RL agent from scratch:

```bash
make train_all
```

This runs:
1. **CNN Feature Extractor** — trains on UK-DALE 1Hz data (mock)
2. **ProtoNet Support Sets** — generates prototype anchors per device class
3. **RL Q-Table** — 1000 episodes of tabular Q-learning

Weights are saved to `backend/models/weights/`.

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
# Terminal 1: MQTT Broker
source venv/bin/activate
python3 scripts/start_broker.py

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
│   └── config.yaml              # Centralized configuration (safety, ProtoNet, RL, analytics)
├── scripts/
│   ├── run_pipeline.py          # Main pipeline orchestrator (9-step architecture)
│   ├── start_broker.py          # Local amqtt MQTT broker
│   ├── train_models.py          # Offline CNN/ProtoNet/RL training
│   └── generate_mock_ukdale.py  # Synthetic UK-DALE HDF5 generator
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI + WebSocket bridge (CORS, REST, heartbeat)
│   ├── database/
│   │   └── session.py           # aiosqlite WAL-mode persistence with batching
│   ├── hardware/
│   │   └── mqtt.py              # Async MQTT client manager (aiomqtt)
│   ├── models/
│   │   ├── protonet.py          # CNN encoder + SupportSetManager (open-set)
│   │   └── thermodynamics.py    # PMV thermal comfort model (ISO 7730)
│   ├── pipeline/
│   │   ├── safety.py            # Real-time safety threshold monitor
│   │   ├── watchdog.py          # Soft anomaly z-score watchdog
│   │   ├── phantom_tracker.py   # Vampire load EMA tracker
│   │   ├── analytics.py         # kWh usage + cost estimation engine
│   │   ├── classifier.py        # Single vs. multi-device mode classifier
│   │   ├── aggregate_nilm.py    # NILM step-change event detector
│   │   └── failure_matrix.py    # Failure → mitigation mapping (graceful degradation)
│   └── rl/
│       └── agent.py             # Tabular Q-Learning agent (ε-greedy, cooldown)
├── backend/
│   ├── scripts/
│   │   └── simulate_esp32.py    # 4-device simulator (fridge, hvac, kettle, tv)
│   ├── data/
│   │   └── mock_ukdale.h5       # Generated synthetic training data
│   └── models/
│       └── weights/             # Trained model weights (CNN, ProtoNet anchors, Q-table)
├── frontend/
│   └── src/
│       ├── App.jsx              # Main dashboard (WebSocket auto-reconnect)
│       ├── App.css              # 6-panel responsive grid layout
│       ├── index.css            # Dark-mode design system (Inter, animations)
│       └── components/
│           ├── DeviceCards.jsx   # Per-device status cards with glow effects
│           ├── RealTimeChart.jsx # Multi-line Recharts with safety threshold
│           ├── SafetyAlerts.jsx  # Severity-based alert feed
│           ├── DigitalTwin.jsx   # PMV gauge + RL log + unknown device prompts
│           ├── PhantomTracker.jsx# Vampire load visualization
│           └── SystemStatus.jsx  # Pipeline health + analytics display
├── tests/
│   ├── test_pipeline.py         # 22 integration tests
│   └── test_api.py              # 5 API endpoint tests
├── data/
│   ├── uk_dale.py               # UK-DALE data loader (stub)
│   ├── redd.py                  # REDD data loader (stub)
│   └── synd.py                  # Synthetic data loader (stub)
├── Makefile                     # install, train_all, test, run, clean
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

All system parameters are defined in [`config/config.yaml`](config/config.yaml):

| Section | Key Parameters |
|---------|---------------|
| `mqtt` | broker host, port, topic patterns |
| `system_safety` | per-device wattage limits, max aggregate |
| `protonet` | CNN weights path, distance threshold, window size |
| `analytics` | cost per kWh |
| `rl` | cooldown seconds, PMV empathy bounds, Q-table path |

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

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 1883 in use | Stop local Mosquitto: `sudo service mosquitto stop` |
| `ModuleNotFoundError: No module named 'src'` | Run from project root with `PYTHONPATH=$(pwd)` or use `make run` |
| Frontend shows "Disconnected" | Ensure the FastAPI backend is running on port 8000 |
| ProtoNet shows ⚠️ disabled | Run `make train_all` to generate model weights |
| Kettle constantly triggering safety | Normal — kettle draws ~2200W; the safety limit is set to 2500W |
| Stale database | Run `make clean` to reset |

---

## 📜 License

This project is developed as part of an academic research initiative.
