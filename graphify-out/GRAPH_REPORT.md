# Graph Report - .  (2026-06-09)

## Corpus Check
- 66 files · ~50,848 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 739 nodes · 1667 edges · 80 communities (59 shown, 21 thin omitted)
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 713 edges (avg confidence: 1.0)
- Token cost: 110,281 input · 3,831 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Synthetic Data & MQTT|Synthetic Data & MQTT]]
- [[_COMMUNITY_Open-Set Model Inference|Open-Set Model Inference]]
- [[_COMMUNITY_Thermal Comfort Thermodynamics|Thermal Comfort Thermodynamics]]
- [[_COMMUNITY_Encoder Architecture & Calibration|Encoder Architecture & Calibration]]
- [[_COMMUNITY_Signal Stability Analysis|Signal Stability Analysis]]
- [[_COMMUNITY_API Integration Testing|API Integration Testing]]
- [[_COMMUNITY_Frontend Dependency Management|Frontend Dependency Management]]
- [[_COMMUNITY_Failure Mitigation Protocols|Failure Mitigation Protocols]]
- [[_COMMUNITY_Database Session Management|Database Session Management]]
- [[_COMMUNITY_Model Training Utilities|Model Training Utilities]]
- [[_COMMUNITY_RL Policy Validation|RL Policy Validation]]
- [[_COMMUNITY_NILM Transient Detection|NILM Transient Detection]]
- [[_COMMUNITY_Frontend UI Components|Frontend UI Components]]
- [[_COMMUNITY_Temporal Attention Mechanisms|Temporal Attention Mechanisms]]
- [[_COMMUNITY_Temporal Anomaly Validation|Temporal Anomaly Validation]]
- [[_COMMUNITY_REST API Endpoints|REST API Endpoints]]
- [[_COMMUNITY_Soft Anomaly Monitoring|Soft Anomaly Monitoring]]
- [[_COMMUNITY_Model Confidence Calibration|Model Confidence Calibration]]
- [[_COMMUNITY_Phantom Load Tracking|Phantom Load Tracking]]
- [[_COMMUNITY_WebSocket Connection Management|WebSocket Connection Management]]
- [[_COMMUNITY_Usage Analytics Engine|Usage Analytics Engine]]
- [[_COMMUNITY_Tabular Q-Learning Agent|Tabular Q-Learning Agent]]
- [[_COMMUNITY_API Event Models|API Event Models]]
- [[_COMMUNITY_Fleet Safety Diagnostics|Fleet Safety Diagnostics]]
- [[_COMMUNITY_REDD Dataset Loader|REDD Dataset Loader]]
- [[_COMMUNITY_UK-DALE Dataset Loader|UK-DALE Dataset Loader]]
- [[_COMMUNITY_Power Mode Classification|Power Mode Classification]]
- [[_COMMUNITY_RL Safety Constraints|RL Safety Constraints]]
- [[_COMMUNITY_CT Sensor Calibration|CT Sensor Calibration]]
- [[_COMMUNITY_ESP32 Firmware Logic|ESP32 Firmware Logic]]
- [[_COMMUNITY_RL State & Reward|RL State & Reward]]
- [[_COMMUNITY_ESP32 Device Simulator|ESP32 Device Simulator]]
- [[_COMMUNITY_Safety Rate-of-Change Testing|Safety Rate-of-Change Testing]]
- [[_COMMUNITY_Load Shedding Configuration|Load Shedding Configuration]]
- [[_COMMUNITY_System Integration Testing|System Integration Testing]]
- [[_COMMUNITY_Calibration Validation Testing|Calibration Validation Testing]]
- [[_COMMUNITY_Relay Protocol Testing|Relay Protocol Testing]]
- [[_COMMUNITY_System Orchestration & Config|System Orchestration & Config]]
- [[_COMMUNITY_Confidence Calibration Logic|Confidence Calibration Logic]]
- [[_COMMUNITY_HDF5 Data Replay|HDF5 Data Replay]]
- [[_COMMUNITY_Confidence Gate Testing|Confidence Gate Testing]]
- [[_COMMUNITY_Label Submission API|Label Submission API]]
- [[_COMMUNITY_Dashboard UI & Documentation|Dashboard UI & Documentation]]
- [[_COMMUNITY_Database Retention Testing|Database Retention Testing]]
- [[_COMMUNITY_CSV Fallback Testing|CSV Fallback Testing]]
- [[_COMMUNITY_Pipeline Latency Testing|Pipeline Latency Testing]]
- [[_COMMUNITY_CSV Persistence Testing|CSV Persistence Testing]]
- [[_COMMUNITY_Phantom Tracker Testing|Phantom Tracker Testing]]
- [[_COMMUNITY_Structured Logging Formatter|Structured Logging Formatter]]
- [[_COMMUNITY_Firmware Deployment Scripts|Firmware Deployment Scripts]]
- [[_COMMUNITY_Mock Data Generation|Mock Data Generation]]
- [[_COMMUNITY_Analytics Data API|Analytics Data API]]
- [[_COMMUNITY_Confidence Monitoring API|Confidence Monitoring API]]
- [[_COMMUNITY_System Readiness API|System Readiness API]]
- [[_COMMUNITY_Training Data Pipeline|Training Data Pipeline]]
- [[_COMMUNITY_Thermal Zone Classification|Thermal Zone Classification]]
- [[_COMMUNITY_Device Lockout Management|Device Lockout Management]]
- [[_COMMUNITY_Asynchronous Action Logging|Asynchronous Action Logging]]
- [[_COMMUNITY_Time-of-Use Pricing|Time-of-Use Pricing]]
- [[_COMMUNITY_Synthetic Model Training|Synthetic Model Training]]
- [[_COMMUNITY_RL Epsilon Decay Testing|RL Epsilon Decay Testing]]
- [[_COMMUNITY_RL Convergence Testing|RL Convergence Testing]]
- [[_COMMUNITY_Confidence Gating Logic|Confidence Gating Logic]]
- [[_COMMUNITY_RL Action Validation|RL Action Validation]]
- [[_COMMUNITY_Frontend Static Assets|Frontend Static Assets]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_REDD Data Loading|REDD Data Loading]]
- [[_COMMUNITY_ESP32 Hardware Simulation|ESP32 Hardware Simulation]]

## God Nodes (most connected - your core abstractions)
1. `TabularQLearningAgent` - 74 edges
2. `OpenMaxWeibull` - 59 edges
3. `DeltaStabilityAnalyzer` - 56 edges
4. `ThermodynamicsModel` - 54 edges
5. `SupportSetManager` - 53 edges
6. `NILMTransientDetector` - 53 edges
7. `TemporalValidator` - 53 edges
8. `PMVThermodynamics` - 52 edges
9. `TemperatureScaler` - 51 edges
10. `FailureMatrix` - 51 edges

## Surprising Connections (you probably didn't know these)
- `EMSOrchestrator` --uses--> `DatabaseSession`  [INFERRED]
  scripts/run_pipeline.py → src/database/session.py
- `EMSOrchestrator` --uses--> `TemperatureScaler`  [INFERRED]
  scripts/run_pipeline.py → src/models/calibration.py
- `EMSOrchestrator` --uses--> `CNN1DEncoder`  [INFERRED]
  scripts/run_pipeline.py → src/models/protonet.py
- `EMSOrchestrator` --uses--> `OpenMaxWeibull`  [INFERRED]
  scripts/run_pipeline.py → src/models/protonet.py
- `EMSOrchestrator` --uses--> `SupportSetManager`  [INFERRED]
  scripts/run_pipeline.py → src/models/protonet.py

## Import Cycles
- 1-file cycle: `src/api/main.py -> src/api/main.py`

## Communities (80 total, 21 thin omitted)

### Community 0 - "Synthetic Data & MQTT"
Cohesion: 0.05
Nodes (26): Synthetic UK-DALE Data Generator. Generates realistic 1 Hz transient power signa, Generates realistic 1 Hz transient power signatures for 10 appliance classes., Returns dict {class_name: np.ndarray (n_samples, seq_len)}., SyntheticUKDALE, MQTTClientManager, ProtoNet, PrototypeRegistry, ProtoNet with temporal attention applied before the CNN encoder.      embed(x) a (+18 more)

### Community 1 - "Open-Set Model Inference"
Cohesion: 0.07
Nodes (22): device, OpenMaxWeibull, x: (batch, 128) → (batch, EMBED_DIM)., Post-training calibration: fit a Weibull tail model to each class's     prototyp, Two call signatures:           fit(class_idx: int, distances: np.ndarray)   ← ne, Legacy fit: stores embeddings and fits Weibull from distances to centroid., Args:             distances:     (N,) distance from query to each prototype, Legacy API: returns scalar open-set probability. (+14 more)

### Community 2 - "Thermal Comfort Thermodynamics"
Cohesion: 0.07
Nodes (21): PMVThermodynamics, ISO 7730 Predicted Mean Vote (PMV) Thermodynamics Model. Category A: PMV in [-0., Backward-compatible wrapper: exposes the old compute_pmv() API used by     exist, Legacy arg-order wrapper (note: met and clo swapped vs ISO signature)., Simple thermal decay model for digital twin simulation., ISO 7730 Predicted Mean Vote (PMV) calculation.     Category A: PMV in [-0.5, +0, Compute PMV. Raises ValueError for out-of-plausible inputs.          Returns:, True if PMV is within ISO 7730 Category A bounds. (+13 more)

### Community 3 - "Encoder Architecture & Calibration"
Cohesion: 0.13
Nodes (23): CNN1DEncoder, EpisodicDataset, 5-layer 1D-CNN mapping (batch, 1, seq_len) → (batch, EMBED_DIM).     Each block:, Legacy episodic sampler kept for backward compat with existing tests., Legacy TemperatureScaler kept for backward compat.     Prefer importing from src, TemperatureScaler, compute_ece(), Computes Expected Calibration Error (ECE).          Args:         confidences: l (+15 more)

### Community 4 - "Signal Stability Analysis"
Cohesion: 0.09
Nodes (16): DeltaStabilityAnalyzer, Delta Stability Analyzer — DFD Level-2 Process P4.3 (Signature Buffer D4.1).  Wh, Legacy API: returns (is_stable: bool, temp_id_if_unstable | None).         Used, Return the last detected stable cluster mean and hit count.         Used by the, Generate a deterministic hash from a mean embedding vector,         quantized to, Return last n anomaly log entries as a standard Python list., Implements DFD Level-2 Process P4.3 (Signature Buffer D4.1).      Usage:, Push one unknown embedding.          Args:             embedding:  (EMBED_DIM,) (+8 more)

### Community 5 - "API Integration Testing"
Cohesion: 0.07
Nodes (19): API endpoint tests — Issue #19. Run with: make test, Issue #7: Should reject if EMS_API_KEY is set but no header provided., Issue #7: Should accept with correct API key., When EMS_API_KEY is not set, any request should be accepted., Issue #9: Reject more than 100 segments., Issue #9: Reject segments with wrong dimension (not 128)., Issue #9: Reject device_id longer than 64 chars., CSV export should return 404 when database file doesn't exist. (+11 more)

### Community 6 - "Frontend Dependency Management"
Cohesion: 0.08
Nodes (24): dependencies, lucide-react, react, react-dom, recharts, devDependencies, eslint, @eslint/js (+16 more)

### Community 7 - "Failure Mitigation Protocols"
Cohesion: 0.09
Nodes (13): FailureMatrix, Module 5: Failure Matrix Integration Maps detected failure modes to automated mi, Initialize the Failure Matrix with predefined escalation protocols., Trigger a mitigation protocol based on failure type.                  Args:, Mitigation: Switch to predictive last-known-good state estimation., Mitigation: Cut main breaker tier-1 sub-branch to isolate., Mitigation: Local edge execution mode, queue data for sync., Mitigation: Fall back to rule-based thermodynamics while queuing retrain. (+5 more)

### Community 8 - "Database Session Management"
Cohesion: 0.11
Nodes (10): DatabaseSession, Synchronous CSV write — safe to call from sync context or via asyncio.to_thread., Non-blocking CSV fallback — runs sync file I/O in thread to avoid stalling event, Phase 2 (WS-6.1): Delete measurements older than retention_days every 24h., Phase 2 (WS-6.2): Import CSV fallback data from previous crash, then archive., Generate a deterministic hash from a mean embedding vector,         quantized to, Persist a stable unknown cluster signature for background pseudo-labeling., Retrieve unmapped clusters with >= min_hits occurrences that have         not ye (+2 more)

### Community 9 - "Model Training Utilities"
Cohesion: 0.12
Nodes (11): CNN1DEncoder, detect_transients(), label_to_canonical(), load_ukdale(), make_synthetic(), ProtoNet, Read power from a nilmtk PyTables compound dataset., Realistic turn-on transient with class-specific decay rate. (+3 more)

### Community 10 - "RL Policy Validation"
Cohesion: 0.10
Nodes (13): PolicyPromotionGate, Tracks validation episodes run in the digital twin sandbox.     A policy is 'pro, Record one digital-twin validation episode., True once >= 50 validation episodes with acceptable PMV penalty., WS-2: Docker compose should define all required services., docker-compose.yml should define mosquitto, pipeline, and api services., Mosquitto config file should exist., Test 20/21: PolicyPromotionGate (GAP 8). (+5 more)

### Community 11 - "NILM Transient Detection"
Cohesion: 0.11
Nodes (13): NILMTransientDetector, Module 3: Aggregate NILM — Savitzky-Golay filter + derivative transient detector, Implements the SG-filter + derivative transient detector described in the     Ph, Push one 1 Hz sample.         Returns (is_transient: bool, segment_array | None), Test 13: SG filter + derivative transient detection (GAP 1)., Push steady signal then a >50W spike — should flag transient., Steady signal should not trigger transient flag., Fewer samples than SG window should return (False, None). (+5 more)

### Community 12 - "Frontend UI Components"
Cohesion: 0.10
Nodes (3): DigitalTwin(), getPmvLabel(), DEVICE_COLORS

### Community 13 - "Temporal Attention Mechanisms"
Cohesion: 0.11
Nodes (9): _LegacyTemporalAttention, PreCNNTemporalAttention, ProtoNet: Full implementation for Phase-1.  Components:   1. CNN1DEncoder, Computes a soft weight vector over the 128 raw time-steps.     High-variance (in, x: (batch, 128) raw segment → (batch, 128) weighted., Episodic forward pass.          Args:             support: (N, K, 128)  N classe, Supports two input shapes:           - (batch, 1, seq_len)   → standard, Legacy attention used by old SupportSetManager tests. (+1 more)

### Community 14 - "Temporal Anomaly Validation"
Cohesion: 0.12
Nodes (12): Temporal Validator — bridges the SoftAnomalyWatchdog and the RL Agent.  Architec, Reset anomaly history for a device, or all devices if None., Temporal anomaly validation layer (DFD spec: Temporal Validation → Suggest Relay, Record a watchdog-flagged anomaly event for temporal validation., Record the anomaly and check whether a soft-control suggestion should be issued., TemporalValidator, §3.1 fix: TemporalValidator bridges Watchdog → RL soft control., Single anomaly should not produce a suggestion. (+4 more)

### Community 15 - "REST API Endpoints"
Cohesion: 0.11
Nodes (19): DeviceState, export_csv(), get_devices(), get_pending_labels(), get_phantom(), get_safety_warnings(), get_status(), health_check() (+11 more)

### Community 16 - "Soft Anomaly Monitoring"
Cohesion: 0.18
Nodes (8): Module 2: Soft Anomaly Watchdog A parallel monitoring layer that detects soft an, Initialize the Soft Anomaly Watchdog.                  Args:             window_, Check if a reading is a soft anomaly based on rolling z-score., SoftAnomalyWatchdog, Should not flag anomaly without enough baseline data., Should detect a massive spike after baseline is established., Normal variance should not trigger anomaly., TestWatchdog

### Community 17 - "Model Confidence Calibration"
Cohesion: 0.15
Nodes (7): Temperature Scaling — post-hoc confidence calibration. Reference: Guo et al., "O, Post-hoc temperature scaling (Guo et al., ICML 2017).     Trains a single scalar, Return calibrated softmax probabilities., TemperatureScaler, Tensor, Test 12: End-to-end pipeline for a known device window., TestFullPipeline

### Community 18 - "Phantom Load Tracking"
Cohesion: 0.15
Nodes (8): PhantomTracker, Module 6: Micro-Load (Phantom) Tracker Identifies and tracks continuous small po, Track potential phantom load for a device.                  Args:             de, Calculate the current total phantom load across the system., Get the devices contributing most to the phantom load.                  Returns:, Initialize the Micro-Load Tracker.                  Args:             baseline_t, WS-3.3: Aggregate state space should be tractable (576 states)., TestStateSpaceSize

### Community 19 - "WebSocket Connection Management"
Cohesion: 0.24
Nodes (8): ConnectionManager, heartbeat_task(), lifespan(), Issue #3 fix: Snapshot list under lock, concurrent send via gather.         Issu, Send periodic heartbeat to keep WebSocket connections alive.     Issue #23: Hear, websocket_endpoint(), FastAPI, WebSocket

### Community 20 - "Usage Analytics Engine"
Cohesion: 0.20
Nodes (6): AnalyticsEngine, Module 4: Analytics Engine Processes historical usage data to generate insights,, Initialize the Analytics Engine.                  Args:             cost_per_kwh, Record power usage for a device.                  Args:             device_id: I, Get the usage summary and cost for a specific day.                  Args:, TestAnalytics

### Community 21 - "Tabular Q-Learning Agent"
Cohesion: 0.18
Nodes (5): 0: NIGHT (0-5), 1: MORNING (6-11), 2: DAY (12-17), 3: EVENING (18-23), Categorize into 0: OFF_PEAK, 1: MID, 2: PEAK, TabularQLearningAgent, State discretization should produce states from the 4×4×3×3×4 space., NEVER_SHED device should never get SHED action.

### Community 22 - "API Event Models"
Cohesion: 0.31
Nodes (11): ActionEventModel, AnalyticsUpdateEvent, DeviceStatusEvent, LabelRequestEvent, LowConfidenceEvent, mqtt_listener_task(), PhantomLoadEvent, PMVUpdateEvent (+3 more)

### Community 23 - "Fleet Safety Diagnostics"
Cohesion: 0.22
Nodes (6): FleetDiagnosticsMonitor, Fleet Diagnostics Monitor (formerly SafetyMonitor)  Production architecture: Saf, Log safety event to a file independent of DB for audit trail., Asynchronous fleet diagnostics, logging, and UI alert dispatcher.      Monitors, Separate asyncio task — never awaits the ML pipeline.         Subscribes directl, Client

### Community 24 - "REDD Dataset Loader"
Cohesion: 0.25
Nodes (5): load_redd(), REDD data loader stub.  Phase 1 uses synthetic data from scripts/generate_mock_u, Yields (timestamp, power) tuples from REDD HDF5 dataset., Load REDD dataset. Phase 2 only., REDDLoader

### Community 25 - "UK-DALE Dataset Loader"
Cohesion: 0.25
Nodes (5): load_ukdale(), UK-DALE data loader stub.  Phase 1 uses synthetic data from scripts/generate_moc, Yields (timestamp, power) tuples from UK-DALE HDF5 dataset.          Phase 2:, Load UK-DALE dataset. Phase 2 only., UKDaleLoader

### Community 26 - "Power Mode Classification"
Cohesion: 0.32
Nodes (3): ModeClassifier, Determines if the current power window represents a single device or a         M, TestModeClassifier

### Community 27 - "RL Safety Constraints"
Cohesion: 0.25
Nodes (4): Check if a device is within its per-device anti-thrashing lockout window., Record a device action timestamp using monotonic clock., Check if a device is in the safety-critical blacklist.         Matches against N, Act based on current state. Returns action string.

### Community 28 - "CT Sensor Calibration"
Cohesion: 0.39
Nodes (3): Calibrator, main(), update_firmware_constants()

### Community 29 - "ESP32 Firmware Logic"
Cohesion: 0.33
Nodes (4): byte, callback(), loop(), reconnectMQTT()

### Community 30 - "RL State & Reward"
Cohesion: 0.33
Nodes (3): Aggregate state space (Phase 2 fix — WS-3.3):         - Total load bin: 4 bins (, Synchronous file write — safe to call from sync context or via asyncio.to_thread, Any

### Community 31 - "ESP32 Device Simulator"
Cohesion: 0.47
Nodes (5): Client, main(), ESP32 Simulator — Mock 1Hz sensor data generator. Simulates 10 devices with real, Simulate a single device publishing at 1Hz., simulate_device()

### Community 32 - "Safety Rate-of-Change Testing"
Cohesion: 0.33
Nodes (4): Normal rate of change should not trigger arc-fault., WS-5.3: Rate-of-change arc-fault proxy detection., Rate-of-change > 1000 W/s should trigger immediate relay OFF., TestRateOfChangeSafety

### Community 33 - "Load Shedding Configuration"
Cohesion: 0.33
Nodes (4): WS-3.4: NEVER_SHED list should load from config tier0 flags., Agent should load NEVER_SHED from config's tier0 flags., RL agent should never return SHED for NEVER_SHED devices., TestNEVERSHEDConfig

### Community 34 - "System Integration Testing"
Cohesion: 0.33
Nodes (4): WS-1/WS-2: MQTT topics must align between firmware, pipeline, and config., Config should define topics matching home/sensor/+/power pattern., ProtoNet seq_len should be 128 across all consumers., TestMQTTTopicAlignment

### Community 35 - "Calibration Validation Testing"
Cohesion: 0.33
Nodes (4): Test 15: TemperatureScaler in src/models/calibration.py (GAP 3)., High T should reduce confidence from near-1 raw softmax., Temperature scaler should save and load correctly., TestTemperatureScalerCalibration

### Community 36 - "Relay Protocol Testing"
Cohesion: 0.33
Nodes (4): WS-5.1: Hardware ACK protocol should clear software cooldowns., Receiving an ACK for a device should reset its action cooldown., ACK for a device not in cooldowns should not crash., TestRelayACKProtocol

### Community 37 - "System Orchestration & Config"
Cohesion: 0.40
Nodes (5): CT Calibration Utility, System Configuration, Docker Compose Orchestration, ESP32 Production Firmware, Pipeline Orchestrator

### Community 38 - "Confidence Calibration Logic"
Cohesion: 0.40
Nodes (3): Fit T by minimising NLL on the calibration set.          Args:             logit, Returns (calibrated_prob_array, max_confidence_float).          Args:, ndarray

### Community 39 - "HDF5 Data Replay"
Cohesion: 0.50
Nodes (4): main(), UK-DALE / REDD HDF5 Replay Script (Phase 2 — WS-4.1)  Reads appliance-level powe, Replay HDF5 appliance data as MQTT messages.      Args:         hdf5_path:    Pa, replay()

### Community 40 - "Confidence Gate Testing"
Cohesion: 0.40
Nodes (3): Test 1: Low confidence should make RL agent return DEFER., Test 2: High confidence should allow RL agent to return a real action., TestConfidenceGate

### Community 41 - "Label Submission API"
Cohesion: 0.50
Nodes (3): LabelSubmission, Submit a user-provided label for an unknown device., submit_label()

### Community 42 - "Dashboard UI & Documentation"
Cohesion: 0.50
Nodes (4): React Dashboard Main, Dashboard Bottom Screenshot, Dashboard Top Screenshot, Digital Twin UI Component

### Community 43 - "Database Retention Testing"
Cohesion: 0.50
Nodes (3): WS-6.1: SQLite retention cleanup., Database schema should use autoincrement ID to avoid PK collision., TestDataRetentionPolicy

### Community 44 - "CSV Fallback Testing"
Cohesion: 0.50
Nodes (3): WS-6.2: CSV fallback replay on startup., CSV fallback should have correct column headers., TestCSVFallbackReplay

### Community 45 - "Pipeline Latency Testing"
Cohesion: 0.50
Nodes (3): WS-7.2: Pipeline latency tracking., time.perf_counter should be available for latency measurement., TestPipelineLatencyInstrumentation

### Community 46 - "CSV Persistence Testing"
Cohesion: 0.50
Nodes (3): §3.2 fix: CSV fallback writer should persist data when DB fails., CSV fallback should create file with header and data row., TestCSVFallbackWriter

## Knowledge Gaps
- **41 isolated node(s):** `byte`, `name`, `private`, `version`, `type` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TabularQLearningAgent` connect `Tabular Q-Learning Agent` to `Synthetic Data & MQTT`, `Open-Set Model Inference`, `Thermal Comfort Thermodynamics`, `Encoder Architecture & Calibration`, `Signal Stability Analysis`, `Failure Mitigation Protocols`, `RL Policy Validation`, `NILM Transient Detection`, `Temporal Anomaly Validation`, `Soft Anomaly Monitoring`, `Model Confidence Calibration`, `Phantom Load Tracking`, `Usage Analytics Engine`, `Power Mode Classification`, `RL Safety Constraints`, `RL State & Reward`, `Safety Rate-of-Change Testing`, `Load Shedding Configuration`, `System Integration Testing`, `Calibration Validation Testing`, `Relay Protocol Testing`, `Confidence Gate Testing`, `Database Retention Testing`, `CSV Fallback Testing`, `Pipeline Latency Testing`, `CSV Persistence Testing`, `Phantom Tracker Testing`, `Thermal Zone Classification`, `Device Lockout Management`, `Asynchronous Action Logging`, `Time-of-Use Pricing`, `RL Epsilon Decay Testing`, `RL Convergence Testing`, `Confidence Gating Logic`, `RL Action Validation`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `EMSOrchestrator` connect `Synthetic Data & MQTT` to `Open-Set Model Inference`, `Thermal Comfort Thermodynamics`, `Encoder Architecture & Calibration`, `Signal Stability Analysis`, `Failure Mitigation Protocols`, `Database Session Management`, `RL Policy Validation`, `NILM Transient Detection`, `Temporal Anomaly Validation`, `Soft Anomaly Monitoring`, `Model Confidence Calibration`, `Phantom Load Tracking`, `Usage Analytics Engine`, `Tabular Q-Learning Agent`, `Power Mode Classification`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `DeltaStabilityAnalyzer` connect `Signal Stability Analysis` to `Synthetic Data & MQTT`, `Open-Set Model Inference`, `Thermal Comfort Thermodynamics`, `Encoder Architecture & Calibration`, `Failure Mitigation Protocols`, `Database Session Management`, `RL Policy Validation`, `NILM Transient Detection`, `Temporal Anomaly Validation`, `Soft Anomaly Monitoring`, `Model Confidence Calibration`, `Phantom Load Tracking`, `Usage Analytics Engine`, `Power Mode Classification`, `Safety Rate-of-Change Testing`, `Load Shedding Configuration`, `System Integration Testing`, `Calibration Validation Testing`, `Relay Protocol Testing`, `Confidence Gate Testing`, `Database Retention Testing`, `CSV Fallback Testing`, `Pipeline Latency Testing`, `CSV Persistence Testing`, `Phantom Tracker Testing`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 39 inferred relationships involving `TabularQLearningAgent` (e.g. with `EMSOrchestrator` and `ndarray`) actually correct?**
  _`TabularQLearningAgent` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `OpenMaxWeibull` (e.g. with `EMSOrchestrator` and `ndarray`) actually correct?**
  _`OpenMaxWeibull` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `DeltaStabilityAnalyzer` (e.g. with `EMSOrchestrator` and `ndarray`) actually correct?**
  _`DeltaStabilityAnalyzer` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `ThermodynamicsModel` (e.g. with `EMSOrchestrator` and `ndarray`) actually correct?**
  _`ThermodynamicsModel` has 39 INFERRED edges - model-reasoned connections that need verification._