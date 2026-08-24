# Exact API Cheatsheet & Anti-Hallucination Reference

> **Target:** Claude (Opus 4.5 / 5 / Sonnet) & Technical Agents  
> **Rule:** NEVER guess or invent method names. Refer to this explicit interface dictionary.

---

## 1. Hardware Simulator Layer (`src/hardware/`)

### `ESP32FirmwareNode` (`src/hardware/esp32_firmware_sim.py`)
Emulates the dual-core FreeRTOS ESP32 node.

```python
class ESP32FirmwareNode:
    def __init__(
        self,
        device_id: str,
        rated_watts: float = 200.0,
        relay_active_low: bool = True,
        mqtt_publish_fn: Optional[Callable[[str, str], Coroutine]] = None,
    ) -> None: ...

    def set_relay(self, on: bool) -> None: ...
    def core0_safety_step(self, sim_dt: float = 0.1) -> None: ...
    async def handle_mqtt_command(self, command: str) -> None: ...
    async def core1_telemetry_tick(self, force_publish: bool = False) -> None: ...

    # Attributes (Safe to access):
    device_id: str
    rated_watts: float
    gpio18_relay_state: bool       # True = ON, False = OFF
    relay_locked: bool             # True when in 300s cooldown
    lock_start_time: float         # time.time() of trip
    safety_lockout_seconds: float  # Default: 300.0
    pzem: VirtualPZEM004T
    shared_power_watts: float
    shared_voltage: float
    shared_current: float
    shared_pf: float
    shared_arc_fault: bool
    shared_arc_fault_roc: float
    _last_watts: float
    _baseline_ring: List[float]    # 5-sample ring buffer
    _baseline_idx: int
    _baseline_fill: int
    topic_power: str
    topic_telemetry: str
    topic_command: str
    topic_status: str
    topic_ack: str
```

❌ **DO NOT USE (Non-existent attributes/methods):**
* `node.relay_state` $\rightarrow$ Use `node.gpio18_relay_state`
* `node.sensor` $\rightarrow$ Use `node.pzem`
* `node.wifi_connected` $\rightarrow$ Use MQTT client `._connected`
* `node.relay_pin` $\rightarrow$ Fixed at GPIO 18
* `node.process_reading()` $\rightarrow$ Use `node.core0_safety_step()`
* `node.read_power()` $\rightarrow$ Access `node.shared_power_watts`

---

### `VirtualPZEM004T` (`src/hardware/esp32_firmware_sim.py`)
Simulates the PZEM-004T v3.0 Modbus RTU metering registers.

```python
class VirtualPZEM004T:
    def __init__(self, voltage: float = 230.0, frequency: float = 50.0) -> None: ...
    def set_load(self, target_watts: float, pf: float = 0.95) -> None: ...

    # Attributes:
    voltage: float
    current: float
    active_power: float
    power_factor: float
    frequency: float
    energy_kwh: float
```

❌ **DO NOT USE:**
* `pzem.parse_modbus_frame()` $\rightarrow$ Frame parsing is internal to C++ library
* `pzem.read_power()` $\rightarrow$ Access `pzem.active_power`

---

### `AsyncMQTTClient` (`src/hardware/mqtt.py`)
In-memory async MQTT client for testing and pipeline integration.

```python
class AsyncMQTTClient:
    def __init__(
        self,
        on_message: Optional[Callable[[str, str], Coroutine]] = None,
        broker: str = "localhost",
        port: int = 1883,
    ) -> None: ...

    async def subscribe(self, topic: str) -> None: ...
    async def publish(self, topic: str, payload: Union[str, bytes]) -> None: ...
    async def disconnect(self) -> None: ...
    async def reconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    async def get_published(self, topic_filter: Optional[str] = None) -> List[Any]: ...

    # Attributes:
    subscriptions: Set[str]
    _connected: bool
    published_messages: List[Tuple[str, str]]
```

---

### `MockMQTTBroker` (`src/hardware/mqtt.py`)
Simulates broker outages and connection drops.

```python
class MockMQTTBroker:
    def __init__(self) -> None: ...
    def register(self, client: AsyncMQTTClient) -> None: ...
    def unregister(self, client: AsyncMQTTClient) -> None: ...
    async def disconnect_all(self) -> None: ...  # NOTE: Must be awaited!
    async def restart(self) -> None: ...         # NOTE: Must be awaited!
```

---

## 2. Pipeline & Safety Analytics (`src/pipeline/`)

### `FleetDiagnosticsMonitor` (`src/pipeline/safety.py`)
Server-side aggregate and device-level safety supervisor.

```python
class FleetDiagnosticsMonitor:
    def __init__(
        self,
        max_aggregate_wattage: Optional[float] = None,
        device_wattage_limits: Optional[Dict[str, float]] = None,
        warning_pct: float = 1.10,
        critical_pct: float = 1.25,
        config: Optional[Dict] = None,
        safety_log_path: str = "safety_events.log",
        db_session = None,
    ) -> None: ...

    async def check_aggregate(self, power_map: Dict[str, float]) -> SafetyEvent: ...
    async def check_roc(self, device: str, prev_power: float, curr_power: float, dt_seconds: float = 1.0) -> Optional[SafetyEvent]: ...
    async def check_device(self, device: str, power: float) -> Optional[SafetyEvent]: ...
    def _log_event_sync(self, level: str, device_id: str, watts: float, pct_or_roc: float) -> None: ...
    async def _log_event_async(self, level: str, device_id: str, watts: float, pct_or_roc: float) -> None: ...

    # Attributes:
    max_aggregate_wattage: float
    device_wattage_limits: Dict[str, float]
    warning_pct: float
    critical_pct: float
    ROC_THRESHOLD: float           # Default: 1000.0
    _prev_readings: Dict[str, float]
    _current_readings: Dict[str, float]
```

❌ **DO NOT USE:**
* `monitor.update_reading()` $\rightarrow$ Direct set `monitor._current_readings[id] = val` or call `check_device()`
* `monitor.log_event()` $\rightarrow$ Use `monitor._log_event_sync()`
* `monitor.trigger_safety_event()` $\rightarrow$ Use `await monitor.check_device()`
* `monitor.is_heartbeat_lost()` $\rightarrow$ Compare `time.time() - last_seen`

---

### `NILMTransientDetector` (`src/pipeline/aggregate_nilm.py`)
Savitzky-Golay filtering and transient onset detector.

```python
class NILMTransientDetector:
    def __init__(
        self,
        window_size: int = 5,
        sg_window: int = 7,
        sg_polyord: int = 2,
        threshold: float = 20.0,
        embed_window: int = 128,
        sample_rate_hz: int = 1,
    ) -> None: ...

    def push(self, power_w: float) -> Tuple[bool, Optional[np.ndarray]]: ...
    def reset(self) -> None: ...

    # Attributes:
    _buffer: List[float]           # Trims to 3 * embed_window
    _cooldown: int
    threshold: float
    embed_window: int
```

---

### `OverlapAwareNILMDetector` (`src/pipeline/aggregate_nilm.py`)
Wraps transient detector with multi-appliance baseline power subtraction.

```python
class OverlapAwareNILMDetector:
    def __init__(
        self,
        base_detector: Optional[NILMTransientDetector] = None,
        appliance_baselines: Optional[Dict[str, float]] = None,
        embed_window: int = 128,
    ) -> None: ...

    def register_appliance_state(self, appliance: str, is_active: bool, power_w: Optional[float] = None) -> None: ...
    def push_sample(self, aggregate_power_w: float) -> List[Tuple[str, np.ndarray]]: ...
```

---

### `SoftAnomalyWatchdog` (`src/pipeline/watchdog.py`)
Rolling z-score anomaly detector for steady-state drift.

```python
class SoftAnomalyWatchdog:
    def __init__(self, window_size: int = 30, threshold: float = 3.0) -> None: ...
    def update(self, device_id: str, reading: float) -> Tuple[bool, float]: ...
    def get_zscore(self, device_id: str, reading: Optional[float] = None) -> float: ...

    # Attributes:
    window_size: int
    threshold: float
    history: Dict[str, deque]
```

---

### `HeuristicApplianceClassifier` (`src/pipeline/heuristic_fallback.py`)
Deterministic nearest-centroid power-signature fallback classifier (zero torch dependency).

```python
class HeuristicApplianceClassifier:
    def __init__(
        self,
        rules: Optional[Sequence[ApplianceRule]] = None,
        on_threshold_w: float = 20.0,
        max_confidence: float = 0.75,
        centroids: Optional[Dict[str, Sequence[float]]] = None,
        feature_scales: Optional[Sequence[float]] = None,
    ) -> None: ...

    def classify(self, window: Sequence[float]) -> HeuristicResult: ...
    def extract_features(self, window: Sequence[float]) -> Dict[str, float]: ...
    def feature_vector(self, f: Dict[str, float]) -> np.ndarray: ...

    # Attributes:
    rules: List[ApplianceRule]
    on_threshold_w: float
    max_confidence: float          # Capped at 0.75 (never reaches 0.90 RL gate)
    centroids: Dict[str, np.ndarray]
    feature_scales: np.ndarray

@dataclass
class HeuristicResult:
    appliance: str
    confidence: float              # <= 0.75
    degraded: bool = True          # Always True
    source: str = "heuristic_fallback"
    features: Dict[str, float]
    runner_up: Optional[str] = None
```

❌ **DO NOT USE:**
* `clf.predict()` $\rightarrow$ Use `clf.classify(window)`
* `clf.infer()` $\rightarrow$ Use `clf.classify(window)`

---

## 3. Models & Calibration (`src/models/`)

### `TemperatureScaler` (`src/models/calibration.py`)
Post-hoc temperature scaling calibration for deep learning logits.

```python
class TemperatureScaler(torch.nn.Module):
    def __init__(self) -> None: ...
    def forward(self, logits: torch.Tensor) -> torch.Tensor: ...  # Clamps T >= 0.05
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float: ...

def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor: ...
def confidence_gate(probabilities: torch.Tensor, threshold: float = 0.90) -> str: ...
# Returns "PASS_RL" if max(prob) >= threshold, else "SKIP_RL"
```

---

## 4. Pipeline Orchestration & Simulation Layer (`scripts/`)

### `EMSOrchestrator` (`scripts/run_pipeline.py`)
Central orchestrator managing parallel safety tasks, ProtoNet + heuristic classification, PMV thermal simulation, and RL policy actions.

```python
class EMSOrchestrator:
    def __init__(
        self,
        config: Optional[Dict] = None,
        stage_hook: Optional[Callable[[str], None]] = None,
        rl_hook: Optional[Callable[[], None]] = None,
    ) -> None: ...

    async def run(self) -> None: ...
    def shutdown(self) -> None: ...
    def handle_label_submitted(self, class_name: str, segments_list: list) -> None: ...
    async def process_raw_mqtt(self, topic: str, payload: Union[str, bytes, bytearray, dict, float, int]) -> PipelineResult: ...
    async def process(self, event: Any) -> PipelineResult: ...

    # Attributes:
    config: Dict
    safety: SafetyMonitor
    env: DigitalTwinEnv
    phantom_tracker: PhantomTracker
    watchdog: SoftAnomalyWatchdog
    analytics: AnalyticsEngine
    heuristic_clf: HeuristicApplianceClassifier
    encoder: Optional[ProtoNet]
    prototype_registry: Optional[PrototypeRegistry]
    weibull: OpenMaxWeibull
    calibrated_scaler: Optional[CalibratedTemperatureScaler]
    nilm_detectors: Dict[str, NILMTransientDetector]
    agent: TabularQLearningAgent
```

### CLI Execution Modes:
* **Standard Profile (3500W ceiling):** `python scripts/run_pipeline.py`
* **Demo Profile (600W bench ceiling, 7 classes):** `python scripts/run_pipeline.py --config config/config.demo.yaml`

---

### `ESP32 Telemetry Simulator` (`backend/scripts/simulate_esp32.py`)
Mock 1Hz sensor telemetry generator for virtual hardware fleets.

```python
# Hardware Profiles:
DEVICES: List[Dict]       # 10 household appliances (fridge, microwave, kettle, hvac, tv, washer, dryer, dishwasher, oven, lighting)
DEMO_DEVICES: List[Dict]  # 5 benchtop electronics (node_laptop, node_desktop, node_monitor, node_projector, node_charger)
```

### CLI Execution Modes:
* **Standard Fleet:** `python backend/scripts/simulate_esp32.py --all`
* **Demo Electronics Fleet:** `python backend/scripts/simulate_esp32.py --demo`

