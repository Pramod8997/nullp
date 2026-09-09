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
        relay_active_low: bool = False,   # tracks the RELAY_ACTIVE_LOW constant in main.cpp
        mqtt_publish_fn: Optional[Callable[[str, str], Coroutine]] = None,
    ) -> None: ...

    def set_relay(self, on: bool) -> None: ...
    def core0_safety_step(self, sim_dt: float = 0.1) -> None: ...
    async def handle_mqtt_command(self, command: str) -> None: ...
    async def core1_telemetry_tick(self, force_publish: bool = False) -> None: ...

    # Attributes (Safe to access):
    device_id: str
    rated_watts: float
    gpio18_relay_state: bool       # Logical relay state: True = ON, False = OFF
    gpio18_level: bool             # Read-only property: electrical pin level
                                   # (True = HIGH). Inverted when relay_active_low.
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
        allowed_classes: Optional[Sequence[str]] = None,   # e.g. config `appliances:`
        reject_radius: float = 6.0,                        # CENTROID_REJECT_RADIUS
    ) -> None: ...

    def classify(self, window: Sequence[float]) -> HeuristicResult: ...
    def extract_features(self, window: Sequence[float]) -> Dict[str, float]: ...
    def feature_vector(self, f: Dict[str, float]) -> np.ndarray: ...
    def classify_batch(self, windows: Sequence[Sequence[float]]) -> List[HeuristicResult]: ...

    # Attributes:
    rules: List[ApplianceRule]     # filtered by allowed_classes at construction
    on_threshold_w: float          # samples below this do not count as "on"
    max_confidence: float          # Capped at 0.75 (never reaches 0.90 RL gate)
    centroids: Dict[str, np.ndarray]
    feature_scales: np.ndarray
    allowed_classes: Optional[set]
    reject_radius: float
    _extra_centroids: set          # centroids with no band rule; exempt from the gate

# ── Module-level, `src/pipeline/heuristic_fallback.py` ──
UNKNOWN = "unknown"                # the reject sentinel; imported by run_pipeline
ON_THRESHOLD_W = 20.0
MAX_HEURISTIC_CONFIDENCE = 0.75
CENTROID_REJECT_RADIUS = 6.0       # scaled-feature units; beyond it -> UNKNOWN
ENVELOPE_SLACK = 0.15              # ±6% mains, P ∝ V², so ~±12% power
DEFAULT_RULES: List[ApplianceRule]
CLASS_CENTROIDS: Dict[str, Tuple[float, ...]]
FEATURE_NAMES  = ("log_steady", "log_peak", "duty", "log_overshoot", "volatility")
FEATURE_SCALES = (0.2063, 0.2233, 1.0000, 0.0272, 0.0474)

def plausible_classes(
    features: Dict[str, float],                     # from extract_features()
    rules: Optional[Sequence[ApplianceRule]] = None,
    slack: float = ENVELOPE_SLACK,
) -> set: ...
# The model-free physical gate. Returns the classes whose measured power
# envelope can contain this window; EMPTY means no known class can draw this
# power, so the caller must report UNKNOWN and request a label. `steady_w` is
# the discriminator; `peak` is tested one-sided.

@dataclass
class HeuristicResult:
    appliance: str
    confidence: float              # <= 0.75
    degraded: bool = True          # Always True
    source: str = "heuristic_fallback"
    features: Dict[str, float]
    runner_up: Optional[str] = None
```

⚠️ **`classify()` returns `UNKNOWN` for a sub-`on_threshold_w` window.** A 10 W
load yields `steady_w == 0.0`, and `feature_vector` then substitutes
`log10(1e-6) = -6` — a fabricated coordinate seven decades below the real power.
Assigning a class from it is meaningless, so the window is rejected. To measure a
sub-20 W load, construct with `on_threshold_w=3.0`. Per CLAUDE.md §1.7, 3–10 W
trickle/standby is `PhantomTracker`'s domain, not this classifier's.

❌ **DO NOT USE:**
* `clf.predict()` $\rightarrow$ Use `clf.classify(window)`
* `clf.infer()` $\rightarrow$ Use `clf.classify(window)`

---

## 3. Models & Calibration (`src/models/`)

### `PrototypeRegistry` (`src/models/protonet.py`)
Few-shot class registry. **This is what `_classify_device` reads** — the live
inference path and the operator-label write path are the same object, which is
what makes the LABEL_REQUEST loop work.

```python
class PrototypeRegistry:
    ENVELOPE_KEY = "__power_envelopes__"      # reserved key inside the saved file

    def __init__(self, encoder: nn.Module, device: str = "cpu") -> None: ...

    def add_class(self, class_name: str, support_segments: np.ndarray) -> None: ...
    # support_segments: (K, 128) raw POWER windows in WATTS, K >= 1.
    # Merges with an existing class (running mean) and WIDENS its envelope, so
    # re-labelling the same device extends the band rather than replacing it.
    def classify(self, segment: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]: ...
    # (best_name, best_squared_distance, {name: squared_distance}).
    # Returns (None, inf, {}) when nothing is enrolled.
    def power_envelope(self, class_name: str) -> Optional[Tuple[float, float]]: ...
    # Observed steady-watt band, or None if unconstrained. `None` is also the
    # test for "shipped class, not operator-enrolled" in _classify_device.
    def class_names(self) -> List[str]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...   # backward-compatible: envelope key popped

    @staticmethod
    def _steady_watts(support_segments: np.ndarray,
                      on_threshold_w: float = 20.0) -> Optional[Tuple[float, float]]: ...
    # Median of samples above threshold, per segment — the SAME definition
    # heuristic_fallback.extract_features uses, so the two envelope sources are
    # directly comparable. None for a sub-20 W load: no bogus band is invented.

    # Attributes:
    prototypes: Dict[str, Tuple[torch.Tensor, int]]   # name -> (embedding, n_support)
    envelopes: Dict[str, Tuple[float, float]]         # name -> (lo_w, hi_w)
```

🔴 **The embedding is power-scale-blind — do not threshold on distance to detect
novelty.** Measured on the shipped demo artefact: a 500 W washing machine lands
`d2 = 1.02` from the **5 W** `router` prototype, and an 800 W heater sits `1.38`
from its nearest class while genuine in-distribution windows reach `8.15`. Novel
distances fall *inside* the known range, so neither a plain threshold nor the
OpenMax Weibull tail can separate known from novel. Absolute watts are the only
reliable novelty signal, which is why rejection is done by `plausible_classes` /
`_eligible_classes` on measured power. Pinned by
`tests/test_ml_pipeline_recognition.py::TestOpenSetRejection`.

⚠️ `SupportSetManager` (same module) is a **separate, legacy** registry populated
only from `protonet.anchors_path` — set in `config/config.yaml` alone. Its
`.raw_windows` is `{}` on the demo and hardware profiles. `_classify_device` only
falls back to it when `raw_windows` is non-empty.

---

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

    # ── Recognition path (rewritten 2026-08-25, defects M-5…M-8) ──
    def _classify_device(
        self, device_id: str, power_watts: float,
        filtered_segment: np.ndarray = None,
    ) -> Tuple[str, float, Dict[str, float]]: ...
    # Returns (class_name, confidence, distances). `class_name` is one of:
    #   a registered class name -> recognised, confident enough to act on
    #   "unknown"               -> UNRECOGNISED; drives the LABEL_REQUEST flow
    #   "pending"               -> not enough samples buffered yet
    #   "error"                 -> inference raised
    # Reads `prototype_registry`, NOT `support_manager`: the latter is populated
    # only from `protonet.anchors_path` (config/config.yaml alone), so on the
    # demo/hardware profiles it returned ("unknown", 0.0, {}) for every event.

    def _eligible_classes(self, window_np: np.ndarray, names: List[str], registry) -> set: ...
    # Envelope source in order: registry (enrolled) -> DEFAULT_RULES ->
    # unconstrained. The gate may only veto where it has knowledge.
    def _registry_heuristic(self, names: List[str]) -> HeuristicApplianceClassifier: ...
    # Confirmation channel B, scoped to the registry's class set, cached per set.
    def _classify_heuristic(self, window_np: np.ndarray) -> Tuple[str, float, dict]: ...
    # Degraded mode only (no encoder / empty registry); gated by
    # heuristic_min_confidence, else UNRECOGNISED.

    # Attributes:
    config: Dict
    safety: SafetyMonitor
    env: DigitalTwinEnv
    phantom_tracker: PhantomTracker
    watchdog: SoftAnomalyWatchdog
    analytics: AnalyticsEngine
    heuristic_clf: HeuristicApplianceClassifier   # scoped by config `appliances:`
    encoder: Optional[ProtoNet]
    prototype_registry: Optional[PrototypeRegistry]
    weibull: OpenMaxWeibull
    calibrated_scaler: Optional[CalibratedTemperatureScaler]
    nilm_detectors: Dict[str, NILMTransientDetector]
    agent: TabularQLearningAgent
    recognition_threshold: float                  # 0.45; NOT confidence_threshold
    heuristic_min_confidence: float               # 0.55
    _unknown_windows: Dict[str, deque]            # raw watt traces for enrollment
    _registry_clf_cache: Dict[tuple, HeuristicApplianceClassifier]

# ── Module-level, `scripts/run_pipeline.py` ──
UNRECOGNISED = UNKNOWN = "unknown"          # imported from heuristic_fallback
UNRECOGNISED_DISPLAY = "Unrecognised device"  # human-facing label only
FullPipeline = EMSOrchestrator                # alias
```

⚠️ **`recognition_threshold` (0.45) is not `confidence_threshold` (0.90).** The
latter gates a single softmax; the former gates the noisy-OR of two independent
channels that have already agreed on the class. They are different quantities and
must not share a number. Both are now explicit in all three config profiles.

⚠️ **`"unknown"` is the wire/internal sentinel and must stay that literal** —
`src/api/main.py`, the dashboard's `DeviceCards.jsx` and 54 test assertions key on
it. `UNRECOGNISED_DISPLAY` is what a human reads.

### `LABEL_REQUEST` event contract

```python
{
  "type": "LABEL_REQUEST", "device_id": str, "power": float, "confidence": float,
  "segments": List[List[float]],   # raw 128-sample POWER windows, in WATTS
  "embedding": List[float],        # display/clustering ONLY — never enroll this
  "suggested_label": str, "message": str,
}
```

🔴 **`segments` is what enrollment consumes; `embedding` is not.** Both are
length-128 float arrays, so the shape check alone could not tell them apart — the
dashboard submitted the embedding and `add_class()` ran the CNN over it as though
it were a power trace, silently building a prototype from nonsense (defect M-8).
Watts are non-negative and a real appliance event clears 20 W; embeddings are
zero-centred and fail both. Now refused at **both** layers —
`LabelSubmission.validate_segments` (HTTP) and `handle_label_submitted` (in-process).

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

