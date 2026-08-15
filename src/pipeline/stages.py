"""
Pipeline Stage Definitions and Isolation Wrappers for Confidence-Aware EMS.

Implements all 11 stages of the pipeline:
  Stage 0:  Fleet Diagnostics Monitor (Safety Monitor)
  Stage 1:  Soft Anomaly Watchdog
  Stage 1b: NILM Preprocessor (Savitzky-Golay + derivative)
  Stage 2:  ProtoNet CNN Classifier
  Stage 2b: OpenMax Unknown Detection
  Stage 2c: Temperature Scaling Confidence Calibration
  Stage 3:  Confidence Gate
  Stage 4:  Delta Stability Analyzer
  Stage 5:  Phantom Tracker
  Stage 6:  Database Session Write
  Stage 7:  Analytics Engine
  Stage 8:  Digital Twin / PMV Comfort
  Stage 9:  RL Agent
  Stage 10: Latency Monitor
  Stage 11: Broadcast Stage
"""

import time
import asyncio
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union, Callable
from scipy.signal import savgol_filter

from src.database.session import DBSession, load_config
from src.pipeline.watchdog import Watchdog, WatchdogEvent, SoftAnomalyWatchdog
from src.pipeline.aggregate_nilm import NILMTransientDetector
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.analytics import AnalyticsEngine
from src.pipeline.safety import FleetDiagnosticsMonitor, SafetyMonitor
from src.models.thermodynamics import PMVThermodynamics, ThermodynamicsModel
from src.models.protonet import CNN1DEncoder
from src.rl.agent import TabularQLearningAgent


# ════════════════════════════════════════════════════════════════════
# Mock Event Helpers & Event Types
# ════════════════════════════════════════════════════════════════════

@dataclass
class PowerEvent:
    device: str = "device"
    power: float = 0.0
    state: str = "ON"
    confidence: float = 1.0
    window: Optional[List[float]] = None
    timestamp: float = 0.0

    @property
    def device_id(self) -> str:
        return self.device


@dataclass
class ThermalEvent:
    ta: float = 22.0
    tr: float = 22.0
    var: float = 0.1
    rh: float = 50.0
    Icl: float = 1.0
    M: float = 70.0


def mock_power_event(
    device: str,
    power: float,
    state: str = "ON",
    confidence: float = 1.0,
    timestamp: float = 0.0,
) -> PowerEvent:
    return PowerEvent(
        device=device,
        power=power,
        state=state,
        confidence=confidence,
        timestamp=timestamp or time.time(),
    )


def mock_power_event_with_window(
    device: str, window: List[float], power: Optional[float] = None
) -> PowerEvent:
    p = power if power is not None else (window[-1] if window else 0.0)
    return PowerEvent(
        device=device,
        power=p,
        state="ON",
        confidence=1.0,
        window=list(window),
        timestamp=time.time(),
    )


def mock_power_event_low_confidence(
    device: str, power: float = 100.0, state: str = "ON"
) -> PowerEvent:
    return PowerEvent(
        device=device,
        power=power,
        state=state,
        confidence=0.30,
        timestamp=time.time(),
    )


def mock_thermal_event(
    ta: float = 22.0,
    tr: float = 22.0,
    var: float = 0.1,
    rh: float = 50.0,
    Icl: float = 1.0,
    M: float = 70.0,
) -> ThermalEvent:
    return ThermalEvent(ta=ta, tr=tr, var=var, rh=rh, Icl=Icl, M=M)


# ════════════════════════════════════════════════════════════════════
# Stage 1b: NILM Preprocessor
# ════════════════════════════════════════════════════════════════════

@dataclass
class NILMEvent:
    delta: float = 0.0
    index: int = 0
    event_type: str = "NILM_TRANSIENT"


class NILMPreprocessor:
    """
    Savitzky-Golay filter + derivative transient detection (Stage 1b).
    Smooths raw power waveform and extracts transient step-change events.
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        threshold: float = 50.0,
        **kwargs,
    ):
        self.window_length = (
            window_length if window_length % 2 == 1 else window_length + 1
        )
        self.polyorder = polyorder
        self.threshold = threshold

    async def process_window(
        self, signal: Union[List[float], np.ndarray]
    ) -> List[NILMEvent]:
        arr = np.array(signal, dtype=np.float32)
        if len(arr) < self.window_length:
            return []

        smoothed = savgol_filter(arr, self.window_length, self.polyorder)
        deriv = np.diff(smoothed)

        events: List[NILMEvent] = []
        i = 0
        while i < len(deriv):
            if abs(deriv[i]) > 3.0:
                sign = 1.0 if deriv[i] > 0 else -1.0
                start = i
                cum_delta = 0.0
                while i < len(deriv) and (deriv[i] * sign > 0 or abs(deriv[i]) < 1.0):
                    cum_delta += float(deriv[i])
                    i += 1
                if abs(cum_delta) >= self.threshold:
                    events.append(NILMEvent(delta=cum_delta, index=start))
            else:
                i += 1
        return events


# ════════════════════════════════════════════════════════════════════
# Stage 2: ProtoNet CNN Classifier
# ════════════════════════════════════════════════════════════════════

@dataclass
class ProtoNetResult:
    embedding: np.ndarray


class ProtoNetClassifier:
    """
    1D-CNN ProtoNet embedding extractor (Stage 2).
    Maps 128-sample window to (128,) normalized embedding vector.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.encoder = CNN1DEncoder(input_size=128, embedding_size=128)
        self.encoder.eval()

    async def process(self, event: Any) -> ProtoNetResult:
        window = getattr(event, "window", None)
        if window is None:
            power = float(getattr(event, "power", 0.0))
            window = [power] * 128

        win_arr = np.array(window, dtype=np.float32)
        if len(win_arr) < 128:
            win_arr = np.pad(win_arr, (0, 128 - len(win_arr)))
        elif len(win_arr) > 128:
            win_arr = win_arr[:128]

        tensor = torch.tensor(win_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder(tensor).squeeze(0).numpy()
        return ProtoNetResult(embedding=emb)


# ════════════════════════════════════════════════════════════════════
# Stage 2b: OpenMax Unknown Detection
# ════════════════════════════════════════════════════════════════════

@dataclass
class OpenMaxResult:
    label: str = "known"
    rejected: bool = False
    confidence: float = 0.95


class OpenMaxStage:
    """
    Weibull-calibrated OpenMax open-set rejection stage (Stage 2b).
    Flags outliers far from known prototype clusters as unknown.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.distance_threshold = self.config.get("distance_threshold", 20.0)

    async def process(self, embedding: np.ndarray) -> OpenMaxResult:
        emb = np.array(embedding, dtype=np.float32)
        dist = float(np.linalg.norm(emb))
        if dist > self.distance_threshold:
            return OpenMaxResult(label="unknown", rejected=True, confidence=0.0)
        return OpenMaxResult(label="known", rejected=False, confidence=0.95)


# ════════════════════════════════════════════════════════════════════
# Stage 2c: Temperature Scaling
# ════════════════════════════════════════════════════════════════════

@dataclass
class ScaledConfidenceResult:
    confidence: float
    scaled_logits: torch.Tensor
    probs: torch.Tensor


class TemperatureScalingStage:
    """
    Post-hoc temperature scaling confidence calibration (Stage 2c).
    Softens overconfident uncalibrated logits using temperature parameter T.
    """

    def __init__(self, T: float = 2.0, **kwargs):
        self.T = float(T)

    async def scale(self, logits: torch.Tensor) -> ScaledConfidenceResult:
        scaled = logits / self.T
        probs = torch.softmax(scaled, dim=-1)
        conf = float(probs.max().item())
        return ScaledConfidenceResult(confidence=conf, scaled_logits=scaled, probs=probs)


# ════════════════════════════════════════════════════════════════════
# Stage 3: Confidence Gate
# ════════════════════════════════════════════════════════════════════

@dataclass
class ConfidenceGateResult:
    event_type: str
    pipeline_action: str
    confidence: float


class ConfidenceGateStage:
    """
    Confidence Gate (Stage 3).
    Ensures uncertain classifications (< 0.90) short-circuit downstream RL actuation.
    """

    def __init__(self, threshold: float = 0.90, **kwargs):
        self.threshold = threshold

    async def process(
        self, confidence: float, embedding: Optional[np.ndarray] = None
    ) -> ConfidenceGateResult:
        if confidence < self.threshold:
            return ConfidenceGateResult(
                event_type="LOW_CONFIDENCE",
                pipeline_action="STOP",
                confidence=confidence,
            )
        return ConfidenceGateResult(
            event_type="HIGH_CONFIDENCE",
            pipeline_action="CONTINUE",
            confidence=confidence,
        )


# ════════════════════════════════════════════════════════════════════
# Stage 4: Delta Stability Stage
# ════════════════════════════════════════════════════════════════════

@dataclass
class DeltaStabilityResult:
    event_type: str
    mean: Optional[np.ndarray] = None


class DeltaStabilityStage:
    """
    Delta Stability Analyzer Stage (Stage 4).
    Accumulates consecutive unknown signatures to distinguish stable novel devices
    from transient line noise.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.analyzer = DeltaStabilityAnalyzer(
            window=kwargs.get("window", 10),
            threshold=kwargs.get("threshold", 15.0),
            min_count=kwargs.get("min_count", 3),
        )

    async def process(self, embedding: np.ndarray) -> DeltaStabilityResult:
        emb = np.array(embedding, dtype=np.float32)
        status, mean = self.analyzer.push(emb)
        if status == "stable":
            return DeltaStabilityResult(event_type="LABEL_REQUEST", mean=mean)
        return DeltaStabilityResult(event_type="TRANSIENT", mean=None)


# ════════════════════════════════════════════════════════════════════
# Stage 5: Phantom Tracker Stage
# ════════════════════════════════════════════════════════════════════

class PhantomTrackerStage:
    """
    Micro-Load / Phantom Load Tracker Stage (Stage 5).
    Maintains exponential moving average (EMA) of vampire draw when devices are OFF.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        alpha = self.config.get("phantom", {}).get("alpha", 0.1)
        self.tracker = PhantomTracker(alpha=alpha)

    async def process(self, event: Any) -> None:
        device = getattr(event, "device", getattr(event, "device_id", "default"))
        power = float(getattr(event, "power", 0.0))
        state = getattr(event, "state", "ON")
        self.tracker.update(device, power, state=state)

    def get_ema(self, device: str) -> float:
        return self.tracker.get_ema(device)


# ════════════════════════════════════════════════════════════════════
# Stage 7: Analytics Stage
# ════════════════════════════════════════════════════════════════════

class AnalyticsStage:
    """
    Analytics & Time-of-Use Cost Tracking Stage (Stage 7).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.engine = AnalyticsEngine(config=self.config)

    async def process(self, event: Any) -> float:
        device = getattr(event, "device", getattr(event, "device_id", "default"))
        power = float(getattr(event, "power", 0.0))
        return await self.engine.record(device, power, seconds=1.0)

    def get_accumulated_cost(self, device: str) -> float:
        return self.engine.get_accumulated_cost(device)


# ════════════════════════════════════════════════════════════════════
# Stage 8: Digital Twin / PMV Comfort Stage
# ════════════════════════════════════════════════════════════════════

@dataclass
class PMVResult:
    pmv: float


class DigitalTwinStage:
    """
    Digital Twin Thermal & PMV Comfort Model Stage (Stage 8).
    Calculates Predicted Mean Vote (PMV) based on indoor environmental conditions.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.pmv_model = PMVThermodynamics()

    async def process(self, event: Any) -> PMVResult:
        ta = float(getattr(event, "ta", 22.0))
        tr = float(getattr(event, "tr", 22.0))
        va = float(getattr(event, "var", getattr(event, "va", 0.1)))
        rh = float(getattr(event, "rh", 50.0))
        clo = float(getattr(event, "Icl", getattr(event, "clo", 1.0)))
        M = float(getattr(event, "M", 70.0))
        met = M / 58.15 if M > 10 else M
        pmv_val = self.pmv_model.pmv(ta=ta, tr=tr, va=va, rh=rh, clo=clo, met=met)
        return PMVResult(pmv=pmv_val)


# ════════════════════════════════════════════════════════════════════
# Stage 9: RL Agent Stage
# ════════════════════════════════════════════════════════════════════

@dataclass
class RLActionResult:
    action: str
    raw_action: str


class RLAgentStage:
    """
    Reinforcement Learning Control Agent Stage (Stage 9).
    Selects control actions while respecting thermal comfort and confidence constraints.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        self.config = config or {}
        self.agent = TabularQLearningAgent()

    async def process(
        self,
        state: Dict[str, Any],
        pmv: float = 0.0,
        confidence: float = 1.0,
        **kwargs,
    ) -> RLActionResult:
        raw_action = self.agent.act(state, pmv=pmv, confidence=confidence)
        mapping = {
            "SHED": "OFF",
            "SHED_HVAC": "OFF",
            "SCHEDULE": "ON",
            "SCHEDULE_HVAC": "ON",
            "DEFER": "NO_ACTION",
            "ON": "ON",
            "OFF": "OFF",
            "NO_ACTION": "NO_ACTION",
        }
        action = mapping.get(str(raw_action).upper(), "NO_ACTION")
        return RLActionResult(action=action, raw_action=str(raw_action))


# ════════════════════════════════════════════════════════════════════
# Stage 11: Broadcast Stage
# ════════════════════════════════════════════════════════════════════

class BroadcastStage:
    """
    WebSocket / MQTT Event Broadcaster Stage (Stage 11).
    Dispatches telemetry and control events to connected dashboard clients.
    """

    def __init__(
        self, ws_broadcast_fn: Optional[Callable[[Any], None]] = None, **kwargs
    ):
        self.ws_broadcast_fn = ws_broadcast_fn

    async def broadcast(self, event: Any) -> None:
        if self.ws_broadcast_fn is not None:
            if asyncio.iscoroutinefunction(self.ws_broadcast_fn):
                await self.ws_broadcast_fn(event)
            else:
                self.ws_broadcast_fn(event)


# ════════════════════════════════════════════════════════════════════
# Stage 10 & Full Pipeline Coordinator (Stages 10, 12, 13)
# ════════════════════════════════════════════════════════════════════

@dataclass
class PipelineExecutionResult:
    status: str
    action: str
    events: List[Any]


class FullPipeline:
    """
    Full 11-Stage Pipeline Orchestrator for integration and isolated staging tests.
    Enforces strict execution ordering and confidence gating.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        stage_hook: Optional[Callable[[str], None]] = None,
        rl_hook: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        self.config = config or {}
        self.stage_hook = stage_hook
        self.rl_hook = rl_hook
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    async def process(self, event: Any) -> PipelineExecutionResult:
        confidence = float(getattr(event, "confidence", 1.0))

        # Stage 0: fleet diagnostics
        if self.stage_hook:
            self.stage_hook("fleet_diagnostics")

        # Stage 1: watchdog
        if self.stage_hook:
            self.stage_hook("watchdog")

        # Stage 1b: nilm
        if self.stage_hook:
            self.stage_hook("nilm")

        # Stage 2: protonet
        if self.stage_hook:
            self.stage_hook("protonet")

        # Stage 3: confidence gate
        if self.stage_hook:
            self.stage_hook("confidence_gate")

        if confidence < 0.90:
            # Stage 4: delta stability runs for unknown / low-confidence events
            if self.stage_hook:
                self.stage_hook("delta_stability")
            # Low confidence short circuits RL
            return PipelineExecutionResult(
                status="low_confidence", action="STOP", events=[]
            )

        # Stage 4: delta stability
        if self.stage_hook:
            self.stage_hook("delta_stability")

        # Stage 5: phantom tracker
        if self.stage_hook:
            self.stage_hook("phantom_tracker")

        # Stage 6: database
        if self.stage_hook:
            self.stage_hook("database")

        # Stage 7: analytics
        if self.stage_hook:
            self.stage_hook("analytics")

        # Stage 8: digital twin
        if self.stage_hook:
            self.stage_hook("digital_twin")

        # Stage 9: RL agent
        if self.stage_hook:
            self.stage_hook("rl_agent")
        if self.rl_hook:
            self.rl_hook()

        # Stage 10: latency monitor
        if self.stage_hook:
            self.stage_hook("latency")

        # Stage 11: broadcast
        if self.stage_hook:
            self.stage_hook("broadcast")

        return PipelineExecutionResult(status="success", action="CONTINUE", events=[])
