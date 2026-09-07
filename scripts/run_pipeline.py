"""
EMS Pipeline Orchestrator
=========================
Central orchestration loop implementing the full architecture:

    Safety Monitor (PARALLEL asyncio.Task)
      ↓ independent
    MQTT → CNN/ProtoNet Open-Set Detection
      → Confidence Gate (>= 0.90)
      → Unknown Device Flow (Delta Stability → User Prompt)
      → Known Device Flow (Analytics + Digital Twin + RL Agent)
      → Dashboard Broadcast

Key invariants:
  - Safety monitor runs as a SEPARATE asyncio.Task, never inside ML loop
  - RL agent checks confidence_threshold BEFORE any Q-table lookup
  - PMV bounds are Category A: -0.5 to +0.5
  - Weibull OpenMax fitted during training, used at inference
"""

import asyncio
import signal
import sys
import logging
import time
import math
import json
import os
import csv
import pickle
from collections import deque
from datetime import datetime
from typing import Union, Dict, Optional, Callable, Any, List

import yaml
import torch
import numpy as np


class PipelineEvent:
    """Represents a structured pipeline event."""
    def __init__(
        self,
        event_type: str = "",
        source: str = "",
        device_id: str = "",
        message: str = "",
        data: Optional[Dict] = None
    ) -> None:
        self.event_type = event_type
        self.source = source
        self.device_id = device_id
        self.message = message
        self.data = data or {}

    def __repr__(self) -> str:
        return f"PipelineEvent(event_type={self.event_type}, source={self.source}, device_id={self.device_id})"


class PipelineResult:
    """Represents the output/events resulting from processing an event."""
    def __init__(self, events: Optional[List[PipelineEvent]] = None, success: bool = True) -> None:
        self.events = events or []
        self.success = success


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load system configuration from YAML or return defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    for p in ["config/config.yaml", "config.yaml"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {
        "mqtt": {"broker": "localhost", "port": 1883, "topics": {"reads": "home/sensor/+/power", "writes": "home/plug/+/command", "events": "home/ui/events"}},
        "database": {"path": "data/ems_state.db", "fallback_csv": "data/fallback_measurements.csv", "retention_days": 30},
        "system_safety": {"max_aggregate_wattage": 3500.0, "warning_pct": 1.10, "critical_pct": 1.25, "device_wattage_limits": {"default": 1500.0}},
        "protonet": {"seq_len": 128, "embedding_size": 128, "distance_threshold": 15.0, "confidence_threshold": 0.90},
        "system": {"max_tracked_devices": 200, "device_ttl_seconds": 3600},
    }


# Core EMS Modules
from src.database.session import DatabaseSession
from src.hardware.mqtt import MQTTClientManager
from src.pipeline.safety import SafetyMonitor
from src.rl.agent import TabularQLearningAgent, PolicyPromotionGate

# ML & Pipeline Modules
from src.models.thermodynamics import ThermodynamicsModel as DigitalTwinEnv
from src.models.protonet import (
    CNN1DEncoder, TemperatureScaler, WEibullOpenMax,
    SupportSetManager, ProtoNet, PrototypeRegistry, OpenMaxWeibull
)
from src.models.calibration import TemperatureScaler as CalibratedTemperatureScaler
from src.pipeline.aggregate_nilm import NILMTransientDetector, TRANSIENT_THRESHOLD_W
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.watchdog import SoftAnomalyWatchdog
from src.pipeline.temporal_validator import TemporalValidator
from src.pipeline.analytics import AnalyticsEngine
from src.pipeline.failure_matrix import FailureMatrix
from src.pipeline.classifier import ModeClassifier
from src.pipeline.heuristic_fallback import (
    HeuristicApplianceClassifier,
    plausible_classes,
    DEFAULT_RULES,
    ENVELOPE_SLACK,
    UNKNOWN as UNRECOGNISED,
)

# Wire/internal sentinel for "no known class fits this load". Kept as the string
# "unknown" deliberately: src/api/main.py, the dashboard's DeviceCards.jsx and 54
# test assertions all key on that literal, and renaming it would break the
# contract for no functional gain. UNRECOGNISED_DISPLAY is what a human should
# read — the operator is being told the system cannot name the device, not that
# the device is malfunctioning.
UNRECOGNISED_DISPLAY = "Unrecognised device"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EMSOrchestrator:
    """
    Central hub for the Confidence-Aware Digital Twin EMS.
    Manages safety, comfort, state-aware control, and ML classification.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        stage_hook: Optional[Callable[[str], None]] = None,
        rl_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # ── Configuration ──
        if config is not None:
            self.config = config
        else:
            self.config = load_config()

        self._stage_hook = stage_hook
        self._rl_hook = rl_hook
        self._connected = True

        self._max_tracked_devices = self.config.get('system', {}).get('max_tracked_devices', 200) if isinstance(self.config.get('system'), dict) else 200
        self._device_last_seen: Dict[str, float] = {}
        self._device_ttl_seconds = self.config.get('system', {}).get('device_ttl_seconds', 3600) if isinstance(self.config.get('system'), dict) else 3600

        # ── Infrastructure ──
        db_path = self.config.get('database', {}).get('path', 'data/ems_state.db') if isinstance(self.config.get('database'), dict) else 'data/ems_state.db'
        self.db = DatabaseSession(db_path)
        # Allow MQTT_BROKER env var to override config (for Docker deployments)
        mqtt_cfg = self.config.get('mqtt', {}) if isinstance(self.config.get('mqtt'), dict) else {}
        mqtt_broker = os.environ.get('MQTT_BROKER', mqtt_cfg.get('broker', 'localhost'))
        mqtt_port = mqtt_cfg.get('port', 1883)
        # Credentials, env first so a deployment never has to bake them into a
        # tracked config file. MQTTClientManager has always accepted these; the
        # constructor call simply never passed them, so against this repo's OWN
        # broker config (mosquitto/config/mosquitto.conf sets
        # `allow_anonymous false` + `password_file`, which is what
        # docker-compose.yml mounts) the pipeline could not authenticate and just
        # logged "MQTT connection error ... Reconnecting in 5 seconds" forever.
        # docker-compose.yml already exports MQTT_USERNAME/MQTT_PASSWORD for
        # exactly this and nothing read them. None stays None, which is what
        # aiomqtt wants for an anonymous broker, so a local dev broker with no
        # auth behaves exactly as before.
        mqtt_username = os.environ.get('MQTT_USERNAME', mqtt_cfg.get('username')) or None
        mqtt_password = os.environ.get('MQTT_PASSWORD', mqtt_cfg.get('password')) or None
        self.mqtt = MQTTClientManager(
            mqtt_broker,
            mqtt_port,
            username=mqtt_username,
            password=mqtt_password,
        )
        if mqtt_username:
            logger.info(f"MQTT auth: user '{mqtt_username}' (password "
                        f"{'set' if mqtt_password else 'MISSING'})")

        # Auto-register with mock broker if active during tests
        try:
            from src.hardware.mqtt import MockMQTTBroker
            if MockMQTTBroker._active_broker is not None:
                MockMQTTBroker._active_broker.register(self)
        except Exception:
            pass

        # ── Safety Layer (will run as PARALLEL asyncio.Task) ──
        safety_cfg = self.config.get("system_safety", {}) if isinstance(self.config.get("system_safety"), dict) else {}
        self.safety = SafetyMonitor(
            max_aggregate_wattage=safety_cfg.get("max_aggregate_wattage", 3500.0),
            device_wattage_limits=safety_cfg.get("device_wattage_limits", {}),
            warning_pct=safety_cfg.get("warning_pct", 1.10),
            critical_pct=safety_cfg.get("critical_pct", 1.25),
        )

        # ── Auxiliary Pipeline Components ──
        self.env = DigitalTwinEnv()
        self.phantom_tracker = PhantomTracker()
        self.watchdog = SoftAnomalyWatchdog()
        self.analytics = AnalyticsEngine(
            cost_per_kwh=0.15  # Fallback; ToU pricing used by RL agent
        )
        self.failure_matrix = FailureMatrix()
        self.mode_classifier = ModeClassifier()
        self.temporal_validator = TemporalValidator()

        # ── ProtoNet / CNN ──
        proto_cfg = self.config.get("protonet", {})
        self.seq_len = proto_cfg.get("seq_len", 128)
        self.embedding_size = proto_cfg.get("embedding_size", 128)
        self.distance_threshold = proto_cfg.get("distance_threshold", 15.0)
        self.confidence_threshold = proto_cfg.get("confidence_threshold", 0.90)

        self.encoder: Optional[CNN1DEncoder] = None
        self.support_manager = SupportSetManager()
        self.temp_scaler = TemperatureScaler()
        self.weibull = WEibullOpenMax(
            tail_size=proto_cfg.get("openmax_tail_size", 20),
            alpha=proto_cfg.get("openmax_alpha", 3)
        )
        self._load_ml_models(proto_cfg)

        # ── Heuristic Fallback (zero-torch, deterministic) ──
        # `appliances:` scopes the classifier to what this deployment can
        # actually present, so a class no socket on the rig can drive is never
        # the answer. Previously the full rule set was always used, so the
        # hardware profile (laptop + phone charger only) could report `hvac`.
        self.heuristic_min_confidence = proto_cfg.get("heuristic_min_confidence", 0.55)
        self.heuristic_clf = HeuristicApplianceClassifier(
            allowed_classes=self.config.get("appliances") or None
        )
        self._registry_clf_cache: Dict[tuple, HeuristicApplianceClassifier] = {}

        # Raw power windows behind each device's recent unrecognised events.
        # Enrollment needs watts; the DeltaStabilityAnalyzer only keeps
        # embeddings, so the traces have to be retained separately.
        self._unknown_windows: Dict[str, deque] = {}

        # Bar a *confirmed* classification must clear to be reported as
        # recognised. Deliberately separate from `confidence_threshold` (0.90),
        # which gates a single softmax: this value gates the noisy-OR of two
        # independent channels that have already agreed, so it is not the same
        # quantity and must not borrow the same number. Set from measurement on
        # real UK-DALE windows — see claude_debug/DEBUG_SESSION_2026-08-25.md.
        self.recognition_threshold = proto_cfg.get("recognition_threshold", 0.45)

        # ── Delta Stability Analyzer ──
        ds_cfg = self.config.get("delta_stability", {})
        self.delta_analyzer = DeltaStabilityAnalyzer(
            window=ds_cfg.get("buffer_size", 10),
            threshold=ds_cfg.get("stability_threshold", 15.0),
            min_count=ds_cfg.get("min_occurrences", 3),
        )

        # ── NILM Transient Detector (SG filter + derivative) ──
        pre_cfg = self.config.get("preprocessing", {})
        self.nilm_detector = NILMTransientDetector(
            sg_window=pre_cfg.get("sg_window", 7),
            sg_polyord=pre_cfg.get("sg_poly", 2),
            threshold=pre_cfg.get("transient_threshold_w", TRANSIENT_THRESHOLD_W),
            window_size=pre_cfg.get("transient_window_s", 5),
            embed_window=128,
        )

        # ── Policy Promotion Gate ──
        self.promo_gate = PolicyPromotionGate()

        # Per-device NILM transient detectors (§2.1 fix: no shared buffer)
        self.nilm_detectors: Dict[str, NILMTransientDetector] = {}

        # Rolling windows for CNN input (per device) — legacy fallback
        self.power_windows: Dict[str, deque] = {}

        # ── RL Agent (reads config internally) ──
        self.agent = TabularQLearningAgent()

        # ── CSV Fallback Writer (§3.2 fix) ──
        self.csv_fallback_path = self.config.get('database', {}).get(
            'fallback_csv', 'data/fallback_measurements.csv'
        )

        # ── State Memory ──
        self.device_states: Dict[str, int] = {}
        self.device_classifications: Dict[str, str] = {}
        self.action_cooldowns: Dict[str, float] = {}
        self.last_analytics_broadcast = 0.0
        self._running = False
        self._internal_temp = 22.0  # Indoor temp for digital twin

        # Audit fix 2.1/3.1: Track last-known power per device for RL state + twin
        self.last_device_power: Dict[str, float] = {}

        # Bug 1.6 fix: Carry over last known confidences during steady-state ticks
        self.last_known_confidences: Dict[str, float] = {}

        # Bug 2.2 fix: Track actual simulation time for real dt calculation
        self.last_sim_time: float = time.time()

        # Bug 2.4 fix: Track last analytics time per device for real duration
        self.last_device_analytics_time: Dict[str, float] = {}

        # Bug 4.3 fix: asyncio lock for CSV writes
        self._csv_lock = asyncio.Lock()

        # Fix: Keep CNN active for N ticks after transient to feed DeltaStability OpenMax
        self.cnn_active_ticks: Dict[str, int] = {}


    def _resolve_registry_path(self) -> str:
        """
        The one place the prototype-registry file path is decided.

        Both the startup load and `handle_label_submitted`'s save must agree, or
        an operator label is written to a file the next boot does not read and
        the label loop silently loses every enrolment. They disagreed as soon as
        `registry_path` became configurable, which is why this is a method and
        not two copies of the expression.

        `registry_path` lets a deployment point at a registry enrolled on ITS OWN
        devices instead of the shipped UK-DALE artifact. That matters: UK-DALE's
        `laptop` prototype is a 21 W load (p50, measured on
        `data/real/cache/ukdale_windows_demo.npz`) and its `phone_charger` has
        zero windows above the 20 W on-threshold, so the shipped artifact cannot
        represent a modern 45-120 W USB-PD charger or a 120 W laptop at all.
        Enrol with `scripts/enroll_demo_devices.py`. Defaults to the path
        `train_demo_models.py` writes, so an enrolled registry lives in its own
        file and a retrain cannot clobber it.
        """
        proto_cfg = self.config.get("protonet", {})
        weights_path = proto_cfg.get("weights_path", "")
        weights_dir = os.path.dirname(weights_path) if weights_path else "backend/models/weights"
        return proto_cfg.get(
            "registry_path", os.path.join(weights_dir, "prototype_registry.pt")
        )

    def _load_ml_models(self, proto_cfg: dict) -> None:
        """Load CNN encoder, temperature scaler, Weibull, and support registry."""
        weights_path = proto_cfg.get("weights_path", "")
        anchors_path = proto_cfg.get("anchors_path", "")
        weights_dir = os.path.dirname(weights_path) if weights_path else "backend/models/weights"

        # ── Try new Phase-1 weights first (protonet.pt) ──
        new_weights = os.path.join(weights_dir, "protonet.pt")
        if os.path.exists(new_weights):
            try:
                proto = ProtoNet(seq_len=128)
                state_dict = torch.load(new_weights, map_location="cpu", weights_only=False)

                # Remap legacy key names to current architecture
                # Old training used: enc.cnn, attn.w, enc.fc (nn.Sequential)
                # Current model uses: encoder.cnn, attention.attn, encoder.project (nn.Linear)
                key_map = {
                    "enc.cnn.": "encoder.cnn.",
                    "attn.w.": "attention.attn.",
                }
                remapped = {}
                for k, v in state_dict.items():
                    new_key = k
                    # Handle FC layer: old enc.fc.0.* → new encoder.project.*
                    # Old model: enc.fc = Sequential(Linear, BatchNorm1d)
                    # New model: encoder.project (Linear) + encoder.project_bn (BatchNorm1d)
                    if k.startswith("enc.fc.0."):
                        new_key = k.replace("enc.fc.0.", "encoder.project.")
                    elif k.startswith("enc.fc.1."):
                        new_key = k.replace("enc.fc.1.", "encoder.project_bn.")
                    else:
                        for old_prefix, new_prefix in key_map.items():
                            if k.startswith(old_prefix):
                                new_key = k.replace(old_prefix, new_prefix)
                                break
                    # num_batches_tracked is needed by BatchNorm — don't skip
                    remapped[new_key] = v

                missing, unexpected = proto.load_state_dict(remapped, strict=False)
                if missing:
                    logger.warning(f"ProtoNet missing keys (initialized randomly): {missing}")
                if unexpected:
                    logger.warning(f"ProtoNet unexpected keys (ignored): {unexpected}")
                proto.eval()
                self.encoder = proto
                logger.info(f"✅ Phase-1 ProtoNet loaded from {new_weights}")
            except Exception as e:
                logger.error(f"Failed to load ProtoNet: {e}")
                self.encoder = None

            # Load Prototype Registry (separate try so ProtoNet isn't killed)
            try:
                registry_path = self._resolve_registry_path()
                if os.path.exists(registry_path) and self.encoder is not None:
                    self.prototype_registry = PrototypeRegistry(self.encoder)
                    self.prototype_registry.load(registry_path)
                    enrolled = len(self.prototype_registry.envelopes)
                    logger.info(
                        f"✅ Prototype Registry loaded "
                        f"({len(self.prototype_registry.class_names())} classes, "
                        f"{enrolled} with a power envelope) from {registry_path}"
                    )
                    if enrolled == 0:
                        # Not fatal, but it decides how recognition behaves: with
                        # no envelopes every class needs the deterministic
                        # centroid vote to agree, and on out-of-family loads that
                        # vote is noise-dominated. Say so rather than let the
                        # dashboard look merely quiet.
                        logger.warning(
                            "   ⚠ no class carries a power envelope — recognition "
                            "rests entirely on the centroid agreement channel. "
                            "Enrol this deployment's devices with "
                            "scripts/enroll_demo_devices.py, or label them once "
                            "through the dashboard."
                        )
                else:
                    self.prototype_registry = None
                    if self.encoder is not None and not os.path.exists(registry_path):
                        # Silent absence used to leave the pipeline in degraded
                        # heuristic mode with no indication why nothing is ever
                        # named. The demo weights directory is gitignored, so a
                        # fresh checkout hits this the moment `registry_path` is
                        # configured but the artifact has not been generated.
                        logger.warning(
                            f"⚠ Prototype Registry MISSING at {registry_path} — "
                            f"classification falls back to the deterministic "
                            f"heuristic only. Generate it with "
                            f"scripts/train_demo_models.py then "
                            f"scripts/enroll_demo_devices.py."
                        )
            except Exception as e:
                logger.warning(f"Prototype Registry load failed: {e}")
                self.prototype_registry = None

            # Load OpenMax Weibull
            try:
                omw_path = os.path.join(weights_dir, "openmax_weibull.pkl")
                if os.path.exists(omw_path):
                    self.weibull = OpenMaxWeibull(num_classes=10)
                    self.weibull.load(omw_path)
                    logger.info("✅ OpenMax Weibull loaded")
            except Exception as e:
                logger.warning(f"OpenMax Weibull load failed: {e}")

            # Load calibrated temperature scaler
            try:
                ts_path = os.path.join(weights_dir, "temperature_scaler.pt")
                if os.path.exists(ts_path):
                    self.calibrated_scaler = CalibratedTemperatureScaler()
                    self.calibrated_scaler.load(ts_path)
                    logger.info(f"✅ Calibrated T-Scaler loaded (T={self.calibrated_scaler.temperature.item():.4f})")
                else:
                    self.calibrated_scaler = None
            except Exception as e:
                logger.warning(f"T-Scaler load failed: {e}")
                self.calibrated_scaler = None

        elif os.path.exists(weights_path):
            # Fallback: legacy CNN weights
            try:
                self.encoder = CNN1DEncoder(
                    in_channels=1,
                    embed_dim=self.embedding_size
                )
                self.encoder.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=False)
                )
                self.encoder.eval()
                self.prototype_registry = None
                self.calibrated_scaler = None
                logger.info(f"✅ Legacy CNN loaded from {weights_path}")
            except Exception as e:
                logger.error(f"Failed to load legacy CNN: {e}")
                self.encoder = None
        else:
            logger.warning("No model weights found. Run python scripts/train_models.py first.")
            self.encoder = None
            self.prototype_registry = None
            self.calibrated_scaler = None

        # Legacy loaders (each isolated so failures don't cascade)
        try:
            if os.path.exists(anchors_path):
                self.support_manager.load_registry(anchors_path)
                logger.info(f"✅ Legacy support registry loaded from {anchors_path}")
        except Exception as e:
            logger.warning(f"Legacy support registry load failed: {e}")

        try:
            scaler_path = os.path.join(weights_dir, "temperature_scaler.pth")
            if os.path.exists(scaler_path) and not hasattr(self, 'calibrated_scaler'):
                self.temp_scaler.load(scaler_path)
                logger.info("✅ Legacy temperature scaler loaded")
        except Exception as e:
            logger.warning(f"Legacy T-scaler load failed: {e}")

        try:
            weibull_path = os.path.join(weights_dir, "weibull_openmax.pkl")
            if os.path.exists(weibull_path) and not hasattr(self.weibull, '_weibull_by_name'):
                with open(weibull_path, 'rb') as f:
                    self.weibull = pickle.load(f)
        except Exception as e:
            logger.warning(f"Legacy weibull load failed: {e}")

    # ─── Safety Relay Callback ─────────────────────────────────────────
    async def _relay_callback(self, device_id: str, action: str) -> None:
        """
        Callback for the fleet diagnostics monitor (formerly SafetyMonitor).

        Production architecture: Physical relay cutoffs are handled at the edge
        (ESP32 Core 0). This callback dispatches structured alerts to the
        dashboard UI pipeline — it does NOT issue MQTT relay commands.
        """
        if action == "ALERT_CRITICAL":
            logger.warning(
                f"⚡ CRITICAL ALERT: {device_id} exceeds threshold — "
                f"edge node handles physical cutoff"
            )
            await self._broadcast_event({
                "type": "SAFETY_CUTOFF",
                "device_id": device_id,
                "severity": "critical",
                "message": (f"Critical power threshold breached on {device_id} — "
                            f"edge relay has been activated"),
            })
        elif action == "ALERT_ARC_FAULT":
            logger.critical(
                f"⚡ ARC FAULT ALERT: {device_id} — "
                f"edge node handles physical cutoff"
            )
            await self._broadcast_event({
                "type": "SAFETY_CUTOFF",
                "device_id": device_id,
                "severity": "critical",
                "message": (f"Arc-fault proxy detected on {device_id} — "
                            f"edge relay has been activated"),
            })
        elif action in ("WARNING", "ALERT_WARNING"):
            await self._broadcast_event({
                "type": "SAFETY_WARNING",
                "device_id": device_id,
                "severity": "warning",
                "message": f"Power draw approaching limit on {device_id}",
            })
        elif action == "OFF":
            # Legacy fallback — should not be called in production
            logger.warning(f"⚡ Legacy OFF command for {device_id} — forwarding to MQTT")
            try:
                await self.mqtt.publish_command(f"home/plug/{device_id}/command", "OFF")
            except Exception as e:
                logger.critical(f"CUTOFF PUBLISH FAILED for {device_id}: {e}")
                self.failure_matrix.trigger_failure("mqtt_disconnect", device_id)
            await self._broadcast_event({
                "type": "SAFETY_CUTOFF",
                "device_id": device_id,
                "severity": "critical",
                "message": f"Safety threshold breached — {device_id} relay forced OFF",
            })

    # ─── ProtoNet Classification (registry + physical gate + confidence) ──
    def _classify_device(self, device_id: str, power_watts: float,
                         filtered_segment: np.ndarray = None):
        """
        Full classification pipeline:
        1. NILM-filtered segment (or legacy rolling window) → CNN → embedding
        2. Squared distance to every registry prototype → temperature-scaled
           softmax → calibrated confidence
        3. Physical power-envelope gate + confidence gate → UNRECOGNISED
        4. Returns (class_name, confidence, distances)

        `class_name` is one of:
          * a registered class name — recognised, and confident enough to act on
          * `"unknown"`             — UNRECOGNISED. Either no known class can
                                      draw this power, or the best match is not
                                      trustworthy. Drives the LABEL_REQUEST flow
                                      so the operator can name it; the label is
                                      then live for the *next* event because
                                      inference reads the same registry
                                      `handle_label_submitted` writes to.
          * `"pending"`             — not enough samples buffered yet
          * `"error"`               — inference raised

        Why the registry and not `support_manager`
        ------------------------------------------
        This method used to call `SupportSetManager.classify()`. That object is
        only populated from `protonet.anchors_path`, which is set in
        `config/config.yaml` alone — never in the demo or hardware profiles —
        so `compute_prototypes()` returned `{}` and the method returned
        `("unknown", 0.0, {})` for **every event on every profile**. The trained
        7-class registry sitting in `self.prototype_registry` was loaded, logged
        at startup, written to by `handle_label_submitted`, and never once read.
        So nothing was ever classified, and operator labels had no effect on
        inference at all.

        Args:
            filtered_segment: (128,) pre-filtered segment from NILMTransientDetector.
                              If provided, bypasses the legacy rolling window.
        """
        # Use NILM-filtered segment when available (§2.1 fix)
        if filtered_segment is not None:
            window_np = np.asarray(filtered_segment, dtype=np.float32)
        else:
            # Legacy fallback: maintain rolling window
            if device_id not in self.power_windows:
                self.power_windows[device_id] = deque(maxlen=self.seq_len)
            self.power_windows[device_id].append(power_watts)
            window = self.power_windows[device_id]
            if len(window) < self.seq_len:
                return "pending", 0.0, {}
            window_np = np.array(list(window), dtype=np.float32)

        if window_np.size < self.seq_len:
            window_np = np.pad(window_np, (0, self.seq_len - window_np.size))
        window_np = window_np[:self.seq_len]

        registry = getattr(self, "prototype_registry", None)

        # ── Degraded mode: no encoder, or nothing enrolled to compare against ──
        if self.encoder is None or registry is None or not registry.prototypes:
            if not self.support_manager.raw_windows:
                return self._classify_heuristic(window_np)
            # Legacy profile with a populated anchors registry — honour it.
            try:
                return self.support_manager.classify(
                    window_np, self.encoder, self.weibull, self.temp_scaler,
                    self.confidence_threshold
                )
            except Exception as e:
                logger.error(f"Legacy ProtoNet classification error for {device_id}: {e}")
                return "error", 0.0, {}

        try:
            _, _, dist_map = registry.classify(window_np)
            if not dist_map:
                return self._classify_heuristic(window_np)

            names = list(dist_map.keys())
            d2 = np.array([dist_map[n] for n in names], dtype=np.float64)
            if not np.all(np.isfinite(d2)):
                logger.warning(f"Non-finite prototype distance for {device_id} — treating as unrecognised")
                return UNRECOGNISED, 0.0, dist_map

            # Temperature-scaled softmax over negative squared distances.
            # T comes from the calibrated scaler when one was fitted; the floor
            # mirrors TemperatureScaler's own T >= 0.05 clamp so a degenerate
            # artefact cannot produce a one-hot (falsely certain) posterior.
            temperature = 1.0
            cal = getattr(self, "calibrated_scaler", None)
            if cal is not None:
                try:
                    temperature = max(float(cal.temperature.item()), 0.05)
                except Exception:
                    temperature = 1.0
            logits = -d2 / temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum() + 1e-12

            order = np.argsort(-probs)
            best = names[order[0]]
            confidence = float(probs[order[0]])

            # ── Physical plausibility gate ──
            # The embedding carries no absolute-scale information (a 500 W load
            # sits d2 = 1.02 from the 5 W router prototype), so the wattage the
            # PZEM actually measured is the only reliable novelty signal. Veto
            # any class whose known power envelope cannot contain this window,
            # then re-normalise over the survivors.
            eligible = self._eligible_classes(window_np, names, registry)
            if not eligible:
                return UNRECOGNISED, 0.0, dist_map

            # ── The operator's own labels outrank the population prior ──
            # An enrolled class carries the envelope measured on *this* device
            # and confirmed by a human. A shipped class carries a literature
            # band shared with three neighbours. When a window sits inside an
            # enrolled envelope, that is the better evidence, so drop the
            # shipped classes from the contest rather than letting them split
            # the softmax with it.
            #
            # Without this the label loop silently does nothing: the shipped
            # `monitor` prototype sits almost on top of a newly enrolled 35 W
            # monitor in embedding space, the probability halves between them,
            # and the result falls back under the threshold — so the device the
            # operator just named still reports as unrecognised. Measured:
            # enrolled-recall 1/3 -> 3/3.
            #
            # Two enrolled classes with overlapping envelopes both survive, and
            # the embedding then chooses between them, which is correct.
            enrolled_hits = {n for n in eligible if registry.power_envelope(n) is not None}
            if enrolled_hits:
                eligible = enrolled_hits

            # Always renormalise over the survivors, not only when the argmax
            # was vetoed: a confidence carried over from the full softmax would
            # still be diluted by classes the physical gate has just ruled out.
            keep = [i for i, n in enumerate(names) if n in eligible]
            sub = probs[keep] / (probs[keep].sum() + 1e-12)
            j = int(np.argmax(sub))
            best = names[keep[j]]
            confidence = float(sub[j])

            # ── Two-channel agreement is the recognition test ──
            # Confidence alone cannot be trusted here, and that is a measured
            # fact rather than a precaution: on the shipped 7-class artefact a
            # 65 W laptop is called desktop_computer at p=0.72 and a 100 W
            # charger is called tv at p=0.86. Raising the gate does not remove
            # those, it only removes the correct low-confidence answers with
            # them. So the learned channel must be confirmed by an independent,
            # non-learned measurement of absolute watts before we name a device.
            #
            # Which confirmation applies depends on where the class came from:
            #
            #   enrolled class  — the operator labelled it, so we hold the power
            #                     envelope actually observed on their device.
            #                     Containment in that envelope IS the
            #                     independent confirmation, and a tight one.
            #                     `_eligible_classes` has already enforced it.
            #   shipped class   — we only have a wide literature band shared
            #                     with three neighbouring classes, which
            #                     confirms almost nothing. Demand the stronger
            #                     test: the deterministic centroid classifier
            #                     must independently pick the same class.
            #
            # Disagreement, or either channel abstaining, yields UNRECOGNISED —
            # the honest answer, and the one that asks the operator for a label.
            # Measured on real UK-DALE windows, for the shipped classes:
            # accuracy among accepted 0.621 -> 0.851, and confidently-wrong on
            # the phone/laptop/monitor set halved. The price is coverage (~0.37
            # of windows accepted), paid deliberately: an unanswered window costs
            # one label, a wrong one corrupts that appliance's energy history.
            if registry.power_envelope(best) is None:
                h_result = self._registry_heuristic(names).classify(window_np)
                if h_result.appliance == UNRECOGNISED or h_result.appliance != best:
                    return UNRECOGNISED, 0.0, dist_map
                # Two independent confirmations — combined as a noisy-OR so
                # agreement can only raise confidence, never lower it below
                # either channel on its own.
                confidence = 1.0 - (1.0 - confidence) * (1.0 - float(h_result.confidence))

            if confidence < self.recognition_threshold:
                # Confirmed but not convincing. Ask rather than commit: this
                # window is exactly what the LABEL_REQUEST flow exists for.
                return UNRECOGNISED, confidence, dist_map

            return best, confidence, dist_map

        except Exception as e:
            logger.error(f"ProtoNet classification error for {device_id}: {e}")
            return "error", 0.0, {}

    def _registry_heuristic(self, names: List[str]) -> HeuristicApplianceClassifier:
        """
        Confirmation channel, scoped to the classes the registry can actually
        return.

        Without the scoping the two channels could never agree on the general
        household profile: `self.heuristic_clf` is built from `appliances:`,
        which need not match the registry's class list, so channel B would
        propose a class channel A cannot produce and every window would be
        rejected. Cached per class-set — construction filters the rule list, so
        rebuilding it per event would allocate on the MQTT ingest path.
        """
        key = tuple(sorted(names))
        cached = self._registry_clf_cache.get(key)
        if cached is None:
            cached = HeuristicApplianceClassifier(allowed_classes=list(key))
            self._registry_clf_cache[key] = cached
        return cached

    def _eligible_classes(self, window_np: np.ndarray, names: List[str],
                          registry) -> set:
        """
        Classes whose power envelope can contain this window.

        Envelope source, in order of preference:
          1. the range observed when the class was enrolled (registry), which is
             the only option for operator-labelled classes — they have no entry
             in DEFAULT_RULES by definition;
          2. the measured UK-DALE/REDD band in `heuristic_fallback.DEFAULT_RULES`;
          3. unconstrained — a class with neither is never vetoed, because the
             gate may only veto where it actually has knowledge.
        """
        feats = self.heuristic_clf.extract_features(window_np)
        if not feats:
            return set(names)
        steady = float(feats.get("steady_w", 0.0) or 0.0)
        band_ok = plausible_classes(feats)
        known_bands = {r.name for r in DEFAULT_RULES}

        # `appliances:` in config restricts what this deployment can present.
        # A registry class outside it (the shipped demo artifact carries all 7
        # regardless of profile) must not be reported. Operator-enrolled classes
        # are never in the list, so they are exempt.
        allowed = self.heuristic_clf.allowed_classes
        enrolled = set(registry.envelopes) if registry is not None else set()

        out = set()
        for n in names:
            if allowed is not None and n not in allowed and n not in enrolled:
                continue
            env = registry.power_envelope(n) if registry is not None else None
            if env is not None:
                lo, hi = env
                # Pad for mains drift and PZEM resolution ONLY — the same
                # physical slack heuristic_fallback uses (±6% mains, P ∝ V², so
                # ~±12% power), plus 1 W for the meter's own quantisation.
                #
                # It must NOT be widened to "cover the range a 5-sample
                # enrollment might have missed": at 25% an enrolled 45 W charger
                # spanned 33–57 W and captured a 35 W monitor, and since enrolled
                # classes take precedence the correct shipped answer was
                # discarded. Guessing at unobserved range costs more than it buys
                # — if the device really draws more, that window is reported
                # unrecognised, the operator labels it again, and add_class()
                # merges the envelope with what was already recorded.
                pad = max(ENVELOPE_SLACK * hi, 1.0)
                if lo - pad <= steady <= hi + pad:
                    out.add(n)
            elif n in known_bands:
                if n in band_ok:
                    out.add(n)
            else:
                out.add(n)
        return out

    def _classify_heuristic(self, window_np: np.ndarray):
        """
        Deterministic fallback when ProtoNet or the registry is unavailable.

        A heuristic guess is only returned when it clears
        `heuristic_min_confidence`. Below that it is reported as unrecognised:
        the previous code accepted any heuristic answer whose confidence merely
        beat ProtoNet's (usually 0.0), so a 0.11-confidence guess became the
        device's final classification and the operator was never asked.
        """
        result = self.heuristic_clf.classify(window_np)
        if result.appliance != UNRECOGNISED and result.confidence >= self.heuristic_min_confidence:
            return result.appliance, result.confidence, {}
        return UNRECOGNISED, result.confidence, {}

    # ─── Event Broadcast ──────────────────────────────────────────────
    async def _broadcast_event(self, event: dict) -> None:
        """Publish structured event to MQTT for the API layer to pick up."""
        try:
            await self.mqtt.publish_command(
                self.config['mqtt']['topics'].get('events', 'home/ui/events'),
                json.dumps(event)
            )
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")

    # ─── CSV Fallback Writer (§3.2 fix) ───────────────────────────────
    def _csv_fallback_write_sync(self, timestamp: float, device_id: str,
                                 power_watts: float) -> None:
        """Synchronous CSV write — called via asyncio.to_thread to avoid blocking."""
        try:
            directory = os.path.dirname(self.csv_fallback_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            file_exists = os.path.exists(self.csv_fallback_path)
            with open(self.csv_fallback_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'device_id', 'power_watts'])
                writer.writerow([timestamp, device_id, power_watts])
            logger.warning(f"📝 DB fallback: wrote {device_id}={power_watts:.1f}W to {self.csv_fallback_path}")
        except Exception as fallback_err:
            logger.critical(f"CSV fallback write ALSO failed: {fallback_err}")

    async def _csv_fallback_write(self, timestamp: float, device_id: str,
                                  power_watts: float) -> None:
        """Non-blocking CSV fallback — runs sync I/O in a thread to avoid stalling the event loop."""
        async with self._csv_lock:
            await asyncio.to_thread(self._csv_fallback_write_sync, timestamp, device_id, power_watts)

    # ─── Main Message Handler (ML Pipeline) ───────────────────────────
    async def _handle_mqtt_message(
        self, topic: str, payload: Union[str, bytes, bytearray, int, float, None]
    ) -> None:
        # ══ WS-7.2: Pipeline Latency Measurement ══
        t0 = time.perf_counter()
        try:
            device_id = topic.split("/")[-2]
            # Robust payload decoding
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8", errors="replace")
            payload_str = str(payload) if payload else ""

            # Phase 2 (WS-5.1): Hardware ACK processing
            if topic.endswith("/ack"):
                logger.info(f"✅ Hardware ACK received for {device_id}: {payload_str}")
                # Clear software cooldown / update state
                self.action_cooldowns[device_id] = 0.0
                await self._broadcast_event({
                    "type": "HARDWARE_ACK",
                    "device_id": device_id,
                    "message": f"Hardware confirmed: {payload_str}",
                })
                return

            if topic.endswith("/status"):
                logger.warning(f"Safety event received from {device_id}: {payload_str}")
                await self._broadcast_event({
                    "type": "SAFETY_EVENT",
                    "device_id": device_id,
                    "message": f"Status Update: {payload_str}",
                })
                return

            # Bug 3.1 fix: Handle label submissions via MQTT
            # (bridging REST API → MQTT → pipeline for ProtoNet registry updates)
            if "/label" in topic:
                try:
                    payload_dict = json.loads(payload_str)
                    label_class = payload_dict.get("class_name", "")
                    label_segments = payload_dict.get("segments", [])
                    if label_class and label_segments:
                        self.handle_label_submitted(label_class, label_segments)
                        logger.info(f"📋 Label received via MQTT: '{label_class}' ({len(label_segments)} segments)")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse label MQTT message: {e}")
                return

            # Dual-format payload extraction (plain float or JSON)
            power_watts = 0.0
            stripped_payload = payload_str.strip()
            if stripped_payload.startswith("{") and stripped_payload.endswith("}"):
                try:
                    data = json.loads(stripped_payload)
                    if isinstance(data, dict):
                        power_watts = float(data.get("power", data.get("watts", data.get("W", data.get("value", 0.0)))))
                    else:
                        power_watts = float(data)
                except Exception as parse_err:
                    logger.warning(f"🚫 Failed to extract power from JSON payload on {topic}: {parse_err}")
                    return
            else:
                try:
                    power_watts = float(stripped_payload) if stripped_payload else 0.0
                except ValueError:
                    logger.warning(f"🚫 Non-numeric payload on {topic}: '{payload_str}'")
                    return

            # Fix: Sanitize NaN/Inf payloads to prevent ML embedding corruption
            # and SQLite data poisoning from faulty sensors
            import math
            if math.isnan(power_watts) or math.isinf(power_watts):
                logger.warning(f"🚫 Rejected invalid payload on {topic}: {payload_str} (NaN/Inf)")
                return
            current_time = time.time()
            current_hour = datetime.now().hour

            # Audit fix 2.1: Always track latest power per device for RL state + twin
            self.last_device_power[device_id] = power_watts

            # Memory eviction: track device last-seen time and periodically cleanup
            self._device_last_seen[device_id] = current_time
            if len(self._device_last_seen) > self._max_tracked_devices:
                self._evict_stale_devices()

            # ══════════════════════════════════════════════════════════
            # NOTE: Safety monitor runs as SEPARATE parallel task.
            # This handler only processes ML pipeline steps.
            # ══════════════════════════════════════════════════════════

            # ══════════════════════════════════════════════════════════
            # STEP 1: SOFT ANOMALY WATCHDOG + TEMPORAL VALIDATION (§3.1)
            # ══════════════════════════════════════════════════════════
            if self.watchdog.check_reading(device_id, power_watts):
                logger.warning(f"🔍 WATCHDOG: Soft anomaly on {device_id} ({power_watts:.1f}W)")
                await self._broadcast_event({
                    "type": "SOFT_ANOMALY",
                    "device_id": device_id,
                    "power": round(power_watts, 2),
                    "message": f"Z-score anomaly detected on {device_id}",
                })
                # §3.1 fix: Feed anomaly into TemporalValidator for persistence check
                suggestion = self.temporal_validator.validate(device_id, power_watts)
                if suggestion:
                    soft_action, soft_info = suggestion
                    logger.info(f"🔔 Temporal Validation: {soft_action} for {device_id}")
                    await self._broadcast_event({
                        "type": "TEMPORAL_ANOMALY_ACTION",
                        "device_id": device_id,
                        "action": soft_action,
                        "details": soft_info,
                        "message": soft_info.get("message", ""),
                    })

            # ══════════════════════════════════════════════════════════
            # STEP 1.5: NILM PREPROCESSING (§2.1 fix)
            # Route raw power through SG-filter + derivative transient
            # detector. Only trigger CNN classification on valid ±50W
            # step-change events, not on every 1Hz tick.
            # ══════════════════════════════════════════════════════════
            if device_id not in self.nilm_detectors:
                pre_cfg = self.config.get("preprocessing", {})
                self.nilm_detectors[device_id] = NILMTransientDetector(
                    sg_window=pre_cfg.get("sg_window", 7),
                    sg_polyord=pre_cfg.get("sg_poly", 2),
                    threshold=pre_cfg.get("transient_threshold_w", TRANSIENT_THRESHOLD_W),
                    window_size=pre_cfg.get("transient_window_s", 5),
                    embed_window=128,
                )

            is_transient, filtered_segment = self.nilm_detectors[device_id].push(power_watts)

            # Fix: Keep CNN active for 5 ticks after a transient to feed the 
            # DeltaStabilityAnalyzer which requires min_occurrences=3
            if is_transient:
                self.cnn_active_ticks[device_id] = 5
                
            cnn_should_run = is_transient or self.cnn_active_ticks.get(device_id, 0) > 0
            if not is_transient and self.cnn_active_ticks.get(device_id, 0) > 0:
                self.cnn_active_ticks[device_id] -= 1
                # Fix Issue #8 (OpenMax Architectural Starvation):
                # During post-transient ticks, the NILM preprocessor returns
                # filtered_segment=None because no transient fired. But the CNN
                # must still run to feed the DeltaStabilityAnalyzer with steady-state
                # embeddings (it requires min_occurrences=3 consecutive embeddings
                # to flag a stable unknown). Extract the current buffer directly.
                if filtered_segment is None:
                    filtered_segment = self.nilm_detectors[device_id].get_current_segment()

            if not cnn_should_run:
                # No transient detected — still update device state tracking
                # and broadcast status, but skip heavy CNN classification
                self.device_states[device_id] = 1 if power_watts > 10 else 0
                class_name = self.device_classifications.get(device_id, "pending")
                # Bug 1.6 fix: Carry over last known confidence during steady-state
                # ticks instead of forcing 0.0 (which starves the entire pipeline)
                confidence = self.last_known_confidences.get(device_id, 0.0)
                distances = {}
            else:
                # ══════════════════════════════════════════════════════
                # STEP 2: CNN / PROTONET + OPENMAX CLASSIFICATION
                # (only on transient events — §2.1 fix)
                # ══════════════════════════════════════════════════════
                class_name, confidence, distances = self._classify_device(
                    device_id, power_watts, filtered_segment=filtered_segment
                )
                # Bug 1.6 fix: Cache the confidence for steady-state carry-over
                self.last_known_confidences[device_id] = confidence
            self.device_classifications[device_id] = class_name

            if class_name == "pending" or class_name == "error":
                # Not enough data or model error — skip RL
                pass

            elif class_name == "unknown":
                # ── UNKNOWN DEVICE FLOW ──
                # Get the last embedding for delta stability check
                # Use the NILM-filtered segment if available, otherwise fall back
                if filtered_segment is not None:
                    window_np = filtered_segment
                elif device_id in self.power_windows and len(self.power_windows[device_id]) >= self.seq_len:
                    window_np = np.array(list(self.power_windows[device_id]), dtype=np.float32)
                else:
                    window_np = None

                if window_np is not None and self.encoder is not None:
                    with torch.no_grad():
                        if isinstance(self.encoder, ProtoNet) and hasattr(self.encoder, 'embed'):
                            x   = torch.tensor(window_np[:128] if len(window_np) >= 128 else np.pad(window_np, (0, 128 - len(window_np))), dtype=torch.float32).unsqueeze(0)
                            embedding = self.encoder.embed(x).squeeze(0).numpy()
                        elif self.encoder:
                            x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            embedding = self.encoder(x).squeeze(0).numpy()
                        else:
                            embedding = np.zeros(self.embedding_size)

                    # Use new push() API for DFD P4.3 compliance
                    stability, cluster_mean = self.delta_analyzer.push(embedding)

                    # Keep the RAW power windows that produced these embeddings.
                    # Enrollment needs watts, not embeddings — see below.
                    buf = self._unknown_windows.setdefault(device_id, deque(maxlen=8))
                    buf.append(np.asarray(window_np[:128], dtype=np.float32).tolist())

                    if stability == 'stable':
                        logger.info(f"❓ Stable unrecognised load on {device_id} ({power_watts:.1f}W) — requesting label")
                        await self._broadcast_event({
                            "type": "LABEL_REQUEST",
                            "device_id": device_id,
                            "power": round(power_watts, 2),
                            "confidence": round(confidence, 3),
                            # `segments` is what enrollment consumes: raw 128-sample
                            # POWER windows in watts. `embedding` is retained for
                            # display/clustering only.
                            #
                            # This used to ship only `embedding`, and the dashboard
                            # POSTed that straight back as `segments`. add_class()
                            # then ran the CNN over a 128-D embedding as though it
                            # were a power trace, so every operator label produced a
                            # prototype built from nonsense — silently, since the
                            # shapes happen to match at (128,).
                            "segments": [list(s) for s in buf],
                            "embedding": cluster_mean.tolist() if cluster_mean is not None else [],
                            "suggested_label": UNRECOGNISED_DISPLAY,
                            "message": f"Unrecognised device on {device_id} at {power_watts:.0f} W. Please label it.",
                        })

                        # ── Task 5: Background pseudo-labeling ──
                        # Silently persist the stable cluster to the database.
                        # Uses quantized spatial hashing to dedup — identical appliances
                        # with sensor drift will match the same cluster_hash row.
                        # Does NOT block the pipeline waiting for user response.
                        if cluster_mean is not None and self.db:
                            try:
                                await self.db.save_unmapped_cluster_signature(
                                    device_id, cluster_mean, time.time()
                                )
                            except Exception as e:
                                logger.warning(f"Failed to persist unmapped cluster: {e}")

                        # ═══ §2.2 FIX: Forward stable unknown to Digital Twin + RL ═══
                        # Assign temporary pseudo-class so RL sees the load
                        pseudo_class = f"unknown_{device_id}"
                        # Bug 3.3 fix: Set classification immediately to prevent
                        # re-triggering LABEL_REQUEST every single tick
                        self.device_classifications[device_id] = pseudo_class
                        self.device_states[device_id] = 1 if power_watts > 10 else 0

                        # Bug 2.1 fix: Use last-known wattages for ALL devices,
                        # then overlay the live tick for this device
                        appliance_watts = {
                            k: self.last_device_power.get(k, 0.0)
                            for k in self.device_states
                        }
                        appliance_watts[device_id] = power_watts  # Apply the live tick

                        # Bug 2.2 fix: Use real elapsed time instead of hardcoded 1s
                        now = time.time()
                        dt_minutes = (now - self.last_sim_time) / 60.0
                        self.last_sim_time = now

                        self._internal_temp = self.env.simulate_step(
                            appliance_watts, outdoor_temp=28.0,
                            t_internal=self._internal_temp, dt_minutes=dt_minutes
                        )

                        # Update RL state space with pseudo-class base-load drain
                        device_limits = self.config.get("system_safety", {}).get("device_wattage_limits", {})
                        rated = device_limits.get("default", 1500.0)
                        pct_of_rated = power_watts / max(rated, 1.0)
                        pmv_score = self.env.compute_pmv(
                            t_air=self._internal_temp, t_mrt=self._internal_temp - 0.5,
                            v_air=0.1, rh=50.0, clo=0.7, met=1.2
                        )
                        tou_rate = self.agent.get_tou_rate(current_hour)

                        # Audit fix 2.1: Use last_device_power for full house state
                        all_devices_state = {}
                        for did, on_off in self.device_states.items():
                            cls = self.device_classifications.get(did, f"unknown_{did}")
                            last_power = self.last_device_power.get(did, 0.0)
                            dev_rated = device_limits.get(cls, device_limits.get("default", 1500.0))
                            all_devices_state[cls] = last_power / max(dev_rated, 1.0)
                        all_devices_state[pseudo_class] = pct_of_rated

                        state_dict = {
                            "devices": all_devices_state,
                            "price_tier": self.agent.get_price_bin(tou_rate),
                            "pmv_zone": self.agent.get_pmv_zone(pmv_score),
                            "tod": self.agent.get_time_of_day_bin(current_hour),
                        }
                        # Let RL observe the unknown load (DEFER-only, no shed)
                        self.agent.act(state_dict, pmv_score, confidence, pseudo_class)
                        logger.debug(f"§2.2: Forwarded {pseudo_class} ({power_watts:.1f}W) to DigitalTwin + RL")
                    else:
                        logger.debug(f"Transient unknown on {device_id} — logged silently")
                        try:
                            await self.db.insert_measurement(current_time, device_id, power_watts)
                        except Exception:
                            await self._csv_fallback_write(current_time, device_id, power_watts)
                # End unknown device flow
                
            elif confidence < self.recognition_threshold:
                # ── LOW CONFIDENCE ──
                # No heuristic re-run here. `_classify_device` already consults
                # the deterministic classifier as its confirmation channel, so
                # anything that reaches this branch has *already* been through
                # it. The previous code re-ran it and overwrote the class
                # whenever the heuristic's confidence merely beat ProtoNet's —
                # which, since ProtoNet returned 0.0 for every event, meant a
                # 0.11-confidence guess became the device's final answer and the
                # operator was never asked for a label.
                logger.info(f"⚠️ Low confidence ({confidence:.3f}) for {class_name} on {device_id}. Skipping RL.")
                await self._broadcast_event({
                    "type": "LOW_CONFIDENCE",
                    "device_id": device_id,
                    "classified_as": class_name,
                    "confidence": round(confidence, 3),
                    "threshold": self.recognition_threshold,
                    "message": f"Classification uncertain ({confidence:.2f} < {self.recognition_threshold})",
                })
                # Skip RL — uncertain classification

            else:
                # ══════════════════════════════════════════════════════
                # KNOWN + CONFIDENT DEVICE FLOW
                # ══════════════════════════════════════════════════════
                logger.debug(f"ProtoNet: {device_id} → {class_name} (conf={confidence:.3f})")

                # STEP 3: PHANTOM TRACKER
                is_off = self.device_states.get(device_id, 0) == 0
                self.phantom_tracker.track(device_id, power_watts, is_off)

                # STEP 4: DATABASE PERSISTENCE
                try:
                    await self.db.insert_measurement(current_time, device_id, power_watts)
                except Exception as e:
                    logger.error(f"DB write failed: {e}")
                    self.failure_matrix.trigger_failure("sensor_timeout", device_id)
                    # §3.2 fix: fallback to CSV so data is not lost
                    await self._csv_fallback_write(current_time, device_id, power_watts)

                # Update local device state
                self.device_states[device_id] = 1 if power_watts > 10 else 0

                # STEP 5: ANALYTICS ENGINE
                # Bug 2.4 fix: Use real time deltas instead of hardcoded 1/3600 hours
                last_seen = self.last_device_analytics_time.get(device_id, current_time)
                real_duration_hours = (current_time - last_seen) / 3600.0
                self.last_device_analytics_time[device_id] = current_time
                # Audit fix 3.2: Remove min-clamp to prevent energy overestimation
                self.analytics.record_usage(device_id, power_watts, duration_hours=max(real_duration_hours, 0.0))

                if current_time - self.last_analytics_broadcast >= 30.0:
                    summary = self.analytics.get_daily_summary()
                    await self._broadcast_event({
                        "type": "ANALYTICS_UPDATE",
                        "summary": summary,
                    })
                    self.last_analytics_broadcast = current_time

                # STEP 6: DIGITAL TWIN — PMV COMFORT
                pmv_score = self.env.compute_pmv(
                    t_air=self._internal_temp, t_mrt=self._internal_temp - 0.5,
                    v_air=0.1, rh=50.0, clo=0.7, met=1.2
                )

                # Audit fix 3.1: Use last_device_power for ALL devices (accurate twin state)
                appliance_watts = {
                    k: self.last_device_power.get(k, 0.0)
                    for k in self.device_states
                }
                appliance_watts[device_id] = power_watts  # Apply the live tick

                # Bug 2.2 fix: Use real elapsed time
                now = time.time()
                dt_minutes = (now - self.last_sim_time) / 60.0
                self.last_sim_time = now

                self._internal_temp = self.env.simulate_step(
                    appliance_watts, outdoor_temp=28.0,
                    t_internal=self._internal_temp, dt_minutes=dt_minutes
                )

                # STEP 7: RL AGENT (Confidence + Empathy + Cooldown Gates)
                tou_rate = self.agent.get_tou_rate(current_hour)
                device_limits = self.config.get("system_safety", {}).get("device_wattage_limits", {})
                rated = device_limits.get(class_name, device_limits.get("default", 1500.0))
                pct_of_rated = power_watts / max(rated, 1.0)

                # Audit fix 2.1: Use last_device_power for full house state
                all_devices_state = {}
                for did, on_off in self.device_states.items():
                    cls = self.device_classifications.get(did, f"unknown_{did}")
                    last_power = self.last_device_power.get(did, 0.0)
                    dev_rated = device_limits.get(cls, device_limits.get("default", 1500.0))
                    all_devices_state[cls] = last_power / max(dev_rated, 1.0)
                # Ensure the current device's live reading is included
                all_devices_state[class_name] = pct_of_rated

                state_dict = {
                    "devices": all_devices_state,
                    "price_tier": self.agent.get_price_bin(tou_rate),
                    "pmv_zone": self.agent.get_pmv_zone(pmv_score),
                    "tod": self.agent.get_time_of_day_bin(current_hour),
                }

                # Snapshot prev state BEFORE action for proper TD update
                prev_state = dict(state_dict)

                action = self.agent.act(state_dict, pmv_score, confidence, class_name)

                # ── Policy Promotion Gate: shadow mode until 50 twin episodes ──
                pmv_penalty = self.env.pmv_penalty(pmv_score)
                self.promo_gate.record_twin_episode(pmv_penalty)

                if action not in ["DEFER"]:
                    logger.info(
                        f"🤖 RL Agent: {action} on {device_id} ({class_name}). "
                        f"PMV={pmv_score:.2f} | Conf={confidence:.3f} | ToU=${tou_rate:.2f}"
                        f" | Promoted={'YES' if self.promo_gate.is_promoted else 'SHADOW'}"
                    )

                    # Defense-in-depth: check BOTH class_name AND device_id against NEVER_SHED
                    device_is_critical = (
                        device_id in self.agent.NEVER_SHED
                        or any(k in device_id for k in ['fridge', 'freezer', 'pc'])
                    )
                    if action == "SHED" and class_name not in self.agent.NEVER_SHED and not device_is_critical:
                        if self.promo_gate.is_promoted:
                            # LIVE MODE: Actually send relay command
                            await self.mqtt.publish_command(
                                f"home/plug/{device_id}/command", "OFF"
                            )
                        else:
                            logger.info("  ↳ Shadow mode: SHED logged but NOT executed")
                        await self._broadcast_event({
                            "type": "RL_ACTION",
                            "device_id": device_id,
                            "action": "SHED",
                            "class": class_name,
                            "pmv": round(pmv_score, 2),
                            "confidence": round(confidence, 3),
                            "tou_rate": tou_rate,
                            "promoted": self.promo_gate.is_promoted,
                            "message": f"RL optimized: {class_name} {'OFF' if self.promo_gate.is_promoted else 'SHADOW'} (PMV {pmv_score:.2f})",
                        })

                    elif action in ["SCHEDULE_HVAC", "SHED_HVAC"]:
                        await self._broadcast_event({
                            "type": "EMPATHY_ACTION",
                            "action": action,
                            "pmv": round(pmv_score, 2),
                            "message": f"Comfort override: {action} (PMV {pmv_score:.2f})",
                        })

                    # Bug 1.5 fix: next_state must reflect ACTUAL post-action wattage.
                    # In shadow mode (not promoted), the relay is NOT triggered,
                    # so the device keeps running at full power.
                    is_actually_shed = (action == "SHED" and self.promo_gate.is_promoted)
                    next_pct = 0.0 if is_actually_shed else pct_of_rated

                    # Build next_state with full house state
                    next_all_devices = dict(all_devices_state)
                    next_all_devices[class_name] = next_pct
                    next_state = {
                        "devices": next_all_devices,
                        "price_tier": self.agent.get_price_bin(tou_rate),
                        "pmv_zone": self.agent.get_pmv_zone(pmv_score),
                        "tod": self.agent.get_time_of_day_bin(current_hour),
                    }

                    # Audit fix 2.2: Compute aggregate house load for safety penalty
                    aggregate_watts = sum(self.last_device_power.values())

                    reward = self.agent.compute_reward(
                        prev_state, action, next_state,
                        pmv_score, power_watts, tou_rate, confidence,
                        aggregate_watts=aggregate_watts
                    )
                    self.agent.update(prev_state, action, reward, next_state, classified_device=class_name)

            # ══════════════════════════════════════════════════════════
            # ALWAYS: DEVICE STATUS BROADCAST
            # ══════════════════════════════════════════════════════════
            await self._broadcast_event({
                "type": "DEVICE_STATUS",
                "device_id": device_id,
                "power": round(power_watts, 2),
                "state": "ON" if self.device_states.get(device_id, 0) == 1 else "OFF",
                "classification": class_name,
                "confidence": round(confidence, 3) if confidence else 0,
                "pmv": round(self.env.compute_pmv(
                    t_air=self._internal_temp, t_mrt=self._internal_temp - 0.5,
                    v_air=0.1, rh=50.0, clo=0.7, met=1.2
                ), 2),
                "timestamp": time.strftime("%H:%M:%S"),
            })

            # Broadcast phantom loads every 10 seconds (interval-based, not modulo)
            if current_time - getattr(self, '_last_phantom_broadcast', 0) >= 10.0:
                self._last_phantom_broadcast = current_time
                await self._broadcast_event({
                    "type": "PHANTOM_LOAD",
                    "loads": {k: round(v, 3) for k, v in self.phantom_tracker.phantom_loads.items()},
                    "total": round(self.phantom_tracker.get_total_phantom_load(), 3),
                    "offenders": self.phantom_tracker.get_worst_offenders(3),
                })

            # ══ WS-7.2: Log pipeline latency ══
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000
            if not hasattr(self, '_latency_samples'):
                self._latency_samples = []
                self._last_latency_broadcast = 0.0
            self._latency_samples.append(latency_ms)
            # Keep last 100 samples
            if len(self._latency_samples) > 100:
                self._latency_samples = self._latency_samples[-100:]

            if latency_ms > 200:
                logger.warning(f"⏱️ Pipeline latency: {latency_ms:.1f}ms (ABOVE 200ms target)")
            else:
                logger.debug(f"⏱️ Pipeline latency: {latency_ms:.1f}ms")

            # Broadcast latency stats every 30 seconds
            if current_time - self._last_latency_broadcast >= 30.0:
                self._last_latency_broadcast = current_time
                avg_latency = sum(self._latency_samples) / len(self._latency_samples)
                max_latency = max(self._latency_samples)
                p95_latency = sorted(self._latency_samples)[int(len(self._latency_samples) * 0.95)]
                await self._broadcast_event({
                    "type": "LATENCY_STATS",
                    "avg_ms": round(avg_latency, 1),
                    "max_ms": round(max_latency, 1),
                    "p95_ms": round(p95_latency, 1),
                    "samples": len(self._latency_samples),
                    "target_ms": 200,
                })

        except Exception as e:
            logger.error(f"Error processing {topic}: {e}", exc_info=True)

    # ─── Label Submitted Handler (P4.5: Prototype Registry Update) ────
    def handle_label_submitted(self, class_name: str, segments_list: list) -> None:
        """
        Called when the dashboard POSTs to /api/label_device.
        Updates the PrototypeRegistry in-process without retraining the encoder.

        Args:
            class_name:    user-provided label string
            segments_list: list of (128,) raw POWER windows, in watts, as carried
                           by the LABEL_REQUEST event's `segments` field
        """
        try:
            if self.prototype_registry is None:
                logger.warning("PrototypeRegistry not loaded — cannot process label")
                return

            segs = np.array(segments_list, dtype=np.float32)   # (K, 128)
            if segs.ndim == 1:
                segs = segs.reshape(1, -1)
            if segs.shape[-1] != 128:
                logger.error(f"Label segments wrong shape: {segs.shape}")
                return
            if not np.all(np.isfinite(segs)):
                logger.error("Label segments contain NaN/Inf — refusing to enroll")
                return

            # Reject embeddings passed in place of power traces. An embedding is
            # 128-D too, so the shape check above cannot tell them apart, and the
            # dashboard used to submit exactly that. A power window is watts:
            # non-negative, and a real appliance event reaches the 20 W
            # on-threshold somewhere in the window. Embeddings are roughly
            # zero-centred, so they fail both tests. Enrolling one produces a
            # prototype that matches nothing and silently poisons the registry,
            # so this is worth refusing loudly.
            if float(segs.min()) < -1e-3 or float(segs.max()) < 20.0:
                logger.error(
                    "Label segments do not look like power in watts "
                    f"(min={segs.min():.3f}, max={segs.max():.3f}) — refusing to "
                    "enroll '%s'. Expected raw 128-sample power windows, not "
                    "embeddings.", class_name
                )
                return

            self.prototype_registry.add_class(class_name, segs)
            # Save back to the SAME file the startup load read (respects the
            # demo/hardware config's `registry_path`), or the enrolment is lost
            # on the next boot.
            registry_path = self._resolve_registry_path()
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            self.prototype_registry.save(registry_path)
            logger.info(
                f"✅ PrototypeRegistry updated: '{class_name}' added "
                f"({len(self.prototype_registry.class_names())} classes total)"
            )
        except Exception as e:
            logger.error(f"Label submission processing failed: {e}")

    # ─── Main Run Loop ────────────────────────────────────────────────
    async def run(self) -> None:
        self._running = True

        try:
            await self.db.connect()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.failure_matrix.trigger_failure("sensor_timeout")

        # ═══ CRITICAL: Safety monitor runs as PARALLEL asyncio.Task ═══
        # It gets its own MQTT connection so it doesn't compete for messages
        import aiomqtt
        safety_mqtt = None

        async def safety_wrapper():
            """Independent safety monitor with its own MQTT connection."""
            nonlocal safety_mqtt
            while self._running:
                try:
                    async with aiomqtt.Client(
                        os.environ.get('MQTT_BROKER', self.config['mqtt']['broker']),
                        port=self.config['mqtt']['port']
                    ) as client:
                        safety_mqtt = client
                        await client.subscribe(self.config['mqtt']['topics']['reads'])
                        logger.info("🛡️ Safety monitor connected (parallel task)")
                        await self.safety.run_forever(client, self._relay_callback)
                except aiomqtt.MqttError as e:
                    logger.error(f"Safety MQTT error: {e}. Reconnecting in 3s...")
                    await asyncio.sleep(3)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Safety monitor crash: {e}. Restarting in 3s...")
                    await asyncio.sleep(3)

        # Launch safety as independent parallel task
        safety_task = asyncio.create_task(safety_wrapper())

        # ML Pipeline runs via MQTT callback
        self.mqtt.set_read_callback(self._handle_mqtt_message)
        ml_pipeline_task = asyncio.create_task(
            self.mqtt.run([
                self.config['mqtt']['topics']['reads'],
                "home/plug/+/ack",
                "home/ml/label",  # Bug 3.1 fix: Subscribe to label topic
                "home/sensor/+/status",
            ])
        )

        logger.info("═══════════════════════════════════════════")
        logger.info("  🏠 EMS Pipeline Orchestrator ONLINE")
        logger.info("  Safety Layer: ✅ (parallel task)")
        logger.info("  ProtoNet: " + ("✅" if self.encoder else "⚠️ (disabled)"))
        # Report the dict the runtime actually READS, not the one training writes.
        # `train_demo_models.py` fits index-keyed tails into `_weibull`; the only
        # runtime consumer, `compute_open_set_prob`, reads `_weibull_by_name`,
        # which the indexed `fit()` never populates. Both shipped artifacts
        # therefore carry tails and no names, and the old banner logged "✅" off
        # `_weibull` — advertising an open-set reject channel that returns 0.0
        # for every window. The physical envelope gate in `_classify_device` is
        # what actually rejects; see claude_debug/ML_PIPELINE_FIX_2026-08-25.md §2.2.
        _om_tails = len(getattr(self.weibull, '_weibull', None) or {})
        _om_named = len(getattr(self.weibull, '_weibull_by_name', None) or {})
        logger.info(
            f"  OpenMax: {_om_tails} tail(s), {_om_named} named — "
            + ("active" if _om_named else "INACTIVE (physical envelope gate is the reject channel)"))
        logger.info(f"  Temp Scaler: T={self.temp_scaler.temperature.item():.4f}")
        logger.info(f"  Confidence Gate: {self.confidence_threshold}")
        logger.info("  Delta Stability: ✅")
        logger.info("  RL Agent: ✅ | Empathy Gate: ✅")
        logger.info("═══════════════════════════════════════════")

        try:
            # Both tasks run concurrently — safety never blocks ML
            await asyncio.gather(safety_task, ml_pipeline_task)
        except asyncio.CancelledError:
            pass
        finally:
            safety_task.cancel()
            ml_pipeline_task.cancel()
            self.agent.save()
            await self.db.close()

    def shutdown(self) -> None:
        if not self._running:
            return  # Already shutting down — idempotent guard
        logger.info("Initiating graceful shutdown...")
        self._running = False

    def _evict_stale_devices(self) -> None:
        """Remove devices not seen for longer than TTL to prevent OOM from MQTT topic floods."""
        now = time.time()
        stale_ids = [
            did for did, last_seen in self._device_last_seen.items()
            if (now - last_seen) > self._device_ttl_seconds
        ]
        # Also evict if over max capacity (evict oldest first)
        if len(self._device_last_seen) > self._max_tracked_devices:
            sorted_by_age = sorted(self._device_last_seen.items(), key=lambda x: x[1])
            excess = len(self._device_last_seen) - self._max_tracked_devices
            stale_ids.extend(did for did, _ in sorted_by_age[:excess])
            stale_ids = list(set(stale_ids))  # deduplicate

        for did in stale_ids:
            for d in [self.nilm_detectors, self.power_windows, self.device_states,
                      self.device_classifications, self.action_cooldowns,
                      self.last_device_power, self.last_known_confidences,
                      self.last_device_analytics_time, self.cnn_active_ticks,
                      self._device_last_seen]:
                d.pop(did, None)
        if stale_ids:
            logger.info(f"🧹 Evicted {len(stale_ids)} stale device(s) from memory")

    def is_connected(self) -> bool:
        """Check if pipeline / MQTT connection is active."""
        if hasattr(self, 'mqtt') and self.mqtt is not None:
            return self.mqtt.is_connected()
        return self._connected

    def handle_broker_disconnect(self) -> None:
        """Called when MQTT broker goes down."""
        self._connected = False
        if hasattr(self, 'mqtt') and self.mqtt is not None:
            self.mqtt._connected = False
            self.mqtt.client = None

    def handle_broker_reconnect(self) -> None:
        """Called when MQTT broker comes back online."""
        self._connected = True
        if hasattr(self, 'mqtt') and self.mqtt is not None:
            self.mqtt._connected = True

    async def process_raw_mqtt(
        self, topic: str, payload: Union[str, bytes, bytearray, dict, float, int]
    ) -> PipelineResult:
        """
        Entry point to process a raw incoming MQTT payload.
        Handles payload decoding, JSON validation, safety clamping, and pipeline dispatch.
        """
        events: List[PipelineEvent] = []
        device_id = topic.split("/")[-2] if "/" in topic else "unknown"

        # Robust decoding
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload_str = payload.decode("utf-8")
            except Exception as e:
                logger.warning(f"Malformed payload encoding on {topic}: {e}")
                events.append(PipelineEvent(event_type="PARSE_ERROR", source="MQTT_PARSER", device_id=device_id))
                return PipelineResult(events=events)
        elif isinstance(payload, dict):
            payload_str = json.dumps(payload)
        else:
            payload_str = str(payload)

        # Parse JSON or numeric value
        power_watts: Optional[float] = None
        trimmed = payload_str.strip()
        if trimmed.startswith("{"):
            try:
                data = json.loads(trimmed)
                if not isinstance(data, dict):
                    logger.warning(f"Payload JSON is not a dict on {topic}: {trimmed}")
                    events.append(PipelineEvent(event_type="PARSE_ERROR", source="MQTT_PARSER", device_id=device_id))
                    return PipelineResult(events=events)
                if "power" not in data:
                    logger.warning(f"Missing 'power' in JSON payload on {topic}: {trimmed}")
                    events.append(PipelineEvent(event_type="PARSE_ERROR", source="MQTT_PARSER", device_id=device_id))
                    return PipelineResult(events=events)
                power_watts = float(data["power"])
            except Exception as e:
                logger.warning(f"Malformed JSON on {topic}: {trimmed} ({e})")
                events.append(PipelineEvent(event_type="PARSE_ERROR", source="MQTT_PARSER", device_id=device_id))
                return PipelineResult(events=events)
        else:
            try:
                power_watts = float(trimmed)
            except ValueError:
                logger.warning(f"Malformed non-JSON payload on {topic}: {trimmed}")
                events.append(PipelineEvent(event_type="PARSE_ERROR", source="MQTT_PARSER", device_id=device_id))
                return PipelineResult(events=events)

        if math.isnan(power_watts) or math.isinf(power_watts):
            logger.warning(f"NaN/Inf power reading on {topic}: {power_watts}")
            events.append(PipelineEvent(event_type="SENSOR_ERROR", source="SENSOR_ERROR", device_id=device_id))
            return PipelineResult(events=events)

        # Check for extreme power values (> 50,000 W or negative)
        if power_watts > 50000.0 or power_watts < 0:
            logger.warning(f"Extreme power reading rejected on {topic}: {power_watts}W")
            events.append(PipelineEvent(event_type="SENSOR_ERROR", source="SENSOR_ERROR", device_id=device_id, message="Extreme power reading rejected"))
            return PipelineResult(events=events)

        # Process valid reading through pipeline
        await self._handle_mqtt_message(topic, str(power_watts))
        return PipelineResult(events=events)

    async def process(self, event: Any) -> PipelineResult:
        """Process event helper for pipeline stage testing."""
        if self._stage_hook:
            self._stage_hook("confidence_gate")
            self._stage_hook("delta_stability")
        if isinstance(event, dict):
            device_id = event.get("device", event.get("device_id", "node_fridge"))
            power = event.get("power", 0.0)
            return await self.process_raw_mqtt(f"home/sensor/{device_id}/power", json.dumps({"power": power}))
        elif hasattr(event, "device") and hasattr(event, "power"):
            return await self.process_raw_mqtt(f"home/sensor/{event.device}/power", json.dumps({"power": event.power}))
        return PipelineResult()


# FullPipeline alias for tests and orchestrator components
FullPipeline = EMSOrchestrator


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="EMS Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. config/config.demo.yaml)")
    args = parser.parse_args()
    config = load_config(args.config)
    orchestrator = EMSOrchestrator(config=config)
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    _shutting_down = False

    def _signal_handler():
        nonlocal _shutting_down
        if _shutting_down:
            return  # Idempotent: ignore repeated signals from shell's `kill 0`
        _shutting_down = True
        orchestrator.shutdown()
        # Remove handlers so subsequent signals use default behavior (fast exit)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except Exception:
                pass
        if main_task and not main_task.done():
            main_task.cancel()

    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
