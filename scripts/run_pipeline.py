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
import json
import os
import csv
import pickle
from collections import deque
from datetime import datetime
from typing import Union, Dict, Optional

import yaml
import torch
import numpy as np

# Core EMS Modules
from src.database.session import DatabaseSession
from src.hardware.mqtt import MQTTClientManager
from src.pipeline.safety import SafetyMonitor
from src.rl.agent import TabularQLearningAgent, PolicyPromotionGate

# ML & Pipeline Modules
from src.models.thermodynamics import ThermodynamicsModel as DigitalTwinEnv, PMVThermodynamics
from src.models.protonet import (
    CNN1DEncoder, TemperatureScaler, WEibullOpenMax,
    SupportSetManager, ProtoNet, PrototypeRegistry, OpenMaxWeibull
)
from src.models.calibration import TemperatureScaler as CalibratedTemperatureScaler
from src.pipeline.aggregate_nilm import NILMTransientDetector
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.watchdog import SoftAnomalyWatchdog
from src.pipeline.temporal_validator import TemporalValidator
from src.pipeline.analytics import AnalyticsEngine
from src.pipeline.failure_matrix import FailureMatrix
from src.pipeline.classifier import ModeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EMSOrchestrator:
    """
    Central hub for the Confidence-Aware Digital Twin EMS.
    Manages safety, comfort, state-aware control, and ML classification.
    """

    def __init__(self) -> None:
        # ── Configuration ──
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # ── Infrastructure ──
        self.db = DatabaseSession(self.config['database']['path'])
        # Allow MQTT_BROKER env var to override config (for Docker deployments)
        mqtt_broker = os.environ.get('MQTT_BROKER', self.config['mqtt']['broker'])
        self.mqtt = MQTTClientManager(
            mqtt_broker,
            self.config['mqtt']['port']
        )

        # ── Safety Layer (will run as PARALLEL asyncio.Task) ──
        safety_cfg = self.config.get("system_safety", {})
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
            threshold=pre_cfg.get("transient_threshold_w", 50.0),
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

        # Bug 1.6 fix: Carry over last known confidences during steady-state ticks
        self.last_known_confidences: Dict[str, float] = {}

        # Bug 2.2 fix: Track actual simulation time for real dt calculation
        self.last_sim_time: float = time.time()

        # Bug 2.4 fix: Track last analytics time per device for real duration
        self.last_device_analytics_time: Dict[str, float] = {}

        # Bug 4.3 fix: asyncio lock for CSV writes
        self._csv_lock = asyncio.Lock()

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
                registry_path = os.path.join(weights_dir, "prototype_registry.pt")
                if os.path.exists(registry_path) and self.encoder is not None:
                    self.prototype_registry = PrototypeRegistry(self.encoder)
                    self.prototype_registry.load(registry_path)
                    logger.info(f"✅ Prototype Registry loaded ({len(self.prototype_registry.class_names())} classes)")
                else:
                    self.prototype_registry = None
            except Exception as e:
                logger.warning(f"Prototype Registry load failed: {e}")
                self.prototype_registry = None

            # Load OpenMax Weibull
            try:
                omw_path = os.path.join(weights_dir, "openmax_weibull.pkl")
                if os.path.exists(omw_path):
                    self.weibull = OpenMaxWeibull(num_classes=10)
                    self.weibull.load(omw_path)
                    logger.info(f"✅ OpenMax Weibull loaded")
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
                logger.info(f"✅ Legacy temperature scaler loaded")
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
        """Callback for safety monitor to trigger relay actions."""
        if action == "OFF":
            logger.warning(f"⚡ Safety Cutoff: Forcing {device_id} OFF.")
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
        elif action == "WARNING":
            await self._broadcast_event({
                "type": "SAFETY_WARNING",
                "device_id": device_id,
                "severity": "warning",
                "message": f"Power draw approaching limit on {device_id}",
            })

    # ─── ProtoNet Classification (with OpenMax + confidence) ──────────
    def _classify_device(self, device_id: str, power_watts: float,
                         filtered_segment: np.ndarray = None):
        """
        Full classification pipeline:
        1. NILM-filtered segment (or legacy rolling window) → CNN → embedding
        2. Distance to prototypes → softmax → calibrated confidence
        3. Weibull OpenMax → unknown detection
        4. Returns (class_name, confidence, distances) or ("pending", 0, {})

        Args:
            filtered_segment: (128,) pre-filtered segment from NILMTransientDetector.
                              If provided, bypasses the legacy rolling window.
        """
        if self.encoder is None:
            return "pending", 0.0, {}

        # Use NILM-filtered segment when available (§2.1 fix)
        if filtered_segment is not None:
            window_np = filtered_segment
        else:
            # Legacy fallback: maintain rolling window
            if device_id not in self.power_windows:
                self.power_windows[device_id] = deque(maxlen=self.seq_len)
            self.power_windows[device_id].append(power_watts)
            window = self.power_windows[device_id]
            if len(window) < self.seq_len:
                return "pending", 0.0, {}
            window_np = np.array(list(window), dtype=np.float32)

        try:
            class_name, confidence, distances = self.support_manager.classify(
                window_np, self.encoder, self.weibull, self.temp_scaler,
                self.confidence_threshold
            )
            return class_name, confidence, distances

        except Exception as e:
            logger.error(f"ProtoNet classification error for {device_id}: {e}")
            return "error", 0.0, {}

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

            power_watts = float(payload_str) if payload_str else 0.0
            current_time = time.time()
            current_hour = datetime.now().hour

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
                    threshold=pre_cfg.get("transient_threshold_w", 50.0),
                    window_size=pre_cfg.get("transient_window_s", 5),
                    embed_window=128,
                )

            is_transient, filtered_segment = self.nilm_detectors[device_id].push(power_watts)

            if not is_transient:
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
                    if stability == 'stable':
                        logger.info(f"❓ Stable unknown on {device_id} ({power_watts:.1f}W) — requesting label")
                        await self._broadcast_event({
                            "type": "LABEL_REQUEST",
                            "device_id": device_id,
                            "power": round(power_watts, 2),
                            "confidence": round(confidence, 3),
                            "embedding": cluster_mean.tolist() if cluster_mean is not None else [],
                            "message": f"Stable unknown signature on {device_id}. Please label this device.",
                        })

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
                            k: (self.power_windows[k][-1] if k in self.power_windows and len(self.power_windows[k]) > 0 else 0)
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

                        # Bug 1.1 fix: Build full house state for RL, not just current device
                        all_devices_state = {}
                        for did, on_off in self.device_states.items():
                            cls = self.device_classifications.get(did, f"unknown_{did}")
                            last_power = self.power_windows[did][-1] if did in self.power_windows and len(self.power_windows[did]) > 0 else 0
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
                
            elif confidence < self.confidence_threshold:
                # ── LOW CONFIDENCE GATE ──
                logger.info(f"⚠️ Low confidence ({confidence:.3f}) for {class_name} on {device_id}. Skipping RL.")
                await self._broadcast_event({
                    "type": "LOW_CONFIDENCE",
                    "device_id": device_id,
                    "classified_as": class_name,
                    "confidence": round(confidence, 3),
                    "threshold": self.confidence_threshold,
                    "message": f"Classification uncertain ({confidence:.2f} < {self.confidence_threshold})",
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
                self.analytics.record_usage(device_id, power_watts, duration_hours=max(real_duration_hours, 1.0 / 3600.0))

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

                # Bug 2.1 fix: Use last-known wattages for ALL devices,
                # not just the currently ticking one (prevents state obliteration)
                appliance_watts = {
                    k: (self.power_windows[k][-1] if k in self.power_windows and len(self.power_windows[k]) > 0 else 0)
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

                # Bug 1.1 fix: Pass the FULL house state to the RL agent,
                # not just the currently ticking device
                all_devices_state = {}
                for did, on_off in self.device_states.items():
                    cls = self.device_classifications.get(did, f"unknown_{did}")
                    last_power = self.power_windows[did][-1] if did in self.power_windows and len(self.power_windows[did]) > 0 else 0
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

                    if action == "SHED" and class_name not in self.agent.NEVER_SHED:
                        if self.promo_gate.is_promoted:
                            # LIVE MODE: Actually send relay command
                            await self.mqtt.publish_command(
                                f"home/plug/{device_id}/command", "OFF"
                            )
                        else:
                            logger.info(f"  ↳ Shadow mode: SHED logged but NOT executed")
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

                    reward = self.agent.compute_reward(
                        prev_state, action, next_state,
                        pmv_score, power_watts, tou_rate, confidence
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
            segments_list: list of (128,) float arrays from the WebSocket broadcast
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

            self.prototype_registry.add_class(class_name, segs)
            registry_path = 'backend/models/weights/prototype_registry.pt'
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
            ])
        )

        logger.info("═══════════════════════════════════════════")
        logger.info("  🏠 EMS Pipeline Orchestrator ONLINE")
        logger.info(f"  Safety Layer: ✅ (parallel task)")
        logger.info(f"  ProtoNet: " + ("✅" if self.encoder else "⚠️ (disabled)"))
        logger.info(f"  OpenMax: " + ("✅" if getattr(self.weibull, '_weibull', None) else "⚠️"))
        logger.info(f"  Temp Scaler: T={self.temp_scaler.temperature.item():.4f}")
        logger.info(f"  Confidence Gate: {self.confidence_threshold}")
        logger.info(f"  Delta Stability: ✅")
        logger.info(f"  RL Agent: ✅ | Empathy Gate: ✅")
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


async def main() -> None:
    orchestrator = EMSOrchestrator()
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
