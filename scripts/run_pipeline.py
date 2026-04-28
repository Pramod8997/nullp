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
from src.rl.agent import TabularQLearningAgent

# ML & Pipeline Modules
from src.models.thermodynamics import ThermodynamicsModel as DigitalTwinEnv
from src.models.protonet import (
    CNN1DEncoder, TemperatureScaler, WEibullOpenMax,
    SupportSetManager, ProtoNet
)
from src.pipeline.delta_stability import DeltaStabilityAnalyzer
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.watchdog import SoftAnomalyWatchdog
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
        self.mqtt = MQTTClientManager(
            self.config['mqtt']['broker'],
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

        # ── ProtoNet / CNN ──
        proto_cfg = self.config.get("protonet", {})
        self.window_size = proto_cfg.get("window_size", 60)
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
            buffer_size=ds_cfg.get("buffer_size", 10),
            stability_threshold=ds_cfg.get("stability_threshold", 3.0),
            min_occurrences=ds_cfg.get("min_occurrences", 3),
        )

        # Rolling windows for CNN input (per device)
        self.power_windows: Dict[str, deque] = {}

        # ── RL Agent (reads config internally) ──
        self.agent = TabularQLearningAgent()

        # ── State Memory ──
        self.device_states: Dict[str, int] = {}
        self.device_classifications: Dict[str, str] = {}
        self.action_cooldowns: Dict[str, float] = {}
        self.last_analytics_broadcast = 0.0
        self._running = False
        self._internal_temp = 22.0  # Indoor temp for digital twin

    def _load_ml_models(self, proto_cfg: dict) -> None:
        """Load CNN encoder, temperature scaler, Weibull, and support registry."""
        weights_path = proto_cfg.get("weights_path", "")
        anchors_path = proto_cfg.get("anchors_path", "")
        weights_dir = os.path.dirname(weights_path) if weights_path else "backend/models/weights"

        try:
            if os.path.exists(weights_path):
                self.encoder = CNN1DEncoder(
                    input_size=self.window_size,
                    embedding_size=self.embedding_size
                )
                self.encoder.load_state_dict(
                    torch.load(weights_path, map_location="cpu", weights_only=False)
                )
                self.encoder.eval()
                logger.info(f"✅ ProtoNet CNN loaded from {weights_path}")
            else:
                logger.warning(f"ProtoNet weights not found at {weights_path}. Classification disabled.")

            if os.path.exists(anchors_path):
                self.support_manager.load_registry(anchors_path)
                logger.info(f"✅ Support registry loaded from {anchors_path}")
            else:
                logger.warning(f"Support registry not found at {anchors_path}.")

            scaler_path = os.path.join(weights_dir, "temperature_scaler.pth")
            if os.path.exists(scaler_path):
                self.temp_scaler.load(scaler_path)
                logger.info(f"✅ Temperature scaler loaded (T={self.temp_scaler.temperature.item():.4f})")

            weibull_path = os.path.join(weights_dir, "weibull_openmax.pkl")
            if os.path.exists(weibull_path):
                with open(weibull_path, 'rb') as f:
                    self.weibull = pickle.load(f)
                logger.info(f"✅ Weibull OpenMax loaded ({len(self.weibull.weibull_models)} classes)")

        except Exception as e:
            logger.error(f"Failed to load ML models: {e}. Falling back to rule-based classification.")
            self.failure_matrix.trigger_failure("model_drift")
            self.encoder = None

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
    def _classify_device(self, device_id: str, power_watts: float):
        """
        Full classification pipeline:
        1. Rolling window → CNN → embedding
        2. Distance to prototypes → softmax → calibrated confidence
        3. Weibull OpenMax → unknown detection
        4. Returns (class_name, confidence, distances) or ("pending", 0, {})
        """
        # Maintain rolling window
        if device_id not in self.power_windows:
            self.power_windows[device_id] = deque(maxlen=self.window_size)
        self.power_windows[device_id].append(power_watts)

        window = self.power_windows[device_id]
        if len(window) < self.window_size or self.encoder is None:
            return "pending", 0.0, {}

        try:
            window_np = np.array(list(window), dtype=np.float32)
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

    # ─── Main Message Handler (ML Pipeline) ───────────────────────────
    async def _handle_mqtt_message(
        self, topic: str, payload: Union[str, bytes, bytearray, int, float, None]
    ) -> None:
        try:
            device_id = topic.split("/")[-2]
            # Robust payload decoding
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8", errors="replace")
            power_watts = float(payload) if payload else 0.0
            current_time = time.time()
            current_hour = datetime.now().hour

            # ══════════════════════════════════════════════════════════
            # NOTE: Safety monitor runs as SEPARATE parallel task.
            # This handler only processes ML pipeline steps.
            # ══════════════════════════════════════════════════════════

            # ══════════════════════════════════════════════════════════
            # STEP 1: SOFT ANOMALY WATCHDOG
            # ══════════════════════════════════════════════════════════
            if self.watchdog.check_reading(device_id, power_watts):
                logger.warning(f"🔍 WATCHDOG: Soft anomaly on {device_id} ({power_watts:.1f}W)")
                await self._broadcast_event({
                    "type": "SOFT_ANOMALY",
                    "device_id": device_id,
                    "power": round(power_watts, 2),
                    "message": f"Z-score anomaly detected on {device_id}",
                })

            # ══════════════════════════════════════════════════════════
            # STEP 2: CNN / PROTONET + OPENMAX CLASSIFICATION
            # ══════════════════════════════════════════════════════════
            class_name, confidence, distances = self._classify_device(device_id, power_watts)
            self.device_classifications[device_id] = class_name

            if class_name == "pending" or class_name == "error":
                # Not enough data or model error — skip RL
                pass

            elif class_name == "unknown":
                # ── UNKNOWN DEVICE FLOW ──
                # Get the last embedding for delta stability check
                if device_id in self.power_windows and len(self.power_windows[device_id]) >= self.window_size:
                    window_np = np.array(list(self.power_windows[device_id]), dtype=np.float32)
                    with torch.no_grad():
                        x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        embedding = self.encoder(x).squeeze(0).numpy() if self.encoder else np.zeros(self.embedding_size)

                    is_stable, temp_id = self.delta_analyzer.check(embedding)
                    if is_stable:
                        logger.info(f"❓ Stable unknown on {device_id} ({power_watts:.1f}W) — requesting label")
                        await self._broadcast_event({
                            "type": "LABEL_REQUEST",
                            "device_id": device_id,
                            "power": round(power_watts, 2),
                            "confidence": round(confidence, 3),
                            "message": f"Stable unknown signature on {device_id}. Please label this device.",
                        })
                    else:
                        logger.debug(f"Transient unknown on {device_id}, assigned {temp_id}")
                        try:
                            await self.db.insert_measurement(current_time, device_id, power_watts)
                        except Exception:
                            pass
                # Skip RL for unknown devices
                
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

                # Update local device state
                self.device_states[device_id] = 1 if power_watts > 10 else 0

                # STEP 5: ANALYTICS ENGINE
                self.analytics.record_usage(device_id, power_watts, duration_hours=1.0 / 3600.0)

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

                # Update internal temperature via thermal simulation
                appliance_watts = {k: (power_watts if k == device_id else 0)
                                   for k in self.device_states}
                self._internal_temp = self.env.simulate_step(
                    appliance_watts, outdoor_temp=28.0,
                    t_internal=self._internal_temp, dt_minutes=1.0/60.0
                )

                # STEP 7: RL AGENT (Confidence + Empathy + Cooldown Gates)
                tou_rate = self.agent.get_tou_rate(current_hour)
                device_limits = self.config.get("system_safety", {}).get("device_wattage_limits", {})
                rated = device_limits.get(class_name, device_limits.get("default", 1500.0))
                pct_of_rated = power_watts / max(rated, 1.0)

                state_dict = {
                    "devices": {class_name: pct_of_rated},
                    "price_tier": self.agent.get_price_bin(tou_rate),
                    "pmv_zone": self.agent.get_pmv_zone(pmv_score),
                    "tod": self.agent.get_time_of_day_bin(current_hour),
                }

                action = self.agent.act(state_dict, pmv_score, confidence, class_name)

                if action not in ["DEFER"]:
                    logger.info(
                        f"🤖 RL Agent: {action} on {device_id} ({class_name}). "
                        f"PMV={pmv_score:.2f} | Conf={confidence:.3f} | ToU=${tou_rate:.2f}"
                    )

                    if action == "SHED" and class_name not in self.agent.NEVER_SHED:
                        await self.mqtt.publish_command(
                            f"home/plug/{device_id}/command", "OFF"
                        )
                        await self._broadcast_event({
                            "type": "RL_ACTION",
                            "device_id": device_id,
                            "action": "SHED",
                            "class": class_name,
                            "pmv": round(pmv_score, 2),
                            "confidence": round(confidence, 3),
                            "tou_rate": tou_rate,
                            "message": f"RL optimized: {class_name} OFF (PMV {pmv_score:.2f})",
                        })

                    elif action in ["SCHEDULE_HVAC", "SHED_HVAC"]:
                        await self._broadcast_event({
                            "type": "EMPATHY_ACTION",
                            "action": action,
                            "pmv": round(pmv_score, 2),
                            "message": f"Comfort override: {action} (PMV {pmv_score:.2f})",
                        })

                    # Q-learning update
                    reward = self.agent.compute_reward(
                        state_dict, action, state_dict,
                        pmv_score, power_watts, tou_rate, confidence
                    )
                    self.agent.update(state_dict, action, reward, state_dict)

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

            # Broadcast phantom loads every 10 seconds
            if int(current_time) % 10 == 0:
                await self._broadcast_event({
                    "type": "PHANTOM_LOAD",
                    "loads": {k: round(v, 3) for k, v in self.phantom_tracker.phantom_loads.items()},
                    "total": round(self.phantom_tracker.get_total_phantom_load(), 3),
                    "offenders": self.phantom_tracker.get_worst_offenders(3),
                })

        except Exception as e:
            logger.error(f"Error processing {topic}: {e}", exc_info=True)

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
                        self.config['mqtt']['broker'],
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
            self.mqtt.run(self.config['mqtt']['topics']['reads'])
        )

        logger.info("═══════════════════════════════════════════")
        logger.info("  🏠 EMS Pipeline Orchestrator ONLINE")
        logger.info(f"  Safety Layer: ✅ (parallel task)")
        logger.info(f"  ProtoNet: " + ("✅" if self.encoder else "⚠️ (disabled)"))
        logger.info(f"  OpenMax: " + ("✅" if self.weibull.weibull_models else "⚠️"))
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
        logger.info("Initiating graceful shutdown...")
        self._running = False


async def main() -> None:
    orchestrator = EMSOrchestrator()
    loop = asyncio.get_running_loop()

    if sys.platform != 'win32':
        loop.add_signal_handler(signal.SIGINT, lambda: orchestrator.shutdown())
        loop.add_signal_handler(signal.SIGTERM, lambda: orchestrator.shutdown())

    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
