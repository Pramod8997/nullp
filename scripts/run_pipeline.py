"""
EMS Pipeline Orchestrator
=========================
Central orchestration loop implementing the full architecture:

    MQTT → Safety Layer → CNN/ProtoNet Open-Set Detection
      → Known Device Flow (Log + Analytics + Dashboard)
      → Unknown Device Flow (Delta Stability + User Prompt)
      → Watchdog (Soft Anomaly Z-Score)
      → Phantom Tracker (Vampire Loads)
      → Digital Twin (PMV Empathy Gate)
      → RL Agent (Cooldown-Protected Decision Engine)
      → Dashboard Broadcast
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
from typing import Union, Dict, Optional

import yaml
import torch
import numpy as np

# Core EMS Modules
from src.database.session import DatabaseSession
from src.hardware.mqtt import MQTTClientManager
from src.pipeline.safety import SafetyMonitor
from src.rl.agent import TabularQLearningAgent

# Phase 2 Integration Modules
from src.models.thermodynamics import ThermodynamicsModel as DigitalTwinEnv
from src.models.protonet import ProtoNet, SupportSetManager
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

        # ── Safety Layer (Parallel, always runs) ──
        safety_cfg = self.config.get("system_safety", {})
        self.safety = SafetyMonitor(
            safety_cfg.get("max_aggregate_wattage", 3500.0),
            safety_cfg.get("device_wattage_limits", {}),
            self._trigger_cutoff
        )

        # ── Phase 2 Components ──
        self.env = DigitalTwinEnv()
        self.phantom_tracker = PhantomTracker()
        self.watchdog = SoftAnomalyWatchdog()
        self.analytics = AnalyticsEngine(
            cost_per_kwh=self.config.get("analytics", {}).get("cost_per_kwh", 0.15)
        )
        self.failure_matrix = FailureMatrix()
        self.mode_classifier = ModeClassifier()

        # ── ProtoNet / CNN ──
        proto_cfg = self.config.get("protonet", {})
        self.window_size = proto_cfg.get("window_size", 60)
        self.embedding_size = proto_cfg.get("embedding_size", 64)
        self.distance_threshold = proto_cfg.get("distance_threshold", 15.0)
        self.protonet: Optional[ProtoNet] = None
        self.support_manager = SupportSetManager()
        self._load_protonet(proto_cfg)

        # Rolling windows for CNN input (per device)
        self.power_windows: Dict[str, deque] = {}

        # ── RL Agent ──
        rl_cfg = self.config.get("rl", {})
        self.agent = TabularQLearningAgent()
        q_path = rl_cfg.get("q_table_path", "backend/models/weights/q_table.pkl")
        if os.path.exists(q_path):
            with open(q_path, "rb") as f:
                self.agent.q_table = pickle.load(f)
            logger.info(f"Loaded Q-table from {q_path}")

        self.COOLDOWN_SECONDS = rl_cfg.get("cooldown_seconds", 15.0)
        self.PMV_MIN = rl_cfg.get("empathy_pmv_min", -1.0)
        self.PMV_MAX = rl_cfg.get("empathy_pmv_max", 1.0)

        # ── State Memory ──
        self.device_states: Dict[str, int] = {}
        self.device_classifications: Dict[str, str] = {}
        self.action_cooldowns: Dict[str, float] = {}
        self.last_analytics_broadcast = 0.0
        self._running = False

    def _load_protonet(self, proto_cfg: dict) -> None:
        """Load CNN/ProtoNet weights for open-set device classification."""
        weights_path = proto_cfg.get("weights_path", "")
        anchors_path = proto_cfg.get("anchors_path", "")

        try:
            if os.path.exists(weights_path):
                self.protonet = ProtoNet(
                    input_size=self.window_size,
                    embedding_size=self.embedding_size
                )
                self.protonet.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
                self.protonet.eval()
                logger.info(f"✅ ProtoNet CNN loaded from {weights_path}")
            else:
                logger.warning(f"ProtoNet weights not found at {weights_path}. Classification disabled.")

            if os.path.exists(anchors_path):
                self.support_manager.support_sets = torch.load(anchors_path, map_location="cpu", weights_only=False)
                logger.info(f"✅ ProtoNet anchors loaded ({len(self.support_manager.support_sets)} classes)")
            else:
                logger.warning(f"ProtoNet anchors not found at {anchors_path}.")

        except Exception as e:
            logger.error(f"Failed to load ProtoNet: {e}. Falling back to rule-based classification.")
            self.failure_matrix.trigger_failure("model_drift")
            self.protonet = None

    # ─── Safety Cutoff ────────────────────────────────────────────────
    async def _trigger_cutoff(self, device_id: str) -> None:
        """Hardware relay cutoff — highest priority action."""
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

    # ─── ProtoNet Classification ──────────────────────────────────────
    def _classify_device(self, device_id: str, power_watts: float) -> str:
        """
        Run CNN feature extraction → ProtoNet distance check → Open-Set Detection.
        Returns 'known:<class>' or 'unknown'.
        """
        # Maintain rolling window
        if device_id not in self.power_windows:
            self.power_windows[device_id] = deque(maxlen=self.window_size)
        self.power_windows[device_id].append(power_watts)

        window = self.power_windows[device_id]
        if len(window) < self.window_size or self.protonet is None:
            return "pending"

        try:
            # Prepare input tensor [1, 1, window_size]
            x = torch.tensor(list(window), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                embedding = self.protonet(x).squeeze(0)  # [embedding_size]

            # Compare to all known prototypes
            best_class = "unknown"
            best_distance = float("inf")

            for class_id, support in self.support_manager.support_sets.items():
                prototype = support.mean(dim=0)
                distance = torch.norm(embedding - prototype).item()
                if distance < best_distance:
                    best_distance = distance
                    best_class = class_id

            # Open-set threshold check
            if best_distance <= self.distance_threshold:
                return f"known:{best_class}"
            else:
                return "unknown"

        except Exception as e:
            logger.error(f"ProtoNet classification error for {device_id}: {e}")
            return "error"

    # ─── Delta Stability Check (Unknown Device Flow) ──────────────────
    def _check_delta_stability(self, device_id: str) -> bool:
        """Check if an unknown device's power draw is stable enough for labeling."""
        window = self.power_windows.get(device_id)
        if window is None or len(window) < 10:
            return False
        recent = list(window)[-10:]
        variance = np.var(recent)
        return variance < 50.0  # Low variance = stable signature

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

    # ─── Main Message Handler ─────────────────────────────────────────
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

            # ══════════════════════════════════════════════════════════
            # STEP 1: PARALLEL SAFETY LAYER (CRITICAL — Always first)
            # ══════════════════════════════════════════════════════════
            is_safe = await self.safety.process_reading(device_id, power_watts)
            if not is_safe:
                # Safety cutoff already triggered. Log and continue monitoring.
                try:
                    await self.db.insert_measurement(current_time, device_id, power_watts)
                except Exception:
                    self.failure_matrix.trigger_failure("sensor_timeout", device_id)
                return

            # ══════════════════════════════════════════════════════════
            # STEP 2: SOFT ANOMALY WATCHDOG (SECONDARY)
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
            # STEP 3: CNN / PROTONET OPEN-SET DETECTION
            # ══════════════════════════════════════════════════════════
            classification = self._classify_device(device_id, power_watts)
            self.device_classifications[device_id] = classification

            if classification.startswith("known:"):
                # ── KNOWN DEVICE FLOW ──
                class_name = classification.split(":")[1]
                logger.debug(f"ProtoNet: {device_id} classified as {class_name}")
            elif classification == "unknown":
                # ── UNKNOWN DEVICE FLOW ──
                is_stable = self._check_delta_stability(device_id)
                if is_stable:
                    logger.info(f"❓ Unknown stable signature on {device_id} ({power_watts:.1f}W) — ready for labeling")
                    await self._broadcast_event({
                        "type": "UNKNOWN_DEVICE",
                        "device_id": device_id,
                        "power": round(power_watts, 2),
                        "stable": True,
                        "message": f"Stable unknown signature on {device_id}. Label this device?",
                    })
                else:
                    logger.debug(f"Unknown unstable signature on {device_id}, assigned unknown_X")

            # ══════════════════════════════════════════════════════════
            # STEP 4: PHANTOM TRACKER
            # ══════════════════════════════════════════════════════════
            is_off = self.device_states.get(device_id, 0) == 0
            self.phantom_tracker.track(device_id, power_watts, is_off)

            # ══════════════════════════════════════════════════════════
            # STEP 5: DATABASE PERSISTENCE
            # ══════════════════════════════════════════════════════════
            try:
                await self.db.insert_measurement(current_time, device_id, power_watts)
            except Exception as e:
                logger.error(f"DB write failed: {e}")
                self.failure_matrix.trigger_failure("sensor_timeout", device_id)

            # Update local device state (ON if > 10W)
            self.device_states[device_id] = 1 if power_watts > 10 else 0

            # ══════════════════════════════════════════════════════════
            # STEP 6: ANALYTICS ENGINE
            # ══════════════════════════════════════════════════════════
            # Record usage (1Hz = 1 sample per second = 1/3600 hours)
            self.analytics.record_usage(device_id, power_watts, duration_hours=1.0 / 3600.0)

            # Broadcast analytics summary every 30 seconds
            if current_time - self.last_analytics_broadcast >= 30.0:
                summary = self.analytics.get_daily_summary()
                await self._broadcast_event({
                    "type": "ANALYTICS_UPDATE",
                    "summary": summary,
                })
                self.last_analytics_broadcast = current_time

            # ══════════════════════════════════════════════════════════
            # STEP 7: DIGITAL TWIN — PMV COMFORT CALCULATION
            # ══════════════════════════════════════════════════════════
            pmv_score = self.env.calculate_pmv(
                t_air=22.5, t_radiant=22.0, v_air=0.1, rh=50.0, met=1.2, clo=0.7
            )

            # ══════════════════════════════════════════════════════════
            # STEP 8: RL AGENT (Cooldown + Empathy Gate)
            # ══════════════════════════════════════════════════════════
            current_hour = time.localtime().tm_hour
            power_bin = min(9, int(power_watts / 500))
            device_names = list(self.device_states.keys())[:self.agent.MAX_RL_DEVICES]
            # Pad to MAX_RL_DEVICES
            while len(device_names) < self.agent.MAX_RL_DEVICES:
                device_names.append(f"pad_{len(device_names)}")
            dev_tuple = tuple(self.device_states.get(k, 0) for k in device_names)

            state = self.agent.get_state_tuple(current_hour, power_bin, dev_tuple)
            action = self.agent.get_action(state)

            # Decode per-device action
            device_idx = device_names.index(device_id) if device_id in device_names else -1
            if device_idx >= 0:
                temp_act = action
                action_decoded = []
                for _ in range(self.agent.MAX_RL_DEVICES):
                    action_decoded.append(temp_act % 3)
                    temp_act //= 3
                dev_action = action_decoded[device_idx]

                if dev_action == 0:  # Agent wants TURN_OFF
                    if self.device_states.get(device_id, 0) == 1:
                        last_action = self.action_cooldowns.get(device_id, 0)
                        if current_time - last_action >= self.COOLDOWN_SECONDS:
                            # ── EMPATHY GATE ──
                            if self.PMV_MIN <= pmv_score <= self.PMV_MAX:
                                logger.info(
                                    f"🤖 RL Agent: Turning OFF {device_id}. "
                                    f"PMV={pmv_score:.2f} | Cooldown OK"
                                )
                                await self.mqtt.publish_command(
                                    f"home/plug/{device_id}/command", "OFF"
                                )
                                self.action_cooldowns[device_id] = current_time

                                await self._broadcast_event({
                                    "type": "RL_ACTION",
                                    "device_id": device_id,
                                    "action": "TURN_OFF",
                                    "pmv": round(pmv_score, 2),
                                    "message": f"RL optimized: {device_id} OFF (PMV {pmv_score:.2f})",
                                })

                                # Q-learning update
                                next_dev_tuple = tuple(
                                    self.device_states.get(k, 0) for k in device_names
                                )
                                next_state = self.agent.get_state_tuple(
                                    current_hour, power_bin, next_dev_tuple
                                )
                                self.agent.update(state, action, reward=1.0, next_state=next_state)
                            else:
                                logger.warning(
                                    f"🛑 EMPATHY GATE: Blocked RL action on {device_id}. "
                                    f"PMV={pmv_score:.2f} out of [{self.PMV_MIN}, {self.PMV_MAX}]"
                                )
                                await self._broadcast_event({
                                    "type": "EMPATHY_BLOCK",
                                    "device_id": device_id,
                                    "pmv": round(pmv_score, 2),
                                    "message": f"Comfort violation — blocked {device_id} OFF",
                                })

            # ══════════════════════════════════════════════════════════
            # STEP 9: DEVICE STATUS BROADCAST
            # ══════════════════════════════════════════════════════════
            await self._broadcast_event({
                "type": "DEVICE_STATUS",
                "device_id": device_id,
                "power": round(power_watts, 2),
                "state": "ON" if self.device_states.get(device_id, 0) == 1 else "OFF",
                "classification": classification,
                "pmv": round(pmv_score, 2),
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

        self.mqtt.set_read_callback(self._handle_mqtt_message)
        mqtt_task = asyncio.create_task(
            self.mqtt.run(self.config['mqtt']['topics']['reads'])
        )

        logger.info("═══════════════════════════════════════════")
        logger.info("  🏠 EMS Pipeline Orchestrator ONLINE")
        logger.info("  Safety Layer: ✅  |  ProtoNet: " + ("✅" if self.protonet else "⚠️ (disabled)"))
        logger.info("  Watchdog: ✅  |  Phantom Tracker: ✅")
        logger.info("  RL Agent: ✅  |  Empathy Gate: ✅")
        logger.info("═══════════════════════════════════════════")

        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            mqtt_task.cancel()
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
