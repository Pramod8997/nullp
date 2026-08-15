"""
Fleet Diagnostics Monitor (formerly SafetyMonitor)

Production architecture: Safety relay cutoffs are now executed at the edge
(ESP32 Core 0 — see firmware/esp32_node/src/main.cpp). This server-side
module is demoted to:
  - Fleet-wide anomaly tracking and threshold monitoring
  - Structured alert dispatching to the dashboard UI pipeline
  - Safety event logging to persistent file for audit trails

It does NOT issue relay commands over MQTT. All physical safety actions
are edge-local and network-independent.
"""
import logging
import math
import asyncio
import time
import os
import yaml
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Any, Optional, List
from src.hardware.mqtt import aiomqtt

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from path or default search paths."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    for p in ["config.yaml", "config/config.yaml"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {
        "system_safety": {
            "max_aggregate_wattage": 3500.0,
            "warning_pct": 1.10,
            "critical_pct": 1.25,
            "device_wattage_limits": {
                "node_fridge": 200.0,
                "node_microwave": 1200.0,
                "node_kettle": 2500.0,
                "node_hvac": 2000.0,
                "default": 1500.0,
            },
        },
        "devices": {
            "node_fridge": {"rated": 200, "tier0": True},
            "node_microwave": {"rated": 1200, "tier0": False},
            "node_kettle": {"rated": 2500, "tier0": False},
            "node_hvac": {"rated": 2000, "tier0": False},
        },
    }


async def slow_rl_agent(delay: float = 0.5):
    """Simulated slow RL agent for concurrency testing."""
    await asyncio.sleep(delay)
    return []


@dataclass
class SafetyEvent:
    event_type: str = "SAFETY_ALERT"
    level: str = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"
    device: str = ""
    watts: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class FleetDiagnosticsMonitor:
    """
    Asynchronous fleet diagnostics, logging, and UI alert dispatcher.

    Monitors power telemetry across all nodes and dispatches structured
    alerts to the dashboard pipeline. Does not actuate relays — that
    responsibility belongs to the edge firmware.
    """

    def __init__(self,
                 max_aggregate_wattage: Optional[float] = None,
                 device_wattage_limits: Optional[Dict[str, float]] = None,
                 warning_pct: float = 1.10,
                 critical_pct: float = 1.25,
                 config: Optional[Dict[str, Any]] = None,
                 protonet_enabled: bool = True,
                 broadcast_fn: Optional[Callable] = None,
                 **kwargs):
        safety_cfg = config.get("system_safety", {}) if config else {}
        self.max_aggregate_wattage = (
            max_aggregate_wattage
            if max_aggregate_wattage is not None
            else safety_cfg.get("max_aggregate_wattage", 3500.0)
        )
        self.device_wattage_limits = (
            device_wattage_limits
            if device_wattage_limits is not None
            else safety_cfg.get("device_wattage_limits", {})
        )
        self.warning_pct = safety_cfg.get("warning_pct", warning_pct)
        self.critical_pct = safety_cfg.get("critical_pct", critical_pct)
        self.default_limit = self.device_wattage_limits.get("default", 1500.0)
        self.protonet_enabled = protonet_enabled
        self.broadcast_fn = broadcast_fn

        # Fleet-wide rate-of-change tracking for anomaly correlation
        self._prev_readings: Dict[str, float] = {}
        self.ROC_THRESHOLD = 1000.0  # W/s — arc-fault proxy (diagnostic only)

        # Aggregate fleet power tracking
        self._current_readings: Dict[str, float] = {}

    async def check_aggregate(self, power_map: Dict[str, float]) -> SafetyEvent:
        """Check fleet aggregate power against ceiling."""
        total = sum(power_map.values())
        if total > self.max_aggregate_wattage:
            evt = SafetyEvent(
                event_type="SAFETY_ALERT",
                level="CRITICAL",
                watts=total,
                details={"power_map": power_map},
            )
            if self.broadcast_fn:
                if asyncio.iscoroutinefunction(self.broadcast_fn):
                    await self.broadcast_fn(evt)
                else:
                    self.broadcast_fn(evt)
            return evt
        elif total >= self.max_aggregate_wattage * self.warning_pct:
            evt = SafetyEvent(
                event_type="SAFETY_ALERT",
                level="WARNING",
                watts=total,
                details={"power_map": power_map},
            )
            return evt
        else:
            return SafetyEvent(
                event_type="",
                level="NORMAL",
                watts=total,
                details={"power_map": power_map},
            )

    async def check_roc(self, device: str, prev_power: float, curr_power: float,
                        dt_seconds: float = 1.0) -> Optional[SafetyEvent]:
        """Check rate of change for arc fault proxy."""
        dt = max(dt_seconds, 1e-6)
        rate_of_change = abs(curr_power - prev_power) / dt
        if rate_of_change > self.ROC_THRESHOLD:
            evt = SafetyEvent(
                event_type="ARC_FAULT",
                level="CRITICAL",
                device=device,
                watts=curr_power,
                details={"roc": rate_of_change},
            )
            if self.broadcast_fn:
                if asyncio.iscoroutinefunction(self.broadcast_fn):
                    await self.broadcast_fn(evt)
                else:
                    self.broadcast_fn(evt)
            return evt
        return None

    async def check_device(self, device: str, power: float) -> Optional[SafetyEvent]:
        """Check single device wattage against rated limits."""
        rated = self.device_wattage_limits.get(device, self.default_limit)
        pct = power / rated if rated > 0 else 1.0
        if pct >= self.critical_pct:
            evt = SafetyEvent(
                event_type="SAFETY_ALERT",
                level="CRITICAL",
                device=device,
                watts=power,
                details={"rated": rated, "pct": pct},
            )
            return evt
        elif power > rated or pct >= self.warning_pct:
            evt = SafetyEvent(
                event_type="SAFETY_ALERT",
                level="WARNING",
                device=device,
                watts=power,
                details={"rated": rated, "pct": pct},
            )
            return evt
        return None

    async def process(self, event: Any) -> Any:
        """Non-blocking pipeline processing for stage 0."""
        return event

    async def run_forever(self, mqtt_client: aiomqtt.Client,
                          relay_callback: Callable[[str, str], Awaitable[None]]):
        """
        Separate asyncio task — never awaits the ML pipeline.
        Subscribes directly to power topics and dispatches alerts.

        The relay_callback is retained for backward compatibility but now
        receives alert-level actions (ALERT_CRITICAL, ALERT_WARNING,
        ALERT_ARC_FAULT) instead of direct relay commands. The orchestrator's
        _relay_callback translates these into dashboard UI events.
        """
        logger.info("🛡️ Fleet Diagnostics Monitor running (edge-safety mode)")
        try:
            async for message in mqtt_client.messages:
                topic_str = str(message.topic)
                if "/power" in topic_str:
                    parts = topic_str.split("/")
                    if len(parts) >= 3:
                        device_id = parts[-2]
                        try:
                            payload_str = (message.payload.decode("utf-8")
                                           if isinstance(message.payload, bytes)
                                           else str(message.payload))
                            watts = float(payload_str)

                            # Patch 9: Input sanitization — reject NaN/Inf/negative
                            if math.isnan(watts) or math.isinf(watts):
                                logger.warning(
                                    f"⚠️ Invalid reading from {device_id}: "
                                    f"{payload_str} — skipping")
                                continue
                            watts = abs(watts)  # Treat negative as faulty sensor

                            # Track fleet aggregate
                            self._current_readings[device_id] = watts

                            rated = self.device_wattage_limits.get(
                                device_id, self.default_limit)
                            pct = watts / rated

                            # ── Rate-of-Change Anomaly Tracking ──
                            prev_watts = self._prev_readings.get(device_id, watts)
                            rate_of_change = abs(watts - prev_watts)
                            self._prev_readings[device_id] = watts

                            # Inrush suppression: low baseline = normal appliance start
                            is_normal_inrush = prev_watts < 50.0

                            if (rate_of_change > self.ROC_THRESHOLD
                                    and not is_normal_inrush):
                                logger.critical(
                                    f"⚡ ARC FAULT DETECTED (fleet diagnostic): "
                                    f"{device_id} dP/dt={rate_of_change:.0f} W/s "
                                    f"(threshold: {self.ROC_THRESHOLD} W/s) — "
                                    f"edge node handles physical cutoff"
                                )
                                # Dispatch alert to dashboard (no relay command)
                                await relay_callback(device_id, "ALERT_ARC_FAULT")
                                # Fix §4.3.2: Non-blocking safety log write
                                await self._log_event_async(
                                    "ARC_FAULT", device_id,
                                    watts, rate_of_change)

                            elif pct >= self.critical_pct:
                                logger.critical(
                                    f"⚡ CRITICAL THRESHOLD: {device_id} drawing "
                                    f"{watts:.1f}W ({pct*100:.1f}% of {rated}W) — "
                                    f"edge node handles physical cutoff"
                                )
                                # Dispatch alert to dashboard (no relay command)
                                await relay_callback(device_id, "ALERT_CRITICAL")
                                # Fix §4.3.2: Non-blocking safety log write
                                await self._log_event_async(
                                    "CRITICAL", device_id, watts, pct)

                            elif pct >= self.warning_pct:
                                logger.warning(
                                    f"⚠️ WARNING: {device_id} drawing "
                                    f"{watts:.1f}W ({pct*100:.1f}% of {rated}W)"
                                )
                                await relay_callback(device_id, "WARNING")
                                # Fix §4.3.2: Non-blocking safety log write
                                await self._log_event_async(
                                    "WARNING", device_id, watts, pct)

                            # ── Fleet aggregate check ──
                            total_fleet = sum(self._current_readings.values())
                            if total_fleet > self.max_aggregate_wattage:
                                logger.warning(
                                    f"⚠️ FLEET AGGREGATE: {total_fleet:.0f}W "
                                    f"> {self.max_aggregate_wattage:.0f}W ceiling"
                                )

                        except ValueError:
                            pass  # invalid float

        except asyncio.CancelledError:
            logger.info("Fleet Diagnostics Monitor task cancelled.")
        except Exception as e:
            logger.error(f"Fleet Diagnostics Monitor error: {e}")

    def _log_event_sync(self, level: str, device_id: str,
                        watts: float, pct_or_roc: float) -> None:
        """Synchronous log write — safe to call from sync context
        or via asyncio.to_thread()."""
        try:
            with open("safety_events.log", "a") as f:
                f.write(f"{time.time()},{level},{device_id},"
                        f"{watts},{pct_or_roc}\n")
        except Exception:
            pass

    async def _log_event_async(self, level: str, device_id: str,
                               watts: float, pct_or_roc: float) -> None:
        """Fix §4.3.2: Non-blocking safety log write.
        Runs sync file I/O in a thread to avoid stalling the event loop."""
        await asyncio.to_thread(
            self._log_event_sync, level, device_id, watts, pct_or_roc
        )

    # Keep legacy name for backward compatibility
    _log_event = _log_event_sync


# Backward compatibility alias — existing imports use SafetyMonitor
SafetyMonitor = FleetDiagnosticsMonitor

