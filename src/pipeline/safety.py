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
import asyncio
import time
from typing import Callable, Awaitable, Dict
from src.hardware.mqtt import aiomqtt

logger = logging.getLogger(__name__)


class FleetDiagnosticsMonitor:
    """
    Asynchronous fleet diagnostics, logging, and UI alert dispatcher.

    Monitors power telemetry across all nodes and dispatches structured
    alerts to the dashboard pipeline. Does not actuate relays — that
    responsibility belongs to the edge firmware.

    Args:
        max_aggregate_wattage: Fleet-wide aggregate power ceiling (W).
        device_wattage_limits: Per-device rated wattage limits.
        warning_pct: Percentage of rated that triggers a WARNING alert.
        critical_pct: Percentage of rated that triggers a CRITICAL alert.
    """

    def __init__(self, max_aggregate_wattage: float,
                 device_wattage_limits: Dict[str, float],
                 warning_pct: float = 1.10,
                 critical_pct: float = 1.25):
        self.max_aggregate_wattage = max_aggregate_wattage
        self.device_wattage_limits = device_wattage_limits
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct
        self.default_limit = device_wattage_limits.get("default", 1500.0)

        # Fleet-wide rate-of-change tracking for anomaly correlation
        self._prev_readings: Dict[str, float] = {}
        self.ROC_THRESHOLD = 1000.0  # W/s — arc-fault proxy (diagnostic only)

        # Aggregate fleet power tracking
        self._current_readings: Dict[str, float] = {}

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
                                self._log_event("ARC_FAULT", device_id,
                                                watts, rate_of_change)

                            elif pct >= self.critical_pct:
                                logger.critical(
                                    f"⚡ CRITICAL THRESHOLD: {device_id} drawing "
                                    f"{watts:.1f}W ({pct*100:.1f}% of {rated}W) — "
                                    f"edge node handles physical cutoff"
                                )
                                # Dispatch alert to dashboard (no relay command)
                                await relay_callback(device_id, "ALERT_CRITICAL")
                                self._log_event("CRITICAL", device_id, watts, pct)

                            elif pct >= self.warning_pct:
                                logger.warning(
                                    f"⚠️ WARNING: {device_id} drawing "
                                    f"{watts:.1f}W ({pct*100:.1f}% of {rated}W)"
                                )
                                await relay_callback(device_id, "WARNING")
                                self._log_event("WARNING", device_id, watts, pct)

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

    def _log_event(self, level: str, device_id: str,
                   watts: float, pct_or_roc: float) -> None:
        """Log safety event to a file independent of DB for audit trail."""
        try:
            with open("safety_events.log", "a") as f:
                f.write(f"{time.time()},{level},{device_id},"
                        f"{watts},{pct_or_roc}\n")
        except Exception:
            pass


# Backward compatibility alias — existing imports use SafetyMonitor
SafetyMonitor = FleetDiagnosticsMonitor
