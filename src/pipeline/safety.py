import logging
import asyncio
import re
from typing import Callable, Awaitable, Dict
from src.hardware.mqtt import aiomqtt

logger = logging.getLogger(__name__)

class SafetyMonitor:
    def __init__(self, max_aggregate_wattage: float, device_wattage_limits: Dict[str, float], 
                 warning_pct: float = 1.10, critical_pct: float = 1.25):
        self.max_aggregate_wattage = max_aggregate_wattage
        self.device_wattage_limits = device_wattage_limits
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct
        self.default_limit = device_wattage_limits.get("default", 1500.0)

        # Phase 2 (WS-5): Rate-of-change tracking for arc-fault proxy detection
        self._prev_readings: Dict[str, float] = {}  # device_id → last watts
        self.ROC_THRESHOLD = 1000.0  # W/s — rapid rate-of-change (arc-fault proxy)

    async def run_forever(self, mqtt_client: aiomqtt.Client, relay_callback: Callable[[str, str], Awaitable[None]]):
        """
        Separate task — never awaits ML pipeline.
        Subscribes directly to power topics and acts immediately.
        """
        logger.info("🛡️ Parallel Safety Monitor running")
        try:
            async for message in mqtt_client.messages:
                topic_str = str(message.topic)
                if "/power" in topic_str:
                    parts = topic_str.split("/")
                    if len(parts) >= 3:
                        device_id = parts[-2]
                        try:
                            payload_str = message.payload.decode("utf-8") if isinstance(message.payload, bytes) else str(message.payload)
                            watts = float(payload_str)
                            
                            rated = self.device_wattage_limits.get(device_id, self.default_limit)
                            pct = watts / rated

                            # ── Phase 2 (WS-5.3): Rate-of-change arc-fault detection ──
                            prev_watts = self._prev_readings.get(device_id, watts)
                            rate_of_change = abs(watts - prev_watts)  # W/s at 1Hz
                            self._prev_readings[device_id] = watts

                            if rate_of_change > self.ROC_THRESHOLD:
                                logger.critical(
                                    f"⚡ ARC FAULT PROXY: {device_id} dP/dt={rate_of_change:.0f} W/s "
                                    f"(threshold: {self.ROC_THRESHOLD} W/s)!"
                                )
                                await relay_callback(device_id, "OFF")
                                self._log_event("ARC_FAULT", device_id, watts, rate_of_change)
                            elif pct >= self.critical_pct:
                                logger.critical(f"⚡ CRITICAL CUTOFF: {device_id} is drawing {watts:.1f}W ({pct*100:.1f}% of {rated}W)!")
                                await relay_callback(device_id, "OFF")
                                self._log_event("CRITICAL", device_id, watts, pct)
                            elif pct >= self.warning_pct:
                                logger.warning(f"⚠️ WARNING: {device_id} is drawing {watts:.1f}W ({pct*100:.1f}% of {rated}W)!")
                                await relay_callback(device_id, "WARNING")
                                self._log_event("WARNING", device_id, watts, pct)
                                
                        except ValueError:
                            pass # invalid float
                            
        except asyncio.CancelledError:
            logger.info("Safety Monitor task cancelled.")
        except Exception as e:
            logger.error(f"Safety Monitor error: {e}")

    def _log_event(self, level: str, device_id: str, watts: float, pct_or_roc: float) -> None:
        """Log safety event to a file independent of DB."""
        try:
            with open("safety_events.log", "a") as f:
                import time
                f.write(f"{time.time()},{level},{device_id},{watts},{pct_or_roc}\n")
        except Exception:
            pass
