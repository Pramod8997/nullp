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

    async def run_forever(self, mqtt_client: aiomqtt.Client, relay_callback: Callable[[str, str], Awaitable[None]]):
        """
        Separate task — never awaits ML pipeline.
        Subscribes directly to power topics and acts immediately.
        """
        logger.info("🛡️ Parallel Safety Monitor running")
        try:
            # We must subscribe again or rely on the fact that the client has subscribed.
            # Assuming the orchestrator subscribes the client to 'home/sensor/+/power'.
            # aiomqtt allows multiple async for loops over client.messages if configured correctly,
            # but usually it's one message queue. 
            # If the orchestrator is already consuming client.messages, we need to ensure both get it.
            # Actually, aiomqtt.Client.messages is an AsyncGenerator. If multiple tasks iterate over it,
            # they compete for messages.
            # Instead, the orchestrator passes messages to us, OR we use a separate MQTT connection.
            # The architectural spec says "subscribes to the same MQTT power topic independently".
            # We'll assume the orchestrator creates a separate aiomqtt.Client for the safety monitor 
            # if we truly want independent subscription, or we just process the message passed to us.
            # Let's write the loop assuming a dedicated client connection is passed.
            
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
                            
                            if pct >= self.critical_pct:
                                logger.critical(f"⚡ CRITICAL CUTOFF: {device_id} is drawing {watts:.1f}W ({pct*100:.1f}% of {rated}W)!")
                                await relay_callback(device_id, "OFF")
                                self._log_event("CRITICAL", device_id, watts, pct)
                            elif pct >= self.warning_pct:
                                logger.warning(f"⚠️ WARNING: {device_id} is drawing {watts:.1f}W ({pct*100:.1f}% of {rated}W)!")
                                await relay_callback(device_id, "WARNING") # Dashboard event or relay warning
                                self._log_event("WARNING", device_id, watts, pct)
                                
                        except ValueError:
                            pass # invalid float
                            
        except asyncio.CancelledError:
            logger.info("Safety Monitor task cancelled.")
        except Exception as e:
            logger.error(f"Safety Monitor error: {e}")

    def _log_event(self, level: str, device_id: str, watts: float, pct: float) -> None:
        """Log safety event to a file independent of DB."""
        try:
            with open("safety_events.log", "a") as f:
                import time
                f.write(f"{time.time()},{level},{device_id},{watts},{pct}\n")
        except Exception:
            pass
