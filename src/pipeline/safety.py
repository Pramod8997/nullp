import logging
from typing import Callable, Awaitable, Dict

logger = logging.getLogger(__name__)

class SafetyMonitor:
    def __init__(self, max_aggregate_wattage: float, device_wattage_limits: Dict[str, float], trigger_cutoff: Callable[[str], Awaitable[None]]):
        self.max_aggregate_wattage = max_aggregate_wattage
        self.device_wattage_limits = device_wattage_limits
        self.trigger_cutoff = trigger_cutoff
        
    async def process_reading(self, device_id: str, power_watts: float) -> bool:
        """
        Check if reading exceeds safety limits.
        Returns True if safe, False if cutoff was triggered.
        """
        limit = self.device_wattage_limits.get(device_id, self.device_wattage_limits.get("default", self.max_aggregate_wattage))
        if power_watts > limit or power_watts > self.max_aggregate_wattage:
            logger.critical(f"SAFETY THRESHOLD BREACHED! Device {device_id} drawing {power_watts}W (Limit: {limit}W). Triggering cutoff!")
            try:
                await self.trigger_cutoff(device_id)
            except Exception as e:
                logger.error(f"Failed to trigger cutoff for {device_id}: {e}")
            return False
        return True
