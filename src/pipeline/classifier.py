import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class ModeClassifier:
    def __init__(self, variance_threshold: float = 50.0):
        self.variance_threshold = variance_threshold
        
    def classify_mode(self, recent_power_window: List[float]) -> str:
        """
        Determines if the current power window represents a single device or a
        MULTI_DEVICE_AGGREGATE based on variance or overlapping unstable baselines.
        """
        if not recent_power_window or len(recent_power_window) < 2:
            return "UNKNOWN"
            
        variance = float(np.var(recent_power_window))
        # Simplified heuristic for [INV-3]: High variance in an otherwise "steady" state 
        # often indicates overlapping inverters/devices constantly shifting.
        if variance > self.variance_threshold:
            logger.debug(f"High variance ({variance:.2f}) detected. Tagging as MULTI_DEVICE_AGGREGATE.")
            return "MULTI_DEVICE_AGGREGATE"
            
        return "SINGLE_DEVICE_STABLE"
