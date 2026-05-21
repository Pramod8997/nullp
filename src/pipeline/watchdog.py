"""
Module 2: Soft Anomaly Watchdog
A parallel monitoring layer that detects soft anomalies (e.g., sensor drift, slow leaks)
before they become critical failures. It works alongside the primary ProtoNet.
"""
import time
from collections import deque
import statistics

class SoftAnomalyWatchdog:
    def __init__(self, window_size: int = 60, z_score_threshold: float = 3.0):
        """
        Initialize the Soft Anomaly Watchdog.
        
        Args:
            window_size: Number of recent samples to keep for baseline.
            z_score_threshold: The z-score beyond which an anomaly is flagged.
        """
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.history = {}
        
    def check_reading(self, device_id: str, reading: float) -> bool:
        """
        Check if a reading is a soft anomaly based on rolling z-score.
        
        Args:
            device_id: Identifier for the device/sensor.
            reading: The current value.
            
        Returns:
            bool: True if an anomaly is detected, False otherwise.
        """
        if device_id not in self.history:
            self.history[device_id] = deque(maxlen=self.window_size)
            
        history = self.history[device_id]
        
        # Need a minimum number of samples to establish a baseline
        if len(history) < 10:
            history.append(reading)
            return False
            
        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 0.0
        
        # Avoid division by zero in perfect stable state
        if stdev < 1e-6:
            stdev = 1e-6

        # Audit fix 4.2: Detect normal ON/OFF state transitions and reset baseline.
        # If the reading deviates massively from a low baseline (e.g., 1.5W → 1500W),
        # this is a normal appliance start, not a soft anomaly.
        if abs(reading - mean) > 100.0 and mean < 10.0:
            history.clear()
            history.append(reading)
            return False
            
        z_score = abs(reading - mean) / stdev
        
        # Update history
        history.append(reading)
        
        return z_score > self.z_score_threshold

watchdog = SoftAnomalyWatchdog()
