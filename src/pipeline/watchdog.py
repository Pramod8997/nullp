"""
Module 2: Soft Anomaly Watchdog
A parallel monitoring layer that detects soft anomalies (e.g., sensor drift, slow leaks)
before they become critical failures. It works alongside the primary ProtoNet.
"""
import time
import math
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

        # Patch 10: Reject poisonous inputs that would corrupt the rolling history
        if not isinstance(reading, (int, float)) or math.isnan(reading) or math.isinf(reading):
            return False

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
        # If the reading deviates massively from a low baseline (e.g., 1.5W → 1500W),
        # this is a normal appliance start, not a soft anomaly.
        # Likewise, if the reading drops massively to <10W, it's a normal appliance stop.
        if abs(reading - mean) > 100.0 and (mean < 10.0 or reading < 10.0):
            history.clear()
            history.append(reading)
            return False
            
        z_score = abs(reading - mean) / stdev
        
        # Update history
        history.append(reading)
        
        return z_score > self.z_score_threshold

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class WatchdogEvent:
    event_type: str = "WATCHDOG_ANOMALY"
    device: str = ""
    reading: float = 0.0
    z_score: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    timestamp: float = 0.0


class Watchdog:
    """
    Rolling z-score anomaly detector for device power telemetry (Stage 1).
    Detects soft anomalies (spikes, sensor drift) based on rolling z-scores.
    """
    def __init__(self, window: int = 30, threshold: float = 3.0, **kwargs):
        self.window = kwargs.get("window_size", window)
        self.threshold = kwargs.get("z_score_threshold", threshold)
        self.history: Dict[str, deque] = {}

    def get_std(self, device_id: str) -> float:
        if device_id not in self.history or len(self.history[device_id]) < 2:
            return 1.0
        stdev = statistics.stdev(self.history[device_id])
        return stdev if stdev > 1e-6 else 1.0

    def get_zscore(self, device_id: str, reading: Optional[float] = None) -> float:
        if device_id not in self.history or len(self.history[device_id]) == 0:
            return 0.0
        history = self.history[device_id]
        if reading is None:
            reading = history[-1]
        if len(history) < 2:
            return 0.0
        mean = statistics.mean(history)
        if abs(reading - mean) < 1e-9:
            return 0.0
        std = self.get_std(device_id)
        return (reading - mean) / std

    def update(self, device_id: str, reading: float) -> Optional[WatchdogEvent]:
        if device_id not in self.history:
            self.history[device_id] = deque(maxlen=self.window)

        if not isinstance(reading, (int, float)) or math.isnan(reading) or math.isinf(reading):
            return None

        history = self.history[device_id]

        if len(history) < self.window:
            history.append(reading)
            return None

        mean = statistics.mean(history)
        std = self.get_std(device_id)

        # Audit fix 4.2: Detect normal ON/OFF state transitions and reset baseline.
        if abs(reading - mean) > 100.0 and (mean < 10.0 or reading < 10.0):
            history.clear()
            history.append(reading)
            return None

        z_score = abs(reading - mean) / std
        history.append(reading)

        if z_score > self.threshold:
            return WatchdogEvent(
                event_type="WATCHDOG_ANOMALY",
                device=device_id,
                reading=reading,
                z_score=z_score,
                mean=mean,
                std=std,
                timestamp=time.time(),
            )
        return None

    async def process(self, event_or_device: Any, reading: Optional[float] = None) -> Optional[WatchdogEvent]:
        if reading is not None:
            device_id = str(event_or_device)
            val = float(reading)
        elif hasattr(event_or_device, "device"):
            device_id = getattr(event_or_device, "device")
            val = float(getattr(event_or_device, "power", getattr(event_or_device, "watts", getattr(event_or_device, "reading", 0.0))))
        elif hasattr(event_or_device, "device_id"):
            device_id = getattr(event_or_device, "device_id")
            val = float(getattr(event_or_device, "power", getattr(event_or_device, "watts", getattr(event_or_device, "reading", 0.0))))
        elif isinstance(event_or_device, dict):
            device_id = event_or_device.get("device", event_or_device.get("device_id", "unknown"))
            val = float(event_or_device.get("power", event_or_device.get("watts", event_or_device.get("reading", 0.0))))
        else:
            device_id = str(event_or_device)
            val = 0.0

        if device_id not in self.history:
            self.history[device_id] = deque(maxlen=self.window)

        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return WatchdogEvent(event_type="NORMAL", device=device_id, reading=val)

        history = self.history[device_id]

        if len(history) < self.window:
            history.append(val)
            return WatchdogEvent(event_type="NORMAL", device=device_id, reading=val)

        mean = statistics.mean(history)
        std = self.get_std(device_id)

        # Normal ON/OFF state transitions check
        if abs(val - mean) > 100.0 and (mean < 10.0 or val < 10.0):
            history.clear()
            history.append(val)
            return WatchdogEvent(event_type="NORMAL", device=device_id, reading=val)

        z_score = abs(val - mean) / std
        history.append(val)

        if z_score > self.threshold:
            return WatchdogEvent(
                event_type="WATCHDOG_ANOMALY",
                device=device_id,
                reading=val,
                z_score=z_score,
                mean=mean,
                std=std,
                timestamp=time.time(),
            )
        return WatchdogEvent(
            event_type="NORMAL",
            device=device_id,
            reading=val,
            z_score=z_score,
            mean=mean,
            std=std,
            timestamp=time.time(),
        )


watchdog = SoftAnomalyWatchdog()
