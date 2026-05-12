"""
Temporal Validator — bridges the SoftAnomalyWatchdog and the RL Agent.

Architecture spec: [Anomaly Detection Module] → [Temporal Validation] → Suggest Relay (soft control).

When the Watchdog flags a soft anomaly (z-score drift), the TemporalValidator
confirms whether the anomaly is persistent (not a one-off transient) and, if so,
suggests a soft-control action to the RL agent (e.g., defer or reduce load on the
degrading appliance).

This prevents the RL agent from ignoring slowly degrading equipment (e.g., a fridge
compressor drawing increasing power over time).
"""
import time
import logging
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
VALIDATION_WINDOW    = 10      # number of anomaly events to buffer
PERSISTENCE_COUNT    = 3       # min consecutive anomalies to confirm persistence
PERSISTENCE_TIMEOUT  = 300.0   # max seconds between anomalies to count as consecutive
COOLDOWN_SECONDS     = 60.0    # min seconds between soft-control suggestions per device


class TemporalValidator:
    """
    Temporal anomaly validation layer (DFD spec: Temporal Validation → Suggest Relay).

    Usage:
        tv = TemporalValidator()
        suggestion = tv.validate(device_id, power_watts)
        if suggestion:
            action, info = suggestion
            # Feed 'action' into RL agent or broadcast to dashboard
    """

    def __init__(self,
                 window:              int   = VALIDATION_WINDOW,
                 persistence_count:   int   = PERSISTENCE_COUNT,
                 persistence_timeout: float = PERSISTENCE_TIMEOUT,
                 cooldown:            float = COOLDOWN_SECONDS):
        self.window             = window
        self.persistence_count  = persistence_count
        self.persistence_timeout = persistence_timeout
        self.cooldown           = cooldown

        # Per-device anomaly event history: deque of (timestamp, power_watts)
        self._anomaly_history: dict = {}
        # Per-device last soft-control suggestion timestamp
        self._last_suggestion: dict = {}

    def record_anomaly(self, device_id: str, power_watts: float) -> None:
        """Record a watchdog-flagged anomaly event for temporal validation."""
        if device_id not in self._anomaly_history:
            self._anomaly_history[device_id] = deque(maxlen=self.window)
        self._anomaly_history[device_id].append((time.time(), power_watts))

    def validate(self, device_id: str, power_watts: float) -> Optional[Tuple[str, dict]]:
        """
        Record the anomaly and check whether a soft-control suggestion should be issued.

        Args:
            device_id:   the device that triggered the anomaly
            power_watts: the anomalous power reading

        Returns:
            None if no action needed, or (action_str, info_dict) for soft control.
            action_str is one of: 'SOFT_DEFER', 'SOFT_SHED_SUGGEST'.
        """
        self.record_anomaly(device_id, power_watts)
        history = self._anomaly_history.get(device_id, deque())

        if len(history) < self.persistence_count:
            return None

        # Check the last `persistence_count` events are temporally consecutive
        recent = list(history)[-self.persistence_count:]
        for i in range(1, len(recent)):
            if recent[i][0] - recent[i - 1][0] > self.persistence_timeout:
                return None  # gap too large; not persistent

        # All recent anomalies are within the timeout window — confirmed persistent anomaly
        # Check cooldown before issuing another suggestion
        last = self._last_suggestion.get(device_id, 0.0)
        if time.time() - last < self.cooldown:
            return None

        # Determine severity of drift
        powers = [p for _, p in recent]
        avg_anomaly_power = sum(powers) / len(powers)
        drift_trend = powers[-1] - powers[0]  # positive = increasing draw

        if drift_trend > 0:
            # Increasing power draw → degrading equipment (e.g., failing compressor)
            action = "SOFT_SHED_SUGGEST"
        else:
            # Decreasing or fluctuating → intermittent issue, defer scheduling
            action = "SOFT_DEFER"

        info = {
            "device_id": device_id,
            "anomaly_count": len(history),
            "persistent_count": self.persistence_count,
            "avg_anomaly_power": round(avg_anomaly_power, 2),
            "drift_trend_w": round(drift_trend, 2),
            "message": (
                f"Persistent anomaly on {device_id}: {len(history)} events, "
                f"avg {avg_anomaly_power:.1f}W, trend {drift_trend:+.1f}W"
            ),
        }

        self._last_suggestion[device_id] = time.time()
        logger.info(f"🔔 TemporalValidator: {action} for {device_id} — {info['message']}")
        return action, info

    def reset(self, device_id: Optional[str] = None) -> None:
        """Reset anomaly history for a device, or all devices if None."""
        if device_id:
            self._anomaly_history.pop(device_id, None)
            self._last_suggestion.pop(device_id, None)
        else:
            self._anomaly_history.clear()
            self._last_suggestion.clear()
