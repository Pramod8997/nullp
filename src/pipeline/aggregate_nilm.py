"""
Module 3: Aggregate NILM — Savitzky-Golay filter + derivative transient detector.

Step 1: smooth signal with Savitzky-Golay filter.
Step 2: compute first derivative (np.diff).
Step 3: flag transient if |derivative| >= TRANSIENT_THRESHOLD_W within
        any 5-sample rolling window.
Returns the windowed signal segment centred on the transient peak for
downstream CNN encoding.

Production Fixes (§2.3 Feasibility Study):
  - Fix 2.1: Configurable sample rate (1Hz → 10Hz) with auto-scaled SG params
  - Fix 2.2: Multi-label overlap detection via power-level subtraction
"""
import time
import logging
import numpy as np
from collections import deque
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WINDOW_SIZE           = 5       # seconds (matches 1 Hz data → 5 samples)
SG_WINDOW             = 7       # Savitzky-Golay window (must be odd, > WINDOW_SIZE)
SG_POLYORD            = 2       # polynomial order
TRANSIENT_THRESHOLD_W = 50.0   # ±50 W triggers event


class NILMTransientDetector:
    """
    Implements the SG-filter + derivative transient detector described in the
    Phase-1 implementation prompt (GAP 1).

    Fix §2.3.1: Now supports configurable sample rates. When sample_rate_hz > 1,
    the SG window and derivative check window are automatically scaled to maintain
    the same temporal coverage (e.g., 5 seconds of history at any sample rate).

    Usage (1 Hz loop):
        detector = NILMTransientDetector()
        is_t, segment = detector.push(power_w)
        if is_t:
            embedding = cnn_encoder(segment)

    Usage (10 Hz loop):
        detector = NILMTransientDetector(sample_rate_hz=10)
        is_t, segment = detector.push(power_w)
    """

    def __init__(self,
                 window_size: int   = WINDOW_SIZE,
                 sg_window:   int   = SG_WINDOW,
                 sg_polyord:  int   = SG_POLYORD,
                 threshold:   float = TRANSIENT_THRESHOLD_W,
                 embed_window: int  = 128,
                 sample_rate_hz: int = 1):
        """
        Args:
            window_size:    Time window in seconds to check for transients.
            sg_window:      Base Savitzky-Golay filter window (for 1Hz).
            sg_polyord:     SG polynomial order.
            threshold:      Power change threshold (W) to flag a transient.
            embed_window:   Number of samples fed to the CNN encoder.
            sample_rate_hz: Input sample rate in Hz (Fix §2.3.1).
                            SG window and derivative check window are scaled
                            proportionally to maintain temporal coverage.
        """
        self.sample_rate_hz = max(1, sample_rate_hz)
        self.threshold      = threshold
        self.sg_polyord     = sg_polyord
        self.embed_window   = embed_window

        # Scale windows by sample rate to maintain temporal coverage
        # At 10Hz, a 5-second window = 50 samples instead of 5
        self.window_size = window_size * self.sample_rate_hz

        # SG window must be odd and > sg_polyord
        scaled_sg = sg_window * self.sample_rate_hz
        self.sg_window = scaled_sg if scaled_sg % 2 == 1 else scaled_sg + 1
        self.sg_window = max(self.sg_window, sg_polyord + 2)

        # Scale threshold: at higher sample rates, per-sample changes are smaller
        # Keep threshold in W/sec units by dividing by sample rate
        self.scaled_threshold = threshold / self.sample_rate_hz

        self._buffer: list = []            # rolling power samples

        logger.info(
            f"NILMTransientDetector initialized: {self.sample_rate_hz}Hz, "
            f"SG_window={self.sg_window}, derivative_window={self.window_size}, "
            f"threshold={self.scaled_threshold:.1f} W/sample"
        )

    def push(self, power_w: float):
        """
        Push one sample at the configured sample rate.
        Returns (is_transient: bool, segment_array | None).
        segment_array is shape (embed_window,), zero-padded if near buffer edge.
        """
        self._buffer.append(float(power_w))
        # Trim to 3× embed_window for efficiency
        if len(self._buffer) > self.embed_window * 3:
            self._buffer = self._buffer[-(self.embed_window * 3):]

        if len(self._buffer) < self.sg_window:
            return False, None

        arr      = np.array(self._buffer, dtype=np.float32)
        smoothed = savgol_filter(arr, self.sg_window, self.sg_polyord)
        deriv    = np.diff(smoothed)

        # Check last window_size derivatives
        recent = deriv[-self.window_size:]
        if np.any(np.abs(recent) >= self.scaled_threshold):
            peak_idx = len(self._buffer) - 1
            half     = self.embed_window // 2
            start    = max(0, peak_idx - half)
            end      = start + self.embed_window
            segment  = arr[start:end]
            # zero-pad if near edges
            if len(segment) < self.embed_window:
                segment = np.pad(segment,
                                 (0, self.embed_window - len(segment)),
                                 mode='constant')

            # If sample rate > 1Hz, downsample the segment to embed_window
            # by taking evenly-spaced samples (preserves transient shape)
            if self.sample_rate_hz > 1 and len(segment) > self.embed_window:
                indices = np.linspace(0, len(segment) - 1,
                                      self.embed_window, dtype=int)
                segment = segment[indices]

            return True, segment.astype(np.float32)

        return False, None

    def reset(self):
        self._buffer.clear()


class OverlapAwareNILMDetector:
    """
    Fix §2.3.2: Multi-label transient overlap handling.

    Extends NILMTransientDetector with overlap detection. When multiple
    transients arrive within a short time window, the detector:
      1. Detects the overlap condition (multiple dP/dt events within overlap_window_s)
      2. Attempts power-level subtraction using known device baselines
      3. Emits multiple candidate segments (one per suspected device)

    This addresses the NILM bottleneck where overlapping transients
    (e.g., fridge compressor + kettle at the same second) produce a
    combined waveform that fails to match any single prototype.

    Usage:
        detector = OverlapAwareNILMDetector(sample_rate_hz=10)
        detector.register_baseline("fridge", 150.0)
        detector.register_baseline("kettle", 2200.0)
        results = detector.push(power_w)
        for is_transient, segment, label_hint in results:
            if is_transient:
                embedding = cnn_encoder(segment)
    """

    def __init__(self,
                 sample_rate_hz: int = 1,
                 overlap_window_s: float = 3.0,
                 threshold: float = TRANSIENT_THRESHOLD_W,
                 embed_window: int = 128,
                 sg_window: int = SG_WINDOW,
                 sg_polyord: int = SG_POLYORD):
        """
        Args:
            sample_rate_hz:   Input sample rate in Hz.
            overlap_window_s: Time window (seconds) within which multiple
                              transients are considered overlapping.
            threshold:        Power change threshold (W) to flag a transient.
            embed_window:     Number of samples fed to the CNN encoder.
        """
        self._base = NILMTransientDetector(
            sample_rate_hz=sample_rate_hz,
            threshold=threshold,
            embed_window=embed_window,
            sg_window=sg_window,
            sg_polyord=sg_polyord,
        )
        self.sample_rate_hz   = max(1, sample_rate_hz)
        self.overlap_window_s = overlap_window_s
        self.embed_window     = embed_window
        self.threshold        = threshold

        # Known device baselines: device_name → rated_power_watts
        self._baselines: Dict[str, float] = {}

        # Recent transient event timestamps for overlap detection
        self._recent_transients: deque = deque(maxlen=20)

        # Power buffer for subtraction (mirrors _base._buffer)
        self._power_buffer: list = []

    def register_baseline(self, device_name: str, rated_watts: float) -> None:
        """Register a known device's rated power for subtraction during overlap."""
        self._baselines[device_name] = rated_watts
        logger.debug(f"Registered baseline: {device_name} = {rated_watts}W")

    def push(self, power_w: float,
             timestamp: Optional[float] = None
             ) -> List[Tuple[bool, Optional[np.ndarray], str]]:
        """
        Push one sample. Returns a list of (is_transient, segment, label_hint)
        tuples. In the common case (no overlap), returns a single element.
        When overlap is detected, returns multiple candidate segments.

        Args:
            power_w:   Power reading in watts.
            timestamp: Optional timestamp (defaults to time.time()).

        Returns:
            List of (is_transient, segment_or_None, label_hint) tuples.
            label_hint is one of: "single", "multi_device", or a device name
            when subtraction successfully isolates a known device.
        """
        ts = timestamp or time.time()
        self._power_buffer.append(float(power_w))
        if len(self._power_buffer) > self.embed_window * 3:
            self._power_buffer = self._power_buffer[-(self.embed_window * 3):]

        is_transient, segment = self._base.push(power_w)

        if not is_transient:
            return [(False, None, "")]

        # Record this transient event
        self._recent_transients.append(ts)

        # Check for overlap: count transients within overlap_window_s
        cutoff = ts - self.overlap_window_s
        recent_count = sum(1 for t in self._recent_transients if t >= cutoff)

        if recent_count <= 1 or not self._baselines:
            # Single transient — standard processing
            return [(True, segment, "single")]

        # ── Overlap detected: attempt power-level subtraction ──
        logger.info(
            f"🔀 Overlap detected: {recent_count} transients in "
            f"{self.overlap_window_s}s window. Attempting subtraction."
        )

        results: List[Tuple[bool, Optional[np.ndarray], str]] = []

        # Emit the raw combined segment as multi_device
        results.append((True, segment, "multi_device"))

        # For each known baseline, subtract it from the segment to
        # produce an isolated residual that may match the other device
        arr = np.array(self._power_buffer, dtype=np.float32)
        for device_name, rated_w in self._baselines.items():
            # Only attempt subtraction if the power level is plausibly
            # high enough to contain this device
            if segment is not None and float(segment.max()) > rated_w * 0.5:
                residual = segment - rated_w
                # Only emit if the residual still has a meaningful transient
                if np.any(np.abs(np.diff(residual)) >= self.threshold / self.sample_rate_hz):
                    # Zero-clamp negative residuals (can't have negative power)
                    residual = np.maximum(residual, 0.0).astype(np.float32)
                    results.append((True, residual, device_name))

        return results

    def reset(self):
        self._base.reset()
        self._recent_transients.clear()
        self._power_buffer.clear()


# Keep a module-level singleton for backward compatibility
nilm_detector = NILMTransientDetector()
