"""
Module 3: Aggregate NILM — Savitzky-Golay filter + derivative transient detector.

Step 1: smooth signal with Savitzky-Golay filter.
Step 2: compute first derivative (np.diff).
Step 3: flag transient if |derivative| >= TRANSIENT_THRESHOLD_W within
        any 5-sample rolling window.
Returns the windowed signal segment centred on the transient peak for
downstream CNN encoding.
"""
import numpy as np
from scipy.signal import savgol_filter

WINDOW_SIZE           = 5       # seconds (matches 1 Hz data → 5 samples)
SG_WINDOW             = 7       # Savitzky-Golay window (must be odd, > WINDOW_SIZE)
SG_POLYORD            = 2       # polynomial order
TRANSIENT_THRESHOLD_W = 50.0   # ±50 W triggers event


class NILMTransientDetector:
    """
    Implements the SG-filter + derivative transient detector described in the
    Phase-1 implementation prompt (GAP 1).

    Usage (1 Hz loop):
        detector = NILMTransientDetector()
        is_t, segment = detector.push(power_w)
        if is_t:
            embedding = cnn_encoder(segment)
    """

    def __init__(self,
                 window_size: int   = WINDOW_SIZE,
                 sg_window:   int   = SG_WINDOW,
                 sg_polyord:  int   = SG_POLYORD,
                 threshold:   float = TRANSIENT_THRESHOLD_W,
                 embed_window: int  = 128):
        self.window_size  = window_size
        self.sg_window    = sg_window
        self.sg_polyord   = sg_polyord
        self.threshold    = threshold
        self.embed_window = embed_window   # samples fed to CNN
        self._buffer: list = []            # rolling 1 Hz power samples

    def push(self, power_w: float):
        """
        Push one 1 Hz sample.
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

        # Check last WINDOW_SIZE derivatives
        recent = deriv[-self.window_size:]
        if np.any(np.abs(recent) >= self.threshold):
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
            return True, segment.astype(np.float32)

        return False, None

    def reset(self):
        self._buffer.clear()


# Keep a module-level singleton for backward compatibility
nilm_detector = NILMTransientDetector()
