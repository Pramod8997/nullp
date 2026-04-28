"""
Delta Stability Analyzer — DFD Level-2 Process P4.3 (Signature Buffer D4.1).

When ProtoNet/OpenMax routes a signature as UNKNOWN, this module determines
whether the unknown is:
  - STABLE:    same signature appearing repeatedly → route to user labeling (P4.5)
  - TRANSIENT: one-off noise spike → log silently (P4.4)

This prevents alert fatigue from transient electrical noise.
"""
import uuid
import numpy as np
from collections import deque
from typing import Tuple, Optional

STABILITY_WINDOW = 10    # number of recent unknown signatures to compare
STABILITY_THRESH = 15.0  # squared Euclidean distance for "same device"
MIN_STABLE_COUNT = 3     # must appear >= 3 times to be flagged as stable unknown


class DeltaStabilityAnalyzer:
    """
    Implements DFD Level-2 Process P4.3 (Signature Buffer D4.1).

    Usage:
        da = DeltaStabilityAnalyzer()
        result, cluster_mean = da.push(embedding, timestamp=time.time())
        if result == 'stable':
            # emit label-request event to dashboard
        else:
            # log silently
    """

    def __init__(self,
                 window:     int   = STABILITY_WINDOW,
                 threshold:  float = STABILITY_THRESH,
                 min_count:  int   = MIN_STABLE_COUNT,
                 # Legacy param names for backward compat
                 buffer_size:         Optional[int]   = None,
                 stability_threshold:  Optional[float] = None,
                 min_occurrences:      Optional[int]   = None):
        # Resolve legacy kwargs
        if buffer_size is not None:
            window = buffer_size
        if stability_threshold is not None:
            threshold = stability_threshold
        if min_occurrences is not None:
            min_count = min_occurrences

        self.window    = window
        self.threshold = threshold
        self.min_count = min_count

        self._buffer   = deque(maxlen=window)   # Signature Buffer D4.1
        self._temp_log = []                      # Temporary Anomaly Log P4.4

    def push(self, embedding: np.ndarray,
             timestamp: Optional[float] = None) -> Tuple[str, Optional[np.ndarray]]:
        """
        Push one unknown embedding.

        Args:
            embedding:  (EMBED_DIM,) numpy array of the unknown signature.
            timestamp:  optional float timestamp.

        Returns:
            ('stable', cluster_mean_array) | ('transient', None)
        """
        self._buffer.append((embedding.copy(), timestamp))

        if len(self._buffer) < self.min_count:
            self._temp_log.append({'embedding': embedding, 'ts': timestamp,
                                   'reason': 'buffer_too_small'})
            return 'transient', None

        embeddings = np.stack([e for e, _ in self._buffer])
        dists      = np.sum((embeddings - embedding) ** 2, axis=1)
        close_count = int(np.sum(dists <= self.threshold))

        if close_count >= self.min_count:
            cluster_mean = embeddings[dists <= self.threshold].mean(axis=0)
            return 'stable', cluster_mean
        else:
            self._temp_log.append({'embedding': embedding, 'ts': timestamp,
                                   'reason': 'isolated_transient'})
            return 'transient', None

    def check(self, embedding: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Legacy API: returns (is_stable: bool, temp_id_if_unstable | None).
        Used by existing tests. Internally calls push().
        """
        result, _ = self.push(embedding)
        if result == 'stable':
            return True, None
        else:
            temp_id = f"Unknown_{uuid.uuid4().hex[:6]}"
            return False, temp_id

    def recent_log(self, n: int = 10) -> list:
        return self._temp_log[-n:]

    def reset(self):
        self._buffer.clear()
        self._temp_log.clear()
