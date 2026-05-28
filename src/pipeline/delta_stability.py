"""
Delta Stability Analyzer — DFD Level-2 Process P4.3 (Signature Buffer D4.1).

When ProtoNet/OpenMax routes a signature as UNKNOWN, this module determines
whether the unknown is:
  - STABLE:    same signature appearing repeatedly → route to user labeling (P4.5)
  - TRANSIENT: one-off noise spike → log silently (P4.4)

This prevents alert fatigue from transient electrical noise.

Production fix: self._temp_log uses deque(maxlen=1000) to prevent unbounded
memory growth from continuous line noise in real-world deployments.
"""
import uuid
import hashlib
import numpy as np
from collections import deque
from typing import Tuple, Optional

STABILITY_WINDOW = 10    # number of recent unknown signatures to compare
STABILITY_THRESH = 15.0  # squared Euclidean distance for "same device"
MIN_STABLE_COUNT = 3     # must appear >= 3 times to be flagged as stable unknown
TEMP_LOG_CAPACITY = 1000 # max anomaly log entries before eviction


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

        self._buffer   = deque(maxlen=window)          # Signature Buffer D4.1
        self._temp_log = deque(maxlen=TEMP_LOG_CAPACITY)  # Fixed-capacity anomaly log

        # Track stable cluster state for background pseudo-labeling (Task 5)
        self._last_stable_mean: Optional[np.ndarray] = None
        self._last_stable_hits: int = 0

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
            self._last_stable_mean = cluster_mean
            self._last_stable_hits = close_count
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

    def get_stable_cluster(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Return the last detected stable cluster mean and hit count.
        Used by the orchestrator for background pseudo-labeling (Task 5).

        Returns:
            (cluster_mean_array, hit_count) or (None, 0) if no stable cluster.
        """
        return self._last_stable_mean, self._last_stable_hits

    @staticmethod
    def quantized_cluster_hash(mean_embedding: np.ndarray,
                               decimals: int = 3) -> str:
        """
        Generate a deterministic hash from a mean embedding vector,
        quantized to `decimals` decimal places to prevent sensor drift
        and line noise from creating duplicate database rows.

        Args:
            mean_embedding: (EMBED_DIM,) numpy array.
            decimals: rounding precision (default 3).

        Returns:
            Hex digest string suitable for database UNIQUE constraint.
        """
        quantized = np.round(mean_embedding, decimals).astype(np.float32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]

    def recent_log(self, n: int = 10) -> list:
        """Return last n anomaly log entries as a standard Python list."""
        return list(self._temp_log)[-n:]

    def reset(self):
        self._buffer.clear()
        self._temp_log.clear()
        self._last_stable_mean = None
        self._last_stable_hits = 0
