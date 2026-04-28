"""
Delta Stability Analyzer (Level 2 DFD P4.3)
Checks if an unknown device signature is stable enough for user labeling
or if it's a transient anomaly to be discarded.
"""
import uuid
import numpy as np
from collections import deque
from typing import Tuple, Optional


class DeltaStabilityAnalyzer:
    def __init__(self, buffer_size: int = 10, stability_threshold: float = 3.0,
                 min_occurrences: int = 3):
        """
        Args:
            buffer_size: How many recent unknown embeddings to retain.
            stability_threshold: Max L2 distance to consider two embeddings as 'same signature'.
            min_occurrences: How many close matches needed to declare a stable unknown.
        """
        self.buffer: deque = deque(maxlen=buffer_size)
        self.stability_threshold = stability_threshold
        self.min_occurrences = min_occurrences

    def check(self, embedding: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_stable, temp_id_if_unstable).
        - If the embedding is within stability_threshold of embeddings 
          already in the buffer >= min_occurrences times → stable unknown
        - Otherwise → transient noise → log as Unknown_X and return False
        """
        distances = [np.linalg.norm(embedding - e) for e in self.buffer]
        close_count = sum(1 for d in distances if d < self.stability_threshold)
        self.buffer.append(embedding.copy())

        if close_count >= self.min_occurrences:
            return True, None  # stable: route to user labeling
        else:
            temp_id = f"Unknown_{uuid.uuid4().hex[:6]}"
            return False, temp_id  # transient: log and discard
