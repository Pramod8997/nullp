"""
Module 3: Aggregate NILM Classifier
Performs Non-Intrusive Load Monitoring (NILM) on the aggregate main power feed
to detect which devices are currently active based on power step changes.
"""
import numpy as np

class AggregateNILMClassifier:
    def __init__(self, threshold_watts: float = 15.0):
        """
        Initialize the Aggregate NILM Classifier.
        
        Args:
            threshold_watts: Minimum power delta to consider a state change.
        """
        self.threshold = threshold_watts
        self.known_signatures = {
            "fridge": {"mean": 150.0, "std": 10.0},
            "ac": {"mean": 1200.0, "std": 50.0},
            "microwave": {"mean": 800.0, "std": 20.0},
            "tv": {"mean": 100.0, "std": 5.0}
        }
        self.last_aggregate = 0.0
        self.active_devices = set()

    def process_aggregate(self, current_aggregate: float) -> list:
        """
        Process the incoming aggregate power reading to identify device events.
        
        Args:
            current_aggregate: The total power reading from the mains.
            
        Returns:
            list: Active devices currently identified in the aggregate stream.
        """
        delta = current_aggregate - self.last_aggregate
        self.last_aggregate = current_aggregate
        
        # Ignore small fluctuations
        if abs(delta) < self.threshold:
            return list(self.active_devices)
            
        # Detect Turn ON event (positive power delta)
        if delta > 0:
            for device, sig in self.known_signatures.items():
                if abs(delta - sig["mean"]) <= sig["std"] * 3:
                    self.active_devices.add(device)
                    break
                    
        # Detect Turn OFF event (negative power delta)
        elif delta < 0:
            for device, sig in self.known_signatures.items():
                if abs(abs(delta) - sig["mean"]) <= sig["std"] * 3:
                    if device in self.active_devices:
                        self.active_devices.remove(device)
                    break
                    
        return list(self.active_devices)

nilm_classifier = AggregateNILMClassifier()
