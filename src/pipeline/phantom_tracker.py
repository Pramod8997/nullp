"""
Module 6: Micro-Load (Phantom) Tracker
Identifies and tracks continuous small power draws (vampire/phantom loads)
that occur when devices are nominally turned off.
"""

class PhantomTracker:
    def __init__(self, baseline_threshold_watts: float = 5.0):
        """
        Initialize the Micro-Load Tracker.
        
        Args:
            baseline_threshold_watts: Maximum power draw considered to be a phantom load.
        """
        self.baseline_threshold = baseline_threshold_watts
        self.phantom_loads = {}
        
    def track(self, device_id: str, power_draw: float, is_nominally_off: bool):
        """
        Track potential phantom load for a device.
        
        Args:
            device_id: Identifier for the device.
            power_draw: Current power draw in watts.
            is_nominally_off: Whether the EMS state registers the device as 'off'.
        """
        # If device is 'off' but drawing a small amount of power
        if is_nominally_off and 0 < power_draw <= self.baseline_threshold:
            # Update the smoothed phantom load estimate (exponential moving average)
            if device_id in self.phantom_loads:
                self.phantom_loads[device_id] = (0.9 * self.phantom_loads[device_id]) + (0.1 * power_draw)
            else:
                self.phantom_loads[device_id] = power_draw
                
        # If device draws 0.0W exactly, it's fully disconnected
        elif power_draw == 0.0:
            self.phantom_loads[device_id] = 0.0
            
    def get_total_phantom_load(self) -> float:
        """Calculate the current total phantom load across the system."""
        return sum(self.phantom_loads.values())
        
    def get_worst_offenders(self, top_n: int = 3) -> list:
        """
        Get the devices contributing most to the phantom load.
        
        Returns:
            list: Tuples of (device_id, phantom_watts) sorted by magnitude.
        """
        sorted_loads = sorted(self.phantom_loads.items(), key=lambda x: x[1], reverse=True)
        return sorted_loads[:top_n]

phantom_tracker = PhantomTracker()
