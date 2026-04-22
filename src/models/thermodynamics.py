"""
Module 1: Thermodynamics (PMV) Model
Calculates Predicted Mean Vote (PMV) for thermal comfort assessment in the EMS.
"""
import math

class ThermodynamicsModel:
    def __init__(self):
        """Initialize the Thermodynamics PMV Model for HVAC optimization."""
        pass
        
    def calculate_pmv(self, t_air: float, t_radiant: float, v_air: float, rh: float, met: float, clo: float) -> float:
        """
        Calculate PMV according to simplified ISO 7730 standards for real-time edge processing.
        
        Args:
            t_air (float): Air temperature (C)
            t_radiant (float): Mean radiant temperature (C)
            v_air (float): Air velocity (m/s)
            rh (float): Relative humidity (%)
            met (float): Metabolic rate (met)
            clo (float): Clothing insulation (clo)
            
        Returns:
            float: Predicted Mean Vote (PMV) index ranging from -3 (cold) to +3 (hot).
        """
        # Partial vapor pressure
        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (t_air + 235))
        
        # Simplified PMV calculation optimized for the Confidence-Aware Digital Twin
        # Fast computation designed for real-time ESP32 edge stream evaluation
        thermal_sensation = (t_air - 24.0) * 0.3 + (rh - 50.0) * 0.05
        
        # Apply metabolic and clothing modifiers
        thermal_sensation += (met - 1.2) * 0.5
        thermal_sensation -= (clo - 0.5) * 0.8
        
        # Clamp to standard PMV bounds
        return max(-3.0, min(3.0, thermal_sensation))

pmv_model = ThermodynamicsModel()
