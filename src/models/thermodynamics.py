"""
Module 1: Thermodynamics (PMV) Model
Calculates Predicted Mean Vote (PMV) using the full ISO 7730 Fanger equation.
"""
import math
import numpy as np
from typing import Dict

class ThermodynamicsModel:
    def __init__(self):
        """Initialize the Thermodynamics PMV Model for HVAC optimization."""
        self.k_loss = 0.01 # heat loss coefficient
        self.c_thermal = 50000.0 # thermal capacity of room (J/K)
        
    def compute_pmv(self, t_air: float, t_mrt: float, v_air: float, rh: float, clo: float, met: float) -> float:
        """
        Full Fanger PMV equation according to ISO 7730 / ASHRAE 55.
        
        Args:
            t_air: air temperature (°C)
            t_mrt: mean radiant temperature (°C)  
            v_air: relative air velocity (m/s)
            rh: relative humidity (%)
            clo: clothing insulation (1.0 = typical indoor clothing)
            met: metabolic rate (1.2 = seated light work)
            
        Returns:
            float: Predicted Mean Vote (PMV) index ranging from -3 (cold) to +3 (hot).
        """
        # Water vapor partial pressure (Pa) using Antoine approximation
        pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (t_air + 235.0))
        
        icl = 0.155 * clo  # Clothing thermal resistance (m²·K/W)
        M = met * 58.15    # Metabolic rate (W/m²)
        W = 0.0            # External work (W/m²)
        MW = M - W         # Internal heat production
        
        # Clothing area factor
        if icl <= 0.078:
            fcl = 1.00 + 1.290 * icl
        else:
            fcl = 1.05 + 0.645 * icl
            
        # Iteratively solve for clothing surface temperature tcl
        # Initial guess
        tcl = t_air + (35.5 - t_air) / (3.5 * (6.45 * icl + 0.1))
        
        for _ in range(150):
            tcl_old = tcl
            
            # Convective heat transfer coefficient
            hcn = 2.38 * abs(tcl - t_air) ** 0.25  # natural convection
            hcf = 12.1 * math.sqrt(max(v_air, 0.0))  # forced convection
            hc = max(hcn, hcf)
            
            # New estimate of tcl
            try:
                rad_term = 3.96e-8 * fcl * ((tcl_old + 273.0) ** 4 - (t_mrt + 273.0) ** 4)
            except (OverflowError, ValueError):
                rad_term = 0.0
            conv_term = fcl * hc * (tcl_old - t_air)
            tcl = 35.7 - 0.028 * MW - icl * (rad_term + conv_term)
            
            # Clamp to physically reasonable range
            tcl = max(min(tcl, 60.0), -10.0)
            
            if abs(tcl - tcl_old) < 0.001:
                break
        
        # Recompute hc with final tcl
        hcn = 2.38 * abs(tcl - t_air) ** 0.25
        hcf = 12.1 * math.sqrt(v_air)
        hc = max(hcn, hcf)
        
        # Heat loss components
        # Skin diffusion
        hl1 = 3.05e-3 * (5733.0 - 6.99 * MW - pa)
        # Sweating
        hl2 = max(0.0, 0.42 * (MW - 58.15))
        # Latent respiration
        hl3 = 1.7e-5 * M * (5867.0 - pa)
        # Dry respiration
        hl4 = 0.0014 * M * (34.0 - t_air)
        # Radiation
        hl5 = 3.96e-8 * fcl * ((tcl + 273.0) ** 4 - (t_mrt + 273.0) ** 4)
        # Convection
        hl6 = fcl * hc * (tcl - t_air)
        
        # Thermal sensation transfer coefficient
        ts = 0.303 * math.exp(-0.036 * M) + 0.028
        
        # PMV = ts * (internal heat - all losses)
        pmv = ts * (MW - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        
        return float(np.clip(pmv, -3.0, 3.0))

    def is_category_a(self, pmv: float) -> bool:
        """Returns True if PMV is within ISO 7730 Category A bounds (-0.5 to 0.5)"""
        return -0.5 <= pmv <= 0.5

    def pmv_penalty(self, pmv: float) -> float:
        """Reward penalty: 0 inside Category A, linear outside"""
        if self.is_category_a(pmv):
            return 0.0
        return abs(pmv) - 0.5

    def simulate_step(self, appliance_states: Dict[str, float], outdoor_temp: float,
                      t_internal: float, dt_minutes: float = 1.0) -> float:
        """
        Simple thermal decay model.
        - Each ON appliance adds heat proportional to its wattage * efficiency
        - Thermal decay: dT/dt = -k*(T_internal - T_outdoor) + Q_appliances/C_thermal
        
        Args:
            appliance_states: dict mapping device_id to current wattage
            outdoor_temp: ambient outdoor temperature (°C)
            t_internal: current internal temperature (°C)
            dt_minutes: simulation step time in minutes
            
        Returns:
            new t_internal after dt_minutes
        """
        # Calculate total heat added by appliances (Joules per minute)
        # Assuming average efficiency where ~20% of power becomes heat, except heating devices
        q_appliances_watts = sum(appliance_states.values()) * 0.20 
        
        # HVAC adds or removes massive heat
        if 'esp32_hvac' in appliance_states and appliance_states['esp32_hvac'] > 100:
            # Assuming it's cooling if outdoor is hot, heating if outdoor is cold
            if outdoor_temp > 24:
                q_appliances_watts -= appliance_states['esp32_hvac'] * 3.0 # COP = 3
            else:
                q_appliances_watts += appliance_states['esp32_hvac'] * 3.0
                
        q_appliances_joules_per_min = q_appliances_watts * 60.0
        
        # Update temp using Euler integration
        dt_t = -self.k_loss * (t_internal - outdoor_temp) + (q_appliances_joules_per_min / self.c_thermal)
        new_temp = t_internal + dt_t * dt_minutes
        
        return new_temp

# Keep existing pmv_model variable for backward compatibility in case it's used directly
pmv_model = ThermodynamicsModel()
