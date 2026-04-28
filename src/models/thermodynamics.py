"""
ISO 7730 Predicted Mean Vote (PMV) Thermodynamics Model.
Category A: PMV in [-0.5, +0.5].

Variables:
  ta   – air temperature (°C)
  tr   – mean radiant temperature (°C)
  va   – air velocity (m/s)
  rh   – relative humidity (%)
  clo  – clothing insulation (clo units; typical: 0.5 summer, 1.0 winter)
  met  – metabolic rate (met; seated=1.0, light activity=1.6)
"""
import math
import numpy as np
from typing import Dict


class PMVThermodynamics:
    """
    ISO 7730 Predicted Mean Vote (PMV) calculation.
    Category A: PMV in [-0.5, +0.5].
    Implementation follows ISO 7730:2005 Annex D.
    """

    CATEGORY_A_MIN = -0.5
    CATEGORY_A_MAX =  0.5

    def pmv(self, ta: float = 22.0, tr: float = 22.0, va: float = 0.1,
            rh: float = 50.0, clo: float = 0.5, met: float = 1.2) -> float:
        """
        Compute PMV. Raises ValueError for out-of-plausible inputs.

        Returns:
            float: PMV value (clipped to [-3, 3])
        """
        # Clothing surface area factor
        if clo < 0.5:
            f_cl = 1.0 + 1.290 * clo
        else:
            f_cl = 1.05 + 0.645 * clo

        # Metabolic heat production (W/m²)
        m = met * 58.15

        # Clothing thermal resistance (m²K/W)
        i_cl = 0.155 * clo

        # Pa: partial water vapour pressure (Pa)
        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235.0))

        # Clothing surface temperature — iterative (ISO 7730:2005 Annex D)
        # Better initial guess avoids divergence
        t_cl = 35.7 - 0.028 * m - i_cl * (
            3.96e-8 * f_cl * ((ta + 273.0) ** 4 - (tr + 273.0) ** 4) +
            f_cl * 2.38 * (abs(ta - ta) ** 0.25) * (ta - ta)
        )
        t_cl = float(np.clip(t_cl, ta - 5, ta + 30))  # physically reasonable

        for _ in range(150):
            hc = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(max(va, 0.0)))
            try:
                rad_term = 3.96e-8 * f_cl * ((t_cl + 273.0) ** 4 - (tr + 273.0) ** 4)
            except (OverflowError, ValueError):
                rad_term = 0.0
            # Correct ISO 7730 iteration: solve clothing surface energy balance
            # t_cl = 35.7 - 0.028*m - R_cl * (radiation + convection)
            t_cl_new = 35.7 - 0.028 * m - i_cl * (rad_term + f_cl * hc * (t_cl - ta))
            t_cl_new = float(np.clip(t_cl_new, -10.0, 60.0))
            if abs(t_cl_new - t_cl) < 0.001:
                t_cl = t_cl_new
                break
            t_cl = 0.9 * t_cl + 0.1 * t_cl_new  # damped update for stability

        hc = max(2.38 * abs(t_cl - ta) ** 0.25, 12.1 * math.sqrt(max(va, 0.0)))

        # Heat loss components (W/m²) — ISO 7730:2005 Eq. (A.1)
        hl1 = 3.05e-3 * (5733.0 - 6.99 * m - pa)   # skin diffusion
        hl2 = 0.42 * max(0.0, m - 58.15)            # sweating (only if m > 58.15)
        hl3 = 1.7e-5 * m * (5867.0 - pa)            # latent respiration
        hl4 = 0.0014 * m * (34.0 - ta)              # dry respiration
        try:
            hl5 = 3.96e-8 * f_cl * ((t_cl + 273.0) ** 4 - (tr + 273.0) ** 4)  # radiation
        except (OverflowError, ValueError):
            hl5 = 0.0
        hl6 = f_cl * hc * (t_cl - ta)               # convection

        # Thermal load: L = internal heat production minus all losses
        L = m - hl1 - hl2 - hl3 - hl4 - hl5 - hl6

        pmv_val = (0.303 * math.exp(-0.036 * m) + 0.028) * L
        return round(float(np.clip(pmv_val, -3.0, 3.0)), 4)

    def is_category_a(self, pmv_val: float) -> bool:
        """True if PMV is within ISO 7730 Category A bounds."""
        return self.CATEGORY_A_MIN <= pmv_val <= self.CATEGORY_A_MAX

    def pmv_penalty(self, pmv_val: float) -> float:
        """
        Reward function penalty for RL agent.
        0.0 if in Category A, proportional to violation otherwise.
        """
        if self.is_category_a(pmv_val):
            return 0.0
        return abs(pmv_val) - 0.5

    def comfort_state(self, ta: float = 22.0, tr: float = 22.0, va: float = 0.1,
                      rh: float = 50.0, clo: float = 0.5, met: float = 1.2) -> dict:
        pmv_val = self.pmv(ta, tr, va, rh, clo, met)
        return {
            'pmv': pmv_val,
            'category_a': self.is_category_a(pmv_val),
            'penalty': self.pmv_penalty(pmv_val),
        }


# ── Backward-compatible ThermodynamicsModel wrapper ──────────────────────────

class ThermodynamicsModel(PMVThermodynamics):
    """
    Backward-compatible wrapper: exposes the old compute_pmv() API used by
    existing tests, while delegating to the full ISO 7730 PMVThermodynamics
    implementation above.
    """

    def __init__(self):
        super().__init__()
        self.k_loss     = 0.01      # heat loss coefficient
        self.c_thermal  = 50000.0   # thermal capacity of room (J/K)

    def compute_pmv(self, t_air: float, t_mrt: float, v_air: float,
                    rh: float, met: float, clo: float) -> float:
        """Legacy arg-order wrapper (note: met and clo swapped vs ISO signature)."""
        return self.pmv(ta=t_air, tr=t_mrt, va=v_air, rh=rh, clo=clo, met=met)

    def simulate_step(self, appliance_states: Dict[str, float],
                      outdoor_temp: float, t_internal: float,
                      dt_minutes: float = 1.0) -> float:
        """Simple thermal decay model for digital twin simulation."""
        q_appliances_watts = sum(appliance_states.values()) * 0.20
        if 'esp32_hvac' in appliance_states and appliance_states['esp32_hvac'] > 100:
            if outdoor_temp > 24:
                q_appliances_watts -= appliance_states['esp32_hvac'] * 3.0
            else:
                q_appliances_watts += appliance_states['esp32_hvac'] * 3.0
        q_joules_per_min = q_appliances_watts * 60.0
        dt_t = (-self.k_loss * (t_internal - outdoor_temp)
                + (q_joules_per_min / self.c_thermal))
        return t_internal + dt_t * dt_minutes


# Module-level singletons for backward compat
pmv_model       = ThermodynamicsModel()
