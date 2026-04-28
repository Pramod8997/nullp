import time
import logging
import pickle
import os
import yaml
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any, Optional
from src.models.thermodynamics import pmv_model

logger = logging.getLogger(__name__)

class TabularQLearningAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        rl_cfg = self.config.get("rl", {})
        self.cooldown = rl_cfg.get("cooldown_seconds", 15.0)
        self.pmv_min = rl_cfg.get("empathy_pmv_min", -0.5)
        self.pmv_max = rl_cfg.get("empathy_pmv_max", 0.5)
        self.q_table_path = rl_cfg.get("q_table_path", "backend/models/weights/q_table.pkl")
        
        self.tou_pricing = self.config.get("analytics", {}).get("tou_pricing", {})
        
        protonet_cfg = self.config.get("protonet", {})
        self.confidence_threshold = protonet_cfg.get("confidence_threshold", 0.90)
        
        safety_cfg = self.config.get("system_safety", {})
        self.max_watts = safety_cfg.get("max_aggregate_wattage", 3500.0)
        self.critical_pct = safety_cfg.get("critical_pct", 1.25)
        self.device_limits = safety_cfg.get("device_wattage_limits", {})
        
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        
        self.MAX_RL_DEVICES = 10
        self.NEVER_SHED = ["esp32_fridge"]  # Hardcoded life-critical devices
        
        # Q-table: state_hash -> {action_hash -> q_value}
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.last_action_time = 0.0
        
        self.twin = pmv_model

        if os.path.exists(self.q_table_path):
            self.load()

    def get_tou_rate(self, hour: int) -> float:
        """Returns the ToU rate for a given hour."""
        for tier, data in self.tou_pricing.items():
            if hour in data.get("hours", []):
                return data.get("rate", 0.15)
        return 0.15  # Fallback

    def get_price_bin(self, rate: float) -> int:
        """Categorize into 0: OFF_PEAK, 1: MID, 2: PEAK"""
        rates = [d["rate"] for d in self.tou_pricing.values() if "rate" in d]
        if not rates:
            return 1
        sorted_rates = sorted(rates)
        if rate <= sorted_rates[0]:
            return 0
        elif rate >= sorted_rates[-1]:
            return 2
        return 1

    def get_pmv_zone(self, pmv: float) -> int:
        """0: COLD, 1: COMFORT, 2: HOT"""
        if pmv < self.pmv_min:
            return 0
        if pmv > self.pmv_max:
            return 2
        return 1

    def get_time_of_day_bin(self, hour: int) -> int:
        """0: NIGHT (0-5), 1: MORNING (6-11), 2: DAY (12-17), 3: EVENING (18-23)"""
        return hour // 6

    def _discretize(self, state_dict: Dict[str, Any]) -> str:
        """
        State space:
        - Per-device: 4 bins (OFF=0, LOW=1, MED=2, HIGH=3)
        - Price tier: 3 bins
        - PMV zone: 3 bins
        - Time of day: 4 bins
        """
        device_states = []
        # Sort devices to ensure consistent state hashing
        for dev_id in sorted(state_dict.get("devices", {}).keys()):
            pct = state_dict["devices"][dev_id]  # Expected to be percentage of rated
            if pct <= 0.05:
                bin_val = 0
            elif pct <= 0.33:
                bin_val = 1
            elif pct <= 0.66:
                bin_val = 2
            else:
                bin_val = 3
            device_states.append(f"{dev_id}:{bin_val}")
            
        parts = [
            f"price:{state_dict.get('price_tier', 1)}",
            f"pmv:{state_dict.get('pmv_zone', 1)}",
            f"tod:{state_dict.get('tod', 2)}",
            "|".join(device_states)
        ]
        return "::".join(parts)

    def act(self, state_dict: Dict[str, Any], pmv: float, confidence: float, classified_device: str) -> str:
        # Gate 1: confidence must be >= threshold
        if confidence < self.confidence_threshold:
            return "DEFER"  # do nothing when uncertain
            
        # Gate 2: PMV empathy gate (Category A bounds: -0.5 to 0.5)
        if pmv < self.pmv_min or pmv > self.pmv_max:
            if pmv < self.pmv_min:
                return "SCHEDULE_HVAC"  # force heating
            else:
                return "SHED_HVAC"      # force cooling
                
        # Gate 3: check cooldown
        if time.time() - self.last_action_time < self.cooldown:
            return "DEFER"

        state_key = self._discretize(state_dict)
        
        # Valid actions for this device
        valid_actions = ["DEFER"]
        if classified_device not in self.NEVER_SHED:
            valid_actions.append("SHED")
            valid_actions.append("SCHEDULE")

        # Explore vs Exploit
        if np.random.rand() < self.epsilon:
            action = str(np.random.choice(valid_actions))
        else:
            # Exploit
            best_action = "DEFER"
            best_q = float('-inf')
            for a in valid_actions:
                q = self.q_table[state_key][a]
                if q > best_q:
                    best_q = q
                    best_action = a
            action = best_action

        # Update last action time if we took a real action
        if action != "DEFER":
            self.last_action_time = time.time()
            
        return action

    def compute_reward(self, prev_state: Dict[str, Any], action: str, next_state: Dict[str, Any], 
                       pmv: float, current_watts: float, tou_rate: float, confidence: float) -> float:
        energy_reward = -current_watts * tou_rate / 1000.0  # cost in kWh
        pmv_penalty = -5.0 * self.twin.pmv_penalty(pmv)     # heavy comfort penalty
        safety_bonus = 0.0 if current_watts < self.max_watts else -10.0
        return energy_reward + pmv_penalty + safety_bonus

    def update(self, state_dict: Dict[str, Any], action: str, reward: float, next_state_dict: Dict[str, Any]) -> None:
        state_key = self._discretize(state_dict)
        next_state_key = self._discretize(next_state_dict)
        
        best_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
        
        # Log to CSV
        self.log_action(state_key, action, reward, next_state_key)

    def log_action(self, state: str, action: str, reward: float, next_state: str) -> None:
        try:
            with open("rl_action_log.csv", "a") as f:
                f.write(f"{time.time()},{state},{action},{reward},{next_state}\n")
        except Exception as e:
            logger.error(f"Failed to log RL action: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, "wb") as f:
            # Convert defaultdict back to dict for pickling
            pickle.dump({k: dict(v) for k, v in self.q_table.items()}, f)

    def load(self) -> None:
        try:
            with open(self.q_table_path, "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    self.q_table[k].update(v)
            logger.info(f"Loaded Q-table from {self.q_table_path}")
        except Exception as e:
            logger.warning(f"Could not load Q-table: {e}")
