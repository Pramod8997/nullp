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

        # Epsilon decay: explore aggressively at start, converge over time
        self.epsilon_start = rl_cfg.get("epsilon_start", 0.3)
        self.epsilon_end = rl_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = rl_cfg.get("epsilon_decay", 0.999)
        self.epsilon = self.epsilon_start
        
        self.MAX_RL_DEVICES = 10

        # Load NEVER_SHED from config: any device with tier0: true is unshedable
        devices_cfg = self.config.get("devices", {})
        self.NEVER_SHED = [
            name for name, cfg in devices_cfg.items()
            if isinstance(cfg, dict) and cfg.get("tier0", False)
        ]
        # Always protect fridge as a fallback even if config doesn't set tier0
        if not any("fridge" in d for d in self.NEVER_SHED):
            self.NEVER_SHED.append("esp32_fridge")
        
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

    def _discretize(self, state_dict: Dict[str, Any], classified_device: str = "") -> str:
        """
        Aggregate state space (Phase 2 fix — WS-3.3):
        - Total load bin: 4 bins (OFF=0, LOW=1, MED=2, HIGH=3)
        - Active device count bin: 4 bins
        - Price tier: 3 bins
        - PMV zone: 3 bins
        - Time of day: 4 bins
        - Target appliance class (Bug 1.3 fix: prevents cross-device policy bleed)
        """
        devices = state_dict.get("devices", {})

        # Aggregate load: average pct-of-rated across all devices
        if devices:
            total_pct = sum(devices.values()) / len(devices)
        else:
            total_pct = 0.0
        total_bin = min(3, int(total_pct * 4))

        # Active device count (those above 5% of rated)
        active_count = sum(1 for v in devices.values() if v > 0.05)
        active_bin = min(3, active_count // 3)

        # Bug 1.3 fix: Include target appliance in state hash so the agent
        # doesn't blindly apply an HVAC policy to a fridge (or vice versa)
        return (f"load:{total_bin}::active:{active_bin}"
                f"::price:{state_dict.get('price_tier', 1)}"
                f"::pmv:{state_dict.get('pmv_zone', 1)}"
                f"::tod:{state_dict.get('tod', 2)}"
                f"::dev:{classified_device}")

    def act(self, state_dict: Dict[str, Any], pmv: float, confidence: float,
            classified_device: str, min_confidence: float = None) -> str:
        """Act based on current state. Returns action string."""
        # Gate 1: confidence gate — block RL when uncertain (FR3 / SRS NF-Accuracy)
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        if confidence < threshold:
            return "DEFER"  # no_op when uncertain
            
        # Gate 2: PMV empathy gate (Category A bounds: -0.5 to 0.5)
        # Bug 2.3 fix: Only apply empathy gate when the ticking device is HVAC,
        # otherwise a TV or Kettle tick would spam HVAC commands every second
        if classified_device and "hvac" in classified_device.lower():
            if pmv < self.pmv_min or pmv > self.pmv_max:
                if pmv < self.pmv_min:
                    return "SCHEDULE_HVAC"  # force heating
                else:
                    return "SHED_HVAC"      # force cooling
                
        # Gate 3: check cooldown
        if time.time() - self.last_action_time < self.cooldown:
            return "DEFER"

        state_key = self._discretize(state_dict, classified_device)
        
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
                       pmv: float, current_watts: float, tou_rate: float, confidence: float,
                       aggregate_watts: float = 0.0) -> float:
        # Bug 1.2 fix: Use projected wattage based on action taken.
        # If agent chose SHED, the device is off → 0W cost. Otherwise use actual watts.
        projected_watts = 0.0 if action == "SHED" else current_watts
        energy_reward = -projected_watts * tou_rate / 1000.0  # cost in kWh
        pmv_penalty = -5.0 * self.twin.pmv_penalty(pmv)     # heavy comfort penalty
        # Audit fix 2.2: Use aggregate house load for safety penalty, not single device
        safety_bonus = 0.0 if aggregate_watts < self.max_watts else -10.0
        return energy_reward + pmv_penalty + safety_bonus

    def update(self, state_dict: Dict[str, Any], action: str, reward: float,
               next_state_dict: Dict[str, Any], classified_device: str = "") -> None:
        state_key = self._discretize(state_dict, classified_device)
        next_state_key = self._discretize(next_state_dict, classified_device)
        
        best_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

        # Epsilon decay after each update
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Log to CSV (synchronous — caller should use log_action_async from async context)
        self._log_action_sync(state_key, action, reward, next_state_key)

    def _log_action_sync(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Synchronous file write — safe to call from sync context or via asyncio.to_thread."""
        try:
            import fcntl
            with open("rl_action_log.csv", "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(f"{time.time()},{state},{action},{reward},{next_state}\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except ImportError:
            # fcntl not available (Windows) — fall back to unprotected write
            try:
                with open("rl_action_log.csv", "a") as f:
                    f.write(f"{time.time()},{state},{action},{reward},{next_state}\n")
            except Exception as e:
                logger.error(f"Failed to log RL action: {e}")
        except Exception as e:
            logger.error(f"Failed to log RL action: {e}")

    async def log_action_async(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Non-blocking RL log — runs sync file I/O in a thread to avoid stalling the event loop."""
        import asyncio
        await asyncio.to_thread(self._log_action_sync, state, action, reward, next_state)

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


# ── Alias ─────────────────────────────────────────────────────────────────────
# QLearningAgent is the canonical name used in run_pipeline.py and tests.
QLearningAgent = TabularQLearningAgent


# ── GAP 8: Policy Promotion Gate ──────────────────────────────────────────────

class PolicyPromotionGate:
    """
    Tracks validation episodes run in the digital twin sandbox.
    A policy is 'promoted' to live relay control only after completing
    MIN_VALIDATION_EPISODES without exceeding the cumulative PMV penalty budget.

    Usage:
        gate = PolicyPromotionGate()
        gate.record_twin_episode(pmv_penalty=thermo.pmv_penalty(pmv))
        if gate.is_promoted:
            issue_relay_command(action)
        else:
            # shadow mode: run in digital twin only
    """

    MIN_VALIDATION_EPISODES = 50
    PMV_PENALTY_LIMIT       = 0.5   # max allowed cumulative PMV penalty

    def __init__(self):
        self._val_episodes:   int   = 0
        self._cumulative_pmv: float = 0.0
        self._promoted:       bool  = False

    def record_twin_episode(self, pmv_penalty: float) -> None:
        """Record one digital-twin validation episode."""
        self._val_episodes   += 1
        self._cumulative_pmv += pmv_penalty

    @property
    def is_promoted(self) -> bool:
        """True once >= 50 validation episodes with acceptable PMV penalty."""
        if self._promoted:
            return True
        if (self._val_episodes >= self.MIN_VALIDATION_EPISODES
                and self._cumulative_pmv <= self.PMV_PENALTY_LIMIT):
            self._promoted = True
        return self._promoted

    def reset(self) -> None:
        self._val_episodes   = 0
        self._cumulative_pmv = 0.0
        self._promoted       = False
