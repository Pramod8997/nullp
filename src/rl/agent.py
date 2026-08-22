import math
from datetime import date
import time
import logging
import pickle
import os
import yaml
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, List
from src.models.thermodynamics import pmv_model
from src.rl.dqn_agent import ReplayBuffer, DQNAgent, Experience

__all__ = [
    "TabularQLearningAgent",
    "QLearningAgent",
    "PolicyPromotionGate",
    "load_config",
    "DEVICE_LOCKOUT_SECONDS",
    "ReplayBuffer",
    "DQNAgent",
    "Experience",
]

logger = logging.getLogger(__name__)

# ── Production Constants ──
DEVICE_LOCKOUT_SECONDS = 300.0  # 5-minute per-device anti-thrashing window

# Safety-critical appliance identifiers that must never be shed,
# regardless of config. Both esp32_ and node_ prefixes are covered.
_HARDCODED_SAFETY_CRITICAL = [
    "esp32_fridge", "node_fridge",
    "esp32_freezer", "node_freezer",
    "esp32_pc", "node_pc",
]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    for p in ["config.yaml", "config/config.yaml"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {
        "rl": {
            "cooldown_seconds": 15.0,
            "empathy_pmv_min": -0.5,
            "empathy_pmv_max": 0.5,
            "epsilon_start": 0.3,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "q_table_path": "backend/models/weights/q_table.pkl",
        },
        "devices": {
            "node_fridge": {"tier0": True},
            "esp32_fridge": {"tier0": True},
            "node_freezer": {"tier0": True},
            "esp32_freezer": {"tier0": True},
            "node_pc": {"tier0": True},
            "esp32_pc": {"tier0": True},
        },
        "system_safety": {
            "max_aggregate_wattage": 3500.0,
            "critical_pct": 1.25,
            "device_wattage_limits": {},
        },
        "analytics": {
            "tou_pricing": {},
        },
        "protonet": {
            "confidence_threshold": 0.90,
        },
    }


class RLActionResult(str):
    def __new__(cls, value="DEFER", blocked_by_cooldown=False, blocked_by_tier0=False,
                blocked_by_pmv_empathy=False, command=None, device=None, event_type=None):
        obj = super().__new__(cls, value)
        obj.blocked_by_cooldown = blocked_by_cooldown
        obj.blocked_by_tier0 = blocked_by_tier0
        obj.blocked_by_pmv_empathy = blocked_by_pmv_empathy
        obj.command = command if command is not None else value
        obj.device = device
        obj.event_type = event_type
        return obj

    def __await__(self):
        async def _coro():
            return self
        return _coro().__await__()


# Alias for backward compatibility
RLAction = RLActionResult


class TabularQLearningAgent:
    def __init__(self, config_path: str = "config/config.yaml",
                 config: Optional[Dict[str, Any]] = None,
                 alpha: Optional[float] = None,
                 gamma: Optional[float] = None,
                 epsilon: Optional[float] = None,
                 epsilon_decay: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 cooldown_s: Optional[float] = None,
                 **kwargs):
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or load_config()
        else:
            self.config = load_config()

        rl_cfg = self.config.get("rl", {})
        self.cooldown = cooldown_s if cooldown_s is not None else rl_cfg.get("cooldown_seconds", 15.0)
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

        self.alpha = alpha if alpha is not None else 0.1
        self.gamma = gamma if gamma is not None else 0.99

        # Epsilon decay: explore aggressively at start, converge over time
        self.epsilon_start = rl_cfg.get("epsilon_start", 0.3)
        self.epsilon_end = epsilon_min if epsilon_min is not None else rl_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else rl_cfg.get("epsilon_decay", 0.999)
        self.epsilon = epsilon if epsilon is not None else self.epsilon_start
        self.epsilon_min = self.epsilon_end

        self.MAX_RL_DEVICES = 10

        # ── Appliance Blacklist (NEVER_SHED) ──
        devices_cfg = self.config.get("devices", {})
        self.NEVER_SHED = [
            name for name, cfg in devices_cfg.items()
            if isinstance(cfg, dict) and cfg.get("tier0", False)
        ]
        for device in _HARDCODED_SAFETY_CRITICAL:
            if device not in self.NEVER_SHED:
                self.NEVER_SHED.append(device)
        if not any("fridge" in d for d in self.NEVER_SHED):
            self.NEVER_SHED.append("esp32_fridge")

        # Q-table: state_hash -> {action_hash -> q_value}
        self.q_table = defaultdict(lambda: defaultdict(float))

        # ── NTP-Resilient Per-Device Anti-Thrashing Lockout ──
        self._device_lockout: Dict[str, float] = {}
        self.lockout_duration = self.cooldown

        self.last_action_time = 0.0
        # Episode-level epsilon decay: one decay per calendar day
        self._last_decay_date: Optional[str] = None
        self.twin = pmv_model

        # Shadow gate improvement tracker
        self._shadow_consecutive_improvements: int = 0
        self._shadow_promoted: bool = False

        if os.path.exists(self.q_table_path):
            self.load()

    def set_q(self, state: str, action: str, value: float) -> None:
        self.q_table[state][action] = float(value)

    def get_q(self, state: str, action: str) -> float:
        return float(self.q_table[state][action])

    def select_action(self, state: str, valid_actions: Optional[List[str]] = None) -> str:
        if valid_actions is None:
            if state in self.q_table and len(self.q_table[state]) > 0:
                valid_actions = list(self.q_table[state].keys())
            else:
                valid_actions = ["ON", "OFF"]
        if np.random.rand() < self.epsilon:
            return str(np.random.choice(valid_actions))
        else:
            best_action = valid_actions[0]
            best_q = float('-inf')
            for a in valid_actions:
                q = self.q_table[state][a]
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def record_shadow_improvement(self, episode: int = 0, improved: bool = True) -> None:
        if improved:
            self._shadow_consecutive_improvements += 1
            if self._shadow_consecutive_improvements >= 50:
                self._shadow_promoted = True
        else:
            self._shadow_consecutive_improvements = 0
            self._shadow_promoted = False

    def shadow_policy_is_promoted(self) -> bool:
        return self._shadow_promoted

    def save_qtable(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({k: dict(v) for k, v in self.q_table.items()}, f)

    def load_qtable(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            for k, v in data.items():
                self.q_table[k].update(v)

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
        devices = state_dict.get("devices", {})

        if devices:
            total_pct = sum(devices.values()) / len(devices)
        else:
            total_pct = 0.0
        total_bin = min(3, int(total_pct * 4))

        active_count = sum(1 for v in devices.values() if v > 0.05)
        active_bin = min(3, active_count // 3)

        return (f"load:{total_bin}::active:{active_bin}"
                f"::price:{state_dict.get('price_tier', 1)}"
                f"::pmv:{state_dict.get('pmv_zone', 1)}"
                f"::tod:{state_dict.get('tod', 2)}"
                f"::dev:{classified_device}")

    def _is_device_locked(self, device_id: str) -> bool:
        if device_id not in self._device_lockout:
            return False
        elapsed = time.time() - self._device_lockout[device_id]
        return elapsed < self.lockout_duration

    def _record_device_action(self, device_id: str) -> None:
        self._device_lockout[device_id] = time.time()

    def clear_device_lockout(self, device_id: str) -> None:
        self._device_lockout.pop(device_id, None)

    def _is_blacklisted(self, device_id: str) -> bool:
        return device_id in self.NEVER_SHED

    def act(self, device_or_state: Any = None, *args, **kwargs) -> RLActionResult:
        # Check if first argument is a device ID string: e.g. act("node_hvac", command="OFF")
        if isinstance(device_or_state, str):
            device_id = device_or_state
            command = kwargs.get("command", "DEFER")
            pmv = kwargs.get("pmv", None)
            confidence = kwargs.get("confidence", 1.0)

            # Check tier0 / NEVER_SHED
            is_tier0 = self._is_blacklisted(device_id) or any(k in device_id for k in ["fridge", "freezer", "pc"])
            if command in ("OFF", "SHED") and is_tier0:
                return RLActionResult("DEFER", blocked_by_tier0=True, command=command, device=device_id)

            # Check PMV empathy only if pmv was explicitly passed
            if pmv is not None and "hvac" in device_id.lower() and command in ("OFF", "SHED"):
                if self.pmv_min <= pmv <= self.pmv_max:
                    return RLActionResult("DEFER", blocked_by_pmv_empathy=True, command=command, device=device_id)

            # Check device lockout / cooldown
            now = time.time()
            if device_id in self._device_lockout:
                if (now - self._device_lockout[device_id]) < self.cooldown:
                    return RLActionResult("DEFER", blocked_by_cooldown=True, command=command, device=device_id)

            self._device_lockout[device_id] = now
            self.last_action_time = now
            return RLActionResult(command, blocked_by_cooldown=False, blocked_by_tier0=False,
                                  blocked_by_pmv_empathy=False, command=command, device=device_id)

        # Standard pipeline call: act(state_dict, pmv, confidence, classified_device, min_confidence=None)
        state_dict = device_or_state if isinstance(device_or_state, dict) else {}
        pmv = kwargs.get("pmv", args[0] if len(args) > 0 else 0.0)
        confidence = kwargs.get("confidence", args[1] if len(args) > 1 else 1.0)
        classified_device = kwargs.get("classified_device", args[2] if len(args) > 2 else "")
        min_confidence = kwargs.get("min_confidence", args[3] if len(args) > 3 else None)

        # Gate 0: NaN safety
        if math.isnan(pmv):
            return RLActionResult("DEFER", device=classified_device)

        # Gate 1: confidence gate
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        if confidence < threshold:
            return RLActionResult("DEFER", device=classified_device)

        # Gate 2: PMV empathy gate
        if classified_device and "hvac" in classified_device.lower():
            if pmv < self.pmv_min or pmv > self.pmv_max:
                if pmv < self.pmv_min:
                    return RLActionResult("SCHEDULE_HVAC", device=classified_device, command="SCHEDULE_HVAC")
                else:
                    return RLActionResult("SHED_HVAC", device=classified_device, command="SHED_HVAC")

        # Gate 3: Global cooldown
        now = time.time()
        if now - self.last_action_time < self.cooldown:
            return RLActionResult("DEFER", blocked_by_cooldown=True, device=classified_device)

        # Gate 4: Per-device anti-thrashing lockout
        if self._is_device_locked(classified_device):
            return RLActionResult("DEFER", blocked_by_cooldown=True, device=classified_device)

        state_key = self._discretize(state_dict, classified_device)

        valid_actions = ["DEFER"]
        if not self._is_blacklisted(classified_device):
            valid_actions.append("SHED")
            valid_actions.append("SCHEDULE")

        if np.random.rand() < self.epsilon:
            action = str(np.random.choice(valid_actions))
        else:
            best_action = "DEFER"
            best_q = float('-inf')
            for a in valid_actions:
                q = self.q_table[state_key][a]
                if q > best_q:
                    best_q = q
                    best_action = a
            action = best_action

        if action == "SHED" and self._is_blacklisted(classified_device):
            logger.info(
                f"[RL INTERCEPTOR] Blocked SHED for blacklisted device '{classified_device}' → RECOMMEND_SCHEDULE"
            )
            return RLActionResult("RECOMMEND_SCHEDULE", blocked_by_tier0=True, command="RECOMMEND_SCHEDULE", device=classified_device)

        if action != "DEFER":
            self.last_action_time = now
            self._record_device_action(classified_device)

        return RLActionResult(action, device=classified_device, command=action)

    def act_with_pmv(self, device: str, command: str, pmv: float) -> RLActionResult:
        return self.act(device, command=command, pmv=pmv)

    async def decide(self, device_states: Dict[str, Any], pmv: float = 0.0) -> List[RLActionResult]:
        results = []
        for dev_id, state in device_states.items():
            is_tier0 = state.get("tier0", False) or self._is_blacklisted(dev_id) or any(k in dev_id for k in ["fridge", "freezer", "pc"])
            if is_tier0:
                results.append(RLActionResult("ON", device=dev_id, command="ON"))
            else:
                cmd = "OFF" if state.get("power", 0) > 1000 else "ON"
                results.append(RLActionResult(cmd, device=dev_id, command=cmd))
        return results

    def compute_reward(self, prev_state: Dict[str, Any], action: str, next_state: Dict[str, Any],
                       pmv: float, current_watts: float, tou_rate: float, confidence: float,
                       aggregate_watts: float = 0.0) -> float:
        projected_watts = 0.0 if action == "SHED" else current_watts
        energy_reward = -projected_watts * tou_rate / 1000.0  # cost in kWh
        pmv_penalty = -5.0 * self.twin.pmv_penalty(pmv)     # heavy comfort penalty
        safety_bonus = 0.0 if aggregate_watts < self.max_watts else -10.0
        return energy_reward + pmv_penalty + safety_bonus

    def update(self, state: Any = None, action: str = "DEFER", reward: float = 0.0,
               next_state: Any = None, classified_device: str = "",
               state_dict: Any = None, next_state_dict: Any = None, **kwargs) -> None:
        s = state if state is not None else state_dict
        s_prime = next_state if next_state is not None else next_state_dict

        if isinstance(s, dict):
            state_key = self._discretize(s, classified_device)
        else:
            state_key = str(s)

        if isinstance(s_prime, dict):
            next_state_key = self._discretize(s_prime, classified_device)
        else:
            next_state_key = str(s_prime)

        best_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0

        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

        # Epsilon decay per calendar day (episode), NOT per step.
        # At 1Hz, per-step decay exhausts exploration in <1 hour.
        # Per-day decay lets the agent explore across full diurnal cycles.
        today = date.today().isoformat()
        if today != self._last_decay_date:
            self._last_decay_date = today
            self.decay_epsilon()

        # Log to CSV (synchronous — caller should use log_action_async from async context)
        self._log_action_sync(state_key, action, reward, next_state_key)

    def _log_action_sync(self, state: str, action: str, reward: float, next_state: str) -> None:
        try:
            import fcntl
            with open("rl_action_log.csv", "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(f"{time.time()},{state},{action},{reward},{next_state}\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except ImportError:
            try:
                with open("rl_action_log.csv", "a") as f:
                    f.write(f"{time.time()},{state},{action},{reward},{next_state}\n")
            except Exception as e:
                logger.error(f"Failed to log RL action: {e}")
        except Exception as e:
            logger.error(f"Failed to log RL action: {e}")

    async def log_action_async(self, state: str, action: str, reward: float, next_state: str) -> None:
        import asyncio
        await asyncio.to_thread(self._log_action_sync, state, action, reward, next_state)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, "wb") as f:
            pickle.dump({k: dict(v) for k, v in self.q_table.items()}, f)

    async def save_async(self) -> None:
        import asyncio
        await asyncio.to_thread(self.save)

    def load(self) -> None:
        try:
            with open(self.q_table_path, "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    self.q_table[k].update(v)
            logger.info(f"Loaded Q-table from {self.q_table_path}")
        except Exception as e:
            logger.warning(f"Could not load Q-table: {e}")

    async def load_async(self) -> None:
        import asyncio
        await asyncio.to_thread(self.load)

    async def save_to_sqlite(self, db_path: str = "data/ems_state.db") -> None:
        import aiosqlite
        import json
        try:
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS q_table (
                        state_key TEXT PRIMARY KEY,
                        action_values TEXT
                    )
                """)
                for state_key, actions in self.q_table.items():
                    await conn.execute(
                        "INSERT OR REPLACE INTO q_table (state_key, action_values) "
                        "VALUES (?, ?)",
                        (state_key, json.dumps(dict(actions)))
                    )
                await conn.commit()
            logger.info(f"Q-table saved to SQLite ({len(self.q_table)} states)")
        except Exception as e:
            logger.error(f"Failed to save Q-table to SQLite: {e}")

    async def load_from_sqlite(self, db_path: str = "data/ems_state.db") -> None:
        import aiosqlite
        import json
        try:
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute("SELECT state_key, action_values FROM q_table")
                rows = await cursor.fetchall()
                for state_key, action_json in rows:
                    actions = json.loads(action_json)
                    self.q_table[state_key].update(actions)
            logger.info(f"Loaded Q-table from SQLite ({len(self.q_table)} states)")
        except Exception as e:
            logger.warning(f"Could not load Q-table from SQLite: {e}")


# ── Alias ─────────────────────────────────────────────────────────────────────
QLearningAgent = TabularQLearningAgent


# ── Policy Promotion Gate ─────────────────────────────────────────────────────

class PolicyPromotionGate:
    MIN_VALIDATION_EPISODES = 50
    PMV_PENALTY_LIMIT       = 0.5   # max allowed cumulative PMV penalty

    def __init__(self):
        self._val_episodes:   int   = 0
        self._cumulative_pmv: float = 0.0
        self._promoted:       bool  = False

    def record_twin_episode(self, pmv_penalty: float) -> None:
        self._val_episodes   += 1
        self._cumulative_pmv += pmv_penalty

    @property
    def is_promoted(self) -> bool:
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
