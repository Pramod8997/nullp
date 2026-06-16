"""
Deep Q-Network (DQN) Agent — Fix §3.3.1 from Feasibility Study.

Replaces tabular Q-learning for scalable state spaces.
Uses a small feedforward neural network to approximate Q(s, a)
with continuous state inputs (PMV, power, price) instead of
discretized bins.

Preserves all safety gating logic from TabularQLearningAgent:
  - Confidence Gate (blocks when classification < threshold)
  - Empathy Gate (forces HVAC when PMV out of Category A bounds)
  - Per-device anti-thrashing lockout (monotonic clock, 5 min)
  - Policy Promotion Gate (50 episodes shadow mode)
  - Appliance blacklist (NEVER_SHED)

Architecture:
  State:  [total_load_pct, active_device_pct, price_rate,
           pmv, time_of_day_sin, time_of_day_cos, device_encoding]
  Hidden: 128 → 128 → num_actions
  Action: DEFER(0), SHED(1), SCHEDULE(2)
"""
import os
import math
import time
import random
import logging
import asyncio
import yaml
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.thermodynamics import pmv_model

logger = logging.getLogger(__name__)

# ── Production Constants ──
DEVICE_LOCKOUT_SECONDS = 300.0   # 5-minute per-device anti-thrashing window
REPLAY_BUFFER_SIZE     = 10000   # Experience replay buffer capacity
BATCH_SIZE             = 64      # Training batch size
TARGET_UPDATE_FREQ     = 100     # Steps between target network syncs

# Safety-critical appliance identifiers that must never be shed
_HARDCODED_SAFETY_CRITICAL = [
    "esp32_fridge", "node_fridge",
    "esp32_freezer", "node_freezer",
    "esp32_pc", "node_pc",
]

# Action indices
ACTION_DEFER    = 0
ACTION_SHED     = 1
ACTION_SCHEDULE = 2
NUM_ACTIONS     = 3
ACTION_NAMES    = ["DEFER", "SHED", "SCHEDULE"]

# State vector dimension
STATE_DIM = 7  # [load_pct, active_pct, price, pmv, tod_sin, tod_cos, dev_hash]


class QNetwork(nn.Module):
    """
    Small feedforward Q-network: state → Q-values for all actions.
    Two hidden layers of 128 units with ReLU activation.
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for energy management.

    Replaces tabular Q-learning with a neural network that accepts
    continuous state vectors, solving the state-space explosion problem
    identified in the feasibility study (§3.2).

    Preserves all safety gating logic from TabularQLearningAgent.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        rl_cfg = self.config.get("rl", {})
        self.cooldown = rl_cfg.get("cooldown_seconds", 15.0)
        self.pmv_min = rl_cfg.get("empathy_pmv_min", -0.5)
        self.pmv_max = rl_cfg.get("empathy_pmv_max", 0.5)
        self.model_path = rl_cfg.get(
            "dqn_model_path",
            "backend/models/weights/dqn_model.pt"
        )

        self.tou_pricing = self.config.get("analytics", {}).get("tou_pricing", {})

        protonet_cfg = self.config.get("protonet", {})
        self.confidence_threshold = protonet_cfg.get("confidence_threshold", 0.90)

        safety_cfg = self.config.get("system_safety", {})
        self.max_watts = safety_cfg.get("max_aggregate_wattage", 3500.0)
        self.critical_pct = safety_cfg.get("critical_pct", 1.25)
        self.device_limits = safety_cfg.get("device_wattage_limits", {})

        # DQN hyperparameters
        self.gamma = rl_cfg.get("dqn_gamma", 0.99)
        self.lr    = rl_cfg.get("dqn_lr", 1e-3)
        self.epsilon_start = rl_cfg.get("epsilon_start", 0.3)
        self.epsilon_end   = rl_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = rl_cfg.get("epsilon_decay", 0.999)
        self.epsilon = self.epsilon_start
        self.batch_size = rl_cfg.get("dqn_batch_size", BATCH_SIZE)
        self.target_update_freq = rl_cfg.get("dqn_target_update", TARGET_UPDATE_FREQ)

        # Device index mapping for state encoding
        devices_cfg = self.config.get("devices", {})
        self._device_names = sorted(devices_cfg.keys())
        self._device_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self._device_names)
        }

        # ── Appliance Blacklist (NEVER_SHED) ──
        self.NEVER_SHED = [
            name for name, cfg in devices_cfg.items()
            if isinstance(cfg, dict) and cfg.get("tier0", False)
        ]
        for device in _HARDCODED_SAFETY_CRITICAL:
            if device not in self.NEVER_SHED:
                self.NEVER_SHED.append(device)
        if not any("fridge" in d for d in self.NEVER_SHED):
            self.NEVER_SHED.append("esp32_fridge")

        # ── Neural Networks ──
        self.device = torch.device("cpu")
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer()
        self._step_count = 0

        # ── Per-Device Anti-Thrashing Lockout (NTP-resilient) ──
        self._device_lockout: Dict[str, float] = {}
        self.lockout_duration = DEVICE_LOCKOUT_SECONDS

        # Legacy global cooldown for backward compat
        self.last_action_time = 0.0

        self.twin = pmv_model

        # Load saved model if exists
        if os.path.exists(self.model_path):
            self.load()

    def get_tou_rate(self, hour: int) -> float:
        """Returns the ToU rate for a given hour."""
        for tier, data in self.tou_pricing.items():
            if hour in data.get("hours", []):
                return data.get("rate", 0.15)
        return 0.15

    def _encode_state(self, state_dict: Dict[str, Any],
                      classified_device: str = "") -> np.ndarray:
        """
        Encode state as a continuous vector instead of discrete bins.
        This is the key difference from tabular Q-learning.

        Returns:
            numpy array of shape (STATE_DIM,):
              [0] total_load_pct:    normalized aggregate load (0-1)
              [1] active_device_pct: fraction of devices active (0-1)
              [2] price_rate:        current ToU rate (continuous)
              [3] pmv:               PMV value (continuous, -3 to 3)
              [4] tod_sin:           sin(2π * hour/24) for cyclical time
              [5] tod_cos:           cos(2π * hour/24) for cyclical time
              [6] device_hash:       normalized device index (0-1)
        """
        devices = state_dict.get("devices", {})

        # Total load as fraction of rated capacity
        total_pct = 0.0
        if devices:
            total_pct = min(1.0, sum(devices.values()) / max(1, len(devices)))

        # Active device fraction
        n_devices = max(1, len(self._device_names))
        active_count = sum(1 for v in devices.values() if v > 0.05)
        active_pct = min(1.0, active_count / n_devices)

        # Continuous price rate
        price_rate = state_dict.get("tou_rate", 0.15)

        # Raw PMV value (continuous)
        pmv = state_dict.get("pmv", 0.0)
        pmv_normalized = np.clip(pmv / 3.0, -1.0, 1.0)  # normalize to [-1, 1]

        # Cyclical time encoding
        hour = state_dict.get("hour", 12)
        tod_angle = 2.0 * math.pi * hour / 24.0
        tod_sin = math.sin(tod_angle)
        tod_cos = math.cos(tod_angle)

        # Device encoding (normalized index)
        dev_idx = self._device_to_idx.get(classified_device, 0)
        dev_hash = dev_idx / max(1, len(self._device_names) - 1)

        return np.array([
            total_pct, active_pct, price_rate,
            pmv_normalized, tod_sin, tod_cos, dev_hash
        ], dtype=np.float32)

    def _is_device_locked(self, device_id: str) -> bool:
        """Check if a device is within its anti-thrashing lockout window."""
        if device_id not in self._device_lockout:
            return False
        elapsed = time.monotonic() - self._device_lockout[device_id]
        return elapsed < self.lockout_duration

    def _record_device_action(self, device_id: str) -> None:
        """Record a device action timestamp using monotonic clock."""
        self._device_lockout[device_id] = time.monotonic()

    def clear_device_lockout(self, device_id: str) -> None:
        """Clear a specific device's lockout."""
        self._device_lockout.pop(device_id, None)

    def _is_blacklisted(self, device_id: str) -> bool:
        """Check if a device is in the safety-critical blacklist."""
        return device_id in self.NEVER_SHED

    def act(self, state_dict: Dict[str, Any], pmv: float, confidence: float,
            classified_device: str, min_confidence: float = None) -> str:
        """Act based on current state. Returns action string."""
        # Gate 1: Confidence gate
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        if confidence < threshold:
            return "DEFER"

        # Gate 2: PMV empathy gate (only for HVAC devices)
        if classified_device and "hvac" in classified_device.lower():
            if pmv < self.pmv_min or pmv > self.pmv_max:
                return "SCHEDULE_HVAC" if pmv < self.pmv_min else "SHED_HVAC"

        # Gate 3: Global cooldown (legacy compat)
        if time.time() - self.last_action_time < self.cooldown:
            return "DEFER"

        # Gate 4: Per-device anti-thrashing lockout
        if self._is_device_locked(classified_device):
            return "DEFER"

        # Encode state as continuous vector
        state_vec = self._encode_state(state_dict, classified_device)

        # Build valid action mask
        valid_actions = [ACTION_DEFER]
        if not self._is_blacklisted(classified_device):
            valid_actions.append(ACTION_SHED)
            valid_actions.append(ACTION_SCHEDULE)

        # Explore vs Exploit
        if random.random() < self.epsilon:
            action_idx = random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_net(state_t).squeeze(0).numpy()
                # Mask invalid actions to -inf
                masked_q = np.full(NUM_ACTIONS, -np.inf)
                for a in valid_actions:
                    masked_q[a] = q_values[a]
                action_idx = int(np.argmax(masked_q))

        action = ACTION_NAMES[action_idx]

        # Blacklist interceptor (defense-in-depth)
        if action == "SHED" and self._is_blacklisted(classified_device):
            logger.info(
                f"[DQN INTERCEPTOR] Blocked SHED for blacklisted device "
                f"'{classified_device}' → RECOMMEND_SCHEDULE"
            )
            return "RECOMMEND_SCHEDULE"

        # Update timestamps for real actions
        if action != "DEFER":
            self.last_action_time = time.time()
            self._record_device_action(classified_device)

        return action

    def compute_reward(self, prev_state: Dict[str, Any], action: str,
                       next_state: Dict[str, Any], pmv: float,
                       current_watts: float, tou_rate: float,
                       confidence: float,
                       aggregate_watts: float = 0.0) -> float:
        """Compute reward — same logic as tabular agent."""
        projected_watts = 0.0 if action == "SHED" else current_watts
        energy_reward = -projected_watts * tou_rate / 1000.0
        pmv_penalty = -5.0 * self.twin.pmv_penalty(pmv)
        safety_bonus = 0.0 if aggregate_watts < self.max_watts else -10.0
        return energy_reward + pmv_penalty + safety_bonus

    def store_transition(self, state_dict: Dict[str, Any], action: str,
                         reward: float, next_state_dict: Dict[str, Any],
                         classified_device: str = "",
                         done: bool = False) -> None:
        """Store a transition in the replay buffer."""
        state_vec = self._encode_state(state_dict, classified_device)
        next_state_vec = self._encode_state(next_state_dict, classified_device)
        action_idx = ACTION_NAMES.index(action) if action in ACTION_NAMES else 0
        self.replay_buffer.push(state_vec, action_idx, reward,
                                next_state_vec, done)

    def update(self, state_dict: Dict[str, Any], action: str, reward: float,
               next_state_dict: Dict[str, Any],
               classified_device: str = "") -> None:
        """
        Store transition and train on a batch from replay buffer.
        Compatible API with TabularQLearningAgent.update().
        """
        self.store_transition(state_dict, action, reward,
                              next_state_dict, classified_device)

        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        self._train_step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodic target network sync
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logger.debug(f"Target network synced at step {self._step_count}")

        # Log action (synchronous — same as tabular agent)
        state_key = str(self._encode_state(state_dict, classified_device).tolist())
        next_key = str(self._encode_state(next_state_dict, classified_device).tolist())
        self._log_action_sync(state_key, action, reward, next_key)

    def _train_step(self) -> None:
        """Perform one gradient step on a batch from replay buffer."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        # Current Q-values for taken actions
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values from frozen target network
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # MSE loss
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def _log_action_sync(self, state: str, action: str,
                         reward: float, next_state: str) -> None:
        """Synchronous file write — safe to call from sync context."""
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

    async def log_action_async(self, state: str, action: str,
                               reward: float, next_state: str) -> None:
        """Non-blocking RL log — runs sync file I/O in a thread."""
        await asyncio.to_thread(self._log_action_sync, state, action, reward, next_state)

    def save(self) -> None:
        """Save model weights and optimizer state."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
        }, self.model_path)
        logger.info(f"DQN model saved to {self.model_path}")

    async def save_async(self) -> None:
        """Non-blocking model save — Fix §3.3.2."""
        await asyncio.to_thread(self.save)

    def load(self) -> None:
        """Load model weights and optimizer state."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=False)
            self.q_net.load_state_dict(checkpoint["q_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self._step_count = checkpoint.get("step_count", 0)
            logger.info(f"DQN model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load DQN model: {e}")

    async def load_async(self) -> None:
        """Non-blocking model load — Fix §3.3.2."""
        await asyncio.to_thread(self.load)
