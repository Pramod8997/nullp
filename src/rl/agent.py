import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class TabularQLearningAgent:
    def __init__(self, num_time_buckets: int = 24, num_power_bins: int = 10, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.MAX_RL_DEVICES = 4 # [INV-5]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_time_buckets = num_time_buckets
        self.num_power_bins = num_power_bins
        
        # Action space: for each device, 0 (off), 1 (on), or 2 (no-op)
        # Total actions = 3 ** MAX_RL_DEVICES
        self.num_actions = 3 ** self.MAX_RL_DEVICES
        
        # State space dimensions: time_bucket, power_bin, device1_state, device2_state...
        # State is represented as a tuple of integers for dictionary lookup
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def get_state_tuple(self, time_bucket: int, power_bin: int, device_states: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(device_states) > self.MAX_RL_DEVICES:
            logger.warning(f"Truncating device states to MAX_RL_DEVICES ({self.MAX_RL_DEVICES})")
            device_states = device_states[:self.MAX_RL_DEVICES]
        elif len(device_states) < self.MAX_RL_DEVICES:
            # Pad with 0s if fewer devices
            device_states = device_states + (0,) * (self.MAX_RL_DEVICES - len(device_states))
            
        return (time_bucket, power_bin) + device_states

    def get_action(self, state: Tuple[int, ...]) -> int:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
            
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        best_next_action = int(np.argmax(self.q_table[next_state]))
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
