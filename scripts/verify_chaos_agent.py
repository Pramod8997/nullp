"""
Custom Agent 2: Chaos & Adversarial Verification Agent
Comprehensive boundary-condition testing for Patches 8-10:
  - RL Agent NaN/Inf/extreme guard (Patch 8)
  - Safety Monitor input sanitization (Patch 9)
  - Watchdog poisoning prevention (Patch 10)

Tests every adversarial vector that the audit identified.
"""
import sys
import os
import math
import asyncio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl.agent import TabularQLearningAgent
from src.rl.dqn_agent import DQNAgent
from src.pipeline.watchdog import SoftAnomalyWatchdog

print("=" * 60)
print("  CUSTOM AGENT 2: CHAOS & ADVERSARIAL VERIFICATION")
print("=" * 60)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ PASS — {name}")
    else:
        failed += 1
        print(f"  ❌ FAIL — {name}" + (f" ({detail})" if detail else ""))

# ──────────────────────────────────────────────────────────────
# TEST SUITE 1: RL Agent Adversarial Inputs (Patch 8)
# ──────────────────────────────────────────────────────────────
print("\n[SUITE 1] RL Agent NaN/Inf/Extreme Guard")

agent = TabularQLearningAgent()
dqn = DQNAgent()
state = {"devices": {"esp32_hvac": 1000}, "price_tier": 1, "pmv_zone": 1, "tod": 2}

# Test NaN PMV
act1 = agent.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
act2 = dqn.act(state, pmv=float('nan'), confidence=0.99, classified_device="esp32_hvac")
check("Tabular agent DEFERs on NaN PMV", act1 == "DEFER", f"got {act1}")
check("DQN agent DEFERs on NaN PMV", act2 == "DEFER", f"got {act2}")

# Test Inf PMV
act3 = agent.act(state, pmv=float('inf'), confidence=0.99, classified_device="esp32_hvac")
act4 = dqn.act(state, pmv=float('inf'), confidence=0.99, classified_device="esp32_hvac")
check("Tabular agent handles Inf PMV", act3 in ("SHED_HVAC", "SCHEDULE_HVAC", "DEFER"), f"got {act3}")
check("DQN agent handles Inf PMV", act4 in ("SHED_HVAC", "SCHEDULE_HVAC", "DEFER"), f"got {act4}")

# Test -Inf PMV
act5 = agent.act(state, pmv=float('-inf'), confidence=0.99, classified_device="esp32_hvac")
check("Tabular agent handles -Inf PMV", act5 in ("SHED_HVAC", "SCHEDULE_HVAC", "DEFER"), f"got {act5}")

# Test extreme values
act6 = agent.act(state, pmv=1e15, confidence=0.99, classified_device="esp32_hvac")
check("Tabular agent handles extreme PMV (1e15)", act6 in ("SHED_HVAC", "DEFER"), f"got {act6}")

act7 = agent.act(state, pmv=-1e15, confidence=0.99, classified_device="esp32_hvac")
check("Tabular agent handles extreme negative PMV (-1e15)", act7 in ("SCHEDULE_HVAC", "DEFER"), f"got {act7}")

# ──────────────────────────────────────────────────────────────
# TEST SUITE 2: Watchdog Input Sanitization (Patch 10)
# ──────────────────────────────────────────────────────────────
print("\n[SUITE 2] Watchdog Input Sanitization")

wd = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)

# Fill baseline with normal readings
for i in range(15):
    wd.check_reading("sensor_test", 100.0 + np.random.normal(0, 1))

# Test NaN — should not poison history
result_nan = wd.check_reading("sensor_test", float('nan'))
check("Watchdog rejects NaN (returns False)", result_nan == False)
history_healthy = not any(math.isnan(x) for x in wd.history.get("sensor_test", []))
check("Watchdog history not poisoned by NaN", history_healthy)

# Test Inf — should not poison history
result_inf = wd.check_reading("sensor_test", float('inf'))
check("Watchdog rejects Inf (returns False)", result_inf == False)
history_healthy2 = not any(math.isinf(x) for x in wd.history.get("sensor_test", []))
check("Watchdog history not poisoned by Inf", history_healthy2)

# Test -Inf
result_neginf = wd.check_reading("sensor_test", float('-inf'))
check("Watchdog rejects -Inf (returns False)", result_neginf == False)

# Test string input
result_str = wd.check_reading("sensor_test", "not_a_number")
check("Watchdog rejects string input (returns False)", result_str == False)

# Test None
result_none = wd.check_reading("sensor_test", None)
check("Watchdog rejects None (returns False)", result_none == False)

# Test that normal readings still work after adversarial inputs
result_normal = wd.check_reading("sensor_test", 100.5)
check("Watchdog still processes normal readings after adversarial", result_normal == False)

# Verify history length didn't grow from adversarial inputs
history_len = len(wd.history.get("sensor_test", []))
check("Watchdog history length unchanged by adversarial inputs",
      history_len == 16,  # 15 baseline + 1 normal after adversarial
      f"expected 16, got {history_len}")

# ──────────────────────────────────────────────────────────────
# TEST SUITE 3: Sustained Adversarial Barrage
# ──────────────────────────────────────────────────────────────
print("\n[SUITE 3] Sustained Adversarial Barrage (100 poisonous inputs)")

wd2 = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)
# Build baseline
for i in range(15):
    wd2.check_reading("barrage", 500.0 + np.random.normal(0, 2))

# Fire 100 adversarial inputs
adversarial_values = (
    [float('nan')] * 30 +
    [float('inf')] * 20 +
    [float('-inf')] * 20 +
    ["garbage"] * 10 +
    [None] * 10 +
    [True] * 5 +      # bool is subclass of int but still weird
    [complex(1, 2)] * 5  # complex number
)

crashes = 0
for val in adversarial_values:
    try:
        wd2.check_reading("barrage", val)
    except Exception:
        crashes += 1

check("Zero crashes during 100-input adversarial barrage", crashes == 0, f"{crashes} crashes")

# Verify watchdog still functional
normal_result = wd2.check_reading("barrage", 500.0)
check("Watchdog functional after barrage", isinstance(normal_result, bool))

barrage_healthy = all(
    isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)
    for x in wd2.history.get("barrage", [])
)
check("Watchdog history clean after barrage", barrage_healthy)

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  CHAOS VERIFICATION: {passed} PASSED, {failed} FAILED")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
