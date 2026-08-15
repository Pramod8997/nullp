import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.safety import SafetyMonitor, SafetyEvent, load_config, slow_rl_agent
from src.rl.agent import QLearningAgent


# TEST 3A-1: NEVER_SHED enforcement — fridge (tier0=True) must NEVER receive
# a shed command, even when aggregate power exceeds 3500W
@pytest.mark.asyncio
async def test_never_shed_fridge_under_any_load():
    safety = SafetyMonitor(config=load_config())
    rl_agent = QLearningAgent(config=load_config())
    
    # Simulate 3600W total load (above 3500W aggregate limit)
    device_states = {
        "node_fridge":   {"power": 200.0,  "tier0": True},
        "node_hvac":     {"power": 2000.0, "tier0": False},
        "node_kettle":   {"power": 2500.0, "tier0": False},
    }
    actions = await rl_agent.decide(device_states, pmv=0.8)
    fridge_actions = [a for a in actions if a.device == "node_fridge"]
    shed_actions   = [a for a in fridge_actions if a.command == "OFF"]
    assert len(shed_actions) == 0, "Fridge (tier0) must NEVER be shed"

# TEST 3A-2: Aggregate safety limit — exactly at 3500W, no critical alert
@pytest.mark.asyncio
async def test_aggregate_exactly_at_limit_no_critical():
    safety = SafetyMonitor(config=load_config())
    result = await safety.check_aggregate({"a": 1000.0, "b": 1000.0, "c": 1500.0})
    assert result.level != "CRITICAL"  # 3500W is at limit, not over

# TEST 3A-3: Aggregate at 3501W must trigger CRITICAL
@pytest.mark.asyncio
async def test_aggregate_one_watt_over_limit_critical():
    safety = SafetyMonitor(config=load_config())
    result = await safety.check_aggregate({"a": 1000.0, "b": 1000.0, "c": 1501.0})
    assert result.level == "CRITICAL"
    assert result.event_type == "SAFETY_ALERT"

# TEST 3A-4: Arc-fault detection — dP/dt > 1000 W/s triggers ARC_FAULT event
@pytest.mark.asyncio
async def test_arc_fault_roc_above_threshold():
    safety = SafetyMonitor(config=load_config())
    # Previous reading: 200W; new reading 1201W after 1 second → dP/dt = 1001 W/s
    result = await safety.check_roc(device="node_kettle", prev_power=200.0,
                                     curr_power=1201.0, dt_seconds=1.0)
    assert result.event_type == "ARC_FAULT"

# TEST 3A-5: dP/dt exactly at 1000 W/s must NOT trigger (boundary)
@pytest.mark.asyncio
async def test_arc_fault_exactly_at_threshold_no_trigger():
    safety = SafetyMonitor(config=load_config())
    result = await safety.check_roc(device="node_kettle", prev_power=200.0,
                                     curr_power=1200.0, dt_seconds=1.0)
    # dP/dt = 1000 W/s exactly — should not trigger (threshold is strict >)
    assert result is None or result.event_type != "ARC_FAULT"

# TEST 3A-6: Safety monitor operates independently of ML pipeline
# If ProtoNet is disabled, safety must still fire on power overage
@pytest.mark.asyncio
async def test_safety_fires_without_protonet():
    safety = SafetyMonitor(config=load_config(), protonet_enabled=False)
    result = await safety.check_aggregate({"device": 4000.0})
    assert result.event_type == "SAFETY_ALERT"

# TEST 3A-7: Per-device wattage limit — kettle rated 2500W, limit 2600W
@pytest.mark.asyncio
async def test_per_device_wattage_limit_kettle():
    safety = SafetyMonitor(config=load_config())
    result = await safety.check_device("node_kettle", power=2601.0)
    assert result is not None and result.level in ("WARNING", "CRITICAL")

# TEST 3A-8: Safety events are broadcast even when RL is mid-decision
@pytest.mark.asyncio
async def test_safety_preempts_rl_action():
    # Safety must not wait for RL to complete; it is a parallel asyncio.Task
    events_emitted = []
    safety = SafetyMonitor(config=load_config(),
                           broadcast_fn=lambda e: events_emitted.append(e))
    rl_task = asyncio.create_task(slow_rl_agent(delay=0.5))  # 500ms RL
    await safety.check_aggregate({"device": 4000.0})  # should fire immediately
    
    # Safety event should appear before RL task finishes
    safety_events = [e for e in events_emitted if e.event_type == "SAFETY_ALERT"]
    assert len(safety_events) > 0
    assert not rl_task.done()  # RL still running when safety fired
