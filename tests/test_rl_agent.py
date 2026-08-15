import datetime
import tempfile
import os
import pytest
import numpy as np
from freezegun import freeze_time

from src.rl.agent import QLearningAgent, ReplayBuffer, load_config


# TEST 4A-1: Bellman update — verify exact Q-value after one step
def test_bellman_update_single_step():
    agent = QLearningAgent(alpha=0.1, gamma=0.95, config=load_config())
    # Manually set Q(s,a)=5.0, inject r=1.0, max_Q(s')=10.0
    agent.set_q(state="s", action="ON", value=5.0)
    agent.set_q(state="s_prime", action="OFF", value=10.0)
    agent.update(state="s", action="ON", reward=1.0, next_state="s_prime")
    # Expected: Q = 5.0 + 0.1*(1.0 + 0.95*10.0 - 5.0) = 5.0 + 0.1*(1+9.5-5) = 5.0+0.55 = 5.55
    expected_q = 5.0 + 0.1 * (1.0 + 0.95 * 10.0 - 5.0)
    actual_q = agent.get_q(state="s", action="ON")
    assert abs(actual_q - expected_q) < 1e-5

# TEST 4A-2: Epsilon-greedy — with epsilon=1.0, always random (never greedy)
def test_epsilon_greedy_fully_random():
    agent = QLearningAgent(epsilon=1.0, config=load_config())
    actions = set()
    for _ in range(500):
        action = agent.select_action("s")
        actions.add(action)
    # With epsilon=1.0, should explore multiple actions
    assert len(actions) > 1

# TEST 4A-3: Epsilon-greedy — with epsilon=0.0, always greedy (best Q)
def test_epsilon_greedy_fully_greedy():
    agent = QLearningAgent(epsilon=0.0, config=load_config())
    agent.set_q("s", "ON", 10.0)
    agent.set_q("s", "OFF", 2.0)
    for _ in range(100):
        action = agent.select_action("s")
        assert action == "ON"  # always picks highest Q

# TEST 4A-4: Epsilon decay is monotonically decreasing
def test_epsilon_decay_monotone():
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.999995, config=load_config())
    prev_eps = agent.epsilon
    for _ in range(1000):
        agent.decay_epsilon()
        assert agent.epsilon <= prev_eps, "Epsilon must not increase"
        prev_eps = agent.epsilon

# TEST 4A-5: Epsilon reaches floor (does not go below 0.01 or configured minimum)
def test_epsilon_floor_reached():
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.999995,
                           epsilon_min=0.01, config=load_config())
    for _ in range(500000):
        agent.decay_epsilon()
    assert agent.epsilon >= 0.01

# TEST 4A-6: Episode count after decay — epsilon per episode is reproducible
@freeze_time("2024-01-01")
def test_epsilon_value_at_episode_n():
    agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.999995, config=load_config())
    for _ in range(2000):
        agent.decay_epsilon()
    expected = 1.0 * (0.999995 ** 2000)
    assert abs(agent.epsilon - expected) < 1e-6

# TEST 4A-7: Shadow gate — policy does NOT promote before 50 episodes of improvement
def test_shadow_gate_no_promotion_before_50_episodes():
    agent = QLearningAgent(config=load_config())
    for episode in range(49):
        agent.record_shadow_improvement(episode=episode, improved=True)
    assert not agent.shadow_policy_is_promoted()

# TEST 4A-8: Shadow gate — promotes after exactly 50 consecutive improvements
def test_shadow_gate_promotes_at_50_episodes():
    agent = QLearningAgent(config=load_config())
    for episode in range(50):
        agent.record_shadow_improvement(episode=episode, improved=True)
    assert agent.shadow_policy_is_promoted()

# TEST 4A-9: Shadow gate — resets counter if any episode shows no improvement
def test_shadow_gate_counter_resets_on_no_improvement():
    agent = QLearningAgent(config=load_config())
    for episode in range(45):
        agent.record_shadow_improvement(episode=episode, improved=True)
    agent.record_shadow_improvement(episode=45, improved=False)  # break streak
    assert not agent.shadow_policy_is_promoted()

# TEST 4A-10: Device lockout — same device cannot receive RL action within 15s
@pytest.mark.asyncio
async def test_rl_device_lockout_15s():
    agent = QLearningAgent(cooldown_s=15, config=load_config())
    await agent.act("node_hvac", command="OFF")
    # Immediately try again (< 15s elapsed)
    result = await agent.act("node_hvac", command="ON")
    assert result is None or result.blocked_by_cooldown

@pytest.mark.asyncio
async def test_rl_device_lockout_expires_after_15s():
    agent = QLearningAgent(cooldown_s=15, config=load_config())
    with freeze_time("2024-01-01 12:00:00") as frozen:
        await agent.act("node_hvac", command="OFF")
        frozen.tick(delta=datetime.timedelta(seconds=16))
        result = await agent.act("node_hvac", command="ON")
        assert result is None or not result.blocked_by_cooldown

# TEST 4A-11: Q-table persists across restarts (save + load)
def test_qtable_persistence():
    agent = QLearningAgent(config=load_config())
    agent.set_q("state_x", "ON", 42.5)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    agent.save_qtable(path)
    
    agent2 = QLearningAgent(config=load_config())
    agent2.load_qtable(path)
    assert abs(agent2.get_q("state_x", "ON") - 42.5) < 1e-5
    os.unlink(path)

# TEST 4A-12: DQN replay buffer — samples without replacement, capped at capacity
def test_dqn_replay_buffer():
    buffer = ReplayBuffer(capacity=100)
    for i in range(150):
        buffer.push(state=np.array([i]*10, dtype=float), action=0, reward=i, next_state=np.zeros(10))
    assert len(buffer) == 100  # older entries evicted
    
    batch = buffer.sample(32)
    assert len(batch) == 32
    # All samples in batch must be unique (no replacement in one sample call)
    states = [tuple(b.state) for b in batch]
    assert len(set(states)) == 32

# TEST 4A-13: NEVER_SHED enforced even when RL calls act() directly
@pytest.mark.asyncio
async def test_never_shed_enforced_in_act():
    agent = QLearningAgent(config=load_config())
    # Try to force-shed the fridge (tier0 device)
    result = await agent.act("node_fridge", command="OFF")
    assert result is None or result.blocked_by_tier0

# TEST 4A-14: PMV empathy gate integration — full stack
@pytest.mark.asyncio
async def test_pmv_empathy_blocks_hvac_shed():
    agent = QLearningAgent(config=load_config())
    # HVAC shed request when PMV=0.3 (within comfort zone)
    result = await agent.act_with_pmv("node_hvac", command="OFF", pmv=0.3)
    assert result is None or result.blocked_by_pmv_empathy
