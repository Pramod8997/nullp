import math
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume

from src.models.protonet import euclidean_distance_squared
from src.models.thermodynamics import compute_pmv, compute_ppd
from src.pipeline.phantom_tracker import PhantomTracker
from src.rl.agent import QLearningAgent, load_config


# TEST 11-1: ProtoNet distance is always non-negative
@given(
    q=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=128, max_size=128),
    p=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=128, max_size=128),
)
def test_distance_non_negative(q, p):
    d = euclidean_distance_squared(np.array(q), np.array(p))
    assert d >= 0.0

# TEST 11-2: PMV is anti-symmetric with respect to temperature change direction
# (colder environment → lower PMV; hotter → higher PMV)
@given(
    ta=st.floats(min_value=15, max_value=35),
    delta=st.floats(min_value=0.1, max_value=5.0),
)
def test_pmv_increases_with_temperature(ta, delta):
    assume(ta + delta <= 40)
    pmv_low  = compute_pmv(M=70, W=0, ta=ta,       tr=ta,       var=0.1, rh=50, Icl=1.0)
    pmv_high = compute_pmv(M=70, W=0, ta=ta+delta, tr=ta+delta, var=0.1, rh=50, Icl=1.0)
    assert pmv_high >= pmv_low, "Higher temperature must produce higher PMV"

# TEST 11-3: PPD is always between 5% and 100%
@given(pmv=st.floats(min_value=-3.5, max_value=3.5, allow_nan=False))
def test_ppd_bounded(pmv):
    ppd = compute_ppd(pmv)
    assert 5.0 <= ppd <= 100.0

# TEST 11-4: Q-value after n Bellman updates is bounded (no divergence)
@given(
    initial_q=st.floats(min_value=-100, max_value=100, allow_nan=False),
    rewards=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False),
                     min_size=1, max_size=1000),
)
def test_qvalue_bounded_after_updates(initial_q, rewards):
    agent = QLearningAgent(alpha=0.1, gamma=0.95, config=load_config())
    agent.set_q("s", "ON", initial_q)
    for r in rewards:
        agent.update("s", "ON", r, "s")  # self-loop
    q = agent.get_q("s", "ON")
    assert not math.isnan(q)
    assert not math.isinf(q)

# TEST 11-5: EMA value always stays between 0 and the maximum reading ever seen
@given(
    alpha=st.floats(min_value=0.01, max_value=0.5),
    readings=st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=500),
)
def test_ema_bounded_by_max_reading(alpha, readings):
    tracker = PhantomTracker(alpha=alpha)
    max_reading = max(readings)
    for r in readings:
        tracker.update("device", r, state="OFF")
    ema = tracker.get_ema("device")
    assert 0.0 <= ema <= max_reading + 0.001
