import datetime
import pytest
from freezegun import freeze_time

from src.pipeline.analytics import AnalyticsEngine, compute_tou_cost


# TEST 1E-1: kWh accumulation formula — 1 hour of 1000W = 1 kWh
@pytest.mark.asyncio
async def test_kwh_accumulation_1kw_1hour():
    engine = AnalyticsEngine()
    # Simulate 3600 readings (1Hz) of 1000W
    for _ in range(3600):
        await engine.record("device", watts=1000.0)
    kwh = engine.get_kwh("device")
    assert abs(kwh - 1.0) < 0.01

# TEST 1E-2: Cost at peak rate — 1 kWh at $0.28 = $0.28
@freeze_time("2024-01-15 14:00:00")  # Peak period
@pytest.mark.asyncio
async def test_cost_peak_rate():
    engine = AnalyticsEngine()
    for _ in range(3600):
        await engine.record("device", watts=1000.0)
    cost = engine.get_cost("device")
    assert abs(cost - 0.28) < 0.005

# TEST 1E-3: Cost at off-peak rate — 1 kWh at $0.09 = $0.09
@freeze_time("2024-01-15 02:00:00")  # Off-peak
@pytest.mark.asyncio
async def test_cost_offpeak_rate():
    engine = AnalyticsEngine()
    for _ in range(3600):
        await engine.record("device", watts=1000.0)
    cost = engine.get_cost("device")
    assert abs(cost - 0.09) < 0.005

# TEST 1E-4: Midnight rollover — daily kWh resets at 00:00
@pytest.mark.asyncio
async def test_daily_kwh_resets_at_midnight():
    with freeze_time("2024-01-15 23:59:59") as frozen_time:
        engine = AnalyticsEngine()
        await engine.record("device", watts=1000.0)
        frozen_time.tick(delta=datetime.timedelta(seconds=2))
        kwh_after_midnight = engine.get_kwh("device")
        # Day has rolled over; counter should be near-zero
        assert kwh_after_midnight < 0.001

# TEST 1E-5: Per-device isolation
@pytest.mark.asyncio
async def test_kwh_per_device_isolated():
    engine = AnalyticsEngine()
    for _ in range(3600):
        await engine.record("device_a", watts=500.0)
    assert engine.get_kwh("device_b") == 0.0

# TEST 1E-6: ToU reward signal for RL — off-peak gives higher reward (lower cost)
def test_tou_reward_peak_vs_offpeak():
    peak_cost = compute_tou_cost(watts=1000.0, seconds=1, period="peak")
    offpeak_cost = compute_tou_cost(watts=1000.0, seconds=1, period="off-peak")
    assert peak_cost > offpeak_cost
    assert abs(peak_cost / offpeak_cost - 0.28/0.09) < 0.01  # ratio must match
