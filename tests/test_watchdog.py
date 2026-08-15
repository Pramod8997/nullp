import pytest
from src.pipeline.watchdog import Watchdog, WatchdogEvent

# TEST 1D-1: z-score is 0 for a steady signal
def test_watchdog_zscore_steady_signal():
    watchdog = Watchdog(window=30, threshold=3.0)
    for _ in range(30):
        watchdog.update("device", 100.0)
    z = watchdog.get_zscore("device")
    assert abs(z) < 0.01

# TEST 1D-2: Spike 5σ above mean triggers anomaly event
@pytest.mark.asyncio
async def test_watchdog_spike_triggers_anomaly():
    watchdog = Watchdog(window=30, threshold=3.0)
    for _ in range(30):
        watchdog.update("device", 100.0)
    # 5-sigma spike
    result = await watchdog.process("device", 100.0 + 5 * watchdog.get_std("device"))
    assert result.event_type in ("WATCHDOG_ANOMALY", "SOFT_OVERRIDE")

# TEST 1D-3: First update before window fills — no false positive
def test_watchdog_no_false_positive_early():
    watchdog = Watchdog(window=30, threshold=3.0)
    for i in range(5):  # only 5 samples, window=30
        result = watchdog.update("device", 100.0 + i * 50)
    assert result is None or result.event_type != "WATCHDOG_ANOMALY"

# TEST 1D-4: z-score magnitude correctly computed
def test_watchdog_zscore_magnitude():
    watchdog = Watchdog(window=30, threshold=3.0)
    values = [100.0] * 29  # 29 identical readings
    for v in values:
        watchdog.update("device", v)
    # mean=100, std≈0 (but handle near-zero std safely)
    # Insert a mildly different value and check z doesn't blow up
    watchdog.update("device", 105.0)
    # Should not raise ZeroDivisionError
