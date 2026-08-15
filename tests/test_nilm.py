import pytest
import numpy as np

from src.pipeline.aggregate_nilm import savitzky_golay, detect_transients
from src.pipeline.delta_stability import DeltaStabilityTracker


# TEST 1B-1: SG filter smooths Gaussian noise without destroying step edges
def test_sg_filter_preserves_step_edge():
    # Synthetic: 100 samples at 100W, then 100 samples at 300W (200W step)
    signal = np.array([100.0]*100 + [300.0]*100, dtype=float)
    noise = np.random.normal(0, 3, 200)
    noisy = signal + noise
    filtered = savitzky_golay(noisy, window_length=11, polyorder=2)
    # After SG, the derivative at sample 100 should be >> 0; far from edge ≈ 0
    derivative = np.diff(filtered)
    assert derivative[99] > 30.0, "SG derivative should detect the 200W step"
    assert abs(derivative[50]) < 5.0, "SG derivative should be near 0 away from edge"

# TEST 1B-2: Transient below 50W threshold must NOT trigger classification
def test_nilm_below_threshold_no_event():
    events = detect_transients(power_window=[200.0]*50 + [240.0]*50)
    assert len(events) == 0, "49W step should not trigger NILM event"

# TEST 1B-3: Transient exactly at 50W threshold MUST trigger
def test_nilm_exact_threshold_triggers():
    events = detect_transients(power_window=[200.0]*50 + [250.0]*50)
    assert len(events) == 1

# TEST 1B-4: Multiple device events in one window — all detected
def test_nilm_multiple_events():
    # Fridge ON (+200W), then kettle ON (+2500W) in same window
    signal = [0.0]*20 + [200.0]*20 + [2700.0]*20
    events = detect_transients(signal)
    assert len(events) == 2

# TEST 1B-5: Negative step (device OFF) also triggers transient detection
def test_nilm_negative_step_detected():
    signal = [300.0]*50 + [100.0]*50  # 200W drop
    events = detect_transients(signal)
    assert len(events) == 1
    assert events[0].delta < 0  # negative transition

# TEST 1B-6: Delta stability buffer — accumulates unknown embeddings correctly
# After 3+ identical (stable) unknown embeddings, LABEL_REQUEST must be emitted
@pytest.mark.asyncio
async def test_delta_stability_emits_label_request_after_min_occurrences():
    tracker = DeltaStabilityTracker(buffer_size=10, std_threshold=3.0, min_occurrences=3)
    embedding = np.random.randn(128).astype(np.float32)
    # Inject nearly identical embeddings 3 times
    for _ in range(3):
        result = await tracker.process(embedding + np.random.normal(0, 0.01, 128))
    assert result is not None and result.event_type == "LABEL_REQUEST"

# TEST 1B-7: Unstable embeddings (high std) should NOT emit LABEL_REQUEST
@pytest.mark.asyncio
async def test_delta_stability_no_emit_on_high_variance():
    tracker = DeltaStabilityTracker(buffer_size=10, std_threshold=3.0, min_occurrences=3)
    for _ in range(10):
        # Random embeddings each time — high variance
        result = await tracker.process(np.random.randn(128).astype(np.float32))
    assert result is None or result.event_type != "LABEL_REQUEST"

# TEST 1B-8: Buffer size cap — oldest embeddings are evicted when buffer is full
def test_delta_stability_buffer_capped_at_max():
    tracker = DeltaStabilityTracker(buffer_size=10, std_threshold=3.0, min_occurrences=3)
    for i in range(15):
        tracker._add_to_buffer(np.ones(128) * i)
    assert len(tracker.buffer) == 10
