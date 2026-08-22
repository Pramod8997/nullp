from src.pipeline.phantom_tracker import PhantomTracker


# TEST 1C-1: EMA formula correctness — single update
def test_phantom_ema_single_step():
    alpha = 0.1  # read from config
    ema_old = 0.0
    reading = 10.0  # 10W phantom when OFF
    ema_new = alpha * reading + (1 - alpha) * ema_old
    tracker = PhantomTracker(alpha=alpha)
    tracker.update("node_fridge", power=10.0, state="OFF")
    assert abs(tracker.get_ema("node_fridge") - ema_new) < 0.001

# TEST 1C-2: EMA converges to true value after many updates
def test_phantom_ema_convergence():
    tracker = PhantomTracker(alpha=0.1)
    true_phantom = 8.5
    for _ in range(500):
        tracker.update("esp32_tv", power=true_phantom, state="OFF")
    assert abs(tracker.get_ema("esp32_tv") - true_phantom) < 0.1

# TEST 1C-3: Phantom tracker should NOT count readings when device is "ON"
def test_phantom_not_accumulated_when_on():
    tracker = PhantomTracker(alpha=0.1)
    # Set baseline phantom
    tracker.update("esp32_tv", power=5.0, state="OFF")
    baseline = tracker.get_ema("esp32_tv")
    # Device turns ON with high wattage — should NOT update phantom EMA
    tracker.update("esp32_tv", power=150.0, state="ON")
    assert tracker.get_ema("esp32_tv") == baseline

# TEST 1C-4: New device starts with EMA=0
def test_phantom_new_device_starts_at_zero():
    tracker = PhantomTracker(alpha=0.1)
    assert tracker.get_ema("new_device") == 0.0

# TEST 1C-5: Per-device isolation — updates to one device don't affect another
def test_phantom_device_isolation():
    tracker = PhantomTracker(alpha=0.1)
    for _ in range(50):
        tracker.update("device_a", power=10.0, state="OFF")
    assert tracker.get_ema("device_b") == 0.0
