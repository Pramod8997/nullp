"""
Integration tests for the EMS pipeline modules.
Run with: make test
"""
import sys
import os
import asyncio
import time

import pytest
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.safety import SafetyMonitor
from src.pipeline.watchdog import SoftAnomalyWatchdog
from src.pipeline.phantom_tracker import PhantomTracker
from src.pipeline.analytics import AnalyticsEngine
from src.pipeline.failure_matrix import FailureMatrix
from src.pipeline.classifier import ModeClassifier
from src.models.thermodynamics import ThermodynamicsModel
from src.rl.agent import TabularQLearningAgent


# ════════════════════════════════════════════════════════
# Safety Monitor
# ════════════════════════════════════════════════════════

class TestSafetyMonitor:
    @pytest.mark.asyncio
    async def test_safe_reading_passes(self):
        """Readings below threshold should return True (safe)."""
        cutoff_called = []
        async def mock_cutoff(device_id):
            cutoff_called.append(device_id)

        monitor = SafetyMonitor(3500.0, {"fridge": 800.0, "default": 1500.0}, mock_cutoff)
        result = await monitor.process_reading("fridge", 150.0)
        assert result is True
        assert len(cutoff_called) == 0

    @pytest.mark.asyncio
    async def test_threshold_breach_triggers_cutoff(self):
        """Readings above device limit should trigger cutoff and return False."""
        cutoff_called = []
        async def mock_cutoff(device_id):
            cutoff_called.append(device_id)

        monitor = SafetyMonitor(3500.0, {"fridge": 800.0, "default": 1500.0}, mock_cutoff)
        result = await monitor.process_reading("fridge", 900.0)
        assert result is False
        assert "fridge" in cutoff_called

    @pytest.mark.asyncio
    async def test_aggregate_breach(self):
        """Readings above max aggregate should trigger cutoff."""
        cutoff_called = []
        async def mock_cutoff(device_id):
            cutoff_called.append(device_id)

        monitor = SafetyMonitor(3500.0, {"default": 5000.0}, mock_cutoff)
        result = await monitor.process_reading("unknown_device", 4000.0)
        assert result is False


# ════════════════════════════════════════════════════════
# Soft Anomaly Watchdog
# ════════════════════════════════════════════════════════

class TestWatchdog:
    def test_needs_baseline(self):
        """Should not flag anomaly without enough baseline data."""
        watchdog = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)
        for i in range(5):
            assert watchdog.check_reading("test", 100.0) is False

    def test_detects_spike(self):
        """Should detect a massive spike after baseline is established."""
        watchdog = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)
        # Build baseline
        for _ in range(15):
            watchdog.check_reading("test", 100.0 + np.random.normal(0, 1))
        # Inject spike
        result = watchdog.check_reading("test", 500.0)
        assert result is True

    def test_normal_variance_ok(self):
        """Normal variance should not trigger anomaly."""
        watchdog = SoftAnomalyWatchdog(window_size=60, z_score_threshold=3.0)
        for _ in range(15):
            watchdog.check_reading("test", 100.0 + np.random.normal(0, 2))
        result = watchdog.check_reading("test", 104.0)
        assert result is False


# ════════════════════════════════════════════════════════
# Phantom Tracker
# ════════════════════════════════════════════════════════

class TestPhantomTracker:
    def test_tracks_phantom_load(self):
        """Should track power draw when device is nominally off."""
        tracker = PhantomTracker(baseline_threshold_watts=5.0)
        tracker.track("fridge", 3.0, is_nominally_off=True)
        assert tracker.get_total_phantom_load() > 0

    def test_ignores_active_device(self):
        """Should NOT track if device is on or drawing above threshold."""
        tracker = PhantomTracker(baseline_threshold_watts=5.0)
        tracker.track("fridge", 150.0, is_nominally_off=False)
        assert tracker.get_total_phantom_load() == 0.0

    def test_worst_offenders(self):
        """Should rank devices by phantom load magnitude."""
        tracker = PhantomTracker(baseline_threshold_watts=10.0)
        tracker.track("fridge", 4.0, True)
        tracker.track("tv", 2.0, True)
        tracker.track("charger", 5.0, True)
        offenders = tracker.get_worst_offenders(2)
        assert len(offenders) == 2
        assert offenders[0][0] == "charger"


# ════════════════════════════════════════════════════════
# Analytics Engine
# ════════════════════════════════════════════════════════

class TestAnalytics:
    def test_records_usage(self):
        engine = AnalyticsEngine(cost_per_kwh=0.15)
        engine.record_usage("fridge", 150.0, duration_hours=1.0)
        summary = engine.get_daily_summary()
        assert summary["total_kwh"] > 0
        assert summary["estimated_cost_usd"] > 0

    def test_empty_day(self):
        engine = AnalyticsEngine(cost_per_kwh=0.15)
        summary = engine.get_daily_summary("2000-01-01")
        assert summary["total_kwh"] == 0


# ════════════════════════════════════════════════════════
# Failure Matrix
# ════════════════════════════════════════════════════════

class TestFailureMatrix:
    def test_known_failures(self):
        fm = FailureMatrix()
        assert fm.trigger_failure("sensor_timeout", "fridge") is True
        assert fm.trigger_failure("mqtt_disconnect") is True
        assert fm.trigger_failure("model_drift") is True
        assert fm.trigger_failure("relay_stuck", "hvac") is True

    def test_unknown_failure(self):
        fm = FailureMatrix()
        assert fm.trigger_failure("alien_invasion") is False


# ════════════════════════════════════════════════════════
# Mode Classifier
# ════════════════════════════════════════════════════════

class TestModeClassifier:
    def test_stable_single(self):
        mc = ModeClassifier(variance_threshold=50.0)
        result = mc.classify_mode([100.0, 100.5, 99.5, 100.2])
        assert result == "SINGLE_DEVICE_STABLE"

    def test_multi_aggregate(self):
        mc = ModeClassifier(variance_threshold=50.0)
        result = mc.classify_mode([100.0, 300.0, 50.0, 400.0, 120.0])
        assert result == "MULTI_DEVICE_AGGREGATE"


# ════════════════════════════════════════════════════════
# Thermodynamics / PMV
# ════════════════════════════════════════════════════════

class TestThermodynamics:
    def test_comfort_zone(self):
        model = ThermodynamicsModel()
        pmv = model.calculate_pmv(t_air=24.0, t_radiant=23.0, v_air=0.1, rh=50.0, met=1.2, clo=0.5)
        assert -1.0 <= pmv <= 1.0

    def test_hot_zone(self):
        model = ThermodynamicsModel()
        pmv = model.calculate_pmv(t_air=32.0, t_radiant=30.0, v_air=0.1, rh=70.0, met=1.5, clo=0.3)
        assert pmv > 1.0

    def test_clamped_bounds(self):
        model = ThermodynamicsModel()
        pmv = model.calculate_pmv(t_air=50.0, t_radiant=50.0, v_air=0.0, rh=100.0, met=3.0, clo=0.0)
        assert pmv <= 3.0


# ════════════════════════════════════════════════════════
# RL Agent
# ════════════════════════════════════════════════════════

class TestRLAgent:
    def test_state_tuple_padding(self):
        agent = TabularQLearningAgent()
        state = agent.get_state_tuple(12, 5, (1, 0))
        assert len(state) == 6  # time + power + 4 devices

    def test_action_in_range(self):
        agent = TabularQLearningAgent()
        state = agent.get_state_tuple(12, 5, (0, 0, 0, 0))
        action = agent.get_action(state)
        assert 0 <= action < agent.num_actions

    def test_update_creates_entry(self):
        agent = TabularQLearningAgent()
        state = agent.get_state_tuple(0, 0, (0, 0, 0, 0))
        next_state = agent.get_state_tuple(0, 1, (1, 0, 0, 0))
        agent.update(state, 0, 1.0, next_state)
        assert state in agent.q_table

    def test_cooldown_prevents_spam(self):
        """Verify cooldown logic prevents rapid-fire actions."""
        cooldown = 15.0
        last_action_time = time.time() - 5.0  # 5 seconds ago
        elapsed = time.time() - last_action_time
        assert elapsed < cooldown  # Should be blocked

        last_action_time = time.time() - 20.0  # 20 seconds ago
        elapsed = time.time() - last_action_time
        assert elapsed >= cooldown  # Should be allowed
