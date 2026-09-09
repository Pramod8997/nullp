import sys
import os
import asyncio
import builtins
import logging
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.safety import SafetyMonitor, load_config, slow_rl_agent
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


# ════════════════════════════════════════════════════════════════════
# Branch coverage of the FleetDiagnosticsMonitor (src/pipeline/safety.py)
# ════════════════════════════════════════════════════════════════════

class MockMessage:
    """Minimal aiomqtt message stand-in: topic string + bytes/str payload."""

    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


class ScriptedMQTTClient:
    """Yields a scripted message sequence, then idles until the task is
    cancelled (so run_forever's CancelledError path is exercised)."""

    def __init__(self, messages):
        self._messages = messages

    @property
    async def messages(self):
        for m in self._messages:
            yield m
        while True:
            await asyncio.sleep(0.05)


async def _drive_monitor(monitor, messages, settle=0.3):
    """Run monitor.run_forever over a scripted stream; return relay callbacks."""
    callbacks = []

    async def relay_callback(device_id, action):
        callbacks.append((device_id, action))

    task = asyncio.create_task(
        monitor.run_forever(ScriptedMQTTClient(messages), relay_callback))
    await asyncio.sleep(settle)  # let the scripted messages drain
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return callbacks


# ── load_config branches ─────────────────────────────────────────────

def test_load_config_explicit_path(tmp_path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text("system_safety:\n  max_aggregate_wattage: 1234.0\n")
    cfg = load_config(str(cfg_file))
    assert cfg["system_safety"]["max_aggregate_wattage"] == 1234.0


def test_load_config_explicit_empty_file_returns_empty_dict(tmp_path):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    assert load_config(str(cfg_file)) == {}


def test_load_config_search_path_empty_yaml_returns_empty(tmp_path, monkeypatch):
    # yaml.safe_load("") -> None -> `or {}` must yield an empty dict
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("")
    assert load_config() == {}


def test_load_config_invalid_yaml_falls_back_to_default(tmp_path, monkeypatch):
    # Unparseable file in the search path must not raise; defaults are used
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("{ not valid yaml ][")
    cfg = load_config()
    assert cfg["system_safety"]["max_aggregate_wattage"] == 3500.0
    assert cfg["devices"]["node_fridge"]["tier0"] is True


def test_load_config_no_files_found_returns_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["system_safety"]["critical_pct"] == 1.25
    assert cfg["system_safety"]["device_wattage_limits"]["default"] == 1500.0


# ── slow_rl_agent / process ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_slow_rl_agent_returns_empty_list():
    assert await slow_rl_agent(delay=0.01) == []


@pytest.mark.asyncio
async def test_process_is_passthrough():
    monitor = SafetyMonitor()
    evt = object()
    assert await monitor.process(evt) is evt


# ── check_aggregate: WARNING band + async broadcast ──────────────────

@pytest.mark.asyncio
async def test_aggregate_warning_band():
    # warning_pct < 1 makes the WARNING band reachable ([800, 1000] here)
    monitor = SafetyMonitor(max_aggregate_wattage=1000.0,
                            warning_pct=0.8, critical_pct=1.25)
    evt = await monitor.check_aggregate({"a": 850.0})
    assert evt.level == "WARNING"
    assert evt.event_type == "SAFETY_ALERT"


@pytest.mark.asyncio
async def test_aggregate_critical_awaits_async_broadcast():
    events = []

    async def broadcast(evt):
        events.append(evt)

    monitor = SafetyMonitor(max_aggregate_wattage=100.0, broadcast_fn=broadcast)
    evt = await monitor.check_aggregate({"a": 150.0})
    assert evt.level == "CRITICAL"
    assert events == [evt]  # awaited coroutine broadcast fired exactly once


# ── check_roc: broadcast dispatch ────────────────────────────────────

@pytest.mark.asyncio
async def test_arc_fault_dispatches_to_async_and_sync_broadcast():
    events = []

    async def async_broadcast(evt):
        events.append(("async", evt))

    monitor = SafetyMonitor(broadcast_fn=async_broadcast)
    evt = await monitor.check_roc("node_kettle", 0.0, 2000.0, dt_seconds=1.0)
    assert evt.event_type == "ARC_FAULT"
    assert events == [("async", evt)]

    sync_events = []
    monitor2 = SafetyMonitor(broadcast_fn=lambda e: sync_events.append(e))
    evt2 = await monitor2.check_roc("node_kettle", 0.0, 2000.0, dt_seconds=1.0)
    assert evt2.event_type == "ARC_FAULT"
    assert sync_events == [evt2]


# ── check_device: CRITICAL / normal / degenerate-rated branches ──────

@pytest.mark.asyncio
async def test_check_device_critical():
    monitor = SafetyMonitor(device_wattage_limits={"node_kettle": 1000.0,
                                                  "default": 1500.0})
    evt = await monitor.check_device("node_kettle", 1300.0)  # 130% >= 125%
    assert evt is not None and evt.level == "CRITICAL"
    assert evt.details["pct"] == pytest.approx(1.3)


@pytest.mark.asyncio
async def test_check_device_within_limit_returns_none():
    monitor = SafetyMonitor(device_wattage_limits={"node_kettle": 1000.0})
    assert await monitor.check_device("node_kettle", 500.0) is None


@pytest.mark.asyncio
async def test_check_device_unknown_device_uses_default_limit():
    monitor = SafetyMonitor(device_wattage_limits={"default": 1000.0})
    evt = await monitor.check_device("who_dis", 2000.0)  # 200% of default
    assert evt is not None and evt.level == "CRITICAL"


@pytest.mark.asyncio
async def test_check_device_zero_rated_limit_treated_as_pct_one():
    # rated <= 0 -> pct guards to 1.0; any positive power > rated -> WARNING
    monitor = SafetyMonitor(device_wattage_limits={"ghost": 0.0},
                            warning_pct=1.10, critical_pct=1.25)
    evt = await monitor.check_device("ghost", 5.0)
    assert evt is not None and evt.level == "WARNING"


@pytest.mark.asyncio
async def test_check_device_warning_on_pct_not_power():
    # power <= rated but pct >= warning_pct (0.5) still warns
    monitor = SafetyMonitor(device_wattage_limits={"node_hvac": 1000.0},
                            warning_pct=0.5)
    evt = await monitor.check_device("node_hvac", 600.0)
    assert evt is not None and evt.level == "WARNING"


# ── run_forever monitor loop ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_forever_warning_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # safety_events.log lands in tmp
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 1000.0, "default": 1500.0},
        warning_pct=1.10, critical_pct=1.25)
    callbacks = await _drive_monitor(
        monitor, [MockMessage("home/sensor/node_kettle/power", b"1100.0")])
    assert ("node_kettle", "WARNING") in callbacks
    log = (tmp_path / "safety_events.log").read_text()
    assert "WARNING,node_kettle,1100.0" in log


@pytest.mark.asyncio
async def test_run_forever_critical_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 1000.0, "default": 1500.0},
        warning_pct=1.10, critical_pct=1.25)
    callbacks = await _drive_monitor(
        monitor, [MockMessage("home/sensor/node_kettle/power", b"1300.0")])
    assert ("node_kettle", "ALERT_CRITICAL") in callbacks
    log = (tmp_path / "safety_events.log").read_text()
    assert "CRITICAL,node_kettle,1300.0" in log


@pytest.mark.asyncio
async def test_run_forever_arc_fault_with_inrush_suppression(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 5000.0, "default": 5000.0},
        warning_pct=1.10, critical_pct=1.25)

    # Baseline >= 50 W then a >1000 W jump -> arc-fault alert
    callbacks = await _drive_monitor(monitor, [
        MockMessage("home/sensor/node_kettle/power", b"200.0"),
        MockMessage("home/sensor/node_kettle/power", b"1500.0"),
    ])
    assert ("node_kettle", "ALERT_ARC_FAULT") in callbacks
    log = (tmp_path / "safety_events.log").read_text()
    assert "ARC_FAULT,node_kettle,1500.0" in log

    # Same jump from a < 50 W baseline is normal inrush: suppressed
    # (fresh monitor — _prev_readings persists across runs by design)
    monitor2 = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 5000.0, "default": 5000.0},
        warning_pct=1.10, critical_pct=1.25)
    callbacks = await _drive_monitor(monitor2, [
        MockMessage("home/sensor/node_kettle/power", b"10.0"),
        MockMessage("home/sensor/node_kettle/power", b"1500.0"),
    ])
    assert ("node_kettle", "ALERT_ARC_FAULT") not in callbacks


@pytest.mark.asyncio
async def test_run_forever_rejects_bad_payloads_and_topics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 1000.0, "default": 1500.0},
        warning_pct=1.10, critical_pct=1.25)
    callbacks = await _drive_monitor(monitor, [
        MockMessage("home/sensor/node_kettle/power", b"nan"),    # NaN -> skip
        MockMessage("home/sensor/node_kettle/power", b"-inf"),   # inf -> skip
        MockMessage("home/sensor/node_kettle/power", b"abc"),    # ValueError
        MockMessage("home/other/topic", b"1200.0"),              # not /power
        MockMessage("sensor/power", b"1200.0"),                  # parts < 3
        MockMessage("home/sensor/node_kettle/power", "900.0"),   # str payload
        MockMessage("home/sensor/node_kettle/power", b"-1200.0"),  # neg -> abs
    ])
    # Only the negative reading survives sanitisation (abs -> 1200 W = 120%)
    assert callbacks == [("node_kettle", "WARNING")]
    # Invalid readings never entered the fleet aggregate map
    assert monitor._current_readings == {"node_kettle": 1200.0}


@pytest.mark.asyncio
async def test_run_forever_fleet_aggregate_ceiling(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=1000.0,
        device_wattage_limits={"default": 5000.0},  # unknown devices -> default
        warning_pct=1.10, critical_pct=1.25)
    with caplog.at_level(logging.WARNING, logger="src.pipeline.safety"):
        callbacks = await _drive_monitor(monitor, [
            MockMessage("home/sensor/dev_a/power", b"600.0"),
            MockMessage("home/sensor/dev_b/power", b"600.0"),
        ])
    assert callbacks == []  # individually fine (12% of default limit)
    assert "FLEET AGGREGATE" in caplog.text  # 1200 W > 1000 W ceiling


@pytest.mark.asyncio
async def test_run_forever_exits_cleanly_when_stream_ends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"node_kettle": 1000.0, "default": 1500.0},
        warning_pct=1.10, critical_pct=1.25)

    class EndingMQTTClient:
        @property
        async def messages(self):
            yield MockMessage("home/sensor/node_kettle/power", b"100.0")

    callbacks = []

    async def relay_callback(device_id, action):
        callbacks.append((device_id, action))

    # Stream ends -> async-for completes -> run_forever returns on its own
    await monitor.run_forever(EndingMQTTClient(), relay_callback)
    assert callbacks == []  # 100 W on 1000 W limit: no alert


@pytest.mark.asyncio
async def test_run_forever_survives_broker_stream_error(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor(
        max_aggregate_wattage=10000.0,
        device_wattage_limits={"default": 5000.0},
        warning_pct=1.10, critical_pct=1.25)

    class ExplodingMQTTClient:
        @property
        async def messages(self):
            yield MockMessage("home/sensor/dev_a/power", b"100.0")
            raise RuntimeError("broker stream exploded")

    callbacks = []

    async def relay_callback(device_id, action):
        callbacks.append((device_id, action))

    with caplog.at_level(logging.ERROR, logger="src.pipeline.safety"):
        # Must terminate cleanly (error logged), never propagate the exception
        await monitor.run_forever(ExplodingMQTTClient(), relay_callback)
    assert "Fleet Diagnostics Monitor error" in caplog.text
    assert callbacks == []


# ── safety event log write paths ─────────────────────────────────────

def test_log_event_sync_writes_csv_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    SafetyMonitor()._log_event_sync("WARNING", "node_kettle", 1200.0, 1.2)
    content = (tmp_path / "safety_events.log").read_text()
    assert "WARNING,node_kettle,1200.0,1.2" in content


def test_log_event_sync_swallows_write_errors(monkeypatch, caplog):
    real_open = builtins.open

    def failing_open(file, *args, **kwargs):
        if str(file) == "safety_events.log":
            raise OSError("disk full")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with caplog.at_level(logging.ERROR, logger="src.pipeline.safety"):
        SafetyMonitor()._log_event_sync("CRITICAL", "x", 1.0, 1.0)
    assert "Failed to write safety event log" in caplog.text


@pytest.mark.asyncio
async def test_log_event_async_offloads_to_thread(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monitor = SafetyMonitor()
    await monitor._log_event_async("WARNING", "node_kettle", 1200.0, 1.2)
    content = (tmp_path / "safety_events.log").read_text()
    assert "WARNING,node_kettle,1200.0,1.2" in content
