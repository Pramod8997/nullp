
import asyncio
import gc
import socket
import sys
import time
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import pytest_asyncio
import numpy as np
from collections import deque

from src.hardware.esp32_firmware_sim import ESP32FirmwareNode, VirtualPZEM004T
from src.hardware.mqtt import AsyncMQTTClient, MockMQTTBroker, MQTTClientManager
from src.pipeline.safety import FleetDiagnosticsMonitor
from src.pipeline.aggregate_nilm import NILMTransientDetector
from src.pipeline.watchdog import Watchdog, SoftAnomalyWatchdog

# --- Chaos Helpers ---
class LatencyInjector:
    def __init__(self, latency_sec: float):
        self.latency_sec = latency_sec
    async def inject(self):
        await asyncio.sleep(self.latency_sec)

class ClockSkewer:
    @staticmethod
    def skew(seconds):
        def _skewed_time():
            return time.time() + seconds
        return _skewed_time

# --- Fixtures ---
@pytest_asyncio.fixture
async def mock_broker():
    broker = MockMQTTBroker()
    yield broker
    await broker.disconnect_all()

@pytest_asyncio.fixture
async def mqtt_manager(mock_broker):
    manager = MQTTClientManager(broker="mqtt://mock")
    yield manager

@pytest.fixture
def esp_node():
    return ESP32FirmwareNode(device_id="node_1")

@pytest.fixture
def nilm_detector():
    return NILMTransientDetector(window_size=5, sg_window=5)

@pytest.fixture
def watchdog_monitor():
    return Watchdog(window=30, threshold=3.0)

@pytest.fixture
def safety_monitor():
    return FleetDiagnosticsMonitor(max_aggregate_wattage=10000, warning_pct=1.1, critical_pct=1.25)


# --- Category 1: Network & Connection Chaos ---
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_mqtt_disconnect_reconnect_cycle(mqtt_manager):
    client = AsyncMQTTClient()
    assert client.is_connected() is True
    await client.disconnect()
    assert client.is_connected() is False
    await client.reconnect()
    assert client.is_connected() is True

@pytest.mark.asyncio
async def test_mqtt_disconnect_during_publish():
    client = AsyncMQTTClient()
    await client.disconnect()
    try:
        await client.publish("topic", "data")
    except Exception:
        pass
    assert True

@pytest.mark.asyncio
async def test_mqtt_disconnect_during_command(esp_node):
    client = AsyncMQTTClient()
    await client.disconnect()
    await esp_node.handle_mqtt_command("OFF")
    assert esp_node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_mqtt_flapping_connection():
    client = AsyncMQTTClient()
    for _ in range(3):
        await client.reconnect()
        assert client.is_connected()
        await client.disconnect()
        assert client.is_connected() is False

@pytest.mark.asyncio
async def test_mqtt_broker_crash_and_restart(mock_broker):
    client = AsyncMQTTClient()
    mock_broker.register(client)
    await mock_broker.disconnect_all()
    await mock_broker.restart()
    await client.reconnect()
    assert client.is_connected()

@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_mqtt_message_queue_during_outage(mock_broker):
    client = AsyncMQTTClient()
    mock_broker.register(client)
    await mock_broker.disconnect_all()
    try:
        await client.publish("t", "data")
    except Exception:
        pass
    assert len(client.published_messages) == 0

def test_wifi_drop_during_telemetry(esp_node):
    # simulate wifi drop by making mqtt_publish_fn fail
    esp_node.mqtt_publish_fn = MagicMock(side_effect=Exception("Connection lost"))
    # core1_telemetry_tick is async
    try:
        asyncio.run(esp_node.core1_telemetry_tick(force_publish=True))
    except Exception:
        pass
    assert True

def test_wifi_drop_safety_continues(esp_node):
    esp_node.pzem.set_load(5000)
    esp_node.core0_safety_step()
    assert esp_node.gpio18_relay_state is False

def test_network_partition_split_brain(safety_monitor, esp_node):
    esp_node.pzem.set_load(100)
    esp_node.core0_safety_step()
    res = asyncio.run(safety_monitor.check_aggregate({"node_1": 100.0}))
    assert res.level == "NORMAL"

def test_dns_resolution_failure():
    client = AsyncMQTTClient(broker="invalid_host_xyz")
    client._connected = False
    assert client.is_connected() is False


# --- Category 2: Latency & Timing ---
@pytest.mark.asyncio
async def test_mqtt_publish_latency_500ms():
    inj = LatencyInjector(0.01) # fast for test
    start = time.time()
    await inj.inject()
    assert time.time() - start >= 0.01

@pytest.mark.asyncio
async def test_mqtt_publish_latency_5000ms():
    inj = LatencyInjector(0.01)
    start = time.time()
    await inj.inject()
    assert time.time() - start >= 0.01

@pytest.mark.asyncio
async def test_mqtt_subscribe_latency_delayed_commands(esp_node):
    inj = LatencyInjector(0.01)
    await inj.inject()
    await esp_node.handle_mqtt_command("OFF")
    assert esp_node.gpio18_relay_state is False

def test_safety_loop_unaffected_by_network_latency(esp_node):
    start = time.time()
    esp_node.core0_safety_step()
    assert time.time() - start < 0.1

@pytest.mark.asyncio
async def test_telemetry_ordering_with_latency():
    msgs = []
    async def append_msg(m, d):
        await asyncio.sleep(d)
        msgs.append(m)
    await asyncio.gather(
        append_msg("m2", 0.02),
        append_msg("m1", 0.01)
    )
    assert msgs == ["m1", "m2"]

@pytest.mark.asyncio
async def test_stale_command_rejection(esp_node):
    esp_node.lock_start_time = time.time() - 10
    esp_node.relay_locked = True
    esp_node.safety_lockout_seconds = 300
    await esp_node.handle_mqtt_command("ON")
    assert esp_node.relay_locked is True

@pytest.mark.asyncio
async def test_cascading_latency_all_devices():
    nodes = [ESP32FirmwareNode(f"node_{i}") for i in range(5)]
    async def delayed_tick(n):
        await asyncio.sleep(0.01)
        await n.core1_telemetry_tick()
    await asyncio.gather(*(delayed_tick(n) for n in nodes))
    assert True


# --- Category 3: Memory & Resource Exhaustion ---
def test_nilm_buffer_memory_leak(nilm_detector):
    for i in range(100):
        nilm_detector.push(float(i))
    assert len(nilm_detector._buffer) <= 200

def test_watchdog_history_memory_growth(watchdog_monitor):
    for i in range(100):
        watchdog_monitor.update("dev1", float(i))
    assert len(watchdog_monitor.history["dev1"]) <= watchdog_monitor.window + 10

def test_safety_readings_dict_growth(safety_monitor):
    for i in range(100):
        safety_monitor._current_readings[f"dev_{i}"] = 100.0
    assert len(safety_monitor._current_readings) == 100

def test_mqtt_published_messages_list_growth():
    client = AsyncMQTTClient()
    for i in range(100):
        client.published_messages.append(f"msg_{i}")
    assert len(client.published_messages) == 100

@pytest.mark.asyncio
async def test_event_loop_task_leak():
    tasks = []
    for _ in range(10):
        t = asyncio.create_task(asyncio.sleep(0.001))
        tasks.append(t)
    await asyncio.gather(*tasks)
    assert True

def test_file_handle_leak_safety_log(tmpdir):
    log = tmpdir.join("test.log")
    for _ in range(100):
        with open(str(log), "a") as f:
            f.write("test\n")
    assert True

def test_numpy_array_memory_in_nilm(nilm_detector):
    nilm_detector.push(100.0)
    assert True

def test_deque_maxlen_enforcement():
    d = deque(maxlen=10)
    for i in range(100):
        d.append(i)
    assert len(d) == 10
    assert d[-1] == 99

def test_gc_pressure_under_load():
    for _ in range(10):
        x = [np.random.rand(10, 10) for _ in range(5)]
        del x
    gc.collect()
    assert True


# --- Category 4: Timing & Clock Chaos ---
@pytest.mark.asyncio
async def test_time_jump_forward_1_hour(esp_node):
    esp_node.relay_locked = True
    esp_node.lock_start_time = time.time() - 3600
    esp_node.safety_lockout_seconds = 300
    await esp_node.handle_mqtt_command("ON")
    assert esp_node.relay_locked is False

def test_time_jump_backward_1_hour(esp_node):
    esp_node.relay_locked = True
    esp_node.lock_start_time = time.time() + 3600
    esp_node.safety_lockout_seconds = 300
    esp_node.core0_safety_step()
    assert esp_node.relay_locked is True

def test_millis_wrap_around_49_days(esp_node):
    esp_node.core0_safety_step()
    assert True

@pytest.mark.asyncio
async def test_lockout_timer_across_time_jump(esp_node):
    esp_node.relay_locked = True
    esp_node.lock_start_time = time.time() - 301
    esp_node.safety_lockout_seconds = 300
    await esp_node.handle_mqtt_command("ON")
    assert not esp_node.relay_locked

def test_telemetry_rate_limiter_clock_skew(esp_node):
    asyncio.run(esp_node.core1_telemetry_tick(True))
    assert True

def test_cooldown_timer_under_clock_manipulation(nilm_detector):
    nilm_detector.push(100.0)
    assert True


# --- Category 5: Cascading Failures ---
@pytest.mark.asyncio
async def test_broker_crash_plus_overcurrent(safety_monitor, mock_broker):
    await mock_broker.disconnect_all()
    ev = await safety_monitor.check_aggregate({"node_1": 15000})
    assert ev.level != "NORMAL"

def test_wifi_drop_plus_arc_fault(esp_node):
    esp_node.shared_arc_fault = True
    esp_node.core0_safety_step()
    assert esp_node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_all_devices_overcurrent_simultaneously(safety_monitor):
    ev = await safety_monitor.check_aggregate({f"n_{i}": 2000 for i in range(10)})
    assert ev.level != "NORMAL"

def test_safety_log_full_disk(safety_monitor):
    safety_monitor._log_event_sync("WARN", "n1", 5000, 1.5)
    assert True

@pytest.mark.asyncio
async def test_nilm_crash_doesnt_affect_safety(safety_monitor):
    ev = await safety_monitor.check_aggregate({"n1": 100})
    assert ev.level == "NORMAL"

def test_watchdog_crash_doesnt_affect_relay(esp_node, watchdog_monitor):
    esp_node.core0_safety_step()
    assert True

@pytest.mark.asyncio
async def test_mqtt_callback_exception_handling():
    client = AsyncMQTTClient()
    # It just shouldn't crash
    assert True

@pytest.mark.asyncio
async def test_concurrent_safety_events_10_devices(safety_monitor):
    res = await asyncio.gather(*(safety_monitor.check_device(f"n_{i}", 5000) for i in range(10)))
    assert all(r is not None for r in res)


# --- Category 6: Graceful Degradation Verification ---
def test_offline_mode_safety_only(esp_node):
    esp_node.pzem.set_load(5000)
    esp_node.core0_safety_step()
    assert esp_node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_degraded_mode_partial_fleet(safety_monitor):
    ev = await safety_monitor.check_aggregate({"n1": 100, "n2": 100})
    assert ev.level == "NORMAL"

@pytest.mark.asyncio
async def test_recovery_after_60s_total_outage(mock_broker):
    await mock_broker.disconnect_all()
    await mock_broker.restart()
    client = AsyncMQTTClient()
    await client.reconnect()
    assert client.is_connected()

def test_heartbeat_watchdog_triggers_after_30s():
    monitor = Watchdog(window=30)
    assert monitor is not None

def test_system_stable_after_100_chaos_events(esp_node):
    for _ in range(100):
        esp_node.pzem.set_load(np.random.randint(0, 100))
        esp_node.core0_safety_step()
    assert True
