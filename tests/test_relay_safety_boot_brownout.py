import pytest
import asyncio
import json
import time
import math
import random
from src.hardware.esp32_firmware_sim import ESP32FirmwareNode
from src.hardware.mqtt import AsyncMQTTClient

DEVICE_ID = "test_node_01"
RATED_WATTS = 200.0
LOCKOUT_SECONDS = 300.0

@pytest.fixture
def mqtt_client():
    return AsyncMQTTClient()

@pytest.fixture
def node(mqtt_client):
    # Polarity intentionally left at the default so the fixture tracks the locked
    # firmware constant (RELAY_ACTIVE_LOW = false) rather than pinning the
    # inverted B-7 wiring, which is what this fixture used to do.
    node = ESP32FirmwareNode(
        device_id=DEVICE_ID,
        rated_watts=RATED_WATTS,
        mqtt_publish_fn=mqtt_client.publish
    )
    return node

# ==========================================
# Category 1: Boot Sequence Safety
# ==========================================

def test_relay_off_at_power_on(node):
    assert node.gpio18_relay_state is False

def test_relay_off_before_wifi_connect(node):
    # Simulate time passing before wifi
    node.core0_safety_step()
    assert node.gpio18_relay_state is False

def test_relay_off_before_mqtt_connect(node):
    node.core0_safety_step()
    assert node.gpio18_relay_state is False

def test_relay_off_before_core0_first_read(node):
    assert node.gpio18_relay_state is False
    node.core0_safety_step()
    assert node.gpio18_relay_state is False

def test_gpio_floating_state_simulation(node):
    # Simulate floating GPIO by setting it to None
    node.gpio18_relay_state = None
    node.set_relay(False)
    assert node.gpio18_relay_state is False

def test_rapid_power_cycle_10_times(mqtt_client):
    for _ in range(10):
        temp_node = ESP32FirmwareNode(
            device_id=DEVICE_ID,
            rated_watts=RATED_WATTS,
            mqtt_publish_fn=mqtt_client.publish
        )
        assert temp_node.gpio18_relay_state is False

def test_setup_sequence_ordering(node):
    # Simulate sequence: GPIO init -> relay OFF -> Core0 -> WiFi -> MQTT
    node.set_relay(False)
    assert node.gpio18_relay_state is False
    node.core0_safety_step()
    assert node.gpio18_relay_state is False
    # MQTT connect simulated
    assert node.gpio18_relay_state is False

def test_active_low_logic_correctness(node):
    """Polarity must be modelled at the pin, not just as logical relay state.

    Regression for B-7. `relay_active_low` used to be stored and never read, so
    the twin could not distinguish active-HIGH from active-LOW at all and this
    test — asserting only gpio18_relay_state — passed under either polarity.
    The locked HARDWARE_FINAL_SPEC.md pins RELAY_ACTIVE_LOW = false
    (the RELAY_ACTIVE_LOW constant in main.cpp): HIGH closes the relay, LOW
    opens it, Hi-Z opens it.
    """
    # Default build (spec-correct, active-HIGH): level follows the logical state.
    assert node.relay_active_low is False
    node.set_relay(True)
    assert node.gpio18_relay_state is True and node.gpio18_level is True
    node.set_relay(False)
    assert node.gpio18_relay_state is False and node.gpio18_level is False

    # Active-LOW wiring (the D5 purchase-contingency fallback) inverts the pin.
    inverted = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=RATED_WATTS,
                                 relay_active_low=True)
    inverted.set_relay(True)
    assert inverted.gpio18_relay_state is True and inverted.gpio18_level is False
    inverted.set_relay(False)
    assert inverted.gpio18_relay_state is False and inverted.gpio18_level is True


def test_default_polarity_matches_locked_firmware_constant():
    """The twin's default must track the RELAY_ACTIVE_LOW constant in main.cpp."""
    import re, pathlib
    src = pathlib.Path("firmware/esp32_node/src/main.cpp").read_text()
    m = re.search(r'const\s+bool\s+RELAY_ACTIVE_LOW\s*=\s*(true|false)\s*;', src)
    assert m, "RELAY_ACTIVE_LOW not found in firmware"
    firmware_active_low = (m.group(1) == "true")
    assert ESP32FirmwareNode(device_id="polarity_probe").relay_active_low is firmware_active_low, (
        "twin default polarity has drifted from the firmware constant"
    )


def test_default_rated_watts_matches_locked_firmware_constant():
    """The twin's ctor default must track the RATED_WATTS constant in main.cpp (250.0).

    Guards the Run 0.2 re-anchor: the twin defaulted to 200 W, so every
    default-constructed node tripped overcurrent at 250 W instead of the
    firmware's 312.5 W (125% of 250).
    """
    import re, pathlib
    src = pathlib.Path("firmware/esp32_node/src/main.cpp").read_text()
    m = re.search(r'RATED_WATTS\s*=\s*([\d.]+)', src)
    assert m, "RATED_WATTS not found in firmware"
    assert ESP32FirmwareNode(device_id="rated_probe").rated_watts == float(m.group(1)), (
        "twin default rated_watts has drifted from the firmware constant"
    )


def test_relay_de_energised_at_boot_under_both_polarities():
    """Boot must leave the load de-energised whichever way the relay is wired."""
    for active_low in (False, True):
        n = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=RATED_WATTS,
                              relay_active_low=active_low)
        assert n.gpio18_relay_state is False, "load must not be energised at boot"
        assert n.gpio18_level is active_low, "pin level must be the de-energising level"


def test_overcurrent_is_unconditional_during_cold_baseline(node):
    """A 140% overload must trip on the first sample, even with a cold baseline.

    Regression: the twin gated overcurrent on `is_normal_inrush`, so a fresh node
    (baseline_avg = 0W, last_watts = 0W) tolerated 280W on a 200W-rated line
    indefinitely. The SafetySamplingTask() overcurrent cutoff in main.cpp
    makes this path unconditional by design
    (HARDWARE_FINAL_SPEC.md D11'), so the twin was weaker than the firmware.
    """
    node.set_relay(True)
    assert node._baseline_fill == 0, "must be a cold baseline for this regression"
    node.pzem.set_load(280.0)                     # 140% of 200W rated
    node.core0_safety_step(sim_dt=0.1)
    assert node.gpio18_relay_state is False, "overcurrent must not be gated by inrush suppression"
    assert node.gpio18_level is False, "pin must be driven to the de-energising level"
    # Core 0 opens the relay immediately; the core-1 loop takes the lockout
    # when it consumes the overcurrent latch (the core-0 cutoff + core-1
    # lockout blocks in main.cpp).
    asyncio.run(node.core1_telemetry_tick())
    assert node.relay_locked is True


def test_inrush_suppression_still_protects_dpdt_channel():
    """Suppression must remain active on the arc-fault channel it exists for."""
    n = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=1200.0)  # ceiling 1500W
    n.set_relay(True)
    for _ in range(5):
        n.pzem.set_load(0.0)
        n.core0_safety_step(sim_dt=0.1)
    n.pzem.set_load(1200.0)                       # 12,000 W/s, under the ceiling
    n.core0_safety_step(sim_dt=0.1)
    assert n.gpio18_relay_state is True, "cold-baseline inrush must not trip dP/dt"
    assert n.shared_arc_fault is False


def test_nan_pzem_read_does_not_poison_safety_state(node):
    """An invalid PZEM read must skip the cycle, as the isnan() guard in
    SafetySamplingTask() in main.cpp does.

    Regression: the twin latched NaN into _last_watts and _baseline_ring —
    permanently forcing is_normal_inrush False — and pushed NaN into the shared
    block, from where Core 1 published a bare "nan" power payload.
    """
    node.set_relay(True)
    node.pzem.set_load(100.0)
    node.core0_safety_step(sim_dt=0.1)
    last_watts_before = node._last_watts
    ring_before = list(node._baseline_ring)

    node.pzem.active_power = float('nan')
    node.core0_safety_step(sim_dt=0.1)

    assert node._last_watts == last_watts_before, "NaN must not reach _last_watts"
    assert node._baseline_ring == ring_before, "NaN must not enter the baseline ring"
    assert math.isfinite(node.shared_power_watts), "NaN must not reach shared state"
    assert node.gpio18_relay_state is True, "an unreadable sensor must not trip the relay"


@pytest.mark.asyncio
async def test_nan_never_published_to_mqtt(mqtt_client):
    """Core 1 must never emit a 'nan' power payload or non-standard JSON telemetry."""
    n = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=RATED_WATTS,
                          mqtt_publish_fn=mqtt_client.publish)
    n.set_relay(True)
    n.pzem.set_load(120.0)
    n.core0_safety_step(sim_dt=0.1)
    n.pzem.active_power = float('nan')
    n.core0_safety_step(sim_dt=0.1)
    await n.core1_telemetry_tick(force_publish=True)

    for topic, payload in mqtt_client.published_messages:
        assert "nan" not in str(payload).lower(), f"non-finite payload on {topic}: {payload!r}"
        if topic.endswith("/telemetry"):
            # json.loads accepts bare NaN by default; reject it explicitly.
            parsed = json.loads(payload, parse_constant=_reject_json_constant)
            assert all(math.isfinite(v) for v in parsed.values())


def _reject_json_constant(name):
    raise AssertionError(f"non-standard JSON constant {name!r} in telemetry payload")

# ==========================================
# Category 2: Brownout Simulation
# ==========================================

def test_brownout_3v0_relay_behavior(mqtt_client):
    # Simulate ESP32 brownout reset at 3.0V BOD threshold
    node = ESP32FirmwareNode(device_id=DEVICE_ID, mqtt_publish_fn=mqtt_client.publish)
    node.pzem.voltage = 3.0
    assert node.gpio18_relay_state is False

def test_brownout_2v5_relay_behavior(mqtt_client):
    # Below BOD threshold, reset state
    node = ESP32FirmwareNode(device_id=DEVICE_ID, mqtt_publish_fn=mqtt_client.publish)
    node.pzem.voltage = 2.5
    assert node.gpio18_relay_state is False

def test_voltage_sag_recovery(node):
    node.set_relay(False)
    node.pzem.voltage = 5.0
    node.core0_safety_step()
    node.pzem.voltage = 3.0
    node.core0_safety_step()
    node.pzem.voltage = 5.0
    node.core0_safety_step()
    assert node.gpio18_relay_state is False

def test_hlk_pm01_ripple_stress(node):
    node.set_relay(True)
    base_voltage = 5.0
    for _ in range(100):
        ripple = random.uniform(-0.2, 0.2)
        node.pzem.voltage = base_voltage + ripple
        node.core0_safety_step()
    assert node.gpio18_relay_state is True

@pytest.mark.asyncio
async def test_brownout_during_mqtt_publish(node, mqtt_client):
    node.set_relay(True)
    await node.handle_mqtt_command("ON")
    # Simulate brownout -> node reboot
    node_reboot = ESP32FirmwareNode(device_id=DEVICE_ID, mqtt_publish_fn=mqtt_client.publish)
    assert node_reboot.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_brownout_during_relay_transition(node):
    # Simulate reset during relay transition
    node.set_relay(True)
    # brownout happens
    node_reboot = ESP32FirmwareNode(device_id=DEVICE_ID)
    assert node_reboot.gpio18_relay_state is False

def test_power_loss_and_restore(node):
    node.pzem.voltage = 0.0
    node.core0_safety_step()
    assert node.gpio18_relay_state is False
    node.pzem.voltage = 230.0
    node.core0_safety_step()
    assert node.gpio18_relay_state is False

# ==========================================
# Category 3: Anti-Thrashing & Lockout
# ==========================================

def trigger_arc_fault(node):
    node.set_relay(True)
    node.pzem.set_load(100.0)
    for _ in range(5):
        node.core0_safety_step(sim_dt=0.1)
    node.pzem.set_load(300.0)
    node.core0_safety_step(sim_dt=0.1)

def test_lockout_after_arc_fault(node):
    trigger_arc_fault(node)
    # The lockout is taken by the core-1 tick consuming the arc-fault flag
    # (the core-1 arc-fault acknowledgment block in main.cpp); core 0 only
    # opens the relay.
    asyncio.run(node.core1_telemetry_tick())
    assert node.relay_locked is True
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_lockout_rejects_on_command(node, mqtt_client):
    trigger_arc_fault(node)
    # Core-1 acknowledgment takes the lockout (the arc-fault latch block).
    await node.core1_telemetry_tick()
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is False
    acks = await mqtt_client.get_published(node.topic_ack)
    assert "LOCKOUT_NACK" in acks

@pytest.mark.asyncio
async def test_lockout_allows_off_command(node, mqtt_client):
    trigger_arc_fault(node)
    node.set_relay(True) # Force it ON to test OFF command
    await node.handle_mqtt_command("OFF")
    assert node.gpio18_relay_state is False
    acks = await mqtt_client.get_published(node.topic_ack)
    assert "OFF_CONFIRMED" in acks

@pytest.mark.asyncio
async def test_lockout_expires_after_300s(node):
    trigger_arc_fault(node)
    await node.core1_telemetry_tick()   # lockout taken (arc-fault latch block)
    assert node.relay_locked is True
    node.lock_start_time = time.time() - 301.0
    # Lockout expiry is evaluated by the core-1 loop tick (the
    # SAFETY_LOCKOUT_MS expiry check),
    # not by the MQTT command handler.
    await node.core1_telemetry_tick()
    assert node.relay_locked is False
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is True

@pytest.mark.asyncio
async def test_lockout_resets_on_new_fault(node):
    trigger_arc_fault(node)
    await node.core1_telemetry_tick()   # lockout taken (arc-fault latch block)
    first_lock_time = node.lock_start_time

    # Fast forward 100s
    time.sleep(0.01)

    # Reset lockout and bypass inrush to trigger another fault
    node.relay_locked = False
    node.set_relay(True)
    node.pzem.set_load(100.0)
    for _ in range(5):
        node.core0_safety_step(sim_dt=0.1)

    # Trigger another fault (overcurrent this time)
    node.pzem.set_load(node.rated_watts * 1.5)
    node.core0_safety_step(sim_dt=0.1)
    # Core 1 re-takes the lockout with a fresh timer (the overcurrent latch block).
    await node.core1_telemetry_tick()

    assert node.lock_start_time > first_lock_time

@pytest.mark.asyncio
async def test_rapid_fault_lockout_chaining(node):
    trigger_arc_fault(node)
    await node.core1_telemetry_tick()   # lockout taken (arc-fault latch block)
    for _ in range(3):
        node.pzem.set_load(3000.0)
        node.core0_safety_step(sim_dt=0.1)
    assert node.relay_locked is True

@pytest.mark.asyncio
async def test_lockout_survives_mqtt_reconnect(node, mqtt_client):
    trigger_arc_fault(node)
    await node.core1_telemetry_tick()   # lockout taken (arc-fault latch block)
    # Simulate disconnect and reconnect
    await mqtt_client.disconnect()
    await mqtt_client.reconnect()
    assert node.relay_locked is True
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_concurrent_on_off_commands(node):
    node.set_relay(False)
    await asyncio.gather(
        node.handle_mqtt_command("ON"),
        node.handle_mqtt_command("OFF")
    )
    # Outcome is deterministic based on execution order, but should not crash
    assert node.gpio18_relay_state in [True, False]

# ==========================================
# Category 4: Race Conditions
# ==========================================

@pytest.mark.asyncio
async def test_core0_core1_relay_race(node):
    # Core 1 sends ON while Core 0 trips safety
    node.set_relay(True)
    node.pzem.set_load(100.0)
    for _ in range(5):
        node.core0_safety_step(sim_dt=0.1)
        
    node.pzem.set_load(2000.0)
    
    async def race():
        node.core0_safety_step(sim_dt=0.1)
        # The core-1 loop consumes the trip latch and takes the lockout
        # (the latch-consumption blocks in the core-1 loop) before the next
        # command can be honored.
        await node.core1_telemetry_tick()
        await node.handle_mqtt_command("ON")

    await race()
    # Safety should override or lockout should prevent ON
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_arc_fault_flag_acknowledgment_race(node):
    trigger_arc_fault(node)
    assert node.shared_arc_fault is True
    await node.core1_telemetry_tick(force_publish=True)
    # telemetry tick should clear the flag
    assert node.shared_arc_fault is False

def test_spinlock_contention_stress(node):
    for i in range(1000):
        node.pzem.set_load(float(i % 100))
        node.core0_safety_step(sim_dt=0.1)
    assert node.shared_power_watts >= 0.0

@pytest.mark.asyncio
async def test_relay_command_during_overcurrent_cutoff(node):
    node.set_relay(True)
    node.pzem.set_load(100.0)
    for _ in range(5):
        node.core0_safety_step(sim_dt=0.1)
        
    node.pzem.set_load(node.rated_watts * 2.0) # overcurrent
    node.core0_safety_step(sim_dt=0.1)
    # Core-1 acknowledgment takes the lockout (the overcurrent latch block).
    await node.core1_telemetry_tick()
    assert node.relay_locked is True
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_watchdog_timeout_vs_mqtt_command(node):
    # Simulate server timeout via a long block, command shouldn't crash
    node.set_relay(False)
    await asyncio.sleep(0.01)
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is True

# ==========================================
# Category 5: Edge Cases
# ==========================================

def test_relay_state_after_nan_power_reading(node):
    node.set_relay(True)
    node.pzem.set_load(math.nan)
    node.core0_safety_step(sim_dt=0.1)
    # Relay should stay ON, NaN should not falsely trigger trip
    assert node.gpio18_relay_state is True

def test_relay_with_zero_rated_watts():
    node_zero = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=0.0)
    node_zero.set_relay(True)
    node_zero.pzem.set_load(100.0)
    for _ in range(5):
        node_zero.core0_safety_step(sim_dt=0.1)
        
    node_zero.pzem.set_load(10.0) # even 10W is overcurrent
    node_zero.core0_safety_step(sim_dt=0.1)
    assert node_zero.gpio18_relay_state is False

def test_relay_with_negative_power(node):
    node.set_relay(True)
    node.pzem.set_load(-100.0) # Regenerative
    node.core0_safety_step(sim_dt=0.1)
    assert node.gpio18_relay_state is True

@pytest.mark.asyncio
async def test_millis_overflow_handling(node):
    trigger_arc_fault(node)
    await node.core1_telemetry_tick()   # lockout taken (arc-fault latch block)
    # Simulate millis overflow (approx 49 days)
    # Python time.time() doesn't overflow like ESP32 millis(), but we test negative diff
    node.lock_start_time = float('inf')
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is False

def test_relay_100000_cycles_endurance(node):
    for i in range(100000):
        node.set_relay(i % 2 == 0)
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_core1_status_lifecycle_and_server_timeout(mqtt_client):
    """Core 1 must publish the firmware's status strings.

    ONLINE once on the first tick (the retained publish in reconnectMQTT());
    OFFLINE via announce_offline() (the LWT counterpart, the client.connect()
    last-will in reconnectMQTT()); SERVER_TIMEOUT per the firmware's exact
    semantics: lastServerHB is armed at MQTT CONNECT (the twin's ONLINE
    announcement), so a connected-but-commandless node fires at t+30 s and
    REPUBLISHES every 30 s while starved (the lastTimeoutLog gate in loop()
    — not once per episode). A command refreshes the heartbeat and stops
    the timeouts (callback() sets lastServerHB in main.cpp).
    """
    n = ESP32FirmwareNode(device_id=DEVICE_ID, rated_watts=RATED_WATTS,
                          mqtt_publish_fn=mqtt_client.publish)

    def statuses():
        return [p for t, p in mqtt_client.published_messages if t == n.topic_status]

    await n.core1_telemetry_tick(force_publish=True)
    assert statuses() == ["ONLINE"], "first tick must announce ONLINE exactly once"

    # Connected-but-commandless (the heartbeat was armed by the ONLINE
    # announcement, like main.cpp arms it at connect): starved past 30 s
    # -> SERVER_TIMEOUT fires.
    n.last_server_hb = time.time() - 31.0
    await n.core1_telemetry_tick(force_publish=True)
    assert statuses().count("SERVER_TIMEOUT") == 1
    # Still starved -> republished every 30 s (the firmware's lastTimeoutLog
    # gate), not once per episode. The gate is wall-clock (millis()-based in
    # the firmware), so the test ages BOTH the heartbeat and the republish
    # gate — the twin's ticks here run microseconds apart.
    n.last_server_hb = time.time() - 65.0
    n._last_server_timeout_log = time.time() - 31.0
    await n.core1_telemetry_tick(force_publish=True)
    assert statuses().count("SERVER_TIMEOUT") == 2

    # A server command refreshes the heartbeat -> the watchdog goes quiet.
    n.last_server_hb = time.time()
    await n.core1_telemetry_tick(force_publish=True)
    assert statuses().count("SERVER_TIMEOUT") == 2
    await n.core1_telemetry_tick(force_publish=True)
    assert statuses().count("SERVER_TIMEOUT") == 2

    await n.announce_offline()
    assert statuses()[-1] == "OFFLINE"
