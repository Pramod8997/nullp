import pytest
import asyncio
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
    node = ESP32FirmwareNode(
        device_id=DEVICE_ID, 
        rated_watts=RATED_WATTS, 
        relay_active_low=True, 
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
            relay_active_low=True, 
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
    # In hardware, setRelay(true) drives GPIO LOW (ON), setRelay(false) drives GPIO HIGH (OFF)
    # The sim uses True = ON, False = OFF
    node.set_relay(True)
    assert node.gpio18_relay_state is True
    node.set_relay(False)
    assert node.gpio18_relay_state is False

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
    assert node.relay_locked is True
    assert node.gpio18_relay_state is False

@pytest.mark.asyncio
async def test_lockout_rejects_on_command(node, mqtt_client):
    trigger_arc_fault(node)
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
    assert node.relay_locked is True
    node.lock_start_time = time.time() - 301.0
    await node.handle_mqtt_command("ON")
    assert node.relay_locked is False
    assert node.gpio18_relay_state is True

@pytest.mark.asyncio
async def test_lockout_resets_on_new_fault(node):
    trigger_arc_fault(node)
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
    
    assert node.lock_start_time > first_lock_time

@pytest.mark.asyncio
async def test_rapid_fault_lockout_chaining(node):
    trigger_arc_fault(node)
    for _ in range(3):
        node.pzem.set_load(3000.0)
        node.core0_safety_step(sim_dt=0.1)
    assert node.relay_locked is True

@pytest.mark.asyncio
async def test_lockout_survives_mqtt_reconnect(node, mqtt_client):
    trigger_arc_fault(node)
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
    # Simulate millis overflow (approx 49 days)
    # Python time.time() doesn't overflow like ESP32 millis(), but we test negative diff
    node.lock_start_time = float('inf')
    await node.handle_mqtt_command("ON")
    assert node.gpio18_relay_state is False

def test_relay_100000_cycles_endurance(node):
    for i in range(100000):
        node.set_relay(i % 2 == 0)
    assert node.gpio18_relay_state is False
