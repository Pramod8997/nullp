import asyncio
import json
import pytest

from src.hardware.mqtt import AsyncMQTTClient, MockMQTTBroker
from scripts.run_pipeline import FullPipeline, load_config


@pytest.fixture
def mock_mqtt_broker():
    broker = MockMQTTBroker()
    yield broker


# TEST 8-1: Topic wildcard — single subscriber handles 10 different device IDs
@pytest.mark.asyncio
async def test_mqtt_wildcard_routes_10_devices():
    received = []
    client = AsyncMQTTClient(on_message=lambda topic, payload: received.append(topic))
    await client.subscribe("home/sensor/+/power")
    
    for device_id in [f"device_{i}" for i in range(10)]:
        await client.publish(f"home/sensor/{device_id}/power", json.dumps({"power": 100.0}))
    
    await asyncio.sleep(0.1)
    assert len(received) == 10
    assert all("home/sensor/" in t and "/power" in t for t in received)

# TEST 8-2: Malformed JSON payload does not crash pipeline
@pytest.mark.asyncio
async def test_mqtt_malformed_json_handled():
    pipeline = FullPipeline(config=load_config())
    # Should log a warning and continue, not raise
    await pipeline.process_raw_mqtt("home/sensor/node_fridge/power",
                                     payload=b"NOT_VALID_JSON")

# TEST 8-3: Missing "power" key in payload handled gracefully
@pytest.mark.asyncio
async def test_mqtt_missing_power_key_handled():
    pipeline = FullPipeline(config=load_config())
    payload = json.dumps({"voltage": 230.0}).encode()  # no "power" key
    await pipeline.process_raw_mqtt("home/sensor/node_fridge/power", payload=payload)
    # No exception; pipeline should emit a parse-error event or skip silently

# TEST 8-4: Extreme power values are rejected / clamped
@pytest.mark.asyncio
async def test_mqtt_extreme_power_values_clamped():
    pipeline = FullPipeline(config=load_config())
    payload = json.dumps({"power": 999999.0}).encode()
    # Should not trigger a real safety alert with clearly invalid data
    result = await pipeline.process_raw_mqtt("home/sensor/node_fridge/power", payload=payload)
    safety_events = [e for e in result.events if e.event_type == "SAFETY_ALERT"]
    # Extreme values should be detected as sensor error, not safety critical
    assert len(safety_events) == 0 or all(e.source == "SENSOR_ERROR" for e in safety_events)

# TEST 8-5: MQTT reconnect — pipeline resumes after broker restart
@pytest.mark.asyncio
async def test_mqtt_reconnects_after_broker_restart(mock_mqtt_broker):
    pipeline = FullPipeline(config=load_config())
    await mock_mqtt_broker.disconnect_all()
    await asyncio.sleep(0.5)
    await mock_mqtt_broker.restart()
    await asyncio.sleep(2.0)
    # Pipeline should have reconnected and be processing again
    assert pipeline.is_connected()
