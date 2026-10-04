"""Failing reproductions for the 2026-10-04 ASTRA release audit.

These tests intentionally capture requirements that the current working tree
does not yet satisfy. They are added before remediation so a later patch must
prove the invariant rather than merely preserve the existing simulator tests.
"""

import asyncio

import pytest

from scripts.run_pipeline import FullPipeline, load_config
from src.hardware.esp32_firmware_sim import ESP32FirmwareNode, PZEM_FAIL_TRIP_COUNT


@pytest.mark.asyncio
async def test_dead_pzem_at_boot_cannot_accept_on_command():
    """H1: unknown measurement health must fail closed at the actuation gate."""
    node = ESP32FirmwareNode("audit_h1")
    node.pzem.voltage = float("nan")

    await node.handle_mqtt_command("ON")
    for _ in range(PZEM_FAIL_TRIP_COUNT * 2):
        node.core0_safety_step()

    assert node.gpio18_relay_state is False, (
        "ON was accepted while the PZEM had never produced a valid read; "
        "the dead-at-boot path leaves the relay energized without a safety "
        "measurement"
    )


@pytest.mark.asyncio
async def test_safety_cutoff_wins_over_on_before_core1_latch_tick():
    """H2: an ON request cannot re-energize a just-tripped circuit."""
    node = ESP32FirmwareNode("audit_h2")
    node.set_relay(True)
    node.pzem.set_load(node.rated_watts * 1.25 + 50.0)

    node.core0_safety_step()
    assert node.gpio18_relay_state is False
    assert node.shared_overcurrent_latch is True

    # This is the inter-core window from the audit: the lockout has not yet
    # been consumed by Core 1, but command handling can currently close relay.
    await node.handle_mqtt_command("ON")
    await node.core1_telemetry_tick()

    assert node.gpio18_relay_state is False, (
        "the relay re-closed after Core 0 safety cutoff; the safety owner must "
        "win over an ON request in the same scheduling window"
    )


@pytest.mark.asyncio
async def test_direct_pipeline_handler_rejects_empty_object_and_negative_power():
    """Parser contract: invalid power must not become live state."""
    pipeline = FullPipeline(config=load_config())

    async def no_broadcast(_event):
        return None

    pipeline._broadcast_event = no_broadcast

    for payload in ("", "{}", "-3"):
        pipeline.last_device_power.clear()
        await pipeline._handle_mqtt_message(
            "home/sensor/audit_parser/power", payload
        )
        assert "audit_parser" not in pipeline.last_device_power, (
            f"invalid payload {payload!r} was accepted into live power state"
        )


class _Message:
    topic = "home/ui/events"
    payload = b'{"type":"DEVICE_STATUS","device_id":"audit_b2","power":"bad"}'


class _OneMalformedMessage:
    def __init__(self):
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            # Let the listener's normal cancellation branch terminate the
            # fake connection after the one malformed frame.
            raise asyncio.CancelledError
        self._sent = True
        return _Message()


class _FakeMQTTClient:
    def __init__(self, **_kwargs):
        self.messages = _OneMalformedMessage()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def subscribe(self, _topic):
        return None


@pytest.mark.asyncio
async def test_malformed_ui_event_does_not_crash_mqtt_bridge(monkeypatch):
    """B2: schema/type errors must be isolated like JSON syntax errors."""
    import src.api.main as api

    monkeypatch.setattr(api.aiomqtt, "Client", _FakeMQTTClient)
    api._shared_mqtt_client = None

    task = asyncio.create_task(api.mqtt_listener_task())
    await asyncio.sleep(0.05)

    if task.done():
        exc = task.exception()
        assert exc is None, f"malformed UI event killed MQTT bridge: {exc!r}"
    else:
        task.cancel()
        await task
