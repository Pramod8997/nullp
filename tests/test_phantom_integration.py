"""
Pipeline wiring tests for the orchestrator's MQTT message handler.

  * Run 1.2 — sub-transient steady loads (a 9 W LED bulb never crosses the
    20 W transient threshold) must still reach PhantomTracker via the
    no-transient branch of _handle_mqtt_message.
  * Run 4 — the home/sensor/+/telemetry subscription stores the latest
    V/I/PF frame per device and broadcasts a "TELEMETRY" WS event.

These test the ORCHESTRATOR wiring; tests/test_phantom.py covers the
PhantomTracker class itself.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_pipeline import FullPipeline  # noqa: E402


def _pipeline(tmp_path):
    cfg = {
        "mqtt": {"broker": "localhost", "port": 1883,
                 "topics": {"reads": "home/sensor/+/power",
                            "writes": "home/plug/+/command",
                            "events": "home/ui/events"}},
        "database": {"path": str(tmp_path / "ems_state.db"),
                     "fallback_csv": str(tmp_path / "fallback.csv")},
        # Point at an absent weights file so no torch model is loaded —
        # these tests never classify.
        "protonet": {"weights_path": str(tmp_path / "absent.pt")},
        "phantom": {"baseline_threshold_watts": 15.0},
    }
    return FullPipeline(config=cfg)


class TestSubTransientPhantomTracking:

    @pytest.mark.asyncio
    async def test_steady_9w_load_with_no_transient_is_phantom_tracked(
            self, tmp_path, monkeypatch):
        """The LED-bulb acceptance criterion: a 9 W steady load that never
        fires the transient detector still appears in the phantom channel."""
        pipeline = _pipeline(tmp_path)

        async def nop(topic, payload, **kwargs):
            pass

        monkeypatch.setattr(pipeline.mqtt, "publish_command", nop)

        for _ in range(30):
            await pipeline._handle_mqtt_message(
                "home/sensor/node_bulb/power", "9.0")

        assert pipeline.device_states["node_bulb"] == 0  # 9 W <= 10 W on-gate
        assert pipeline.phantom_tracker.phantom_loads.get("node_bulb") \
            == pytest.approx(9.0)

    @pytest.mark.asyncio
    async def test_load_above_the_10w_state_gate_stays_unattributed(
            self, tmp_path, monkeypatch):
        # Documented ceiling: the no-transient branch marks devices ON above
        # 10 W, so a load in (10, 15] W is never phantom-tracked even though
        # it is below the 15 W baseline threshold.
        pipeline = _pipeline(tmp_path)

        async def nop(topic, payload, **kwargs):
            pass

        monkeypatch.setattr(pipeline.mqtt, "publish_command", nop)

        for _ in range(30):
            await pipeline._handle_mqtt_message(
                "home/sensor/node_mid/power", "12.0")

        assert pipeline.device_states["node_mid"] == 1
        assert "node_mid" not in pipeline.phantom_tracker.phantom_loads


class TestTelemetrySubscription:

    @pytest.mark.asyncio
    async def test_telemetry_frame_is_stored_and_broadcast(
            self, tmp_path, monkeypatch):
        pipeline = _pipeline(tmp_path)
        events = []

        async def capture(event):
            events.append(event)

        monkeypatch.setattr(pipeline, "_broadcast_event", capture)

        frame = '{"v": 230.4, "i": 0.041, "w": 9.2, "pf": 0.94}'
        await pipeline._handle_mqtt_message(
            "home/sensor/node_bulb/telemetry", frame)

        assert pipeline.device_telemetry["node_bulb"] == {
            "v": 230.4, "i": 0.041, "w": 9.2, "pf": 0.94}
        # CONTRACT: event type exactly "TELEMETRY"; fields device_id, v, i, pf.
        tel = [e for e in events if e.get("type") == "TELEMETRY"]
        assert len(tel) == 1
        assert tel[0]["device_id"] == "node_bulb"
        assert tel[0]["v"] == pytest.approx(230.4)
        assert tel[0]["i"] == pytest.approx(0.041)
        assert tel[0]["pf"] == pytest.approx(0.94)

    @pytest.mark.asyncio
    async def test_bad_telemetry_payload_is_dropped(
            self, tmp_path, monkeypatch):
        pipeline = _pipeline(tmp_path)
        events = []

        async def capture(event):
            events.append(event)

        monkeypatch.setattr(pipeline, "_broadcast_event", capture)

        await pipeline._handle_mqtt_message(
            "home/sensor/node_bulb/telemetry", "not-json{")
        await pipeline._handle_mqtt_message(
            "home/sensor/node_bulb/telemetry", '{"v": 230.0}')  # missing i/w/pf

        assert "node_bulb" not in pipeline.device_telemetry
        assert not [e for e in events if e.get("type") == "TELEMETRY"]

    def test_run_subscribes_the_telemetry_topic(self):
        # The subscription list lives inside run(), which needs a live broker
        # to execute — pin the wiring by source until a broker fixture exists.
        import inspect
        from scripts import run_pipeline
        src = inspect.getsource(run_pipeline.EMSOrchestrator.run)
        assert '"home/sensor/+/telemetry"' in src
