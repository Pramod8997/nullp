"""Isolated presentation contract checks; no sockets, services, or hardware."""

import asyncio
import importlib.util
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


BRIDGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "presentation_bridge.py"
SPEC = importlib.util.spec_from_file_location("isolated_presentation_bridge", BRIDGE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    for key, value in {
        "CLIENTS": set(), "DEMO_REPLAY": None, "REPLAY_SAMPLES": [],
        "REPLAY_REPEATS": 3, "REPLAY_PATH": None, "SAMPLE": 0, "LAST_RELAY": "OFF",
    }.items():
        monkeypatch.setattr(bridge, key, value)


class DashboardSocket:
    def __init__(self):
        self.frames = []

    async def send(self, payload):
        self.frames.append(json.loads(payload))

    async def wait_closed(self):
        pass


def assert_generated(payload):
    assert payload["source"] == "generated_sample_replay"
    assert payload["generated"] is True
    assert payload["presentation_only"] is True
    assert payload["recording_replay"] is False
    assert payload["pzem_connected"] is False
    assert payload["relay_action"] == "NONE"
    assert payload["inference_performed"] is False


def test_connection_immediately_initializes_populated_dashboard(monkeypatch):
    replay = bridge.GeneratedReplay()
    monkeypatch.setattr(bridge, "DEMO_REPLAY", replay)
    monkeypatch.setattr(bridge.time, "time", lambda: 1234567890.0)
    socket = DashboardSocket()
    asyncio.run(bridge.websocket_handler(socket))

    initial, event = socket.frames
    assert initial["type"] == "init_state"
    assert initial["pipeline_status"] == "SAMPLE_DATA_REPLAY"
    assert_generated(initial)
    assert set(initial["devices"]) == {"sample_phone", "sample_laptop", "sample_projector"}
    assert 0 < sum(device["power"] for device in initial["devices"].values()) <= 250
    assert len(initial["power_history"]) == 12
    assert initial["analytics"]["total_kwh"] > 0
    assert initial["analytics"]["tariff_inr_per_kwh"] == 8
    assert initial["analytics"]["replay"] == {
        "cycle": 1, "total_cycles": 3, "sample_index": 12,
        "total_samples": 60, "generated": True, "group": 1,
    }
    assert_generated(initial["analytics"])
    assert initial["playback_speed"] == 300
    assert initial["represented_interval_s"] == 300
    assert initial["timestamp_semantics"] == "bridge_emit_time_not_capture"
    for device_id, device in initial["devices"].items():
        assert_generated(device)
        assert device["classification"] == "known:" + device_id.removeprefix("sample_")
        assert device["classification_source"] == "generated_scenario_label"
        assert device["confidence"] is None
        assert device["last_seen"] == 1234567890.0
        assert "latency_ms" not in device
        telemetry = initial["telemetry"][device_id]
        assert_generated(telemetry)
        assert telemetry["timestamp"] == 1234567890.0
        assert telemetry["v"] * telemetry["i"] * telemetry["pf"] == pytest.approx(device["power"])
    for row in initial["power_history"]:
        assert_generated(row)
        assert row["time"].startswith("C1 S")
        assert "timestamp" not in row
    for row in initial["analytics"]["energy_history"]:
        assert_generated(row)
        assert {"time", "today", "yesterday"} <= row.keys()
    assert event["type"] == "SAFETY_WARNING"
    assert_generated(event)
    assert "Generated" in event["message"]
    assert socket not in bridge.CLIENTS


def test_all_generated_power_is_finite_bounded_and_only_three_classes():
    replay = bridge.GeneratedReplay()
    observed_cycles = set()
    for _ in range(600):
        snapshot = replay.snapshot()
        assert len(snapshot["power_history"]) <= 120
        assert len(snapshot["analytics"]["energy_history"]) <= 120
        assert all(math.isfinite(power) and power >= 0 for power in replay.readings.values())
        assert sum(replay.readings.values()) <= 250
        assert {d["classification"] for d in snapshot["devices"].values()} == {
            "known:phone", "known:laptop", "known:projector",
        }
        observed_cycles.add(snapshot["analytics"]["replay"]["cycle"])
        # Disallow NaN/Infinity anywhere on the wire, including analytics and telemetry.
        json.dumps(snapshot, allow_nan=False)
        replay.advance()
    assert observed_cycles == {1, 2, 3}


@pytest.mark.parametrize("tariff", [0.0, 8.0, 11.75])
def test_energy_baseline_savings_and_cost_are_integrated_from_samples(tariff):
    replay = bridge.GeneratedReplay(tariff)
    device_ids = set(replay.readings)
    expected = sum(sum(row[device_id] for device_id in device_ids) for row in replay.power_history) * 300 / 3_600_000
    # Check beyond the rolling history window; totals still include the entire group.
    for _ in range(150):
        summary = replay.snapshot()["analytics"]
        baseline = 215 * replay.position * 300 / 3_600_000
        assert summary["total_kwh"] == pytest.approx(expected)
        assert sum(summary["device_usage_kwh"].values()) == pytest.approx(expected)
        assert summary["estimated_cost_inr"] == summary["total_kwh"] * tariff
        assert summary["baseline_kwh"] == pytest.approx(baseline)
        assert summary["energy_saved_kwh"] == pytest.approx(max(baseline - expected, 0))
        assert summary["represented_duration_s"] == replay.position * 300
        row = summary["energy_history"][-1]
        assert row["today"] == pytest.approx(sum(replay.readings.values()) * 300 / 3_600_000)
        assert row["yesterday"] == pytest.approx(215 * 300 / 3_600_000)
        replay.advance()
        expected += sum(replay.readings.values()) * 300 / 3_600_000


def test_three_cycle_groups_repeat_deterministically_and_reset_totals():
    replay = bridge.GeneratedReplay()
    initial = replay.snapshot()
    replay.reset()
    first_group = replay.group
    cycles = []
    cycle_energy = []
    for cycle in range(1, 4):
        rows = []
        for index in range(1, 61):
            replay.advance()
            summary = replay.snapshot()["analytics"]
            assert summary["replay"]["cycle"] == cycle
            assert summary["replay"]["sample_index"] == index
            rows.append(dict(replay.readings))
        cycles.append(rows)
        cycle_energy.append(replay.snapshot()["analytics"]["total_kwh"])
    assert cycles[0] == cycles[1] == cycles[2]
    assert cycle_energy[1] == pytest.approx(cycle_energy[0] * 2)
    assert cycle_energy[2] == pytest.approx(cycle_energy[0] * 3)
    assert len(replay.power_history) == 120

    replay.advance()
    summary = replay.snapshot()["analytics"]
    assert summary["replay"]["group"] == first_group + 1
    assert summary["replay"]["cycle"] == summary["replay"]["sample_index"] == 1
    assert replay.readings == cycles[0][0]
    assert summary["total_kwh"] == pytest.approx(sum(cycles[0][0].values()) * 300 / 3_600_000)
    assert len(replay.power_history) == 1
    assert initial["analytics"]["replay"]["sample_index"] == 12
    assert len(initial["power_history"]) == 12  # Previously sent snapshots stay stable.
    for _ in range(11):
        replay.advance()
    assert replay.snapshot()["analytics"]["total_kwh"] == initial["analytics"]["total_kwh"]


class StopDemo(Exception):
    pass


def test_demo_cli_streams_at_one_hz_without_serial_or_recording_reads(monkeypatch):
    socket = DashboardSocket()
    delays = []
    serial_constructor = Mock(side_effect=AssertionError("demo must not read USB/PZEM"))
    serial_loop = AsyncMock(side_effect=AssertionError("demo must not select serial loop"))
    file_loader = Mock(side_effect=AssertionError("demo must not load a hardware recording"))
    monkeypatch.setattr(bridge.serial, "Serial", serial_constructor)
    monkeypatch.setattr(bridge, "serial_loop", serial_loop)
    monkeypatch.setattr(bridge, "load_recording", file_loader)
    monkeypatch.setattr(bridge, "heartbeat_loop", AsyncMock())

    @asynccontextmanager
    async def fake_server(handler, host, port):
        assert (host, port) == ("127.0.0.1", 8000)
        await handler(socket)
        bridge.CLIENTS.add(socket)
        yield

    async def tick(delay):
        delays.append(delay)
        if len(delays) == 172:
            raise StopDemo()

    monkeypatch.setattr(bridge, "serve", fake_server)
    monkeypatch.setattr(bridge.asyncio, "sleep", tick)
    with pytest.raises(StopDemo):
        asyncio.run(bridge.main(["--demo", "--demo-tariff", "9.5"]))
    serial_constructor.assert_not_called()
    serial_loop.assert_not_called()
    file_loader.assert_not_called()
    assert delays == [1.0] * 172
    assert socket.frames[0]["type"] == "init_state"
    stream = socket.frames[2:]
    summaries = []
    for start in range(0, len(stream), 8):
        batch = stream[start:start + 8]
        assert [frame["type"] for frame in batch] == [
            "power_batch", "DEVICE_STATUS", "DEVICE_STATUS", "DEVICE_STATUS",
            "ANALYTICS_UPDATE", "TELEMETRY", "TELEMETRY", "TELEMETRY",
        ]
        for frame in batch:
            assert_generated(frame)
        readings = batch[0]["readings"]
        assert sum(readings.values()) <= 250
        assert {frame["device_id"]: frame["power"] for frame in batch[1:4]} == readings
        summary = batch[4]["summary"]
        assert summary["tariff_inr_per_kwh"] == 9.5
        assert summary["estimated_cost_inr"] == summary["total_kwh"] * 9.5
        summaries.append(summary)
    assert [s["replay"]["sample_index"] for s in summaries[:3]] == [12, 13, 14]
    assert [s["replay"]["cycle"] for s in summaries[168:171]] == [3, 1, 1]
    assert summaries[169]["replay"]["sample_index"] == 1
    assert summaries[169]["replay"]["group"] == 2
    assert summaries[169]["total_kwh"] < summaries[168]["total_kwh"]
    assert summaries[-1]["replay"]["sample_index"] == 3
    assert all(frame["type"] not in {"SAFETY_CUTOFF", "LATENCY_STATS", "RL_ACTION"} for frame in socket.frames)


def test_heartbeat_advertises_sample_data_mode(monkeypatch):
    monkeypatch.setattr(bridge, "DEMO_REPLAY", bridge.GeneratedReplay())
    broadcast = AsyncMock()
    monkeypatch.setattr(bridge, "broadcast", broadcast)
    monkeypatch.setattr(bridge.asyncio, "sleep", AsyncMock(side_effect=StopDemo()))
    with pytest.raises(StopDemo):
        asyncio.run(bridge.heartbeat_loop())
    assert json.loads(broadcast.call_args.args[0]) == {"type": "heartbeat", "status": "SAMPLE_DATA_REPLAY"}


def test_file_replay_preserves_power_timing_and_repeats_without_invented_cutoffs(tmp_path, monkeypatch):
    recording = tmp_path / "trace.jsonl"
    recording.write_text('# real file input\n{"power": 42.5, "relay": "ON", "delay_s": 0.1}\n{"power": 0, "relay": "OFF", "delay_s": 2}\n', encoding="utf-8")
    samples = bridge.load_recording(str(recording))
    monkeypatch.setattr(bridge, "REPLAY_SAMPLES", samples)
    monkeypatch.setattr(bridge, "REPLAY_REPEATS", 2)
    socket = DashboardSocket()
    asyncio.run(bridge.websocket_handler(socket))
    initial = socket.frames[0]
    assert initial["pipeline_status"] == "RECORDED_DATA_REPLAY"
    device = initial["devices"][bridge.DEVICE_ID]
    assert device["recording_replay"] is True
    assert device["pzem_connected"] is False
    assert device["confidence"] is None
    bridge.CLIENTS.add(socket)
    sleep = AsyncMock()
    monkeypatch.setattr(bridge.asyncio, "sleep", sleep)
    asyncio.run(bridge.replay_loop())
    statuses = [frame for frame in socket.frames if frame["type"] == "DEVICE_STATUS"]
    assert [frame["power"] for frame in statuses] == [42.5, 0, 42.5, 0]
    assert [frame["replay_cycle"] for frame in statuses] == [1, 1, 2, 2]
    assert [frame["relay_state"] for frame in statuses] == ["ON", "OFF", "ON", "OFF"]
    assert [frame["state"] for frame in statuses] == ["ON", "OFF", "ON", "OFF"]
    assert all(frame["recording_replay"] and not frame["pzem_connected"] for frame in statuses)
    assert all(frame["relay_action"] == "NONE" for frame in statuses)
    assert not any(frame["type"] == "SAFETY_CUTOFF" for frame in socket.frames)
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 2, 0.1, 2]


def test_missing_recording_is_rejected_without_fallback_or_server_start(tmp_path, monkeypatch):
    missing = tmp_path / "missing.jsonl"
    server = Mock(side_effect=AssertionError("must reject before opening a server"))
    monkeypatch.setattr(bridge, "serve", server)
    with pytest.raises(FileNotFoundError, match="no generated substitute"):
        bridge.load_recording(str(missing))
    with pytest.raises(SystemExit) as error:
        asyncio.run(bridge.main(["--replay", str(missing)]))
    assert error.value.code == 2
    assert bridge.DEMO_REPLAY is None
    server.assert_not_called()


@pytest.mark.parametrize("row", [
    '{"power": NaN}', '{"power": Infinity}', '{"power": -1}',
    '{"power": 42, "delay_s": NaN}', '{"power": 42, "delay_s": Infinity}',
    '{"power": 42, "delay_s": -1}', '[]', '{"relay": "OFF"}', '', '# no samples',
])
def test_invalid_or_empty_recordings_are_rejected(tmp_path, row):
    recording = tmp_path / "invalid.jsonl"
    recording.write_text(row, encoding="utf-8")
    with pytest.raises(ValueError):
        bridge.load_recording(str(recording))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_demo_tariffs_are_rejected(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        bridge.GeneratedReplay(value)
    with pytest.raises(SystemExit) as error:
        asyncio.run(bridge.main(["--demo", f"--demo-tariff={value}"]))
    assert error.value.code == 2


@pytest.mark.parametrize("other", [["--replay", "trace.jsonl"], ["--port", "COM9"]])
def test_demo_is_an_explicit_exclusive_mode(other):
    with pytest.raises(SystemExit) as error:
        asyncio.run(bridge.main(["--demo", *other]))
    assert error.value.code == 2


@pytest.mark.parametrize("args, expected", [([], "COM3"), (["--port", "COM7"], "COM7")])
def test_serial_mode_still_selects_requested_or_default_port(monkeypatch, args, expected):
    @asynccontextmanager
    async def fake_server(*_):
        yield

    serial_loop = AsyncMock(side_effect=StopDemo())
    monkeypatch.setattr(bridge, "serve", fake_server)
    monkeypatch.setattr(bridge, "serial_loop", serial_loop)
    monkeypatch.setattr(bridge, "heartbeat_loop", AsyncMock())
    with pytest.raises(StopDemo):
        asyncio.run(bridge.main(args))
    serial_loop.assert_awaited_once_with(expected)
    assert bridge.DEMO_REPLAY is None
    assert bridge.pipeline_status() == "USB_COLLECTION_MODE"


def test_serial_parser_and_status_keep_real_power_without_inference_claim(monkeypatch):
    assert bridge.parse_usb_power("[USB] PZEM=INVALID Relay=OFF") is None
    assert bridge.parse_usb_power("[USB] P=-3 Relay=ON") is None
    parsed = bridge.parse_usb_power("[USB] P=57.2 Relay=ON")
    assert parsed == (57.2, "ON")
    socket = DashboardSocket()
    bridge.CLIENTS.add(socket)
    asyncio.run(bridge.publish_sample(*parsed))
    batch, status = socket.frames
    assert batch["readings"] == {bridge.DEVICE_ID: 57.2}
    assert status["source"] == "usb_collection"
    assert status["recording_replay"] is False
    assert status["relay_state"] == "ON"
    assert status["relay_action"] == "NONE"
    assert status["confidence"] is None
    assert status["inference_performed"] is False
