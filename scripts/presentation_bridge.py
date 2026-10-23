"""Presentation bridge for USB collection, file replay, or explicit sample demo.

--demo generates labelled scenario data without reading hardware. --replay
requires a real JSONL file and never substitutes generated data for a missing
recording. This bridge does not perform inference or issue relay commands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import serial
from websockets.asyncio.server import ServerConnection, serve


DEVICE_ID = "node_bench_agg"
CLIENTS: set[ServerConnection] = set()
SAMPLE = 0
LAST_RELAY = "OFF"
REPLAY_SAMPLES: list[dict[str, Any]] = []
REPLAY_REPEATS = 3
REPLAY_PATH: Path | None = None
DEMO_REPLAY: GeneratedReplay | None = None
DEMO_SOURCE = "generated_sample_replay"
DEMO_METADATA = {
    "source": DEMO_SOURCE,
    "generated": True,
    "presentation_only": True,
    "recording_replay": False,
    "pzem_connected": False,
    "relay_action": "NONE",
    "inference_performed": False,
}


class GeneratedReplay:
    """Deterministic scenario; energy totals cover the current three-cycle group.

    Preload twelve *scenario* intervals, not backdated wall-clock measurements.
    One emitted second represents five scenario minutes (300x playback).
    """

    total_samples = 60
    total_cycles = 3
    represented_interval_s = 300.0
    publish_interval_s = 1.0
    rated_watts = {"phone": 30.0, "laptop": 65.0, "projector": 120.0}

    def __init__(self, tariff_inr_per_kwh: float = 8.0) -> None:
        if not math.isfinite(tariff_inr_per_kwh) or tariff_inr_per_kwh < 0:
            raise ValueError("Demo tariff must be finite and non-negative")
        self.tariff = tariff_inr_per_kwh
        self.group = 0
        self.reset()
        for _ in range(12):
            self.advance()

    def reset(self) -> None:
        self.group += 1
        self.position = 0
        self.device_usage_kwh = {f"sample_{name}": 0.0 for name in self.rated_watts}
        self.baseline_kwh = 0.0
        self.readings: dict[str, float] = {}
        self.power_history: list[dict[str, Any]] = []
        self.energy_history: list[dict[str, Any]] = []

    def advance(self) -> None:
        if self.position == self.total_samples * self.total_cycles:
            self.reset()
        index = self.position % self.total_samples
        self.position += 1
        # All three loads fit the <=250 W aggregate envelope, even at their peaks.
        powers = {
            "phone": 27 + 3 * math.sin(index / 4) if index < 40 else 0.0,
            "laptop": 55 + 10 * math.sin(index / 7) if index < 52 else 0.0,
            "projector": 100 + 20 * math.sin(index / 6) if index < 30 or index >= 42 else 0.0,
        }
        self.readings = {f"sample_{name}": round(watts, 1) for name, watts in powers.items()}
        interval_kwh = self.represented_interval_s / 3_600_000
        for device_id, power in self.readings.items():
            self.device_usage_kwh[device_id] += power * interval_kwh
        baseline = sum(self.rated_watts.values()) * interval_kwh
        self.baseline_kwh += baseline
        cycle = (self.position - 1) // self.total_samples + 1
        label = f"C{cycle} S{index + 1} (+{self.position * self.represented_interval_s / 60:g} min)"
        # Relative scenario labels deliberately contain no invented capture timestamps.
        self.power_history.append({"time": label, **self.readings, **DEMO_METADATA})
        self.power_history = self.power_history[-120:]
        self.energy_history.append({
            "time": label,
            "today": sum(self.readings.values()) * interval_kwh,
            "yesterday": baseline,
            **DEMO_METADATA,
        })
        self.energy_history = self.energy_history[-120:]

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        timing = {
            "playback_speed": self.represented_interval_s / self.publish_interval_s,
            "represented_interval_s": self.represented_interval_s,
            "publish_interval_s": self.publish_interval_s,
            "timestamp_semantics": "bridge_emit_time_not_capture",
        }
        devices = {}
        telemetry = {}
        for name, rated in self.rated_watts.items():
            device_id = f"sample_{name}"
            power = self.readings.get(device_id, 0.0)
            devices[device_id] = {
                **DEMO_METADATA,
                **timing,
                "power": power,
                "state": "ON" if power > 0 else "OFF",
                "classification": f"known:{name}",
                "classification_source": "generated_scenario_label",
                "confidence": None,
                "rated": rated,
                "label": f"Generated sample replay — {name}",
                "last_seen": now,
            }
            telemetry[device_id] = {
                **DEMO_METADATA,
                **timing,
                "v": 230.0,
                "i": power / (230.0 * 0.95),
                "pf": 0.95,
                "timestamp": now,
                "electrical_model": "generated assumption: 230 V, power factor 0.95; I=P/(V*PF)",
            }
        total = sum(self.device_usage_kwh.values())
        summary = {
            **DEMO_METADATA,
            **timing,
            "tariff_inr_per_kwh": self.tariff,
            "total_kwh": total,
            "estimated_cost_inr": total * self.tariff,
            "baseline_kwh": self.baseline_kwh,
            "energy_saved_kwh": max(self.baseline_kwh - total, 0.0),
            "baseline_power_w": sum(self.rated_watts.values()),
            "baseline_description": "Generated comparison: all three loads always at scenario rated power; not measured savings",
            "energy_scope": "current_three_cycle_group; resets at next group",
            "represented_duration_s": self.position * self.represented_interval_s,
            "device_usage_kwh": dict(self.device_usage_kwh),
            "energy_history": list(self.energy_history),
            "energy_history_scope": "last_120_represented_intervals",
            # Existing EnergyChart keys are aliases, not claims about calendar days.
            "energy_history_labels": {"today": "Generated scenario", "yesterday": "Generated baseline"},
            "replay": {
                "cycle": (max(self.position, 1) - 1) // self.total_samples + 1,
                "total_cycles": self.total_cycles,
                "sample_index": (self.position - 1) % self.total_samples + 1 if self.position else 0,
                "total_samples": self.total_samples,
                "generated": True,
                "group": self.group,
            },
        }
        return {
            **DEMO_METADATA,
            **timing,
            "pipeline_status": "SAMPLE_DATA_REPLAY",
            "devices": devices,
            "analytics": summary,
            "power_history": list(self.power_history),
            "telemetry": telemetry,
        }


def message(message_type: str, **payload: Any) -> str:
    return json.dumps({"type": message_type, **payload})


async def broadcast(payload: str) -> None:
    if not CLIENTS:
        return
    dead: list[ServerConnection] = []
    for client in CLIENTS:
        try:
            await client.send(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        CLIENTS.discard(client)


def replay_mode() -> bool:
    return bool(REPLAY_SAMPLES)


def pipeline_status() -> str:
    if DEMO_REPLAY is not None:
        return "SAMPLE_DATA_REPLAY"
    return "RECORDED_DATA_REPLAY" if replay_mode() else "USB_COLLECTION_MODE"


def device_label() -> str:
    return "Recorded session replay — power trace" if replay_mode() else "USB collection — awaiting valid PZEM data"


def load_recording(path: str) -> list[dict[str, Any]]:
    recording_path = Path(path)
    if not recording_path.is_file():
        raise FileNotFoundError(
            f"No telemetry recording found at {recording_path}. "
            "Replay requires a JSONL file captured from the hardware; no generated substitute was used."
        )

    samples: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(recording_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw_line)
            power = float(row["power"])
            delay = float(row.get("delay_s", 1.0))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid recording row {line_number} in {recording_path}: {exc}") from exc
        if not math.isfinite(power) or power < 0:
            raise ValueError(f"Invalid non-finite/negative power at row {line_number} in {recording_path}")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError(f"Invalid non-finite/negative delay at row {line_number} in {recording_path}")
        samples.append({
            "power": round(power, 1),
            "relay": str(row.get("relay", "OFF")),
            "delay_s": max(0.05, delay),
        })

    if not samples:
        raise ValueError(f"Telemetry recording {recording_path} contains no samples")
    return samples


async def websocket_handler(websocket: ServerConnection) -> None:
    CLIENTS.add(websocket)
    try:
        if DEMO_REPLAY is not None:
            await websocket.send(message("init_state", **DEMO_REPLAY.snapshot()))
            await websocket.send(message(
                "SAFETY_WARNING",
                **DEMO_METADATA,
                severity="info",
                message="Generated sample-data replay (300x scenario time); no hardware capture, inference, or relay action.",
            ))
        else:
            initial_power = REPLAY_SAMPLES[0]["power"] if REPLAY_SAMPLES else 0.0
            await websocket.send(message(
                "init_state",
                devices={
                    DEVICE_ID: {
                        "power": initial_power,
                        "state": "ON" if initial_power > 0 else "OFF",
                        "classification": "unknown",
                        "confidence": None,
                        "inference_performed": False,
                        "rated": 250,
                        "label": device_label(),
                        "source": "recorded_session_replay" if replay_mode() else "usb_collection",
                        "presentation_only": True,
                        "recording_replay": replay_mode(),
                        "pzem_connected": False,
                        "last_seen": time.time(),
                    }
                },
                pipeline_status=pipeline_status(),
            ))
            await websocket.send(message(
                "SAFETY_WARNING",
                device_id=DEVICE_ID,
                severity="info",
                presentation_only=True,
                recording_replay=replay_mode(),
                pzem_connected=False,
                relay_action="NONE",
                message=(
                    "Recorded-session replay: file power trace; no live PZEM connection or relay commands."
                    if replay_mode()
                    else "USB collection mode: waiting for valid PZEM samples; this bridge issues no relay commands."
                ),
            ))
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)


async def heartbeat_loop() -> None:
    while True:
        await broadcast(message("heartbeat", status=pipeline_status()))
        await asyncio.sleep(2)


async def demo_loop() -> None:
    assert DEMO_REPLAY is not None
    while True:
        snapshot = DEMO_REPLAY.snapshot()
        await broadcast(message(
            "power_batch",
            **DEMO_METADATA,
            readings=dict(DEMO_REPLAY.readings),
            timestamp=time.time(),
            playback_speed=snapshot["playback_speed"],
            represented_interval_s=snapshot["represented_interval_s"],
            timestamp_semantics=snapshot["timestamp_semantics"],
        ))
        for device_id, device in snapshot["devices"].items():
            await broadcast(message("DEVICE_STATUS", device_id=device_id, timestamp=device["last_seen"], **device))
        await broadcast(message("ANALYTICS_UPDATE", **DEMO_METADATA, summary=snapshot["analytics"]))
        for device_id, telemetry in snapshot["telemetry"].items():
            await broadcast(message("TELEMETRY", device_id=device_id, **telemetry))
        await asyncio.sleep(DEMO_REPLAY.publish_interval_s)
        DEMO_REPLAY.advance()


async def publish_sample(power: float, relay: str = "OFF", replay_cycle: int | None = None) -> None:
    global SAMPLE, LAST_RELAY
    SAMPLE += 1
    LAST_RELAY = relay
    now = time.time()
    await broadcast(
        message(
            "power_batch",
            readings={DEVICE_ID: power},
        )
    )
    await broadcast(
        message(
            "DEVICE_STATUS",
            device_id=DEVICE_ID,
            power=power,
            state="ON" if power > 0 else "OFF",
            classification="unknown",
            confidence=None,
            inference_performed=False,
            timestamp=now,
            label=device_label(),
            rated=250,
            source="recorded_session_replay" if replay_mode() else "usb_collection",
            presentation_only=True,
            recording_replay=replay_mode(),
            pzem_connected=False,
            relay_action="NONE",
            relay_state=LAST_RELAY,
            replay_cycle=replay_cycle,
        )
    )


def parse_usb_power(line: str) -> tuple[float, str] | None:
    match = re.search(r"\bP=([-+]?\d+(?:\.\d+)?)", line)
    if not match:
        return None
    power = float(match.group(1))
    if not math.isfinite(power) or power < 0:
        return None
    relay = "ON" if "Relay=ON" in line else "OFF"
    return power, relay


async def replay_loop() -> None:
    for replay_cycle in range(1, REPLAY_REPEATS + 1):
        for sample in REPLAY_SAMPLES:
            await publish_sample(sample["power"], sample["relay"], replay_cycle)
            await asyncio.sleep(sample["delay_s"])
    print(f"[BRIDGE] Completed {REPLAY_REPEATS} replay cycles from {REPLAY_PATH}", flush=True)


async def serial_loop(port: str) -> None:
    while True:
        serial_port = None
        try:
            print(f"[BRIDGE] Opening ESP32 serial port {port} at 115200", flush=True)
            serial_port = serial.Serial(port, 115200, timeout=1)
            while True:
                raw = await asyncio.to_thread(serial_port.readline)
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("[USB]"):
                        print(line, flush=True)
                        parsed = parse_usb_power(line)
                        if parsed is not None:
                            await publish_sample(*parsed)
                else:
                    await asyncio.sleep(0.05)
        except (serial.SerialException, PermissionError) as exc:
            print(f"[BRIDGE] Serial temporarily unavailable: {exc}; retrying", flush=True)
            await asyncio.sleep(2)
        finally:
            if serial_port is not None:
                serial_port.close()


async def main(argv: list[str] | None = None) -> None:
    global DEMO_REPLAY, REPLAY_REPEATS, REPLAY_PATH, REPLAY_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--port", help="USB serial port (default COM3 when neither replay mode is selected)")
    modes.add_argument("--replay", help="JSONL hardware recording; missing files are rejected")
    modes.add_argument("--demo", action="store_true", help="Generated sample replay at 1 Hz, no USB/PZEM; loops in 3-cycle groups")
    parser.add_argument("--repeats", type=int, default=3, choices=(2, 3), help="File replay cycles (default: 3); demo always loops in 3-cycle groups")
    parser.add_argument("--demo-tariff", type=float, default=8.0, metavar="INR_PER_KWH", help="Generated demo tariff (default: 8 INR/kWh)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.demo_tariff) or args.demo_tariff < 0:
        parser.error("--demo-tariff must be finite and non-negative")

    DEMO_REPLAY = None
    REPLAY_PATH = None
    REPLAY_SAMPLES = []
    REPLAY_REPEATS = args.repeats
    if args.demo:
        DEMO_REPLAY = GeneratedReplay(args.demo_tariff)
        print("[BRIDGE] GENERATED sample-data replay: 1 Hz, 300x scenario time, repeating 3-cycle groups; no hardware reads or relay actions", flush=True)
    elif args.replay:
        REPLAY_PATH = Path(args.replay)
        try:
            REPLAY_SAMPLES = load_recording(args.replay)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"[BRIDGE] Replaying {len(REPLAY_SAMPLES)} file samples, {REPLAY_REPEATS} times", flush=True)

    async with serve(websocket_handler, "127.0.0.1", 8000):
        print("[BRIDGE] Dashboard WebSocket: ws://127.0.0.1:8000/ws")
        if DEMO_REPLAY is not None:
            data_loop = demo_loop()
        elif replay_mode():
            data_loop = replay_loop()
        else:
            data_loop = serial_loop(args.port or "COM3")
        data_task = asyncio.create_task(data_loop)
        heartbeat_task = asyncio.create_task(heartbeat_loop())
        try:
            await asyncio.gather(data_task, heartbeat_task)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except serial.SerialException as exc:
        print(f"[BRIDGE] Serial error: {exc}")
        print("[BRIDGE] Re-run after connecting the ESP32 USB cable.")
