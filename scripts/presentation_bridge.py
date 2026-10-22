"""USB-only presentation bridge for the ESP32 dashboard.

This is deliberately a presentation tool, not the production MQTT path. It
reads the ESP32 USB serial heartbeat and publishes a clearly-labelled,
synthetic 1 Hz load to the existing dashboard WebSocket when the PZEM is not
connected. No mains or PZEM data is claimed by this mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from typing import Any

import serial
from websockets.asyncio.server import ServerConnection, serve


DEVICE_ID = "node_bench_agg"
DEVICE_LABEL = "USB DEMO — synthetic laptop (PZEM not connected)"
DEMO_CUTOFF_AFTER_SAMPLES = 5
CLIENTS: set[ServerConnection] = set()
SAMPLE = 0
LAST_RELAY = "OFF"
DEMO_CUTOFF_SENT = False


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


def demo_power(sample: int) -> float:
    # Laptop-like movement, always below the 250 W physical prototype envelope.
    return round(105.0 + 45.0 * math.sin(sample / 4.0) + 12.0 * math.sin(sample / 1.7), 1)


def demo_cutoff_event() -> str:
    """Return a presentation marker; it does not represent a physical cutoff."""
    return message(
        "SAFETY_CUTOFF",
        device_id=DEVICE_ID,
        severity="critical",
        presentation_only=True,
        demo=True,
        pzem_connected=False,
        relay_action="NONE",
        message=(
            "DEMO ONLY: synthetic overcurrent cutoff marker for presentation; "
            "no PZEM measurement, relay command, or mains contact action was issued."
        ),
    )


async def websocket_handler(websocket: ServerConnection) -> None:
    CLIENTS.add(websocket)
    initial_power = demo_power(SAMPLE)
    await websocket.send(
        message(
            "init_state",
            devices={
                DEVICE_ID: {
                    "power": initial_power,
                    "state": "ON",
                    "classification": "demo:laptop",
                    "confidence": 0.96,
                    "rated": 250,
                    "label": DEVICE_LABEL,
                    "source": "synthetic_usb_demo",
                    "presentation_only": True,
                    "pzem_connected": False,
                    "last_seen": time.time(),
                }
            },
            pipeline_status="USB_PRESENTATION_MODE",
        )
    )
    await websocket.send(
        message(
            "SAFETY_WARNING",
            device_id=DEVICE_ID,
            severity="info",
            message="USB presentation mode: synthetic telemetry; PZEM is not connected.",
        )
    )
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)


async def heartbeat_loop() -> None:
    while True:
        await broadcast(message("heartbeat", status="USB_PRESENTATION_MODE"))
        await asyncio.sleep(2)


async def publish_sample(serial_line: str) -> None:
    global SAMPLE, LAST_RELAY, DEMO_CUTOFF_SENT
    SAMPLE += 1
    if "Relay=ON" in serial_line:
        LAST_RELAY = "ON"
    elif "Relay=OFF" in serial_line:
        LAST_RELAY = "OFF"

    power = demo_power(SAMPLE)
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
            state="ON",
            classification="demo:laptop",
            confidence=0.96,
            pmv=0.1,
            timestamp=now,
            label=DEVICE_LABEL,
            rated=250,
            source="synthetic_usb_demo",
            presentation_only=True,
            pzem_connected=False,
        )
    )
    if not DEMO_CUTOFF_SENT and SAMPLE >= DEMO_CUTOFF_AFTER_SAMPLES:
        DEMO_CUTOFF_SENT = True
        await broadcast(demo_cutoff_event())


async def serial_loop(port: str) -> None:
    print(f"[BRIDGE] Opening ESP32 serial port {port} at 115200")
    serial_port = serial.Serial(port, 115200, timeout=1)
    try:
        while True:
            raw = await asyncio.to_thread(serial_port.readline)
            if raw:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("[USB]"):
                    print(line, flush=True)
                    await publish_sample(line)
            else:
                await asyncio.sleep(0.05)
    finally:
        serial_port.close()


async def offline_loop() -> None:
    print("[BRIDGE] No serial port supplied; using timed synthetic samples")
    while True:
        await publish_sample("[USB] PZEM=INVALID Relay=OFF")
        await asyncio.sleep(1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    args = parser.parse_args()

    async with serve(websocket_handler, "127.0.0.1", 8000):
        print("[BRIDGE] Dashboard WebSocket: ws://127.0.0.1:8000/ws")
        serial_task = asyncio.create_task(serial_loop(args.port))
        heartbeat_task = asyncio.create_task(heartbeat_loop())
        try:
            await asyncio.gather(serial_task, heartbeat_task)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except serial.SerialException as exc:
        print(f"[BRIDGE] Serial error: {exc}")
        print("[BRIDGE] Re-run after connecting the ESP32 USB cable.")
