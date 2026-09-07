#!/usr/bin/env python3
"""
Capture REAL power windows from the live PZEM/ESP32 node over MQTT.

This is the missing producer for `scripts/enroll_demo_devices.py --capture`:
the enrolment mechanism accepts an .npz of measured watt windows, and until now
nothing in the repo could create one from hardware.

It records ONLY what the broker delivers. It generates nothing, models nothing
and fills in nothing. If the node is not publishing, it times out and writes
no file.

Method
------
The firmware publishes `home/sensor/{DEVICE_ID}/power` as a plain float at 1 Hz
(main.cpp:441-446), so one 128-sample window spans 128 SECONDS of wall clock.
Capturing K windows back-to-back would take K*128 s, so windows are taken with
a sliding stride (default 16 s): K=8 windows costs 128 + 7*16 = 240 s per class.
Strided windows overlap and are therefore correlated — fine for a prototype mean
and a power envelope, and stated here rather than hidden.

Capture STEADY-STATE running load, one appliance at a time on an otherwise idle
socket. That is what `PrototypeRegistry._steady_watts` measures the envelope
from. It is deliberately NOT the switch-on transient: at the transient instant
the detector hands the classifier the 128 s BEFORE the event
(aggregate_nilm.py), a different window shape entirely.

Usage — one class per run, appending to the same file:
    python scripts/capture_bench_windows.py --class laptop        --out bench.npz
    python scripts/capture_bench_windows.py --class phone_charger --out bench.npz
    python scripts/capture_bench_windows.py --class projector     --out bench.npz
    python scripts/capture_bench_windows.py --class monitor       --out bench.npz

Then:
    python scripts/enroll_demo_devices.py --capture bench.npz \
        --out backend/models/weights_demo/prototype_registry_bench.pt
"""
import argparse
import asyncio
import getpass
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

try:
    import aiomqtt
except ImportError:
    sys.exit("aiomqtt not installed — activate the project venv (.venv)")

SEQ_LEN = 128
PROVENANCE_KEY = "__provenance__"
VALID_CLASSES = ("phone_charger", "laptop", "projector", "monitor",
                 "desktop_computer", "incandescent_lamp")


async def collect(broker, port, username, password, topic, n_windows, stride,
                  timeout_s, quiet_s):
    """
    Returns (windows (K,128) float32, stats dict). Raises SystemExit on no data.
    """
    samples: list[float] = []
    windows: list[np.ndarray] = []
    t_first = t_last = None
    next_at = SEQ_LEN                      # sample count at which to cut window 1
    bad = 0

    print(f"connecting to {broker}:{port} as {username or 'anonymous'} ...")
    async with aiomqtt.Client(broker, port=port,
                              username=username, password=password) as client:
        await client.subscribe(topic)
        print(f"subscribed to {topic}")
        print(f"need {SEQ_LEN} samples for the first window "
              f"(~{SEQ_LEN} s at 1 Hz), then one per {stride} s\n")
        deadline = time.monotonic() + timeout_s
        last_msg = time.monotonic()

        async def pump():
            nonlocal t_first, t_last, next_at, bad
            async for m in client.messages:
                raw = m.payload.decode(errors="replace").strip()
                try:
                    val = float(raw)
                except ValueError:
                    bad += 1
                    continue
                if not np.isfinite(val):
                    bad += 1
                    continue
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                samples.append(val)
                if len(samples) >= next_at:
                    windows.append(np.asarray(samples[-SEQ_LEN:], dtype=np.float32))
                    next_at = len(samples) + stride
                    w = windows[-1]
                    on = w[w > 20.0]
                    print(f"  window {len(windows)}/{n_windows}: "
                          f"mean={w.mean():7.2f} W  min={w.min():7.2f}  "
                          f"max={w.max():7.2f}  samples>20W={on.size}/{SEQ_LEN}")
                    if len(windows) >= n_windows:
                        return

        task = asyncio.create_task(pump())
        while not task.done():
            await asyncio.sleep(0.5)
            if time.monotonic() > deadline:
                task.cancel()
                break
            if samples and time.monotonic() - last_msg > quiet_s:
                pass
            if samples:
                last_msg = time.monotonic()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if not samples:
        raise SystemExit(
            f"\nNO DATA on {topic} within {timeout_s} s.\n"
            "The node is not publishing. Check, in order:\n"
            "  1. ESP32 powered and flashed;\n"
            "  2. its serial log shows [WiFi] Connected and [MQTT] Connected;\n"
            "  3. the broker accepts non-loopback connections "
            "(listener 1883 0.0.0.0);\n"
            "  4. EMS_MQTT_SERVER in include/secrets.h is this host's LAN IP;\n"
            "  5. EMS_DEVICE_ID matches the topic you are capturing."
        )
    if len(windows) < n_windows:
        print(f"\n⚠ captured only {len(windows)}/{n_windows} windows before the "
              f"{timeout_s} s timeout — keeping what arrived.")
    if not windows:
        raise SystemExit(
            f"\nOnly {len(samples)} samples arrived; a window needs {SEQ_LEN}. "
            f"At 1 Hz that is {SEQ_LEN} s — raise --timeout."
        )

    dur = (t_last - t_first) if (t_first and t_last) else 0.0
    stats = {
        "samples": len(samples),
        "bad_payloads": bad,
        "duration_s": round(dur, 1),
        "measured_rate_hz": round(len(samples) / dur, 3) if dur > 0 else None,
        "first_utc": datetime.fromtimestamp(t_first, timezone.utc).isoformat(),
        "last_utc": datetime.fromtimestamp(t_last, timezone.utc).isoformat(),
    }
    return np.stack(windows), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True, choices=VALID_CLASSES,
                    help="appliance class being measured RIGHT NOW on the socket")
    ap.add_argument("--out", default="bench_capture.npz")
    ap.add_argument("--device", default="node_bench_agg",
                    help="must match EMS_DEVICE_ID in the firmware's secrets.h")
    ap.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
    ap.add_argument("--username", default=os.environ.get("MQTT_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("MQTT_PASSWORD"),
                    help="prefer the MQTT_PASSWORD env var over the command line")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16,
                    help="seconds between window cuts; smaller = faster, more correlated")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--min-watts", type=float, default=20.0,
                    help="on-threshold; a window with nothing above it cannot "
                         "produce a power envelope")
    ap.add_argument("--attest-physical", default=None, metavar="NOTE",
                    help="operator attestation that a REAL appliance was on a REAL "
                         "PZEM for this capture, e.g. \"Dell 65W brick, ref meter "
                         "reads 63W\". Recorded verbatim in the file's provenance. "
                         "Without it the capture is stamped UNATTESTED, because the "
                         "transport alone cannot tell a physical node from "
                         "backend/scripts/simulate_esp32.py publishing to the same "
                         "topic.")
    args = ap.parse_args()

    topic = f"home/sensor/{args.device}/power"
    print(f"\n=== CAPTURE: {args.cls} ===")
    print(f"Run ONLY this appliance on the socket, in steady state, "
          f"and leave it running.\n")

    windows, stats = asyncio.run(collect(
        args.broker, args.port, args.username, args.password, topic,
        args.windows, args.stride, args.timeout, quiet_s=30.0))

    # Refuse to hand the enroller a file that cannot yield an envelope.
    usable = [i for i, w in enumerate(windows) if (w > args.min_watts).any()]
    if not usable:
        raise SystemExit(
            f"\nEvery window is below the {args.min_watts:.0f} W on-threshold "
            f"(max seen {windows.max():.1f} W). Either the load is not drawing, "
            f"the relay is OPEN, or the PZEM is reading zero. Nothing written."
        )
    if len(usable) < len(windows):
        print(f"\n⚠ dropping {len(windows) - len(usable)} window(s) with nothing "
              f"above {args.min_watts:.0f} W")
        windows = windows[usable]

    steady = float(np.median([np.median(w[w > args.min_watts]) for w in windows]))
    print(f"\ncaptured {len(windows)} window(s); median steady = {steady:.1f} W")
    print(f"measured publish rate = {stats['measured_rate_hz']} Hz "
          f"(firmware publishes 1 Hz; a large deviation means dropped messages)")
    if stats["bad_payloads"]:
        print(f"⚠ {stats['bad_payloads']} unparseable/non-finite payload(s) skipped")

    # Merge into an existing capture rather than clobbering other classes.
    payload, prov = {}, []
    if os.path.exists(args.out):
        with np.load(args.out, allow_pickle=False) as z:
            for k in z.files:
                if k == PROVENANCE_KEY:
                    prov = json.loads(str(z[k]))
                else:
                    payload[k] = z[k]
        if args.cls in payload:
            print(f"⚠ replacing the previous '{args.cls}' capture in {args.out}")
    payload[args.cls] = windows

    prov.append({
        "class": args.cls, "windows": int(len(windows)),
        "seq_len": SEQ_LEN, "stride_s": args.stride,
        "median_steady_w": round(steady, 2),
        "transport": "LIVE_MQTT", "broker": f"{args.broker}:{args.port}",
        "topic": topic, "device_id": args.device,
        # The transport says nothing about what generated the numbers: the demo
        # simulator publishes to this same topic. Only the operator can attest
        # that a real appliance sat on a real PZEM.
        "physical_attestation": args.attest_physical or "UNATTESTED",
        "captured_by": f"{getpass.getuser()}@{socket.gethostname()}",
        **stats,
    })
    payload[PROVENANCE_KEY] = np.array(json.dumps(prov))
    np.savez(args.out, **payload)

    if not args.attest_physical:
        print("\n⚠ stamped UNATTESTED: no --attest-physical note was given, so this "
              "file is NOT evidence of a physical measurement.")

    have = sorted(k for k in payload if k != PROVENANCE_KEY)
    print(f"\nwrote {args.out}  classes now present: {have}")
    print("\nNext:")
    print("  python scripts/enroll_demo_devices.py --capture "
          f"{args.out} --out backend/models/weights_demo/prototype_registry_bench.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
