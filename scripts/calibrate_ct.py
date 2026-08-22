#!/usr/bin/env python3
"""
PZEM-004T v3.0 In-Field Calibration & Electrical Diagnostics Tool
Validates and calibrates physical ESP32 + PZEM-004T power nodes over MQTT.

Features:
  1. Authenticated MQTT telemetry subscriber (with username/password support).
  2. Live statistics: Mean Power, Voltage, Current, Power Factor, RMS Noise Floor.
  3. Power scaling factor calculation against known reference load (e.g. 100W bulb or 1000W heater).
  4. Line noise floor & dP/dt Arc-Fault Margin analysis (ensures idle noise << 1000 W/s).
  5. Automatically saves calibration profile to data/calibration_<device_id>.json.

Usage:
  python scripts/calibrate_ct.py node_fridge --ref-power 100.0 --samples 20
  python scripts/calibrate_ct.py node_kettle --ref-power 2000.0 --samples 30
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import List, Optional
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CALIBRATOR")


class PZEMCalibrator:
    def __init__(
        self,
        device_id: str,
        broker: str = "localhost",
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.device_id = device_id
        self.broker = broker
        self.port = port
        self.username = username or os.getenv("MQTT_USERNAME", "pipeline")
        self.password = password or os.getenv("MQTT_PASSWORD", "changeme_pipeline_password")

        self.power_samples: List[float] = []
        self.telemetry_samples: List[dict] = []
        self.timestamps: List[float] = []

        self.topic_power = f"home/sensor/{device_id}/power"
        self.topic_telemetry = f"home/sensor/{device_id}/telemetry"
        self.topic_status = f"home/sensor/{device_id}/status"

        self.client = mqtt.Client(client_id=f"calibrator_{device_id}_{int(time.time())}")
        if self.username:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
            client.subscribe([(self.topic_power, 0), (self.topic_telemetry, 0), (self.topic_status, 0)])
            self._connected = True
        else:
            logger.error(f"MQTT connection failed with code {rc} (Check username/password)")

    def _on_message(self, client, userdata, msg):
        payload_str = msg.payload.decode("utf-8", errors="replace").strip()
        topic = msg.topic

        if topic == self.topic_power:
            try:
                # Handle plain float or JSON
                if payload_str.startswith("{") and payload_str.endswith("}"):
                    data = json.loads(payload_str)
                    w = float(data.get("power", data.get("watts", data.get("W", data.get("w", 0.0)))))
                else:
                    w = float(payload_str)

                if not math.isnan(w) and not math.isinf(w):
                    self.power_samples.append(w)
                    self.timestamps.append(time.time())
                    logger.info(f"[{self.device_id}] Power reading #{len(self.power_samples)}: {w:.2f} W")
            except ValueError:
                pass

        elif topic == self.topic_telemetry:
            try:
                data = json.loads(payload_str)
                self.telemetry_samples.append(data)
                logger.info(f"[{self.device_id}] Diagnostics: V={data.get('v')}V, I={data.get('i')}A, PF={data.get('pf')}")
            except Exception:
                pass

        elif topic == self.topic_status:
            logger.info(f"[{self.device_id}] Status update: {payload_str}")

    def collect(self, num_samples: int = 20, timeout: float = 45.0) -> bool:
        logger.info(f"Connecting to broker and collecting {num_samples} samples from '{self.device_id}'...")
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False

        start = time.time()
        while len(self.power_samples) < num_samples and (time.time() - start) < timeout:
            time.sleep(0.2)

        self.client.loop_stop()
        self.client.disconnect()

        if len(self.power_samples) < num_samples:
            logger.warning(
                f"Collection timed out: gathered {len(self.power_samples)}/{num_samples} samples. "
                f"Ensure the ESP32 node '{self.device_id}' is powered on and transmitting on {self.topic_power}."
            )
            return len(self.power_samples) > 0

        return True

    def analyze(self, ref_power: Optional[float] = None, ref_voltage: float = 230.0) -> dict:
        if not self.power_samples:
            return {}

        n = len(self.power_samples)
        mean_p = sum(self.power_samples) / n
        var_p = sum((p - mean_p) ** 2 for p in self.power_samples) / max(1, n - 1)
        std_p = math.sqrt(var_p)
        min_p = min(self.power_samples)
        max_p = max(self.power_samples)

        # Compute dP/dt rate-of-change jitter
        rocs = []
        for i in range(1, len(self.power_samples)):
            dt = max(0.001, self.timestamps[i] - self.timestamps[i - 1])
            rocs.append(abs(self.power_samples[i] - self.power_samples[i - 1]) / dt)
        max_roc = max(rocs) if rocs else 0.0

        scale_factor = 1.0
        if ref_power is not None and ref_power > 0 and mean_p > 0:
            scale_factor = ref_power / mean_p

        # Extract latest voltage and PF if available
        latest_v = ref_voltage
        latest_pf = 1.0
        if self.telemetry_samples:
            latest_v = self.telemetry_samples[-1].get("v", ref_voltage)
            latest_pf = self.telemetry_samples[-1].get("pf", 1.0)

        # Arc-fault margin: how far is max idle jitter from 1000 W/s threshold
        arc_fault_margin_pct = max(0.0, (1.0 - (max_roc / 1000.0)) * 100.0)

        report = {
            "device_id": self.device_id,
            "sample_count": n,
            "mean_power_watts": round(mean_p, 3),
            "std_dev_watts": round(std_p, 3),
            "min_power_watts": round(min_p, 2),
            "max_power_watts": round(max_p, 2),
            "ref_power_watts": ref_power,
            "power_scaling_factor": round(scale_factor, 6),
            "latest_voltage_v": round(latest_v, 1),
            "latest_power_factor": round(latest_pf, 2),
            "max_dp_dt_w_per_s": round(max_roc, 2),
            "arc_fault_margin_pct": round(arc_fault_margin_pct, 1),
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="PZEM-004T In-Field Calibration & Diagnostics Utility")
    parser.add_argument("device_id", help="Target node identifier (e.g. node_fridge, node_kettle)")
    parser.add_argument("--ref-power", type=float, default=None, help="Known reference load wattage (e.g. 100.0W)")
    parser.add_argument("--samples", type=int, default=20, help="Number of telemetry samples to gather (default: 20)")
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"), help="MQTT broker address")
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")), help="MQTT broker port")
    parser.add_argument("--username", default=os.getenv("MQTT_USERNAME", "pipeline"), help="MQTT username")
    parser.add_argument("--password", default=os.getenv("MQTT_PASSWORD", "changeme_pipeline_password"), help="MQTT password")
    args = parser.parse_args()

    calibrator = PZEMCalibrator(
        device_id=args.device_id,
        broker=args.broker,
        port=args.port,
        username=args.username,
        password=args.password,
    )

    success = calibrator.collect(num_samples=args.samples)
    if not success:
        logger.error(f"❌ Failed to gather sufficient samples for '{args.device_id}'.")
        sys.exit(1)

    results = calibrator.analyze(ref_power=args.ref_power)

    print("\n" + "═" * 70)
    print(f" 📊 PZEM-004T CALIBRATION & ELECTRICAL DIAGNOSTICS: {args.device_id}")
    print("═" * 70)
    print(f" • Samples Gathered:        {results['sample_count']}")
    print(f" • Mean Measured Power:     {results['mean_power_watts']:.2f} W (±{results['std_dev_watts']:.2f} W)")
    print(f" • Min / Max Power:         {results['min_power_watts']:.1f} W / {results['max_power_watts']:.1f} W")
    if args.ref_power:
        print(f" • Reference Standard Load: {results['ref_power_watts']:.1f} W")
        print(f" • Power Scale Factor:      {results['power_scaling_factor']:.6f}")
    print(f" • Grid Voltage:            {results['latest_voltage_v']:.1f} V")
    print(f" • Power Factor:            {results['latest_power_factor']:.2f}")
    print(f" • Max Idle dP/dt:          {results['max_dp_dt_w_per_s']:.1f} W/s")
    print(f" • Arc-Fault Safety Margin: {results['arc_fault_margin_pct']:.1f}% (Threshold: 1000 W/s)")
    print("═" * 70)

    # Persist calibration profile
    os.makedirs("data", exist_ok=True)
    out_path = f"data/calibration_{args.device_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved calibration profile to: {out_path}\n")


if __name__ == "__main__":
    main()
