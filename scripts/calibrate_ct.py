#!/usr/bin/env python3
"""Calibration utility for ESP32 CT‑clamp sensors.

Usage:
    python scripts/calibrate_ct.py <DEVICE_ID>

The script:
1. Publishes a `CALIBRATE_REQUEST` MQTT message for the target device.
2. Waits for a `POWER_READING` response while a known 100 W load is attached.
3. Computes a correction factor based on the measured RMS power.
4. Updates the constants `CT_RATIO` and `BURDEN_R` in `firmware/esp32_node/src/main.cpp`.
5. Writes the updated values back to the source file.

The correction factor is persisted in `Phase2_Implementation_Plan.md` manually by the operator.
"""

import argparse, json, time, os, sys
import paho.mqtt.client as mqtt

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
CALIBRATE_TOPIC = "home/calibrate/{device_id}"
POWER_TOPIC = "home/sensor/{device_id}/power"

# Simple blocking wait for a single power reading
class Calibrator:
    def __init__(self, device_id):
        self.device_id = device_id
        self.reading = None
        self.client = mqtt.Client()
        self.client.on_message = self._on_message
        self.client.connect(MQTT_BROKER, 1883, 60)
        self.client.subscribe(POWER_TOPIC.format(device_id=device_id))
        self.client.loop_start()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.reading = payload.get("rms_power")
        except Exception:
            pass

    def request_calibration(self):
        topic = CALIBRATE_TOPIC.format(device_id=self.device_id)
        self.client.publish(topic, json.dumps({"action": "start", "load_w": 100}))

    def get_reading(self, timeout=10):
        start = time.time()
        while self.reading is None and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self.reading

def update_firmware_constants(device_id, correction_factor):
    src_path = os.path.abspath("firmware/esp32_node/src/main.cpp")
    with open(src_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if "#define CT_RATIO" in line:
            parts = line.split()
            orig = float(parts[-1])
            new = orig * correction_factor
            new_line = f"#define CT_RATIO {new:.6f}\n"
            new_lines.append(new_line)
        elif "#define BURDEN_R" in line:
            # For simplicity keep the same value; real calibration may adjust this.
            new_lines.append(line)
        else:
            new_lines.append(line)
    with open(src_path, "w") as f:
        f.writelines(new_lines)
    print(f"Updated CT_RATIO for {device_id} with factor {correction_factor:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Calibrate CT‑clamp for a device")
    parser.add_argument("device_id", help="Device identifier (e.g., node_fridge)")
    args = parser.parse_args()

    cal = Calibrator(args.device_id)
    cal.request_calibration()
    reading = cal.get_reading()
    if reading is None:
        sys.exit("No power reading received – ensure a 100 W load is attached and the device is online.")
    # Expected 100 W, compute correction factor
    factor = 100.0 / reading
    update_firmware_constants(args.device_id, factor)
    cal.client.loop_stop()
    cal.client.disconnect()

if __name__ == "__main__":
    main()
