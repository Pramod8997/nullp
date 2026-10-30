#!/usr/bin/env python3
"""
Cross-platform Make runner for Windows and environments without GNU make.
Dispatches standard Makefile targets directly to Python/Node tools.
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Find Python executable
PYTHON = sys.executable

TARGETS = {
    "run": [PYTHON, str(ROOT / "scripts" / "run_hardware_system.py")],
    "dev": [PYTHON, str(ROOT / "scripts" / "run_hardware_system.py")],
    "live": [PYTHON, str(ROOT / "scripts" / "run_hardware_system.py")],
    "hardware": [PYTHON, str(ROOT / "scripts" / "run_hardware_system.py")],
    "demo": [PYTHON, str(ROOT / "scripts" / "demo_full_system.py"), "--demo"],
    "broker": [PYTHON, str(ROOT / "scripts" / "start_broker.py"), "0.0.0.0"],
    "dev-backend": [PYTHON, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
    "dev-frontend": ["npm", "run", "dev"],
    "dev-pipeline-hardware": [PYTHON, str(ROOT / "scripts" / "run_pipeline.py"), "--config", "config/config.hardware.yaml"],
    "dev-pipeline-demo": [PYTHON, str(ROOT / "scripts" / "run_pipeline.py"), "--config", "config/config.demo.yaml"],
    "relay-on": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'ON', hostname='127.0.0.1'); print('Sent ON (Energize Both)')"],
    "relay-off": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'OFF', hostname='127.0.0.1'); print('Sent OFF (Emergency Cutoff Both)')"],
    "relay-in1-on": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN1_ON', hostname='127.0.0.1'); print('Sent IN1_ON (Relay 1 / Laptop ON)')"],
    "relay-in1-off": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN1_OFF', hostname='127.0.0.1'); print('Sent IN1_OFF (Relay 1 / Laptop OFF)')"],
    "relay-in2-on": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN2_ON', hostname='127.0.0.1'); print('Sent IN2_ON (Relay 2 / Phone ON)')"],
    "relay-in2-off": [PYTHON, "-c", "import paho.mqtt.publish as pub; pub.single('home/plug/node_bench_agg/command', 'IN2_OFF', hostname='127.0.0.1'); print('Sent IN2_OFF (Relay 2 / Phone OFF)')"],
    "test": [PYTHON, "-m", "pytest", "tests/"],
    "test-backend": [PYTHON, "-m", "pytest", "tests/"],
    "test-frontend": ["npm", "test", "--", "--run"],
    "firmware-build": ["pio", "run", "-d", "firmware/esp32_node", "-e", "esp32dev"],
    "firmware-upload": ["pio", "run", "-d", "firmware/esp32_node", "-e", "esp32dev", "-t", "upload"],
    "firmware-monitor": ["pio", "device", "monitor", "-d", "firmware/esp32_node", "-b", "115200"],
}


def print_help():
    print("=" * 66)
    print("  Confidence-Aware Digital Twin EMS -- Windows Make Runner")
    print("=" * 66)
    print("  Available Targets:")
    for target in sorted(TARGETS.keys()):
        print(f"    make {target}")
    print("=" * 66)


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "run"
    if target in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    if target not in TARGETS:
        print(f"Unknown target: '{target}'. Available targets:")
        for t in sorted(TARGETS.keys()):
            print(f"  - {t}")
        sys.exit(1)

    cmd = TARGETS[target]
    cwd = ROOT / "frontend" if target in ("dev-frontend", "test-frontend") else ROOT

    use_shell = sys.platform == "win32" and (cmd[0] in ("npm", "pio"))
    try:
        sys.exit(subprocess.call(cmd, cwd=cwd, shell=use_shell))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
