#!/usr/bin/env python3
"""
Full-System Digital Twin Software Demo & Verification Runner
Orchestrates the entire Smart Home EMS in pure software (no physical hardware required).

Launches concurrently:
  1. MQTT Broker (via local mosquitto or background docker)
  2. Python NILM + ProtoNet + RL Pipeline Orchestrator (`scripts/run_pipeline.py`)
  3. FastAPI REST & WebSocket Backend (`src.api.main:app` on Port 8000)
  4. Virtual ESP32+PZEM Hardware Fleet Simulator (`backend/scripts/simulate_esp32.py --all`)

Provides an interactive CLI control panel to test:
  • Live Real-Time Dashboard at http://localhost:5173
  • Real-time ProtoNet NILM classifications (10 appliance classes)
  • OpenMax unknown device detection & Few-Shot labeling
  • ISO 7730 PMV thermal comfort & RL load-shedding actuation
  • Indian DISCOM Time-of-Use tariff tracking (₹ INR)
  • Safety cutoffs (Overcurrent >125% and Arc-Fault >1000 W/s)
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
import shutil

# Ensure workspace root is in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_ROOT)


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         ⚡ CONFIDENCE-AWARE DIGITAL TWIN EMS — FULL SOFTWARE DEMO         ║
╚════════════════════════════════════════════════════════════════════════════╝
  Mode: 100% Pure Software Simulation (Virtual Hardware Fleet)
  Stack: Mosquitto MQTT + 1D-CNN ProtoNet + RL + FastAPI + React 19 Frontend
  Grid:  Indian Grid (230V / 50Hz) | DISCOM ToU Slabs (₹8.0/₹6.5/₹4.0 per kWh)
    """)


class SystemOrchestrator:
    def __init__(self):
        self.processes = []
        self.running = True
        self.demo_mode = '--demo' in sys.argv or os.environ.get('EMS_DEMO', '') == '1'

    def check_broker(self) -> bool:
        """Check if Mosquitto broker is reachable on port 1883."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(("localhost", 1883))
            s.close()
            return True
        except Exception:
            return False

    def start_process(self, cmd: list, name: str, env_extra: dict = None) -> subprocess.Popen:
        env = os.environ.copy()
        env["PYTHONPATH"] = WORKSPACE_ROOT
        if env_extra:
            env.update(env_extra)
        
        proc = subprocess.Popen(
            cmd,
            cwd=WORKSPACE_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.processes.append((name, proc))
        print(f"  🚀 Started {name:<20} [PID {proc.pid}]")
        return proc

    def stop_all(self):
        print("\n\n  🛑 Shutting down all services gracefully...")
        for name, proc in reversed(self.processes):
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
                print(f"  ✓ Stopped {name}")
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("  ✅ All processes terminated cleanly.\n")

    def run(self):
        print_banner()

        # Step 1: Verify / Start MQTT Broker
        broker_active = self.check_broker()
        if broker_active:
            print("  ✅ Local MQTT broker detected on port 1883")
        else:
            if shutil.which("mosquitto"):
                print("  🔌 Launching local Mosquitto daemon...")
                self.start_process(
                    ["mosquitto", "-c", "mosquitto/config/mosquitto.conf"],
                    "Mosquitto Broker"
                )
                time.sleep(1.0)
            else:
                print("  ⚠️ Mosquitto not found in PATH — attempting python embedded broker...")
                self.start_process(
                    [sys.executable, "scripts/start_broker.py"],
                    "Python MQTT Broker"
                )
                time.sleep(1.5)

        # Step 2: Start Pipeline Orchestrator
        pipeline_cmd = [sys.executable, "scripts/run_pipeline.py"]
        if self.demo_mode:
            pipeline_cmd.extend(["--config", "config/config.demo.yaml"])
        self.start_process(pipeline_cmd, "Pipeline Orchestrator")
        time.sleep(1.5)

        # Step 3: Start FastAPI Backend
        self.start_process(
            [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            "FastAPI Backend"
        )
        time.sleep(1.5)

        # Step 4: Start ESP32 Virtual Hardware Simulator
        sim_cmd = [sys.executable, "backend/scripts/simulate_esp32.py"]
        if self.demo_mode:
            sim_cmd.append("--demo")
        else:
            sim_cmd.append("--all")
        self.start_process(sim_cmd, "Virtual Hardware Fleet")
        time.sleep(1.0)

        print("\n" + "═" * 78)
        print(" 🌐 SYSTEM ACCESS URLS:")
        print("   • React 19 Frontend Dashboard:  http://localhost:5173")
        print("   • FastAPI Swagger API Docs:     http://localhost:8000/docs")
        print("   • Live WebSockets Stream:       ws://localhost:8000/ws")
        print("   • REST Healthcheck:             http://localhost:8000/health")
        print("═" * 78)
        print(" 💡 To start the React frontend in another terminal:")
        print("    cd frontend && npm run dev")
        print("═" * 78)
        print("\n  [Press Ctrl+C at any time to stop the full simulation stack]\n")

        # Stream active logs
        try:
            while self.running:
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        print(f"  ⚠️ Process {name} exited with code {proc.returncode}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full-System EMS Demo")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with consumer electronics")
    args = parser.parse_args()
    if args.demo:
        os.environ['EMS_DEMO'] = '1'
    orchestrator = SystemOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        orchestrator.stop_all()
        print(f"Error: {e}")
        sys.exit(1)
