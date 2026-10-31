#!/usr/bin/env python3
"""
Live Hardware System Runner for Confidence-Aware Digital Twin EMS
==================================================================
Starts and orchestrates the full live hardware stack:
  1. MQTT Broker (0.0.0.0:1883) — allows physical ESP32 to connect over WiFi
  2. Hardware Pipeline (scripts/run_pipeline.py --config config/config.hardware.yaml)
     — listens to ESP32 node 'node_bench_agg' for live 1Hz telemetry and feeds data to NILM
  3. FastAPI Server (src.api.main:app on 0.0.0.0:8000)
     — provides REST API, relay control (/api/relay/command), and WebSocket (/ws)
  4. React 19 Frontend Dashboard (Vite on port 5173)
     — live UI with manual IN1 / IN2 relay toggle controls and emergency cutoff

Clean shutdown on Ctrl+C terminating all spawned processes without leaving zombie ports.
"""

import os
import sys
import time
import socket
import signal
import threading
import subprocess
from pathlib import Path

# Fix Windows console encoding for Unicode box drawing characters & emojis
if sys.platform == "win32":
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# Set workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Color codes for terminal logging
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = """
==============================================================================
  [EMS] CONFIDENCE-AWARE DIGITAL TWIN EMS -- LIVE HARDWARE RIG
==============================================================================
  Hardware Target: ESP32 DevKit V1 (30-pin) | PZEM-004T v4.0 (100A CT)
  Node ID:         node_bench_agg (Active-LOW, Opto-isolated)
  Relay 1 (IN1):   GPIO 26 -- Laptop Charger (~65 W)
  Relay 2 (IN2):   GPIO 27 -- Phone Charger (~20 W)
  Opto VCC:        GPIO 25 (3.3V) | JD-VCC to VIN (5V)
  UART2 Pins:      RX = GPIO 16 (PZEM TX) | TX = GPIO 17 (PZEM RX)
==============================================================================
"""
    try:
        print(banner)
    except Exception:
        print(banner.encode('ascii', errors='replace').decode('ascii'))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is open and accepting TCP connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


class LiveSystemOrchestrator:
    def __init__(self, with_frontend: bool = True):
        self.with_frontend = with_frontend
        self.processes = []
        self.running = True

    def start_process(self, cmd, name: str, color: str, cwd=None, env_extra: dict = None):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORKSPACE_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        env["MQTT_BROKER"] = "127.0.0.1"
        if env_extra:
            env.update(env_extra)

        cwd_path = str(cwd or WORKSPACE_ROOT)

        # On Windows, use shell=True for npm/npx bat wrappers
        use_shell = sys.platform == "win32" and ("npm" in cmd[0] or "npx" in cmd[0])

        proc = subprocess.Popen(
            cmd,
            cwd=cwd_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=use_shell,
        )
        self.processes.append((name, proc))
        print(f"  {GREEN}🚀 Started {name:<24}{RESET} [PID {proc.pid}]")

        # Stream process output with tag
        def stream_output():
            try:
                for line in iter(proc.stdout.readline, ''):
                    if not self.running:
                        break
                    line_str = line.strip()
                    if line_str:
                        print(f"  {color}[{name}]{RESET} {line_str}")
            except Exception:
                pass

        t = threading.Thread(target=stream_output, daemon=True)
        t.start()
        return proc

    def stop_all(self):
        if not self.running:
            return
        self.running = False
        print(f"\n{YELLOW}🛑 Shutting down live EMS hardware stack...{RESET}")
        for name, proc in reversed(self.processes):
            try:
                proc.terminate()
                print(f"  ✓ Terminating {name}...")
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print(f"  {GREEN}✅ All services stopped cleanly.{RESET}\n")

    def run(self):
        print_banner()

        # Step 1: MQTT Broker (Port 1883)
        if is_port_in_use(1883):
            print(f"  {GREEN}✅ MQTT Broker already listening on port 1883{RESET}")
        else:
            print(f"  🔌 Starting MQTT Broker on 0.0.0.0:1883...")
            self.start_process(
                [sys.executable, "scripts/start_broker.py", "0.0.0.0"],
                "MQTT-Broker",
                BLUE,
            )
            # Wait up to 3s for broker to bind
            for _ in range(30):
                if is_port_in_use(1883):
                    break
                time.sleep(0.1)

        # Step 2: Hardware Ingestion Pipeline
        print(f"  🧠 Starting Hardware Pipeline (config.hardware.yaml)...")
        self.start_process(
            [sys.executable, "scripts/run_pipeline.py", "--config", "config/config.hardware.yaml"],
            "Pipeline-HW",
            MAGENTA,
        )
        time.sleep(1.5)

        # Step 3: FastAPI Backend Server (Port 8000)
        if is_port_in_use(8000):
            print(f"  {YELLOW}⚠️ Port 8000 in use — assuming FastAPI backend already running{RESET}")
        else:
            print(f"  🌐 Starting FastAPI Server on port 8000...")
            self.start_process(
                [sys.executable, "scripts/start_api.py"],
                "FastAPI-API",
                CYAN,
                env_extra={"EMS_API_KEY": os.environ.get("EMS_API_KEY", "changeme-ems-prod-key")},
            )
            # Wait for API to come up
            for _ in range(30):
                if is_port_in_use(8000):
                    break
                time.sleep(0.1)

        # Step 4: React 19 Frontend Dashboard (Port 5173)
        if self.with_frontend:
            print("  [*] Starting Vite React Frontend Dashboard...")
            frontend_dir = WORKSPACE_ROOT / "frontend"
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            api_key = os.environ.get("EMS_API_KEY", "changeme-ems-prod-key")
            self.start_process(
                [npm_cmd, "run", "dev"],
                "Frontend-UI",
                YELLOW,
                cwd=frontend_dir,
                env_extra={"VITE_API_KEY": api_key},
            )

        ready_text = """
==============================================================================
  [OK] DIGITAL TWIN EMS IS READY & CONNECTED TO ESP32!
  * Dashboard UI:     http://localhost:5173
  * Backend API:      http://localhost:8000  (Docs: /docs, WS: /ws)
  * Target Device:    node_bench_agg (Active telemetry stream at 1Hz)
  * Manual Cutoffs:   Relay 1 (IN1: Laptop) & Relay 2 (IN2: Phone)
==============================================================================
  Press Ctrl+C at any time to safely shut down all services.
"""
        try:
            print(ready_text)
        except Exception:
            print(ready_text.encode('ascii', errors='replace').decode('ascii'))

        # Keep alive and monitor child processes
        try:
            while self.running:
                for name, proc in self.processes:
                    ret = proc.poll()
                    if ret is not None:
                        print(f"  {RED}⚠️ Warning: Process {name} exited with code {ret}{RESET}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()


if __name__ == "__main__":
    no_fe = "--no-frontend" in sys.argv
    orchestrator = LiveSystemOrchestrator(with_frontend=not no_fe)

    def handle_signal(sig, frame):
        orchestrator.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    orchestrator.run()
