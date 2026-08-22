"""
ESP32 + PZEM-004T Dual-Core Firmware Simulator
Exact software model of firmware/esp32_node/src/main.cpp for full-stack closed-loop simulation.

Replicates:
  • Core 0 (High Priority @ 100ms):
      - PZEM-004T Modbus register polling (V, I, W, PF)
      - Sliding baseline inrush suppression
      - Edge Arc-Fault trip (dP/dt > 1000 W/s)
      - Overcurrent cutoff (125% rated)
      - Hardware relay cutoff with zero network dependency
  • Core 1 (Standard Priority @ 1Hz):
      - 1Hz fast power telemetry (`home/sensor/{id}/power`)
      - 10s diagnostic telemetry (`home/sensor/{id}/telemetry`)
      - Relay command handling (ON/OFF/WARNING)
      - Relay ACKs (`ON_CONFIRMED`, `OFF_CONFIRMED`, `LOCKOUT_NACK`)
      - 5-minute anti-thrashing lockout
"""

import asyncio
import json
import logging
import math
import random
import time
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("ESP32_FIRMWARE_SIM")


class VirtualPZEM004T:
    """Simulates PZEM-004T v3.0 Modbus RTU registers."""
    def __init__(self, voltage: float = 230.0, frequency: float = 50.0):
        self.voltage = voltage
        self.frequency = frequency
        self.current = 0.0
        self.active_power = 0.0
        self.power_factor = 0.95
        self.energy_kwh = 0.0

    def set_load(self, target_watts: float, pf: float = 0.95):
        self.active_power = max(0.0, target_watts)
        self.power_factor = pf
        if self.voltage > 0:
            apparent_power = self.active_power / max(0.1, self.power_factor)
            self.current = apparent_power / self.voltage
        else:
            self.current = 0.0


class ESP32FirmwareNode:
    """Exact emulation of ESP32 firmware running Dual-Core FreeRTOS."""
    def __init__(
        self,
        device_id: str,
        rated_watts: float = 200.0,
        relay_active_low: bool = True,
        mqtt_publish_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        self.device_id = device_id
        self.rated_watts = rated_watts
        self.relay_active_low = relay_active_low
        self.mqtt_publish = mqtt_publish_fn

        # Hardware Pins & State
        self.gpio18_relay_state = False   # False = OFF, True = ON
        self.relay_locked = False
        self.lock_start_time = 0.0
        self.safety_lockout_seconds = 300.0  # 5-minute lockout

        # PZEM Instance
        self.pzem = VirtualPZEM004T()

        # Shared State (Spinlock protected in C++)
        self.shared_power_watts = 0.0
        self.shared_voltage = 230.0
        self.shared_current = 0.0
        self.shared_pf = 0.95
        self.shared_arc_fault = False
        self.shared_arc_fault_roc = 0.0

        # Core 0 State
        self._last_watts = 0.0
        self._baseline_ring = [0.0] * 5
        self._baseline_idx = 0
        self._baseline_fill = 0
        self._last_read_time = time.time()
        self._core0_running = True

        # Core 1 State
        self._last_1hz_msg = 0.0
        self._last_10s_telemetry = 0.0

        # Topics
        self.topic_power = f"home/sensor/{device_id}/power"
        self.topic_telemetry = f"home/sensor/{device_id}/telemetry"
        self.topic_command = f"home/plug/{device_id}/command"
        self.topic_status = f"home/sensor/{device_id}/status"
        self.topic_ack = f"home/plug/{device_id}/ack"

    def set_relay(self, on: bool):
        """Actuate physical relay GPIO pin."""
        self.gpio18_relay_state = on
        if not on:
            self.pzem.set_load(0.0)

    # ═════════════════════════════════════════════════════════════════════
    # CORE 0: High-Priority Safety Loop (Runs every 100ms)
    # ═════════════════════════════════════════════════════════════════════
    def core0_safety_step(self, sim_dt: float = 0.1):
        """Execute one 100ms Core 0 cycle."""
        now = time.time()
        power_w = self.pzem.active_power if self.gpio18_relay_state else 0.0
        voltage = self.pzem.voltage
        current = self.pzem.current
        pf = self.pzem.power_factor

        # 1. Calculate pre-step sliding baseline average from history
        baseline_avg = sum(self._baseline_ring[:self._baseline_fill]) / max(1, self._baseline_fill) if self._baseline_fill > 0 else 0.0
        is_normal_inrush = (baseline_avg < 50.0) and (self._last_watts < (baseline_avg + 100.0))

        # 2. Edge Arc-Fault Proxy Detection (dP/dt in W/s) — only on positive power surges
        if sim_dt > 0.0 and (power_w > self._last_watts):
            roc = (power_w - self._last_watts) / sim_dt
            if roc > 1000.0 and not is_normal_inrush:
                # ⚡ IMMEDIATE PHYSICAL CUTOFF — ZERO NETWORK LATENCY
                self.set_relay(False)
                self.shared_arc_fault = True
                self.shared_arc_fault_roc = roc
                self.relay_locked = True
                self.lock_start_time = now
                logger.warning(
                    f"[CORE 0] ⚡ EDGE ARC-FAULT on {self.device_id}! "
                    f"dP/dt={roc:.0f} W/s > 1000 W/s -> Relay CUTOFF instantly!"
                )

        # 3. Overcurrent Cutoff (125% of rated) — allow transient inrush during motor startup
        critical_watts = self.rated_watts * 1.25
        if power_w > critical_watts and not is_normal_inrush:
            self.set_relay(False)
            self.relay_locked = True
            self.lock_start_time = now
            logger.warning(
                f"[CORE 0] ⚡ OVERCURRENT on {self.device_id}! "
                f"{power_w:.1f}W > {critical_watts:.1f}W -> Relay CUTOFF instantly!"
            )

        # 4. Update sliding baseline (5 samples) with current measurement
        self._baseline_ring[self._baseline_idx] = power_w
        self._baseline_idx = (self._baseline_idx + 1) % 5
        if self._baseline_fill < 5:
            self._baseline_fill += 1

        self._last_watts = power_w

        # 5. Write to shared memory
        self.shared_power_watts = power_w
        self.shared_voltage = voltage
        self.shared_current = current
        self.shared_pf = pf

    # ═════════════════════════════════════════════════════════════════════
    # CORE 1: Standard Priority Arduino Loop (MQTT + Telemetry)
    # ═════════════════════════════════════════════════════════════════════
    async def handle_mqtt_command(self, command: str):
        """Simulates Core 1 callback() receiving an MQTT relay command."""
        cmd = command.strip().upper()
        now = time.time()

        # Check 5-minute lockout
        if self.relay_locked:
            if (now - self.lock_start_time) > self.safety_lockout_seconds:
                self.relay_locked = False
                logger.info(f"[{self.device_id}] 5-minute safety lockout expired. Relay unlocked.")

        if cmd == "ON":
            if not self.relay_locked:
                self.set_relay(True)
                if self.mqtt_publish:
                    await self.mqtt_publish(self.topic_ack, "ON_CONFIRMED")
                logger.info(f"[{self.device_id}] Relay turned ON (ACK: ON_CONFIRMED)")
            else:
                if self.mqtt_publish:
                    await self.mqtt_publish(self.topic_ack, "LOCKOUT_NACK")
                logger.warning(f"[{self.device_id}] ON rejected: Relay locked out (NACK sent)")

        elif cmd == "OFF":
            self.set_relay(False)
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_ack, "OFF_CONFIRMED")
            logger.info(f"[{self.device_id}] Relay turned OFF (ACK: OFF_CONFIRMED)")

    async def core1_telemetry_tick(self, force_publish: bool = False):
        """Simulates Core 1 1Hz power broadcast and 10s diagnostic broadcast."""
        now = time.time()
        power_w = self.shared_power_watts

        # Check for arc-fault alert to publish
        if self.shared_arc_fault:
            alert_msg = f"EDGE_ARC_FAULT:dP/dt={self.shared_arc_fault_roc:.0f}W/s"
            self.shared_arc_fault = False
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_status, alert_msg)

        # 1Hz Fast Power publish (plain float string)
        if force_publish or (now - self._last_1hz_msg >= 1.0):
            self._last_1hz_msg = now
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_power, f"{power_w:.2f}")

        # 10s Rich Diagnostics publish (JSON)
        if force_publish or (now - self._last_10s_telemetry >= 10.0):
            self._last_10s_telemetry = now
            diag_payload = json.dumps({
                "v": round(self.shared_voltage, 1),
                "i": round(self.shared_current, 2),
                "w": round(power_w, 1),
                "pf": round(self.shared_pf, 2)
            })
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_telemetry, diag_payload)
