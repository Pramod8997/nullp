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
      - Status alerts (`ONLINE`/`OFFLINE`/`OVERCURRENT:`/`SERVER_TIMEOUT`/`EDGE_ARC_FAULT:`)
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
        # PF init 1.0 mirrors the POWER_FACTOR constant / sharedPf init in
        # main.cpp.
        self.power_factor = 1.0
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
        # Default mirrors the RATED_WATTS constant in main.cpp (250.0 — the
        # 250 W prototype envelope; 125% cutoff trips at 312 W, 3.7x under
        # the 5 A fuse).
        rated_watts: float = 250.0,
        relay_active_low: bool = False,
        mqtt_publish_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        # relay_active_low mirrors the RELAY_ACTIVE_LOW constant in main.cpp,
        # which the
        # locked HARDWARE_FINAL_SPEC.md pins to false (active-HIGH net at GPIO
        # 18: HIGH = closed, LOW = open, Hi-Z = open). Defaulting this to True
        # modelled the inverted polarity of defect B-7 — the one that energised
        # the load at boot and made every cutoff CLOSE the relay.
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
        # PF init 1.0 mirrors the sharedPf init in main.cpp.
        self.shared_pf = 1.0
        self.shared_arc_fault = False
        self.shared_arc_fault_roc = 0.0
        # Core-0 -> core-1 overcurrent latch. Mirrors the arc-fault flag
        # handshake: core 0 opens the relay and raises the latch; the core-1
        # tick consumes it and takes the lockout (the core-1 loop's overcurrent
        # latch block in main.cpp does the lock, not the safety task).
        self.shared_overcurrent_latch = False

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
        # ONLINE is announced once on the first core-1 tick, mirroring the
        # retained publish on MQTT connect (reconnectMQTT() in main.cpp).
        self._announced_online = False
        # Server heartbeat (the lastServerHB refresh in callback() in
        # main.cpp). The twin has no server, so
        # this stays None (no timeout possible) until the ONLINE announcement
        # (the connect equivalent) or a command handler call or a test sets it.
        self.last_server_hb: Optional[float] = None

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

    @property
    def gpio18_level(self) -> bool:
        """Electrical level driven onto GPIO 18 (True = HIGH, False = LOW).

        Mirrors setRelay() in main.cpp. `gpio18_relay_state` is the *logical*
        relay state (True = load energised); this is the pin level that produces
        it, so a reintroduced polarity inversion (B-7) becomes observable in the
        twin instead of being invisible as it was while relay_active_low was
        stored and never read.
        """
        if self.relay_active_low:
            return not self.gpio18_relay_state
        return self.gpio18_relay_state

    # ═════════════════════════════════════════════════════════════════════
    # CORE 0: High-Priority Safety Loop (Runs every 100ms)
    # ═════════════════════════════════════════════════════════════════════
    def core0_safety_step(self, sim_dt: float = 0.1):
        """Execute one 100ms Core 0 cycle."""
        power_w = self.pzem.active_power if self.gpio18_relay_state else 0.0
        voltage = self.pzem.voltage
        current = self.pzem.current
        pf = self.pzem.power_factor

        # Reject invalid PZEM reads before they can reach safety state.
        # Mirrors the isnan() guard in SafetySamplingTask() in main.cpp, which
        # skips the whole
        # cycle: _last_watts, the baseline ring and the shared block are all
        # left untouched. Without this the twin latched NaN into _last_watts and
        # _baseline_ring (permanently forcing is_normal_inrush False) and Core 1
        # published a bare "nan" on the power topic plus non-standard JSON
        # ({"w": NaN}) on the telemetry topic, poisoning the NILM stage.
        if not all(math.isfinite(v) for v in (power_w, voltage, current, pf)):
            logger.warning(
                f"[CORE 0] Invalid PZEM read on {self.device_id} "
                f"(W={power_w}, V={voltage}, I={current}, PF={pf}) -> cycle skipped."
            )
            return

        # 1. Calculate pre-step sliding baseline average from history
        baseline_avg = sum(self._baseline_ring[:self._baseline_fill]) / max(1, self._baseline_fill) if self._baseline_fill > 0 else 0.0
        is_normal_inrush = (baseline_avg < 50.0) and (self._last_watts < (baseline_avg + 100.0))

        # 2. Edge Arc-Fault Proxy Detection (dP/dt in W/s) — only on positive power surges
        if sim_dt > 0.0 and (power_w > self._last_watts):
            # Effective dt is clamped to [0.134, 0.16] s: the PZEM-004T updates
            # its registers every 200 ms and the Modbus transaction itself takes
            # time, so hardware dt is ~0.134-0.16 s even though the twin is
            # stepped faster. PZEM reads themselves stay instantaneous (no
            # register-value lag) so first-sample overcurrent trips still work.
            # Under this clamp a 120 W step reads 750-896 W/s (no trip, matching
            # hardware); steps > 134 W still exceed the 1000 W/s threshold
            # (134/0.134 is exactly 1000 — the boundary itself does not trip,
            # matching main.cpp's strict >).
            effective_dt = min(max(sim_dt, 0.134), 0.16)
            roc = (power_w - self._last_watts) / effective_dt
            if roc > 1000.0 and not is_normal_inrush:
                # ⚡ IMMEDIATE PHYSICAL CUTOFF — ZERO NETWORK LATENCY
                # The relay opens here in core 0; the lockout is taken by the
                # core-1 tick when it consumes shared_arc_fault, mirroring the
                # arc-fault cutoff branch in SafetySamplingTask() + the core-1
                # arc-fault acknowledgment block in loop() (main.cpp).
                self.set_relay(False)
                self.shared_arc_fault = True
                self.shared_arc_fault_roc = roc
                logger.warning(
                    f"[CORE 0] ⚡ EDGE ARC-FAULT on {self.device_id}! "
                    f"dP/dt={roc:.0f} W/s > 1000 W/s -> Relay CUTOFF instantly!"
                )

        # 3. Overcurrent Cutoff (125% of rated) — UNCONDITIONAL.
        # Inrush suppression deliberately does NOT gate this path, matching
        # the EDGE_ROC_THRESHOLD / isNormalInrush arc-fault branch in
        # SafetySamplingTask() (main.cpp) and HARDWARE_FINAL_SPEC.md D11':
        # suppression exists
        # only to stop a starting surge reading as an arc fault on the dP/dt
        # channel. Gating overcurrent on is_normal_inrush made the twin *weaker*
        # than the firmware — a 140% overload arriving while the baseline was
        # still cold (e.g. 280W on a 200W line at boot) left the relay closed in
        # simulation while real hardware opens it on the first sample.
        # The relay opens here (immediate, core 0); the anti-thrashing lockout
        # is set by the core-1 tick consuming shared_overcurrent_latch,
        # mirroring the core-0 overcurrent cutoff in SafetySamplingTask() +
        # the core-1 overcurrent lockout block in loop() (main.cpp).
        critical_watts = self.rated_watts * 1.25
        if power_w > critical_watts:
            self.set_relay(False)
            self.shared_overcurrent_latch = True
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
        """Simulates Core 1 callback() receiving an MQTT relay command.

        Mirrors the callback() MQTT command handler in main.cpp exactly:
          - any received message refreshes the server heartbeat (lastServerHB);
          - payloads over MAX_MQTT_PAYLOAD (256) bytes are dropped;
          - the payload is compared with == against "ON"/"OFF"/"WARNING" —
            case-sensitive, no strip()/upper() normalization;
          - "WARNING" is log-only;
          - the lockout is NOT expired here; only the core-1 loop tick does
            that (the SAFETY_LOCKOUT_MS expiry check in loop()).
        """
        # Heartbeat refresh (lastServerHB in callback())
        self.last_server_hb = time.time()

        # Payload size cap (MAX_MQTT_PAYLOAD in callback())
        if len(command) > 256:
            logger.warning(
                f"[{self.device_id}] [MQTT] Payload too large "
                f"({len(command)} bytes), dropping."
            )
            return

        if command == "ON":
            if not self.relay_locked:
                self.set_relay(True)
                if self.mqtt_publish:
                    await self.mqtt_publish(self.topic_ack, "ON_CONFIRMED")
                logger.info(f"[{self.device_id}] Relay turned ON (ACK: ON_CONFIRMED)")
            else:
                if self.mqtt_publish:
                    await self.mqtt_publish(self.topic_ack, "LOCKOUT_NACK")
                logger.warning(f"[{self.device_id}] ON rejected: Relay locked out (NACK sent)")

        elif command == "OFF":
            self.set_relay(False)
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_ack, "OFF_CONFIRMED")
            logger.info(f"[{self.device_id}] Relay turned OFF (ACK: OFF_CONFIRMED)")

        elif command == "WARNING":
            # Log-only branch (WARNING in callback())
            logger.info(f"[{self.device_id}] [SAFETY] Warning received from server")

    async def announce_offline(self):
        """Publish the OFFLINE status (mirrors the MQTT Last Will & Testament
        retained message configured by the client.connect() LWT in
        reconnectMQTT() in main.cpp)."""
        if self.mqtt_publish:
            await self.mqtt_publish(self.topic_status, "OFFLINE")
        logger.info(f"[{self.device_id}] Announced OFFLINE (LWT)")

    async def core1_telemetry_tick(self, force_publish: bool = False):
        """Simulates Core 1 1Hz power broadcast and 10s diagnostic broadcast."""
        now = time.time()
        power_w = self.shared_power_watts

        # ONLINE lifecycle — announced once on the first tick, mirroring the
        # retained publish on MQTT connect (reconnectMQTT() in main.cpp).
        # announce_offline() is the LWT counterpart. The connect also ARMS
        # the server-heartbeat watchdog (main.cpp sets lastServerHB = millis()
        # inside client.connect()), so a connected-but-commandless node
        # timeout-fires exactly like the firmware does.
        if not self._announced_online:
            self._announced_online = True
            self.last_server_hb = now
            # Republish gate, mirroring the firmware's static lastTimeoutLog
            # (initialized 0): the FIRST timeout fires at t+30 after connect.
            self._last_server_timeout_log = 0.0
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_status, "ONLINE")

        # 5-Minute Anti-Thrashing Lockout — evaluated on EVERY tick,
        # mirroring the SAFETY_LOCKOUT_MS expiry check in the core-1 loop
        # (the command handler does not expire it).
        if self.relay_locked and (now - self.lock_start_time) > self.safety_lockout_seconds:
            self.relay_locked = False
            logger.info(f"[{self.device_id}] 5-minute safety lockout expired. Relay unlocked.")

        # Core-0 arc-fault latch -> lockout + best-effort alert publish
        # (mirrors the core-1 arc-fault acknowledgment block in loop())
        if self.shared_arc_fault:
            alert_msg = f"EDGE_ARC_FAULT:dP/dt={self.shared_arc_fault_roc:.0f}W/s"
            self.shared_arc_fault = False
            self.relay_locked = True
            self.lock_start_time = now
            logger.warning(
                f"[{self.device_id}] [SAFETY] Edge arc-fault acknowledged -> relay LOCKED."
            )
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_status, alert_msg)

        # Core-0 overcurrent latch -> lockout + alert publish. main.cpp has no
        # latch here — its core-1 loop is a LEVEL-triggered check on
        # sharedPowerWatts (powerWatts > criticalWatts && !relayLocked). The
        # twin's edge-triggered flag is a behaviorally equivalent modelling
        # construct: the relay-open zeroes the read either way, and the flag
        # guarantees the lock is taken exactly once per trip.
        if self.shared_overcurrent_latch:
            self.shared_overcurrent_latch = False
            self.relay_locked = True
            self.lock_start_time = now
            critical_watts = self.rated_watts * 1.25
            logger.warning(
                f"[{self.device_id}] [SAFETY] OVERCURRENT! "
                f"{power_w:.1f}W > {critical_watts:.1f}W. Relay LOCKED."
            )
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_status, f"OVERCURRENT:{power_w:.1f}")

        # Server Heartbeat Watchdog (mirrors the lastServerHB / SERVER_TIMEOUT
        # watchdog block in loop() in main.cpp). Arming matches the firmware:
        # lastServerHB is set at MQTT CONNECT (the ONLINE announcement above),
        # so a connected-but-commandless node fires at t+30 s. Repeat cadence
        # matches the firmware's lastTimeoutLog gate: republished every 30 s
        # while starved (not once per episode), and a command refreshes
        # last_server_hb (the callback sets lastServerHB in main.cpp).
        if (self.last_server_hb is not None
                and (now - self.last_server_hb) > 30.0
                and (now - getattr(self, "_last_server_timeout_log", 0.0)) > 30.0):
            self._last_server_timeout_log = now
            logger.warning(f"[{self.device_id}] [WATCHDOG] No server heartbeat for 30s")
            if self.mqtt_publish:
                await self.mqtt_publish(self.topic_status, "SERVER_TIMEOUT")

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
