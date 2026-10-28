# Current Low-Voltage Wiring — Actual 100A CT Build (PZEM-004T v4.0)

**Board:** 38-pin ESP32-WROOM-32 DevKit (or 30-pin DevKit V1). **Mains disconnected.** Dupont wires
are permitted only on this low-voltage side.

The meter is the **PZEM-004T v4.0 (100A) external-CT variant**.
Its external CT measures one live conductor; its high-voltage screw terminals are voltage sense and CT secondary input.

---

## 1. Complete Pin Map

| ESP32 Pin | Connect To | Target Component & Function | Required Condition / Logic |
|---|---|---|---|
| **GPIO 17 / TX2** | PZEM TTL **RX** | PZEM-004T v4.0 UART2 Receive | Direct 3.3V TX from ESP32 |
| **GPIO 16 / RX2** | PZEM TTL **TX** | PZEM-004T v4.0 UART2 Transmit | **1kΩ / 2kΩ divider** (or 5V-to-3.3V level shifter) |
| **GPIO 26** | Relay **IN1** | 4-Ch Relay Module Channel 1 | Active-LOW (`LOW` = ON, `HIGH` = OFF) |
| **GPIO 27** | Relay **IN2** | 4-Ch Relay Module Channel 2 | Active-LOW (`LOW` = ON, `HIGH` = OFF) |
| **GPIO 14** | Relay **IN3** | 4-Ch Relay Module Channel 3 | Active-LOW (`LOW` = ON, `HIGH` = OFF) |
| **GPIO 32** | Relay **IN4** | 4-Ch Relay Module Channel 4 | Active-LOW (Remapped off strapping pin `GPIO 12`) |
| **GND** | Relay GND & PZEM GND | Common Low-Voltage DC Ground | Single unified ground reference |
| **VIN / 5V** | Relay VCC & PZEM 5V | Isolated 5V Rail (≥ 2A supply) | Dedicated BIS-marked 5V supply |

---

## 2. PZEM-004T v4.0 Wiring

### Low-Voltage 4-Pin TTL Header
- **5V**: Connects to external **5V Rail** (`ESP32 VIN`)
- **RX**: Connects to **ESP32 GPIO 17 (UART2 TX)**
- **TX**: Connects through **1kΩ / 2kΩ voltage divider** to **ESP32 GPIO 16 (UART2 RX)**
- **GND**: Connects to **Common DC GND**

```text
PZEM TX ── 1 kΩ ──┬── ESP32 GPIO 16 (UART2 RX)
                  │
             2 kΩ to GND
```

### High-Voltage 4-Pin Screw Terminals
- **Terminal 1 (`L`)**: 230V AC Live (via 1A fast-blow inline ceramic fuse)
- **Terminal 2 (`N`)**: 230V AC Neutral
- **Terminal 3 (`CT+`)**: 100A Split-Core CT Red lead wire
- **Terminal 4 (`CT-`)**: 100A Split-Core CT Black lead wire

> **CRITICAL SAFETY NOTE**: Tighten screw terminals 3 & 4 securely **BEFORE** clamping the split-core CT around the single insulated Live (`L`) conductor. Never leave CT secondary terminals open while energized. Note that the PZEM-004T AC side must be powered by 230V AC for the onboard metering ASIC to respond to UART polls.

---

## 3. 4-Channel Opto-Isolated Relay Module

- **VCC**: Connects to external **5V Rail** (≥ 2A external supply recommended; or `JD-VCC` = 5V and `VCC` = 3.3V with jumper removed for full optical isolation).
- **GND**: Connects to **Common DC GND**.
- **IN1**: `GPIO 26`
- **IN2**: `GPIO 27`
- **IN3**: `GPIO 14`
- **IN4**: `GPIO 32` (⚠️ **CRITICAL FIX**: Remapped from `GPIO 12` / `MTDI` strapping pin. `GPIO 12` pulled HIGH at boot switches `VDD_SDIO` to 1.8V and bricks flash boot).
- **Relay Logic Polarity**: **ACTIVE-LOW** (`RELAY_ON = LOW`, `RELAY_OFF = HIGH`). Firmware pre-loads pins to `HIGH` before configuring output mode to eliminate boot chatter. Autonomous safety trips immediately drive all 4 channels `HIGH`.

---

## 4. Power & USB Separation

- **Runtime**: BIS-marked isolated 5V 2A USB charger → USB-A-female-to-screw-terminal breakout → 5V / GND rail.
- **Flashing**: Flash using laptop USB with mains and high-voltage AC disconnected. For runtime operation, disconnect laptop USB.
- **Mains Safety**: Never use Dupont jumpers, breadboards, or thin wires for 230V AC mains conductors. Use 1.0 mm² insulated mains cable.
