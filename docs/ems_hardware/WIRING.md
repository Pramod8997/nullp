# Current Low-Voltage Wiring — Actual 100A CT Build (PZEM-004T v4.0)

**Board:** ESP32 DevKit V1 (30-pin). **Mains disconnected.** Dupont wires
are permitted only on this low-voltage side.

The meter is the **PZEM-004T v4.0 (100A) external-CT variant**.
Its external CT measures one live conductor; its high-voltage screw terminals are voltage sense and CT secondary input.

---

## 1. Complete Pin Map

| ESP32 Pin | Connect To | Target Component & Function | Required Condition / Logic |
|---|---|---|---|
| **GPIO 17 / TX2** | PZEM TTL **RX** | PZEM-004T v4.0 UART2 Receive | 9600 baud, Modbus-RTU |
| **GPIO 16 / RX2** | PZEM TTL **TX** | PZEM-004T v4.0 UART2 Transmit | Direct 3.3V logic |
| **GPIO 25** | Relay **VCC** | Relay Optocoupler LED VCC source | Driven HIGH as 3.3V source (a few mA) |
| **GPIO 26** | Relay **IN1** | Relay Channel 1 (Laptop Charger) | Active-LOW (`LOW` = ON, `HIGH` = OFF) |
| **GPIO 27** | Relay **IN2** | Relay Channel 2 (Phone Charger) | Active-LOW (`LOW` = ON, `HIGH` = OFF) |
| **GND** | Relay GND & PZEM GND | Common Low-Voltage DC Ground | Single unified ground reference |
| **3V3** | PZEM TTL **3V3 / VCC** | PZEM Low-voltage Logic Supply | 3.3V logic supply from ESP32 |
| **VIN / 5V** | Relay **JD-VCC** | Relay Coil Supply (5V) | Jumper removed between VCC & JD-VCC |

> **Strapping Pins & UART Reservations**: Do NOT use GPIO 12 (MTDI strapping pin). Do NOT use GPIO 16/17 for anything else.

---

## 2. PZEM-004T v4.0 Wiring

### Low-Voltage 4-Pin TTL Header
- **3V3 / VCC**: Connects to **ESP32 3V3 pin** (3.3V logic)
- **RX**: Connects to **ESP32 GPIO 17 (UART2 TX)**
- **TX**: Connects to **ESP32 GPIO 16 (UART2 RX)**
- **GND**: Connects to **Common DC GND**

### High-Voltage 4-Pin Screw Terminals
- **Terminal 1 (`L`)**: 230V AC Live (via 1A fast-blow inline ceramic fuse)
- **Terminal 2 (`N`)**: 230V AC Neutral
- **Terminal 3 (`CT+`)**: 100A Split-Core CT Red lead wire
- **Terminal 4 (`CT-`)**: 100A Split-Core CT Black lead wire

> **CRITICAL SAFETY NOTE**: Tighten screw terminals 3 & 4 securely **BEFORE** clamping the split-core CT around the single insulated Live (`L`) conductor. Never leave CT secondary terminals open while energized. Note that the PZEM-004T AC side must be powered by 230V AC for the onboard metering ASIC to respond to UART polls.

---

## 3. 2-Channel Opto-Isolated Relay Module

- **VCC**: Connects to **ESP32 GPIO 25** (used as 3.3V source driven HIGH in setup; feeds only opto LEDs, a few mA).
- **JD-VCC**: Connects to **External 5V / VIN Rail** (remove the jumper between VCC and JD-VCC).
- **GND**: Connects to **Common DC GND**.
- **IN1**: `GPIO 26` (Laptop charger)
- **IN2**: `GPIO 27` (Phone charger)
- **Relay Logic Polarity**: **ACTIVE-LOW** (`RELAY_ON = LOW`, `RELAY_OFF = HIGH`). Relays are held OFF through boot and reset:
  1. IN1 (GPIO26) and IN2 (GPIO27) are set to OUTPUT and driven HIGH (RELAY_OFF).
  2. THEN RELAY_VCC_PIN (GPIO25) is set to OUTPUT and driven HIGH.
  3. Delay ~100 ms, then init PZEM on Serial2.
- **Safe State**: Safe-state function `allRelaysOff()` writes HIGH to GPIO26/27 on any fatal error, watchdog trip, comms loss, or restart.

---

## 4. Power & USB Separation

- **Runtime**: BIS-marked isolated 5V 2A USB charger → USB-A-female-to-screw-terminal breakout → 5V / GND rail.
- **Flashing**: Flash using laptop USB with mains and high-voltage AC disconnected. For runtime operation, disconnect laptop USB.
- **Mains Safety**: Never use Dupont jumpers, breadboards, or thin wires for 230V AC mains conductors. Use 1.0 mm² insulated mains cable.
