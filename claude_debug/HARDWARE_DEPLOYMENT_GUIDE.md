# Real-World Hardware Deployment & Electrical Safety Guide

> **Project:** Smart Home Energy Monitoring & Edge Safety Platform (EMS)  
> **Target Hardware:** ESP32 DevKit V1 + PZEM-004T v3.0 + 30A SLA Relay Module + HLK-5M05  
> **Status:** Production Hardware Audit Completed

---

## 1. Physical Architecture & Schematic Overview

```
[ 230V AC Mains In (L) ] ──[ 0.5A Fuse ]──┬──[ HLK-5M05 AC-L ]
                                          ├──[ 10D471K MOV ]──[ Mains (N) ]
                                          └──[ Relay COM ] ──┬──[ Relay NO ]──> [ Appliance Load ]
                                                             │
                                                    [ RC Snubber: 100Ω 2W ]
                                                             │
                                                    [ 0.1µF 275VAC X2 Cap ]
                                                             │
                                                             └── (across COM & NO)

[ HLK-5M05 DC Out +5V ] ──┬──[ 1000µF 16V Low-ESR Cap ]──┬──> [ ESP32 VIN (5V) ]
                          └──[ 100nF Ceramic Cap      ]──┼──> [ Relay VCC (+5V) ]
                                                         └──> [ PZEM-004T VCC (+5V) ]

[ Common Ground (GND) ] ──────────────────────────────────────> [ ESP32 GND / Relay GND / PZEM GND ]

[ ESP32 GPIO 18 (3.3V) ] ──[ 1kΩ ]──> [ Gate: 2N7000 MOSFET ]
                                      [ Drain: 2N7000        ] ────> [ Relay Module "IN" Pin ]
                                      [ Source: 2N7000       ] ────> [ Common Ground ]

[ ESP32 GPIO 16 (RX2)  ] <───────────────────────────────────────── [ PZEM-004T TX ]
[ ESP32 GPIO 17 (TX2)  ] ─────────────────────────────────────────> [ PZEM-004T RX ]

[ 100A CT Clamp ] ────────────────────────────────────────────────> [ PZEM-004T CT Input Terminals ]
(Clamped ONLY on Live conductor)
```

---

## 2. The 6 Critical Real-World Electrical Hazards & Solutions

### Hazard 1: 3.3V GPIO to 5V Active-LOW Optocoupler Mismatch
* **Symptom:** Relay fails to turn OFF or chatters when GPIO 18 is driven HIGH ($3.3\text{V}$).
* **Root Cause:** Standard optoisolated relay modules tie the optocoupler anode to $+5\text{V}$ through an onboard LED ($V_f \approx 1.2\text{V}$). Driving $3.3\text{V}$ leaves $\Delta V = 5.0\text{V} - 3.3\text{V} = 1.7\text{V} > 1.2\text{V}$, keeping the internal phototransistor conducting.
* **Solution:** Interpose a 2N7000 or BSS138 N-channel MOSFET as an open-drain inverter. Driving GPIO 18 HIGH pulls the Drain to GND (Relay ON); driving GPIO 18 LOW leaves the Drain floating (Relay OFF).

---

### Hazard 2: Relay Contact Arcing on Inductive Loads
* **Symptom:** Relay contacts weld closed after days of switching compressors, microwave transformers, or vacuum motors.
* **Root Cause:** Inductive loads store energy in magnetic fields ($E = \frac{1}{2} L I^2$). Opening contacts induces massive back-EMF ($V = -L \frac{di}{dt}$) exceeding 1,500V, causing an intense electrical arc that melts contact metallurgy.
* **Solution:** Connect an **RC Snubber** ($100\Omega \text{ 2W flameproof resistor} + 0.1\mu\text{F 275VAC Class-X2 safety capacitor}$) in parallel across the Relay COM and NO screw terminals.

---

### Hazard 3: HLK-PM01 Power Sags & WiFi Brownout Bootloops
* **Symptom:** ESP32 enters an infinite bootloop when attempting to connect to WiFi while the relay is energized.
* **Root Cause:** HLK-PM01 is rated for only 600mA (3W). An energized 30A relay coil draws ~180mA; ESP32 WiFi calibration bursts pull up to 500mA ($180 + 500 = 680\text{mA} > 600\text{mA}$). The 5V rail collapses below 4.4V, causing the 3.3V LDO to drop out.
* **Solution:**
  1. Upgrade to the **HLK-5M05 (5V 1A / 5W)**.
  2. Place a **$1000\mu\text{F } 16\text{V}$ Low-ESR electrolytic capacitor** directly across 5V and GND.

---

### Hazard 4: PZEM-004T Metering Refresh Latency
* **Symptom:** High-frequency $100\text{ms}$ polling yields identical repeated power readings for 4–8 cycles, then a step jump.
* **Root Cause:** The internal metering IC (SD3004 / RN8209) calculates true RMS voltage and current over 25–50 AC line cycles, updating registers at $1\text{--}2\text{ Hz}$ ($500\text{--}1000\text{ms}$).
* **Solution:** Understand that the edge $dP/dt$ rate-of-change proxy operates over a physical ~500ms quantization window. It provides excellent overcurrent and thermal runaway disconnects, but cannot substitute for dedicated microsecond hardware spark gap detectors.

---

### Hazard 5: PCB Trace Sizing & High-Voltage Isolation
* **Symptom:** PCB traces carrying 15A–30A overheat and delaminate; mains voltage arcs over to the 3.3V digital plane.
* **Root Cause:** Standard 1oz copper (35µm) requires a trace width $>12\text{mm}$ to carry 20A without exceeding a $30^\circ\text{C}$ temperature rise. Creepage distances $<6.3\text{mm}$ fail IEC 62368 safety standards.
* **Solution:**
  * Specify **2oz copper (70µm)** with exposed soldermask heavily reinforced with thick solder bridging on AC load traces.
  * Mill **air isolation slots** between AC mains pins and low-voltage DC planes.

---

### Hazard 6: Inverter-Driven Appliance Dynamics
* **Symptom:** Savitzky-Golay transient detector misses activations of modern Inverter ACs or Inverter Fridges.
* **Root Cause:** Inverter compressors ramp wattage smoothly (e.g., $100\text{W} \rightarrow 700\text{W}$ over 45 seconds) without step discontinuities.
* **Solution:** Rely on the `SoftAnomalyWatchdog` and baseline aggregate drift models for continuous variable loads; use transient edge classification for discrete two-state appliances (kettles, geysers, toasters, microwave ovens).

---

## 3. Bill of Materials (BOM)

| Item | Description | Exact Part Number | Qty |
| :---: | :--- | :--- | :---: |
| 1 | ESP32 DevKit V1 Development Board | ESP32-WROOM-32D | 1 |
| 2 | AC Energy Metering Module (UART) | PZEM-004T v3.0 | 1 |
| 3 | Split-Core Current Transformer | 100A / 50mA Matched CT | 1 |
| 4 | 30A Power Relay Module (5V Coil) | SLA-05VDC-SL-C | 1 |
| 5 | Isolated AC-DC Power Supply Module | Hi-Link HLK-5M05 (5V 1A) | 1 |
| 6 | Bulk Decoupling Capacitor | $1000\mu\text{F } 16\text{V}$ Low-ESR Electrolytic | 1 |
| 7 | High-Frequency Decoupling Capacitor | $100\text{nF } 50\text{V}$ Ceramic (X7R) | 1 |
| 8 | MOSFET Level Inverter | 2N7000 N-Channel MOSFET (TO-92) | 1 |
| 9 | Gate Resistor | $1\text{k}\Omega \text{ } 0.25\text{W}$ Metal Film | 1 |
| 10 | RC Snubber Resistor | $100\Omega \text{ } 2\text{W}$ Flameproof Metal Oxide | 1 |
| 11 | RC Snubber Capacitor | $0.1\mu\text{F } 275\text{VAC}$ Class-X2 Polypropylene | 1 |
| 12 | Mains Fuse | 0.5A 250V Slow-Blow Glass/Ceramic | 1 |
| 13 | Metal Oxide Varistor (MOV) | 10D471K (275VAC Clamp) | 1 |
| 14 | Enclosure | DIN Rail ABS V0 Flame Retardant | 1 |
