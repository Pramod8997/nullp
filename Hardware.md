# Hardware Pinout & Wiring Specification

* **PZEM-004T Metering UART:**
  * ESP32 GPIO 16 (RX2)  <──  PZEM-004T TX (Verify 3.3V logic level)
  * ESP32 GPIO 17 (TX2)  ──>  PZEM-004T RX
* **30A Relay Actuation:**
  * ESP32 GPIO 18 (3.3V) ──[ 1kΩ ]──> Gate: 2N7000 MOSFET
  * Drain: 2N7000 ──> 30A Relay Module IN (Active-LOW, 5V coil)
  * Source: 2N7000 ──> Shared Common Ground
  * RC Snubber: 100Ω 2W + 0.1µF 275VAC Class-X2 across Relay COM & NO
* **Power Supply (5V Rail):**
  * Hi-Link HLK-5M05 (5V 1A / 5W) with 0.5A slow-blow fuse on AC-L
  * 1000µF 16V Low-ESR bulk electrolytic + 100nF ceramic decoupling cap
  * ESP32 VIN (5V), Relay VCC (+5V), PZEM VCC (+5V)
* **AC Mains & Protective Earth (PE):**
  * Mains Live (L) ──> 16A MCB / Load Fuse ──> Relay COM ──[ Relay NO ]──> Load Live
  * Mains Neutral (N) ──> PZEM Terminal N & HLK-5M05 AC-N & Load Neutral
  * Protective Earth (PE) ──> Direct DIN rail enclosure earth block ──> Load PE (Unswitched)
  * CT Clamp (100A / 50mA or 10A Shunt) ──> Clamped around LIVE wire only
