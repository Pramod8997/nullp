# Real-World Physical & Electrical Testing Protocol

> **System:** Smart Home Energy Monitoring & Disaggregation Platform (EMS)  
> **Target Environment:** Physical Electronics Benchtop & Mains Commissioning (230V AC / 50Hz)  
> **Status:** Production Hardware Testing Standard

---

## 1. Physical Benchtop Instrumentation Required

To execute physical validation before mains installation, prepare the following bench equipment:

1. **Variac (Variable AC Autotransformer):** 0–270V AC, 1kVA (for testing brownouts, grid sags, and swells).
2. **Current-Limited DC Bench Power Supply:** 0–30V, 0–5A (for testing 5V rail stability and ESP32 brownout margins).
3. **Digital Storage Oscilloscope (DSO):** Minimum 2 channels, 100MHz (for measuring 5V ripple during WiFi TX bursts and relay contact flyback arcs).
4. **True-RMS Digital Multimeter:** AC Voltage, AC Current, Resistance, Diode check.
5. **High-Voltage Differential Probe:** 1000V rated (for measuring inductive kick across relay contacts).
6. **Thermal Camera or K-Type Thermocouple:** For measuring $I^2R$ PCB trace heating at 16A/25A continuous.
7. **Test Loads:**
   * **Resistive:** 100W incandescent test bulb, 2000W electric kettle.
   * **High Inrush / Inductive:** 500W halogen work lamp (10x cold inrush), 1/4 HP refrigerator compressor motor / drill.

---

## 2. The 8 Real-World Physical Test Procedures

### Test 1: 5V DC Rail Stability & WiFi TX Brownout Pulse Test
* **Objective:** Ensure the 5V DC rail does not collapse below 4.4V when the ESP32 performs high-power WiFi RF transmissions simultaneously with 30A relay coil pull.
* **Setup:** Connect Channel 1 of DSO (AC Coupling, 20mV/div) to the ESP32 `VIN` (5V) pin.
* **Procedure:**
  1. Trigger relay `ON` command over MQTT while streaming high-rate telemetry at 10Hz.
  2. Capture oscilloscope voltage droop during the WiFi beacon transmission.
* **Pass Criteria:** Voltage droop must be $< 250\text{mV}$; minimum rail voltage must remain $> 4.75\text{V}$ (preventing 3.3V LDO dropout).

---

### Test 2: Mains Voltage Sags & Swell Transients (Variac Test)
* **Objective:** Verify PZEM-004T metering and ESP32 power supply operation under severe grid voltage fluctuations.
* **Setup:** Power the system through a Variac; connect a 100W test bulb through the relay.
* **Procedure:**
  1. Sweep mains voltage from 230V down to **160V AC** (simulating peak evening grid brownout).
  2. Verify ESP32 stays online and PZEM-004T measures voltage accurately ($\pm 1\%$).
  3. Increase voltage to **270V AC** (simulating lightning surge / transformer tap change).
* **Pass Criteria:** No reboots, no false overcurrent cutoffs, and power factor calculation remains stable.

---

### Test 3: Inductive Load Arcing & Snubber Damping Test
* **Objective:** Verify the RC Snubber ($100\Omega \text{ 2W} + 0.1\mu\text{F X2}$) clamps inductive back-EMF during relay cutoff.
* **Setup:** Connect an inductive test motor to the relay. Attach High-Voltage differential probe across Relay COM and NO terminals.
* **Procedure:**
  1. Energize the motor.
  2. Issue an emergency `OFF` cutoff command via Core 0 safety trip.
  3. Capture peak back-EMF spike on the oscilloscope.
* **Pass Criteria:** Peak voltage spike must be clamped below $450\text{V}_{\text{peak}}$ (preventing air dielectric breakdown and contact welding).

---

### Test 4: Motor Cold Inrush vs. Arc-Fault Discrimination Test
* **Objective:** Verify that cold-start motor inrush does not trigger spurious edge arc-fault trips.
* **Setup:** Connect a 500W Halogen lamp (10x cold inrush) or Refrigerator compressor to the relay.
* **Procedure:**
  1. Power-on the load cold (0W $\rightarrow$ 1200W transient in $<20\text{ms}$).
  2. Observe Core 0 debug serial logs: `[CORE 0] INRUSH SUPPRESSION ACTIVE (baseline < 50W)`.
* **Pass Criteria:** Relay stays closed; inrush is tolerated.
* **Sub-test (Genuine Arc Fault):** While running steady at 150W, short a 1000W secondary load across the line; verify instantaneous trip within $<100\text{ms}$ with 300s lockout.

---

### Test 5: 30A Continuous Current Thermal Stress Test
* **Objective:** Verify PCB trace width and terminal block temperature under continuous high current (IPC-2221 compliance).
* **Setup:** Pass 16A continuous (for 2 hours) and 25A continuous (for 30 minutes) through the relay COM-NO PCB traces using a dummy load bank. Monitor with a thermal camera.
* **Pass Criteria:** Maximum trace temperature rise must not exceed $\Delta T < 30^\circ\text{C}$ above ambient room temperature ($<65^\circ\text{C}$ absolute).

---

### Test 6: CT Clamp Reversed Orientation & Saturation Test
* **Objective:** Verify the system safely handles reverse-clamped CTs and grid over-saturation.
* **Procedure:**
  1. Install CT clamp in reverse direction on the Live wire.
  2. Verify firmware and server sanitize negative power values (`abs()` or clamp to `0.0W`) without math exceptions.
  3. Inject 120A instantaneous current. Verify overcurrent trip triggers instantly.
* **Pass Criteria:** Zero division or NaN errors; instantaneous hardware lockout.

---

### Test 7: Total Harmonic Distortion (THD) & Electrical Noise Test
* **Objective:** Verify NILM transient detector does not trigger false positives when non-linear loads (LED dimmers, phase-angle SCR motor speed controllers) inject line harmonics.
* **Procedure:** Run an electric drill / blender on the same circuit branch. Stream 1Hz power data through `NILMTransientDetector`.
* **Pass Criteria:** Zero false transient events triggered during steady-state phase-angle chopped running.

---

### Test 8: Sudden Mains Power Cutoff during Flash Storage
* **Objective:** Verify non-volatile memory (NVS) integrity when 230V mains is killed abruptly during telemetry or database write operations.
* **Procedure:** Power cycle the 230V AC breaker randomly 50 times while the system is actively writing SQLite records and FreeRTOS state variables.
* **Pass Criteria:** On power restoration, SQLite database recovers with zero corruption (WAL mode verification); ESP32 boots cleanly into safe default state (Relay OFF).

---

## 3. Automated Real-World Stress Test CLI

To execute automated simulation models of these 8 physical scenarios:

```bash
source .venv/bin/activate
python scripts/real_world_physical_stress.py
python scripts/hil_hardware_test.py
```
