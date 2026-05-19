# Phase 2 Execution Report: Hardware Deployment & Physical Pipeline Integration

## 1. Objective
Transition the Smart Home Energy Management System from a Python-based synthetic UK-DALE simulation to a physical, edge-enabled cyber-physical system. Phase 2 validates the Q-learning agent and few-shot classification against live electrical transients using ESP32 edge nodes.

## 2. Hardware Layer Execution (Tier-0 Safety)
The most critical aspect of Phase 2 is isolating the Tier-0 safety logic from the machine learning orchestration to guarantee hardware-level overcurrent protection.

* **BOM Assembly (Per Node):**
    * ESP32 DevKit V1
    * SCT-013-030 Non-invasive CT Sensor (30A)
    * 5V, 10A Relay Module
    * 33Ω Burden Resistor & 3.5mm Jack
* **Firmware Implementation (`main.cpp`):**
    * The ESP32 must sample the ADC1_CH6 pin to calculate RMS current.
    * **Hardcoded Safety:** The microcontroller must hold a local `critical_pct` threshold (125% of rated watts). If breached, the ESP32 must toggle the GPIO 5 relay pin to `LOW` in under 50ms, *independently* of the MQTT connection state.
    * **Data Publishing:** Publish $P = V \times I \times PF$ to `home/sensor/{device_id}/power` at a strict 1Hz interval.

## 3. Data Integration & Environment Setup
For stable edge node flashing and NILMTK dataset processing, ensure your Pop!_OS development environment is properly configured with the latest `esptool.py` and USB-to-UART bridge drivers (CP210x).

* **Deprecating the Simulator:** Remove `backend/scripts/simulate_esp32.py` from the execution stack.
* **NILMTK Integration:** Connect the physical pipeline to the live MQTT broker. For benchmarking against real-world data, utilize the NILMTK toolkit to stream real UK-DALE and REDD datasets through the local Mosquitto broker as if they were live physical sensors.

## 4. Pipeline Remediation (Pre-Deployment)
Before physical relay actuation is permitted, the following Phase 1 pipeline gaps must be patched:
1.  **Activate Preprocessing:** Wrap the MQTT payload ingestion in `run_pipeline.py` with `self.nilm_detector.process()`. The CNN must only see SG-filtered transient signatures to improve the Kettle/Microwave classification boundaries.
2.  **Enable Soft Anomaly Control:** Update the RL agent's state space to consume the `SOFT_ANOMALY` events, allowing the Q-table to learn deferral policies for appliances exhibiting early-stage mechanical degradation.
3.  **Bridge the Unknown Flow:** Modify the pipeline so that temporally stable unknown devices (assigned a `Temp ID`) are forwarded to the Digital Twin as baseline loads, allowing the RL agent to schedule known devices around the new, unknown energy footprint.

## 5. Team Coordination & Milestones
To accelerate deployment before the final review with Mr. Ramesh Sunder Nayak, parallelize the execution tasks:
* **Hardware & Soldering:** Coordinate with Aadi Gupta and Prajwal K A to assemble and calibrate the 4 physical ESP32 CT-clamp nodes.
* **Firmware & MQTT Bridge:** Work with Abhishek Raj P to ensure the `main.cpp` publishes standard JSON payloads and correctly subscribes to the `home/plug/+/command` topics for the RL agent's actuation commands.

## 6. Evaluation Metrics for Phase 2
* **Latency Testing:** Verify that the end-to-end classification pipeline (MQTT publish $\rightarrow$ CNN $\rightarrow$ Confidence Gate $\rightarrow$ Q-Learning $\rightarrow$ MQTT command) executes in $< 50$ms.
* **Thermal Comfort Validation:** Cross-reference the Digital Twin's predicted PMV values against real-world ambient room temperature changes when physical HVAC loads are shedded. 
* **Energy Savings:** Quantify the reduction in idle power consumption against a standard Time-of-Use rule-based baseline.