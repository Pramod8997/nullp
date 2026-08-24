# Deep Technical Review & Architectural Design Document

> **System:** Smart Home Energy Monitoring & Disaggregation Platform (EMS)  
> **Prepared for:** Claude (Opus 4.5 / 5 / Sonnet) & Systems Architecture Teams  
> **Status:** Production-Verified

---

## 1. End-to-End System Topology

```mermaid
graph TD
    subgraph Edge_Node [Physical / Simulated ESP32 Node]
        Mains["230V AC Mains (50Hz)"] --> CT["100A Split-Core CT Clamp"]
        CT --> PZEM["PZEM-004T v3.0 (Modbus RTU)"]
        PZEM -->|UART Serial2 9600 baud| ESP32["ESP32 DevKit V1 (Xtensa LX6)"]
        
        subgraph FreeRTOS [Dual-Core FreeRTOS Partitioning]
            Core0["Core 0: SafetySamplingTask (100ms)"]
            Core1["Core 1: Arduino Loop + MQTT Task"]
            Spinlock["portMUX_TYPE sharedMux Spinlock"]
            
            Core0 <-->|Atomic Float Data| Spinlock
            Spinlock <-->|Atomic Float Data| Core1
        end
        
        Core0 -->|Overcurrent / Arc Fault| RelayDriver["2N7000 MOSFET Inverter"]
        RelayDriver --> Relay["30A Active-LOW Relay (GPIO 18)"]
        Relay --> Load["Appliance Load"]
    end

    subgraph Transport_Layer [Network & Protocol Bus]
        Core1 -->|WiFi 802.11 b/g/n| Broker["Mosquitto MQTT Broker (:1883)"]
        Broker -->|QoS 0 Telemetry / QoS 1 Commands| Pipeline["Python EMS Pipeline"]
    end

    subgraph Pipeline_Processing [Backend Analytics & ML]
        Pipeline --> SafetyCheck["FleetDiagnosticsMonitor"]
        Pipeline --> SGFilter["Savitzky-Golay Transient Detector"]
        SGFilter --> OverlapSub["OverlapAwareNILMDetector"]
        OverlapSub --> ProtoNetEmbed["ProtoNet 1D-CNN Metric Learner"]
        ProtoNetEmbed --> TempScaler["Temperature Scaler (T >= 0.05)"]
        TempScaler --> Gate["Confidence Gate (Threshold >= 0.90)"]
        Gate --> Watchdog["SoftAnomalyWatchdog (Z-Score >= 3.0)"]
        Gate --> RLAgent["PPO / DQN Demand-Response Agent"]
        RLAgent -->|home/plug/{id}/command| Broker
    end
```

---

## 2. Dual-Core FreeRTOS State Machine (Firmware Layer)

### 2.1 Core 0: `SafetySamplingTask` (Safety Critical)
* **Execution Interval:** Every $100\text{ms}$ ($\pm 5\text{ms}$). Priority: 2 (High).
* **Execution Path:**
  1. Read power registers from PZEM-004T: `P_curr = pzem.power()`.
  2. If `isnan(P_curr)` or `isinf(P_curr)`: Skip cycle to avoid state poisoning.
  3. Evaluate rate-of-change:
     $$\text{RoC} = \frac{P_{\text{curr}} - P_{\text{last}}}{\Delta t}$$
  4. Evaluate sliding baseline inrush:
     $$\text{baseline\_avg} = \frac{1}{5} \sum_{i=1}^{5} \text{RingBuffer}[i]$$
     $$\text{is\_inrush} = (\text{baseline\_avg} < 50.0\text{W}) \land (P_{\text{last}} < \text{baseline\_avg} + 100.0\text{W})$$
  5. **Safety Cutoff Decision:**
     $$\text{Trip} = (\text{RoC} > 1000.0\text{W/s} \land \neg\text{is\_inrush}) \lor (P_{\text{curr}} > 1.25 \times P_{\text{rated}} \land \neg\text{is\_inrush})$$
  6. If $\text{Trip} == \text{True}$:
     * Set `GPIO 18 = HIGH` (Relay OFF).
     * Set `relayLocked = True`.
     * Record `lockStartMs = millis()`.
  7. Acquire `taskENTER_CRITICAL(&sharedMux)`, update `sharedPowerWatts = P_curr`, and call `taskEXIT_CRITICAL(&sharedMux)`.

### 2.2 Core 1: `Arduino Loop` (Network & Telemetry)
* **Execution Interval:** Non-blocking async loop. Priority: 1 (Standard).
* **Execution Path:**
  1. Maintain WiFi and MQTT connection states with exponential backoff (1s $\rightarrow$ 30s).
  2. Call `client.loop()` to process incoming command topics.
  3. Every $1.0\text{s}$: Publish `sharedPowerWatts` to `home/sensor/{id}/power`.
  4. Every $10.0\text{s}$: Publish diagnostic JSON `{v, i, w, pf}` to `home/sensor/{id}/telemetry`.
  5. Handle incoming `ON` commands:
     * If `relayLocked == True` and $(t_{\text{now}} - t_{\text{lock}}) < 300,000\text{ms}$: Publish `LOCKOUT_NACK` and drop command.
     * Else if $(t_{\text{now}} - t_{\text{lock}}) \ge 300,000\text{ms}$: Clear `relayLocked = False`, set `GPIO 18 = LOW` (Relay ON), publish `ON_CONFIRMED`.

---

## 3. Mathematical Formulations & ML Pipeline

### 3.1 Savitzky-Golay Filtering & Derivative
The transient detector fits local polynomials of degree $p=2$ over window $w=7$ to compute smooth numerical derivatives without amplifying high-frequency sensor noise:

$$P_{\text{smooth}}(n) = \sum_{k=-m}^{m} c_k P(n+k), \quad m = \frac{w-1}{2}$$

The rate of change is evaluated over consecutive filtered samples:

$$\Delta P(n) = P_{\text{smooth}}(n) - P_{\text{smooth}}(n-1)$$

A transient onset is triggered when $|\Delta P(n)| \ge 20.0\text{W}$, initiating a 128-sample embedding window and activating a 5-second cooldown timer.

---

### 3.2 Overlap-Aware Power Subtraction
When a new transient $\Delta P_{\text{new}}$ is detected while existing registered appliances $A_1, \dots, A_k$ are active with known steady-state powers $P(A_i)$:

$$P_{\text{isolated}}(t) = \max\left(0.0, P_{\text{aggregate}}(t) - \sum_{i=1}^{k} P(A_i)\right)$$

The isolated transient waveform is zero-padded to length 128 and fed into the ProtoNet embedding network.

---

### 3.3 ProtoNet Metric Learning & Temperature Calibration
1. **ProtoNet 1D-CNN Encoder:** Encodes transient waveform $x \in \mathbb{R}^{128}$ into embedding space $f_\phi(x) \in \mathbb{R}^{64}$.
2. **Euclidean Distance to Prototypes:**
   $$d(f_\phi(x), c_k) = \| f_\phi(x) - c_k \|_2^2$$
3. **Logits:** $z_k = -d(f_\phi(x), c_k)$.
4. **Temperature-Scaled Probabilities:**
   $$p_k = \frac{\exp(z_k / T)}{\sum_{j} \exp(z_j / T)}, \quad \text{where } T = \max(T_{\text{learned}}, 0.05)$$
5. **Confidence Gate:**
   $$\text{Action} = \begin{cases} \text{PASS\_RL}, & \text{if } \max_k(p_k) \ge 0.90 \\ \text{SKIP\_RL}, & \text{if } \max_k(p_k) < 0.90 \end{cases}$$

---

### 3.4 Rolling Z-Score Anomaly Watchdog
Tracks rolling mean $\mu$ and standard deviation $\sigma$ over window $N=30$:

$$\mu_t = \frac{1}{N} \sum_{i=0}^{N-1} x_{t-i}, \quad \sigma_t = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} (x_{t-i} - \mu_t)^2}$$

$$\sigma_{\text{safe}} = \max(\sigma_t, 10^{-6})$$

$$Z_t = \frac{|x_t - \mu_t|}{\sigma_{\text{safe}}}$$

If $Z_t \ge 3.0$ and $|x_t - \mu_t| > 50\text{W}$, an anomaly event is dispatched. If $|x_t - \mu_t| > 100\text{W}$ and $(x_t < 10\text{W} \lor \mu_t < 10\text{W})$, a discrete ON/OFF baseline reset is executed.

---

## 4. MQTT Topic Schema & Payload Formats

| Topic String | Direction | QoS | Retain | Payload Format | Description |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `home/sensor/{id}/power` | ESP32 $\rightarrow$ Server | 0 | False | Plain float string (`"204.5"`) | 1Hz continuous power readings |
| `home/sensor/{id}/telemetry` | ESP32 $\rightarrow$ Server | 0 | False | JSON (`{"v":230,"i":1,"w":200,"pf":1}`) | 10s full diagnostics packet |
| `home/sensor/{id}/status` | ESP32 $\rightarrow$ Server | 1 | True | Plain string (`"EDGE_ARC_FAULT"`) | Emergency safety alerts |
| `home/plug/{id}/command` | Server $\rightarrow$ ESP32 | 1 | False | Plain string (`"ON"`, `"OFF"`, `"WARNING"`) | Remote relay actuation commands |
| `home/plug/{id}/ack` | ESP32 $\rightarrow$ Server | 1 | False | Plain string (`"ON_CONFIRMED"`, `"LOCKOUT_NACK"`) | Actuation acknowledgments |
| `home/ui/events` | Server $\rightarrow$ Frontend | 0 | False | JSON Event Object | Real-time WebSocket UI stream |
