/**
 * Module 7: Production ESP32 Firmware — Dual-Core FreeRTOS (Hybrid Node)
 *
 * Edge-hybrid safety architecture for the Smart Home EMS.
 * ESP32 DevKit V1 (30-pin / 38-pin)
 * 2-Channel Active-LOW Opto-Isolated Relay Module:
 *   - Relay 1 (Laptop Charger, ~65W) = GPIO 26 (High Priority)
 *   - Relay 2 (Phone Charger,  ~25W) = GPIO 27 (Low Priority, Auto-Shed)
 *   - Relay Opto VCC                 = GPIO 25 (3.3V source driven HIGH)
 *   - Active-LOW: LOW = ON, HIGH = OFF
 *   - JD-VCC = VIN (5V)
 *
 * SIMULATED TELEMETRY (PZEM Disconnected):
 *   - Realistic mains AC: ~230V AC fluctuating (228.5V - 231.5V), ~50.0 Hz
 *   - Active Power: 0W (both OFF), ~65W (Relay 1), ~25W (Relay 2), ~90W (Both)
 *   - Current: Calculated dynamically as P / (V * PF)
 *   - Power Factor: 0.00 (OFF), ~0.95 (Relay 1), ~0.88 (Relay 2), ~0.92 (Both)
 *   - Energy (kWh): Real-time numeric integration
 *
 * LOAD SHEDDING PROTECTION (80W Ceiling):
 *   - If total power exceeds 80W (both relays ON ~90W), 3-second grace countdown triggers.
 *   - After 3 continuous seconds, Relay 2 (Phone Charger) is automatically shed.
 *   - Relay 1 (Laptop Charger) is kept safely ON.
 *
 * COMMUNICATIONS:
 *   - Dual-Mode WiFi: Connects to local WiFi or falls back to SoftAP "Smart_EMS_Demo"
 *   - Web Dashboard on Port 80: Modern dark-mode HTML/JS UI with live /status AJAX
 *   - MQTT: Subscribes to home/plug/{DEVICE_ID}/command (IN1_ON, IN1_OFF, IN2_ON, IN2_OFF, ON, OFF)
 *           Publishes to home/sensor/{DEVICE_ID}/power and telemetry
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include <esp_task_wdt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#if !__has_include("secrets.h")
#error "Missing firmware/esp32_node/include/secrets.h - copy secrets.h.example to secrets.h and fill in credentials."
#endif
#include "secrets.h"
#include "hardware_config.h"

// ═══════════════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════════════
const char* DEVICE_ID      = EMS_DEVICE_ID;
const char* ssid           = EMS_WIFI_SSID;
const char* password       = EMS_WIFI_PASSWORD;
const char* mqtt_server    = EMS_MQTT_SERVER;
const int   mqtt_port      = EMS_MQTT_PORT;
const char* mqtt_user      = EMS_MQTT_USER;
const char* mqtt_password  = EMS_MQTT_PASSWORD;

// Fallback SoftAP Hotspot
const char* AP_SSID        = "Smart_EMS_Demo";
const char* AP_PASSWORD    = "12345678";

// Hardware Pins
const int PIN_RELAY_1      = RELAY1_PIN;     // GPIO 26
const int PIN_RELAY_2      = RELAY2_PIN;     // GPIO 27
const int PIN_OPTO_VCC     = RELAY_VCC_PIN; // GPIO 25

// Relay logic levels (Active-LOW opto-isolated relay: LOW = ON / Closed, HIGH = OFF / Cutoff)
#define RELAY_ON  LOW
#define RELAY_OFF HIGH

// Overload Protection Thresholds
const float OVERLOAD_POWER_THRESHOLD = 80.0f; // Watts
const unsigned long OVERLOAD_TIME_MS = 3000;   // 3 Seconds

// ═══════════════════════════════════════════════════════
//  SHARED STATE (Core 0 <-> Core 1)
// ═══════════════════════════════════════════════════════
SemaphoreHandle_t stateMutex = nullptr;

inline void acquireStateLock() {
    if (stateMutex != nullptr) {
        xSemaphoreTake(stateMutex, portMAX_DELAY);
    }
}

inline void releaseStateLock() {
    if (stateMutex != nullptr) {
        xSemaphoreGive(stateMutex);
    }
}

volatile float sharedPowerWatts       = 0.0f;
volatile float sharedVoltage          = 230.0f;
volatile float sharedCurrent          = 0.0f;
volatile float sharedPf               = 0.00f;
volatile float sharedEnergy           = 0.0f;
volatile float sharedFrequency        = 50.0f;
volatile float sharedApparentPowerVa  = 0.0f;
volatile float sharedReactivePowerVar = 0.0f;

volatile bool  relay1Closed           = false;
volatile bool  relay2Closed           = false;
volatile bool  relayClosed            = false;
volatile bool  autoOverloadEnabled    = true;

volatile bool  overloadTimingActive   = false;
volatile unsigned long overloadStartMs = 0;
volatile float overloadCountdownSec   = 0.0f;
volatile unsigned int overloadTripCount = 0;
volatile bool  overloadShedEvent      = false;

// Command queues from Core 1 / MQTT to Core 0
volatile bool  pendingRelayOn         = false;
volatile bool  pendingRelayOff        = false;
volatile bool  pendingCh1On           = false;
volatile bool  pendingCh1Off          = false;
volatile bool  pendingCh2On           = false;
volatile bool  pendingCh2Off          = false;
volatile bool  pendingAck             = false;
volatile uint8_t pendingAckType       = 0; // 1=ON, 2=OFF, 4=CH1_OFF, 5=CH1_ON, 6=CH2_OFF, 7=CH2_ON

TaskHandle_t   safetyTaskHandle       = nullptr;
String         sharedAlertMessage     = "System Initialized. Circuits Nominal.";

// ═══════════════════════════════════════════════════════
//  CORE 1 STATE (Network, WebServer, MQTT)
// ═══════════════════════════════════════════════════════
WiFiClient espClient;
PubSubClient client(espClient);
WebServer server(80);

char activeDeviceId[32]    = "node_bench_agg";
unsigned long lastMsgMs    = 0;
unsigned long lastTelemetryMs = 0;
unsigned long lastServerHB = 0;
unsigned long lastReconnectAttempt = 0;

char topicPower[64];
char topicTelemetry[64];
char topicCommand[64];
char topicStatus[64];
char topicAck[64];

// ═══════════════════════════════════════════════════════
//  HARDWARE RELAY CONTROL
// ═══════════════════════════════════════════════════════
void applyRelayHardware(uint8_t channel, bool on) {
    uint8_t pinState = on ? RELAY_ON : RELAY_OFF;
    if (channel == 1) {
        digitalWrite(PIN_RELAY_1, pinState);
        relay1Closed = on;
    } else if (channel == 2) {
        digitalWrite(PIN_RELAY_2, pinState);
        relay2Closed = on;
    }
    relayClosed = (relay1Closed || relay2Closed);
    Serial.printf("[RELAY] Channel %d -> %s (GPIO %d: %s)\n", channel, on ? "ON (Current Flowing)" : "OFF (Cutoff)", channel == 1 ? PIN_RELAY_1 : PIN_RELAY_2, on ? "HIGH 3.3V" : "LOW 0V");
}

void allRelaysOff() {
    applyRelayHardware(1, false);
    applyRelayHardware(2, false);
    relayClosed = false;
}

// ═══════════════════════════════════════════════════════
//  EMBEDDED WEB DASHBOARD HTML (Stored in PROGMEM)
// ═══════════════════════════════════════════════════════
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Home EMS | ESP32 Node</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.25);
      --accent-green: #10b981;
      --accent-green-glow: rgba(16, 185, 129, 0.35);
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-red-glow: rgba(239, 68, 68, 0.45);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
    body {
      background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0b0f19 75%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 20px 14px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container { width: 100%; max-width: 900px; }
    header {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 18px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .header-titles h1 {
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
      background: linear-gradient(135deg, #fff 30%, #38bdf8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .header-titles p { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }
    .badge-bar { display: flex; flex-wrap: wrap; gap: 8px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      padding: 5px 10px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: #cbd5e1;
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
    .badge-dot.pulse { animation: pulseAnim 1.5s infinite; }
    @keyframes pulseAnim { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } }

    #alertBanner {
      display: none;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--accent-red);
      border-radius: 14px;
      padding: 14px 18px;
      margin-bottom: 18px;
      box-shadow: 0 0 20px var(--accent-red-glow);
    }
    #alertBanner.warning-state {
      background: rgba(245, 158, 11, 0.15);
      border-color: var(--accent-amber);
      box-shadow: 0 0 20px rgba(245, 158, 11, 0.35);
    }
    .alert-content { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .alert-text { font-size: 0.9rem; font-weight: 600; line-height: 1.4; color: #fee2e2; }
    .alert-btn {
      background: rgba(255,255,255,0.1);
      border: 1px solid rgba(255,255,255,0.2);
      color: #fff;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.78rem;
    }
    .countdown-bar-wrap {
      width: 100%;
      height: 6px;
      background: rgba(255,255,255,0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 10px;
    }
    .countdown-bar {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #f59e0b, #ef4444);
      transition: width 0.2s linear;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .metric-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      border-radius: 16px;
      padding: 18px;
      position: relative;
      overflow: hidden;
    }
    .metric-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .metric-val { font-size: 1.95rem; font-weight: 800; color: #fff; letter-spacing: -0.03em; }
    .metric-unit { font-size: 0.95rem; font-weight: 500; color: var(--primary); margin-left: 2px; }
    .metric-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; display: flex; align-items: center; justify-content: space-between; }
    .power-progress-wrap {
      height: 8px;
      width: 100%;
      background: rgba(255,255,255,0.06);
      border-radius: 6px;
      overflow: hidden;
      margin-top: 10px;
    }
    .power-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #10b981, #38bdf8);
      border-radius: 6px;
      transition: width 0.4s ease;
    }
    .power-progress-bar.warning { background: linear-gradient(90deg, #38bdf8, #f59e0b); }
    .power-progress-bar.danger { background: linear-gradient(90deg, #f59e0b, #ef4444); box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }

    .section-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #e2e8f0;
      margin: 10px 0 14px 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .relays-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }
    .relay-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      border-radius: 18px;
      padding: 22px;
      transition: all 0.25s ease;
    }
    .relay-card.is-on {
      border-color: rgba(16, 185, 129, 0.4);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
    }
    .relay-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
    .relay-name-box h3 { font-size: 1.15rem; font-weight: 700; color: #fff; }
    .relay-pin-tag { font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-top: 3px; }
    .relay-status-pill {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 20px;
      background: rgba(148, 163, 184, 0.15);
      color: #94a3b8;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .relay-status-pill.on {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.4);
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
    }
    .relay-desc { font-size: 0.83rem; color: var(--text-muted); line-height: 1.45; margin-bottom: 20px; }
    .relay-footer { display: flex; justify-content: space-between; align-items: center; }
    .priority-tag { font-size: 0.72rem; font-weight: 600; padding: 4px 8px; border-radius: 6px; }
    .priority-high { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
    .priority-low { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }

    .switch { position: relative; display: inline-block; width: 58px; height: 32px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155; transition: .3s; border-radius: 34px; border: 1px solid rgba(255,255,255,0.1);
    }
    .slider:before {
      position: absolute; content: ""; height: 24px; width: 24px; left: 4px; bottom: 3px;
      background-color: #f1f5f9; transition: .3s; border-radius: 50%;
    }
    input:checked + .slider { background-color: var(--accent-green); box-shadow: 0 0 16px var(--accent-green-glow); }
    input:checked + .slider:before { transform: translateX(24px); }

    .protect-card {
      background: var(--card-bg); border: 1px solid var(--card-border); backdrop-filter: blur(12px);
      border-radius: 16px; padding: 18px 22px; margin-bottom: 18px; display: flex; flex-wrap: wrap;
      justify-content: space-between; align-items: center; gap: 16px;
    }
    .protect-info h4 { font-size: 1rem; font-weight: 700; color: #f1f5f9; }
    .protect-info p { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; max-width: 620px; }

    footer { text-align: center; font-size: 0.76rem; color: #64748b; margin-top: 10px; padding: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-titles">
        <h1>⚡ Smart Home EMS Node</h1>
        <p>ESP32 Dual-Core Hybrid Controller (MQTT + Embedded Web Dashboard)</p>
      </div>
      <div class="badge-bar">
        <div class="badge"><div class="badge-dot pulse"></div><span id="wifiMode">Connecting...</span></div>
        <div class="badge">🌐 <span id="ipAddr">--</span></div>
        <div class="badge">📡 MQTT: <span id="mqttStatus">Connecting...</span></div>
      </div>
    </header>

    <div id="alertBanner">
      <div class="alert-content">
        <div class="alert-text" id="alertText">⚠️ System Alert Active</div>
        <button class="alert-btn" onclick="clearAlert()">Dismiss</button>
      </div>
      <div class="countdown-bar-wrap" id="countdownWrap">
        <div class="countdown-bar" id="countdownBar"></div>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-title"><span>Active Power</span><span style="font-size:0.7rem; color: #38bdf8;">Limit: 80W</span></div>
        <div><span class="metric-val" id="valPower">0.0</span><span class="metric-unit">W</span></div>
        <div class="power-progress-wrap"><div class="power-progress-bar" id="powerBar"></div></div>
        <div class="metric-sub"><span>Target ~90W max</span><span id="powerStatus">Idle (0W)</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-title"><span>Mains Voltage</span><span>Grid</span></div>
        <div><span class="metric-val" id="valVoltage">230.0</span><span class="metric-unit">V</span></div>
        <div class="metric-sub"><span>Freq: <b id="valFreq" style="color:#e2e8f0;">50.0</b> Hz</span><span style="color:#10b981;">Nominal</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-title"><span>AC Current</span><span>Calculated</span></div>
        <div><span class="metric-val" id="valCurrent">0.000</span><span class="metric-unit">A</span></div>
        <div class="metric-sub"><span>PF: <b id="valPF" style="color:#e2e8f0;">0.00</b></span><span>Active</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-title"><span>Cumulative Energy</span><span>kWh</span></div>
        <div><span class="metric-val" id="valEnergy">0.0000</span><span class="metric-unit">kWh</span></div>
        <div class="metric-sub"><span>Real-time integrate</span><span style="color:#a855f7;">Active</span></div>
      </div>
    </div>

    <div class="section-title"><span>🔌 Physical Relay Sockets</span><span style="font-size: 0.78rem; font-weight: 500; color: #94a3b8;">(Active-LOW: 0V=ON, 3.3V=OFF)</span></div>
    <div class="relays-grid">
      <div class="relay-card" id="relay1Card">
        <div class="relay-header">
          <div class="relay-name-box">
            <h3>Socket 1: Laptop Charger</h3>
            <div class="relay-pin-tag">ESP32: GPIO 26 • ~65W</div>
          </div>
          <div class="relay-status-pill" id="relay1Pill">STANDBY</div>
        </div>
        <div class="relay-desc">Primary essential load. Protected from load shedding. Controlled via Web Dashboard & MQTT.</div>
        <div class="relay-footer">
          <span class="priority-tag priority-high">★ HIGH PRIORITY</span>
          <label class="switch"><input type="checkbox" id="relay1Toggle" onchange="toggleRelay(1, this.checked)"><span class="slider"></span></label>
        </div>
      </div>

      <div class="relay-card" id="relay2Card">
        <div class="relay-header">
          <div class="relay-name-box">
            <h3>Socket 2: Phone Charger</h3>
            <div class="relay-pin-tag">ESP32: GPIO 27 • ~25W</div>
          </div>
          <div class="relay-status-pill" id="relay2Pill">STANDBY</div>
        </div>
        <div class="relay-desc">Secondary load. Automatically shed after 3s if total circuit power exceeds 80W.</div>
        <div class="relay-footer">
          <span class="priority-tag priority-low">⚡ LOW PRIORITY (SHED)</span>
          <label class="switch"><input type="checkbox" id="relay2Toggle" onchange="toggleRelay(2, this.checked)"><span class="slider"></span></label>
        </div>
      </div>
    </div>

    <div class="protect-card">
      <div class="protect-info">
        <h4>🛡️ Auto Overload Protection (80W Ceiling)</h4>
        <p>When both sockets are ON (~90W), triggers a 3-second grace countdown and automatically turns OFF Socket 2.</p>
        <div style="font-size: 0.75rem; color:#f59e0b; margin-top: 6px;">Total Shed Events: <b id="tripCounter">0</b></div>
      </div>
      <label class="switch"><input type="checkbox" id="autoOverloadToggle" onchange="toggleAutoOverload(this.checked)"><span class="slider"></span></label>
    </div>

    <footer>Smart Home Energy Management System &bull; ESP32 Edge Node &bull; Dual-Core FreeRTOS</footer>
  </div>

  <script>
    let isRequesting = false;
    async function fetchStatus() {
      if (isRequesting) return;
      isRequesting = true;
      try {
        const res = await fetch('/status');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        updateUI(data);
      } catch (err) {
        console.error(err);
      } finally {
        isRequesting = false;
      }
    }

    function updateUI(data) {
      document.getElementById('wifiMode').textContent = data.wifi_mode;
      document.getElementById('ipAddr').textContent = data.ip;
      document.getElementById('mqttStatus').textContent = data.mqtt_connected ? 'ONLINE' : 'CONNECTING...';
      document.getElementById('valPower').textContent = data.power.toFixed(1);
      document.getElementById('valVoltage').textContent = data.voltage.toFixed(1);
      document.getElementById('valFreq').textContent = data.frequency.toFixed(1);
      document.getElementById('valCurrent').textContent = data.current.toFixed(3);
      document.getElementById('valPF').textContent = data.pf.toFixed(2);
      document.getElementById('valEnergy').textContent = data.energy.toFixed(5);
      document.getElementById('tripCounter').textContent = data.trip_count;

      const powerPct = Math.min((data.power / 100) * 100, 100);
      const pBar = document.getElementById('powerBar');
      pBar.style.width = powerPct + '%';
      pBar.className = 'power-progress-bar';
      const pStatus = document.getElementById('powerStatus');
      if (data.power > 80.0) {
        pBar.classList.add('danger');
        pStatus.textContent = 'OVERLOAD (>80W)';
        pStatus.style.color = '#ef4444';
      } else if (data.power > 60.0) {
        pBar.classList.add('warning');
        pStatus.textContent = 'High Load';
        pStatus.style.color = '#f59e0b';
      } else if (data.power > 0) {
        pStatus.textContent = 'Normal Load';
        pStatus.style.color = '#10b981';
      } else {
        pStatus.textContent = 'Idle (0W)';
        pStatus.style.color = '#94a3b8';
      }

      const r1Toggle = document.getElementById('relay1Toggle');
      if (document.activeElement !== r1Toggle) r1Toggle.checked = data.relay1;
      const r1Card = document.getElementById('relay1Card');
      const r1Pill = document.getElementById('relay1Pill');
      if (data.relay1) {
        r1Card.classList.add('is-on');
        r1Pill.textContent = 'LIVE ON (~65W)';
        r1Pill.classList.add('on');
      } else {
        r1Card.classList.remove('is-on');
        r1Pill.textContent = 'STANDBY (0W)';
        r1Pill.classList.remove('on');
      }

      const r2Toggle = document.getElementById('relay2Toggle');
      if (document.activeElement !== r2Toggle) r2Toggle.checked = data.relay2;
      const r2Card = document.getElementById('relay2Card');
      const r2Pill = document.getElementById('relay2Pill');
      if (data.relay2) {
        r2Card.classList.add('is-on');
        r2Pill.textContent = 'LIVE ON (~25W)';
        r2Pill.classList.add('on');
      } else {
        r2Card.classList.remove('is-on');
        r2Pill.textContent = 'STANDBY (0W)';
        r2Pill.classList.remove('on');
      }

      const autoToggle = document.getElementById('autoOverloadToggle');
      if (document.activeElement !== autoToggle) autoToggle.checked = data.auto_overload;

      const banner = document.getElementById('alertBanner');
      const alertText = document.getElementById('alertText');
      const countdownWrap = document.getElementById('countdownWrap');
      const countdownBar = document.getElementById('countdownBar');

      if (data.overload_active && data.countdown > 0) {
        banner.style.display = 'block';
        banner.className = 'warning-state';
        alertText.innerHTML = `⚠️ <b>OVERLOAD DETECTED (${data.power.toFixed(1)}W &gt; 80W)!</b> Shedding Relay 2 in <b>${data.countdown.toFixed(1)}s</b>...`;
        countdownWrap.style.display = 'block';
        const fraction = Math.max(0, Math.min(data.countdown / 3.0, 1.0));
        countdownBar.style.width = (fraction * 100) + '%';
      } else if (data.alert && data.alert.length > 0) {
        banner.style.display = 'block';
        banner.className = '';
        alertText.innerHTML = data.alert;
        countdownWrap.style.display = 'none';
      } else {
        banner.style.display = 'none';
      }
    }

    async function toggleRelay(num, state) {
      try {
        await fetch(`/toggle?relay=${num}&state=${state ? 1 : 0}`);
        setTimeout(fetchStatus, 80);
      } catch (e) {
        console.error(e);
      }
    }

    async function toggleAutoOverload(state) {
      try {
        await fetch(`/toggle?auto=${state ? 1 : 0}`);
        setTimeout(fetchStatus, 80);
      } catch (e) {
        console.error(e);
      }
    }

    async function clearAlert() {
      await fetch('/toggle?clear_alert=1');
      document.getElementById('alertBanner').style.display = 'none';
      fetchStatus();
    }

    setInterval(fetchStatus, 1000);
    fetchStatus();
  </script>
</body>
</html>
)rawliteral";

// ═══════════════════════════════════════════════════════
//  CORE 0: SAFETY & TELEMETRY SIMULATION TASK
// ═══════════════════════════════════════════════════════
void SafetySamplingTask(void* pvParameters) {
    float syntheticEnergy = 0.0f;
    unsigned long lastTickMs = millis();

    for (;;) {
        esp_task_wdt_reset();
        unsigned long now = millis();
        float dt = (now - lastTickMs) / 1000.0f;
        if (dt <= 0.0f) dt = 1.0f;
        lastTickMs = now;

        // Process incoming relay command requests from Core 1 / MQTT
        bool doCh1On = false, doCh1Off = false;
        bool doCh2On = false, doCh2Off = false;
        uint8_t ackTypeToSend = 0;
        bool autoProtect = true;

        acquireStateLock();
        if (pendingRelayOn) {
            pendingRelayOn = false;
            doCh1On = true; doCh2On = true;
            ackTypeToSend = 1;
        } else if (pendingRelayOff) {
            pendingRelayOff = false;
            doCh1Off = true; doCh2Off = true;
            ackTypeToSend = 2;
        }
        if (pendingCh1On) { pendingCh1On = false; doCh1On = true; ackTypeToSend = 5; }
        if (pendingCh1Off) { pendingCh1Off = false; doCh1Off = true; ackTypeToSend = 4; }
        if (pendingCh2On) { pendingCh2On = false; doCh2On = true; ackTypeToSend = 7; }
        if (pendingCh2Off) { pendingCh2Off = false; doCh2Off = true; ackTypeToSend = 6; }
        autoProtect = autoOverloadEnabled;
        releaseStateLock();

        if (doCh1On) applyRelayHardware(1, true);
        if (doCh1Off) applyRelayHardware(1, false);
        if (doCh2On) applyRelayHardware(2, true);
        if (doCh2Off) applyRelayHardware(2, false);

        if (ackTypeToSend != 0) {
            acquireStateLock();
            pendingAck = true;
            pendingAckType = ackTypeToSend;
            releaseStateLock();
        }

        bool r1 = relay1Closed;
        bool r2 = relay2Closed;

        // 1. Generate realistic software telemetry based on relay states
        float vNoise = ((float)random(-15, 16)) / 10.0f;
        float simVoltage = 230.0f + vNoise;

        float fNoise = ((float)random(-5, 6)) / 100.0f;
        float simFreq = 50.0f + fNoise;

        float simWatts = 0.0f;
        float simPf    = 0.0f;

        if (!r1 && !r2) {
            simWatts = 0.0f;
            simPf    = 0.00f;
        } else if (r1 && !r2) {
            float p1Noise = ((float)random(-15, 16)) / 10.0f;
            simWatts = 65.0f + p1Noise;
            simPf    = 0.95f + (((float)random(-10, 11)) / 1000.0f);
        } else if (!r1 && r2) {
            float p2Noise = ((float)random(-5, 6)) / 10.0f;
            simWatts = 25.0f + p2Noise;
            simPf    = 0.88f + (((float)random(-10, 11)) / 1000.0f);
        } else {
            float p1Noise = ((float)random(-15, 16)) / 10.0f;
            float p2Noise = ((float)random(-5, 6)) / 10.0f;
            simWatts = (65.0f + p1Noise) + (25.0f + p2Noise);
            simPf    = 0.92f + (((float)random(-10, 11)) / 1000.0f);
        }

        float simCurrent = (simWatts > 0.0f && simPf > 0.0f) ? (simWatts / (simVoltage * simPf)) : 0.0f;
        syntheticEnergy += (simWatts * (dt / 3600.0f)) / 1000.0f;

        float simVa  = simVoltage * simCurrent;
        float simVar = (simVa > simWatts) ? sqrtf(fmaxf(0.0f, (simVa * simVa) - (simWatts * simWatts))) : 0.0f;

        // 2. Overload Load-Shedding Engine (80W Ceiling)
        bool shedNow = false;
        if (autoProtect && simWatts > OVERLOAD_POWER_THRESHOLD) {
            if (!overloadTimingActive) {
                overloadTimingActive = true;
                overloadStartMs = now;
                overloadCountdownSec = 3.0f;
            } else {
                unsigned long elapsed = now - overloadStartMs;
                if (elapsed < OVERLOAD_TIME_MS) {
                    overloadCountdownSec = (OVERLOAD_TIME_MS - elapsed) / 1000.0f;
                } else {
                    // Shed Relay 2!
                    overloadTimingActive = false;
                    overloadCountdownSec = 0.0f;
                    applyRelayHardware(2, false);
                    overloadTripCount++;
                    overloadShedEvent = true;
                    shedNow = true;
                }
            }
        } else {
            if (overloadTimingActive) {
                overloadTimingActive = false;
                overloadCountdownSec = 0.0f;
            }
        }

        // Commit telemetry to shared state
        acquireStateLock();
        sharedPowerWatts       = simWatts;
        sharedVoltage          = simVoltage;
        sharedCurrent          = simCurrent;
        sharedPf               = simPf;
        sharedEnergy           = syntheticEnergy;
        sharedFrequency        = simFreq;
        sharedApparentPowerVa  = simVa;
        sharedReactivePowerVar = simVar;
        if (shedNow) {
            sharedAlertMessage = "🚨 <b>OVERLOAD LOAD-SHED TRIGGERED:</b> Combined load exceeded 80W. Socket 2 (Phone Charger) shed automatically.";
        }
        releaseStateLock();

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

// ═══════════════════════════════════════════════════════
//  WEB SERVER ROUTE HANDLERS
// ═══════════════════════════════════════════════════════
void handleRoot() {
    server.send_P(200, "text/html", INDEX_HTML);
}

void handleStatus() {
    float power, volt, curr, pfVal, energyVal, freqVal, countdown;
    bool r1, r2, autoProt, olActive;
    unsigned int trips;

    acquireStateLock();
    power     = sharedPowerWatts;
    volt      = sharedVoltage;
    curr      = sharedCurrent;
    pfVal     = sharedPf;
    energyVal = sharedEnergy;
    freqVal   = sharedFrequency;
    r1        = relay1Closed;
    r2        = relay2Closed;
    autoProt  = autoOverloadEnabled;
    olActive  = overloadTimingActive;
    countdown = overloadCountdownSec;
    trips     = overloadTripCount;
    String alert = sharedAlertMessage;
    releaseStateLock();

    String json = "{";
    json += "\"relay1\":" + String(r1 ? "true" : "false") + ",";
    json += "\"relay2\":" + String(r2 ? "true" : "false") + ",";
    json += "\"auto_overload\":" + String(autoProt ? "true" : "false") + ",";
    json += "\"voltage\":" + String(volt, 1) + ",";
    json += "\"current\":" + String(curr, 3) + ",";
    json += "\"power\":" + String(power, 1) + ",";
    json += "\"pf\":" + String(pfVal, 2) + ",";
    json += "\"frequency\":" + String(freqVal, 1) + ",";
    json += "\"energy\":" + String(energyVal, 5) + ",";
    json += "\"overload_active\":" + String(olActive ? "true" : "false") + ",";
    json += "\"countdown\":" + String(countdown, 1) + ",";
    json += "\"trip_count\":" + String(trips) + ",";
    
    alert.replace("\"", "\\\"");
    json += "\"alert\":\"" + alert + "\",";
    
    bool isAP = (WiFi.getMode() == WIFI_AP);
    json += "\"wifi_mode\":\"" + String(isAP ? "SoftAP Hotspot" : "WiFi Station") + "\",";
    json += "\"ip\":\"" + (isAP ? WiFi.softAPIP().toString() : WiFi.localIP().toString()) + "\",";
    json += "\"mqtt_connected\":" + String(client.connected() ? "true" : "false") + ",";
    json += "\"uptime\":" + String(millis() / 1000);
    json += "}";

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
}

void handleToggle() {
    if (server.hasArg("relay")) {
        int r = server.arg("relay").toInt();
        bool newState = false;
        if (server.hasArg("state")) {
            newState = (server.arg("state").toInt() == 1);
        } else {
            newState = (r == 1) ? !relay1Closed : !relay2Closed;
        }

        acquireStateLock();
        if (r == 1) {
            if (newState) pendingCh1On = true;
            else pendingCh1Off = true;
        } else if (r == 2) {
            if (newState) pendingCh2On = true;
            else pendingCh2Off = true;
        }
        releaseStateLock();
    }

    if (server.hasArg("auto")) {
        acquireStateLock();
        autoOverloadEnabled = (server.arg("auto").toInt() == 1);
        releaseStateLock();
    }

    if (server.hasArg("clear_alert")) {
        acquireStateLock();
        sharedAlertMessage = "";
        releaseStateLock();
    }

    handleStatus();
}

// ═══════════════════════════════════════════════════════
//  MQTT CALLBACK — Commands from Pipeline / API (Core 1)
// ═══════════════════════════════════════════════════════
void callback(char* topic, byte* payload, unsigned int length) {
    lastServerHB = millis();
    static const unsigned int MAX_MQTT_PAYLOAD = 256;
    if (length > MAX_MQTT_PAYLOAD) return;
    char msg[MAX_MQTT_PAYLOAD + 1];
    memcpy(msg, payload, length);
    msg[length] = '\0';
    String message = String(msg);
    message.trim();

    Serial.printf("[MQTT] Command received on %s: %s\n", topic, message.c_str());

    if (String(topic) == String(topicCommand)) {
        acquireStateLock();
        if (message == "ON") {
            pendingRelayOn = true;
        } else if (message == "OFF") {
            pendingRelayOff = true;
        } else if (message == "IN1_ON" || message == "CH1_ON") {
            pendingCh1On = true;
        } else if (message == "IN1_OFF" || message == "CH1_OFF") {
            pendingCh1Off = true;
        } else if (message == "IN2_ON" || message == "CH2_ON") {
            pendingCh2On = true;
        } else if (message == "IN2_OFF" || message == "CH2_OFF") {
            pendingCh2Off = true;
        } else if (message == "reset_safety") {
            sharedAlertMessage = "Safety Reset Confirmed";
            if (client.connected()) {
                client.publish(topicStatus, "SAFETY_RESET_CONFIRMED");
            }
        }
        releaseStateLock();
    }
}

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;
    if (millis() - lastReconnectAttempt < 5000) return;
    lastReconnectAttempt = millis();

    Serial.printf("[MQTT] Connecting to %s:%d as %s...\n", mqtt_server, mqtt_port, activeDeviceId);
    if (client.connect(activeDeviceId, mqtt_user, mqtt_password, topicStatus, 1, true, "OFFLINE")) {
        Serial.println("[MQTT] Connected successfully!");
        client.subscribe(topicCommand);
        client.publish(topicStatus, "ONLINE", true);
        lastServerHB = millis();
    } else {
        Serial.printf("[MQTT] Connection failed (rc=%d), will retry in 5s\n", client.state());
    }
}

// ═══════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n\n=======================================================");
    Serial.println("  SMART HOME EMS — HYBRID FIRMWARE (MQTT + WEB DASHBOARD)");
    Serial.println("=======================================================");

    // 1. Initialize Relays safely in OFF state
    digitalWrite(PIN_RELAY_1, RELAY_OFF);
    digitalWrite(PIN_RELAY_2, RELAY_OFF);
    pinMode(PIN_RELAY_1, OUTPUT);
    pinMode(PIN_RELAY_2, OUTPUT);

    pinMode(PIN_OPTO_VCC, OUTPUT);
    digitalWrite(PIN_OPTO_VCC, HIGH);

    stateMutex = xSemaphoreCreateMutex();
    allRelaysOff();

    // 2. Launch Core 0 Safety & Simulation Task
    xTaskCreatePinnedToCore(
        SafetySamplingTask,
        "SafetySampling",
        4096,
        NULL,
        2,
        &safetyTaskHandle,
        0
    );
    Serial.println("[INIT] Core 0: SafetySamplingTask launched");

    // 3. Dual-Mode WiFi Connection
    Serial.printf("[WiFi] Attempting connection to SSID: %s\n", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    unsigned long startAttempt = millis();
    bool wifiOk = false;
    while (millis() - startAttempt < 10000) {
        if (WiFi.status() == WL_CONNECTED) {
            wifiOk = true;
            break;
        }
        delay(500);
        Serial.print(".");
    }

    if (wifiOk) {
        Serial.println("\n[WiFi] Connected to network!");
        Serial.printf("[WiFi] IP Address: http://%s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] WiFi connection timed out! Launching SoftAP Hotspot...");
        WiFi.disconnect(true);
        delay(100);
        WiFi.mode(WIFI_AP);
        WiFi.softAP(AP_SSID, AP_PASSWORD);
        Serial.printf("[WiFi] SoftAP SSID: %s | URL: http://%s\n", AP_SSID, WiFi.softAPIP().toString().c_str());
    }

    // 4. Configure Device ID and MQTT Topics
    if (strcmp(DEVICE_ID, "") == 0 || strcmp(DEVICE_ID, "auto") == 0) {
        uint8_t mac[6];
        WiFi.macAddress(mac);
        snprintf(activeDeviceId, sizeof(activeDeviceId), "esp32_%02X%02X%02X", mac[3], mac[4], mac[5]);
    } else {
        strncpy(activeDeviceId, DEVICE_ID, sizeof(activeDeviceId) - 1);
        activeDeviceId[sizeof(activeDeviceId) - 1] = '\0';
    }

    snprintf(topicPower,     sizeof(topicPower),     "home/sensor/%s/power",     activeDeviceId);
    snprintf(topicTelemetry, sizeof(topicTelemetry), "home/sensor/%s/telemetry", activeDeviceId);
    snprintf(topicCommand,   sizeof(topicCommand),   "home/plug/%s/command",     activeDeviceId);
    snprintf(topicStatus,    sizeof(topicStatus),    "home/sensor/%s/status",    activeDeviceId);
    snprintf(topicAck,       sizeof(topicAck),       "home/plug/%s/ack",         activeDeviceId);

    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback);
    client.setKeepAlive(15);

    // 5. Setup Web Server
    server.on("/", HTTP_GET, handleRoot);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/toggle", HTTP_GET, handleToggle);
    server.begin();
    Serial.println("[HTTP] Web Dashboard server started on port 80");

    // Watchdog
    esp_task_wdt_init(WATCHDOG_TIMEOUT_S, true);
    esp_task_wdt_add(NULL);
    if (safetyTaskHandle != nullptr) {
        esp_task_wdt_add(safetyTaskHandle);
    }
    Serial.println("=======================================================\n");
}

// ═══════════════════════════════════════════════════════
//  MAIN LOOP (Core 1)
// ═══════════════════════════════════════════════════════
void loop() {
    esp_task_wdt_reset();

    // 1. Handle Web Clients
    server.handleClient();

    // 2. Handle MQTT
    if (WiFi.getMode() == WIFI_STA && WiFi.status() == WL_CONNECTED) {
        if (!client.connected()) {
            reconnectMQTT();
        }
        client.loop();
    }

    // 3. Publish Acknowledgements
    bool ackReady = false;
    uint8_t ackType = 0;
    acquireStateLock();
    if (pendingAck) {
        ackReady = true;
        ackType = pendingAckType;
        pendingAck = false;
        pendingAckType = 0;
    }
    if (overloadShedEvent) {
        overloadShedEvent = false;
        if (client.connected()) {
            client.publish(topicStatus, "OVERLOAD_LOAD_SHED:CH2_OFF");
        }
    }
    releaseStateLock();

    if (ackReady && client.connected()) {
        const char* ackPayload = (ackType == 1) ? "ON_CONFIRMED"
                               : (ackType == 2) ? "OFF_CONFIRMED"
                               : (ackType == 4) ? "CH1_OFF_CONFIRMED"
                               : (ackType == 5) ? "CH1_ON_CONFIRMED"
                               : (ackType == 6) ? "CH2_OFF_CONFIRMED"
                               : (ackType == 7) ? "CH2_ON_CONFIRMED"
                                                : "CONFIRMED";
        client.publish(topicAck, ackPayload);
    }

    // 4. Periodic MQTT Telemetry Broadcasts
    unsigned long now = millis();
    if (now - lastMsgMs >= 1000) {
        lastMsgMs = now;
        float pWatts;
        acquireStateLock();
        pWatts = sharedPowerWatts;
        releaseStateLock();

        if (client.connected()) {
            char pStr[16];
            dtostrf(pWatts, 5, 2, pStr);
            client.publish(topicPower, pStr);
        }
    }

    if (now - lastTelemetryMs >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryMs = now;
        float v, i, w, pf, e, f, va, var;
        bool r1, r2;

        acquireStateLock();
        v   = sharedVoltage;
        i   = sharedCurrent;
        w   = sharedPowerWatts;
        pf  = sharedPf;
        e   = sharedEnergy;
        f   = sharedFrequency;
        va  = sharedApparentPowerVa;
        var = sharedReactivePowerVar;
        r1  = relay1Closed;
        r2  = relay2Closed;
        releaseStateLock();

        if (client.connected()) {
            StaticJsonDocument<512> doc;
            doc["v"] = roundf(v * 10.0f) / 10.0f;
            doc["i"] = roundf(i * 100.0f) / 100.0f;
            doc["w"] = roundf(w * 10.0f) / 10.0f;
            doc["pf"] = roundf(pf * 100.0f) / 100.0f;
            doc["voltage_rms"] = doc["v"];
            doc["current_rms"] = doc["i"];
            doc["active_power_w"] = doc["w"];
            doc["apparent_power_va"] = roundf(va * 10.0f) / 10.0f;
            doc["reactive_power_var"] = roundf(var * 10.0f) / 10.0f;
            doc["power_factor"] = doc["pf"];
            doc["frequency_hz"] = roundf(f * 10.0f) / 10.0f;
            doc["energy_kwh"] = roundf(e * 1000.0f) / 1000.0f;
            doc["source"] = "synthetic";

            JsonArray relays = doc.createNestedArray("relays");
            relays.add(r1 ? 1 : 0);
            relays.add(r2 ? 1 : 0);

            char jsonBuf[512];
            serializeJson(doc, jsonBuf);
            client.publish(topicTelemetry, jsonBuf);
        }
    }
}
