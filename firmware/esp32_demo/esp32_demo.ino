/*
 * ==============================================================================
 *  SMART HOME EMS — ESP32 HARDWARE DEMO & REALTIME TELEMETRY SIMULATOR
 * ==============================================================================
 *  Target Board : ESP32 DevKit V1 (30-pin / 38-pin WROOM)
 *  Framework    : Arduino IDE / PlatformIO
 *  Dependencies : Built-in ESP32 core ONLY (<WiFi.h>, <WebServer.h>)
 *                 NO external third-party libraries required!
 *
 *  HARDWARE SETUP (PZEM Disconnected):
 *  ------------------------------------------------------------------------------
 *  - Relay 1 (High Priority: Laptop Charger, ~65W) -> GPIO 26
 *  - Relay 2 (Low Priority : Phone Charger,  ~25W) -> GPIO 27
 *  - Relay Opto VCC (3.3V supply for optocouplers) -> GPIO 25 (driven HIGH)
 *  - Relay Coil JD-VCC                             -> External 5V / VIN
 *  - Active-LOW Logic: LOW = Relay ON (Closed), HIGH = Relay OFF (Open)
 *
 *  FEATURES:
 *  ------------------------------------------------------------------------------
 *  1. Dual-Mode WiFi:
 *     - Tries connecting to user WiFi credentials for 10 seconds.
 *     - If unavailable or fails, automatically falls back to SoftAP Hotspot:
 *       SSID: "Smart_EMS_Demo" | Password: "12345678" | IP: 192.168.4.1
 *  2. Software Energy Telemetry Simulator:
 *     - Updates at 1Hz (1-second loop) based on physical relay states:
 *       • Voltage     : 230V AC mains with realistic grid fluctuations (228.5V - 231.5V)
 *       • Frequency   : ~50.0 Hz (49.95 - 50.05 Hz)
 *       • Active Power: 0W (both OFF), ~65W +/- 1.5W (Relay 1 ON), ~25W +/- 0.5W (Relay 2 ON),
 *                       ~90W total (+/- jitter) when both are ON.
 *       • Power Factor: 0.00 (OFF), ~0.95 (Relay 1), ~0.88 (Relay 2), ~0.92 (Both)
 *       • Current     : Power / (Voltage * PF)
 *       • Energy (kWh): Accumulates realistically over time based on active wattage.
 *  3. Automatic Overload Load-Shedding (80W Ceiling):
 *     - When both Relay 1 & 2 are turned ON, total power (~90W) exceeds 80W.
 *     - If Auto-Protection is enabled, a 3-second countdown initiates.
 *     - If overload persists for 3 continuous seconds, Relay 2 (Phone Charger) is
 *       automatically shed (turned OFF), protecting Relay 1 (Laptop Charger).
 *  4. Embedded Dark-Mode Web Dashboard:
 *     - Served on Port 80, 100% standalone (no CDN, works offline in hotspot).
 *     - Auto-polls /status every 1s via AJAX/Fetch for instant, flicker-free updates.
 *     - Live toggle switches for Relay 1, Relay 2, and Auto-Overload Protection.
 * ==============================================================================
 */

#include <WiFi.h>
#include <WebServer.h>

// ==============================================================================
//  USER CONFIGURATION — WiFi Credentials
// ==============================================================================
// Enter your local WiFi details here:
const char* WIFI_SSID     = "Pramodsharma’s iPhone"; // Replace with your WiFi SSID
const char* WIFI_PASSWORD = "!Xnxxx6969420@69-6969:420(Xxx)@1"; // Replace with your WiFi Password

// Fallback SoftAP Hotspot Credentials
const char* AP_SSID       = "Smart_EMS_Demo";
const char* AP_PASSWORD   = "12345678";

// ==============================================================================
//  HARDWARE PIN DEFINITIONS
// ==============================================================================
const int PIN_RELAY_1     = 26; // High-Priority Load: Laptop Charger (~65W)
const int PIN_RELAY_2     = 27; // Low-Priority Load : Phone Charger  (~25W)
const int PIN_OPTO_VCC    = 25; // 3.3V source for Relay Module Opto-Isolators

// Relay Configuration (Active-LOW: LOW = Relay Energized / ON, HIGH = CUTOFF / OFF)
#define RELAY_ACTIVE_ON   LOW
#define RELAY_ACTIVE_OFF  HIGH

// Overload Protection Thresholds
const float OVERLOAD_POWER_THRESHOLD = 80.0f; // Watts
const unsigned long OVERLOAD_TIME_MS = 3000;   // 3 Seconds

// ==============================================================================
//  GLOBAL STATE VARIABLES
// ==============================================================================
bool relay1State          = false; // false = OFF, true = ON
bool relay2State          = false; // false = OFF, true = ON
bool autoOverloadEnabled  = true;  // Automatic 80W load shedding enabled by default

// Simulated Telemetry Values
float simVoltage          = 230.0f; // Volts AC
float simCurrent          = 0.000f; // Amperes
float simPower            = 0.0f;   // Active Watts
float simEnergyKWh        = 0.00000f; // Cumulative kWh
float simFrequency        = 50.0f;  // Hertz
float simPF               = 0.00f;  // Power Factor

// Overload Tracking
bool overloadTimingActive = false;
unsigned long overloadStartTime = 0;
float overloadCountdownSec = 0.0f;
unsigned int overloadTripCount = 0;
String alertMessage       = "System Initialized. All circuits nominal.";

// Timing
unsigned long lastTelemetryUpdate = 0;

// Web Server instance on port 80
WebServer server(80);

// Forward declarations
void updateTelemetry();
void checkOverloadProtection();
void applyRelayHardware(int relayNum, bool state);
void handleRoot();
void handleStatus();
void handleToggle();
void handleNotFound();

// ==============================================================================
//  EMBEDDED DASHBOARD HTML / CSS / JS (Stored in Flash Memory - PROGMEM)
// ==============================================================================
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Home EMS | ESP32 Hardware Node</title>
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
    
    /* Header */
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

    /* Alert Banner */
    #alertBanner {
      display: none;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--accent-red);
      border-radius: 14px;
      padding: 14px 18px;
      margin-bottom: 18px;
      animation: alertPulse 1.8s infinite;
      box-shadow: 0 0 20px var(--accent-red-glow);
    }
    #alertBanner.warning-state {
      background: rgba(245, 158, 11, 0.15);
      border-color: var(--accent-amber);
      box-shadow: 0 0 20px rgba(245, 158, 11, 0.35);
    }
    @keyframes alertPulse { 0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); } 50% { box-shadow: 0 0 28px rgba(239, 68, 68, 0.6); } }
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

    /* Telemetry Grid */
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
      transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.3); }
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
      border: 1px solid rgba(255,255,255,0.05);
    }
    .power-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #10b981, #38bdf8);
      border-radius: 6px;
      transition: width 0.4s ease, background 0.4s ease;
    }
    .power-progress-bar.warning { background: linear-gradient(90deg, #38bdf8, #f59e0b); }
    .power-progress-bar.danger { background: linear-gradient(90deg, #f59e0b, #ef4444); box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }

    /* Relay Control Cards */
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
      position: relative;
    }
    .relay-card.is-on {
      border-color: rgba(16, 185, 129, 0.4);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
    }
    .relay-card.is-overload-shed {
      border-color: rgba(239, 68, 68, 0.5);
      box-shadow: 0 8px 24px rgba(239, 68, 68, 0.2);
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
      transition: all 0.2s ease;
    }
    .relay-status-pill.on {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.4);
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
    }
    .relay-desc { font-size: 0.83rem; color: var(--text-muted); line-height: 1.45; margin-bottom: 20px; }
    .relay-footer { display: flex; justify-content: space-between; align-items: center; }
    .priority-tag {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 6px;
    }
    .priority-high { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
    .priority-low { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }

    /* Custom Toggle Switch */
    .switch {
      position: relative;
      display: inline-block;
      width: 58px;
      height: 32px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      transition: .3s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 34px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 24px;
      width: 24px;
      left: 4px;
      bottom: 3px;
      background-color: #f1f5f9;
      transition: .3s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 50%;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    input:checked + .slider {
      background-color: var(--accent-green);
      box-shadow: 0 0 16px var(--accent-green-glow);
    }
    input:checked + .slider:before {
      transform: translateX(24px);
      background-color: #ffffff;
    }

    /* Protection Mode Card */
    .protect-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      border-radius: 16px;
      padding: 18px 22px;
      margin-bottom: 18px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .protect-info h4 { font-size: 1rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 8px; }
    .protect-info p { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; max-width: 620px; }
    
    /* Footer & System Info */
    footer {
      text-align: center;
      font-size: 0.76rem;
      color: #64748b;
      margin-top: 10px;
      padding: 10px;
    }
    .btn-reset-kwh {
      background: transparent;
      border: 1px solid rgba(255,255,255,0.1);
      color: #94a3b8;
      font-size: 0.7rem;
      padding: 2px 6px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-reset-kwh:hover { border-color: var(--primary); color: #fff; }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="header-titles">
        <h1>⚡ Smart Home EMS Node</h1>
        <p>ESP32 DevKit V1 Hardware Controller & Real-Time Telemetry Simulator</p>
      </div>
      <div class="badge-bar">
        <div class="badge"><div class="badge-dot pulse"></div><span id="wifiMode">Connecting...</span></div>
        <div class="badge">🌐 <span id="ipAddr">--</span></div>
        <div class="badge">⏱️ <span id="uptime">0s</span></div>
      </div>
    </header>

    <!-- Alert / Load-Shedding Countdown Banner -->
    <div id="alertBanner">
      <div class="alert-content">
        <div class="alert-text" id="alertText">⚠️ System Alert Active</div>
        <button class="alert-btn" onclick="clearAlert()">Dismiss</button>
      </div>
      <div class="countdown-bar-wrap" id="countdownWrap">
        <div class="countdown-bar" id="countdownBar"></div>
      </div>
    </div>

    <!-- Telemetry Cards -->
    <div class="metrics-grid">
      <!-- Active Power -->
      <div class="metric-card">
        <div class="metric-title">
          <span>Active Power</span>
          <span style="font-size:0.7rem; color: #38bdf8;">Limit: 80W</span>
        </div>
        <div>
          <span class="metric-val" id="valPower">0.0</span>
          <span class="metric-unit">W</span>
        </div>
        <div class="power-progress-wrap">
          <div class="power-progress-bar" id="powerBar"></div>
        </div>
        <div class="metric-sub">
          <span>Target ~90W max</span>
          <span id="powerStatus">Idle (0W)</span>
        </div>
      </div>

      <!-- Mains Voltage & Frequency -->
      <div class="metric-card">
        <div class="metric-title"><span>Mains Voltage</span><span>Grid</span></div>
        <div>
          <span class="metric-val" id="valVoltage">230.0</span>
          <span class="metric-unit">V</span>
        </div>
        <div class="metric-sub">
          <span>Freq: <b id="valFreq" style="color:#e2e8f0;">50.0</b> Hz</span>
          <span style="color:#10b981;">Nominal</span>
        </div>
      </div>

      <!-- Current & Power Factor -->
      <div class="metric-card">
        <div class="metric-title"><span>AC Current</span><span>Calculated</span></div>
        <div>
          <span class="metric-val" id="valCurrent">0.000</span>
          <span class="metric-unit">A</span>
        </div>
        <div class="metric-sub">
          <span>PF: <b id="valPF" style="color:#e2e8f0;">0.00</b></span>
          <span id="loadType">Resistive</span>
        </div>
      </div>

      <!-- Cumulative Energy -->
      <div class="metric-card">
        <div class="metric-title">
          <span>Accumulated Energy</span>
          <button class="btn-reset-kwh" title="Reset Energy Counter" onclick="resetKWh()">Reset</button>
        </div>
        <div>
          <span class="metric-val" id="valEnergy">0.0000</span>
          <span class="metric-unit">kWh</span>
        </div>
        <div class="metric-sub">
          <span>Real-time integrate</span>
          <span style="color:#a855f7;">Active</span>
        </div>
      </div>
    </div>

    <!-- Relay Switching Section -->
    <div class="section-title">
      <span>🔌 Physical Relay Socket Controls</span>
      <span style="font-size: 0.78rem; font-weight: 500; color: #94a3b8;">(Active-LOW: 0V = ON, 3.3V = OFF)</span>
    </div>

    <div class="relays-grid">
      <!-- Relay 1 -->
      <div class="relay-card" id="relay1Card">
        <div class="relay-header">
          <div class="relay-name-box">
            <h3>Socket 1: Laptop Charger</h3>
            <div class="relay-pin-tag">ESP32 Pin: GPIO 26 • Nominal ~65W</div>
          </div>
          <div class="relay-status-pill" id="relay1Pill">STANDBY</div>
        </div>
        <div class="relay-desc">
          Primary essential appliance socket. Protected against automated load shedding to prevent data interruption.
        </div>
        <div class="relay-footer">
          <span class="priority-tag priority-high">★ HIGH PRIORITY</span>
          <label class="switch">
            <input type="checkbox" id="relay1Toggle" onchange="toggleRelay(1, this.checked)">
            <span class="slider"></span>
          </label>
        </div>
      </div>

      <!-- Relay 2 -->
      <div class="relay-card" id="relay2Card">
        <div class="relay-header">
          <div class="relay-name-box">
            <h3>Socket 2: Phone Charger</h3>
            <div class="relay-pin-tag">ESP32 Pin: GPIO 27 • Nominal ~25W</div>
          </div>
          <div class="relay-status-pill" id="relay2Pill">STANDBY</div>
        </div>
        <div class="relay-desc">
          Secondary non-critical socket. Sheds automatically after 3 seconds if combined load exceeds 80W ceiling.
        </div>
        <div class="relay-footer">
          <span class="priority-tag priority-low">⚡ LOW PRIORITY (SHED)</span>
          <label class="switch">
            <input type="checkbox" id="relay2Toggle" onchange="toggleRelay(2, this.checked)">
            <span class="slider"></span>
          </label>
        </div>
      </div>
    </div>

    <!-- Auto-Overload Protection Setting -->
    <div class="protect-card">
      <div class="protect-info">
        <h4>🛡️ Auto Overload Protection (80W Ceiling)</h4>
        <p>
          Simulates smart home grid overload protection: When both sockets are energized simultaneously (~90W total),
          the controller triggers a 3-second grace countdown and automatically turns OFF Socket 2 (Phone Charger) to protect the circuit.
        </p>
        <div style="font-size: 0.75rem; color:#f59e0b; margin-top: 6px;">
          Total Overload Shed Events: <b id="tripCounter">0</b>
        </div>
      </div>
      <label class="switch">
        <input type="checkbox" id="autoOverloadToggle" onchange="toggleAutoOverload(this.checked)">
        <span class="slider"></span>
      </label>
    </div>

    <!-- Architecture & Pinout Reference -->
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 14px; padding: 16px 20px; margin-bottom: 20px; font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
      <b style="color: #f1f5f9;">Hardware Bench Wiring Status:</b><br>
      • <b>Relay 1:</b> GPIO 26 &rarr; IN1 | <b>Relay 2:</b> GPIO 27 &rarr; IN2 | <b>Opto VCC:</b> GPIO 25 &rarr; 3.3V Source<br>
      • <b>PZEM-004T:</b> Burnt-out sensor bypassed; 1Hz telemetry dynamically simulated in firmware.
    </div>

    <footer>
      Smart Home Energy Management System &bull; ESP32 Hardware Edge Node &bull; Zero External Dependencies
    </footer>
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
        console.error('Fetch error:', err);
      } finally {
        isRequesting = false;
      }
    }

    function updateUI(data) {
      // WiFi & Meta
      document.getElementById('wifiMode').textContent = data.wifi_mode;
      document.getElementById('ipAddr').textContent = data.ip;
      
      const secs = data.uptime;
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      const s = secs % 60;
      document.getElementById('uptime').textContent = (h > 0 ? h + 'h ' : '') + m + 'm ' + s + 's';

      // Telemetry Values
      document.getElementById('valPower').textContent = data.power.toFixed(1);
      document.getElementById('valVoltage').textContent = data.voltage.toFixed(1);
      document.getElementById('valFreq').textContent = data.frequency.toFixed(1);
      document.getElementById('valCurrent').textContent = data.current.toFixed(3);
      document.getElementById('valPF').textContent = data.pf.toFixed(2);
      document.getElementById('valEnergy').textContent = data.energy.toFixed(5);
      document.getElementById('tripCounter').textContent = data.trip_count;

      // Power Bar & Status
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

      // Relay 1 UI
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

      // Relay 2 UI
      const r2Toggle = document.getElementById('relay2Toggle');
      if (document.activeElement !== r2Toggle) r2Toggle.checked = data.relay2;
      const r2Card = document.getElementById('relay2Card');
      const r2Pill = document.getElementById('relay2Pill');
      if (data.relay2) {
        r2Card.classList.add('is-on');
        r2Card.classList.remove('is-overload-shed');
        r2Pill.textContent = 'LIVE ON (~25W)';
        r2Pill.classList.add('on');
      } else {
        r2Card.classList.remove('is-on');
        r2Pill.textContent = 'STANDBY (0W)';
        r2Pill.classList.remove('on');
      }

      // Auto Overload Setting
      const autoToggle = document.getElementById('autoOverloadToggle');
      if (document.activeElement !== autoToggle) autoToggle.checked = data.auto_overload;

      // Alerts & Overload Countdown Bar
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

    async function resetKWh() {
      if (confirm('Reset cumulative energy counter to 0.00000 kWh?')) {
        await fetch('/toggle?reset_kwh=1');
        fetchStatus();
      }
    }

    async function clearAlert() {
      await fetch('/toggle?clear_alert=1');
      document.getElementById('alertBanner').style.display = 'none';
      fetchStatus();
    }

    // Auto-refresh every 1 second
    setInterval(fetchStatus, 1000);
    fetchStatus();
  </script>
</body>
</html>
)rawliteral";

// ==============================================================================
//  HARDWARE RELAY HELPER
// ==============================================================================
void applyRelayHardware(int relayNum, bool state) {
  int pin = (relayNum == 1) ? PIN_RELAY_1 : PIN_RELAY_2;
  // Active-LOW: LOW = ON (energized), HIGH = OFF (open)
  digitalWrite(pin, state ? RELAY_ACTIVE_ON : RELAY_ACTIVE_OFF);
  Serial.printf("[RELAY] Socket %d (GPIO %d) -> %s\n", 
                relayNum, pin, state ? "ON (Active-LOW: 0V)" : "OFF (Active-LOW: 3.3V)");
}

// ==============================================================================
//  TELEMETRY SIMULATOR & OVERLOAD PROTECTION ENGINE
// ==============================================================================
void updateTelemetry() {
  unsigned long now = millis();
  float dtSec = (now - lastTelemetryUpdate) / 1000.0f;
  if (dtSec <= 0.0f) dtSec = 1.0f;
  lastTelemetryUpdate = now;

  // 1. Grid Voltage: 230V AC fluctuating slightly around 228.5V to 231.5V
  float vNoise = ((float)random(-15, 16)) / 10.0f; // -1.5V to +1.5V
  simVoltage = 230.0f + vNoise;

  // 2. Frequency: ~50.0 Hz (49.95 to 50.05 Hz)
  float fNoise = ((float)random(-5, 6)) / 100.0f;  // -0.05 to +0.05 Hz
  simFrequency = 50.0f + fNoise;

  // 3. Active Power & Power Factor based on physical relay states
  if (!relay1State && !relay2State) {
    // Both OFF
    simPower = 0.0f;
    simPF    = 0.00f;
  } else if (relay1State && !relay2State) {
    // Only Relay 1 ON (Laptop Charger: ~65W +/- 1.5W jitter, PF ~0.95)
    float p1Noise = ((float)random(-15, 16)) / 10.0f;
    float pfNoise = ((float)random(-10, 11)) / 1000.0f;
    simPower = 65.0f + p1Noise;
    simPF    = 0.95f + pfNoise;
  } else if (!relay1State && relay2State) {
    // Only Relay 2 ON (Phone Charger: ~25W +/- 0.5W jitter, PF ~0.88)
    float p2Noise = ((float)random(-5, 6)) / 10.0f;
    float pfNoise = ((float)random(-10, 11)) / 1000.0f;
    simPower = 25.0f + p2Noise;
    simPF    = 0.88f + pfNoise;
  } else {
    // Both Relays ON (Combined ~90W total +/- jitter, PF ~0.92)
    float p1Noise = ((float)random(-15, 16)) / 10.0f;
    float p2Noise = ((float)random(-5, 6)) / 10.0f;
    float pfNoise = ((float)random(-10, 11)) / 1000.0f;
    simPower = (65.0f + p1Noise) + (25.0f + p2Noise);
    simPF    = 0.92f + pfNoise;
  }

  // 4. Current Calculation: I = P / (V * PF)
  if (simPower > 0.0f && simPF > 0.0f && simVoltage > 0.0f) {
    simCurrent = simPower / (simVoltage * simPF);
  } else {
    simCurrent = 0.000f;
  }

  // 5. Energy Accumulator (kWh)
  simEnergyKWh += (simPower * (dtSec / 3600.0f)) / 1000.0f;
}

void checkOverloadProtection() {
  unsigned long now = millis();

  // Overload condition: Total power exceeds 80W ceiling
  if (autoOverloadEnabled && simPower > OVERLOAD_POWER_THRESHOLD) {
    if (!overloadTimingActive) {
      // Start 3-second grace countdown
      overloadTimingActive = true;
      overloadStartTime = now;
      overloadCountdownSec = 3.0f;
      Serial.printf("[OVERLOAD] Power %.1fW exceeds 80W threshold! Starting 3s timer.\n", simPower);
    } else {
      unsigned long elapsed = now - overloadStartTime;
      if (elapsed < OVERLOAD_TIME_MS) {
        overloadCountdownSec = (OVERLOAD_TIME_MS - elapsed) / 1000.0f;
      } else {
        // 3 SECONDS ELAPSED: Automatic Load Shedding of Low Priority Load (Relay 2)
        overloadTimingActive = false;
        overloadCountdownSec = 0.0f;
        relay2State = false;
        applyRelayHardware(2, false);
        overloadTripCount++;

        alertMessage = "🚨 <b>OVERLOAD LOAD-SHEDDING TRIGGERED:</b> Total power exceeded 80W for 3 seconds. Socket 2 (Phone Charger) automatically turned OFF to protect system.";
        Serial.println("[LOAD-SHED] Relay 2 (Phone Charger) shed successfully. Relay 1 kept active.");
      }
    }
  } else {
    // Normal operation or power dropped back below 80W
    if (overloadTimingActive) {
      overloadTimingActive = false;
      overloadCountdownSec = 0.0f;
      Serial.println("[OVERLOAD] Power normalized below 80W. Timer cancelled.");
    }
  }
}

// ==============================================================================
//  WEB SERVER ROUTE HANDLERS
// ==============================================================================
void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleStatus() {
  // Construct JSON response without external libraries
  String json = "{";
  json += "\"relay1\":" + String(relay1State ? "true" : "false") + ",";
  json += "\"relay2\":" + String(relay2State ? "true" : "false") + ",";
  json += "\"auto_overload\":" + String(autoOverloadEnabled ? "true" : "false") + ",";
  json += "\"voltage\":" + String(simVoltage, 1) + ",";
  json += "\"current\":" + String(simCurrent, 3) + ",";
  json += "\"power\":" + String(simPower, 1) + ",";
  json += "\"pf\":" + String(simPF, 2) + ",";
  json += "\"frequency\":" + String(simFrequency, 1) + ",";
  json += "\"energy\":" + String(simEnergyKWh, 5) + ",";
  json += "\"overload_active\":" + String(overloadTimingActive ? "true" : "false") + ",";
  json += "\"countdown\":" + String(overloadCountdownSec, 1) + ",";
  json += "\"trip_count\":" + String(overloadTripCount) + ",";
  
  // Escape quotes in alertMessage
  String safeAlert = alertMessage;
  safeAlert.replace("\"", "\\\"");
  json += "\"alert\":\"" + safeAlert + "\",";
  
  bool isAP = (WiFi.getMode() == WIFI_AP);
  json += "\"wifi_mode\":\"" + String(isAP ? "SoftAP Hotspot" : "WiFi Station") + "\",";
  json += "\"ip\":\"" + (isAP ? WiFi.softAPIP().toString() : WiFi.localIP().toString()) + "\",";
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
      newState = (r == 1) ? !relay1State : !relay2State;
    }

    if (r == 1) {
      relay1State = newState;
      applyRelayHardware(1, relay1State);
    } else if (r == 2) {
      relay2State = newState;
      applyRelayHardware(2, relay2State);
    }
  }

  if (server.hasArg("auto")) {
    autoOverloadEnabled = (server.arg("auto").toInt() == 1);
    Serial.printf("[CONFIG] Auto-Overload Protection: %s\n", autoOverloadEnabled ? "ENABLED" : "DISABLED");
  }

  if (server.hasArg("reset_kwh")) {
    simEnergyKWh = 0.0f;
    Serial.println("[METER] Energy counter reset to 0.00000 kWh.");
  }

  if (server.hasArg("clear_alert")) {
    alertMessage = "";
  }

  // Return updated status immediately
  handleStatus();
}

void handleNotFound() {
  server.send(404, "text/plain", "404 Not Found");
}

// ==============================================================================
//  SETUP ROUTINE
// ==============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=======================================================");
  Serial.println("  SMART HOME EMS — ESP32 HARDWARE RELAY & SIMULATOR");
  Serial.println("=======================================================");

  // 1. Initialize Hardware Pins
  // IMPORTANT: Set initial output state to HIGH (OFF) before configuring pinMode
  // to prevent any relay clicks or power glitches during ESP32 boot!
  digitalWrite(PIN_RELAY_1, RELAY_ACTIVE_OFF);
  digitalWrite(PIN_RELAY_2, RELAY_ACTIVE_OFF);
  pinMode(PIN_RELAY_1, OUTPUT);
  pinMode(PIN_RELAY_2, OUTPUT);

  // Power the Optocoupler LEDs (GPIO 25 driven HIGH as 3.3V source)
  pinMode(PIN_OPTO_VCC, OUTPUT);
  digitalWrite(PIN_OPTO_VCC, HIGH);

  Serial.println("[HARDWARE] Relays initialized (Active-LOW: LOW=ON, HIGH=OFF)");
  Serial.printf("[HARDWARE] Relay 1 on GPIO %d (OFF)\n", PIN_RELAY_1);
  Serial.printf("[HARDWARE] Relay 2 on GPIO %d (OFF)\n", PIN_RELAY_2);
  Serial.printf("[HARDWARE] Opto VCC on GPIO %d (HIGH)\n", PIN_OPTO_VCC);

  // 2. Dual-Mode WiFi Connection
  Serial.printf("[WIFI] Attempting connection to SSID: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();
  bool connected = false;

  // Wait up to 10 seconds for WiFi connection
  while (millis() - startAttemptTime < 10000) {
    if (WiFi.status() == WL_CONNECTED) {
      connected = true;
      break;
    }
    delay(500);
    Serial.print(".");
  }

  if (connected) {
    Serial.println("\n[WIFI] Connected to local network!");
    Serial.print("[WIFI] Station IP Address: http://");
    Serial.println(WiFi.localIP());
  } else {
    // Fallback to SoftAP Hotspot
    Serial.println("\n[WIFI] Station connection timed out! Activating SoftAP hotspot fallback...");
    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD);

    Serial.print("[WIFI] SoftAP SSID: ");
    Serial.println(AP_SSID);
    Serial.print("[WIFI] SoftAP Password: ");
    Serial.println(AP_PASSWORD);
    Serial.print("[WIFI] SoftAP Dashboard URL: http://");
    Serial.println(WiFi.softAPIP());
  }

  // 3. Configure Web Server Routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/toggle", HTTP_GET, handleToggle);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("[HTTP] Web Server started on port 80");
  Serial.println("=======================================================\n");

  lastTelemetryUpdate = millis();
}

// ==============================================================================
//  MAIN LOOP
// ==============================================================================
void loop() {
  // 1. Process Web Server Requests
  server.handleClient();

  // 2. Periodic Telemetry Update & Overload Check (every 1 second)
  static unsigned long lastSecTick = 0;
  if (millis() - lastSecTick >= 1000) {
    lastSecTick = millis();
    updateTelemetry();
    checkOverloadProtection();
  }
}
