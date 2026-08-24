/**
 * Module 7: Production ESP32 Firmware — Dual-Core FreeRTOS
 *
 * Edge-hybrid safety architecture for the Smart Home EMS.
 * PZEM-004T v3.0 UART Edition
 *
 * CORE 0 (High Priority — SafetySamplingTask):
 *   - Continuous PZEM polling at 100ms intervals
 *   - Edge-local arc-fault proxy (dP/dt > 1000 W/s)
 *   - Dynamic inrush suppression via 5-sample sliding baseline
 *   - Immediate relay cutoff — zero network dependency
 *
 * CORE 1 (Standard Priority — Arduino loop):
 *   - Non-blocking MQTT client.loop()
 *   - 1Hz telemetry broadcast (plain float)
 *   - Incoming relay command handler (ON/OFF/WARNING)
 *   - Best-effort EDGE_ARC_FAULT alert publishing
 *
 * Shared Memory:
 *   portMUX_TYPE spinlock protects volatile float sharedPowerWatts.
 *   32-bit aligned float on Xtensa is atomic — spinlock avoids
 *   scheduler overhead and priority inversion of heavy semaphores.
 *
 * MQTT Topics (must match pipeline config.yaml):
 *   Publish:   home/sensor/{DEVICE_ID}/power   (plain float string)
 *   Subscribe: home/plug/{DEVICE_ID}/command    (ON/OFF/WARNING)
 *   Publish:   home/sensor/{DEVICE_ID}/status   (alerts)
 *   Publish:   home/plug/{DEVICE_ID}/ack        (relay confirmations)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <PZEM004Tv30.h>

// ═══════════════════════════════════════════════════════
//  CONFIGURATION — CHANGE THESE PER NODE
// ═══════════════════════════════════════════════════════
// Single aggregate sense point: ONE PZEM measures ONE IS 1293 6 A socket, and a
// 3-pin earthed multi-plug adapter in that socket puts the laptop brick and the
// phone charger behind the same shunt at the same time. That simultaneity is
// what NILM disaggregates -- one load at a time would be single-appliance
// classification, not disaggregation.
// See claude_debug/HARDWARE_FINAL_SPEC.md (scope S1/S2/S5, decision D12').
const char* DEVICE_ID      = "node_bench_agg";   // Unique per node
const char* ssid           = "YOUR_WIFI_SSID";
const char* password       = "YOUR_WIFI_PASSWORD";
const char* mqtt_server    = "192.168.1.100";     // EMS Backend IP
const char* mqtt_user      = "pipeline";          // Broker username (matches mosquitto.conf)
const char* mqtt_password  = "changeme_pipeline_password"; // Broker password
// 250 W prototype envelope -> CRITICAL_PCT 1.25 trips the relay at 312 W
// (1.36 A @ 230 V), which stays 3.7x below the 5 A load fuse so the relay always
// acts first. Coordination ladder: HARDWARE_FINAL_SPEC.md D8.
//
// Do NOT restore 600 W here. At 600 W the trip is 750 W = 3.26 A, and one socket
// carrying a laptop + phone charger draws only 150-320 W -- the cutoff could
// never fire, making the headline safety feature undemonstrable in hardware.
// That is blocking issue B-9; the 600 W ceiling belongs to the *simulated*
// `make demo` fleet in config/config.demo.yaml, which has no hardware.
const float RATED_WATTS    = 250.0;               // Rated power for this node
const float POWER_FACTOR   = 1.0;                 // Reference only; PZEM reports measured PF

// ═══════════════════════════════════════════════════════
//  HARDWARE PINS & CONSTANTS
// ═══════════════════════════════════════════════════════
const int   RELAY_PIN      = 18;
// NET polarity at the GPIO. Under HARDWARE_FINAL_SPEC.md D3' there are now ZERO
// inversions in the chain: the relay module jumper is set to H (high-trigger)
// and GPIO 18 drives the opto LED directly (~2.1 mA), so active-HIGH is the
// literal reading. A 100 kOhm IN->GND pull-down holds the input at 0 V while
// GPIO 18 is high-impedance during reset/boot, so OPEN is the only boot state.
//
// (The superseded D3 got to the same constant the long way round: an inverting
// BSS138 feeding an active-LOW module input, two inversions cancelling. The
// BSS138 is SOT-23 SMD and was deleted as unsolderable by hand -- B-10.)
//
// Setting this true energised the load at boot and made every safety cutoff
// CLOSE the relay. See HARDWARE_FINAL_SPEC.md D4/D5 (B-7). Do not reintroduce.
// Set true ONLY if the D5 purchase-contingency fallback was taken (module has
// no H/L jumper -> run module VCC at 3.3 V, JD-VCC at 5 V, and move the 100 kOhm
// to a pull-UP). Re-run bring-up Stage 2 and confirm OPEN-at-boot if you do.
const bool  RELAY_ACTIVE_LOW = false;
const int   PZEM_RX_PIN    = 16;
const int   PZEM_TX_PIN    = 17;
const float VOLTAGE        = 230.0;    // Mains voltage (India: 230V) for reference
const float CRITICAL_PCT   = 1.25;     // 125% of rated → hardware cutoff

// PZEM Instance
PZEM004Tv30 pzem(Serial2, PZEM_RX_PIN, PZEM_TX_PIN);

// ── Edge Arc-Fault Detection Constants ──
const float EDGE_ROC_THRESHOLD = 1000.0; // W/s — rapid dP/dt trip
const int   BASELINE_WINDOW    = 5;      // Sliding baseline sample count
const float BASELINE_INRUSH_CEIL = 50.0; // Baseline avg must be below this for inrush suppression
const float INRUSH_HEADROOM    = 100.0;  // Extra W above baseline avg to tolerate during inrush

// ── Anti-Thrashing Constants ──
const unsigned long SAFETY_LOCKOUT_MS = 300000;  // 5-minute relay lockout after safety trip

// ═══════════════════════════════════════════════════════
//  SHARED STATE (Core 0 ↔ Core 1)
// ═══════════════════════════════════════════════════════
portMUX_TYPE sharedMux = portMUX_INITIALIZER_UNLOCKED;

volatile float sharedPowerWatts  = 0.0;
volatile float sharedVoltage     = 230.0;
volatile float sharedCurrent     = 0.0;
volatile float sharedPf          = 1.0;
volatile bool  sharedArcFault    = false;
volatile float sharedArcFaultRoC = 0.0;

// ═══════════════════════════════════════════════════════
//  CORE 1 STATE (Arduino loop — not shared)
// ═══════════════════════════════════════════════════════
WiFiClient espClient;
PubSubClient client(espClient);

char activeDeviceId[32]    = "node_fridge";
bool relayLocked           = false;
unsigned long lockStartMs  = 0;
unsigned long lastMsgMs    = 0;
unsigned long lastTelemetryMs = 0;
unsigned long lastServerHB = 0;

char topicPower[64];
char topicTelemetry[64];
char topicCommand[64];
char topicStatus[64];
char topicAck[64];

// ═══════════════════════════════════════════════════════
//  RELAY HELPER
// ═══════════════════════════════════════════════════════
void setRelay(bool on) {
    if (RELAY_ACTIVE_LOW) {
        digitalWrite(RELAY_PIN, on ? LOW : HIGH);
    } else {
        digitalWrite(RELAY_PIN, on ? HIGH : LOW);
    }
}

// ═══════════════════════════════════════════════════════
//  CORE 0: HIGH-PRIORITY SAFETY SAMPLING TASK
// ═══════════════════════════════════════════════════════
void SafetySamplingTask(void* pvParameters) {
    float lastWatts = 0.0;
    float baselineRing[BASELINE_WINDOW];
    int   baselineIdx   = 0;
    int   baselineFill  = 0;
    for (int i = 0; i < BASELINE_WINDOW; i++) baselineRing[i] = 0.0;
    
    unsigned long lastReadMs = millis();

    for (;;) {
        float powerWatts = pzem.power();
        float pzemVoltage = pzem.voltage();
        float pzemCurrent = pzem.current();
        float pzemPf = pzem.pf();

        unsigned long nowMs = millis();
        float dt = (nowMs - lastReadMs) / 1000.0f; 
        lastReadMs = nowMs;

        if (isnan(powerWatts) || isnan(pzemVoltage) || isnan(pzemCurrent) || isnan(pzemPf)) {
            // Read failure, skip this cycle
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // ── Calculate historical sliding baseline average ──
        float baselineAvg = 0.0;
        if (baselineFill > 0) {
            for (int i = 0; i < baselineFill; i++) baselineAvg += baselineRing[i];
            baselineAvg /= (float)baselineFill;
        }

        // ── Inrush suppression flag ──
        // Declared at task scope: the dP/dt path below consults it, and it must
        // remain in scope afterwards. (It was previously declared inside the
        // dP/dt `if` block yet read by the overcurrent check below it, which
        // does not compile.)
        bool isNormalInrush = (baselineAvg < BASELINE_INRUSH_CEIL)
                           && (lastWatts < (baselineAvg + INRUSH_HEADROOM));

        // ── Edge Arc-Fault Proxy Detection (dP/dt in W/s) ──
        if (dt > 0.0f && (powerWatts > lastWatts)) {
            float rateOfChange = (powerWatts - lastWatts) / dt;

            if (rateOfChange > EDGE_ROC_THRESHOLD && !isNormalInrush) {
                // ⚡ IMMEDIATE PHYSICAL RELAY CUTOFF — NO NETWORK DEPENDENCY
                setRelay(false);

                taskENTER_CRITICAL(&sharedMux);
                sharedArcFault    = true;
                sharedArcFaultRoC = rateOfChange;
                taskEXIT_CRITICAL(&sharedMux);

                Serial.printf("[CORE0] ⚡ EDGE ARC-FAULT! dP/dt=%.0fW/s "
                              "(threshold: %.0f). Relay CUTOFF.\n",
                              rateOfChange, EDGE_ROC_THRESHOLD);
            }
        }

        // ── Overcurrent Cutoff (% of rated) ──
        // UNCONDITIONAL. Inrush suppression deliberately does NOT gate this
        // path: it exists to stop a motor's starting surge from reading as an
        // arc fault on the dP/dt channel, and under HARDWARE_FINAL_SPEC.md D11'
        // there is no inductive load left in scope at all (SMPS bricks plus a
        // resistive lamp). A sustained draw above 125% of rated must open the
        // relay on the very first sample that sees it -- D8 makes the relay the
        // functional protective element, ahead of the 5 A fuse at 3.7x margin.
        float criticalWatts = RATED_WATTS * CRITICAL_PCT;
        if (powerWatts > criticalWatts) {
            setRelay(false);
            Serial.printf("[CORE0] ⚡ OVERCURRENT! %.1fW > %.1fW. Relay CUTOFF.\n",
                          powerWatts, criticalWatts);
        }

        // ── Update sliding baseline ring buffer with current sample ──
        baselineRing[baselineIdx] = powerWatts;
        baselineIdx = (baselineIdx + 1) % BASELINE_WINDOW;
        if (baselineFill < BASELINE_WINDOW) baselineFill++;

        lastWatts = powerWatts;

        // ── Write shared measurements under spinlock ──
        taskENTER_CRITICAL(&sharedMux);
        sharedPowerWatts = powerWatts;
        sharedVoltage    = pzemVoltage;
        sharedCurrent    = pzemCurrent;
        sharedPf         = pzemPf;
        taskEXIT_CRITICAL(&sharedMux);

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// ═══════════════════════════════════════════════════════
//  MQTT CALLBACK — Relay Commands from Pipeline (Core 1)
// ═══════════════════════════════════════════════════════
void callback(char* topic, byte* payload, unsigned int length) {
    lastServerHB = millis();

    static const unsigned int MAX_MQTT_PAYLOAD = 256;
    if (length > MAX_MQTT_PAYLOAD) {
        Serial.printf("[MQTT] Payload too large (%u bytes), dropping.\n", length);
        return;
    }
    char msg[MAX_MQTT_PAYLOAD + 1];
    memcpy(msg, payload, length);
    msg[length] = '\0';
    String message = String(msg);

    if (String(topic) == String(topicCommand)) {
        if (message == "ON") {
            if (!relayLocked) {
                setRelay(true);
                Serial.println("[RELAY] ON via server command");
                client.publish(topicAck, "ON_CONFIRMED");
            } else {
                Serial.println("[RELAY] ON rejected — relay locked");
                client.publish(topicAck, "LOCKOUT_NACK");
            }
        } else if (message == "OFF") {
            setRelay(false);
            Serial.println("[RELAY] OFF via server command");
            client.publish(topicAck, "OFF_CONFIRMED");
        } else if (message == "WARNING") {
            Serial.println("[SAFETY] Warning received from server");
        }
    }
}

// ═══════════════════════════════════════════════════════
//  WIFI + MQTT SETUP (Core 1)
// ═══════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    
    // Initialize PZEM Serial
    Serial2.begin(9600, SERIAL_8N1, PZEM_RX_PIN, PZEM_TX_PIN);

    pinMode(RELAY_PIN, OUTPUT);
    setRelay(false); // Start with relay OFF for safety

    // ═══ Launch Core 0 Safety Sampling Task ═══
    // Launched BEFORE WiFi so safety works offline
    xTaskCreatePinnedToCore(
        SafetySamplingTask,   // Task function
        "SafetySampling",     // Name
        4096,                 // Stack size (bytes)
        NULL,                 // Parameters
        2,                    // Priority (higher than loop)
        NULL,                 // Task handle (not needed)
        0                     // Core 0
    );
    Serial.println("[INIT] Core 0: SafetySamplingTask launched (priority 2)");

    // Connect WiFi with timeout
    WiFi.setAutoReconnect(true);
    WiFi.begin(ssid, password);
    Serial.print("[WiFi] Connecting");
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - wifiStart < 30000)) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] Connection timeout. Proceeding offline.");
    }

    // Auto-provision Device ID if left blank or set to "auto"
    if (strcmp(DEVICE_ID, "") == 0 || strcmp(DEVICE_ID, "auto") == 0) {
        uint8_t mac[6];
        WiFi.macAddress(mac);
        snprintf(activeDeviceId, sizeof(activeDeviceId), "esp32_%02X%02X%02X", mac[3], mac[4], mac[5]);
    } else {
        strncpy(activeDeviceId, DEVICE_ID, sizeof(activeDeviceId) - 1);
        activeDeviceId[sizeof(activeDeviceId) - 1] = '\0';
    }

    // Build topic strings dynamically from activeDeviceId
    snprintf(topicPower,     sizeof(topicPower),     "home/sensor/%s/power",     activeDeviceId);
    snprintf(topicTelemetry, sizeof(topicTelemetry), "home/sensor/%s/telemetry", activeDeviceId);
    snprintf(topicCommand,   sizeof(topicCommand),   "home/plug/%s/command",     activeDeviceId);
    snprintf(topicStatus,    sizeof(topicStatus),    "home/sensor/%s/status",    activeDeviceId);
    snprintf(topicAck,       sizeof(topicAck),       "home/plug/%s/ack",         activeDeviceId);

    Serial.printf("[INIT] Device ID: %s\n", activeDeviceId);

    // Configure MQTT
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
    client.setKeepAlive(15);
    
    Serial.println("[INIT] Core 1: MQTT + Telemetry (Arduino loop)");
}

// ═══════════════════════════════════════════════════════
//  MQTT RECONNECT (Core 1)
// ═══════════════════════════════════════════════════════
unsigned long lastReconnectAttempt = 0;

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;
    
    if (millis() - lastReconnectAttempt < 5000) {
        return;
    }
    lastReconnectAttempt = millis();

    Serial.printf("[MQTT] Connecting as %s...\n", activeDeviceId);
    // Connect with auth credentials and Last Will & Testament (LWT)
    if (client.connect(activeDeviceId, mqtt_user, mqtt_password, topicStatus, 1, true, "OFFLINE")) {
        Serial.println("[MQTT] Connected");
        client.subscribe(topicCommand);
        // Announce online status
        client.publish(topicStatus, "ONLINE", true);
        lastServerHB = millis();
    } else {
        Serial.printf("[MQTT] Failed rc=%d, will retry in 5s\n", client.state());
    }
}

// ═══════════════════════════════════════════════════════
//  MAIN LOOP (Core 1 — MQTT + Telemetry)
// ═══════════════════════════════════════════════════════
void loop() {
    if (!client.connected()) {
        reconnectMQTT();
    }
    client.loop();

    // ── Read shared power under spinlock ──
    float powerWatts;
    taskENTER_CRITICAL(&sharedMux);
    powerWatts = sharedPowerWatts;
    taskEXIT_CRITICAL(&sharedMux);

    // ── 5-Minute Anti-Thrashing Lockout ──
    if (relayLocked && (millis() - lockStartMs > SAFETY_LOCKOUT_MS)) {
        relayLocked = false;
        Serial.println("[SAFETY] 5-minute lockout complete. Relay unlocked.");
    }

    // ── Check for edge arc-fault flag from Core 0 ──
    bool arcFaultTripped = false;
    float arcRoC = 0.0;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedArcFault) {
        arcFaultTripped = true;
        arcRoC = sharedArcFaultRoC;
        sharedArcFault = false;  // Acknowledge
    }
    taskEXIT_CRITICAL(&sharedMux);

    if (arcFaultTripped) {
        relayLocked = true;
        lockStartMs = millis();
        // Best-effort alert publish
        if (client.connected()) {
            char alertMsg[80];
            snprintf(alertMsg, sizeof(alertMsg),
                     "EDGE_ARC_FAULT:dP/dt=%.0fW/s", arcRoC);
            client.publish(topicStatus, alertMsg);
        }
    }

    // ── Overcurrent Alert Publishing ──
    float criticalWatts = RATED_WATTS * CRITICAL_PCT;
    if (powerWatts > criticalWatts && !relayLocked) {
        relayLocked = true;
        lockStartMs = millis();
        Serial.printf("[SAFETY] OVERCURRENT! %.1fW > %.1fW. Relay LOCKED.\n",
                      powerWatts, criticalWatts);
        if (client.connected()) {
            char alertMsg[64];
            snprintf(alertMsg, sizeof(alertMsg), "OVERCURRENT:%.1f", powerWatts);
            client.publish(topicStatus, alertMsg);
        }
    }

    // ── Server Heartbeat Watchdog ──
    if (millis() - lastServerHB > 30000 && lastServerHB > 0) {
        static unsigned long lastTimeoutLog = 0;
        if (millis() - lastTimeoutLog > 30000) {
            Serial.println("[WATCHDOG] No server heartbeat for 30s");
            if (client.connected()) {
                client.publish(topicStatus, "SERVER_TIMEOUT");
            }
            lastTimeoutLog = millis();
        }
    }

    // ── Publish Fast Power at 1Hz (plain float for NILM transient detection) ──
    if (millis() - lastMsgMs > 1000) {
        lastMsgMs = millis();
        char payload[16];
        dtostrf(powerWatts, 6, 2, payload);
        client.publish(topicPower, payload);
    }

    // ── Publish Rich Electrical Diagnostics at 0.1Hz (every 10s JSON) ──
    if (millis() - lastTelemetryMs > 10000) {
        lastTelemetryMs = millis();
        float v, i, pf;
        taskENTER_CRITICAL(&sharedMux);
        v  = sharedVoltage;
        i  = sharedCurrent;
        pf = sharedPf;
        taskEXIT_CRITICAL(&sharedMux);

        char jsonPayload[128];
        snprintf(jsonPayload, sizeof(jsonPayload),
                 "{\"v\":%.1f,\"i\":%.2f,\"w\":%.1f,\"pf\":%.2f}",
                 v, i, powerWatts, pf);
        client.publish(topicTelemetry, jsonPayload);
    }
}
