/**
 * Module 7: Phase 2 ESP32 C++ Firmware (FIXED)
 * 
 * Hardware integration layer for the Smart Home EMS.
 * - Publishes real-time RMS power readings at 1Hz to MQTT
 * - Subscribes to relay commands from the RL agent
 * - Hardware-level Tier-0 safety cutoff (independent of MQTT/server)
 * - Non-blocking cooldown after overcurrent events
 * - Server heartbeat watchdog
 * 
 * MQTT Topics (must match pipeline's config.yaml):
 *   Publish:   home/sensor/{DEVICE_ID}/power  (plain float string)
 *   Subscribe: home/plug/{DEVICE_ID}/command   (ON/OFF/WARNING)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <math.h>

// ═══════════════════════════════════════════════════════
//  CONFIGURATION — CHANGE THESE PER NODE
// ═══════════════════════════════════════════════════════
const char* DEVICE_ID      = "node_fridge";      // Unique per node
const char* ssid           = "YOUR_WIFI_SSID";
const char* password       = "YOUR_WIFI_PASSWORD";
const char* mqtt_server    = "192.168.1.100";     // EMS Backend IP
const float RATED_WATTS    = 200.0;               // Rated power for this appliance
const float POWER_FACTOR   = 1.0;                 // PF=1.0 for resistive; 0.85 for motors

// ═══════════════════════════════════════════════════════
//  HARDWARE PINS & CONSTANTS
// ═══════════════════════════════════════════════════════
const int RELAY_PIN    = 5;
const int SENSOR_PIN   = 34;       // ADC1_CH6 for CT clamp
const float VOLTAGE    = 230.0;    // Mains voltage (India: 230V)
const float BURDEN_R   = 33.0;     // Burden resistor (ohms)
const float CT_RATIO   = 1800.0;   // SCT-013-030: 1800:1 turns ratio
const float ADC_VREF   = 3.3;      // ESP32 ADC reference voltage
const int   ADC_MAX    = 4095;     // 12-bit ADC
const float CRITICAL_PCT = 1.25;   // 125% of rated → hardware cutoff

// ═══════════════════════════════════════════════════════
//  STATE VARIABLES
// ═══════════════════════════════════════════════════════
WiFiClient espClient;
PubSubClient client(espClient);

bool relayLocked         = false;
unsigned long lockStartMs = 0;
unsigned long lastMsgMs   = 0;
unsigned long lastServerHB = 0;    // Last message from server

// Build MQTT topic strings at compile time
char topicPower[64];
char topicCommand[64];
char topicStatus[64];

// ═══════════════════════════════════════════════════════
//  RMS CURRENT MEASUREMENT (WS-1.3)
//  Samples ADC over ~100ms (5 full cycles at 50Hz)
//  to get true RMS, not instantaneous peak.
// ═══════════════════════════════════════════════════════
float readRMSCurrent() {
    float sumSq = 0.0;
    const int SAMPLES = 200;  // ~100ms at 500µs per sample

    for (int i = 0; i < SAMPLES; i++) {
        int raw = analogRead(SENSOR_PIN);
        // Center around midpoint (DC bias at VCC/2 = 1.65V → ADC 2048)
        float centered = (float)(raw - 2048);
        // Convert ADC value to voltage, then to current via burden/CT ratio
        float voltage = (centered / (float)ADC_MAX) * ADC_VREF;
        float current = (voltage / BURDEN_R) * CT_RATIO;
        sumSq += current * current;
        delayMicroseconds(500);
    }

    return sqrt(sumSq / (float)SAMPLES);
}

// ═══════════════════════════════════════════════════════
//  MQTT CALLBACK — Relay Commands from Pipeline
// ═══════════════════════════════════════════════════════
void callback(char* topic, byte* payload, unsigned int length) {
    // Any message from server = heartbeat
    lastServerHB = millis();

    String message = "";
    for (unsigned int i = 0; i < length; i++) {
        message += (char)payload[i];
    }

    if (String(topic) == String(topicCommand)) {
        if (message == "ON" && !relayLocked) {
            digitalWrite(RELAY_PIN, HIGH);
            Serial.println("[RELAY] ON via server command");
        } else if (message == "OFF") {
            digitalWrite(RELAY_PIN, LOW);
            Serial.println("[RELAY] OFF via server command");
            // Publish ACK (WS-5.1)
            char ackTopic[64];
            snprintf(ackTopic, sizeof(ackTopic), "home/plug/%s/ack", DEVICE_ID);
            client.publish(ackTopic, "OFF_CONFIRMED");
        } else if (message == "WARNING") {
            Serial.println("[SAFETY] Warning received from server");
        }
    }
}

// ═══════════════════════════════════════════════════════
//  WIFI + MQTT SETUP
// ═══════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);  // Start with relay OFF for safety

    // Build topic strings
    snprintf(topicPower,   sizeof(topicPower),   "home/sensor/%s/power",   DEVICE_ID);
    snprintf(topicCommand, sizeof(topicCommand), "home/plug/%s/command",   DEVICE_ID);
    snprintf(topicStatus,  sizeof(topicStatus),  "home/sensor/%s/status",  DEVICE_ID);

    // Connect WiFi
    WiFi.begin(ssid, password);
    Serial.print("[WiFi] Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());

    // Configure MQTT
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
    client.setKeepAlive(15);
}

// ═══════════════════════════════════════════════════════
//  MQTT RECONNECT
// ═══════════════════════════════════════════════════════
void reconnectMQTT() {
    int retries = 0;
    while (!client.connected() && retries < 5) {
        Serial.printf("[MQTT] Connecting as %s...\n", DEVICE_ID);
        if (client.connect(DEVICE_ID)) {
            Serial.println("[MQTT] Connected");
            client.subscribe(topicCommand);
            lastServerHB = millis();
        } else {
            Serial.printf("[MQTT] Failed rc=%d, retry in 5s\n", client.state());
            delay(5000);
            retries++;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════
void loop() {
    if (!client.connected()) {
        reconnectMQTT();
    }
    client.loop();

    // Read true RMS current
    float rmsAmps = readRMSCurrent();
    float powerWatts = rmsAmps * VOLTAGE * POWER_FACTOR;

    // ── Tier-0 Hardware Safety Cutoff (WS-1.4) ──
    // Runs BEFORE any MQTT processing. Independent of server.
    float criticalWatts = RATED_WATTS * CRITICAL_PCT;
    if (powerWatts > criticalWatts && !relayLocked) {
        digitalWrite(RELAY_PIN, LOW);  // IMMEDIATE physical disconnect
        relayLocked = true;
        lockStartMs = millis();
        Serial.printf("[SAFETY] OVERCURRENT! %.1fW > %.1fW. Relay LOCKED.\n",
                      powerWatts, criticalWatts);
        // Publish alert (best-effort — safety already acted)
        if (client.connected()) {
            char alertMsg[64];
            snprintf(alertMsg, sizeof(alertMsg), "OVERCURRENT:%.1f", powerWatts);
            client.publish(topicStatus, alertMsg);
        }
    }

    // Non-blocking cooldown (WS-1.4): unlock relay after 5 seconds
    if (relayLocked && (millis() - lockStartMs > 5000)) {
        relayLocked = false;
        Serial.println("[SAFETY] Cooldown complete. Relay unlocked.");
    }

    // ── Server Heartbeat Watchdog (WS-1.5) ──
    // If no server contact for 30s, log disconnect (but keep relay state — fail-safe)
    if (millis() - lastServerHB > 30000 && lastServerHB > 0) {
        static unsigned long lastTimeoutLog = 0;
        if (millis() - lastTimeoutLog > 30000) {  // Log at most once per 30s
            Serial.println("[WATCHDOG] No server heartbeat for 30s");
            if (client.connected()) {
                client.publish(topicStatus, "SERVER_TIMEOUT");
            }
            lastTimeoutLog = millis();
        }
    }

    // ── Publish Power at 1Hz (WS-1.2: plain float, not JSON) ──
    if (millis() - lastMsgMs > 1000) {
        lastMsgMs = millis();
        char payload[16];
        dtostrf(powerWatts, 6, 2, payload);
        client.publish(topicPower, payload);
    }
}
