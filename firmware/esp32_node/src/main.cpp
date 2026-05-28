/**
 * Module 7: Production ESP32 Firmware — Dual-Core FreeRTOS
 *
 * Edge-hybrid safety architecture for the Smart Home EMS.
 *
 * CORE 0 (High Priority — SafetySamplingTask):
 *   - Continuous ADC sampling at ~500µs intervals
 *   - True RMS computed on 20ms AC wave-cycle boundaries (50Hz India grid)
 *   - 100ms windows (5 full cycles) for published power value
 *   - Edge-local arc-fault proxy (dP/dt > 800 W/cycle)
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

// ── Edge Arc-Fault Detection Constants ──
const float EDGE_ROC_THRESHOLD = 800.0;  // W/cycle — rapid dP/dt trip
const int   BASELINE_WINDOW    = 5;      // Sliding baseline sample count
const float BASELINE_INRUSH_CEIL = 50.0; // Baseline avg must be below this for inrush suppression
const float INRUSH_HEADROOM    = 100.0;  // Extra W above baseline avg to tolerate during inrush

// ── Anti-Thrashing Constants ──
const unsigned long SAFETY_LOCKOUT_MS = 300000;  // 5-minute relay lockout after safety trip

// ═══════════════════════════════════════════════════════
//  SHARED STATE (Core 0 ↔ Core 1)
//  portMUX spinlock: lightweight, no scheduler overhead.
//  volatile ensures compiler doesn't optimize away reads.
// ═══════════════════════════════════════════════════════
portMUX_TYPE sharedMux = portMUX_INITIALIZER_UNLOCKED;

volatile float sharedPowerWatts  = 0.0;
volatile bool  sharedArcFault    = false;   // Flag from Core 0 → Core 1 for alert publishing
volatile float sharedArcFaultRoC = 0.0;     // The dP/dt value that triggered the trip

// ═══════════════════════════════════════════════════════
//  CORE 1 STATE (Arduino loop — not shared)
// ═══════════════════════════════════════════════════════
WiFiClient espClient;
PubSubClient client(espClient);

bool relayLocked           = false;
unsigned long lockStartMs  = 0;
unsigned long lastMsgMs    = 0;
unsigned long lastServerHB = 0;

char topicPower[64];
char topicCommand[64];
char topicStatus[64];

// ═══════════════════════════════════════════════════════
//  CORE 0: HIGH-PRIORITY SAFETY SAMPLING TASK
//
//  Runs pinned to Core 0 at priority 2.
//  Continuously samples ADC, computes true RMS on 20ms
//  AC wave-cycle boundaries (50Hz), and evaluates
//  edge-local arc-fault proxy with dynamic inrush
//  suppression via a sliding baseline average.
// ═══════════════════════════════════════════════════════
void SafetySamplingTask(void* pvParameters) {
    // ── Per-cycle accumulation state ──
    float sumSqCycle     = 0.0;
    int   samplesCycle   = 0;
    unsigned long cycleStartUs = micros();

    // ── Multi-cycle RMS accumulation (5 cycles = 100ms) ──
    float sumSqWindow    = 0.0;
    int   samplesWindow  = 0;
    int   completedCycles = 0;
    const int CYCLES_PER_WINDOW = 5;  // 5 × 20ms = 100ms

    // ── Arc-fault state ──
    float lastWatts = 0.0;
    float baselineRing[BASELINE_WINDOW];
    int   baselineIdx   = 0;
    int   baselineFill  = 0;
    for (int i = 0; i < BASELINE_WINDOW; i++) baselineRing[i] = 0.0;

    for (;;) {
        // ── Sample ADC ──
        int raw = analogRead(SENSOR_PIN);
        float centered = (float)(raw - 2048);
        float voltSensor = (centered / (float)ADC_MAX) * ADC_VREF;
        float current = (voltSensor / BURDEN_R) * CT_RATIO;
        sumSqCycle += current * current;
        samplesCycle++;

        // ── Check for 20ms AC wave-cycle boundary (50Hz) ──
        unsigned long nowUs = micros();
        unsigned long elapsedUs = nowUs - cycleStartUs;

        if (elapsedUs >= 20000) {  // 20ms = one full 50Hz cycle
            // Accumulate this cycle into the multi-cycle window
            sumSqWindow  += sumSqCycle;
            samplesWindow += samplesCycle;
            completedCycles++;

            // Reset per-cycle accumulators
            sumSqCycle   = 0.0;
            samplesCycle = 0;
            cycleStartUs = nowUs;

            // ── After 5 complete cycles (100ms), compute final RMS ──
            if (completedCycles >= CYCLES_PER_WINDOW) {
                float rmsAmps = sqrt(sumSqWindow / (float)samplesWindow);
                float powerWatts = rmsAmps * VOLTAGE * POWER_FACTOR;

                // ── Update sliding baseline ──
                baselineRing[baselineIdx] = powerWatts;
                baselineIdx = (baselineIdx + 1) % BASELINE_WINDOW;
                if (baselineFill < BASELINE_WINDOW) baselineFill++;

                float baselineAvg = 0.0;
                for (int i = 0; i < baselineFill; i++) baselineAvg += baselineRing[i];
                baselineAvg /= (float)baselineFill;

                // ── Edge Arc-Fault Proxy Detection (dP/dt) ──
                float rateOfChange = fabs(powerWatts - lastWatts);

                // Dynamic inrush suppression:
                // Only suppress if the baseline is genuinely low (appliance cold-start)
                // AND the previous reading was within normal inrush headroom of baseline.
                // If a 150W cooler is already running, baseline > 50W → inrush check fails
                // → arc-fault detection remains armed (correct behavior).
                bool isNormalInrush = (baselineAvg < BASELINE_INRUSH_CEIL)
                                   && (lastWatts < (baselineAvg + INRUSH_HEADROOM));

                if (rateOfChange > EDGE_ROC_THRESHOLD && !isNormalInrush) {
                    // ⚡ IMMEDIATE PHYSICAL RELAY CUTOFF — NO NETWORK DEPENDENCY
                    digitalWrite(RELAY_PIN, LOW);

                    // Signal Core 1 to publish alert (best-effort)
                    taskENTER_CRITICAL(&sharedMux);
                    sharedArcFault    = true;
                    sharedArcFaultRoC = rateOfChange;
                    taskEXIT_CRITICAL(&sharedMux);

                    Serial.printf("[CORE0] ⚡ EDGE ARC-FAULT! dP/dt=%.0fW/cycle "
                                  "(threshold: %.0f). Relay CUTOFF.\n",
                                  rateOfChange, EDGE_ROC_THRESHOLD);
                }

                // ── Overcurrent Cutoff (% of rated) ──
                float criticalWatts = RATED_WATTS * CRITICAL_PCT;
                if (powerWatts > criticalWatts) {
                    digitalWrite(RELAY_PIN, LOW);
                    Serial.printf("[CORE0] ⚡ OVERCURRENT! %.1fW > %.1fW. Relay CUTOFF.\n",
                                  powerWatts, criticalWatts);
                }

                lastWatts = powerWatts;

                // ── Write shared power under spinlock ──
                taskENTER_CRITICAL(&sharedMux);
                sharedPowerWatts = powerWatts;
                taskEXIT_CRITICAL(&sharedMux);

                // Reset multi-cycle accumulators
                sumSqWindow     = 0.0;
                samplesWindow   = 0;
                completedCycles = 0;
            }
        }

        // ~500µs between samples (same as original, but non-blocking via vTaskDelay)
        // Using 1 tick at configTICK_RATE_HZ=1000 gives ~1ms granularity;
        // for sub-ms we use delayMicroseconds since this task owns Core 0.
        delayMicroseconds(500);
    }
}

// ═══════════════════════════════════════════════════════
//  MQTT CALLBACK — Relay Commands from Pipeline (Core 1)
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
//  WIFI + MQTT SETUP (Core 1)
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

    // ═══ Launch Core 0 Safety Sampling Task ═══
    // Priority 2 (above default Arduino loop priority 1)
    // Stack size 4096 bytes, pinned to Core 0
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
    Serial.println("[INIT] Core 1: MQTT + Telemetry (Arduino loop)");
}

// ═══════════════════════════════════════════════════════
//  MQTT RECONNECT (Core 1)
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

    // ── 5-Minute Anti-Thrashing Lockout (upgraded from 5s) ──
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
        // Best-effort alert publish (safety already acted on Core 0)
        if (client.connected()) {
            char alertMsg[80];
            snprintf(alertMsg, sizeof(alertMsg),
                     "EDGE_ARC_FAULT:dP/dt=%.0fW/cycle", arcRoC);
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

    // ── Publish Power at 1Hz (plain float, not JSON) ──
    if (millis() - lastMsgMs > 1000) {
        lastMsgMs = millis();
        char payload[16];
        dtostrf(powerWatts, 6, 2, payload);
        client.publish(topicPower, payload);
    }
}
