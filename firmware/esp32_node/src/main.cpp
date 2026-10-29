/**
 * Module 7: Production ESP32 Firmware — Dual-Core FreeRTOS
 *
 * Edge-hybrid safety architecture for the Smart Home EMS.
 * PZEM-004T v4.0 (100A split-core CT) UART Edition
 * 4-Channel Active-LOW Opto-Isolated Relay Module
 *
 * CORE 0 (High Priority — SafetySamplingTask):
 *   - Continuous PZEM polling at 100ms intervals over Hardware UART2
 *   - Edge-local arc-fault proxy (dP/dt > 1000 W/s)
 *   - Dynamic inrush suppression via 5-sample sliding baseline
 *   - Unconditional overcurrent, overvoltage & undervoltage protection
 *   - Hardware task watchdog feeding (5-second timeout)
 *   - PZEM loss-of-measurement watchdog (bounded successful-read age -> cutoff)
 *   - Immediate 4-channel relay cutoff — zero network dependency
 *
 * CORE 1 (Standard Priority — Arduino loop):
 *   - Non-blocking MQTT client.loop() and reconnect
 *   - Hardware task watchdog feeding (5-second timeout)
 *   - 1Hz fast power broadcast (plain float string)
 *   - 1Hz/10s rich electrical diagnostic telemetry (JSON object with canonical & shorthand keys)
 *   - Incoming relay command handler (ON/OFF/WARNING/reset_safety)
 *   - Safety trip latch consumption -> lockout & MQTT status alert publishing
 *
 * MQTT Topics (must match pipeline config.yaml):
 *   Publish:   home/sensor/{DEVICE_ID}/power       (plain float string)
 *   Publish:   home/sensor/{DEVICE_ID}/telemetry   (rich JSON diagnostics)
 *   Subscribe: home/plug/{DEVICE_ID}/command       (ON/OFF/WARNING/reset_safety)
 *   Publish:   home/sensor/{DEVICE_ID}/status      (alerts and lifecycle)
 *   Publish:   home/plug/{DEVICE_ID}/ack           (relay confirmations)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include <esp_task_wdt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <PZEM004Tv30.h>

#if !__has_include("secrets.h")
#error "Missing firmware/esp32_node/include/secrets.h - copy secrets.h.example to secrets.h and fill in credentials."
#endif
#include "secrets.h"
#include "hardware_config.h"

// ═══════════════════════════════════════════════════════
//  CONFIGURATION — set these in include/secrets.h
// ═══════════════════════════════════════════════════════
const char* DEVICE_ID      = EMS_DEVICE_ID;
const char* ssid           = EMS_WIFI_SSID;
const char* password       = EMS_WIFI_PASSWORD;
const char* mqtt_server    = EMS_MQTT_SERVER;
const int   mqtt_port      = EMS_MQTT_PORT;
const char* mqtt_user      = EMS_MQTT_USER;
const char* mqtt_password  = EMS_MQTT_PASSWORD;

const float RATED_WATTS    = ems_hardware::RATED_WATTS;
const float POWER_FACTOR   = ems_hardware::POWER_FACTOR;

// ═══════════════════════════════════════════════════════
//  HARDWARE PINS & CONSTANTS
// ═══════════════════════════════════════════════════════
const int   RELAY_PIN        = ems_hardware::RELAY_PIN;
const bool  RELAY_ACTIVE_LOW = ems_hardware::RELAY_ACTIVE_LOW;
// const int   PZEM_RX_PIN      = 16;
// const int   PZEM_TX_PIN      = 17;
const float VOLTAGE          = ems_hardware::MAINS_VOLTAGE_V;
const float CRITICAL_PCT     = ems_hardware::CRITICAL_PCT;

// 4-Channel Active-LOW Relay Pins (CH4 remapped to GPIO 32 to eliminate MTDI strapping pin hazard)
const int RELAY_PINS[4] = {RELAY_CH1_PIN, RELAY_CH2_PIN, RELAY_CH3_PIN, RELAY_CH4_PIN};

// PZEM Instance on Hardware UART2 (PZEM_SERIAL = Serial2, RX: GPIO 16, TX: GPIO 17)
PZEM004Tv30 pzem(PZEM_SERIAL, PZEM_RX_PIN, PZEM_TX_PIN);

// Edge Arc-Fault & Safety Constants
const float EDGE_ROC_THRESHOLD      = ems_hardware::EDGE_ROC_THRESHOLD_W_PER_S;
const int   BASELINE_WINDOW         = ems_hardware::BASELINE_WINDOW;
const float BASELINE_INRUSH_CEIL    = ems_hardware::BASELINE_INRUSH_CEIL_W;
const float INRUSH_HEADROOM         = ems_hardware::INRUSH_HEADROOM_W;
const unsigned long SAFETY_LOCKOUT_MS = ems_hardware::SAFETY_LOCKOUT_MS;

// PZEM Loss-of-Measurement Watchdog
const int PZEM_FAIL_TRIP_COUNT      = ems_hardware::PZEM_FAIL_TRIP_COUNT;
const unsigned long PZEM_MAX_BLIND_MS = ems_hardware::PZEM_MAX_BLIND_MS;
const float PZEM_MIN_VALID_VOLTAGE_V  = ems_hardware::PZEM_MIN_VALID_VOLTAGE_V;
const float PZEM_MAX_VALID_VOLTAGE_V  = ems_hardware::PZEM_MAX_VALID_VOLTAGE_V;
const float PZEM_MAX_VALID_CURRENT_A  = ems_hardware::PZEM_MAX_VALID_CURRENT_A;
const float PZEM_MAX_VALID_POWER_W    = ems_hardware::PZEM_MAX_VALID_POWER_W;

// ═══════════════════════════════════════════════════════
//  SHARED STATE (Core 0 <-> Core 1)
// ═══════════════════════════════════════════════════════
portMUX_TYPE sharedMux = portMUX_INITIALIZER_UNLOCKED;

volatile float sharedPowerWatts      = 0.0f;
volatile float sharedVoltage         = ems_hardware::MAINS_VOLTAGE_V;
volatile float sharedCurrent         = 0.0f;
volatile float sharedPf              = ems_hardware::POWER_FACTOR;
volatile float sharedEnergy          = 0.0f;
volatile float sharedFrequency       = ems_hardware::MAINS_FREQUENCY_HZ;
volatile float sharedApparentPowerVa = 0.0f;
volatile float sharedReactivePowerVar = 0.0f;

volatile bool  sharedArcFault        = false;
volatile float sharedArcFaultRoC     = 0.0f;
volatile bool  sharedOvercurrentLatch = false;
volatile bool  sharedOvervoltageLatch = false;
volatile bool  sharedUndervoltageLatch = false;
volatile bool  sharedPzemFault       = false;

// Autonomous safety trip latch (must be cleared explicitly by reset_safety command or lockout)
volatile bool  safetyTripped         = false;

// Command queues and gates
volatile bool  pendingRelayOn        = false;
volatile bool  pendingRelayOff       = false;
volatile bool  pendingAck            = false;
volatile uint8_t pendingAckType      = 0; // 0=none, 1=ON_CONFIRMED, 2=OFF_CONFIRMED, 3=LOCKOUT_NACK
volatile bool  safetyInhibit         = false;
volatile bool  measurementFresh      = false;
volatile unsigned long lastSuccessfulPzemReadMs = 0;
volatile bool  safetyTaskRunning     = false;
TaskHandle_t   safetyTaskHandle      = nullptr;

volatile bool  relayClosed           = false;

// ═══════════════════════════════════════════════════════
//  CORE 1 STATE (Arduino loop — not shared)
// ═══════════════════════════════════════════════════════
WiFiClient espClient;
PubSubClient client(espClient);

char activeDeviceId[32]    = "node_bench_agg";
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
//  RELAY HELPER (Controls all 4 opto-isolated channels)
// ═══════════════════════════════════════════════════════
void setRelay(bool on) {
    uint8_t pinState = on ? RELAY_ON : RELAY_OFF;
    for (int i = 0; i < 4; i++) {
        digitalWrite(RELAY_PINS[i], pinState);
    }
#ifdef RELAY_PIN
    if (RELAY_ACTIVE_LOW) {
        digitalWrite(RELAY_PIN, on ? LOW : HIGH);
    } else {
        digitalWrite(RELAY_PIN, on ? HIGH : LOW);
    }
#endif
    relayClosed = on;
}

// ═══════════════════════════════════════════════════════
//  CORE 0: HIGH-PRIORITY SAFETY SAMPLING TASK
// ═══════════════════════════════════════════════════════
void SafetySamplingTask(void* pvParameters) {
    safetyTaskRunning = true;
    float lastWatts = 0.0f;
    float baselineRing[BASELINE_WINDOW];
    int   baselineIdx   = 0;
    int   baselineFill  = 0;
    int   pzemFailCount = 0;
    bool  pzemEverValid = false;
    for (int i = 0; i < BASELINE_WINDOW; i++) baselineRing[i] = 0.0f;

    unsigned long lastReadMs = millis();

    for (;;) {
        // Feed task watchdog (must run inside task to prevent hardware watchdog trigger)
        esp_task_wdt_reset();

        float powerWatts  = pzem.power();
        float pzemVoltage = pzem.voltage();
        float pzemCurrent = pzem.current();
        float pzemPf      = pzem.pf();
        float pzemEnergy  = pzem.energy();
        float pzemFreq    = pzem.frequency();

        unsigned long nowMs = millis();
        float dt = (nowMs - lastReadMs) / 1000.0f;
        lastReadMs = nowMs;

        // ── Validate PZEM readings with isfinite() and range checks ──
        if (!isfinite(powerWatts) || !isfinite(pzemVoltage)
            || !isfinite(pzemCurrent) || !isfinite(pzemPf)
            || pzemVoltage < PZEM_MIN_VALID_VOLTAGE_V
            || pzemVoltage > PZEM_MAX_VALID_VOLTAGE_V
            || pzemCurrent < 0.0f
            || pzemCurrent > PZEM_MAX_VALID_CURRENT_A
            || powerWatts < 0.0f
            || powerWatts > PZEM_MAX_VALID_POWER_W
            || pzemPf < 0.0f || pzemPf > 1.0f) {
            
            measurementFresh = false;

            taskENTER_CRITICAL(&sharedMux);
            if (pendingRelayOn) {
                pendingRelayOn = false;
                pendingAck = true;
                pendingAckType = 3; // LOCKOUT_NACK
            }
            if (pendingRelayOff) {
                pendingRelayOff = false;
                setRelay(false);
                pendingAck = true;
                pendingAckType = 2; // OFF_CONFIRMED
            }
            taskEXIT_CRITICAL(&sharedMux);

            if (pzemFailCount < PZEM_FAIL_TRIP_COUNT) pzemFailCount++;

            const bool successfulReadExpired =
                lastSuccessfulPzemReadMs > 0
                && (unsigned long)(nowMs - lastSuccessfulPzemReadMs)
                   >= PZEM_MAX_BLIND_MS;

            if (pzemFailCount >= PZEM_FAIL_TRIP_COUNT && pzemEverValid
                    && relayClosed && successfulReadExpired) {
                taskENTER_CRITICAL(&sharedMux);
                safetyInhibit = true;
                safetyTripped = true;
                setRelay(false);
                sharedPzemFault = true;
                taskEXIT_CRITICAL(&sharedMux);

                Serial.printf("[CORE0] ⚠ PZEM FAULT! %d consecutive invalid "
                              "reads (%.1fs blind). Relay CUTOFF.\n",
                              pzemFailCount, pzemFailCount * 0.1f);
            }

            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        pzemFailCount = 0;
        pzemEverValid = true;
        measurementFresh = true;
        lastSuccessfulPzemReadMs = nowMs;

        // ── Compute derived electrical metrics safely ──
        float apparent_power_va = pzemVoltage * pzemCurrent;
        float reactive_power_var = sqrtf(fmaxf(0.0f, (apparent_power_va * apparent_power_va) - (powerWatts * powerWatts)));
        float safeEnergy = isfinite(pzemEnergy) ? pzemEnergy : 0.0f;
        float safeFreq   = isfinite(pzemFreq) ? pzemFreq : 50.0f;

        // ── Calculate historical sliding baseline average ──
        float baselineAvg = 0.0f;
        if (baselineFill > 0) {
            for (int i = 0; i < baselineFill; i++) baselineAvg += baselineRing[i];
            baselineAvg /= (float)baselineFill;
        }

        // ── Inrush suppression flag ──
        bool isNormalInrush = (baselineAvg < BASELINE_INRUSH_CEIL)
                           && (lastWatts < (baselineAvg + INRUSH_HEADROOM));

        // ── Edge Arc-Fault Proxy Detection (dP/dt in W/s) ──
        if (dt > 0.0f && (powerWatts > lastWatts)) {
            float rateOfChange = (powerWatts - lastWatts) / dt;

            if (rateOfChange > EDGE_ROC_THRESHOLD && !isNormalInrush) {
                taskENTER_CRITICAL(&sharedMux);
                safetyInhibit   = true;
                safetyTripped   = true;
                setRelay(false);
                sharedArcFault    = true;
                sharedArcFaultRoC = rateOfChange;
                taskEXIT_CRITICAL(&sharedMux);

                Serial.printf("[CORE0] ⚡ EDGE ARC-FAULT! dP/dt=%.0fW/s "
                              "(threshold: %.0f). Relay CUTOFF.\n",
                              rateOfChange, EDGE_ROC_THRESHOLD);
            }
        }

        // ── Autonomous Overcurrent Protection ──
        float criticalWatts = RATED_WATTS * CRITICAL_PCT;
        if (powerWatts > criticalWatts || pzemCurrent > MAX_SAFE_CURRENT_A) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            safetyTripped = true;
            setRelay(false);
            sharedOvercurrentLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ OVERCURRENT! %.1fW > %.1fW (%.2fA > %.1fA). Relay CUTOFF.\n",
                          powerWatts, criticalWatts, pzemCurrent, MAX_SAFE_CURRENT_A);
        }

        // ── Autonomous Overvoltage Protection ──
        if (pzemVoltage > MAX_SAFE_VOLTAGE_V) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            safetyTripped = true;
            setRelay(false);
            sharedOvervoltageLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ OVERVOLTAGE! %.1fV > %.1fV. Relay CUTOFF.\n",
                          pzemVoltage, MAX_SAFE_VOLTAGE_V);
        }

        // ── Autonomous Undervoltage Protection ──
        if (pzemVoltage < MIN_SAFE_VOLTAGE_V && pzemVoltage >= PZEM_MIN_VALID_VOLTAGE_V) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            safetyTripped = true;
            setRelay(false);
            sharedUndervoltageLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ UNDERVOLTAGE! %.1fV < %.1fV. Relay CUTOFF.\n",
                          pzemVoltage, MIN_SAFE_VOLTAGE_V);
        }

        // ── Update sliding baseline ring buffer with current sample ──
        baselineRing[baselineIdx] = powerWatts;
        baselineIdx = (baselineIdx + 1) % BASELINE_WINDOW;
        if (baselineFill < BASELINE_WINDOW) baselineFill++;

        lastWatts = powerWatts;

        // ── Write shared measurements under spinlock ──
        taskENTER_CRITICAL(&sharedMux);
        sharedPowerWatts       = powerWatts;
        sharedVoltage          = pzemVoltage;
        sharedCurrent          = pzemCurrent;
        sharedPf               = pzemPf;
        sharedEnergy           = safeEnergy;
        sharedFrequency        = safeFreq;
        sharedApparentPowerVa  = apparent_power_va;
        sharedReactivePowerVar = reactive_power_var;
        taskEXIT_CRITICAL(&sharedMux);

        // ── Consume queued relay requests (Core 0 is sole owner) ──
        taskENTER_CRITICAL(&sharedMux);
        if (pendingRelayOff) {
            pendingRelayOff = false;
            setRelay(false);
            pendingAck = true;
            pendingAckType = 2; // OFF_CONFIRMED
        } else if (pendingRelayOn) {
            pendingRelayOn = false;
            if (safetyTaskRunning && !safetyInhibit && measurementFresh && !relayLocked) {
                setRelay(true);
                pendingAck = true;
                pendingAckType = 1; // ON_CONFIRMED
            } else {
                pendingAck = true;
                pendingAckType = 3; // LOCKOUT_NACK
            }
        }
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
            taskENTER_CRITICAL(&sharedMux);
            pendingRelayOn = true;
            taskEXIT_CRITICAL(&sharedMux);
        } else if (message == "OFF") {
            taskENTER_CRITICAL(&sharedMux);
            pendingRelayOff = true;
            pendingRelayOn = false;
            taskEXIT_CRITICAL(&sharedMux);
        } else if (message == "reset_safety") {
            taskENTER_CRITICAL(&sharedMux);
            safetyTripped = false;
            safetyInhibit = false;
            relayLocked   = false;
            taskEXIT_CRITICAL(&sharedMux);
            Serial.println("[SAFETY] Safety reset command received. Lockout cleared.");
            if (client.connected()) {
                client.publish(topicStatus, "SAFETY_RESET_CONFIRMED");
            }
        } else if (message == "WARNING") {
            Serial.println("[SAFETY] Warning received from server");
        }
    }
}

// ═══════════════════════════════════════════════════════
//  SETUP (Core 1)
// ═══════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);

    // CRITICAL: Initialize all 4 relay pins to RELAY_OFF (HIGH) BEFORE enabling outputs
    // to prevent boot-time relay chatter
    for (int i = 0; i < 4; i++) {
        digitalWrite(RELAY_PINS[i], RELAY_OFF);
        pinMode(RELAY_PINS[i], OUTPUT);
    }
#ifdef RELAY_PIN
    digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);
    pinMode(RELAY_PIN, OUTPUT);
#endif
    setRelay(false);

    // Initialize PZEM Hardware Serial2 (9600 baud, 8N1, RX=16, TX=17)
    PZEM_SERIAL.begin(PZEM_UART_BAUD, SERIAL_8N1, PZEM_RX_PIN, PZEM_TX_PIN);

#if EMS_BENCH_MODE
    Serial.println("[BENCH] DRY MODE: mains and relay contacts must be disconnected");
    Serial.println("[BENCH] Commands: DRY_ON, DRY_OFF, STATUS");
    return;
#endif

    // Launch Core 0 Safety Sampling Task before WiFi for offline protection
    BaseType_t safetyTaskResult = xTaskCreatePinnedToCore(
        SafetySamplingTask,
        "SafetySampling",
        4096,
        NULL,
        2,
        &safetyTaskHandle,
        0
    );
    if (safetyTaskResult != pdPASS || safetyTaskHandle == nullptr) {
        safetyTaskRunning = false;
        safetyInhibit = true;
        setRelay(false);
        Serial.println("[INIT] SAFETY TASK CREATION FAILED — relay inhibited");
    } else {
        Serial.println("[INIT] Core 0: SafetySamplingTask launched (priority 2)");
    }

    // Connect WiFi with timeout
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);

    esp_task_wdt_reset();
    Serial.printf("[WiFi] Scanning for '%s'...\n", ssid);
    int scanCount = WiFi.scanNetworks();
    esp_task_wdt_reset();
    Serial.printf("[WiFi] Found %d APs:\n", scanCount);
    const char* matchedSsid = ssid;
    for (int i = 0; i < scanCount; i++) {
        String found = WiFi.SSID(i);
        Serial.printf("  - '%s' (RSSI: %d, Ch: %d)\n", found.c_str(), WiFi.RSSI(i), WiFi.channel(i));
        if (found.indexOf("Pramodsharma") >= 0 && found.indexOf("iPhone") >= 0) {
            matchedSsid = strdup(found.c_str());
            Serial.printf("  ==> Auto-matched iPhone AP: '%s'\n", matchedSsid);
        }
    }

    WiFi.setAutoReconnect(true);
    WiFi.begin(matchedSsid, password);
    Serial.print("[WiFi] Connecting");
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - wifiStart < 30000)) {
        esp_task_wdt_reset();
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

    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback);
    client.setKeepAlive(15);

    Serial.println("[INIT] Core 1: MQTT + Telemetry initialized");

    // Initialize 5-second hardware task watchdog for steady-state runtime
    esp_task_wdt_init(WATCHDOG_TIMEOUT_S, true);
    esp_task_wdt_add(NULL);
    if (safetyTaskHandle != nullptr) {
        esp_task_wdt_add(safetyTaskHandle);
    }
}

// ═══════════════════════════════════════════════════════
//  MQTT RECONNECT (Core 1 — Non-blocking)
// ═══════════════════════════════════════════════════════
unsigned long lastReconnectAttempt = 0;

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;
    
    if (millis() - lastReconnectAttempt < 5000) {
        return;
    }
    lastReconnectAttempt = millis();

    Serial.printf("[MQTT] Connecting as %s...\n", activeDeviceId);
    if (client.connect(activeDeviceId, mqtt_user, mqtt_password, topicStatus, 1, true, "OFFLINE")) {
        Serial.println("[MQTT] Connected");
        client.subscribe(topicCommand);
        client.publish(topicStatus, "ONLINE", true);
        lastServerHB = millis();
    } else {
        Serial.printf("[MQTT] Failed rc=%d, will retry in 5s\n", client.state());
    }
}

// ═══════════════════════════════════════════════════════
//  MAIN LOOP (Core 1)
// ═══════════════════════════════════════════════════════
void loop() {
    esp_task_wdt_reset();

#if EMS_BENCH_MODE
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "DRY_ON") {
            setRelay(true);
            Serial.println("[BENCH] DRY_ON: relay closed; verify COM-NO continuity");
        } else if (command == "DRY_OFF") {
            setRelay(false);
            Serial.println("[BENCH] DRY_OFF: relay open; verify COM-NO open");
        } else if (command == "STATUS") {
            Serial.printf("[BENCH] Relay=%s\n", relayClosed ? "ON" : "OFF");
        }
    }
    delay(10);
    return;
#endif

    if (!client.connected()) {
        reconnectMQTT();
    }
    client.loop();

    // Read shared measurements under spinlock
    float powerWatts, voltage, current, pf, energy, freq, va, var;
    bool relayState, measurementState;
    taskENTER_CRITICAL(&sharedMux);
    powerWatts       = sharedPowerWatts;
    voltage          = sharedVoltage;
    current          = sharedCurrent;
    pf               = sharedPf;
    energy           = sharedEnergy;
    freq             = sharedFrequency;
    va               = sharedApparentPowerVa;
    var              = sharedReactivePowerVar;
    relayState       = relayClosed;
    measurementState = measurementFresh;
    taskEXIT_CRITICAL(&sharedMux);

    // ── 5-Minute Anti-Thrashing Lockout ──
    if (relayLocked && (millis() - lockStartMs > SAFETY_LOCKOUT_MS)) {
        relayLocked = false;
        taskENTER_CRITICAL(&sharedMux);
        if (!safetyTripped) {
            safetyInhibit = false;
        }
        taskEXIT_CRITICAL(&sharedMux);
        Serial.println("[SAFETY] 5-minute lockout complete. Relay unlocked.");
    }

    // ── Check for Edge Arc-Fault from Core 0 ──
    bool arcFaultTripped = false;
    float arcRoC = 0.0f;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedArcFault) {
        arcFaultTripped = true;
        arcRoC = sharedArcFaultRoC;
        sharedArcFault = false;
    }
    taskEXIT_CRITICAL(&sharedMux);

    if (arcFaultTripped) {
        relayLocked = true;
        lockStartMs = millis();
        if (client.connected()) {
            char alertMsg[80];
            snprintf(alertMsg, sizeof(alertMsg), "EDGE_ARC_FAULT:dP/dt=%.0fW/s", arcRoC);
            client.publish(topicStatus, alertMsg);
        }
    }

    // ── Check for Overcurrent latch from Core 0 ──
    bool overcurrentTripped = false;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedOvercurrentLatch) {
        overcurrentTripped = true;
        sharedOvercurrentLatch = false;
    }
    taskEXIT_CRITICAL(&sharedMux);

    if (overcurrentTripped) {
        relayLocked = true;
        lockStartMs = millis();
        float criticalWatts = RATED_WATTS * CRITICAL_PCT;
        Serial.printf("[SAFETY] OVERCURRENT! %.1fW > %.1fW. Relay LOCKED.\n",
                      powerWatts, criticalWatts);
        if (client.connected()) {
            char alertMsg[64];
            snprintf(alertMsg, sizeof(alertMsg), "OVERCURRENT:%.1f", powerWatts);
            client.publish(topicStatus, alertMsg);
        }
    }

    // ── Check for Overvoltage / Undervoltage latches ──
    bool ovTripped = false, uvTripped = false;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedOvervoltageLatch) {
        ovTripped = true;
        sharedOvervoltageLatch = false;
    }
    if (sharedUndervoltageLatch) {
        uvTripped = true;
        sharedUndervoltageLatch = false;
    }
    taskEXIT_CRITICAL(&sharedMux);

    if (ovTripped) {
        relayLocked = true;
        lockStartMs = millis();
        Serial.printf("[SAFETY] OVERVOLTAGE! %.1fV. Relay LOCKED.\n", voltage);
        if (client.connected()) {
            char alertMsg[64];
            snprintf(alertMsg, sizeof(alertMsg), "OVERVOLTAGE:%.1f", voltage);
            client.publish(topicStatus, alertMsg);
        }
    }
    if (uvTripped) {
        relayLocked = true;
        lockStartMs = millis();
        Serial.printf("[SAFETY] UNDERVOLTAGE! %.1fV. Relay LOCKED.\n", voltage);
        if (client.connected()) {
            char alertMsg[64];
            snprintf(alertMsg, sizeof(alertMsg), "UNDERVOLTAGE:%.1f", voltage);
            client.publish(topicStatus, alertMsg);
        }
    }

    // ── Check for PZEM Loss-of-Measurement latch from Core 0 ──
    bool pzemFaultTripped = false;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedPzemFault) {
        pzemFaultTripped = true;
        sharedPzemFault = false;
    }
    taskEXIT_CRITICAL(&sharedMux);

    if (pzemFaultTripped) {
        relayLocked = true;
        lockStartMs = millis();
        Serial.println("[SAFETY] PZEM FAULT! No valid measurement. Relay LOCKED.");
        if (client.connected()) {
            client.publish(topicStatus, "PZEM_FAULT");
        }
    }

    // ── Publish acknowledgements ──
    bool ackReady = false;
    uint8_t ackType = 0;
    taskENTER_CRITICAL(&sharedMux);
    if (pendingAck) {
        ackReady = true;
        ackType = pendingAckType;
        pendingAck = false;
        pendingAckType = 0;
    }
    taskEXIT_CRITICAL(&sharedMux);
    if (ackReady && client.connected()) {
        const char* ackPayload = (ackType == 1) ? "ON_CONFIRMED"
                              : (ackType == 2) ? "OFF_CONFIRMED"
                                               : "LOCKOUT_NACK";
        client.publish(topicAck, ackPayload);
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
    if (millis() - lastMsgMs >= 1000) {
        lastMsgMs = millis();
        char payload[16];
        dtostrf(powerWatts, 6, 2, payload);
        client.publish(topicPower, payload);
        if (measurementState) {
            Serial.printf("[USB] V=%.1fV I=%.2fA P=%.1fW PF=%.2f VA=%.1f VAR=%.1f Relay=%s\n",
                          voltage, current, powerWatts, pf, va, var,
                          relayState ? "ON" : "OFF");
        } else {
            Serial.printf("[USB] PZEM=INVALID Relay=%s\n", relayState ? "ON" : "OFF");
        }
    }

    // ── Publish Rich Electrical Diagnostics at TELEMETRY_INTERVAL_MS (1000ms JSON) ──
    if (millis() - lastTelemetryMs >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryMs = millis();
        
        // Construct JSON telemetry payload containing BOTH canonical and shorthand keys
        StaticJsonDocument<384> doc;
        doc["v"] = roundf(voltage * 10.0f) / 10.0f;
        doc["i"] = roundf(current * 100.0f) / 100.0f;
        doc["w"] = roundf(powerWatts * 10.0f) / 10.0f;
        doc["pf"] = roundf(pf * 100.0f) / 100.0f;

        // Canonical contract keys
        doc["voltage_rms"] = doc["v"];
        doc["current_rms"] = doc["i"];
        doc["active_power_w"] = doc["w"];
        doc["apparent_power_va"] = roundf(va * 10.0f) / 10.0f;
        doc["reactive_power_var"] = roundf(var * 10.0f) / 10.0f;
        doc["power_factor"] = doc["pf"];
        doc["frequency_hz"] = roundf(freq * 10.0f) / 10.0f;
        doc["energy_kwh"] = roundf(energy * 1000.0f) / 1000.0f;

        // Relay channel array (4 channels)
        JsonArray relays = doc.createNestedArray("relays");
        relays.add(relayState ? 1 : 0);
        relays.add(relayState ? 1 : 0);
        relays.add(relayState ? 1 : 0);
        relays.add(relayState ? 1 : 0);

        char jsonBuffer[384];
        serializeJson(doc, jsonBuffer);
        client.publish(topicTelemetry, jsonBuffer);
    }
}
