/**
 * Module 7: Production ESP32 Firmware — Dual-Core FreeRTOS
 *
 * Edge-hybrid safety architecture for the Smart Home EMS.
 * ESP32 DevKit V1 (30-pin)
 * PZEM-004T-100A V4.0 (100A split-core CT) on UART2 (RX=16, TX=17, 9600 baud, 3.3V logic)
 * 2-Channel Active-LOW Opto-Isolated Relay Module:
 *   - Relay 1 (Laptop Charger) = GPIO 26
 *   - Relay 2 (Phone Charger)  = GPIO 27
 *   - Relay Opto VCC           = GPIO 25 (3.3V source driven HIGH)
 *   - Active-LOW: LOW = ON, HIGH = OFF
 *   - JD-VCC = VIN (5V)
 *
 * CORE 0 (High Priority — SafetySamplingTask):
 *   - PZEM polling at 1-second interval over Hardware UART2
 *   - Edge-local arc-fault proxy (dP/dt > 1000 W/s)
 *   - Dynamic inrush suppression via 5-sample sliding baseline
 *   - Unconditional overcurrent, overvoltage & undervoltage protection
 *   - Hardware task watchdog feeding (5-second timeout)
 *   - PZEM loss-of-measurement watchdog (default 5 consecutive failed reads)
 *   - Synthetic fallback telemetry generation when PZEM is offline or disabled
 *   - Immediate relay cutoff / safe state (allRelaysOff) — zero network dependency
 *
 * CORE 1 (Standard Priority — Arduino loop):
 *   - Non-blocking MQTT client.loop() and reconnect
 *   - Hardware task watchdog feeding (5-second timeout)
 *   - 1Hz fast power broadcast (plain float string)
 *   - 1Hz rich electrical diagnostic telemetry (JSON object with canonical & shorthand keys, source=real|synthetic)
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

// 2-Channel Active-LOW Relay Pins (Relay 1 = GPIO 26, Relay 2 = GPIO 27)
const int RELAY_PINS[2] = {RELAY1_PIN, RELAY2_PIN};

// PZEM Instance on Hardware UART2 (PZEM_SERIAL = Serial2, RX: GPIO 16, TX: GPIO 17, 9600 baud)
PZEM004Tv30 pzem(PZEM_SERIAL, PZEM_RX, PZEM_TX);

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
volatile bool  sharedSampleIsSynthetic = false;

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

volatile bool  relay1Closed          = false;
volatile bool  relay2Closed          = false;
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
//  RELAY CONTROL (3. Single function for active-LOW mapping)
// ═══════════════════════════════════════════════════════
void setRelay(uint8_t channel, bool on) {
    uint8_t pinState = on ? RELAY_ON : RELAY_OFF;
    if (channel == 1) {
        digitalWrite(RELAY1_PIN, pinState);
        relay1Closed = on;
    } else if (channel == 2) {
        digitalWrite(RELAY2_PIN, pinState);
        relay2Closed = on;
    }
    relayClosed = on;
    if (!relay1Closed && !relay2Closed) {
        relayClosed = false;
    }
}

// Convenience overload for controlling all relay channels
void setRelay(bool on) {
    setRelay(1, on);
    setRelay(2, on);
    relayClosed = on;
}

// 4. Safe-state function: writes HIGH (RELAY_OFF) to GPIO26 and GPIO27
void allRelaysOff() {
    setRelay(1, false);
    setRelay(2, false);
    relayClosed = false;
}

#if EMS_BENCH_MODE
void printPzemStatus() {
    float v = pzem.voltage();
    float i = pzem.current();
    float p = pzem.power();
    float pf = pzem.pf();
    if (!isfinite(v) || !isfinite(i) || !isfinite(p)) {
        Serial.println("[BENCH] PZEM status: OFFLINE / READ FAILED (normal when 230V AC mains is disconnected)");
    } else {
        Serial.printf("[BENCH] PZEM status: OK -> V=%.1fV, I=%.3fA, P=%.1fW, PF=%.2f\n", v, i, p, pf);
    }
}
#endif

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

    unsigned long lastPzemPollMs = 0;
    float syntheticEnergy = 0.0f;

    for (;;) {
        // Feed task watchdog (must run inside task to prevent hardware watchdog trigger)
        esp_task_wdt_reset();

        unsigned long nowMs = millis();

        // ── 5. Poll PZEM no faster than once per second (>= 1000 ms) ──
        if (lastPzemPollMs > 0 && (nowMs - lastPzemPollMs < 1000)) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        float dt = (lastPzemPollMs == 0) ? 1.0f : ((nowMs - lastPzemPollMs) / 1000.0f);
        lastPzemPollMs = nowMs;

        float powerWatts  = pzem.power();
        float pzemVoltage = pzem.voltage();
        float pzemCurrent = pzem.current();
        float pzemPf      = pzem.pf();
        float pzemEnergy  = pzem.energy();
        float pzemFreq    = pzem.frequency();

        bool isSyntheticSample = false;

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

            // ── 6. USE_SYNTHETIC_FALLBACK handling ──
            // If USE_SYNTHETIC_FALLBACK is enabled, or if PZEM never responds after initial tries
            bool fallbackActive = (USE_SYNTHETIC_FALLBACK != 0) || (!pzemEverValid && (pzemFailCount >= PZEM_FAIL_TRIP_COUNT));

            if (fallbackActive) {
                isSyntheticSample = true;
                pzemVoltage = 230.0f;
                pzemFreq = 50.0f;

                // Simulated load based on relay states:
                // Relay 1 (Laptop Charger) = 65.0W, Relay 2 (Phone Charger) = 20.0W
                float simWatts = 0.0f;
                if (relay1Closed) simWatts += 65.0f;
                if (relay2Closed) simWatts += 20.0f;

                powerWatts = simWatts;
                pzemPf = (simWatts > 0.0f) ? 0.95f : 1.0f;
                pzemCurrent = (simWatts > 0.0f) ? (simWatts / (pzemVoltage * pzemPf)) : 0.0f;
                syntheticEnergy += (simWatts * (dt / 3600.0f)) / 1000.0f; // kWh
                pzemEnergy = syntheticEnergy;
            } else {
                // Treat NaN or failed read as invalid (never as 0 W)
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
                    allRelaysOff();
                    pendingAck = true;
                    pendingAckType = 2; // OFF_CONFIRMED
                }
                taskEXIT_CRITICAL(&sharedMux);

                if (pzemFailCount < PZEM_FAIL_TRIP_COUNT) pzemFailCount++;

                const bool successfulReadExpired =
                    lastSuccessfulPzemReadMs > 0
                    && (unsigned long)(nowMs - lastSuccessfulPzemReadMs)
                       >= PZEM_MAX_BLIND_MS;

                // After N consecutive failures (default 5) log a warning and, if the safety layer requires it, call allRelaysOff()
                if (pzemFailCount >= PZEM_FAIL_TRIP_COUNT) {
                    Serial.printf("[CORE0] ⚠ Warning: PZEM read failed (%d consecutive failures)\n", pzemFailCount);
                }

                if (pzemFailCount >= PZEM_FAIL_TRIP_COUNT && pzemEverValid
                        && relayClosed && successfulReadExpired) {
                    taskENTER_CRITICAL(&sharedMux);
                    safetyInhibit = true;
                    safetyTripped = true;
                    setRelay(false);
                    allRelaysOff();
                    sharedPzemFault = true;
                    taskEXIT_CRITICAL(&sharedMux);

                    Serial.printf("[CORE0] ⚠ PZEM FAULT! %d consecutive invalid "
                                  "reads (%.1fs blind). Safe state -> allRelaysOff().\n",
                                  pzemFailCount, pzemFailCount * 1.0f);
                }

                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
        }

        // Reading is valid (either real from sensor or synthetic fallback)
        if (!isSyntheticSample) {
            pzemFailCount = 0;
            pzemEverValid = true;
        }
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
                allRelaysOff();
                sharedArcFault    = true;
                sharedArcFaultRoC = rateOfChange;
                taskEXIT_CRITICAL(&sharedMux);

                Serial.printf("[CORE0] ⚡ EDGE ARC-FAULT! dP/dt=%.0fW/s "
                              "(threshold: %.0f). Safe state -> allRelaysOff().\n",
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
            allRelaysOff();
            sharedOvercurrentLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ OVERCURRENT! %.1fW > %.1fW (%.2fA > %.1fA). Safe state -> allRelaysOff().\n",
                          powerWatts, criticalWatts, pzemCurrent, MAX_SAFE_CURRENT_A);
        }

        // ── Autonomous Overvoltage Protection ──
        if (pzemVoltage > MAX_SAFE_VOLTAGE_V) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            safetyTripped = true;
            setRelay(false);
            allRelaysOff();
            sharedOvervoltageLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ OVERVOLTAGE! %.1fV > %.1fV. Safe state -> allRelaysOff().\n",
                          pzemVoltage, MAX_SAFE_VOLTAGE_V);
        }

        // ── Autonomous Undervoltage Protection ──
        if (pzemVoltage < MIN_SAFE_VOLTAGE_V && pzemVoltage >= PZEM_MIN_VALID_VOLTAGE_V) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            safetyTripped = true;
            setRelay(false);
            allRelaysOff();
            sharedUndervoltageLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

            Serial.printf("[CORE0] ⚡ UNDERVOLTAGE! %.1fV < %.1fV. Safe state -> allRelaysOff().\n",
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
        sharedSampleIsSynthetic = isSyntheticSample;
        taskEXIT_CRITICAL(&sharedMux);

        // ── Consume queued relay requests (Core 0 is sole owner) ──
        taskENTER_CRITICAL(&sharedMux);
        if (pendingRelayOff) {
            pendingRelayOff = false;
            setRelay(false);
            allRelaysOff();
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

    // ── 2a: Configure RELAY1_PIN and RELAY2_PIN as outputs and write RELAY_OFF (HIGH) ──
    pinMode(RELAY1_PIN, OUTPUT);
    digitalWrite(RELAY1_PIN, RELAY_OFF);
    pinMode(RELAY2_PIN, OUTPUT);
    digitalWrite(RELAY2_PIN, RELAY_OFF);

    // ── 2b: THEN configure RELAY_VCC_PIN as output and write HIGH ──
    pinMode(RELAY_VCC_PIN, OUTPUT);
    digitalWrite(RELAY_VCC_PIN, HIGH);

    // ── 2c: Delay ~100 ms, then init the PZEM on Serial2 (GPIO16/17, 9600) ──
    delay(100);
    PZEM_SERIAL.begin(PZEM_UART_BAUD, SERIAL_8N1, PZEM_RX_PIN, PZEM_TX_PIN);

    // Ensure all relays stay in safe OFF state through boot and reset
    allRelaysOff();

    // ── 2d: Only after that start the main logic ──
#if EMS_BENCH_MODE
    Serial.println("====================================================");
    Serial.println("[BENCH] LOW-VOLTAGE BENCH-TEST MODE ACTIVE");
    Serial.println("[BENCH] Mains and relay load contacts MUST be disconnected!");
    Serial.println("[BENCH] Cycling: Relay 1 (1s) -> Relay 2 (1s) -> All OFF (1s)");
    Serial.println("[BENCH] Interactive commands: DRY_ON, DRY_OFF, STATUS");
    Serial.println("====================================================");
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
        allRelaysOff();
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
    // 8. Bench-test mode: with mains side disconnected, toggles each relay for 1s in turn and prints PZEM status
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        if (command == "DRY_ON") {
            setRelay(1, true);
            setRelay(2, true);
            Serial.println("[BENCH] DRY_ON: Both relays energized (GPIO26=LOW, GPIO27=LOW)");
            printPzemStatus();
            return;
        } else if (command == "DRY_OFF") {
            allRelaysOff();
            Serial.println("[BENCH] DRY_OFF: Both relays de-energized (GPIO26=HIGH, GPIO27=HIGH)");
            printPzemStatus();
            return;
        } else if (command == "STATUS") {
            Serial.printf("[BENCH] Relay1(GPIO26)=%s, Relay2(GPIO27)=%s, RelayVCC(GPIO25)=HIGH\n",
                          relay1Closed ? "ON" : "OFF", relay2Closed ? "ON" : "OFF");
            printPzemStatus();
            return;
        }
    }

    Serial.println("\n[BENCH] --- Step 1: Relay 1 (Laptop Charger, GPIO26) ON for 1s ---");
    setRelay(1, true);
    setRelay(2, false);
    printPzemStatus();
    delay(1000);

    Serial.println("[BENCH] --- Step 2: Relay 2 (Phone Charger, GPIO27) ON for 1s ---");
    setRelay(1, false);
    setRelay(2, true);
    printPzemStatus();
    delay(1000);

    Serial.println("[BENCH] --- Step 3: All Relays OFF for 1s ---");
    allRelaysOff();
    printPzemStatus();
    delay(1000);

    return;
#endif

    if (!client.connected()) {
        reconnectMQTT();
    }
    client.loop();

    // Read shared measurements under spinlock
    float powerWatts, voltage, current, pf, energy, freq, va, var;
    bool relayState, relay1State, relay2State, measurementState, sampleIsSynthetic;
    taskENTER_CRITICAL(&sharedMux);
    powerWatts        = sharedPowerWatts;
    voltage           = sharedVoltage;
    current           = sharedCurrent;
    pf                = sharedPf;
    energy            = sharedEnergy;
    freq              = sharedFrequency;
    va                = sharedApparentPowerVa;
    var               = sharedReactivePowerVar;
    relayState        = relayClosed;
    relay1State       = relay1Closed;
    relay2State       = relay2Closed;
    measurementState  = measurementFresh;
    sampleIsSynthetic = sharedSampleIsSynthetic;
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
            Serial.println("[WATCHDOG] No server heartbeat for 30s. Comms loss -> allRelaysOff()");
            allRelaysOff();
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
            Serial.printf("[USB] V=%.1fV I=%.2fA P=%.1fW PF=%.2f VA=%.1f VAR=%.1f Relay1=%s Relay2=%s source=%s\n",
                          voltage, current, powerWatts, pf, va, var,
                          relay1State ? "ON" : "OFF", relay2State ? "ON" : "OFF",
                          sampleIsSynthetic ? "synthetic" : "real");
        } else {
            Serial.printf("[USB] PZEM=INVALID Relay1=%s Relay2=%s\n",
                          relay1State ? "ON" : "OFF", relay2State ? "ON" : "OFF");
        }
    }

    // ── Publish Rich Electrical Diagnostics at TELEMETRY_INTERVAL_MS (1000ms JSON) ──
    if (millis() - lastTelemetryMs >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryMs = millis();
        
        // Construct JSON telemetry payload containing canonical keys, shorthand keys, source and relays
        StaticJsonDocument<512> doc;
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

        // Data source tracking: real or synthetic
        doc["source"] = sampleIsSynthetic ? "synthetic" : "real";

        // Relay channel array (2 channels: CH1=Laptop on GPIO26, CH2=Phone on GPIO27)
        JsonArray relays = doc.createNestedArray("relays");
        relays.add(relay1State ? 1 : 0);
        relays.add(relay2State ? 1 : 0);

        char jsonBuffer[512];
        serializeJson(doc, jsonBuffer);
        client.publish(topicTelemetry, jsonBuffer);
    }
}
