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
 *   - Unconditional overcurrent cutoff (125% of rated)
 *   - PZEM loss-of-measurement watchdog (bounded successful-read age → cutoff)
 *   - Immediate relay cutoff — zero network dependency
 *
 * CORE 1 (Standard Priority — Arduino loop):
 *   - Non-blocking MQTT client.loop()
 *   - 1Hz telemetry broadcast (plain float)
 *   - Incoming relay command handler (ON/OFF/WARNING)
 *   - Consumes the core-0 safety latches (arc-fault / overcurrent / PZEM
 *     fault) → 5-minute lockout + best-effort status alert publishing.
 *     Reporting only: every cutoff already happened on core 0.
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

// Per-node credentials. Copy include/secrets.h.example to include/secrets.h
// and fill it in; secrets.h is gitignored so a real Wi-Fi or broker password
// never lands in version control. Kept out of this file deliberately: the
// placeholders that used to live here ("YOUR_WIFI_SSID", "192.168.1.100") are
// tracked, so filling them in risks committing credentials.
#if !__has_include("secrets.h")
#error "Missing firmware/esp32_node/include/secrets.h - copy secrets.h.example to secrets.h and fill in your Wi-Fi SSID/password, broker IP and broker password."
#endif
#include "secrets.h"

// ═══════════════════════════════════════════════════════
//  CONFIGURATION — set these in include/secrets.h
// ═══════════════════════════════════════════════════════
// Single aggregate sense point: ONE PZEM measures ONE IS 1293 6 A socket, and a
// 3-pin earthed multi-plug adapter in that socket puts the laptop brick and the
// phone charger behind the same shunt at the same time. That simultaneity is
// what NILM disaggregates -- one load at a time would be single-appliance
// classification, not disaggregation.
// See claude_debug/HARDWARE_FINAL_SPEC.md (scope S1/S2/S5, decision D12').
const char* DEVICE_ID      = EMS_DEVICE_ID;
const char* ssid           = EMS_WIFI_SSID;
const char* password       = EMS_WIFI_PASSWORD;
const char* mqtt_server    = EMS_MQTT_SERVER;
const int   mqtt_port      = EMS_MQTT_PORT;
const char* mqtt_user      = EMS_MQTT_USER;
const char* mqtt_password  = EMS_MQTT_PASSWORD;
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

// ── PZEM Loss-of-Measurement Watchdog ──
// Consecutive failed/non-finite PZEM reads before the relay is opened. This is
// the tuning knob: 30 samples at the 100 ms core-0 cadence = 3 s of measuring
// nothing.
//   - Long enough that transient Modbus CRC errors / UART timeouts on a noisy
//     bench line (which arrive in ones and twos, not thirties) cannot
//     nuisance-trip the demo.
//   - Short enough that an energised socket is never left blind for a
//     meaningful time: with no measurement BOTH the overcurrent and arc-fault
//     channels are off, and at the ≤312 W trip ceiling of D8 (1.36 A through a
//     5 A fuse and 1.0 mm² wire) 3 s is thermally irrelevant.
// Raise it only with evidence of nuisance trips; never above ~100 (10 s).
const int PZEM_FAIL_TRIP_COUNT = 30;
// Elapsed-age guard for the successful PZEM transaction, independent of how
// long a failed getter/library timeout takes. The value is a software claim
// until Gate 8 measures the installed dependency on the target ESP32/PZEM.
const unsigned long PZEM_MAX_BLIND_MS = 3000;

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

// ── Core 0 → Core 1 safety latches ──
// Core 0 has ALREADY opened the relay by the time either of these is raised.
// Core 1 only consumes them to take the 5-minute lockout and publish the
// alert, so a dead broker delays the *report*, never the cutoff.
//
// sharedOvercurrentLatch closes a real hole. Core 1 used to LEVEL-test
// sharedPowerWatts > criticalWatts, but core 0 opens the relay on the very
// sample that offends, so a spike that tripped core 0 and collapsed before
// core 1 next looked took NO lockout at all — and the next `ON` re-closed
// straight into the fault. Core 0 sees every 100 ms sample and raises the
// latch on each trip, so the latch is a superset of anything the level check
// could have observed: sharedPowerWatts only ever holds a value core 0 has
// already run the overcurrent test against.
volatile bool  sharedOvercurrentLatch = false;

// Raised by the PZEM loss-of-measurement watchdog (PZEM_FAIL_TRIP_COUNT).
volatile bool  sharedPzemFault   = false;

// Core 0 is the only runtime relay owner. Core 1's MQTT callback queues a
// request here; the safety task consumes it only after validating the current
// PZEM sample and evaluating all cutoff conditions. This prevents a command
// from winning the interval between a Core-0 cutoff and Core-1 lockout.
volatile bool  pendingRelayOn  = false;
volatile bool  pendingRelayOff = false;
volatile bool  pendingAck      = false;
// 0 = none, 1 = ON_CONFIRMED, 2 = OFF_CONFIRMED, 3 = LOCKOUT_NACK.
volatile uint8_t pendingAckType = 0;
volatile bool  safetyInhibit   = false;
volatile bool  measurementFresh = false;
volatile unsigned long lastSuccessfulPzemReadMs = 0;
volatile bool safetyTaskRunning = false;
TaskHandle_t safetyTaskHandle = nullptr;

// Last commanded relay state, written by setRelay() by setup or Core 0 only.
// Read by core 0 solely to arm the PZEM watchdog: the watchdog's own
// cutoff clears it, so the trip fires once per fault episode instead of once
// per 100 ms sample, and only a server `ON` (which the lockout gates) re-arms
// it. An open relay is not an unprotected energised socket, so there is
// nothing to trip on. It does NOT gate overcurrent or arc-fault — those stay
// unconditional. A single aligned bool is atomic on Xtensa, so no spinlock.
volatile bool  relayClosed       = false;

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
    // Recorded AFTER the pin write on purpose. If core 0's cutoff preempts a
    // core-1 `ON` mid-call, this ordering can only leave relayClosed=true over
    // an OPEN relay, which costs one redundant watchdog cutoff next sample.
    // The reverse ordering could leave relayClosed=false over a CLOSED relay —
    // a disarmed watchdog on an energised socket.
    relayClosed = on;
}

// ═══════════════════════════════════════════════════════
//  CORE 0: HIGH-PRIORITY SAFETY SAMPLING TASK
// ═══════════════════════════════════════════════════════
void SafetySamplingTask(void* pvParameters) {
    safetyTaskRunning = true;
    float lastWatts = 0.0;
    float baselineRing[BASELINE_WINDOW];
    int   baselineIdx   = 0;
    int   baselineFill  = 0;
    int   pzemFailCount = 0;   // Consecutive bad reads; 0 on any good read
    // Watchdog ARMING latch: set by the first all-finite read and never
    // cleared. Task-local (not volatile, no spinlock): core 0 is its only
    // reader and only writer — core 1 never looks at it. See the trip branch
    // for why the watchdog must stay disarmed until the PZEM proves itself.
    bool  pzemEverValid = false;
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

        // ── Reject any non-finite PZEM read ──
        // !isfinite, not isnan: the library signals a failed Modbus frame with
        // NAN, but a corrupted frame decoded into a float can also land on
        // ±Inf, and an Inf reaching lastWatts/the baseline ring poisons the
        // dP/dt channel permanently (every later comparison against it is
        // false). isnan is a strict subset of !isfinite, so this only ever
        // rejects more.
        if (!isfinite(powerWatts) || !isfinite(pzemVoltage)
            || !isfinite(pzemCurrent) || !isfinite(pzemPf)) {
            measurementFresh = false;
            // A pending ON is rejected as soon as the safety task observes
            // that the measurement channel is unavailable. OFF remains
            // serviceable and is still owned by this task.
            taskENTER_CRITICAL(&sharedMux);
            if (pendingRelayOn) {
                pendingRelayOn = false;
                pendingAck = true;
                pendingAckType = 3;
            }
            if (pendingRelayOff) {
                pendingRelayOff = false;
                setRelay(false);
                pendingAck = true;
                pendingAckType = 2;
            }
            taskEXIT_CRITICAL(&sharedMux);
            // ── PZEM Loss-of-Measurement Watchdog ──
            // This branch used to `continue` forever with the relay LEFT
            // CLOSED: a dead PZEM, a severed UART or a pulled 5 V rail
            // silently disabled BOTH the overcurrent and the arc-fault channel
            // while the socket stayed energised — fail-DANGEROUS. After
            // PZEM_FAIL_TRIP_COUNT consecutive blind samples, open the relay
            // and hand the lockout to core 1. Zero network dependency: the
            // cutoff is complete before core 1 ever hears about it.
            if (pzemFailCount < PZEM_FAIL_TRIP_COUNT) pzemFailCount++;  // Saturate: no overflow on a permanently dead sensor
            // Three conditions, ALL required:
            //  1. PZEM_FAIL_TRIP_COUNT consecutive blind samples (3 s). Any
            //     good read rewinds the counter to 0.
            //  2. relayClosed — an open relay is not an unprotected energised
            //     socket, so there is nothing to protect. The cutoff below
            //     clears it, so the trip fires once per fault episode, not
            //     once per 100 ms sample. The counter is NOT cleared on close,
            //     so a node whose PZEM died while open trips 100 ms after the
            //     next `ON` instead of waiting another 3 s — deliberate.
            //  3. pzemEverValid — DISARMED until the PZEM has returned one
            //     all-finite read since boot. A node that has never measured
            //     anything has never observed mains, so there is provably no
            //     energised socket to protect: this refines the invariant, it
            //     does not weaken it. The PZEM sits UPSTREAM of the relay
            //     (WIRING_STEP_BY_STEP.md:154) and is mains-powered, so with
            //     no mains it returns NaN forever regardless of relay state —
            //     and the documented bring-up runs exactly that way ON PURPOSE.
            //     Without this, BRINGUP_RUNBOOK.md Gate 7 ("Relay, dry, NO
            //     MAINS") is unperformable: the saturated counter trips within
            //     100 ms of the commanded close, so the contacts re-open and
            //     lock out for 5 minutes before COM–NO continuity can be
            //     metered. Once the sensor has proven itself alive, losing it
            //     is a real fault and still trips in 3 s.
            const bool successfulReadExpired =
                lastSuccessfulPzemReadMs > 0
                && (unsigned long)(nowMs - lastSuccessfulPzemReadMs)
                   >= PZEM_MAX_BLIND_MS;
            if (pzemFailCount >= PZEM_FAIL_TRIP_COUNT && pzemEverValid
                    && relayClosed && successfulReadExpired) {
                taskENTER_CRITICAL(&sharedMux);
                safetyInhibit = true;
                setRelay(false);   // Clears relayClosed → one trip per episode
                sharedPzemFault = true;
                taskEXIT_CRITICAL(&sharedMux);

                Serial.printf("[CORE0] ⚠ PZEM FAULT! %d consecutive invalid "
                              "reads (%.1fs blind). Relay CUTOFF.\n",
                              pzemFailCount, pzemFailCount * 0.1f);
            }
            // Read failure, skip the rest of this cycle: lastWatts, the
            // baseline ring and the shared block stay untouched.
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        pzemFailCount = 0;
        pzemEverValid = true;   // Latched forever: the watchdog is now armed
        measurementFresh = true;
        lastSuccessfulPzemReadMs = nowMs;

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
                taskENTER_CRITICAL(&sharedMux);
                safetyInhibit   = true;
                setRelay(false);
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
        //
        // The latch is what makes the trip stick. Opening the relay collapses
        // the reading, so by the time core 1 next reads sharedPowerWatts the
        // overload is gone; without the latch the 5-minute lockout was simply
        // never taken and the next `ON` re-closed into the fault.
        float criticalWatts = RATED_WATTS * CRITICAL_PCT;
        if (powerWatts > criticalWatts) {
            taskENTER_CRITICAL(&sharedMux);
            safetyInhibit = true;
            setRelay(false);
            sharedOvercurrentLatch = true;
            taskEXIT_CRITICAL(&sharedMux);

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

        // ── Consume queued relay requests (Core 0 is sole owner) ──
        taskENTER_CRITICAL(&sharedMux);
        if (pendingRelayOff) {
            pendingRelayOff = false;
            setRelay(false);
            pendingAck = true;
            pendingAckType = 2;
        } else if (pendingRelayOn) {
            pendingRelayOn = false;
            if (safetyTaskRunning && !safetyInhibit && measurementFresh && !relayLocked) {
                setRelay(true);
                pendingAck = true;
                pendingAckType = 1;
            } else {
                pendingAck = true;
                pendingAckType = 3;
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
    BaseType_t safetyTaskResult = xTaskCreatePinnedToCore(
        SafetySamplingTask,   // Task function
        "SafetySampling",     // Name
        4096,                 // Stack size (bytes)
        NULL,                 // Parameters
        2,                    // Priority (higher than loop)
        &safetyTaskHandle,    // Task handle for creation/liveness gate
        0                     // Core 0
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
    client.setServer(mqtt_server, mqtt_port);
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
        taskENTER_CRITICAL(&sharedMux);
        safetyInhibit = false;
        taskEXIT_CRITICAL(&sharedMux);
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

    // ── Check for overcurrent latch from Core 0 ──
    // Latch-consume, NOT a level check on sharedPowerWatts. The level check
    // this replaces (powerWatts > criticalWatts && !relayLocked) could miss a
    // trip outright: core 0 opens the relay on the offending sample, so the
    // reading collapses to a safe value before core 1 next looks and the
    // lockout was never taken — leaving the next `ON` free to re-close into
    // the fault. Core 0 evaluates every 100 ms sample, core 1 only ever saw
    // the last value written to shared memory, so the latch strictly
    // dominates: every value that could satisfy the level check was already
    // tested by core 0 in the same cycle that wrote it.
    bool overcurrentTripped = false;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedOvercurrentLatch) {
        overcurrentTripped = true;
        sharedOvercurrentLatch = false;  // Acknowledge
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

    // ── Check for PZEM loss-of-measurement latch from Core 0 ──
    // Core 0 has already opened the relay; the lockout stops an `ON` from
    // re-energising a socket we cannot measure for the next 5 minutes.
    bool pzemFaultTripped = false;
    taskENTER_CRITICAL(&sharedMux);
    if (sharedPzemFault) {
        pzemFaultTripped = true;
        sharedPzemFault = false;  // Acknowledge
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

    // Publish acknowledgements generated by the Core-0 relay owner. The MQTT
    // callback never writes GPIO or claims success before the safety task has
    // evaluated the request.
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
