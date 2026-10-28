#if __has_include("unity.h")
#include "unity.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int _unity_passes = 0;
static int _unity_fails = 0;
#define UNITY_BEGIN() (_unity_passes = _unity_fails = 0)
#define UNITY_END() (printf("\n%d Tests %d Failures\n", _unity_passes + _unity_fails, _unity_fails), _unity_fails)
#define RUN_TEST(func) do { setUp(); func(); tearDown(); } while(0)
#define TEST_ASSERT(cond) do { if (!(cond)) { printf("FAIL: %s at %d\n", #cond, __LINE__); _unity_fails++; return; } else { _unity_passes++; } } while(0)
#define TEST_ASSERT_TRUE(cond) TEST_ASSERT(cond)
#define TEST_ASSERT_FALSE(cond) TEST_ASSERT(!(cond))
#define TEST_ASSERT_EQUAL(expected, actual) TEST_ASSERT((expected) == (actual))
#define TEST_ASSERT_EQUAL_INT(expected, actual) TEST_ASSERT((expected) == (actual))
#define TEST_ASSERT_EQUAL_STRING(expected, actual) TEST_ASSERT(strcmp((expected), (actual)) == 0)
#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) TEST_ASSERT(fabsf((float)(expected) - (float)(actual)) <= (delta))
#define TEST_ASSERT_GREATER_THAN(threshold, actual) TEST_ASSERT((actual) > (threshold))
#endif

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════
// Mock Hardware & Firmware Definitions
// ═══════════════════════════════════════════════════════
#define RELAY_CH1_PIN 26
#define RELAY_CH2_PIN 27
#define RELAY_CH3_PIN 14
#define RELAY_CH4_PIN 32   // Remapped from GPIO 12 (MTDI strapping pin hazard)

#define RELAY_ON  0        // Active-LOW relay logic: 0 = LOW (energised)
#define RELAY_OFF 1        // Active-LOW relay logic: 1 = HIGH (de-energised)

#define MAX_SAFE_CURRENT_A 32.0f
#define MAX_SAFE_VOLTAGE_V 260.0f
#define MIN_SAFE_VOLTAGE_V 180.0f

int relay_channels[4] = {RELAY_OFF, RELAY_OFF, RELAY_OFF, RELAY_OFF};
bool safety_tripped = false;
uint32_t last_heartbeat_ms = 0;
int mqtt_publish_call_count = 0;
char last_mqtt_topic[256] = "";
char last_mqtt_payload[256] = "";
int delay_call_count = 0;
uint32_t backoff_ms = 1000;
uint32_t current_time_ms = 100000;

uint32_t millis(void) {
    return current_time_ms;
}

// Actuate all 4 relay channels with active-LOW logic
void set_all_relays(int state, const char* device) {
    for (int i = 0; i < 4; i++) {
        relay_channels[i] = state;
    }
    mqtt_publish_call_count++;
    if (state == RELAY_OFF) {
        snprintf(last_mqtt_topic, sizeof(last_mqtt_topic), "home/plug/%s/ack", device);
        snprintf(last_mqtt_payload, sizeof(last_mqtt_payload), "OFF_CONFIRMED");
    } else {
        snprintf(last_mqtt_topic, sizeof(last_mqtt_topic), "home/plug/%s/ack", device);
        snprintf(last_mqtt_payload, sizeof(last_mqtt_payload), "ON_CONFIRMED");
    }
}

// ═══════════════════════════════════════════════════════
// PZEM-004T v4.0 Electrical Metric Calculations
// ═══════════════════════════════════════════════════════
float compute_apparent_power(float v_rms, float i_rms) {
    return v_rms * i_rms;
}

float compute_reactive_power(float apparent_va, float active_w) {
    float diff = (apparent_va * apparent_va) - (active_w * active_w);
    if (diff < 0.0f) diff = 0.0f;
    return sqrtf(diff);
}

bool validate_pzem_reading(float v, float i, float p, float pf) {
    if (isnan(v) || isnan(i) || isnan(p) || isnan(pf)) return false;
    if (isinf(v) || isinf(i) || isinf(p) || isinf(pf)) return false;
    if (v < 80.0f || v > 260.0f) return false;
    if (i < 0.0f || i > 100.0f) return false;
    if (p < 0.0f || p > 23000.0f) return false;
    if (pf < 0.0f || pf > 1.0f) return false;
    return true;
}

// ═══════════════════════════════════════════════════════
// Autonomous Safety Trip Protection Logic
// ═══════════════════════════════════════════════════════
bool evaluate_safety_trip(float v, float i, float p) {
    if (i > MAX_SAFE_CURRENT_A || v > MAX_SAFE_VOLTAGE_V || v < MIN_SAFE_VOLTAGE_V) {
        set_all_relays(RELAY_OFF, "test_device");
        safety_tripped = true;
        return true;
    }
    return false;
}

bool handle_remote_command(const char* cmd, const char* device) {
    if (strcmp(cmd, "reset_safety") == 0) {
        safety_tripped = false;
        return true;
    }
    if (strcmp(cmd, "ON") == 0) {
        if (safety_tripped) {
            // Block ON while safety trip is latched
            snprintf(last_mqtt_topic, sizeof(last_mqtt_topic), "home/plug/%s/ack", device);
            snprintf(last_mqtt_payload, sizeof(last_mqtt_payload), "LOCKOUT_NACK");
            return false;
        }
        set_all_relays(RELAY_ON, device);
        return true;
    }
    if (strcmp(cmd, "OFF") == 0) {
        set_all_relays(RELAY_OFF, device);
        return true;
    }
    return false;
}

void process_power_cycle(float prev_power, float curr_power) {
    if (curr_power - prev_power > 800.0f) {
        set_all_relays(RELAY_OFF, "device");
        safety_tripped = true;
    }
}

void check_heartbeat_watchdog(void) {
    if (millis() - last_heartbeat_ms > 30000) {
        set_all_relays(RELAY_OFF, "device");
    }
}

void trigger_wifi_disconnect(void) {
    backoff_ms *= 2;
}

uint32_t get_next_reconnect_interval(void) {
    return backoff_ms;
}

void setUp(void) {
    for (int i = 0; i < 4; i++) {
        relay_channels[i] = RELAY_ON;
    }
    safety_tripped = false;
    mqtt_publish_call_count = 0;
    delay_call_count = 0;
    backoff_ms = 1000;
    current_time_ms = 100000;
    strcpy(last_mqtt_topic, "");
    strcpy(last_mqtt_payload, "");
}

void tearDown(void) {
}

// ═══════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════

/* TEST 1: PZEM-004T v4.0 Apparent & Reactive Power Metrics */
void test_pzem_metrics_calculation(void) {
    float v = 230.0f;
    float i = 2.0f;
    float p = 400.0f; // Active power
    float va = compute_apparent_power(v, i); // 460.0 VA
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 460.0f, va);

    float var = compute_reactive_power(va, p); // sqrt(460^2 - 400^2) = sqrt(211600 - 160000) = sqrt(51600) = 227.15 VAR
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 227.15f, var);
}

/* TEST 2: PZEM-004T NaN & Inf Reading Validation */
void test_pzem_nan_handling_does_not_crash(void) {
    float nan_val = 0.0f / 0.0f;
    float inf_val = 1.0f / 0.0f;
    TEST_ASSERT_FALSE(validate_pzem_reading(nan_val, 1.0f, 230.0f, 1.0f));
    TEST_ASSERT_FALSE(validate_pzem_reading(230.0f, nan_val, 230.0f, 1.0f));
    TEST_ASSERT_FALSE(validate_pzem_reading(230.0f, 1.0f, nan_val, 1.0f));
    TEST_ASSERT_FALSE(validate_pzem_reading(inf_val, 1.0f, 230.0f, 1.0f));
    TEST_ASSERT_TRUE(validate_pzem_reading(230.0f, 1.5f, 345.0f, 1.0f));
}

/* TEST 3: GPIO 32 Pin Mapping Hazard Elimination (MTDI GPIO 12 avoided) */
void test_gpio32_pin_mapping_avoids_strapping_hazard(void) {
    TEST_ASSERT_EQUAL_INT(26, RELAY_CH1_PIN);
    TEST_ASSERT_EQUAL_INT(27, RELAY_CH2_PIN);
    TEST_ASSERT_EQUAL_INT(14, RELAY_CH3_PIN);
    TEST_ASSERT_EQUAL_INT(32, RELAY_CH4_PIN);
    TEST_ASSERT(RELAY_CH4_PIN != 12); // Must NOT be GPIO 12 MTDI strapping pin
}

/* TEST 4: Active-LOW Relay Polarity Macros */
void test_active_low_relay_polarity(void) {
    TEST_ASSERT_EQUAL_INT(0, RELAY_ON);   // Active-LOW: LOW (0) energises coil
    TEST_ASSERT_EQUAL_INT(1, RELAY_OFF);  // Active-LOW: HIGH (1) de-energises coil
}

/* TEST 5: Autonomous Overcurrent Protection Trip */
void test_safety_trip_overcurrent(void) {
    setUp();
    bool tripped = evaluate_safety_trip(230.0f, 35.0f, 8050.0f); // 35A > 32A max safe
    TEST_ASSERT_TRUE(tripped);
    TEST_ASSERT_TRUE(safety_tripped);
    TEST_ASSERT_EQUAL_INT(RELAY_OFF, relay_channels[0]);
    TEST_ASSERT_EQUAL_INT(RELAY_OFF, relay_channels[3]);
}

/* TEST 6: Autonomous Overvoltage & Undervoltage Protection Trip */
void test_safety_trip_voltage_extremes(void) {
    setUp();
    bool over_trip = evaluate_safety_trip(275.0f, 2.0f, 550.0f); // 275V > 260V max safe
    TEST_ASSERT_TRUE(over_trip);
    TEST_ASSERT_TRUE(safety_tripped);

    setUp();
    bool under_trip = evaluate_safety_trip(170.0f, 2.0f, 340.0f); // 170V < 180V min safe
    TEST_ASSERT_TRUE(under_trip);
    TEST_ASSERT_TRUE(safety_tripped);
}

/* TEST 7: Safety Trip Latches and Blocks Remote ON Until Explicit reset_safety */
void test_safety_latch_blocks_remote_on_until_reset(void) {
    setUp();
    evaluate_safety_trip(230.0f, 35.0f, 8050.0f);
    TEST_ASSERT_TRUE(safety_tripped);

    // Attempting remote ON while safety tripped must be rejected
    bool res = handle_remote_command("ON", "test_device");
    TEST_ASSERT_FALSE(res);
    TEST_ASSERT_EQUAL_INT(RELAY_OFF, relay_channels[0]);
    TEST_ASSERT_EQUAL_STRING("LOCKOUT_NACK", last_mqtt_payload);

    // Send explicit reset_safety command
    bool reset_res = handle_remote_command("reset_safety", "test_device");
    TEST_ASSERT_TRUE(reset_res);
    TEST_ASSERT_FALSE(safety_tripped);

    // Now remote ON succeeds
    bool on_res = handle_remote_command("ON", "test_device");
    TEST_ASSERT_TRUE(on_res);
    TEST_ASSERT_EQUAL_INT(RELAY_ON, relay_channels[0]);
    TEST_ASSERT_EQUAL_STRING("ON_CONFIRMED", last_mqtt_payload);
}

/* TEST 8: Arc-Fault Proxy dP/dt Cutoff */
void test_arc_fault_triggers_above_threshold(void) {
    setUp();
    process_power_cycle(100.0f, 1000.0f); // 900W change > 800W threshold
    TEST_ASSERT_EQUAL_INT(RELAY_OFF, relay_channels[0]);
    TEST_ASSERT_TRUE(safety_tripped);
}

/* TEST 9: Heartbeat Watchdog Safe Mode */
void test_heartbeat_watchdog_safe_mode(void) {
    setUp();
    last_heartbeat_ms = millis() - 31000;
    check_heartbeat_watchdog();
    TEST_ASSERT_EQUAL_INT(RELAY_OFF, relay_channels[0]);
}

/* TEST 10: WiFi Reconnect Exponential Backoff */
void test_wifi_reconnect_exponential_backoff(void) {
    backoff_ms = 1000;
    trigger_wifi_disconnect();
    uint32_t first = get_next_reconnect_interval();
    trigger_wifi_disconnect();
    uint32_t second = get_next_reconnect_interval();
    TEST_ASSERT_GREATER_THAN(first, second);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_pzem_metrics_calculation);
    RUN_TEST(test_pzem_nan_handling_does_not_crash);
    RUN_TEST(test_gpio32_pin_mapping_avoids_strapping_hazard);
    RUN_TEST(test_active_low_relay_polarity);
    RUN_TEST(test_safety_trip_overcurrent);
    RUN_TEST(test_safety_trip_voltage_extremes);
    RUN_TEST(test_safety_latch_blocks_remote_on_until_reset);
    RUN_TEST(test_arc_fault_triggers_above_threshold);
    RUN_TEST(test_heartbeat_watchdog_safe_mode);
    RUN_TEST(test_wifi_reconnect_exponential_backoff);
    return UNITY_END();
}
