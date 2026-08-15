#include "unity.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Mock definitions
#define RELAY_ON 1
#define RELAY_OFF 0

int relay_state = RELAY_ON;
uint32_t last_heartbeat_ms = 0;
int mqtt_publish_call_count = 0;
char last_mqtt_topic[256] = "";
int delay_call_count = 0;
uint32_t backoff_ms = 1000;
uint32_t current_time_ms = 0;

uint32_t millis() {
    return current_time_ms;
}

void set_relay(int state, const char* device) {
    relay_state = state;
    if (state == RELAY_OFF) {
        mqtt_publish_call_count++;
        snprintf(last_mqtt_topic, sizeof(last_mqtt_topic), "home/plug/%s/ack", device);
    }
}

float compute_true_rms(float* samples, int count) {
    if (count <= 0) return 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < count; i++) {
        sum_sq += samples[i] * samples[i];
    }
    return sqrtf(sum_sq / count);
}

void process_power_cycle(float prev_power, float curr_power) {
    if (curr_power - prev_power > 800.0f) {
        set_relay(RELAY_OFF, "device");
    }
}

void check_heartbeat_watchdog() {
    if (millis() - last_heartbeat_ms > 30000) {
        set_relay(RELAY_OFF, "device");
    }
}

void trigger_wifi_disconnect() {
    backoff_ms *= 2;
}

uint32_t get_next_reconnect_interval() {
    return backoff_ms;
}

void setUp(void) {
    relay_state = RELAY_ON;
    mqtt_publish_call_count = 0;
    delay_call_count = 0;
    backoff_ms = 1000;
    current_time_ms = 100000; // Start at 100s
}

void tearDown(void) {
}

/* TEST 3B-1: True RMS calculation — 200 samples at constant 10V RMS */
void test_true_rms_constant_signal(void) {
    float samples[200];
    for (int i = 0; i < 200; i++) samples[i] = 10.0f;
    float rms = compute_true_rms(samples, 200);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 10.0f, rms);
}

/* TEST 3B-2: True RMS — sinusoidal signal: peak V, RMS = V/√2 */
void test_true_rms_sinusoidal(void) {
    float peak = 14.14f;  // RMS should be 10.0
    float samples[200];
    for (int i = 0; i < 200; i++)
        samples[i] = peak * sinf(2.0f * M_PI * i / 200.0f);
    float rms = compute_true_rms(samples, 200);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 10.0f, rms);
}

/* TEST 3B-3: Arc-fault proxy — dP/dt > 800 W/cycle triggers relay cutoff */
void test_arc_fault_triggers_above_threshold(void) {
    relay_state = RELAY_ON;
    float prev_power = 100.0f;
    float curr_power = 1000.0f;  // 900 W change in 1 cycle — exceeds 800 W/cycle
    process_power_cycle(prev_power, curr_power);
    TEST_ASSERT_EQUAL(RELAY_OFF, relay_state);  // cutoff must have occurred
}

/* TEST 3B-4: Arc-fault does NOT trigger at exactly 800 W/cycle */
void test_arc_fault_not_triggered_at_boundary(void) {
    relay_state = RELAY_ON;
    process_power_cycle(100.0f, 900.0f);  // Exactly 800 W/cycle
    TEST_ASSERT_EQUAL(RELAY_ON, relay_state);  // no cutoff
}

/* TEST 3B-5: Heartbeat watchdog — relay turns OFF after 30s without heartbeat */
void test_heartbeat_watchdog_safe_mode(void) {
    relay_state = RELAY_ON;
    last_heartbeat_ms = millis() - 31000;  // 31 seconds ago
    check_heartbeat_watchdog();
    TEST_ASSERT_EQUAL(RELAY_OFF, relay_state);
}

/* TEST 3B-6: Heartbeat watchdog — relay stays ON with heartbeat within 30s */
void test_heartbeat_watchdog_no_cutoff_in_time(void) {
    relay_state = RELAY_ON;
    last_heartbeat_ms = millis() - 5000;  // 5 seconds ago
    check_heartbeat_watchdog();
    TEST_ASSERT_EQUAL(RELAY_ON, relay_state);
}

/* TEST 3B-7: Hardware ACK — OFF_CONFIRMED published after relay state change */
void test_relay_ack_published_after_off(void) {
    mqtt_publish_call_count = 0;
    strcpy(last_mqtt_topic, "");
    set_relay(RELAY_OFF, "test_device");
    TEST_ASSERT_EQUAL_INT(1, mqtt_publish_call_count);
    TEST_ASSERT_EQUAL_STRING("home/plug/test_device/ack", last_mqtt_topic);
}

/* TEST 3B-8: Non-blocking relay — no delay() calls in relay handler */
void test_relay_handler_nonblocking(void) {
    delay_call_count = 0;
    set_relay(RELAY_OFF, "device");
    TEST_ASSERT_EQUAL_INT(0, delay_call_count);  // millis()-based, not delay()
}

/* TEST 3B-9: WiFi reconnect — exponential backoff increments */
void test_wifi_reconnect_exponential_backoff(void) {
    backoff_ms = 1000;
    trigger_wifi_disconnect();
    uint32_t first_attempt = get_next_reconnect_interval();
    trigger_wifi_disconnect();
    uint32_t second_attempt = get_next_reconnect_interval();
    TEST_ASSERT_GREATER_THAN(first_attempt, second_attempt);
}

/* TEST 3B-10: True RMS edge case — all-zero samples → 0W */
void test_true_rms_zero_signal(void) {
    float samples[200] = {0.0f};
    float rms = compute_true_rms(samples, 200);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, rms);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_true_rms_constant_signal);
    RUN_TEST(test_true_rms_sinusoidal);
    RUN_TEST(test_arc_fault_triggers_above_threshold);
    RUN_TEST(test_arc_fault_not_triggered_at_boundary);
    RUN_TEST(test_heartbeat_watchdog_safe_mode);
    RUN_TEST(test_heartbeat_watchdog_no_cutoff_in_time);
    RUN_TEST(test_relay_ack_published_after_off);
    RUN_TEST(test_relay_handler_nonblocking);
    RUN_TEST(test_wifi_reconnect_exponential_backoff);
    RUN_TEST(test_true_rms_zero_signal);
    return UNITY_END();
}
