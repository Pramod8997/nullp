#ifndef EMS_HARDWARE_CONFIG_H
#define EMS_HARDWARE_CONFIG_H

// Production is the default. The dedicated `bench` PlatformIO environment
// defines EMS_BENCH_MODE=1 and removes Wi-Fi/MQTT so DRY_ON/DRY_OFF can only
// be used for a mains-disconnected continuity test.
#ifndef EMS_BENCH_MODE
#define EMS_BENCH_MODE 0
#endif

// ═══════════════════════════════════════════════════════
// PZEM-004T v4.0 (100A) UART2 Configuration
// ═══════════════════════════════════════════════════════
#define PZEM_RX_PIN 16
#define PZEM_TX_PIN 17
#define PZEM_SERIAL Serial2
#define PZEM_UART_BAUD 9600

// ═══════════════════════════════════════════════════════
// 4-Channel Active-LOW Opto-Isolated Relay Configuration
// Remapped CH4 from GPIO 12 to GPIO 32 to eliminate MTDI flash strapping pin boot hazard
// ═══════════════════════════════════════════════════════
#define RELAY_CH1_PIN 26
#define RELAY_CH2_PIN 27
#define RELAY_CH3_PIN 14
#define RELAY_CH4_PIN 32

// Keep active-low fail-safe relay macros
#ifndef RELAY_ON
#define RELAY_ON LOW
#endif
#ifndef RELAY_OFF
#define RELAY_OFF HIGH
#endif

// ═══════════════════════════════════════════════════════
// Production Safety Thresholds (230V / 100A CT)
// ═══════════════════════════════════════════════════════
#define ENABLE_LOCAL_PROTECTION 1
#define MAX_SAFE_CURRENT_A 32.0f
#define MAX_SAFE_VOLTAGE_V 260.0f
#define MIN_SAFE_VOLTAGE_V 180.0f
#define TELEMETRY_INTERVAL_MS 1000
#define WATCHDOG_TIMEOUT_S 5

// Compile-time hardware contract for the ESP32 node.
namespace ems_hardware {

constexpr float RATED_WATTS = 250.0f;
constexpr float POWER_FACTOR = 1.0f;
constexpr float MAINS_VOLTAGE_V = 230.0f;
constexpr float MAINS_FREQUENCY_HZ = 50.0f;

constexpr int RELAY_PIN = 18;  // Legacy single-channel binding for test compatibility
constexpr int RELAY_CH1 = RELAY_CH1_PIN;
constexpr int RELAY_CH2 = RELAY_CH2_PIN;
constexpr int RELAY_CH3 = RELAY_CH3_PIN;
constexpr int RELAY_CH4 = RELAY_CH4_PIN;
constexpr bool RELAY_ACTIVE_LOW = false;

constexpr float CRITICAL_PCT = 1.25f;
constexpr float EDGE_ROC_THRESHOLD_W_PER_S = 1000.0f;
constexpr int BASELINE_WINDOW = 5;
constexpr float BASELINE_INRUSH_CEIL_W = 50.0f;
constexpr float INRUSH_HEADROOM_W = 100.0f;
constexpr unsigned long SAFETY_LOCKOUT_MS = 300000UL;
constexpr int PZEM_FAIL_TRIP_COUNT = 30;
constexpr unsigned long PZEM_MAX_BLIND_MS = 3000UL;

constexpr unsigned long CORE0_SAMPLE_PERIOD_MS = 100UL;
constexpr unsigned long WIFI_CONNECT_TIMEOUT_MS = 30000UL;
constexpr unsigned long MQTT_RECONNECT_BACKOFF_MS = 5000UL;
constexpr unsigned long SERVER_HEARTBEAT_TIMEOUT_MS = 30000UL;
constexpr unsigned int MQTT_PAYLOAD_MAX_BYTES = 256U;
constexpr unsigned long POWER_TELEMETRY_PERIOD_MS = 1000UL;
constexpr unsigned long DIAGNOSTIC_TELEMETRY_PERIOD_MS = 10000UL;

// PZEM-004T-100A(V4.0) digital metering limits
constexpr float PZEM_MIN_VALID_VOLTAGE_V = 80.0f;
constexpr float PZEM_MAX_VALID_VOLTAGE_V = 260.0f;
constexpr float PZEM_MAX_VALID_CURRENT_A = 100.0f;
constexpr float PZEM_MAX_VALID_POWER_W = 23000.0f;
constexpr float VOLTAGE_SCALE = 1.0f;
constexpr float CURRENT_SCALE = 1.0f;
constexpr float POWER_SCALE = 1.0f;

}  // namespace ems_hardware

#endif  // EMS_HARDWARE_CONFIG_H
