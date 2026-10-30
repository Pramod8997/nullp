#ifndef EMS_HARDWARE_CONFIG_H
#define EMS_HARDWARE_CONFIG_H

// Production is the default. The dedicated `bench` PlatformIO environment
// defines EMS_BENCH_MODE=1 and removes Wi-Fi/MQTT so DRY_ON/DRY_OFF/CYCLE can only
// be used for a mains-disconnected continuity and relay test.
#ifndef EMS_BENCH_MODE
#define EMS_BENCH_MODE 0
#endif

// Flag to enable synthetic telemetry fallback when PZEM is offline or fails
#ifndef USE_SYNTHETIC_FALLBACK
#define USE_SYNTHETIC_FALLBACK 1
#endif

// ═══════════════════════════════════════════════════════
// ESP32 DevKit V1 (30-pin) Pinout & Hardware Configuration
// ═══════════════════════════════════════════════════════
// PZEM-004T-100A V4.0 on Hardware UART2:
//   - ESP32 RX = GPIO16 (UART2 RX2, wired to PZEM TX)
//   - ESP32 TX = GPIO17 (UART2 TX2, wired to PZEM RX)
//   - Baud rate: 9600, 8N1, Modbus-RTU, default address 0xF8
//   - Logic supply: PZEM powered from ESP32 3V3 pin (3.3 V logic)
//   - Note: Do NOT use GPIO16 or GPIO17 for anything else.
//   - Note: Do NOT use GPIO12 (boot strap pin).
constexpr uint8_t PZEM_RX = 16;
constexpr uint8_t PZEM_TX = 17;
constexpr uint32_t PZEM_BAUD = 9600;
#define PZEM_RX_PIN 16
#define PZEM_TX_PIN 17
#define PZEM_UART_BAUD 9600
#define PZEM_SERIAL Serial2

// 2-Channel Active-LOW Opto-Isolated Relay Module:
//   - Relay module VCC: Powered from GPIO25, used as a 3.3 V source (driven HIGH).
//     Feeds only the optocoupler LEDs (a few mA) and eliminates boot chatter.
//   - Relay 1 (laptop charger): IN1 = GPIO26
//   - Relay 2 (phone charger):  IN2 = GPIO27
//   - Relay coil supply (JD-VCC): VIN (5 V external supply), jumper removed.
//   - Relay logic: ACTIVE-LOW (LOW = relay ON / closed, HIGH = relay OFF / open).
constexpr uint8_t RELAY_VCC_PIN = 25;
constexpr uint8_t RELAY1_PIN = 26;
constexpr uint8_t RELAY2_PIN = 27;

// Backward-compatibility aliases
#define RELAY_CH1_PIN 26
#define RELAY_CH2_PIN 27

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

// Pin assignments
constexpr uint8_t PZEM_RX_PIN_VAL = 16;
constexpr uint8_t PZEM_TX_PIN_VAL = 17;
constexpr uint32_t PZEM_BAUD_VAL = 9600;
constexpr uint8_t RELAY_VCC_PIN_VAL = 25;
constexpr uint8_t RELAY1_PIN_VAL = 26;
constexpr uint8_t RELAY2_PIN_VAL = 27;

// Backward-compatibility aliases
constexpr int RELAY_PIN = 26;
constexpr int RELAY_CH1 = 26;
constexpr int RELAY_CH2 = 27;
constexpr bool RELAY_ACTIVE_LOW = true;

constexpr float CRITICAL_PCT = 1.25f;
constexpr float EDGE_ROC_THRESHOLD_W_PER_S = 1000.0f;
constexpr int BASELINE_WINDOW = 5;
constexpr float BASELINE_INRUSH_CEIL_W = 50.0f;
constexpr float INRUSH_HEADROOM_W = 100.0f;
constexpr unsigned long SAFETY_LOCKOUT_MS = 300000UL;
constexpr int PZEM_FAIL_TRIP_COUNT = 5;  // Default 5 consecutive failures before warning/cutoff
constexpr unsigned long PZEM_MAX_BLIND_MS = 5000UL;

constexpr unsigned long CORE0_SAMPLE_PERIOD_MS = 1000UL;  // Poll no faster than once per second
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
