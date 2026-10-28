#ifndef EMS_HARDWARE_CONFIG_H
#define EMS_HARDWARE_CONFIG_H

// Production is the default. The dedicated `bench` PlatformIO environment
// defines EMS_BENCH_MODE=1 and removes Wi-Fi/MQTT so DRY_ON/DRY_OFF can only
// be used for a mains-disconnected continuity test.
#ifndef EMS_BENCH_MODE
#define EMS_BENCH_MODE 0
#endif

// Compile-time hardware contract for the ESP32 node.
//
// Keep physical pins, safety thresholds, metering scales and task timing in
// one tracked header.  The runtime YAML profiles mirror these values and the
// parity tests compare both representations.  Credentials remain in
// secrets.h and are deliberately not part of this file.
namespace ems_hardware {

constexpr float RATED_WATTS = 250.0f;
constexpr float POWER_FACTOR = 1.0f;
constexpr float MAINS_VOLTAGE_V = 230.0f;
constexpr float MAINS_FREQUENCY_HZ = 50.0f;

constexpr int RELAY_PIN = 18;
constexpr bool RELAY_ACTIVE_LOW = false;
constexpr int PZEM_RX_PIN = 16;
constexpr int PZEM_TX_PIN = 17;
constexpr int PZEM_UART_BAUD = 9600;

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

// The physically available meter is a PZEM-004T-100A(V4.0)-type external-CT
// unit.  The installed library uses the v3/v4-compatible Modbus register
// protocol, but the exact board, CT polarity, TX level and calibration still
// require physical validation before mains release.  There is no ESP32 ADC,
// SCT-013 burden, ZMPT101B or ACS712 path in this firmware.
constexpr float PZEM_MIN_VALID_VOLTAGE_V = 80.0f;
constexpr float PZEM_MAX_VALID_VOLTAGE_V = 260.0f;
constexpr float PZEM_MAX_VALID_CURRENT_A = 100.0f;
constexpr float PZEM_MAX_VALID_POWER_W = 23000.0f;
constexpr float VOLTAGE_SCALE = 1.0f;
constexpr float CURRENT_SCALE = 1.0f;
constexpr float POWER_SCALE = 1.0f;

}  // namespace ems_hardware

#endif  // EMS_HARDWARE_CONFIG_H
