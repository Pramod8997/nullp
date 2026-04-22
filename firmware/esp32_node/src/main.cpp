/**
 * Module 7: Phase 2 ESP32 C++ Firmware
 * 
 * Hardware integration layer. Connects to the MQTT broker, publishes real-time 
 * power metrics (simulated via ADC reading), and subscribes to relay control
 * commands from the Confidence-Aware EMS. Includes local safety cutoff logic.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "192.168.1.100"; // EMS Backend IP

WiFiClient espClient;
PubSubClient client(espClient);

const int RELAY_PIN = 5;
const int SENSOR_PIN = 34; // Analog pin for CT sensor
const float VOLTAGE = 230.0;
const float MAX_CURRENT_AMPS = 15.0; // Tier 0 local hardware cutoff

void callback(char* topic, byte* payload, unsigned int length) {
    String message = "";
    for (int i = 0; i < length; i++) {
        message += (char)payload[i];
    }
    
    if (String(topic) == "ems/control/relay_1") {
        if (message == "ON") {
            digitalWrite(RELAY_PIN, HIGH); // Assuming active-high relay
        } else if (message == "OFF") {
            digitalWrite(RELAY_PIN, LOW);
        }
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW); // Start with relay off for safety
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
}

void loop() {
    if (!client.connected()) {
        while (!client.connected()) {
            if (client.connect("ESP32_Node_1")) {
                client.subscribe("ems/control/relay_1");
            } else {
                delay(5000);
            }
        }
    }
    client.loop();
    
    // Read analog value from CT sensor (0-4095 on ESP32)
    int adcValue = analogRead(SENSOR_PIN);
    
    // Convert to Amps (Simplified conversion for Phase 2)
    float current = (adcValue / 4095.0) * 30.0; // Assuming 30A max CT sensor
    float powerWatts = current * VOLTAGE;
    
    // Tier 0 Local Hardware Cutoff (Sub-10ms response)
    // Overrides any software commands if physical current is too high
    if (current > MAX_CURRENT_AMPS) {
        digitalWrite(RELAY_PIN, LOW); // Immediate physical disconnect
        client.publish("ems/alerts/hardware_cutoff", "OVERCURRENT_DETECTED");
        delay(5000); // Cool down before allowing loop to continue
        return;
    }
    
    // Publish power metrics every 1 second to the backend
    static unsigned long lastMsg = 0;
    unsigned long now = millis();
    if (now - lastMsg > 1000) {
        lastMsg = now;
        String payload = "{\"device_id\":\"node_1\", \"power_w\":" + String(powerWatts) + "}";
        client.publish("ems/telemetry/power", payload.c_str());
    }
}
