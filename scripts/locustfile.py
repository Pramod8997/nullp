import time
import random
from locust import task, events, User, between
import paho.mqtt.client as mqtt

# We use standard Paho MQTT for the Locust clients
# to simulate the ESP32 devices publishing telemetry.

class MQTTUser(User):
    # Simulate high frequency data (10Hz = 0.1 seconds between tasks)
    wait_time = between(0.09, 0.11)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign a random simulated ESP32 device ID to this user
        self.device_id = f"esp32_node_{random.randint(1, 100):03d}"
        self.client = mqtt.Client(client_id=self.device_id)
        
    def on_start(self):
        # Connect to the local broker when the simulated user starts
        self.client.connect("localhost", 1883, 60)
        self.client.loop_start()
        
    def on_stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    @task
    def publish_telemetry(self):
        topic = f"home/sensor/{self.device_id}/power"
        
        # Simulate baseline power draw with occasional spikes
        if random.random() < 0.05:
            # 5% chance of a transient spike (appliance turning on/off)
            power = random.uniform(1000.0, 3500.0)
        else:
            # Normal baseline variance
            power = random.uniform(1.0, 15.0)
            
        payload = str(round(power, 2))
        
        start_time = time.time()
        try:
            # QoS 0 is used for high-frequency telemetry in our system
            self.client.publish(topic, payload, qos=0)
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="MQTT Publish",
                name=topic,
                response_time=total_time,
                response_length=len(payload),
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="MQTT Publish",
                name=topic,
                response_time=total_time,
                response_length=0,
                exception=e,
            )

# Run instructions:
# Activate virtual environment
# pip install locust
# locust -f scripts/locustfile.py --headless -u 100 -r 20 -t 1m
