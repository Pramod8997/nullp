import asyncio
import random
import logging
import json
import aiomqtt
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def simulate_device(client: aiomqtt.Client, device_id: str, base_power: float, variance: float, spike_probability: float, cutoff_threshold: float):
    topic = f"home/sensor/{device_id}/power"
    logger.info(f"Starting simulation for {device_id} on {topic}")
    
    while True:
        try:
            # 1Hz loop representing ESP32 sampling rate
            await asyncio.sleep(1.0)
            
            # Base load with random variance
            power = max(0.0, random.gauss(base_power, variance))
            
            # Inject a random massive spike to trigger the safety layer
            if random.random() < spike_probability:
                power = cutoff_threshold + random.uniform(100, 500)
                logger.warning(f"💉 Injecting massive spike for {device_id}: {power:.2f}W")
            
            # Publish as raw string payload, typical of basic ESP32 implementations
            payload = f"{power:.2f}"
            await client.publish(topic, payload=payload)
            logger.debug(f"Published {payload}W to {topic}")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in device {device_id} loop: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Mock ESP32 1Hz Data Generator")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT Broker address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker port")
    args = parser.parse_args()

    # Configuration for simulated devices
    devices = [
        {"id": "esp32_fridge", "base": 150.0, "var": 10.0, "spike_prob": 0.05, "cutoff": 3500.0},
        {"id": "esp32_hvac", "base": 1200.0, "var": 50.0, "spike_prob": 0.02, "cutoff": 3500.0},
    ]

    while True:
        try:
            async with aiomqtt.Client(args.broker, port=args.port) as client:
                logger.info(f"Connected to MQTT broker at {args.broker}:{args.port}")
                
                tasks = []
                for cfg in devices:
                    task = asyncio.create_task(
                        simulate_device(client, cfg["id"], cfg["base"], cfg["var"], cfg["spike_prob"], cfg["cutoff"])
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
        except aiomqtt.MqttError as e:
            logger.error(f"MQTT Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation stopped manually.")
