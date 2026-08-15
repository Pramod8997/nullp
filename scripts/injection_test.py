import asyncio
import time
import json
import logging
from aiomqtt import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_injection_test():
    logger.info("Starting Deep Integration Injection Test...")
    
    async with Client("localhost", port=1883) as client:
        
        # ─── TEST 1: Arc-Fault / Edge Safety ───
        logger.info("TEST 1: Injecting massive transient to trigger Arc-Fault on 'esp32_test_plug'")
        # Baseline
        await client.publish("home/sensor/esp32_test_plug/power", "100.0")
        await asyncio.sleep(1)
        # Massive transient (dP/dt = 2900 W/s)
        await client.publish("home/sensor/esp32_test_plug/power", "3000.0")
        logger.info("Test 1 injected. Monitor logs for SAFETY_CUTOFF.")
        
        await asyncio.sleep(2)
        
        # ─── TEST 2: Watchdog & Temporal Validation ───
        logger.info("TEST 2: Injecting chaotic, impossible data to trigger Watchdog on 'esp32_chaotic'")
        for power in [5.0, 3500.0, 0.0, 4000.0, 10.0, 3900.0]:
            await client.publish("home/sensor/esp32_chaotic/power", str(power))
            await asyncio.sleep(0.5)
        logger.info("Test 2 injected. Monitor logs for Watchdog Soft/Hard anomalies.")
        
        await asyncio.sleep(2)
        
        # ─── TEST 3: OpenMax (Unknown Device Detection) ───
        logger.info("TEST 3: Injecting stable but weird signature for 'esp32_alien' to trigger OpenMax")
        # Send a stable bizarre signature (1337W oscillating slightly)
        for _ in range(30):
            import random
            val = 1337.0 + random.uniform(-2.0, 2.0)
            await client.publish("home/sensor/esp32_alien/power", str(val))
            await asyncio.sleep(0.5)
            
        logger.info("Test 3 injected. Monitor logs for LOW_CONFIDENCE / LABEL_REQUEST.")
        
        await asyncio.sleep(3)
        logger.info("Injection tests complete. Please check the orchestrator logs to verify the system's response.")

if __name__ == "__main__":
    asyncio.run(run_injection_test())
