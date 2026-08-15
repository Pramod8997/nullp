import asyncio
import logging
from aiomqtt import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_openmax_test():
    logger.info("Starting OpenMax Transient Test...")
    async with Client("localhost", port=1883) as client:
        # 1. Baseline off - MUST fill the 128-sample buffer
        logger.info("Filling 128-sample CNN buffer with zeros...")
        for _ in range(130):
            await client.publish("home/sensor/esp32_alien2/power", "0.0")
            await asyncio.sleep(0.05) # Send fast just to fill buffer
            
        # 2. Huge transient turn on (> 50W dP/dt to trigger NILM)
        logger.info("Injecting turn-on transient...")
        await client.publish("home/sensor/esp32_alien2/power", "1337.0")
        await asyncio.sleep(0.5)
        
        # 3. Stable signature to pass DeltaStability (min_occurrences = 3)
        # Needs to be sent at normal 1Hz pace so the pipeline processes them
        logger.info("Injecting stable signature...")
        for _ in range(15):
            await client.publish("home/sensor/esp32_alien2/power", "1337.0")
            await asyncio.sleep(1.0)
            
    logger.info("Test complete.")

if __name__ == "__main__":
    asyncio.run(run_openmax_test())
