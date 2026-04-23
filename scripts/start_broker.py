import asyncio
import logging
from amqtt.broker import Broker

# Set up logging to see broker activity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = {
    'listeners': {
        'default': {
            'type': 'tcp',
            'bind': '0.0.0.0:1883',
        },
    },
    'sys_interval': 10,
    'auth': {
        'allow-anonymous': True,
    }
}

async def start_broker():
    broker = Broker(config)
    try:
        await broker.start()
        logger.info("MQTT Broker started on 0.0.0.0:1883")
    except Exception as e:
        logger.error(f"Failed to start MQTT Broker: {e}")

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_broker())
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Stopping MQTT Broker...")
