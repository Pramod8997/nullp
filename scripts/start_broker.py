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
        # Keep running until cancelled
        stop_event = asyncio.Event()
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Broker shutting down...")
    except Exception as e:
        logger.error(f"Failed to start MQTT Broker: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(start_broker())
    except KeyboardInterrupt:
        logger.info("Stopping MQTT Broker...")
