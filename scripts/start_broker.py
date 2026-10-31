import sys
import os
import asyncio
import logging

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from amqtt.broker import Broker

host = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MQTT_BROKER_HOST", "0.0.0.0")
port = int(os.getenv("MQTT_BROKER_PORT", 1883))

config = {
    'listeners': {
        'default': {
            'type': 'tcp',
            'bind': f'{host}:{port}',
        },
    },
    'plugins': [
        'amqtt.plugins.authentication.AnonymousAuthPlugin',
    ],
    'sys_interval': 10,
    'auth': {
        'allow-anonymous': True,
    }
}

async def start_broker():
    logger.info(f"MQTT Broker starting on {host}:{port}...")
    broker = Broker(config)
    try:
        await broker.start()
        logger.info(f"MQTT Broker successfully started and listening on {host}:{port}")
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
