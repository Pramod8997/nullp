import asyncio
import logging
from amqtt.broker import Broker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("bench_broker")

config = {
    "listeners": {
        "default": {
            "type": "tcp",
            "bind": "0.0.0.0:1883",
        },
    },
    "sys_interval": 10,
    "plugins": [
        "amqtt.plugins.authentication.AnonymousAuthPlugin"
    ],
    "auth": {
        "allow-anonymous": True,
    },
}

async def run():
    broker = Broker(config)
    await broker.start()
    logger.info("Bench MQTT Broker listening on 0.0.0.0:1883 (accepting all bench clients)")
    stop_event = asyncio.Event()
    await stop_event.wait()

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Broker stopped.")
