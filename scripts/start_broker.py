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
            # Loopback only: this is the last-resort dev fallback broker and
            # it allows anonymous connects — binding 0.0.0.0 would silently
            # expose an unauthenticated broker (direct relay control) to the LAN.
            'bind': '127.0.0.1:1883',
        },
    },
    'sys_interval': 10,
    'auth': {
        'allow-anonymous': True,
    }
}

async def start_broker():
    # Loud on purpose: this is an anonymous, loopback-only dev fallback —
    # never to be mistaken for the authenticated broker the rig uses.
    print("!!! SECURITY: anonymous dev fallback broker on 127.0.0.1:1883 !!!\n"
          "!!! Loopback-only, no authentication — NOT for the rig or any LAN !!!\n"
          "!!! Use the authenticated mosquitto stack for anything real.      !!!")
    logger.warning("Anonymous dev fallback broker starting on 127.0.0.1:1883 "
                   "(loopback-only, NOT for the rig)")
    broker = Broker(config)
    try:
        await broker.start()
        logger.info("MQTT Broker started on 127.0.0.1:1883")
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
