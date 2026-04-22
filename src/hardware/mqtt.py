import aiomqtt
import asyncio
import logging
from typing import Callable, Awaitable, Optional, Union

logger = logging.getLogger(__name__)

class MQTTClientManager:
    def __init__(self, broker: str, port: int = 1883):
        self.broker = broker
        self.port = port
        self.client: Optional[aiomqtt.Client] = None
        self._read_callback: Optional[Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]] = None

    def set_read_callback(self, callback: Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]) -> None:
        self._read_callback = callback

    async def run(self, read_topic: str) -> None:
        # Reconnect loop
        while True:
            try:
                async with aiomqtt.Client(self.broker, port=self.port) as client:
                    self.client = client
                    logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
                    await client.subscribe(read_topic)
                    logger.info(f"Subscribed to reads on {read_topic}")
                    async for message in client.messages:
                        if self._read_callback and message.topic:
                            try:
                                await self._read_callback(str(message.topic), message.payload)
                            except Exception as e:
                                logger.error(f"Error processing MQTT message: {e}")
            except aiomqtt.MqttError as e:
                logger.error(f"MQTT connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                logger.info("MQTT client task cancelled.")
                break

    async def publish_command(self, write_topic: str, payload: str) -> None:
        if not self.client:
            logger.error("MQTT client not connected, cannot publish.")
            return
        try:
            await self.client.publish(write_topic, payload=payload)
            logger.debug(f"Published to {write_topic}: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish to {write_topic}: {e}")
