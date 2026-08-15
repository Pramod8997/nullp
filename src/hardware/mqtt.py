import re
import aiomqtt
import asyncio
import logging
from typing import Callable, Awaitable, Optional, Union, Any, List, Set, Dict

logger = logging.getLogger(__name__)

class MQTTClientManager:
    def __init__(self, broker: str, port: int = 1883):
        self.broker = broker
        self.port = port
        self.client: Optional[aiomqtt.Client] = None
        self._read_callback: Optional[Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]] = None
        self._connected: bool = True

    def set_read_callback(self, callback: Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]) -> None:
        self._read_callback = callback

    def is_connected(self) -> bool:
        if self.client is not None:
            return True
        return self._connected

    async def run(self, read_topic: Union[str, list]) -> None:
        # Reconnect loop
        while True:
            try:
                async with aiomqtt.Client(self.broker, port=self.port) as client:
                    self.client = client
                    self._connected = True
                    logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
                    topics = [read_topic] if isinstance(read_topic, str) else read_topic
                    for t in topics:
                        await client.subscribe(t, qos=1)
                        logger.info(f"Subscribed to reads on {t}")
                    async for message in client.messages:
                        if self._read_callback and message.topic:
                            try:
                                await self._read_callback(str(message.topic), message.payload)
                            except Exception as e:
                                logger.error(f"Error processing MQTT message: {e}")
            except aiomqtt.MqttError as e:
                self.client = None
                self._connected = False
                logger.error(f"MQTT connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.client = None
                self._connected = False
                logger.info("MQTT client task cancelled.")
                break

    async def publish_command(self, write_topic: str, payload: str) -> None:
        if not self.client:
            logger.error("MQTT client not connected, cannot publish.")
            return
        try:
            await self.client.publish(write_topic, payload=payload, qos=1)
            logger.debug(f"Published to {write_topic}: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish to {write_topic}: {e}")


def topic_matches_sub(sub: str, topic: str) -> bool:
    """MQTT topic wildcard matching (+ for single-level, # for multi-level)."""
    pattern = "^" + re.escape(sub).replace(r"\+", r"[^/]+").replace(r"\#", r".*") + "$"
    return bool(re.match(pattern, topic))


class AsyncMQTTClient:
    """
    Asynchronous MQTT client supporting wildcard subscriptions, publishing,
    and mock/in-memory message delivery for tests and local simulation.
    """
    def __init__(
        self,
        on_message: Optional[Callable[[str, Union[str, bytes]], Any]] = None,
        broker: str = "localhost",
        port: int = 1883
    ):
        self.on_message = on_message
        self.broker = broker
        self.port = port
        self.subscriptions: Set[str] = set()
        self._connected: bool = True
        self.published_messages: List[tuple[str, Any]] = []

    async def subscribe(self, topic: str) -> None:
        self.subscriptions.add(topic)

    async def publish(self, topic: str, payload: Union[str, bytes]) -> None:
        if not self._connected:
            return
        self.published_messages.append((topic, payload))
        if self.on_message:
            for sub in self.subscriptions:
                if topic_matches_sub(sub, topic):
                    if asyncio.iscoroutinefunction(self.on_message):
                        await self.on_message(topic, payload)
                    else:
                        self.on_message(topic, payload)

    async def disconnect(self) -> None:
        self._connected = False

    async def reconnect(self) -> None:
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    async def get_published(self, topic_filter: Optional[str] = None) -> List[Any]:
        if topic_filter is None:
            return [p for _, p in self.published_messages]
        return [p for t, p in self.published_messages if topic_matches_sub(topic_filter, t)]


class MockMQTTBroker:
    """Mock MQTT broker to simulate connection drops, disconnects, and restarts."""
    _active_broker: Optional['MockMQTTBroker'] = None

    def __init__(self) -> None:
        self.clients: List[Any] = []
        self._running: bool = True
        MockMQTTBroker._active_broker = self

    def register(self, client: Any) -> None:
        if client not in self.clients:
            self.clients.append(client)

    def unregister(self, client: Any) -> None:
        if client in self.clients:
            self.clients.remove(client)

    async def disconnect_all(self) -> None:
        self._running = False
        for client in list(self.clients):
            if hasattr(client, "handle_broker_disconnect"):
                res = client.handle_broker_disconnect()
                if asyncio.iscoroutine(res):
                    await res
            elif hasattr(client, "disconnect"):
                await client.disconnect()
            elif hasattr(client, "_connected"):
                client._connected = False

    async def restart(self) -> None:
        self._running = True
        for client in list(self.clients):
            if hasattr(client, "handle_broker_reconnect"):
                res = client.handle_broker_reconnect()
                if asyncio.iscoroutine(res):
                    await res
            elif hasattr(client, "reconnect"):
                await client.reconnect()
            elif hasattr(client, "_connected"):
                client._connected = True

