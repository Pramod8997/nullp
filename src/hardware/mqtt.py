import re
import aiomqtt
import asyncio
import logging
from typing import Callable, Awaitable, Optional, Union, Any, List, Set, Tuple

logger = logging.getLogger(__name__)

class MQTTClientManager:
    def __init__(self, broker: str, port: int = 1883,
                 username: Optional[str] = None, password: Optional[str] = None,
                 ingress_queue_size: int = 256):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client: Optional[aiomqtt.Client] = None
        self._read_callback: Optional[Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]] = None
        self._connected: bool = True
        self._run_started = False
        self._ingress_queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue(
            maxsize=max(1, ingress_queue_size)
        )
        self._dropped_messages = 0

    def set_read_callback(self, callback: Callable[[str, Union[str, bytes, bytearray, int, float, None]], Awaitable[None]]) -> None:
        self._read_callback = callback

    def is_connected(self) -> bool:
        if self._run_started:
            return self.client is not None and self._connected
        # Preserve the historical pre-run contract used by local pipeline
        # construction/tests; once run() starts, only a live broker session is
        # reported as connected.
        return self._connected

    @property
    def ingress_queue_size(self) -> int:
        return self._ingress_queue.qsize()

    @property
    def dropped_messages(self) -> int:
        return self._dropped_messages

    def _enqueue_message(self, topic: str, payload: Any) -> None:
        """Bound broker ingress and prefer fresh telemetry under overload."""
        item = (topic, payload)
        if self._ingress_queue.full():
            # Telemetry is periodic and replaceable; retain control/status and
            # label frames whenever possible by evicting the oldest telemetry
            # already buffered.  The queue is intentionally bounded even when
            # every item is a control frame.
            try:
                self._ingress_queue.get_nowait()
                self._ingress_queue.task_done()
                self._dropped_messages += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self._ingress_queue.put_nowait(item)
        except asyncio.QueueFull:
            self._dropped_messages += 1

    async def _dispatch_ingress(self) -> None:
        while True:
            topic, payload = await self._ingress_queue.get()
            try:
                if self._read_callback:
                    await self._read_callback(topic, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # One bad frame must not terminate the consumer/connection.
                logger.exception("Error processing MQTT message on %s: %s", topic, exc)
            finally:
                self._ingress_queue.task_done()

    def _clear_ingress_queue(self) -> None:
        while True:
            try:
                self._ingress_queue.get_nowait()
                self._ingress_queue.task_done()
            except asyncio.QueueEmpty:
                return

    async def run(self, read_topic: Union[str, list]) -> None:
        # Reconnect loop
        self._run_started = True
        self._connected = False
        while True:
            dispatcher: Optional[asyncio.Task] = None
            try:
                async with aiomqtt.Client(
                    self.broker, port=self.port,
                    username=self.username, password=self.password
                ) as client:
                    self.client = client
                    self._connected = True
                    logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
                    topics = [read_topic] if isinstance(read_topic, str) else read_topic
                    for t in topics:
                        await client.subscribe(t, qos=1)
                        logger.info(f"Subscribed to reads on {t}")
                    dispatcher = asyncio.create_task(
                        self._dispatch_ingress(), name="mqtt-ingress-dispatcher"
                    )
                    async for message in client.messages:
                        if self._read_callback and message.topic:
                            self._enqueue_message(str(message.topic), message.payload)
            except aiomqtt.MqttError as e:
                self.client = None
                self._connected = False
                logger.error(f"MQTT connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.client = None
                self._connected = False
                self._clear_ingress_queue()
                logger.info("MQTT client task cancelled.")
                break
            except Exception as e:
                self.client = None
                self._connected = False
                logger.exception("MQTT consumer stopped unexpectedly: %s", e)
                await asyncio.sleep(5)
            finally:
                if dispatcher is not None:
                    dispatcher.cancel()
                    try:
                        await dispatcher
                    except asyncio.CancelledError:
                        pass
                self._clear_ingress_queue()

    async def publish_command(self, write_topic: str, payload: str,
                              retries: int = 3, retry_delay: float = 1.0) -> None:
        """Publish a relay command with retry logic for reliability.
        
        Safety-critical relay commands (ON/OFF) must not be silently lost
        during momentary WiFi dropouts. Retries up to `retries` times
        with exponential backoff.
        """
        for attempt in range(retries):
            if not self.client:
                if attempt < retries - 1:
                    logger.warning(
                        f"MQTT not connected, retry {attempt+1}/{retries} "
                        f"for {write_topic} in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                    continue
                logger.error(
                    f"MQTT client not connected after {retries} retries, "
                    f"relay command LOST: {write_topic} = {payload}"
                )
                return
            try:
                await self.client.publish(write_topic, payload=payload, qos=1)
                logger.debug(f"Published to {write_topic}: {payload}")
                return  # Success
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Publish failed (attempt {attempt+1}/{retries}): {e}. "
                        f"Retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Failed to publish to {write_topic} after {retries} "
                        f"attempts: {e}"
                    )


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

