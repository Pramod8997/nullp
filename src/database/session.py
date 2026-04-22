import aiosqlite
import asyncio
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class DatabaseSession:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_queue: asyncio.Queue[Tuple[str, tuple]] = asyncio.Queue()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                timestamp REAL,
                device_id TEXT,
                power REAL,
                PRIMARY KEY (timestamp, device_id)
            )
            """
        )
        await self._conn.commit()
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"Database connected and WAL enabled at {self.db_path}")

    async def insert_measurement(self, timestamp: float, device_id: str, power: float) -> None:
        if not self._running:
            raise RuntimeError("Database not running")
        query = "INSERT OR REPLACE INTO measurements (timestamp, device_id, power) VALUES (?, ?, ?)"
        await self._write_queue.put((query, (timestamp, device_id, power)))

    async def _flush_loop(self) -> None:
        while self._running or not self._write_queue.empty():
            try:
                batch: List[Tuple[str, tuple]] = []
                try:
                    if self._write_queue.empty() and not self._running:
                        break
                    # Wait for 10 seconds for batching, as per INV-6
                    item = await asyncio.wait_for(self._write_queue.get(), timeout=10.0)
                    batch.append(item)
                    self._write_queue.task_done()
                    
                    while not self._write_queue.empty():
                        batch.append(self._write_queue.get_nowait())
                        self._write_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                if batch and self._conn:
                    for query, params in batch:
                        await self._conn.execute(query, params)
                    await self._conn.commit()
                    logger.debug(f"Flushed {len(batch)} records to database.")

            except Exception as e:
                logger.error(f"Error in DB flush loop: {e}")

    async def close(self) -> None:
        self._running = False
        if self._flush_task:
            await self._flush_task
        if self._conn:
            await self._conn.close()
        logger.info("Database connection closed gracefully.")
