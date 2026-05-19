import aiosqlite
import asyncio
import csv
import logging
import os
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class DatabaseSession:
    def __init__(self, db_path: str, fallback_csv: str = "data/fallback_measurements.csv",
                 retention_days: int = 30):
        self.db_path = db_path
        self.fallback_csv = fallback_csv
        self.retention_days = retention_days
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_queue: asyncio.Queue[Tuple[str, tuple]] = asyncio.Queue()
        self._flush_task: Optional[asyncio.Task] = None
        self._retention_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        # Phase 2 (WS-6.3): Use autoincrement ID to avoid PK collision
        # when two messages arrive with identical time.time() values
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                device_id TEXT,
                power REAL
            )
            """
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts_dev ON measurements(timestamp, device_id)"
        )
        await self._conn.commit()
        self._running = True

        # Phase 2 (WS-6.2): Replay any CSV fallback data from previous crashes
        await self._replay_csv_fallback()

        self._flush_task = asyncio.create_task(self._flush_loop())
        # Phase 2 (WS-6.1): Daily data retention cleanup
        self._retention_task = asyncio.create_task(self._retention_loop())
        logger.info(f"Database connected and WAL enabled at {self.db_path}")

    async def insert_measurement(self, timestamp: float, device_id: str, power: float) -> None:
        if not self._running:
            raise RuntimeError("Database not running")
        query = "INSERT INTO measurements (timestamp, device_id, power) VALUES (?, ?, ?)"
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
                    # Bug 4.6 fix: Removed task_done() calls — they can cause
                    # ValueError if calls get desynchronized without join()
                    
                    while not self._write_queue.empty():
                        batch.append(self._write_queue.get_nowait())
                except asyncio.TimeoutError:
                    pass

                if batch and self._conn:
                    try:
                        for query, params in batch:
                            await self._conn.execute(query, params)
                        await self._conn.commit()
                        logger.debug(f"Flushed {len(batch)} records to database.")
                    except Exception as db_err:
                        # Bug 4.1 fix: When the actual DB write fails, fall back
                        # to CSV. insert_measurement never throws (it just queues),
                        # so CSV fallback must happen HERE where the real write is.
                        logger.error(f"DB write error in flush loop: {db_err}")
                        self._csv_fallback_batch(batch)

            except asyncio.CancelledError:
                # Bug 4.2 fix: Drain ALL remaining items from the queue
                # before exiting, so no data is lost on shutdown
                while not self._write_queue.empty():
                    try:
                        batch.append(self._write_queue.get_nowait())
                    except Exception:
                        break
                if batch and self._conn:
                    try:
                        for query, params in batch:
                            await self._conn.execute(query, params)
                        await self._conn.commit()
                        logger.info(f"Flushed {len(batch)} records during shutdown.")
                    except Exception as shutdown_err:
                        logger.error(f"Shutdown flush failed: {shutdown_err}")
                        self._csv_fallback_batch(batch)
                raise
            except Exception as e:
                logger.error(f"Error in DB flush loop: {e}")

    def _csv_fallback_batch(self, batch: List[Tuple[str, tuple]]) -> None:
        """Bug 4.1 fix: Write a batch of failed DB records to CSV fallback."""
        import csv as csv_mod
        try:
            directory = os.path.dirname(self.fallback_csv)
            if directory:
                os.makedirs(directory, exist_ok=True)
            file_exists = os.path.exists(self.fallback_csv)
            with open(self.fallback_csv, 'a', newline='') as f:
                writer = csv_mod.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'device_id', 'power_watts'])
                for query, params in batch:
                    writer.writerow(list(params))
            logger.warning(f"📝 CSV fallback: wrote {len(batch)} records to {self.fallback_csv}")
        except Exception as e:
            logger.critical(f"CSV fallback write ALSO failed: {e}")

    async def _retention_loop(self) -> None:
        """Phase 2 (WS-6.1): Delete measurements older than retention_days every 24h."""
        while self._running:
            try:
                await asyncio.sleep(86400)  # Run once per day
                if self._conn and self.retention_days > 0:
                    import time
                    cutoff = time.time() - (self.retention_days * 86400)
                    cursor = await self._conn.execute(
                        "DELETE FROM measurements WHERE timestamp < ?", (cutoff,)
                    )
                    await self._conn.commit()
                    logger.info(
                        f"🗂️ Retention cleanup: deleted rows older than {self.retention_days} days "
                        f"(cutoff timestamp: {cutoff:.0f})"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retention loop error: {e}")

    async def _replay_csv_fallback(self) -> None:
        """Phase 2 (WS-6.2): Import CSV fallback data from previous crash, then archive."""
        if not os.path.exists(self.fallback_csv):
            return
        try:
            count = 0
            with open(self.fallback_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._conn:
                        await self._conn.execute(
                            "INSERT INTO measurements (timestamp, device_id, power) VALUES (?, ?, ?)",
                            (float(row['timestamp']), row['device_id'], float(row['power_watts']))
                        )
                        count += 1
            if count > 0 and self._conn:
                await self._conn.commit()
            archived = self.fallback_csv + '.imported'
            os.rename(self.fallback_csv, archived)
            logger.info(f"📥 Replayed {count} CSV fallback records into database (archived to {archived})")
        except Exception as e:
            logger.error(f"CSV fallback replay failed: {e}")

    async def close(self) -> None:
        self._running = False
        if self._retention_task:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except asyncio.CancelledError:
                pass
        if self._flush_task:
            await self._flush_task
        if self._conn:
            await self._conn.close()
        logger.info("Database connection closed gracefully.")
