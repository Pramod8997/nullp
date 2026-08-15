import aiosqlite
import asyncio
import csv
import hashlib
import logging
import os
import numpy as np
from typing import Optional, List, Tuple, Dict

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
        # Fix Issue #5: asyncio.Lock for CSV fallback writes to prevent
        # race condition with EMSOrchestrator's concurrent CSV writer
        self._csv_lock = asyncio.Lock()

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
        # Task 5: Unmapped cluster signatures for background pseudo-labeling.
        # Quantized spatial hashing prevents duplicate rows from sensor drift.
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unmapped_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_hash TEXT UNIQUE,
                mean_embedding BLOB,
                hit_count INTEGER DEFAULT 1,
                first_seen REAL,
                last_seen REAL,
                device_id TEXT,
                labeled INTEGER DEFAULT 0
            )
            """
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cluster_hash ON unmapped_clusters(cluster_hash)"
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
                        await self._csv_fallback_batch_async(batch)

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
                        await self._csv_fallback_batch_async(batch)
                raise
            except Exception as e:
                logger.error(f"Error in DB flush loop: {e}")

    def _csv_fallback_batch_sync(self, batch: List[Tuple[str, tuple]]) -> None:
        """Synchronous CSV write — safe to call from sync context or via asyncio.to_thread."""
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

    async def _csv_fallback_batch_async(self, batch: List[Tuple[str, tuple]]) -> None:
        """Non-blocking CSV fallback — runs sync file I/O in thread to avoid stalling event loop.
        Fix Issue #5: Uses asyncio.Lock to prevent race conditions with other CSV writers."""
        async with self._csv_lock:
            await asyncio.to_thread(self._csv_fallback_batch_sync, batch)

    # Keep legacy name as alias for backward compat
    _csv_fallback_batch = _csv_fallback_batch_sync

    async def _retention_loop(self) -> None:
        """Phase 2 (WS-6.1): Delete measurements older than retention_days every 24h.
        Fix: Runs VACUUM after deletion to reclaim disk space and prevent .db file bloat."""
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
                    # Fix: Reclaim disk space after bulk deletion.
                    # Without VACUUM, SQLite marks pages as free but never
                    # shrinks the file, causing infinite .db growth.
                    try:
                        await self._conn.execute("VACUUM")
                        logger.info("🗂️ VACUUM completed — disk space reclaimed.")
                    except Exception as vacuum_err:
                        logger.warning(f"VACUUM failed (non-fatal): {vacuum_err}")
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

    @staticmethod
    def _quantized_cluster_hash(mean_embedding: np.ndarray,
                                decimals: int = 3) -> str:
        """
        Generate a deterministic hash from a mean embedding vector,
        quantized to `decimals` decimal places to prevent sensor drift
        and line noise from creating duplicate database rows.
        """
        quantized = np.round(mean_embedding, decimals).astype(np.float32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]

    async def save_unmapped_cluster_signature(
        self, device_id: str, mean_embedding: np.ndarray,
        timestamp: float
    ) -> None:
        """
        Persist a stable unknown cluster signature for background pseudo-labeling.
        Uses quantized spatial hashing to dedup — identical appliances with minor
        sensor drift will match the same cluster_hash row.

        Upserts: if the cluster_hash already exists, increments hit_count and
        updates last_seen. Otherwise inserts a new row.
        """
        if not self._conn:
            logger.warning("DB not connected — cannot save unmapped cluster")
            return

        cluster_hash = self._quantized_cluster_hash(mean_embedding)
        embedding_blob = mean_embedding.astype(np.float32).tobytes()

        try:
            cursor = await self._conn.execute(
                "SELECT id, hit_count FROM unmapped_clusters WHERE cluster_hash = ?",
                (cluster_hash,)
            )
            row = await cursor.fetchone()

            if row:
                # Existing cluster — increment hit_count, update last_seen
                await self._conn.execute(
                    """
                    UPDATE unmapped_clusters
                    SET hit_count = hit_count + 1,
                        last_seen = ?,
                        device_id = ?
                    WHERE id = ?
                    """,
                    (timestamp, device_id, row[0])
                )
                logger.debug(
                    f"Cluster {cluster_hash[:8]}… hit_count → {row[1] + 1} "
                    f"for {device_id}"
                )
            else:
                # New cluster — insert
                await self._conn.execute(
                    """
                    INSERT INTO unmapped_clusters
                        (cluster_hash, mean_embedding, hit_count,
                         first_seen, last_seen, device_id, labeled)
                    VALUES (?, ?, 1, ?, ?, ?, 0)
                    """,
                    (cluster_hash, embedding_blob, timestamp,
                     timestamp, device_id)
                )
                logger.info(
                    f"New unmapped cluster {cluster_hash[:8]}… "
                    f"for {device_id}"
                )

            await self._conn.commit()

        except Exception as e:
            logger.error(f"Failed to save unmapped cluster: {e}")

    async def get_pending_clusters(
        self, min_hits: int = 5
    ) -> List[Dict[str, any]]:
        """
        Retrieve unmapped clusters with >= min_hits occurrences that have
        not yet been labeled. Used for weekly dashboard roll-up prompts.

        Returns:
            List of dicts with keys: id, cluster_hash, hit_count,
            first_seen, last_seen, device_id.
        """
        if not self._conn:
            return []

        try:
            cursor = await self._conn.execute(
                """
                SELECT id, cluster_hash, hit_count, first_seen,
                       last_seen, device_id
                FROM unmapped_clusters
                WHERE hit_count >= ? AND labeled = 0
                ORDER BY hit_count DESC
                """,
                (min_hits,)
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "cluster_hash": r[1],
                    "hit_count": r[2],
                    "first_seen": r[3],
                    "last_seen": r[4],
                    "device_id": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to query pending clusters: {e}")
            return []

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


import datetime
import json
import tempfile
from typing import Union, Any


def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    for p in ["config.yaml", "config/config.yaml"]:
        if os.path.exists(p):
            try:
                import yaml
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {
        "database": {"path": "data/ems.db", "retention_days": 30},
        "safety": {"aggregate_limit_w": 3500.0, "roc_threshold_w_per_s": 1000.0},
        "nilm": {"threshold_w": 50.0},
        "analytics": {"rates": {"peak": 0.28, "mid": 0.18, "offpeak": 0.09}},
        "phantom": {"alpha": 0.1},
        "watchdog": {"window": 30, "threshold": 3.0},
    }


class DBSession:
    def __init__(
        self,
        config: Optional[Dict] = None,
        csv_fallback_path: Optional[str] = None,
        batch_interval_s: int = 10,
        db_path: Optional[str] = None,
        retention_days: int = 30,
    ):
        self.config = config or {}
        if db_path is not None:
            self.db_path = db_path
        else:
            self._temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self.db_path = self._temp_db.name
            self._temp_db.close()

        if csv_fallback_path is not None:
            self.csv_fallback_path = csv_fallback_path
        else:
            self._temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            self.csv_fallback_path = self._temp_csv.name
            self._temp_csv.close()

        self.batch_interval_s = batch_interval_s
        self.retention_days = retention_days
        self._conn: Optional[aiosqlite.Connection] = None
        self._queue: List[Dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._conn is None:
            dirname = os.path.dirname(self.db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute("PRAGMA synchronous=NORMAL;")
            await self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS power_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device TEXT,
                    power REAL,
                    timestamp TEXT
                )
                """
            )
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
        self._initialized = True

    async def query_scalar(self, sql: str, params: tuple = ()) -> Any:
        if self._conn is None:
            await self.initialize()
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        if row is not None and len(row) > 0:
            return row[0]
        return None

    async def queue_write(self, record: Dict) -> None:
        self._queue.append(record)

    async def flush(self) -> None:
        if not self._queue:
            return
        batch = list(self._queue)
        self._queue.clear()
        try:
            await self._write_to_db(batch)
        except Exception:
            if self.csv_fallback_path:
                directory = os.path.dirname(self.csv_fallback_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(self.csv_fallback_path, "a") as f:
                    for item in batch:
                        f.write(json.dumps(item) + "\n")

    async def _write_to_db(self, records: Union[Dict, List[Dict]]) -> None:
        if isinstance(records, dict):
            records = [records]
        if not records:
            return
        if self._conn is None:
            await self.initialize()
        for r in records:
            device = r.get("device", r.get("device_id", ""))
            power = float(r.get("power", 0.0))
            ts = r.get("timestamp")
            if ts is None:
                ts = datetime.datetime.now().isoformat()
            else:
                ts = str(ts)
            await self._conn.execute(
                "INSERT INTO power_log (device, power, timestamp) VALUES (?, ?, ?)",
                (device, power, ts),
            )
        await self._conn.commit()

    async def replay_csv_fallback(self) -> None:
        if not self.csv_fallback_path or not os.path.exists(self.csv_fallback_path):
            return
        if self._conn is None:
            await self.initialize()
        with open(self.csv_fallback_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                device = record.get("device", record.get("device_id", ""))
                power = float(record.get("power", 0.0))
                ts = str(record.get("timestamp", ""))
                cursor = await self._conn.execute(
                    "SELECT 1 FROM power_log WHERE device = ? AND power = ? AND timestamp = ?",
                    (device, power, ts),
                )
                if await cursor.fetchone() is None:
                    await self._conn.execute(
                        "INSERT INTO power_log (device, power, timestamp) VALUES (?, ?, ?)",
                        (device, power, ts),
                    )
        await self._conn.commit()

    async def purge_old_records(self, retention_days: int = 30) -> None:
        if self._conn is None:
            await self.initialize()
        cutoff_dt = (
            datetime.datetime.now()
            - datetime.timedelta(days=retention_days)
            - datetime.timedelta(seconds=5)
        )
        cutoff_iso = cutoff_dt.isoformat()
        await self._conn.execute(
            "DELETE FROM power_log WHERE timestamp < ?",
            (cutoff_iso,),
        )
        await self._conn.commit()

    async def count_records(self, device: Optional[str] = None) -> int:
        if device:
            val = await self.query_scalar(
                "SELECT COUNT(*) FROM power_log WHERE device = ?", (device,)
            )
        else:
            val = await self.query_scalar("SELECT COUNT(*) FROM power_log")
        return val or 0

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

