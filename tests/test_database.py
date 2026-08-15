import datetime
import json
import os
import tempfile
from unittest.mock import patch

import aiosqlite
import pytest

from src.database.session import DBSession, load_config


# TEST 6-1: SQLite opened in WAL mode
@pytest.mark.asyncio
async def test_sqlite_wal_mode():
    session = DBSession(config=load_config())
    await session.initialize()
    journal_mode = await session.query_scalar("PRAGMA journal_mode;")
    assert journal_mode == "wal"

# TEST 6-2: CSV fallback activates on simulated DB lock
@pytest.mark.asyncio
async def test_csv_fallback_on_db_lock():
    import tempfile, os
    csv_path = tempfile.mktemp(suffix=".csv")
    session = DBSession(config=load_config(), csv_fallback_path=csv_path,
                        batch_interval_s=0)
    # Simulate a locked DB
    with patch.object(session, "_write_to_db", side_effect=Exception("DB locked")):
        await session.queue_write({"device": "node_fridge", "power": 200.0})
        await session.flush()
    
    assert os.path.exists(csv_path), "CSV fallback file should be created"
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) >= 1
    os.unlink(csv_path)

# TEST 6-3: CSV replay on restart — records re-inserted without duplicates
@pytest.mark.asyncio
async def test_csv_replay_no_duplicates():
    import tempfile, os
    csv_path = tempfile.mktemp(suffix=".csv")
    db_path  = tempfile.mktemp(suffix=".db")
    
    # Write 3 records to CSV
    records = [{"device": "x", "power": i * 100.0, "timestamp": i} for i in range(3)]
    with open(csv_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    session = DBSession(config=load_config(), csv_fallback_path=csv_path, db_path=db_path)
    await session.initialize()
    await session.replay_csv_fallback()
    
    count = await session.query_scalar("SELECT COUNT(*) FROM power_log;")
    assert count == 3
    
    # Replay again — should NOT duplicate
    await session.replay_csv_fallback()
    count2 = await session.query_scalar("SELECT COUNT(*) FROM power_log;")
    assert count2 == 3
    
    os.unlink(csv_path)
    os.unlink(db_path)

# TEST 6-4: 30-day retention — records older than 30 days are deleted
@pytest.mark.asyncio
async def test_30day_retention_deletes_old_records():
    session = DBSession(config=load_config())
    await session.initialize()
    
    # Insert a record with timestamp 31 days ago
    old_ts = (datetime.datetime.now() - datetime.timedelta(days=31)).isoformat()
    await session._write_to_db({"device": "x", "power": 100.0, "timestamp": old_ts})
    
    # Run retention purge
    await session.purge_old_records(retention_days=30)
    count = await session.query_scalar(
        "SELECT COUNT(*) FROM power_log WHERE device='x';"
    )
    assert count == 0

# TEST 6-5: Record from exactly 30 days ago is NOT deleted
@pytest.mark.asyncio
async def test_30day_retention_keeps_recent_record():
    session = DBSession(config=load_config())
    await session.initialize()
    exact_ts = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
    await session._write_to_db({"device": "x", "power": 100.0, "timestamp": exact_ts})
    await session.purge_old_records(retention_days=30)
    count = await session.query_scalar(
        "SELECT COUNT(*) FROM power_log WHERE device='x';"
    )
    assert count == 1

# TEST 6-6: Batch write groups records into one transaction
@pytest.mark.asyncio
async def test_batch_write_uses_single_transaction():
    commit_count = [0]
    original_commit = aiosqlite.Connection.commit
    
    async def counting_commit(self):
        commit_count[0] += 1
        return await original_commit(self)
    
    with patch.object(aiosqlite.Connection, "commit", counting_commit):
        session = DBSession(config=load_config(), batch_interval_s=0)
        for _ in range(10):
            await session.queue_write({"device": "x", "power": 100.0})
        await session.flush()
    
    assert commit_count[0] == 1  # all 10 records in one transaction
