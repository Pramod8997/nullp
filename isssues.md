🔴 CRITICAL / HIGH PRIORITY ISSUES
1. CSV Export: Async Database Iteration Bug
File: src/api/main.py (lines 369–387)
Severity: 🔴 High
Issue: Async iteration over aiosqlite cursor (async for row in cursor) is unreliable in many versions; may raise runtime errors or exhibit unexpected behavior.
Impact: CSV export endpoint fails or returns incomplete data; large result sets may cause OOM.
Fix: Use fetchmany() with batching instead of async iteration.
2. Race Conditions on Shared system_state
File: src/api/main.py (global state, lines 24–35)
Severity: 🔴 High
Issue: Multiple asyncio tasks (mqtt_listener_task, heartbeat_task, REST handlers, WebSocket) mutate system_state dict concurrently without synchronization. While individual dict operations are atomic, compound operations (read-modify-write, list append + trim) are NOT atomic.
Locations: Lines 101, 110, 113, 115, 129–131, 144–145, 156, 160–167, 178–181, 188.
Impact: Data corruption, lost updates, inconsistent state, rare crashes. Example: two tasks try to append to pending_labels and trim simultaneously — one trim may discard the other's append.
Fix: Add asyncio.Lock() around all compound state mutations.
3. WebSocket Broadcast: Concurrent Modification & Race Condition
File: src/api/main.py (lines 39–63, ConnectionManager class)
Severity: 🔴 High
Issues:
Iterates manager.active_connections while the list can be mutated by connect()/disconnect() from other tasks → RuntimeError.
No lock protecting active_connections list.
Sequential sending to each client is slow if many are connected; concurrent send (with asyncio.gather) is more efficient.
Impact: Broadcast crashes with "list changed during iteration"; slow real-time updates; missed broadcasts to some clients.
Fix: Use lock, snapshot list before iterating, or use concurrent send with bounded concurrency.
4. Ephemeral State Pollution in system_state
File: src/api/main.py (lines 185–189)
Severity: 🟡 Medium-High
Issue: Throttling timestamps stored as _ws_last_{device_id} keys directly in system_state dict. This pollutes persistent state with transient keys and creates name-collision risk.
Impact: Hard-to-debug state pollution, confusing API responses, potential collisions with real device IDs.
Fix: Move throttling state into separate dict or ConnectionManager; don't mix with business logic state.
5. Inconsistent Timestamp Format (string vs float)
File: src/api/main.py
Severity: 🟡 Medium-High
Issue:
DEVICE_STATUS uses event_data.get("timestamp", "") (string)
Other events use time.time() (float epoch)
system_state["devices"][did]["last_seen"] assigned from string; no normalization.
Impact: Clients cannot reliably sort/filter by time; type inconsistency in JSON responses; potential crashes if code assumes float.
Fix: Standardize all timestamps to epoch float or ISO8601 string; convert on entry.
6. MQTT Reconnection: No Exponential Backoff or Jitter
File: src/api/main.py (lines 209–211)
Severity: 🟡 Medium-High
Issue: Fixed 3-second retry interval with no exponential backoff or jitter → thundering herd when MQTT restarts.
Impact: Sudden spike in reconnect attempts; network thrashing; cascading failures.
Fix: Implement exponential backoff with jitter (e.g., 1s, 2s, 4s, 8s, capped at 60s).
7. Security: No Authentication on Control Endpoints
File: src/api/main.py (lines 316–344, submit-label endpoint)
Severity: 🔴 High
Issue: /api/submit-label (write endpoint) accepts labels with no authentication or authorization. Anyone can relabel devices or pollute the ML model training data.
Impact: Unauthorized device relabeling, ML model poisoning, loss of data integrity.
Fix: Add JWT/API key authentication, require role-based authorization (e.g., "admin"), audit log label submissions.
8. Security: CORS Overly Permissive
File: src/api/main.py (lines 238–244)
Severity: 🔴 High
Issue: allow_origins=["*"] allows any origin to call your API. Combined with no auth, this is a security risk (CSRF, XSS injection, unauthorized access).
Impact: Any website can call your endpoints; potential for abuse or data theft.
Fix: Restrict CORS to specific trusted origins (e.g., your frontend domain). Require HTTPS in production.
9. Input Validation: No Limits on Label Embeddings
File: src/api/main.py (lines 310–313, LabelSubmission model)
Severity: 🟡 Medium-High
Issue: segments: List[List[float]] = [] has no size validation. Attacker can POST enormous payloads (megabytes of floats) → OOM or DoS.
Impact: Denial of service (crash), resource exhaustion.
Fix: Validate max segment count (e.g., ≤100) and dimension (e.g., exactly 128). Reject oversized requests with 413 Payload Too Large.
10. Lifespan Shutdown: Tasks Not Awaited
File: src/api/main.py (lines 226–232)
Severity: 🟡 Medium-High
Issue: mqtt_task.cancel() and hb_task.cancel() are called but NOT awaited. Context managers and DB connections may not clean up properly. Tasks may still be running when the app exits.
Impact: Resource leaks (unclosed DB connections, MQTT sockets), data loss, unclean shutdown.
Fix: After cancel, await each task with timeout and CancelledError handling.
11. Logging: Silent Swallowing of Errors
File: src/api/main.py
Severity: 🟡 Medium
Locations: Lines 170–171 (json.JSONDecodeError pass), 194–195 (ValueError pass), 409–410 (WebSocket init Exception pass).
Issue: Exceptions silently caught with pass or generic except Exception without logging → impossible to debug.
Impact: Silent failures; hard to diagnose root causes in production.
Fix: Log at least logger.debug() with context (topic, payload size, etc.).
12. Redundant Dependencies: Both aiomqtt and amqtt
File: requirements.txt (lines 4, 17)
Severity: 🟡 Medium
Issue: Two different MQTT client libraries listed. Code imports aiomqtt, not amqtt. This wastes disk/image space and confuses dependencies.
Impact: Bloated container image, confusion for future maintainers.
Fix: Remove unused package (likely amqtt).
🟡 MEDIUM PRIORITY ISSUES
13. No Database Connection Pooling / Reuse
File: src/api/main.py (lines 328–334, 370)
Severity: 🟡 Medium
Issue: Each label submission and CSV export creates a new MQTT client (async with aiomqtt.Client(...)) and DB connection. No connection pooling.
Impact: High latency, connection churn, potential "too many open files" errors.
Fix: Create singleton/global MQTT client and DB connection pool at startup; reuse connections.
14. Health Check Incomplete (No Dependency Checks)
File: src/api/main.py (lines 253–255)
Severity: 🟡 Medium
Issue: /health returns 200 OK always, even if MQTT or DB is down. No readiness/liveness probe distinction.
Impact: Orchestrators (Kubernetes, Docker Swarm) think app is healthy when downstream services fail → cascading failures.
Fix: Add /ready (checks dependencies), /live (checks self only), include dependency status in response.
15. No Rate Limiting on Endpoints
Severity: 🟡 Medium
Issue: No rate limiting on any endpoint → DoS vulnerability. Attacker can spam /api/submit-label or /api/export-csv.
Impact: Resource exhaustion, service unavailability.
Fix: Add rate limiting middleware (e.g., slowapi); per-IP or per-API-key limits.
16. WebSocket: No Per-Client Backpressure or Buffer Management
Severity: 🟡 Medium
Issue: ws.send_json() with timeout mitigates some issues, but no per-client buffer or acknowledgment flow control. Slow clients may cause memory buildup.
Impact: Slow clients starve fast clients; potential memory leak.
Fix: Implement per-client message queue with max size; drop old messages or disconnect slow clients.
17. No Structured Logging / Metrics
Severity: 🟡 Medium
Issue: Logging is ad-hoc print statements, not structured (JSON). No Prometheus metrics for monitoring.
Impact: Hard to parse logs in production; no visibility into performance or issues.
Fix: Use structured logging (json logs) and export Prometheus metrics (MQTT connections, WS clients, message rates, latencies).
18. No Audit Logging for Safety-Critical Events
Severity: 🟡 Medium
Issue: Safety warnings, mitigations, and cutoffs logged locally but no persistent audit trail.
Impact: Cannot trace who relabeled a device or when a mitigation was triggered; regulatory/compliance risk.
Fix: Log all label submissions, mitigations, and safety events to a dedicated audit log (DB or syslog).
19. No Tests or Continuous Integration
Severity: 🟡 Medium
Issue: No pytest tests for CSV export, MQTT listener, broadcast, submit_label. No CI/CD pipeline.
Impact: Regressions go undetected; hard to refactor safely.
Fix: Add unit/integration tests (pytest-asyncio); set up GitHub Actions CI.
🟢 LOW PRIORITY / NICE-TO-HAVE
20. Inconsistent Field Naming: total_phantom vs total_watts
File: src/api/main.py
Locations: Lines 27 ("total_phantom": 0.0), 275 ("total_watts": system_state["total_phantom"]), 286.
Severity: 🟢 Low
Issue: API field name and internal state name don't match.
Fix: Standardize to one name throughout (e.g., total_phantom_watts).
21. Missing Type Hints on system_state
Severity: 🟢 Low
Issue: system_state is plain dict with no type hints. Hard to track what fields exist and their types.
Fix: Define a TypedDict or dataclass for system_state structure; use mypy for type checking.
22. Lifespan Errors Not Caught / Logged
File: src/api/main.py (lines 226–232)
Severity: 🟢 Low
Issue: If mqtt_listener_task or heartbeat_task raise errors, the app may not handle them gracefully.
Fix: Wrap task creation in try/except and log errors; optionally retry or gracefully degrade.
23. WebSocket Init State May Be Sent Multiple Times
File: src/api/main.py (lines 400–410)
Severity: 🟢 Low
Issue: Init state sent on connection but also in heartbeat. Client may receive stale/duplicate init data.
Fix: Send init state once; client can request full state via explicit message if needed.
24. No Request/Response Validation & Serialization Errors
Severity: 🟢 Low
Issue: No explicit validation of MQTT payloads; no serialization error handling for JSON responses.
Fix: Add Pydantic models for all MQTT event types; validate on deserialize.
25. Container / Deployment Not Production-Ready
File: Dockerfile, docker-compose.yml
Severity: 🟢 Low
Issue: No info provided, but typical issues: base image too large, no health checks, no secrets management, torch/h5py bloat.
Fix: Use slim base, multi-stage build, add HEALTHCHECK, use .env for secrets, separate ML workloads.
SUMMARY TABLE
#	Issue	Severity	File	Type
1	CSV Export Async Iteration	🔴 High	main.py:369–387	Correctness
2	Race Conditions on system_state	🔴 High	main.py:24–35, multi	Concurrency
3	WebSocket Broadcast Concurrent Modification	🔴 High	main.py:39–63	Concurrency
4	Ephemeral State Pollution	🟡 Med-High	main.py:185–189	Design
5	Inconsistent Timestamp Format	🟡 Med-High	main.py:multi	Correctness
6	MQTT Reconnection No Backoff	🟡 Med-High	main.py:209–211	Robustness
7	No Auth on Control Endpoints	🔴 High	main.py:316–344	Security
8	CORS Allow *	🔴 High	main.py:238–244	Security
9	No Input Validation on Embeddings	🟡 Med-High	main.py:310–313	Security
10	Lifespan Tasks Not Awaited	🟡 Med-High	main.py:226–232	Robustness
11	Silent Exception Swallowing	🟡 Medium	main.py:170, 194, 409	Observability
12	Redundant Dependencies	🟡 Medium	requirements.txt:4,17	Hygiene
13	No Connection Pooling	🟡 Medium	main.py:328, 370	Performance
14	Incomplete Health Check	🟡 Medium	main.py:253–255	Observability
15	No Rate Limiting	🟡 Medium	main.py:global	Security
16	No WebSocket Backpressure	🟡 Medium	main.py:39–63	Robustness
17	No Structured Logging	🟡 Medium	main.py:global	Observability
18	No Audit Logging	🟡 Medium	main.py:316+	Compliance
19	No Tests / CI	🟡 Medium	tests/	Quality
20	Inconsistent Field Names	🟢 Low	main.py:27, 275, 286	Hygiene
21	Missing Type Hints	🟢 Low	main.py:24	Code Quality
22	Lifespan Errors Not Logged	🟢 Low	main.py:226	Observability
23	Duplicate Init State	🟢 Low	main.py:400–410	Logic
24	No Validation Models for MQTT	🟢 Low	main.py:88–207	Design
25	Deployment Not Production-Ready	🟢 Low	Dockerfile, compose	DevOps