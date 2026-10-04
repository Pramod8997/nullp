# Safety Findings — Agent 2

**Scope:** H1/H2/H3 only: dead-at-boot PZEM behavior, actuation ownership and
races, lockout/stale or retained `ON`, network-loss behavior, PZEM transaction
freshness/timing, and safety-task creation/reboot behavior.

**Physical validation:** **NOT RUN.** All findings below are from simulator
execution, static firmware inspection, and existing test fixtures. No mains,
relay contacts, PZEM UART timing, brownout waveform, or ESP32 task execution was
physically exercised.

## H1 — Dead PZEM at boot can still accept `ON`

**Root cause:** The actuation path does not require a valid/fresh PZEM state.
`callback()` accepts `ON` when only `relayLocked == false` and immediately calls
`setRelay(true)` (`main.cpp:396-405`). The Core 0 loss watchdog is gated by
`pzemEverValid` (`main.cpp:263-283`), so a node whose PZEM has never produced a
finite read can accumulate/saturate `pzemFailCount` indefinitely without
opening the relay. These two gates leave the never-valid path able to energize
the socket without a measurement.

**Current evidence:**

- Failing reproducer:
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_dead_pzem_at_boot_cannot_accept_on_command -q --tb=short`
  → **FAIL**. After `ON` and 60 invalid Core 0 steps, the simulator reports
  `gpio18_relay_state is True`.
- The same behavior is encoded by the passing focused test
  `test_pzem_watchdog_stays_disarmed_until_first_valid_read` and by
  `test_bringup_gate7_dry_relay_close_stays_closed`. Those tests document the
  current `pzemEverValid` exception; they do not prove the H1 safety invariant.
- The network-loss/heartbeat checks
  `test_wifi_drop_safety_continues`, `test_keepalive_timeout_handling`, and
  `test_network_partition_split_brain` pass in simulation for an already-valid
  overcurrent path. Firmware `SERVER_TIMEOUT` only logs/publishes status
  (`main.cpp:604-614`); it does not de-energize the relay. After reconnect,
  `client.subscribe(topicCommand)` can deliver a retained `ON`, and the same
  callback has no measurement-health gate.
- Reboot initialization does call `setRelay(false)` before Wi-Fi/MQTT
  (`main.cpp:419-437`), and simulator reboot/boot tests pass. That establishes
  the software initial state only; it does not prevent a later stale/retained
  `ON` from closing the relay.

**Evidence class:** **Simulator + static evidence only.** The failure is
deterministic in the digital twin; no physical relay or PZEM was used.

**Exact next test needed:** Add/run a deterministic retained-command
reproducer with this sequence: construct/reboot the node with every PZEM value
invalid; allow at least `PZEM_FAIL_TRIP_COUNT * 2` Core 0 cycles; simulate
network loss and reconnect; deliver a retained `ON` before any finite PZEM
sample; assert the physical relay GPIO remains at its de-energizing level, no
`ON_CONFIRMED` is emitted, and the node reports the sensor fault. Then repeat
the same sequence on the qualified bench with the relay contacts metered and
**no mains connected** before any energized test is considered.

## H2 — Core 0 cutoff and Core 1 `ON` have competing actuation authority

**Root cause:** Both cores can call `setRelay()`, but lockout is established
only later by Core 1. Core 0 opens the relay and raises
`sharedOvercurrentLatch` (`main.cpp:349-355`); before Core 1 consumes that latch
(`main.cpp:563-581`), the Core 1 MQTT callback checks only `relayLocked` and can
call `setRelay(true)` (`main.cpp:396-400`). There is therefore no single
authoritative actuation owner or atomic safety-inhibit check spanning cutoff,
lockout, and `ON` handling.

**Current evidence:**

- Failing reproducer:
  `PYTHONPATH=venv/lib/python3.10/site-packages:. /usr/bin/python3.10 -m pytest tests/test_audit_reproductions.py::test_safety_cutoff_wins_over_on_before_core1_latch_tick -q --tb=short`
  → **FAIL**. Core 0 first opens the relay and sets the latch; `ON` then
  re-closes it before the Core 1 latch tick, and the final simulator state is
  `gpio18_relay_state is True`.
- The focused tests that wait for Core 1 to consume the latch pass:
  `test_brief_overcurrent_spike_still_takes_lockout_and_nacks_on`,
  `test_relay_command_during_overcurrent_cutoff`, and
  `test_lockout_rejects_on_command`. They verify the post-consumption lockout,
  not the inter-core window.
- `test_core0_core1_relay_race` also passes, but its ordering is
  Core 0 cutoff → Core 1 telemetry tick → `ON`; it does not reproduce the
  failing cutoff → `ON` → Core 1 tick interleaving.
- Network-loss tests pass only for the local Core 0 cutoff in the simulator;
  loss of MQTT does not remove the two firmware actuation callers or make the
  Core 1 lockout update atomic.

**Evidence class:** **Simulator + static evidence only.** No dual-core ESP32
scheduling trace or physical relay observation was obtained.

**Exact next test needed:** Run a deterministic firmware interleaving test
that forces `SafetySamplingTask` to execute the overcurrent cutoff, injects an
`ON` callback before the next `loop()` latch-consumption block, and then runs
the latch block. Assert all of: GPIO is open/de-energizing, `relayLocked` is
true, the `ON` acknowledgement is `LOCKOUT_NACK` (never
`ON_CONFIRMED`), and a subsequent retained/reconnect `ON` is also rejected.
Repeat with the MQTT broker disconnected so the cutoff must still complete;
then verify the actual relay contact remains open on the qualified bench.

## H3 — PZEM watchdog timing is cycle-count based, not transaction/freshness based

**Root cause:** The firmware treats one loop containing four PZEM getter calls
as one 100 ms sample. It calls `pzem.power()`, `pzem.voltage()`,
`pzem.current()`, and `pzem.pf()` sequentially before taking `nowMs`
(`main.cpp:224-234`), then delays another 100 ms (`main.cpp:296` or
`main.cpp:376`). `pzemFailCount` increments once per aggregate loop and resets
on any finite getter result (`main.cpp:243-300`). There is no timestamp for the
last successful transaction, no per-transaction timeout budget in this code,
and no check that the returned finite values are newer than the last successful
measurement. Consequently, the claimed `30 * 100 ms = 3 s` blind bound is an
assumption: actual blind time includes all library UART transaction/timeout
durations, and a finite cached/stale response can reset the watchdog.

The same gap extends to task health: `xTaskCreatePinnedToCore()` is launched
before Wi-Fi, but its return value and task handle are discarded
(`main.cpp:428-439`). `loop()` therefore has no evidence that the Core 0 safety
task was created or is still running before it accepts commands. Reboot does
start with the relay off, but there is no retained measurement-health gate
between reboot and a later `ON`.

**Current evidence:**

- No existing failing test captures H3. `FAIL-H3` remains open in
  `docs/omnirush/FAILURE_LOG.md`; the current test matrix explicitly requires
  a PZEM timeout/measurement timing trace.
- Passing tests are model/parity checks only:
  `test_pzem_watchdog_trips_at_exactly_the_trip_count`,
  `test_pzem_fail_trip_count_matches_twin`, and
  `test_twin_roc_dt_clamp_matches_pzem_cadence`. They assume the simulator's
  100 ms stepping / fixed 0.134–0.16 s ROC clamp and do not measure the
  PZEM004Tv30 library's four real UART transactions or timeout behavior.
- The selected UART corruption tests pass, but
  `tests/test_hil_uart_corruption.py` states that it has no raw Modbus parser
  and injects register values directly. It cannot establish transaction
  completion time, stale-register age, or task liveness.
- `BRINGUP_RUNBOOK.md` Gate 8 requires temporary measurement of Core 0 loop
  mean/min/max and consecutive byte-identical PZEM reads over 60 s; that gate
  is documented but has not been executed. Gate 6 also says constant readings
  indicate that the PZEM is not refreshing.
- The existing simulator watchdog tests pass, including
  `test_pzem_watchdog_trips_at_exactly_the_trip_count`; this proves only the
  simulator's assumed count, not the physical elapsed-time bound.

**Evidence class:** **Static + simulator evidence only.** H3 is presently an
untested timing/freshness risk, not a physically reproduced 15 s (or other)
cutoff measurement. The exact elapsed value must not be claimed until the
bench trace exists.

**Exact next test needed:** Execute `BRINGUP_RUNBOOK.md` Gate 8 on the target
ESP32/PZEM with temporary, reverted instrumentation around each getter and the
full Core 0 iteration: record per-getter duration, total iteration period,
PZEM refresh/byte-identical runs, and timeout values for at least 60 s. After
one known-good measurement, interrupt the PZEM UART or sensor supply under a
qualified controlled bench condition and timestamp the last successful
transaction and relay de-assertion. Assert that relay opening occurs within
the approved blind-time limit based on the measured successful-transaction
age, that stale finite values do not reset freshness, and that the safety task
is confirmed running after boot/task creation. Record raw serial/GPIO timing
artifacts; without those artifacts H3 remains **NOT RUN**.

## Summary

| Issue | Current result | Evidence | Physical status |
|---|---|---|---|
| H1 | **OPEN; reproduced** | Failing simulator reproducer plus static callback/watchdog trace | NOT RUN |
| H2 | **OPEN; reproduced** | Failing simulator interleaving reproducer plus static dual-owner trace | NOT RUN |
| H3 | **OPEN; timing not reproduced** | Static transaction/freshness gap plus simulator-only parity tests | NOT RUN |
