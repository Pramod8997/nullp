# Physical Bring-Up Runbook

> Companion to the hardware readiness audit. Gates 1-4 are **DONE** (software).
> Gates 5-12 need the rig on the bench and are **NOT EXECUTED**.
>
> Rule for every gate: **do not proceed past a failed safety or hardware test.**
> Record PASS/FAIL and the actual numbers, not "looked fine".

---

## PROCEDURAL RULES (read before every bench session)

* 🔴 **Never run `backend/scripts/simulate_esp32.py` while the physical rig is live.**
  It ACKs commands for **ANY device id** — including `node_bench_agg`, clearing the
  pipeline's cooldowns with fake hardware ACKs — and injects simulated telemetry on the
  same topics the real PZEM publishes to. One stray run mid-demo corrupts the demo and
  the safety state.
* Exactly **one broker** may own port 1883 (see the Gate 0 pre-step). Two listeners =
  silently split traffic.
* Every tool that talks to the authed broker needs credentials — including
  `mosquitto_sub`/`mosquitto_pub` and `scripts/calibrate_ct.py` (Gate 6 note).

---

## Completed before the bench session

| Gate | Status | Evidence |
| :--- | :--- | :--- |
| 1. Compile firmware | ✅ PASS | `firmware.bin` 756,989 B (57.8% flash), RAM 45,304 B (13.8%); `src/main.cpp.o` rebuilt with zero warnings; PZEM-004T-v30 @ 1.1.2, PubSubClient @ 2.8.0 |
| 2. Fix compile errors | ✅ N/A | none existed |
| 3. Network config | ✅ code ready | credentials moved to gitignored `include/secrets.h`; `#error` guard fires if absent |
| 4. MQTT auth | ✅ PASS | pipeline + API now pass `MQTT_USERNAME`/`MQTT_PASSWORD`; anonymous path unchanged; 527 tests pass |

Toolchain lives in an isolated venv: **`/tmp/pio-venv/bin/pio`** (kept out of `.venv`
because installing it there downgrades `uvicorn`/`starlette`). Recreate with
`python3 -m venv /tmp/pio-venv && /tmp/pio-venv/bin/pip install platformio`.

---

## GATE 0 — Broker: accept the ESP32 (needs sudo, no mains)

The system broker is bound to loopback, so no ESP32 can reach it. Decision taken:
**bind 0.0.0.0 with password auth**, canonical user **`ems_pipeline`**.

**Pre-step — exactly one broker owner.** The docker-compose stack, the system service and
`scripts/start_broker.py` all want port 1883; two of them running means silently split
traffic (the collision is verification finding BROKER-1). Before anything else:

```bash
docker ps --format '{{.Names}}  {{.Ports}}' | grep 1883   # if the compose broker is up:
docker compose down                                        # stop it before the bench session
systemctl status mosquitto --no-pager                      # confirm which broker is live
ss -ltn | grep 1883                                        # must show exactly ONE listener
```

```bash
# 1. Pick a broker password and create the user (you will type it twice)
sudo mosquitto_passwd -c /etc/mosquitto/passwd ems_pipeline

# 1b. Dedicated ESP32 device user — the firmware must NOT hold the pipeline credential
#     (with only the shared user, anyone holding the one password can publish
#     home/plug/node_bench_agg/command, i.e. toggle mains). Set EMS_MQTT_USER in
#     secrets.h to esp32.
sudo mosquitto_passwd /etc/mosquitto/passwd esp32

# 2. Open the listener and require auth, and enforce the ACL grant matrix
sudo tee /etc/mosquitto/conf.d/ems.conf >/dev/null <<'EOF'
listener 1883 0.0.0.0
allow_anonymous false
password_file /etc/mosquitto/passwd
acl_file /etc/mosquitto/acl
EOF

# 2b. System-broker equivalent of the repo ACL: copy the grant matrix and deploy it.
#     The repo's mosquitto/config/acl (docker path) is the authority — the same
#     matrix applies here. Grants, in full (a sketchy ACL breaks the stack):
#       esp32       read  home/plug/+/command;
#                   write home/sensor/+/power, home/sensor/+/telemetry,
#                               home/sensor/+/status, home/plug/+/ack
#       ems_pipeline read  home/sensor/#, home/plug/+/ack, home/ml/label, $SYS/#
#                   write home/plug/+/command, home/ui/events, home/ml/label
#     ($SYS/# read is what the docker healthcheck needs; the home/ml/label write is
#     what the API label loop needs. Do not trim either.)
sudo cp mosquitto/config/acl /etc/mosquitto/acl
sudo chown mosquitto:mosquitto /etc/mosquitto/acl

sudo systemctl restart mosquitto
ss -ltn | grep 1883          # must now show 0.0.0.0:1883, not 127.0.0.1:1883
ip -4 addr show scope global | grep -oP 'inet \K[\d.]+'   # note this LAN IP
```

**PASS** = `0.0.0.0:1883` listening, and from another machine on the same subnet:
`mosquitto_sub -h <LAN_IP> -u ems_pipeline -P <pw> -t 'home/#' -v` connects.

> ⚠ The backend now needs the same credentials in its environment:
> `export MQTT_USERNAME=ems_pipeline MQTT_PASSWORD=<pw>` before
> `python scripts/run_pipeline.py --config config/config.hardware.yaml`.
> Without them the pipeline will log `MQTT connection error ... Reconnecting`.
>
> ⚠ This host's current LAN IP is in the `172.20.10.x` range, which is a phone
> hotspot. That address changes every session — re-check it each time and keep
> `EMS_MQTT_SERVER` in step 1 in sync.

---

## GATE 1b — secrets.h (no mains)

```bash
cp firmware/esp32_node/include/secrets.h.example \
   firmware/esp32_node/include/secrets.h
$EDITOR firmware/esp32_node/include/secrets.h
```

Fill in: 2.4 GHz SSID (**the ESP32 has no 5 GHz radio**), Wi-Fi password, the LAN
IP from Gate 0, the **`esp32` user's** password from Gate 0 step 1b. Set
`EMS_MQTT_USER` to `esp32` (the dedicated device user — the firmware must not hold
the pipeline credential) and `EMS_DEVICE_ID` to `node_bench_agg` — the latter must
match `devices:` in `config/config.hardware.yaml` or the backend silently ignores
every reading.

```bash
cd firmware/esp32_node && /tmp/pio-venv/bin/pio run     # PASS = SUCCESS
```

---

## GATE 5 — First packet (USB power only, NO MAINS on the PZEM)

```bash
cd firmware/esp32_node
/tmp/pio-venv/bin/pio run --target upload
/tmp/pio-venv/bin/pio device monitor -b 115200
```

Serial log must show, in order:
`[INIT] Core 0: SafetySamplingTask launched` → `[WiFi] Connected: <ip>` →
`[INIT] Device ID: node_bench_agg` → `[MQTT] Connected`.

Then from the backend host:
```bash
mosquitto_sub -h localhost -u ems_pipeline -P <pw> -t 'home/#' -v
```

**PASS** = `home/sensor/node_bench_agg/status ONLINE` plus
`home/sensor/node_bench_agg/power` arriving about once per second.

**If `[MQTT] Failed rc=`**: `rc=-2` is network/DNS, `rc=4` is bad credentials,
`rc=5` is not authorised. Fix the credential or IP, do not move on.

**Expect zeros or NaN-skips from the PZEM here** — no mains is connected yet.
Core 0 skips non-finite reads (the `isnan()` guard in `SafetySamplingTask`), so the power topic
may publish `0.00`. That is correct at this gate.

---

## GATE 6 — Real PZEM readings (MAINS LIVE — ⚠ safety gate)

> **STOP.** Everything from here involves 230 V. Do not proceed without the
> locked wiring of `HARDWARE_FINAL_SPEC.md`: PZEM 10 A direct-connect, SRD 10 A
> relay, 100 kΩ IN→GND pull-down, 5 A load fuse, earthed 3-pin socket.
> Confirm the relay is OPEN (Gate 7 below) *before* energising a load.

Single resistive load first: the 100 W incandescent lamp (spec BOM item 19,
PF = 1, no SMPS soft-start).

Check `home/sensor/node_bench_agg/telemetry` (JSON `{v,i,w,pf}`, every 10 s)
against a reference meter.

**PASS** = V within a few % of the reference, W within ~5%, PF ≈ 1.00,
I ≈ W / V. **FAIL** = zeros (UART wiring: RX/TX swapped on GPIO 16/17), or
constant values (PZEM not refreshing).

> If W is off by > 5 %, run the calibration tool with the 100 W lamp as the
> reference — **it needs credentials against the authed broker** (its defaults are
> the wrong user `pipeline` / a placeholder password, so auth fails silently into
> a connection error):
> `python scripts/calibrate_ct.py node_bench_agg --ref-power 100 --username ems_pipeline --password <pw>`
> (or `export MQTT_USERNAME=ems_pipeline MQTT_PASSWORD=<pw>` — same effect).

> (Resolved 2026-09-10: the backend now subscribes `home/sensor/+/telemetry`
> and broadcasts V/I/PF to the dashboard — read them there, or with
> `mosquitto_sub` for this gate.)

---

## GATE 7 — Relay, dry, NO MAINS (⚠ do this BEFORE Gate 6 if at all possible)

Continuity meter across the relay's load contacts, nothing plugged in.

1. Power-on / reset → contacts must read **OPEN**. The pre-Wi-Fi `setRelay(false)` call in
   `setup()` and the 100 kΩ pull-down holds the input low while
   GPIO 18 is high-impedance during boot.
2. `mosquitto_pub -t home/plug/node_bench_agg/command -m ON` → **CLOSED**, and
   `home/plug/node_bench_agg/ack` publishes `ON_CONFIRMED`.
3. `-m OFF` → **OPEN**, ack `OFF_CONFIRMED`.

**PASS** = all three. **FAIL on step 1 = stop entirely**: an inverted relay
energises the load at boot and makes every safety cutoff *close* the contacts
(defect B-7). Do not connect mains until step 1 passes.

---

## GATE 8 — Measure the actual Core 0 loop period

This decides audit item **D6** and cannot be observed from outside the node.
Apply this temporary instrumentation, measure, then **revert it**:

```c
// TEMPORARY — inside SafetySamplingTask, right after `lastReadMs = nowMs;`
static uint32_t n = 0; static uint32_t sum = 0, mn = 9999, mx = 0;
uint32_t d = (uint32_t)(dt * 1000.0f);
sum += d; if (d < mn) mn = d; if (d > mx) mx = d;
if (++n % 100 == 0) Serial.printf("[LOOP] n=%u mean=%.1fms min=%u max=%u\n",
                                  n, sum / (float)n, mn, mx);
```

Record mean / min / max over 60 s. Also note how many consecutive PZEM reads
return byte-identical values (the sensor refreshes slower than the 100 ms poll).

**Why it matters:** the arc-fault channel trips when
`Δwatts / period_s > 1000`. At a 100 ms period, **any step above 100 W trips**.
At 150 ms, above 150 W. Write the number down before Gate 9.

---

## GATE 9 — Arc-fault behaviour (⚠ expect a trip; safety gate)

Inrush suppression only applies while the 5-sample baseline is **below 50 W**
(the `BASELINE_INRUSH_CEIL` constant, via `isNormalInrush`). The rig's intended baseline is 65-220 W, so
suppression is off for every load added after the first.

| test | action | predicted |
| :--- | :--- | :--- |
| 9a | idle socket → switch in the 100 W lamp | holds (baseline < 50 W) |
| 9b | lamp running → switch in a ~120 W load | **TRIP + 5-minute lockout** |

On the digital twin (same constants as `main.cpp`): 65 W → 185 W = 1200 W/s →
relay OPEN, `relay_locked = True`.

**Record what actually happens. Do NOT change `EDGE_ROC_THRESHOLD` or
`CRITICAL_PCT` to make 9b pass.** If 9b trips, stop and report: the honest lever
is the *suppression scope* (`BASELINE_INRUSH_CEIL = 50 W` never covered this
rig's operating range), and that is a safety decision for the owner, not a
tuning knob. Sequencing the demo one-load-from-idle also avoids it.

---

## GATE 10 — Overcurrent cutoff (the headline safety demo, ⚠ mains)

Ramp the socket past **312.5 W** (`RATED_WATTS 250 × CRITICAL_PCT 1.25`).

**PASS** = relay opens on the first sample above 312.5 W; serial shows
`[CORE0] ⚡ OVERCURRENT!`; the 5 A fuse (1150 W, 3.7× margin) does **not** blow.
Repeat with Wi-Fi disconnected — the cutoff must still fire, because Core 0 is
launched before Wi-Fi and never depends on it.

---

## GATE 11 — Capture real windows (one appliance at a time, ⚠ mains)

One load on an otherwise idle socket, steady state, left running. A window is
128 samples at 1 Hz = **128 s**; strided windows cost ~4 min per class.

```bash
export MQTT_USERNAME=ems_pipeline MQTT_PASSWORD=<pw>
for CLS in phone_charger laptop projector monitor; do
  echo ">>> plug in ONLY the $CLS, leave it running, then press Enter"; read
  python scripts/capture_bench_windows.py --class $CLS --out bench.npz \
      --windows 8 --attest-physical "REPLACE: what was plugged in, reference meter reading"
done
```

`--attest-physical` is not optional in spirit: without it the file is stamped
`UNATTESTED`, because the transport alone cannot distinguish a real PZEM from
`simulate_esp32.py` publishing to the same topic.

**PASS** = 8 windows per class, `measured publish rate ≈ 1 Hz` (a large deviation
means dropped messages), and a median steady watts that matches your reference
meter.

---

## GATE 12 — Enrol and test physical recognition

```bash
python scripts/enroll_demo_devices.py --capture bench.npz \
    --out backend/models/weights_demo/prototype_registry_bench.pt
```

Confirm the printed provenance shows `✅ operator attestation` for all four, then
add to `config/config.hardware.yaml` under `protonet:`:

```yaml
  registry_path: "backend/models/weights_demo/prototype_registry_bench.pt"
```

Restart the pipeline and present each appliance again, **one at a time on an
idle socket**.

**PASS** = each of phone / laptop / projector / monitor is named correctly, and a
load that is none of them reports `unknown`.

**Known limits — record these as expected, not as failures:**
- **Simultaneous loads will be wrong.** With one load already running,
  `steady_w` is the median over the whole 128 s window, i.e. the *pre-event*
  aggregate. Measured: 65 W running + 120 W added → `steady_w` = 65 W → reported
  `tv` at confidence 1.000. `OverlapAwareNILMDetector` is exported but never
  instantiated (audit D4 / open item M-2). One load at a time is the supported
  case tonight.
- **Equal-wattage loads are not separable.** A 1 Hz power window carries no
  scale-independent shape cue and the embedding is power-scale-blind, so any
  unknown load inside an enrolled envelope takes that label.

---

## Reporting template

```
GATE n — <name>
  RESULT:   PASS | FAIL
  EVIDENCE: <measured numbers / serial excerpt / mosquitto_sub output>
  NEXT:     <next gate, or STOP + reason>
```


---

## DEMO-DAY NOTES (added 2026-09-10, wave-3 close-out)

1. **Viewing the dashboard from another device (LAN):** run the frontend with
   `npm run dev -- --host` AND export `CORS_ORIGINS=http://<lan-ip>:5173` for
   the API — the WS origin check reuses the CORS allowlist, and the defaults
   cover only localhost/127.0.0.1. Without this the page loads but the live
   event stream is silently rejected.
2. **Same-socket event spacing:** space plug-in/unplug events on one socket at
   least ~20 s apart (the delta-overlap baseline refills over a 30-sample
   history; two events closer than the stability window land inside one
   pending delta event and are treated as near-simultaneous — out of scope).
3. **Sequencing:** projector-first (cold start is inrush-suppressed); never
   add the projector onto a >= 50 W running baseline (arc-fault trip,
   expected firmware behavior — see GOD_TIER_PLAN C3).
