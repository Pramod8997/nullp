import React from 'react';
import { test, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DeviceCards from '../components/DeviceCards';
import SafetyAlerts from '../components/SafetyAlerts';
import DigitalTwin from '../components/DigitalTwin';
import SystemStatus from '../components/SystemStatus';
import ApplianceTable from '../components/ApplianceTable';
import AlertsPage from '../pages/AlertsPage/AlertsPage';
import SummaryCards from '../components/SummaryCards/SummaryCards';
import EnergyChart from '../components/EnergyChart/EnergyChart';

// TEST 9A-1: Renders N cards for N devices
test("renders one card per device", () => {
    const devices = {
        node_fridge: { power: 200, state: "ON", label: "Fridge" },
        node_hvac:   { power: 2000, state: "ON", label: "HVAC" },
    };
    render(<DeviceCards devices={devices} />);
    expect(screen.getAllByTestId("device-card")).toHaveLength(2);
});

// TEST 9A-2: Shows "No Devices" when empty
test("shows empty state when no devices", () => {
    render(<DeviceCards devices={{}} />);
    expect(screen.getByText(/no devices/i)).toBeInTheDocument();
});

test("does not invent energy cost or savings without backend analytics", () => {
    render(<SummaryCards devices={{ node_a: { power: 100 } }} powerHistory={[{ node_a: 100 }]} />);
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/15% optimized/i)).not.toBeInTheDocument();
});

test("does not generate historical energy without measured history", () => {
    render(<EnergyChart powerHistory={[]} />);
    expect(screen.getByText(/no measured energy history/i)).toBeInTheDocument();
});

// TEST 9A-3: HIGH power state triggers glow CSS class
test("glow effect applied when power > 80% of rated", () => {
    const devices = { node_kettle: { power: 2200, state: "ON", rated: 2500, label: "Kettle" } };
    render(<DeviceCards devices={devices} />);
    const card = screen.getByTestId("device-card-node_kettle");
    expect(card.className).toMatch(/glow/i);
});

// TEST 9A-4: "OFF" state shows power as 0W
test("OFF device shows 0W", () => {
    const devices = { esp32_tv: { power: 0, state: "OFF", label: "TV" } };
    render(<DeviceCards devices={devices} />);
    expect(screen.getByText(/0.*W/i)).toBeInTheDocument();
});

// TEST 9A-5: PZEM telemetry (V/I/PF) line renders when present
test("shows V/I/PF telemetry line when telemetry exists for device", () => {
    const devices = { node_bench_agg: { power: 120, state: "ON", label: "Bench" } };
    const telemetry = { node_bench_agg: { v: 230.4, i: 0.52, pf: 0.98, ts: 1 } };
    render(<DeviceCards devices={devices} telemetry={telemetry} />);
    expect(screen.getByTestId("device-telemetry-node_bench_agg").textContent).toContain("230.4V");
    expect(screen.getByTestId("device-telemetry-node_bench_agg").textContent).toContain("0.52A");
    expect(screen.getByTestId("device-telemetry-node_bench_agg").textContent).toContain("PF 0.98");
});

// TEST 9A-6: no telemetry line when none published
test("hides telemetry line when no telemetry exists", () => {
    const devices = { node_bench_agg: { power: 120, state: "ON", label: "Bench" } };
    render(<DeviceCards devices={devices} telemetry={{}} />);
    expect(screen.queryByTestId("device-telemetry-node_bench_agg")).not.toBeInTheDocument();
});

// TEST 9A-7: ApplianceTable renders confidence as a percentage
test("appliance table shows confidence percentage when provided", () => {
    const devices = { node_bench_agg: { power: 45, state: "ON", classification: "known:phone", confidence: 0.873 } };
    render(<ApplianceTable devices={devices} />);
    expect(screen.getByText("87%")).toBeInTheDocument();
});

// TEST 9A-8: ApplianceTable shows em-dash when confidence is missing
test("appliance table shows placeholder when confidence missing", () => {
    const devices = { node_bench_agg: { power: 45, state: "ON" } };
    render(<ApplianceTable devices={devices} />);
    expect(screen.getByText("—")).toBeInTheDocument();
});

// TEST 9B-1: Critical alert renders in red
test("CRITICAL alert has critical styling", () => {
    const alerts = [{ id: 1, level: "CRITICAL", message: "Overcurrent", device: "node_kettle" }];
    render(<SafetyAlerts alerts={alerts} />);
    const alert = screen.getByTestId("alert-1");
    expect(alert.className).toMatch(/critical/i);
});

// TEST 9B-2: ARC_FAULT event uses distinct icon / color
test("ARC_FAULT event is visually distinct", () => {
    const alerts = [{ id: 2, level: "ARC_FAULT", message: "Arc fault", device: "node_kettle" }];
    render(<SafetyAlerts alerts={alerts} />);
    expect(screen.getByTestId("alert-icon-2")).toHaveClass("arc-fault-icon");
});

// TEST 9B-3: Alert feed is capped at 50 items (oldest evicted)
test("feed shows max 50 alerts", () => {
    const alerts = Array.from({ length: 60 }, (_, i) => ({
        id: i, level: "WARNING", message: `Alert ${i}`, device: "x"
    }));
    render(<SafetyAlerts alerts={alerts} maxAlerts={50} />);
    expect(screen.getAllByTestId(/^alert-/)).toHaveLength(50);
});

// TEST 9C-1: PMV gauge at 0 shows center (neutral comfort)
test("PMV gauge renders at center for PMV=0", () => {
    render(<DigitalTwin pmv={0} ppd={5} rlLog={[]} />);
    const gauge = screen.getByTestId("pmv-gauge");
    expect(gauge).toHaveAttribute("data-pmv", "0");
    expect(gauge.className).toMatch(/neutral|comfort/i);
});

// TEST 9C-2: LABEL_REQUEST event renders the wired label prompt (event-log card)
test("shows label prompt for LABEL_REQUEST event", () => {
    const events = [
        { type: "LABEL_REQUEST", device_id: "esp32_mystery", message: "Unknown load detected", power: 45 },
    ];
    render(<DigitalTwin pmv={0} ppd={5} events={events} />);
    expect(screen.getByPlaceholderText(/appliance name/i)).toBeInTheDocument();
});

// TEST 9C-3: label submit posts to /api/submit-label and confirms enrollment
test("label submit posts to API and confirms enrollment", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    const segments = [Array.from({ length: 128 }, () => 45)];
    const events = [
        { type: "LABEL_REQUEST", device_id: "esp32_mystery", message: "Unknown load detected", power: 45, segments },
    ];
    render(<DigitalTwin pmv={0} ppd={5} events={events} />);
    await userEvent.type(screen.getByPlaceholderText(/appliance name/i), "Kettle");
    await userEvent.click(screen.getByText(/label device/i));
    // Parent swaps the card for the confirmation line once onLabeled fires.
    await waitFor(() => expect(screen.getByText(/Enrolled as "Kettle"/i)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/submit-label"),
        expect.objectContaining({ method: "POST" })
    );
    vi.unstubAllGlobals();
});

// TEST 9D-1: Latency panel turns red when p95 > 200ms
test("latency panel shows red when p95 > 200ms", () => {
    render(<SystemStatus latency={{ avg: 150, p95: 250, max: 300 }} />);
    const panel = screen.getByTestId("latency-panel");
    expect(panel.className).toMatch(/danger|red|over-sla/i);
});

// TEST 9D-2: Latency panel is green when p95 < 200ms
test("latency panel shows green when p95 < 200ms", () => {
    render(<SystemStatus latency={{ avg: 80, p95: 150, max: 180 }} />);
    const panel = screen.getByTestId("latency-panel");
    expect(panel.className).toMatch(/ok|green|within-sla/i);
});

// TEST 9D-3: Disconnected banner shown when WebSocket drops
test("shows disconnected banner on WS drop", () => {
    render(<SystemStatus wsConnected={false} latency={null} />);
    expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
});

// TEST 9E-1: Breaker status derived from alerts — TRIPPED on SAFETY_CUTOFF
test("breaker status shows TRIPPED when a cutoff alert exists", () => {
    const alerts = [
        { id: Date.now(), type: "SAFETY_CUTOFF", severity: "critical", device_id: "node_bench_agg",
          message: "Critical power threshold breached — edge relay has been activated" },
    ];
    render(<AlertsPage alerts={alerts} />);
    const breaker = screen.getByTestId("breaker-status");
    expect(breaker.textContent).toMatch(/tripped/i);
    expect(breaker.textContent).toMatch(/relay open/i);
    expect(breaker.textContent).not.toMatch(/armed & nominal/i);
});

// TEST 9E-3: A cutoff alert OLDER than the 5-minute lockout leaves the
// breaker Armed & Nominal — the lockout expired and the relay can be
// re-energized, so a stale alert must not claim TRIPPED forever.
test("breaker status resets after the lockout window expires", () => {
    const alerts = [
        { id: Date.now() - 6 * 60 * 1000, type: "SAFETY_CUTOFF", severity: "critical",
          device_id: "node_bench_agg", message: "Critical power threshold breached" },
    ];
    render(<AlertsPage alerts={alerts} />);
    expect(screen.getByTestId("breaker-status").textContent).toMatch(/armed & nominal/i);
});

// TEST 9E-2: Breaker status stays Armed without cutoff/overcurrent/arc alerts
test("breaker status shows Armed & Nominal without trip alerts", () => {
    const alerts = [
        { id: 1, type: "SAFETY_WARNING", severity: "warning", device_id: "x",
          message: "Power draw approaching limit on x" },
    ];
    render(<AlertsPage alerts={alerts} />);
    expect(screen.getByTestId("breaker-status").textContent).toMatch(/armed & nominal/i);
});
