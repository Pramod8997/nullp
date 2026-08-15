import React from 'react';
import { test, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DeviceCards from '../components/DeviceCards';
import SafetyAlerts from '../components/SafetyAlerts';
import DigitalTwin from '../components/DigitalTwin';
import SystemStatus from '../components/SystemStatus';

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
    render(<DigitalTwin pmv={0} ppd={5} rlLog={[]} unknownDevices={[]} />);
    const gauge = screen.getByTestId("pmv-gauge");
    expect(gauge).toHaveAttribute("data-pmv", "0");
    expect(gauge.className).toMatch(/neutral|comfort/i);
});

// TEST 9C-2: LABEL_REQUEST prompt appears for unknown device
test("shows label prompt for unknown device", () => {
    const unknownDevices = [{ id: "esp32_mystery", requestId: "req_1" }];
    render(<DigitalTwin pmv={0} ppd={5} rlLog={[]} unknownDevices={unknownDevices} />);
    expect(screen.getByTestId("label-request-req_1")).toBeInTheDocument();
});

// TEST 9C-3: Label prompt dismisses after user submission
test("label prompt dismisses on submit", async () => {
    const onLabel = vi.fn();
    render(<DigitalTwin pmv={0} ppd={5} rlLog={[]} unknownDevices={[{id: "x", requestId: "r1"}]}
                        onLabel={onLabel} />);
    await userEvent.type(screen.getByRole("textbox"), "Dishwasher");
    await userEvent.click(screen.getByText(/submit|confirm/i));
    expect(onLabel).toHaveBeenCalledWith("r1", "Dishwasher");
    await waitFor(() => expect(screen.queryByTestId("label-request-r1")).not.toBeInTheDocument());
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
