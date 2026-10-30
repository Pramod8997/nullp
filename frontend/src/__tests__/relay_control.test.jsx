import React from 'react';
import { test, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RelayControlPanel from '../components/RelayControlPanel';
import * as clientModule from '../api/client';

vi.mock('../api/client', () => ({
  requestJson: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
});

test('RelayControlPanel renders IN1 and IN2 controls', () => {
  render(
    <RelayControlPanel
      deviceId="node_bench_agg"
      devices={{ node_bench_agg: { state: 'ON', relays: [1, 0] } }}
      telemetry={{ node_bench_agg: { relays: [1, 0], w: 65.0 } }}
    />
  );

  expect(screen.getByText(/Manual Relay Cutoff Console/i)).toBeInTheDocument();
  expect(screen.getByText(/Relay 1: IN1 \(Laptop Charger\)/i)).toBeInTheDocument();
  expect(screen.getByText(/Relay 2: IN2 \(Phone Charger\)/i)).toBeInTheDocument();
  expect(screen.getAllByText(/65\.0/i).length).toBeGreaterThan(0);
});

test('Clicking IN1 toggle dispatches channel 1 command', async () => {
  clientModule.requestJson.mockResolvedValueOnce({
    status: 'published',
    command: 'IN1_OFF',
    topic: 'home/plug/node_bench_agg/command',
  });

  render(
    <RelayControlPanel
      deviceId="node_bench_agg"
      devices={{ node_bench_agg: { state: 'ON', relays: [1, 0] } }}
      telemetry={{ node_bench_agg: { relays: [1, 0], w: 65.0 } }}
    />
  );

  const in1Toggle = screen.getByRole('button', { name: /toggle in1/i });
  await userEvent.click(in1Toggle);

  expect(clientModule.requestJson).toHaveBeenCalledWith('/api/relay/command', {
    method: 'POST',
    body: JSON.stringify({
      device_id: 'node_bench_agg',
      command: 'OFF',
      channel: 1,
    }),
  });
});

test('Clicking Emergency Cutoff dispatches master OFF', async () => {
  clientModule.requestJson.mockResolvedValueOnce({
    status: 'published',
    command: 'OFF',
    topic: 'home/plug/node_bench_agg/command',
  });

  render(
    <RelayControlPanel
      deviceId="node_bench_agg"
      devices={{ node_bench_agg: { state: 'ON', relays: [1, 1] } }}
      telemetry={{ node_bench_agg: { relays: [1, 1], w: 85.0 } }}
    />
  );

  const emergencyBtn = screen.getByRole('button', { name: /emergency cutoff \(off\)/i });
  await userEvent.click(emergencyBtn);

  expect(clientModule.requestJson).toHaveBeenCalledWith('/api/relay/command', {
    method: 'POST',
    body: JSON.stringify({
      device_id: 'node_bench_agg',
      command: 'OFF',
      channel: null,
    }),
  });
});

test('Clicking Master ON dispatches master ON', async () => {
  clientModule.requestJson.mockResolvedValueOnce({
    status: 'published',
    command: 'ON',
    topic: 'home/plug/node_bench_agg/command',
  });

  render(
    <RelayControlPanel
      deviceId="node_bench_agg"
      devices={{ node_bench_agg: { state: 'OFF', relays: [0, 0] } }}
      telemetry={{ node_bench_agg: { relays: [0, 0], w: 0.0 } }}
    />
  );

  const masterOnBtn = screen.getByRole('button', { name: /master on \(both\)/i });
  await userEvent.click(masterOnBtn);

  expect(clientModule.requestJson).toHaveBeenCalledWith('/api/relay/command', {
    method: 'POST',
    body: JSON.stringify({
      device_id: 'node_bench_agg',
      command: 'ON',
      channel: null,
    }),
  });
});
