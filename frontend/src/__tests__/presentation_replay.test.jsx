import React from 'react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';

// JSDOM has no layout. Keep the real charts, but give their container a size.
vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    ResponsiveContainer: ({ children }) => React.cloneElement(children, { width: 800, height: 240 }),
  };
});

let socket;

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;

  constructor() {
    this.readyState = FakeWebSocket.CONNECTING;
    socket = this;
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

const generatedAnalytics = {
  source: 'generated_sample_replay',
  generated: true,
  tariff_inr_per_kwh: 8,
  total_kwh: 1.25,
  estimated_cost_inr: 10,
  baseline_kwh: 1.5,
  energy_saved_kwh: 0.25,
  playback_speed: 300,
  replay: {
    cycle: 2,
    total_cycles: 3,
    sample_index: 20,
    total_samples: 60,
    generated: true,
  },
  energy_history: [
    { time: '00:10', today: 0.5, yesterday: 0.6 },
    { time: '00:20', today: 0.75, yesterday: 0.9 },
  ],
};

function generatedInitState(historyCount = 60) {
  return {
    type: 'init_state',
    pipeline_status: 'SAMPLE_DATA_REPLAY',
    devices: {
      node_bench_agg: {
        label: 'Generated laptop sample',
        power: 75,
        state: 'ON',
        classification: 'known:laptop',
        source: 'generated_sample_replay',
        generated: true,
        presentation_only: true,
      },
    },
    analytics: generatedAnalytics,
    power_history: Array.from({ length: historyCount }, (_, index) => ({
      time: `sample-${index + 1}`,
      node_bench_agg: 40 + index,
    })),
  };
}

async function emit(...payloads) {
  await act(async () => {
    payloads.forEach((payload) => socket.onmessage({ data: JSON.stringify(payload) }));
  });
}

async function mountDashboard(initialState) {
  const view = render(<App />);
  await waitFor(() => expect(socket?.onmessage).toBeTypeOf('function'));
  await act(async () => socket.open());
  if (initialState) await emit(initialState);
  return view;
}

beforeEach(() => {
  vi.stubGlobal('WebSocket', FakeWebSocket);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  socket = undefined;
});

test('generated replay identifies its source and displays cycle and sample progress', async () => {
  await mountDashboard(generatedInitState());

  const header = within(screen.getByRole('banner'));
  expect(header.getByText(/sample-data replay/i)).toBeVisible();
  expect(screen.getByText('Source: generated samples · No hardware recording · No relay action')).toBeVisible();
  expect(screen.getByText(/300× scenario time.*three-cycle group/i)).toBeVisible();
  expect(screen.getAllByText(/(?:cycle|replay)\s*2\s*(?:\/|of)\s*3/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/samples?\s*20\s*(?:\/|of)\s*60/i).length).toBeGreaterThan(0);
  expect(header.queryByText(/recorded.*replay|live 1hz stream/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/captured (?:hardware|load)|recorded (?:hardware|session) replay|relay (?:open|activated)|physical relay (?:action|activated)/i)).not.toBeInTheDocument();

  await emit({
    type: 'ANALYTICS_UPDATE',
    summary: {
      ...generatedAnalytics,
      replay: { ...generatedAnalytics.replay, cycle: 3, sample_index: 40 },
    },
  });
  await waitFor(() => expect(screen.getAllByText(/(?:cycle|replay)\s*3\s*(?:\/|of)\s*3/i).length).toBeGreaterThan(0));
  expect(screen.getAllByText(/samples?\s*40\s*(?:\/|of)\s*60/i).length).toBeGreaterThan(0);
});

test('init_state populates generated consumption, cost, savings and real chart series before any live updates', async () => {
  const { container } = await mountDashboard(generatedInitState());

  expect(screen.getByText('Scenario Consumption')).toBeVisible();
  expect(screen.getByText('1.250')).toBeVisible();
  expect(screen.getByText(/₹\s*10\.00/)).toBeVisible();
  expect(screen.getByText('0.250')).toBeVisible();
  expect(screen.getByText(/₹\s*8(?:\.00)?\s*\/\s*kWh/i)).toBeVisible();
  expect(screen.getByTestId('device-card-node_bench_agg')).toHaveTextContent('75.0');
  expect(container.querySelector('[data-history-count]')).toHaveAttribute('data-history-count', '60');
  expect(screen.queryByText(/no measured energy history/i)).not.toBeInTheDocument();
  await waitFor(() => {
    expect(container.querySelector('.recharts-line-curve')).toBeInTheDocument();
    expect(container.querySelector('.recharts-bar-rectangle')).toBeInTheDocument();
  });
});

test.each([false, true])('generated init_state accepts optional telemetry (present: %s)', async (withTelemetry) => {
  const payload = generatedInitState();
  if (withTelemetry) payload.telemetry = { node_bench_agg: { v: 230.4, i: 0.33, pf: 0.98 } };
  await mountDashboard(payload);

  expect(screen.getByTestId('device-card-node_bench_agg')).toHaveTextContent('Generated laptop sample');
  if (withTelemetry) {
    expect(screen.getByTestId('device-telemetry-node_bench_agg')).toHaveTextContent('230.4V · 0.33A · PF 0.98');
  } else {
    expect(screen.queryByTestId('device-telemetry-node_bench_agg')).not.toBeInTheDocument();
  }
});

test('replay power history stays capped at 120 rows as new readings arrive', async () => {
  const { container } = await mountDashboard(generatedInitState(120));
  expect(container.querySelector('[data-history-count]')).toHaveAttribute('data-history-count', '120');

  await emit(
    { type: 'power_reading', device_id: 'node_bench_agg', power: 81 },
    { type: 'power_batch', readings: { node_bench_agg: 82 } },
  );

  await waitFor(() => expect(screen.getByTestId('device-card-node_bench_agg')).toHaveTextContent('82.0'));
  expect(container.querySelector('[data-history-count]')).toHaveAttribute('data-history-count', '120');
});

test('sample-data replay does not fabricate a measured inference latency or healthy SLA', async () => {
  const user = userEvent.setup();
  await mountDashboard(generatedInitState());
  await user.click(screen.getByRole('button', { name: 'Analytics' }));

  expect(screen.getByRole('heading', { name: /energy analytics/i })).toBeVisible();
  expect(screen.getByText('1.250 kWh')).toBeVisible();
  expect(screen.getByText(/₹\s*10\.00/)).toBeVisible();
  expect(screen.queryByText(/\d+(?:\.\d+)?\s*ms\s*\/\s*\d+(?:\.\d+)?\s*ms/)).not.toBeInTheDocument();
  expect(screen.queryByText(/healthy\s*[<≤]\s*200ms/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/round-trip ESP32 to WebSocket inference latency/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/pipeline latency trend/i)).not.toBeInTheDocument();
});

test('generated cutoff markers stay in alert history across replay initialization without a fullscreen popup or physical relay claim', async () => {
  const user = userEvent.setup();
  await mountDashboard(generatedInitState());
  await emit(...[1, 2, 3].map((cycle) => ({
    type: 'SAFETY_CUTOFF',
    device_id: 'node_bench_agg',
    severity: 'critical',
    presentation_only: true,
    generated: true,
    source: 'generated_sample_replay',
    relay_action: 'NONE',
    message: `Generated sample replay: ARC_FAULT cutoff marker cycle ${cycle}; no relay action.`,
  })));
  expect(document.querySelector('#arc-fault-overlay')).not.toBeInTheDocument();

  await emit(generatedInitState());
  await user.click(within(screen.getByRole('navigation')).getByRole('button', { name: /^Alerts/ }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'All (3)' })).toBeVisible());
  const alertRows = screen.getAllByTestId(/^alert-\d/);
  expect(alertRows).toHaveLength(3);
  expect(alertRows[0]).toHaveTextContent('cycle 3');
  expect(alertRows[2]).toHaveTextContent('cycle 1');
  expect(screen.getByTestId('breaker-status')).toHaveTextContent(/no relay action/i);
  expect(screen.getByTestId('breaker-status')).not.toHaveTextContent(/TRIPPED|relay open/i);
  expect(screen.queryByText(/recorded-session replay/i)).not.toBeInTheDocument();
  expect(document.querySelector('#arc-fault-overlay')).not.toBeInTheDocument();
});

test('recorded replay cutoffs preserve their feed and recording provenance without a fullscreen red popup', async () => {
  const user = userEvent.setup();
  await mountDashboard({ type: 'init_state', pipeline_status: 'RECORDED_DATA_REPLAY' });

  const payload = {
    type: 'SAFETY_CUTOFF',
    device_id: 'node_bench_agg',
    severity: 'critical',
    presentation_only: true,
    recording_replay: true,
    relay_action: 'NONE',
    message: 'Recorded-session replay: overcurrent cutoff marker; no relay command or mains contact action was issued.',
  };

  await emit(payload, payload, payload);
  expect(document.querySelector('#arc-fault-overlay')).not.toBeInTheDocument();
  await emit({ type: 'init_state', pipeline_status: 'RECORDED_DATA_REPLAY' });
  await user.click(within(screen.getByRole('navigation')).getByRole('button', { name: /^Alerts/ }));

  await waitFor(() => expect(screen.getByRole('button', { name: 'All (3)' })).toBeVisible());
  expect(screen.getAllByTestId(/^alert-\d/)).toHaveLength(3);
  expect(screen.getAllByText(payload.message)).toHaveLength(3);
  expect(within(screen.getByRole('banner')).getByText(/recorded[ -]data replay/i)).toBeVisible();
  expect(screen.getByTestId('breaker-status')).toHaveTextContent(/replay event.*no relay action/i);
  expect(screen.getByTestId('breaker-status')).not.toHaveTextContent(/TRIPPED|relay open/i);
  expect(screen.queryByText(/Source: generated samples/i)).not.toBeInTheDocument();
  expect(document.querySelector('#arc-fault-overlay')).not.toBeInTheDocument();
});
