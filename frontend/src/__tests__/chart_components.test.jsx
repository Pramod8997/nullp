import React from 'react';
import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import RealTimeChart from '../components/RealTimeChart';
import EnergyChart from '../components/EnergyChart/EnergyChart';
import {
  buildEnergyChartData,
  buildRealTimeChartData,
  MAX_ENERGY_POINTS,
  MAX_REALTIME_POINTS,
} from '../components/chartDataUtils';

test('real-time chart normalizes, sorts, merges, and filters readings without mutating input', () => {
  const data = [
    { timestamp: '2026-10-27T10:00:02Z', device_id: 'node_b', power: '20' },
    { time: '10:00:01', node_a: 10, node_b: Number.NaN, ignored: -1 },
    { timestamp: '2026-10-27T10:00:02.000Z', node_a: 30 },
    { time: 'not-a-time', node_a: 99 },
  ];
  const original = structuredClone(data);

  expect(buildRealTimeChartData(data)).toEqual([
    { time: '10:00:01', node_a: 10 },
    { time: '2026-10-27T10:00:02Z', node_b: 20, node_a: 30 },
  ]);
  expect(data).toEqual(original);
});

test('real-time chart keeps only the newest bounded history points', () => {
  const data = Array.from({ length: MAX_REALTIME_POINTS + 10 }, (_, index) => ({
    time: new Date(Date.UTC(2026, 9, 27, 10, 0, index)).toISOString(),
    node_a: index,
  }));

  const result = buildRealTimeChartData(data);
  expect(result).toHaveLength(MAX_REALTIME_POINTS);
  expect(result[0].node_a).toBe(10);
  expect(result.at(-1).node_a).toBe(MAX_REALTIME_POINTS + 9);
});

test('energy chart sorts, merges partial duplicate rows, bounds history, and rejects invalid values', () => {
  const history = [
    { timestamp: '2026-10-27T10:00:02+02:00', today: '2.5' },
    { time: '2026-10-27T07:59:59Z', yesterday: 1 },
    { timestamp: '2026-10-27T08:00:02Z', yesterday: 2 },
    { time: 'bad', today: 4 },
    { time: '2026-10-27T08:00:02Z', today: 3, yesterday: -1 },
  ];
  const original = structuredClone(history);

  expect(buildEnergyChartData(history)).toEqual([
    { time: '2026-10-27T07:59:59Z', yesterday: 1, today: null },
    { time: '2026-10-27T10:00:02+02:00', yesterday: 2, today: 3 },
  ]);
  expect(history).toEqual(original);

  const bounded = buildEnergyChartData(Array.from({ length: MAX_ENERGY_POINTS + 1 }, (_, index) => ({
    time: 1_700_000_000 + index,
    today: index,
  })));
  expect(bounded).toHaveLength(MAX_ENERGY_POINTS);
  expect(bounded[0].today).toBe(1);
});

test('charts expose truthful loading, error, and empty states', () => {
  render(<RealTimeChart loading />);
  expect(screen.getByText(/loading power history/i)).toBeInTheDocument();

  render(<EnergyChart error={new Error('backend unavailable')} />);
  expect(screen.getByText(/unable to display measured energy history/i)).toBeInTheDocument();

  render(<RealTimeChart data={[{ time: 'bad', node_a: 10 }]} devices={{ node_a: {} }} />);
  expect(screen.getByText(/no valid power readings/i)).toBeInTheDocument();
});
