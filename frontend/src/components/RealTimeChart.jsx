import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useTheme } from '../contexts/ThemeContext';

const DEVICE_COLORS = {
  node_fridge: '#10b981',
  node_microwave: '#f43f5e',
  node_kettle: '#f59e0b',
  node_hvac: '#06b6d4',
  esp32_fridge: '#3b82f6',
  esp32_hvac: '#8b5cf6',
  esp32_kettle: '#f59e0b',
  esp32_tv: '#06b6d4',
  esp32_washer: '#a78bfa',
  esp32_dryer: '#f472b6',
  esp32_dishwasher: '#34d399',
  esp32_oven: '#fb923c',
  esp32_lighting: '#facc15',
  default: '#60a5fa',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-950/95 dark:bg-gray-900/95 backdrop-blur-md p-3 rounded-xl border border-gray-800 shadow-2xl text-xs z-50">
      <p className="text-gray-400 font-mono mb-2 pb-1 border-b border-gray-800">{label}</p>
      <div className="space-y-1">
        {payload.map((entry, i) => (
          <div key={i} className="flex items-center justify-between gap-3 font-mono">
            <span className="font-semibold" style={{ color: entry.color }}>
              {entry.dataKey}:
            </span>
            <span className="text-white font-bold">{Number(entry.value).toFixed(1)} W</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const RealTimeChart = ({ data = [], devices = {} }) => {
  const { isDark } = useTheme();

  // Merge power readings into unified time-series rows
  const chartData = useMemo(() => {
    const timeMap = {};
    data.forEach((point) => {
      const key = point.time;
      if (!timeMap[key]) timeMap[key] = { time: key };
      Object.keys(point).forEach((k) => {
        if (k !== 'time' && k !== 'device_id') {
          timeMap[key][k] = point[k];
        }
      });
    });
    return Object.values(timeMap).slice(-60);
  }, [data]);

  const deviceIds = Object.keys(devices || {});

  const gridColor = isDark ? '#1f2937' : '#f3f4f6';
  const axisColor = isDark ? '#6b7280' : '#9ca3af';

  return (
    <div className="w-full h-[280px] pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 5, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
          <XAxis
            dataKey="time"
            stroke={axisColor}
            tick={{ fill: axisColor, fontSize: 10 }}
            tickMargin={8}
            minTickGap={35}
          />
          <YAxis
            stroke={axisColor}
            tick={{ fill: axisColor, fontSize: 10 }}
            unit="W"
            domain={[0, 'auto']}
            width={45}
          />
          <Tooltip content={<CustomTooltip />} isAnimationActive={false} />

          {/* Safety threshold reference line */}
          <ReferenceLine
            y={1500}
            stroke="#ef4444"
            strokeDasharray="5 5"
            strokeOpacity={0.7}
            label={{
              value: 'Safety Cutoff (1500W)',
              fill: '#ef4444',
              fontSize: 10,
              position: 'insideTopRight',
            }}
          />

          {/* Per-device lines */}
          {deviceIds.map((deviceId) => (
            <Line
              key={deviceId}
              type="monotone"
              dataKey={deviceId}
              stroke={DEVICE_COLORS[deviceId] || DEVICE_COLORS.default}
              strokeWidth={2}
              dot={false}
              activeDot={{
                r: 4,
                strokeWidth: 2,
                fill: DEVICE_COLORS[deviceId] || DEVICE_COLORS.default,
              }}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RealTimeChart;
