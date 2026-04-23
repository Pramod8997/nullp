import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const DEVICE_COLORS = {
  esp32_fridge:  '#3b82f6',
  esp32_hvac:    '#8b5cf6',
  esp32_kettle:  '#f59e0b',
  esp32_tv:      '#06b6d4',
  default:       '#ec4899',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0f172a',
      padding: '10px 14px',
      border: '1px solid #334155',
      borderRadius: '8px',
      boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
    }}>
      <p style={{ color: '#94a3b8', margin: '0 0 6px', fontSize: '0.75rem' }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{
          color: entry.color,
          fontWeight: 600,
          margin: '2px 0',
          fontSize: '0.85rem',
          fontFamily: "'Inter', monospace",
        }}>
          {entry.dataKey}: {Number(entry.value).toFixed(1)} W
        </p>
      ))}
    </div>
  );
};

const RealTimeChart = ({ data, devices }) => {
  // Merge power readings into unified time-series rows
  const chartData = useMemo(() => {
    const timeMap = {};
    data.forEach(point => {
      const key = point.time;
      if (!timeMap[key]) timeMap[key] = { time: key };
      if (point.device_id && point[point.device_id] !== undefined) {
        timeMap[key][point.device_id] = point[point.device_id];
      }
    });
    return Object.values(timeMap).slice(-60);
  }, [data]);

  const deviceIds = Object.keys(devices);

  return (
    <div className="chart-container">
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="time"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickMargin={8}
            minTickGap={40}
          />
          <YAxis
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 11 }}
            unit="W"
            domain={[0, 'auto']}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} isAnimationActive={false} />

          {/* Safety threshold reference line */}
          <ReferenceLine y={1500} stroke="#ef4444" strokeDasharray="6 4" strokeOpacity={0.5} label={{ value: 'Safety Limit', fill: '#ef4444', fontSize: 10, position: 'right' }} />

          {/* Per-device lines */}
          {deviceIds.map(deviceId => (
            <Line
              key={deviceId}
              type="monotone"
              dataKey={deviceId}
              stroke={DEVICE_COLORS[deviceId] || DEVICE_COLORS.default}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, fill: DEVICE_COLORS[deviceId] || DEVICE_COLORS.default }}
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
