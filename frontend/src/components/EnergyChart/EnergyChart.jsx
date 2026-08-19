import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import './EnergyChart.css';

// Generate mock hourly data for the chart
const generateHourlyData = () => {
  const data = [];
  for (let h = 0; h <= 24; h++) {
    const hour = `${String(h).padStart(2, '0')}:00`;
    // Simulate a realistic energy curve: low at night, peaks at morning/evening
    let base = 0.2;
    if (h >= 6 && h <= 9) base = 0.8 + Math.random() * 0.7;
    else if (h >= 11 && h <= 14) base = 1.0 + Math.random() * 1.0;
    else if (h >= 17 && h <= 22) base = 1.2 + Math.random() * 1.5;
    else if (h >= 1 && h <= 5) base = 0.1 + Math.random() * 0.2;
    else base = 0.4 + Math.random() * 0.5;

    data.push({
      time: hour,
      today: parseFloat(base.toFixed(2)),
      yesterday: parseFloat((base * (0.85 + Math.random() * 0.3)).toFixed(2)),
    });
  }
  return data;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="energy-chart-tooltip">
      <p className="energy-chart-tooltip__label">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="energy-chart-tooltip__value" style={{ color: entry.color }}>
          {entry.name}: {Number(entry.value).toFixed(2)} kWh
        </p>
      ))}
    </div>
  );
};

const EnergyChart = ({ powerHistory = [] }) => {
  const [timeRange, setTimeRange] = useState('Day');
  const ranges = ['Day', 'Week', 'Month', 'Year'];

  const chartData = useMemo(() => generateHourlyData(), [timeRange]);

  return (
    <div className="energy-chart-panel">
      <div className="energy-chart__header">
        <h3>ENERGY CONSUMPTION (kWh)</h3>
        <div className="energy-chart__range-toggle">
          {ranges.map((range) => (
            <button
              key={range}
              className={`energy-chart__range-btn ${timeRange === range ? 'active' : ''}`}
              onClick={() => setTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      <div className="energy-chart__legend-custom">
        <span className="energy-chart__legend-item">
          <span className="energy-chart__legend-line" style={{ background: '#6366f1' }} />
          Yesterday
        </span>
        <span className="energy-chart__legend-item">
          <span className="energy-chart__legend-line" style={{ background: '#22c55e' }} />
          Today
        </span>
      </div>

      <div className="energy-chart__container">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 5, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickMargin={8}
              minTickGap={30}
            />
            <YAxis
              stroke="#475569"
              tick={{ fill: '#64748b', fontSize: 10 }}
              domain={[0, 'auto']}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar
              dataKey="yesterday"
              fill="#6366f1"
              radius={[2, 2, 0, 0]}
              opacity={0.5}
              name="Yesterday"
            />
            <Bar
              dataKey="today"
              fill="#22c55e"
              radius={[2, 2, 0, 0]}
              name="Today"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EnergyChart;
