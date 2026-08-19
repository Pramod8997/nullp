import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useTheme } from '../../contexts/ThemeContext';
import { BarChart3 } from 'lucide-react';

const generateDataForRange = (range) => {
  const data = [];
  const count = range === 'Week' ? 7 : range === 'Month' ? 30 : range === 'Year' ? 12 : 24;

  for (let i = 0; i <= count; i++) {
    const label =
      range === 'Week'
        ? `Day ${i + 1}`
        : range === 'Month'
        ? `Day ${i + 1}`
        : range === 'Year'
        ? `M${i + 1}`
        : `${String(i).padStart(2, '0')}:00`;

    let base = 0.2;
    if (i >= 6 && i <= 9) base = 0.8 + Math.random() * 0.7;
    else if (i >= 11 && i <= 14) base = 1.0 + Math.random() * 1.0;
    else if (i >= 17 && i <= 22) base = 1.2 + Math.random() * 1.5;
    else if (i >= 1 && i <= 5) base = 0.1 + Math.random() * 0.2;
    else base = 0.4 + Math.random() * 0.5;

    data.push({
      time: label,
      today: parseFloat(base.toFixed(2)),
      yesterday: parseFloat((base * (0.85 + Math.random() * 0.3)).toFixed(2)),
    });
  }
  return data;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-950/95 dark:bg-gray-900/95 backdrop-blur-md p-3 rounded-xl border border-gray-800 shadow-2xl text-xs z-50">
      <p className="text-gray-400 font-mono mb-2 pb-1 border-b border-gray-800">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="font-mono font-semibold flex items-center justify-between gap-3" style={{ color: entry.color }}>
          <span>{entry.name}:</span>
          <span>{Number(entry.value).toFixed(2)} kWh</span>
        </p>
      ))}
    </div>
  );
};

const EnergyChart = ({ powerHistory = [] }) => {
  const { isDark } = useTheme();
  const [timeRange, setTimeRange] = useState('Day');
  const ranges = ['Day', 'Week', 'Month', 'Year'];

  const chartData = useMemo(() => generateDataForRange(timeRange), [timeRange]);

  const gridColor = isDark ? '#1f2937' : '#f3f4f6';
  const axisColor = isDark ? '#6b7280' : '#9ca3af';

  return (
    <div className="w-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-5 shadow-sm flex flex-col justify-between" data-history-count={powerHistory.length}>
      {/* Header & Range Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-gray-200/60 dark:border-gray-700/60">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400">
            <BarChart3 size={18} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              Energy Consumption (kWh)
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">Historical vs current period</p>
          </div>
        </div>

        {/* Range Buttons */}
        <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-900/60 p-1 rounded-xl border border-gray-200/60 dark:border-gray-700/60 self-start sm:self-auto">
          {ranges.map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                timeRange === range
                  ? 'bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-end gap-4 text-xs font-medium py-2">
        <span className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
          <span className="h-2.5 w-2.5 rounded-sm bg-indigo-500 opacity-60" />
          Previous
        </span>
        <span className="flex items-center gap-1.5 text-gray-900 dark:text-gray-200">
          <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
          Current
        </span>
      </div>

      {/* Chart */}
      <div className="w-full h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
            <XAxis
              dataKey="time"
              stroke={axisColor}
              tick={{ fill: axisColor, fontSize: 10 }}
              tickMargin={8}
              minTickGap={30}
            />
            <YAxis
              stroke={axisColor}
              tick={{ fill: axisColor, fontSize: 10 }}
              domain={[0, 'auto']}
              width={35}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)' }} />
            <Bar
              dataKey="yesterday"
              fill="#6366f1"
              radius={[3, 3, 0, 0]}
              opacity={0.4}
              name="Previous"
            />
            <Bar
              dataKey="today"
              fill="#10b981"
              radius={[3, 3, 0, 0]}
              name="Current"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EnergyChart;
