import React, { memo, useMemo } from 'react';
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
import { BarChart3, Calendar, ChevronDown } from 'lucide-react';
import {
  buildEnergyChartData,
  finiteNonNegative,
} from '../chartDataUtils';

const finiteEnergy = finiteNonNegative;

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const validEntries = payload.filter((entry) => finiteEnergy(entry?.value) !== null);
  if (validEntries.length === 0) return null;
  return (
    <div className="bg-gray-950/95 dark:bg-[#0c1424]/95 backdrop-blur-md p-3 rounded-xl border border-gray-800 dark:border-[#172338] shadow-2xl text-xs z-50">
      <p className="text-gray-400 font-mono mb-2 pb-1 border-b border-gray-800 dark:border-[#172338]">{label}</p>
      {validEntries.map((entry, i) => (
        <p key={i} className="font-mono font-semibold flex items-center justify-between gap-3" style={{ color: entry.color }}>
          <span>{entry.name}:</span>
          <span>{finiteEnergy(entry.value).toFixed(2)} kWh</span>
        </p>
      ))}
    </div>
  );
};

const EnergyChart = ({ powerHistory = [], analytics = {}, loading = false, error = null }) => {
  const { isDark } = useTheme();

  const chartData = useMemo(
    () => buildEnergyChartData(analytics?.energy_history),
    [analytics]
  );
  const hasHistory = chartData.length > 0 && !loading && !error;

  const gridColor = isDark ? '#172338' : '#f3f4f6';
  const axisColor = isDark ? '#64748b' : '#9ca3af';

  return (
    <div className="w-full bg-white/90 dark:bg-[#0c1424]/90 backdrop-blur-xl border border-gray-200/80 dark:border-[#172338] rounded-3xl p-5 md:p-6 shadow-sm flex flex-col justify-between" data-history-count={Array.isArray(powerHistory) ? powerHistory.length : 0}>
      {/* Header & Range Filters */}
      <div className="flex items-center justify-between gap-3 pb-4 border-b border-gray-200/60 dark:border-[#172338]/60">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-400 border border-emerald-500/20">
            <BarChart3 size={17} />
          </div>
          <div>
            <h3 className="text-xs font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              ENERGY CONSUMPTION (KWH)
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Measured backend history
            </p>
          </div>
        </div>

        {/* Range Pill */}
        <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-gray-100 text-gray-700 dark:bg-[#131c2d] dark:text-gray-300 border border-gray-200 dark:border-[#223048] cursor-pointer hover:bg-gray-200 dark:hover:bg-[#182338] transition-colors">
          <Calendar size={13} className="text-gray-400" />
          <span>Today</span>
          <ChevronDown size={12} className="text-gray-400 ml-0.5" />
        </div>
      </div>

      {loading || error ? (
        <div className="flex min-h-[240px] items-center justify-center text-center text-sm text-gray-500 dark:text-gray-400" role="status" aria-live="polite">
          {loading ? 'Loading measured energy history…' : 'Unable to display measured energy history.'}
        </div>
      ) : hasHistory ? (
        <>
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
                <Tooltip content={<CustomTooltip />} isAnimationActive={false} cursor={{ fill: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)' }} />
                <Bar
                  dataKey="yesterday"
                  fill="#6366f1"
                  radius={[3, 3, 0, 0]}
                  opacity={0.4}
                  name="Previous"
                  isAnimationActive={false}
                />
                <Bar
                  dataKey="today"
                  fill="#10b981"
                  radius={[3, 3, 0, 0]}
                  name="Current"
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="flex min-h-[240px] items-center justify-center text-center text-sm text-gray-500 dark:text-gray-400">
          No measured energy history available.
        </div>
      )}
    </div>
  );
};

const areEqual = (previous, next) => (
  previous.analytics?.energy_history === next.analytics?.energy_history
  && previous.powerHistory === next.powerHistory
  && previous.loading === next.loading
  && previous.error === next.error
);

export default memo(EnergyChart, areEqual);
