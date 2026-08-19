import React from 'react';
import {
  Activity,
  Wifi,
  WifiOff,
  Database,
  Brain,
  Clock,
  Download,
  AlertCircle,
  DollarSign,
} from 'lucide-react';
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

const API_BASE = `http://${window.location.hostname}:8000`;

const LatencyTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-950/95 dark:bg-gray-900/95 backdrop-blur-md p-2.5 rounded-xl border border-gray-800 shadow-xl text-xs z-50">
      <p className="text-gray-400 font-mono mb-1.5">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="font-mono font-semibold" style={{ color: entry.color }}>
          {entry.dataKey}: {Number(entry.value).toFixed(1)} ms
        </p>
      ))}
    </div>
  );
};

const SystemStatus = ({
  connectionStatus = 'connected',
  pipelineStatus,
  analytics,
  deviceCount,
  latencyStats,
  latencyHistory,
  latency,
  wsConnected,
}) => {
  const { isDark } = useTheme();

  const isWsConnected = wsConnected !== undefined ? wsConnected : connectionStatus === 'connected';
  const isWsReconnecting = connectionStatus === 'reconnecting';

  const lat =
    latency ||
    (latencyStats
      ? {
          avg: latencyStats.avg_ms ?? latencyStats.avg,
          p95: latencyStats.p95_ms ?? latencyStats.p95,
          max: latencyStats.max_ms ?? latencyStats.max,
        }
      : null);

  const isDanger = lat && lat.p95 !== undefined && lat.p95 > 200;
  const latencyClass = isDanger ? 'danger red over-sla' : 'ok green within-sla';

  const handleExportCSV = () => {
    window.open(`${API_BASE}/api/export-csv`, '_blank');
  };

  const gridColor = isDark ? '#1f2937' : '#f3f4f6';
  const axisColor = isDark ? '#6b7280' : '#9ca3af';

  return (
    <div className="w-full space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200/80 dark:border-gray-700/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-cyan-50 dark:bg-cyan-950/60 text-cyan-600 dark:text-cyan-400">
            <Activity size={18} />
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">System Telemetry & Health</h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">Real-time pipeline metrics and latency SLAs</p>
          </div>
        </div>
      </div>

      {!isWsConnected && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-50 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300 border border-rose-300 dark:border-rose-800 text-xs font-semibold">
          <AlertCircle size={16} />
          <span>Disconnected from WebSocket server — Attempting auto-reconnect...</span>
        </div>
      )}

      {/* Grid of Status telemetry pills */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {/* WebSocket */}
        <div className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-200/60 dark:border-gray-700/60">
          <span className="flex items-center gap-2 text-xs font-medium text-gray-600 dark:text-gray-300">
            {isWsConnected ? <Wifi size={16} className="text-emerald-500" /> : <WifiOff size={16} className="text-rose-500" />}
            WebSocket Stream
          </span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono ${
            isWsConnected
              ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
              : isWsReconnecting
              ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border border-amber-200 dark:border-amber-800'
              : 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border border-rose-200 dark:border-rose-800'
          }`}>
            {isWsConnected ? 'CONNECTED' : isWsReconnecting ? 'RECONNECTING' : 'OFFLINE'}
          </span>
        </div>

        {/* Pipeline */}
        <div className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-200/60 dark:border-gray-700/60">
          <span className="flex items-center gap-2 text-xs font-medium text-gray-600 dark:text-gray-300">
            <Database size={16} className="text-blue-500" />
            NILM Pipeline
          </span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono ${
            pipelineStatus === 'connected'
              ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
              : 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-200 dark:border-blue-800'
          }`}>
            {pipelineStatus?.toUpperCase() || 'ONLINE'}
          </span>
        </div>

        {/* Active Nodes */}
        {deviceCount !== undefined && (
          <div className="flex items-center justify-between p-3.5 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-200/60 dark:border-gray-700/60">
            <span className="flex items-center gap-2 text-xs font-medium text-gray-600 dark:text-gray-300">
              <Brain size={16} className="text-purple-500" />
              Active Channels
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-bold font-mono bg-purple-50 text-purple-700 dark:bg-purple-950/60 dark:text-purple-400 border border-purple-200 dark:border-purple-800">
              {deviceCount} Nodes
            </span>
          </div>
        )}
      </div>

      {/* Analytics row if present */}
      {analytics?.total_kwh !== undefined && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 p-3.5 rounded-xl bg-gray-50 dark:bg-gray-800/40 border border-gray-200/60 dark:border-gray-700/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Accumulated Ingestion</span>
            <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
              {analytics.total_kwh.toFixed(3)} kWh
            </span>
          </div>
          {analytics.estimated_cost_usd !== undefined && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500 dark:text-gray-400 font-medium flex items-center gap-1">
                <DollarSign size={12} /> Projected Cost
              </span>
              <span className="text-xs font-mono font-bold text-amber-600 dark:text-amber-400">
                ${analytics.estimated_cost_usd.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Latency Panel widget */}
      {lat && (
        <div
          data-testid="latency-panel"
          className={`latency-panel ${latencyClass} p-4 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
            isDanger
              ? 'bg-rose-50 dark:bg-rose-950/30 border-rose-300 dark:border-rose-800 text-rose-900 dark:text-rose-200'
              : 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800 text-emerald-900 dark:text-emerald-200'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <div className={`p-2 rounded-lg ${isDanger ? 'bg-rose-500/20 text-rose-600' : 'bg-emerald-500/20 text-emerald-600'}`}>
              <Clock size={18} />
            </div>
            <div>
              <div className="text-xs font-bold uppercase tracking-wider">
                Pipeline Ingestion SLA ({isDanger ? 'BREACHED > 200ms' : 'HEALTHY < 200ms'})
              </div>
              <p className="text-xs opacity-80">Round-trip ESP32 to WebSocket inference latency</p>
            </div>
          </div>

          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-xs">Avg / P95:</span>
            <span className="text-lg font-black">
              {lat.avg?.toFixed ? lat.avg.toFixed(1) : lat.avg}ms / {lat.p95?.toFixed ? lat.p95.toFixed(1) : lat.p95}ms
            </span>
          </div>
        </div>
      )}

      {/* Latency Trend History Chart */}
      {latencyHistory && latencyHistory.length > 1 && (
        <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/40 border border-gray-200/60 dark:border-gray-700/60 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-gray-600 dark:text-gray-300">
            <span className="flex items-center gap-1.5">
              <Clock size={14} className="text-cyan-500" />
              Pipeline Latency Trend (ms)
            </span>
            <span className="text-[11px] text-gray-400">Target SLA: 200ms</span>
          </div>
          <div className="w-full h-36">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={latencyHistory} margin={{ top: 4, right: 10, bottom: 4, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis dataKey="time" stroke={axisColor} tick={{ fill: axisColor, fontSize: 9 }} tickMargin={4} minTickGap={35} />
                <YAxis stroke={axisColor} tick={{ fill: axisColor, fontSize: 9 }} unit="ms" domain={[0, 'auto']} width={35} />
                <Tooltip content={<LatencyTooltip />} isAnimationActive={false} />
                <ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.7} />
                <Line type="monotone" dataKey="avg" stroke="#06b6d4" strokeWidth={2} dot={false} name="Avg" isAnimationActive={false} />
                <Line type="monotone" dataKey="p95" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="P95" isAnimationActive={false} />
                <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} dot={false} name="Max" strokeDasharray="4 2" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Export Button */}
      <div className="flex justify-end pt-2">
        <button
          id="export-csv-btn"
          onClick={handleExportCSV}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs font-semibold border border-gray-200 dark:border-gray-700 transition-all cursor-pointer"
          title="Download last 24 hours of energy data as CSV"
        >
          <Download size={14} />
          <span>Export 24h Energy CSV</span>
        </button>
      </div>
    </div>
  );
};

export default SystemStatus;
