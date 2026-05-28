import React from 'react';
import { Activity, Wifi, WifiOff, Database, Brain, DollarSign, Clock, Download } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const API_BASE = `http://${window.location.hostname}:8000`;

const LatencyTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0f172a',
      padding: '8px 12px',
      border: '1px solid #334155',
      borderRadius: '8px',
      boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
    }}>
      <p style={{ color: '#94a3b8', margin: '0 0 4px', fontSize: '0.7rem' }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{
          color: entry.color,
          fontWeight: 600,
          margin: '1px 0',
          fontSize: '0.75rem',
          fontFamily: "'Inter', monospace",
        }}>
          {entry.dataKey}: {Number(entry.value).toFixed(1)} ms
        </p>
      ))}
    </div>
  );
};

const SystemStatus = ({ connectionStatus, pipelineStatus, analytics, deviceCount, latencyStats, latencyHistory }) => {
  const wsStatus = connectionStatus === 'connected' ? 'ok' : connectionStatus === 'reconnecting' ? 'warn' : 'error';
  const pipeStatus = pipelineStatus === 'connected' ? 'ok' : pipelineStatus === 'mqtt_reconnecting' ? 'warn' : 'error';

  const handleExportCSV = () => {
    window.open(`${API_BASE}/api/export-csv`, '_blank');
  };

  return (
    <div>
      <div className="panel-header">
        <h2><Activity size={18} color="#06b6d4" /> System Status</h2>
      </div>

      <div className="status-grid">
        <div className="status-row">
          <span className="status-row__label">
            {wsStatus === 'ok' ? <Wifi size={14} style={{ marginRight: 6 }} /> : <WifiOff size={14} style={{ marginRight: 6 }} />}
            WebSocket
          </span>
          <span className={`status-row__value ${wsStatus}`}>
            {connectionStatus === 'connected' ? 'CONNECTED' : connectionStatus === 'reconnecting' ? 'RECONNECTING' : 'OFFLINE'}
          </span>
        </div>

        <div className="status-row">
          <span className="status-row__label">
            <Database size={14} style={{ marginRight: 6 }} />
            Pipeline
          </span>
          <span className={`status-row__value ${pipeStatus}`}>
            {pipelineStatus?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>

        <div className="status-row">
          <span className="status-row__label">
            <Brain size={14} style={{ marginRight: 6 }} />
            Devices
          </span>
          <span className="status-row__value ok">{deviceCount}</span>
        </div>

        {analytics?.total_kwh !== undefined && (
          <>
            <div className="status-row" style={{ marginTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
              <span className="status-row__label">
                <Activity size={14} style={{ marginRight: 6 }} />
                Today's Usage
              </span>
              <span className="status-row__value ok">{analytics.total_kwh.toFixed(3)} kWh</span>
            </div>
            <div className="status-row">
              <span className="status-row__label">
                <DollarSign size={14} style={{ marginRight: 6 }} />
                Est. Cost
              </span>
              <span className="status-row__value" style={{ color: '#f59e0b' }}>
                ${analytics.estimated_cost_usd?.toFixed(4) || '0.00'}
              </span>
            </div>
          </>
        )}

        {latencyStats && latencyStats.avg_ms > 0 && (
          <>
            <div className="status-row" style={{ marginTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
              <span className="status-row__label">
                <Clock size={14} style={{ marginRight: 6 }} />
                Avg Latency
              </span>
              <span className={`status-row__value ${latencyStats.avg_ms < 200 ? 'ok' : 'warn'}`}>
                {latencyStats.avg_ms.toFixed(1)} ms
              </span>
            </div>
            <div className="status-row">
              <span className="status-row__label">
                <Clock size={14} style={{ marginRight: 6 }} />
                P95 / Max
              </span>
              <span className={`status-row__value ${latencyStats.p95_ms < 200 ? 'ok' : 'warn'}`}>
                {latencyStats.p95_ms.toFixed(1)} / {latencyStats.max_ms.toFixed(1)} ms
              </span>
            </div>
          </>
        )}

        {/* ── Latency Trend Chart ── */}
        {latencyHistory && latencyHistory.length > 1 && (
          <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem', fontWeight: 600 }}>
              <Clock size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              Pipeline Latency Trend
            </div>
            <div style={{ width: '100%', height: 140 }}>
              <ResponsiveContainer>
                <LineChart data={latencyHistory} margin={{ top: 4, right: 4, bottom: 4, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="time" stroke="#475569" tick={{ fill: '#64748b', fontSize: 9 }} tickMargin={4} minTickGap={40} />
                  <YAxis stroke="#475569" tick={{ fill: '#64748b', fontSize: 9 }} unit="ms" domain={[0, 'auto']} width={40} />
                  <Tooltip content={<LatencyTooltip />} isAnimationActive={false} />
                  <ReferenceLine y={200} stroke="#ef4444" strokeDasharray="6 4" strokeOpacity={0.6} label={{ value: '200ms', fill: '#ef4444', fontSize: 9, position: 'right' }} />
                  <Line type="monotone" dataKey="avg" stroke="#06b6d4" strokeWidth={2} dot={false} name="Avg" isAnimationActive={false} />
                  <Line type="monotone" dataKey="p95" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="P95" isAnimationActive={false} />
                  <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={1} dot={false} name="Max" strokeDasharray="4 2" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── Export CSV Button ── */}
        <button
          id="export-csv-btn"
          onClick={handleExportCSV}
          className="export-btn"
          title="Download last 24 hours of energy data as CSV"
        >
          <Download size={14} style={{ marginRight: 6 }} />
          Export 24h Energy Data
        </button>
      </div>
    </div>
  );
};

export default SystemStatus;
