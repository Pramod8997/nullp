import React from 'react';
import { Activity, Wifi, WifiOff, Database, Brain, DollarSign } from 'lucide-react';

const SystemStatus = ({ connectionStatus, pipelineStatus, analytics, deviceCount }) => {
  const wsStatus = connectionStatus === 'connected' ? 'ok' : connectionStatus === 'reconnecting' ? 'warn' : 'error';
  const pipeStatus = pipelineStatus === 'connected' ? 'ok' : pipelineStatus === 'mqtt_reconnecting' ? 'warn' : 'error';

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
      </div>
    </div>
  );
};

export default SystemStatus;
