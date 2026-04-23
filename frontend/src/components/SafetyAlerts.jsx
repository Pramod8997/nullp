import React from 'react';
import { AlertTriangle, ShieldAlert, ShieldCheck } from 'lucide-react';

const SafetyAlerts = ({ alerts }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="empty-state">
        <ShieldCheck size={48} color="#10b981" />
        <p style={{ fontWeight: 600, color: '#10b981' }}>System Nominal</p>
        <p>No safety thresholds breached.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '400px', overflowY: 'auto' }}>
      {alerts.slice(0, 20).map((alert) => (
        <div key={alert.id} className={`alert-item ${alert.severity}`}>
          {alert.severity === 'critical' ? (
            <ShieldAlert size={20} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
          ) : (
            <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: 2 }} />
          )}
          <div>
            <div className="alert-item__title" style={{ color: alert.severity === 'critical' ? '#ef4444' : '#f59e0b' }}>
              {alert.severity === 'critical' ? 'CRITICAL CUTOFF' : 'WARNING'}
              {alert.device_id ? ` — ${alert.device_id}` : ''}
            </div>
            <div className="alert-item__message">{alert.message}</div>
            <div className="alert-item__time">{alert.timestamp}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default SafetyAlerts;
