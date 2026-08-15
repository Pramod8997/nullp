import React from 'react';
import { AlertTriangle, ShieldAlert, ShieldCheck } from 'lucide-react';

const SafetyAlerts = ({ alerts = [], maxAlerts = 50 }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="empty-state">
        <ShieldCheck size={48} color="#10b981" />
        <p style={{ fontWeight: 600, color: '#10b981' }}>System Nominal</p>
        <p>No safety thresholds breached.</p>
      </div>
    );
  }

  const alertList = alerts.slice(0, maxAlerts);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '400px', overflowY: 'auto' }}>
      {alertList.map((alert) => {
        const level = (alert.level || alert.severity || 'WARNING').toUpperCase();
        const isCritical = level === 'CRITICAL';
        const isArcFault = level === 'ARC_FAULT';
        const itemClass = `alert-item ${isCritical ? 'critical' : isArcFault ? 'arc-fault' : 'warning'}`;

        return (
          <div key={alert.id} data-testid={`alert-${alert.id}`} className={itemClass}>
            {isArcFault ? (
              <span data-testid={`alert-icon-${alert.id}`} className="arc-fault-icon">
                <AlertTriangle size={20} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
              </span>
            ) : isCritical ? (
              <ShieldAlert size={20} color="#ef4444" style={{ flexShrink: 0, marginTop: 2 }} />
            ) : (
              <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0, marginTop: 2 }} />
            )}
            <div>
              <div className="alert-item__title" style={{ color: isCritical || isArcFault ? '#ef4444' : '#f59e0b' }}>
                {isCritical ? 'CRITICAL CUTOFF' : isArcFault ? 'ARC FAULT' : 'WARNING'}
                {alert.device_id || alert.device ? ` — ${alert.device_id || alert.device}` : ''}
              </div>
              <div className="alert-item__message">{alert.message}</div>
              <div className="alert-item__time">{alert.timestamp}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SafetyAlerts;
