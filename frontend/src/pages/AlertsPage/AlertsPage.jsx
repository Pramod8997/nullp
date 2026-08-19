import React from 'react';
import SafetyAlerts from '../../components/SafetyAlerts';
import { Bell } from 'lucide-react';
import './AlertsPage.css';

const AlertsPage = ({ alerts }) => {
  return (
    <div className="alerts-page">
      <div className="alerts-page__header">
        <h2>
          <Bell size={22} style={{ marginRight: 8, verticalAlign: 'middle' }} />
          Alerts & Notifications
        </h2>
        <p className="alerts-page__subtitle">
          Real-time safety alerts, anomaly detection, and system notifications.
        </p>
      </div>
      <div className="panel">
        <div className="panel-header">
          <h2>Safety Layer</h2>
          <span className="panel-badge">{alerts.length} events</span>
        </div>
        <SafetyAlerts alerts={alerts} />
      </div>
    </div>
  );
};

export default AlertsPage;
