import React from 'react';
import RealTimeChart from '../../components/RealTimeChart';
import EnergyChart from '../../components/EnergyChart/EnergyChart';
import SystemStatus from '../../components/SystemStatus';
import './AnalyticsPage.css';

const AnalyticsPage = ({
  devices,
  powerHistory,
  connectionStatus,
  pipelineStatus,
  analytics,
  latencyStats,
  latencyHistory,
}) => {
  return (
    <div className="analytics-page">
      <div className="analytics-page__header">
        <h2>Energy Analytics</h2>
        <p className="analytics-page__subtitle">
          Detailed energy consumption patterns and system performance metrics.
        </p>
      </div>

      <div className="analytics-page__row">
        <div className="analytics-page__col">
          <div className="panel">
            <div className="panel-header">
              <h2>⚡ Real-Time Power Monitor</h2>
              <span className="panel-badge">{Object.keys(devices).length} devices</span>
            </div>
            <RealTimeChart data={powerHistory} devices={devices} />
          </div>
        </div>
        <div className="analytics-page__col">
          <EnergyChart powerHistory={powerHistory} />
        </div>
      </div>

      <div className="panel">
        <SystemStatus
          connectionStatus={connectionStatus}
          pipelineStatus={pipelineStatus}
          analytics={analytics}
          deviceCount={Object.keys(devices).length}
          latencyStats={latencyStats}
          latencyHistory={latencyHistory}
        />
      </div>
    </div>
  );
};

export default AnalyticsPage;
