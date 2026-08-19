import React from 'react';
import RealTimeChart from '../../components/RealTimeChart';
import EnergyChart from '../../components/EnergyChart/EnergyChart';
import SystemStatus from '../../components/SystemStatus';
import { BarChart3, Zap, Activity } from 'lucide-react';

const AnalyticsPage = ({
  devices = {},
  powerHistory = [],
  connectionStatus,
  pipelineStatus,
  analytics,
  latencyStats,
  latencyHistory,
}) => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20">
          <BarChart3 size={22} />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
            Energy Analytics & Pipeline Telemetry
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Deep-dive consumption patterns, load shapes, and system inference latency.
          </p>
        </div>
      </div>

      {/* Row 1: Charts Bento */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Real Time Power Monitor */}
        <div className="lg:col-span-7 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-gray-200/60 dark:border-gray-700/60">
            <div className="flex items-center gap-2">
              <Zap size={18} className="text-blue-500" />
              <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
                Real-Time Power Monitor
              </h2>
            </div>
            <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
              {Object.keys(devices).length} Channels
            </span>
          </div>
          <RealTimeChart data={powerHistory} devices={devices} />
        </div>

        {/* Energy Chart */}
        <div className="lg:col-span-5 flex flex-col">
          <EnergyChart powerHistory={powerHistory} />
        </div>
      </div>

      {/* Row 2: System Telemetry & SLA Panel */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
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
