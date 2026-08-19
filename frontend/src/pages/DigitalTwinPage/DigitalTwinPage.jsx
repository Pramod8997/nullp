import React from 'react';
import DigitalTwin from '../../components/DigitalTwin';
import DigitalTwinVisualization from '../../components/DigitalTwinVisualization/DigitalTwinVisualization';
import RealTimeChart from '../../components/RealTimeChart';
import DeviceCards from '../../components/DeviceCards';
import PhantomTracker from '../../components/PhantomTracker';
import { Zap } from 'lucide-react';

const DigitalTwinPage = ({
  devices = {},
  powerHistory = [],
  twinEvents = [],
  pmvScore = 0,
  phantomData = {},
  pendingUnknowns = [],
}) => {
  return (
    <div className="space-y-6">
      {/* Top: Full-Width 4-Zone Digital Twin Visualization */}
      <DigitalTwinVisualization devices={devices} />

      {/* Middle Row: Power Telemetry + AI RL Digital Twin Agent */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-gray-200/60 dark:border-gray-700/60">
            <div className="flex items-center gap-2">
              <Zap size={18} className="text-blue-500" />
              <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
                Real-Time Telemetry Stream
              </h2>
            </div>
            <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
              {Object.keys(devices).length} Channels
            </span>
          </div>
          <RealTimeChart data={powerHistory} devices={devices} />
        </div>

        <div className="lg:col-span-5 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
          <DigitalTwin events={twinEvents} pmvScore={pmvScore} unknownDevices={pendingUnknowns} />
        </div>
      </div>

      {/* Bottom Row: Device Fleet + Phantom Load Tracker */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
          <DeviceCards devices={devices} />
        </div>

        <div className="lg:col-span-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
          <PhantomTracker data={phantomData} />
        </div>
      </div>
    </div>
  );
};

export default DigitalTwinPage;
