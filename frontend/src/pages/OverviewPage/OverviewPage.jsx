import React from 'react';
import SummaryCards from '../../components/SummaryCards/SummaryCards';
import DeviceCards from '../../components/DeviceCards';
import RealTimeChart from '../../components/RealTimeChart';
import EnergyChart from '../../components/EnergyChart/EnergyChart';
import DigitalTwinVisualization from '../../components/DigitalTwinVisualization/DigitalTwinVisualization';
import { Zap } from 'lucide-react';

const OverviewPage = ({ devices = {}, powerHistory = [] }) => {
  return (
    <div className="space-y-6">
      {/* ── Top Row: Summary KPI Widgets ── */}
      <section>
        <SummaryCards devices={devices} powerHistory={powerHistory} />
      </section>

      {/* ── Middle Bento Row: Telemetry & Historical Charts ── */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Real-time 1Hz Power Monitor Bento Box */}
        <div className="lg:col-span-7 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between pb-4 border-b border-gray-200/60 dark:border-gray-700/60">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
                <Zap size={18} />
              </div>
              <div>
                <h2 className="text-base font-bold text-gray-900 dark:text-white">
                  Real-Time Power Monitor
                </h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Live 1Hz disaggregated wattage stream
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
                {Object.keys(devices).length} Channels
              </span>
            </div>
          </div>

          <div className="py-2">
            <RealTimeChart data={powerHistory} devices={devices} />
          </div>
        </div>

        {/* Energy Consumption History Bento Box */}
        <div className="lg:col-span-5 flex flex-col">
          <EnergyChart powerHistory={powerHistory} />
        </div>
      </section>

      {/* ── Bottom Bento Row: Floor Plan Digital Twin & Device Fleet ── */}
      <section className="space-y-6">
        {/* Digital Twin 4-Room Interactive Blueprint */}
        <DigitalTwinVisualization devices={devices} />

        {/* Device Fleet Cards */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
          <DeviceCards devices={devices} />
        </div>
      </section>
    </div>
  );
};

export default OverviewPage;
