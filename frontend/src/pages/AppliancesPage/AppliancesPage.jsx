import React from 'react';
import DeviceCards from '../../components/DeviceCards';
import ApplianceTable from '../../components/ApplianceTable/ApplianceTable';
import { Cpu, Zap, Activity } from 'lucide-react';

const AppliancesPage = ({ devices = {} }) => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20">
          <Cpu size={22} />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
            Appliance Fleet & Sub-Metering
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Real-time individual load disaggregation, wattage limits, and state control.
          </p>
        </div>
      </div>

      {/* Device Fleet Cards */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
        <DeviceCards devices={devices} />
      </div>

      {/* Detailed Appliance Table */}
      <ApplianceTable devices={devices} />
    </div>
  );
};

export default AppliancesPage;
