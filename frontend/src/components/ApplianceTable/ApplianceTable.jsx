import React from 'react';
import { Cpu, Zap, Activity } from 'lucide-react';

const getConfidenceInfo = (confidence) => {
  if (confidence === undefined || confidence === null) return { label: '—', cls: 'bg-gray-100 dark:bg-gray-800 text-gray-400' };
  const pct = Math.round(confidence * 100);
  if (pct >= 80) return { label: `${pct}%`, cls: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800' };
  if (pct >= 40) return { label: `${pct}%`, cls: 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border border-amber-200 dark:border-amber-800' };
  return { label: `${pct}%`, cls: 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border border-rose-200 dark:border-rose-800' };
};

const formatDeviceName = (id) => {
  return id
    .replace('esp32_', '')
    .replace('node_', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const ApplianceTable = ({ devices = {} }) => {
  const deviceEntries = Object.entries(devices);

  if (deviceEntries.length === 0) {
    return (
      <div className="w-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-5 shadow-sm">
        <div className="flex items-center gap-2 pb-3 border-b border-gray-200/60 dark:border-gray-700/60">
          <Cpu size={18} className="text-blue-500" />
          <h3 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
            Real-Time Appliance Status
          </h3>
        </div>
        <div className="flex flex-col items-center justify-center p-8 text-center text-xs text-gray-400">
          No appliances detected in telemetry
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-5 shadow-sm">
      <div className="flex items-center justify-between pb-3 border-b border-gray-200/60 dark:border-gray-700/60 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
            <Cpu size={18} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              Real-Time Appliance Status
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">Granular sub-metering breakdown</p>
          </div>
        </div>
        <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
          {deviceEntries.length} Units
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-gray-100 dark:border-gray-700/60 text-gray-500 dark:text-gray-400 uppercase tracking-wider font-semibold">
              <th className="pb-3 pt-1 px-3">Appliance</th>
              <th className="pb-3 pt-1 px-3">Status</th>
              <th className="pb-3 pt-1 px-3">Power (W)</th>
              <th className="pb-3 pt-1 px-3">Energy (kWh)</th>
              <th className="pb-3 pt-1 px-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-700/40">
            {deviceEntries.map(([id, dev]) => {
              const power = dev?.power || 0;
              const state = dev?.state || (power > 10 ? 'ON' : 'OFF');
              const isOn = state === 'ON';
              const energy = (power / 1000) * 0.5;
              const confidence = dev?.confidence;
              const confInfo = getConfidenceInfo(confidence);

              return (
                <tr key={id} className="hover:bg-gray-50/60 dark:hover:bg-gray-700/30 transition-colors">
                  <td className="py-3 px-3 font-semibold text-gray-900 dark:text-white">
                    {dev?.label || formatDeviceName(id)}
                  </td>
                  <td className="py-3 px-3">
                    <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold ${
                      isOn
                        ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
                        : 'bg-gray-100 text-gray-500 dark:bg-gray-700/50 dark:text-gray-400'
                    }`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${isOn ? 'bg-emerald-500 animate-pulse' : 'bg-gray-400'}`} />
                      {state}
                    </span>
                  </td>
                  <td className="py-3 px-3 font-mono font-bold text-gray-900 dark:text-white">
                    {isOn ? power.toFixed(0) : '0'} W
                  </td>
                  <td className="py-3 px-3 font-mono text-gray-600 dark:text-gray-300">
                    {isOn ? energy.toFixed(2) : '0.00'}
                  </td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded-md font-mono text-[11px] font-semibold ${confInfo.cls}`}>
                      {confInfo.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-[11px] text-gray-400 dark:text-gray-500 pt-4 mt-2 border-t border-gray-100 dark:border-gray-700/40">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-500" /> High Confidence (&gt; 80%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-amber-500" /> Medium (40% - 80%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-rose-500" /> Unknown (&lt; 40%)
        </span>
      </div>
    </div>
  );
};

export default ApplianceTable;
