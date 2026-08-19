import React from 'react';
import { Cpu, Power, PowerOff, Zap, AlertCircle } from 'lucide-react';

const formatDeviceName = (id) => {
  return id
    .replace('esp32_', '')
    .replace('node_', '')
    .replace(/_/g, ' ')
    .toUpperCase();
};

const DeviceCards = ({ devices = {} }) => {
  const deviceIds = Object.keys(devices || {});

  if (deviceIds.length === 0) {
    return (
      <div className="w-full">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-gray-200/80 dark:border-gray-700/80">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
              <Cpu size={18} />
            </div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">Device Fleet</h2>
          </div>
          <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
            Waiting...
          </span>
        </div>
        <div className="flex flex-col items-center justify-center p-12 bg-white/40 dark:bg-gray-800/40 rounded-2xl border border-dashed border-gray-200 dark:border-gray-700 text-center">
          <Cpu className="h-10 w-10 text-gray-400 dark:text-gray-500 mb-2 opacity-50" />
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">No Devices Connected</p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">Telemetry stream waiting for ESP32 nodes</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Fleet Header */}
      <div className="flex items-center justify-between pb-4 mb-6 border-b border-gray-200/80 dark:border-gray-700/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 shadow-sm">
            <Cpu size={18} />
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">Device Fleet</h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">Active telemetry and load disaggregation</p>
          </div>
        </div>
        <span className="px-3 py-1 text-xs font-bold font-mono rounded-full bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
          {deviceIds.length} Active
        </span>
      </div>

      {/* Modern Responsive Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {deviceIds.map((id) => {
          const dev = devices[id];
          const power = dev?.power ?? 0;
          const state = dev?.state || (power > 10 ? 'ON' : 'OFF');
          const classification = dev?.classification || 'pending';
          const isPending = classification === 'pending' || classification === 'syncing';
          const isOn = state === 'ON';
          const isGlow = dev?.rated ? power > 0.8 * dev.rated : false;

          return (
            <div
              key={id}
              data-testid={`device-card-${id}`}
              className={`relative bg-white dark:bg-gray-800 shadow-sm hover:shadow-md border rounded-xl p-5 transition-all duration-300 flex flex-col justify-between overflow-hidden group ${
                isGlow
                  ? 'glow border-amber-400 dark:border-amber-500 ring-2 ring-amber-400/40 dark:ring-amber-500/40 shadow-lg shadow-amber-500/10'
                  : isOn
                  ? 'border-gray-200 dark:border-gray-700 hover:border-emerald-400 dark:hover:border-emerald-600'
                  : 'border-gray-200 dark:border-gray-700 opacity-80 hover:opacity-100'
              }`}
            >
              {/* Inner card with test id */}
              <div data-testid="device-card" className={`flex flex-col h-full justify-between ${isGlow ? 'glow' : ''}`}>
                {/* Header: Status icon & Device name */}
                <div className="flex items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2 truncate">
                    {isOn ? (
                      <span className="relative flex h-2.5 w-2.5 shrink-0">
                        <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
                      </span>
                    ) : (
                      <PowerOff size={14} className="text-gray-400 dark:text-gray-500 shrink-0" />
                    )}
                    <span className="font-bold text-sm text-gray-900 dark:text-white tracking-tight truncate">
                      {dev?.label || formatDeviceName(id)}
                    </span>
                  </div>

                  {/* Rated limit or warning badge */}
                  {dev?.rated && (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-500 dark:text-gray-400 shrink-0">
                      Max {dev.rated}W
                    </span>
                  )}
                </div>

                {/* Power Metric Display */}
                <div className="my-3">
                  {power === 0 || !isOn ? (
                    <div className="text-2xl font-black font-mono tracking-tight text-gray-400 dark:text-gray-500">
                      0W
                    </div>
                  ) : (
                    <div className="text-2xl font-black font-mono tracking-tight text-gray-900 dark:text-white flex items-baseline gap-1">
                      <span>{power.toFixed(1)}</span>
                      <span className="text-sm font-semibold text-gray-500 dark:text-gray-400 font-sans">W</span>
                    </div>
                  )}
                </div>

                {/* Meta row: State badge & NILM Classification */}
                <div className="flex items-center justify-between gap-2 pt-3 border-t border-gray-100 dark:border-gray-700/60 text-xs">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full font-semibold ${
                      isOn
                        ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200/80 dark:border-emerald-800/80'
                        : 'bg-gray-100 text-gray-500 dark:bg-gray-700/50 dark:text-gray-400 border border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    {state}
                  </span>

                  {isPending ? (
                    <div className="animate-pulse bg-gray-200 dark:bg-gray-700 h-5 w-20 rounded-md" />
                  ) : (
                    <span
                      className={`font-mono text-[11px] px-2 py-0.5 rounded-md font-medium truncate max-w-[130px] ${
                        classification.startsWith('known:')
                          ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                          : classification === 'unknown'
                          ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {classification.startsWith('known:')
                        ? `✓ ${classification.split(':')[1]}`
                        : classification === 'unknown'
                        ? '? Unknown'
                        : classification}
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DeviceCards;
