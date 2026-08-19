import React from 'react';
import { Ghost, Zap } from 'lucide-react';

const PhantomTracker = ({ data = {} }) => {
  const { loads = {}, total = 0 } = data;
  const loadEntries = Object.entries(loads).filter(([, v]) => v > 0);

  return (
    <div className="w-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-200/80 dark:border-gray-700/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400">
            <Ghost size={18} />
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">Phantom Load Tracker</h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">Standby vampire power consumption</p>
          </div>
        </div>
        <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border border-amber-200 dark:border-amber-800">
          {loadEntries.length} Detected
        </span>
      </div>

      {/* Vampire Meter Hero Card */}
      <div className="p-4 rounded-2xl bg-gradient-to-br from-amber-50/50 via-white/40 to-orange-50/30 dark:from-amber-950/20 dark:via-gray-800/40 dark:to-orange-950/10 border border-amber-200/80 dark:border-amber-800/60 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400">
            <Zap size={24} />
          </div>
          <div>
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Total Vampire Draw
            </span>
            <div className="text-2xl font-black font-mono text-gray-900 dark:text-white">
              {total.toFixed(2)} <span className="text-sm font-sans font-semibold text-amber-600">W</span>
            </div>
          </div>
        </div>
        <span className="text-[11px] font-semibold text-amber-700 dark:text-amber-400 bg-amber-100/70 dark:bg-amber-900/60 px-2.5 py-1 rounded-lg">
          ~₹{((total * 24 * 30 * 8) / 1000).toFixed(0)}/mo waste
        </span>
      </div>

      {/* Per-device list */}
      {loadEntries.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-8 bg-gray-50/50 dark:bg-gray-800/40 rounded-2xl border border-dashed border-gray-200 dark:border-gray-700 text-center">
          <Ghost size={24} className="text-gray-400 dark:text-gray-500 mb-1 opacity-50" />
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">No phantom loads detected</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
          {loadEntries
            .sort(([, a], [, b]) => b - a)
            .map(([deviceId, watts]) => (
              <div
                key={deviceId}
                className="flex items-center justify-between p-2.5 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-100 dark:border-gray-700/60 text-xs"
              >
                <span className="font-semibold text-gray-800 dark:text-gray-200">
                  {deviceId.replace('esp32_', '').replace('node_', '').toUpperCase()}
                </span>
                <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                  {watts.toFixed(3)} W
                </span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};

export default PhantomTracker;
