import React from 'react';
import { AlertTriangle, ShieldAlert, ShieldCheck, Zap, Clock, Cpu } from 'lucide-react';

const SafetyAlerts = ({ alerts = [], maxAlerts = 50 }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 bg-white/40 dark:bg-gray-800/40 rounded-2xl border border-dashed border-gray-200 dark:border-gray-700 text-center">
        <div className="p-3 rounded-full bg-emerald-50 dark:bg-emerald-950/60 text-emerald-500 mb-3">
          <ShieldCheck size={36} />
        </div>
        <p className="text-sm font-bold text-emerald-600 dark:text-emerald-400">System Nominal</p>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">No safety thresholds breached. Relay breakers active.</p>
      </div>
    );
  }

  const alertList = alerts.slice(0, maxAlerts);

  return (
    <div className="flex flex-col gap-3 max-h-[500px] overflow-y-auto pr-1">
      {alertList.map((alert) => {
        const level = (alert.level || alert.severity || 'WARNING').toUpperCase();
        const isCritical = level === 'CRITICAL' || level === 'SAFETY_CUTOFF';
        const isArcFault = level === 'ARC_FAULT' || (alert.message && alert.message.includes('ARC'));
        
        // CSS class string maintaining test compatibility
        const itemClass = `alert-item ${isCritical ? 'critical' : ''} ${isArcFault ? 'arc-fault' : ''} ${!isCritical && !isArcFault ? 'warning' : ''} relative p-4 rounded-xl border transition-all duration-200 flex items-start gap-3.5 shadow-sm ${
          isArcFault
            ? 'bg-rose-50 text-rose-900 dark:bg-rose-950/30 dark:text-rose-200 border-rose-300 dark:border-rose-800/60 shadow-[0_0_12px_rgba(244,63,94,0.15)]'
            : isCritical
            ? 'bg-red-50 text-red-900 dark:bg-red-950/30 dark:text-red-200 border-red-300 dark:border-red-800/60'
            : 'bg-amber-50 text-amber-900 dark:bg-amber-950/30 dark:text-amber-200 border-amber-200 dark:border-amber-800/60'
        }`;

        const deviceName = alert.device_id || alert.device;

        return (
          <div key={alert.id} data-testid={`alert-${alert.id}`} className={itemClass}>
            {/* Icon */}
            <div className="shrink-0 mt-0.5">
              {isArcFault ? (
                <span data-testid={`alert-icon-${alert.id}`} className="arc-fault-icon p-2 rounded-lg bg-rose-500/20 text-rose-600 dark:text-rose-400 inline-flex">
                  <Zap size={18} className="animate-pulse" />
                </span>
              ) : isCritical ? (
                <div className="p-2 rounded-lg bg-red-500/20 text-red-600 dark:text-red-400 inline-flex">
                  <ShieldAlert size={18} />
                </div>
              ) : (
                <div className="p-2 rounded-lg bg-amber-500/20 text-amber-600 dark:text-amber-400 inline-flex">
                  <AlertTriangle size={18} />
                </div>
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-extrabold uppercase tracking-wide px-2 py-0.5 rounded-md ${
                    isArcFault
                      ? 'bg-rose-200/70 text-rose-900 dark:bg-rose-900/80 dark:text-rose-200'
                      : isCritical
                      ? 'bg-red-200/70 text-red-900 dark:bg-red-900/80 dark:text-red-200'
                      : 'bg-amber-200/70 text-amber-900 dark:bg-amber-900/80 dark:text-amber-200'
                  }`}>
                    {isArcFault ? 'ARC FAULT DETECTED' : isCritical ? 'CRITICAL SAFETY CUTOFF' : 'WARNING'}
                  </span>

                  {deviceName && (
                    <span className="flex items-center gap-1 text-xs font-mono font-bold text-gray-700 dark:text-gray-300">
                      <Cpu size={12} />
                      {deviceName}
                    </span>
                  )}
                </div>

                {alert.timestamp && (
                  <span className="flex items-center gap-1 text-[11px] font-mono text-gray-500 dark:text-gray-400">
                    <Clock size={11} />
                    {alert.timestamp}
                  </span>
                )}
              </div>

              <p className="text-sm font-medium leading-relaxed break-words">
                {alert.message}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SafetyAlerts;
