import React, { useState, useEffect } from 'react';
import SafetyAlerts from '../../components/SafetyAlerts';
import { Bell, ShieldAlert, ShieldCheck, AlertTriangle, Zap, CheckCircle2 } from 'lucide-react';

const AlertsPage = ({ alerts = [] }) => {
  const [filter, setFilter] = useState('all');

  const criticalCount = alerts.filter(
    (a) => (a.level || a.severity || '').toUpperCase() === 'CRITICAL' || (a.message && a.message.includes('ARC'))
  ).length;
  const warningCount = alerts.length - criticalCount;

  // Breaker status is derived from the live alert feed, not hardcoded:
  // a cutoff / overcurrent / arc-fault alert WITHIN THE LOCKOUT WINDOW
  // (5 min, the firmware's SAFETY_LOCKOUT_MS) means the relay is open.
  // Older alerts leave it "Armed & Nominal" — the lockout has expired and
  // the relay can be re-energized, so a stale alert must not claim TRIPPED
  // forever. The alert id is Date.now()-based at append time, so it doubles
  // as the machine timestamp; the interval re-render lets the window expire.
  const [nowMs, setNowMs] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 5000);
    return () => clearInterval(id);
  }, []);
  const LOCKOUT_MS = 5 * 60 * 1000;
  const isTripped = alerts.some(
    (a) =>
      (a?.type === 'SAFETY_CUTOFF' ||
        /OVERCURRENT|ARC[_ ]?FAULT|RELAY.*FORCED OFF|RELAY FORCED OFF/i.test(a?.message || '')) &&
      nowMs - (a?.id || 0) < LOCKOUT_MS
  );

  const filteredAlerts = alerts.filter((alert) => {
    if (filter === 'critical') {
      return (alert.level || alert.severity || '').toUpperCase() === 'CRITICAL' || (alert.message && alert.message.includes('ARC'));
    }
    if (filter === 'warning') {
      return (alert.level || alert.severity || '').toUpperCase() !== 'CRITICAL' && !(alert.message && alert.message.includes('ARC'));
    }
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-rose-500 to-red-600 text-white shadow-md shadow-red-500/20">
              <Bell size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
                Safety Alerts & Incident Log
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Real-time safety cutoffs, arc-fault detection, and anomaly notifications.
              </p>
            </div>
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-2 bg-white dark:bg-gray-800 p-1 rounded-xl border border-gray-200 dark:border-gray-700 self-start sm:self-auto">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filter === 'all'
                ? 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900 shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            All ({alerts.length})
          </button>
          <button
            onClick={() => setFilter('critical')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filter === 'critical'
                ? 'bg-rose-600 text-white shadow-sm'
                : 'text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40'
            }`}
          >
            Critical ({criticalCount})
          </button>
          <button
            onClick={() => setFilter('warning')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              filter === 'warning'
                ? 'bg-amber-500 text-white shadow-sm'
                : 'text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-950/40'
            }`}
          >
            Warnings ({warningCount})
          </button>
        </div>
      </div>

      {/* Safety Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div
          data-testid="breaker-status"
          className={`bg-white/80 dark:bg-gray-800/80 backdrop-blur-md rounded-2xl p-4 border shadow-sm flex items-center justify-between ${
            isTripped
              ? 'border-rose-300 dark:border-rose-800'
              : 'border-gray-200/80 dark:border-gray-700/80'
          }`}
        >
          <div>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Breaker Status</span>
            <div
              className={`text-lg font-bold mt-0.5 flex items-center gap-1.5 ${
                isTripped
                  ? 'text-rose-600 dark:text-rose-400'
                  : 'text-emerald-600 dark:text-emerald-400'
              }`}
            >
              {isTripped ? <ShieldAlert size={16} /> : <CheckCircle2 size={16} />}
              <span>{isTripped ? 'TRIPPED — relay open' : 'Armed & Nominal'}</span>
            </div>
          </div>
          <div
            className={`p-2.5 rounded-xl ${
              isTripped ? 'bg-rose-500/10 text-rose-500' : 'bg-emerald-500/10 text-emerald-500'
            }`}
          >
            {isTripped ? <ShieldAlert size={20} /> : <ShieldCheck size={20} />}
          </div>
        </div>

        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md rounded-2xl p-4 border border-gray-200/80 dark:border-gray-700/80 shadow-sm flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Arc Fault Cutoffs</span>
            <div className="text-lg font-bold text-gray-900 dark:text-white font-mono mt-0.5">
              {criticalCount} Triggered
            </div>
          </div>
          <div className="p-2.5 rounded-xl bg-rose-500/10 text-rose-500">
            <Zap size={20} />
          </div>
        </div>

        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md rounded-2xl p-4 border border-gray-200/80 dark:border-gray-700/80 shadow-sm flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Soft Anomalies</span>
            <div className="text-lg font-bold text-amber-600 dark:text-amber-400 font-mono mt-0.5">
              {warningCount} Active
            </div>
          </div>
          <div className="p-2.5 rounded-xl bg-amber-500/10 text-amber-500">
            <AlertTriangle size={20} />
          </div>
        </div>
      </div>

      {/* Main Alert Feed Container */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-gray-200/60 dark:border-gray-700/60">
          <div className="flex items-center gap-2">
            <ShieldAlert size={18} className="text-rose-500" />
            <h2 className="text-base font-bold text-gray-900 dark:text-white">Safety Layer Events</h2>
          </div>
          <span className="text-xs font-bold font-mono px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
            {filteredAlerts.length} Events
          </span>
        </div>

        <SafetyAlerts alerts={filteredAlerts} />
      </div>
    </div>
  );
};

export default AlertsPage;
