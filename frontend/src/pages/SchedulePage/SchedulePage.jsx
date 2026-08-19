import React, { useState } from 'react';
import { Calendar, Clock, Plus, Zap, Sparkles, CheckCircle2 } from 'lucide-react';

const mockSchedules = [
  {
    id: 1,
    appliance: 'Dishwasher',
    zone: 'Kitchen',
    time: '02:00 AM - 04:00 AM',
    tariff: 'Super Off-Peak ($0.08/kWh)',
    status: 'Scheduled',
    autoOptimized: true,
  },
  {
    id: 2,
    appliance: 'Washing Machine',
    zone: 'Utility & Bathroom',
    time: '04:00 AM - 05:30 AM',
    tariff: 'Super Off-Peak ($0.08/kWh)',
    status: 'Scheduled',
    autoOptimized: true,
  },
  {
    id: 3,
    appliance: 'HVAC Pre-Cooling',
    zone: 'Living Room',
    time: '01:00 PM - 02:00 PM',
    tariff: 'Pre-Peak Buffer ($0.12/kWh)',
    status: 'Active',
    autoOptimized: true,
  },
];

const SchedulePage = () => {
  const [schedules, setSchedules] = useState(mockSchedules);

  const toggleScheduleStatus = (id) => {
    setSchedules((prev) =>
      prev.map((s) =>
        s.id === id ? { ...s, status: s.status === 'Active' ? 'Paused' : 'Active' } : s
      )
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-600 text-white shadow-md shadow-indigo-500/20">
            <Calendar size={22} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
              Automated Load Scheduling & ToU Dispatch
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Shift high-demand flexible loads into cheapest tariff windows autonomously.
            </p>
          </div>
        </div>

        <button className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-md shadow-blue-500/20 transition-all cursor-pointer self-start sm:self-auto">
          <Plus size={15} />
          <span>New Schedule Rule</span>
        </button>
      </div>

      {/* Hero Banner: AI Optimization */}
      <div className="p-5 rounded-3xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="p-3 rounded-2xl bg-white/10 backdrop-blur-md">
            <Sparkles size={24} className="text-amber-300 animate-pulse" />
          </div>
          <div>
            <h2 className="text-base font-bold">Autonomous Time-of-Use Optimizer Active</h2>
            <p className="text-xs text-blue-100 mt-0.5 max-w-xl">
              Algorithms continuously analyze day-ahead grid tariffs and dynamically schedule heavy appliances during off-peak valleys, saving up to 28% monthly.
            </p>
          </div>
        </div>
        <div className="px-4 py-2 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 text-xs font-mono font-bold whitespace-nowrap">
          Next Off-Peak: 11:00 PM
        </div>
      </div>

      {/* Schedule Table/Cards */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-gray-200/60 dark:border-gray-700/60">
          <h2 className="text-base font-bold text-gray-900 dark:text-white">Active Dispatch Queue</h2>
          <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-800">
            {schedules.length} Active Rules
          </span>
        </div>

        <div className="space-y-3">
          {schedules.map((item) => (
            <div
              key={item.id}
              onClick={() => toggleScheduleStatus(item.id)}
              className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-800/60 border border-gray-200/60 dark:border-gray-700/60 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:border-blue-300 dark:hover:border-blue-700 transition-all cursor-pointer"
            >
              <div className="flex items-center gap-3.5">
                <div className="p-2.5 rounded-xl bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border border-gray-200 dark:border-gray-700 shadow-sm">
                  <Zap size={18} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-gray-900 dark:text-white">
                      {item.appliance}
                    </span>
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-200/70 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                      {item.zone}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 mt-1">
                    <span className="flex items-center gap-1">
                      <Clock size={12} />
                      {item.time}
                    </span>
                    <span>•</span>
                    <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                      {item.tariff}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 self-end sm:self-center">
                {item.autoOptimized && (
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-950/60 border border-purple-200 dark:border-purple-800 px-2.5 py-1 rounded-full">
                    <Sparkles size={11} />
                    Auto-Optimized
                  </span>
                )}
                <span className={`text-xs font-bold px-3 py-1 rounded-full border ${
                  item.status === 'Active'
                    ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/60 border-emerald-200 dark:border-emerald-800'
                    : 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
                }`}>
                  {item.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SchedulePage;
