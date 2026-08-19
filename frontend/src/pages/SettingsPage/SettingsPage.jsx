import React, { useState } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import {
  Settings as SettingsIcon,
  Moon,
  Sun,
  Shield,
  Zap,
  Save,
  CheckCircle2,
  Sliders,
  DollarSign,
  Bell,
} from 'lucide-react';

const SettingsPage = () => {
  const { theme, toggleTheme, isDark } = useTheme();

  const [costPerKwh, setCostPerKwh] = useState('8.00');
  const [maxSafetyThreshold, setMaxSafetyThreshold] = useState('4500');
  const [arcFaultEnabled, setArcFaultEnabled] = useState(true);
  const [touOptimization, setTouOptimization] = useState(true);
  const [empathyComfortGate, setEmpathyComfortGate] = useState(true);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [savedStatus, setSavedStatus] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    setSavedStatus(true);
    setTimeout(() => setSavedStatus(false), 3000);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 pb-4 border-b border-gray-200/80 dark:border-gray-800/80">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-blue-600 to-cyan-500 text-white shadow-md shadow-blue-500/20">
            <SettingsIcon size={22} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">
              System Settings & Preferences
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Configure telemetry parameters, safety thresholds, and visual appearance.
            </p>
          </div>
        </div>

        {savedStatus && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 text-xs font-semibold">
            <CheckCircle2 size={14} />
            <span>Preferences Saved</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Section 1: Appearance & Dark Mode */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Sliders size={18} className="text-blue-500" />
            <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              Visual Appearance
            </h2>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200/60 dark:border-gray-700/60">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700 text-amber-500 dark:text-indigo-400">
                {isDark ? <Moon size={20} /> : <Sun size={20} />}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                  Dark Mode
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Toggle between modern sleek dark theme and crisp daylight theme.
                </p>
              </div>
            </div>

            {/* Modern Tailwind-styled Toggle Switch */}
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 capitalize">
                {theme} Mode
              </span>
              <button
                type="button"
                onClick={toggleTheme}
                role="switch"
                aria-checked={isDark}
                className={`relative inline-flex h-7 w-14 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  isDark ? 'bg-blue-600' : 'bg-gray-300'
                }`}
              >
                <span className="sr-only">Toggle theme</span>
                <span
                  className={`pointer-events-none inline-block h-6 w-6 transform rounded-full bg-white shadow-lg ring-0 transition duration-300 ease-in-out flex items-center justify-center text-xs ${
                    isDark ? 'translate-x-7' : 'translate-x-0'
                  }`}
                >
                  {isDark ? <Moon size={12} className="text-blue-600" /> : <Sun size={12} className="text-amber-500" />}
                </span>
              </button>
            </div>
          </div>
        </div>

        {/* Section 2: Energy & Tariffs */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Zap size={18} className="text-amber-500" />
            <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              Grid Tariffs & Power Thresholds
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                Electricity Tariff (₹ per kWh)
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.1"
                  value={costPerKwh}
                  onChange={(e) => setCostPerKwh(e.target.value)}
                  className="w-full pl-8 pr-4 py-2.5 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-xl text-sm font-mono text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <DollarSign size={14} className="absolute left-3 top-3.5 text-gray-400" />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                Max Safety Power Ceiling (Watts)
              </label>
              <input
                type="number"
                step="100"
                value={maxSafetyThreshold}
                onChange={(e) => setMaxSafetyThreshold(e.target.value)}
                className="w-full px-4 py-2.5 bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded-xl text-sm font-mono text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700/60 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                Time-of-Use (ToU) Optimization
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Automatically shift flexible heavy loads to off-peak tariff periods.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setTouOptimization(!touOptimization)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                touOptimization ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-gray-700'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
                  touOptimization ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>

        {/* Section 3: Safety Layer & Empathy Controls */}
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Shield size={18} className="text-red-500" />
            <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
              Safety Breaker & Notifications
            </h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200/60 dark:border-gray-700/60">
              <div>
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                  Arc Fault Automated Relay Trip
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Instantly open hardware relay on high frequency arc signatures.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setArcFaultEnabled(!arcFaultEnabled)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                  arcFaultEnabled ? 'bg-red-500' : 'bg-gray-300 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
                    arcFaultEnabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200/60 dark:border-gray-700/60">
              <div>
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                  RL Empathy Comfort Gate
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Block power curtailment if predicted thermal comfort (PMV) breaches comfort band.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setEmpathyComfortGate(!empathyComfortGate)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                  empathyComfortGate ? 'bg-indigo-500' : 'bg-gray-300 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
                    empathyComfortGate ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200/60 dark:border-gray-700/60">
              <div className="flex items-center gap-2">
                <Bell size={16} className="text-blue-500" />
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                    Push Notifications & Telemetry Warnings
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Receive realtime popups when anomalous wattage or arc spikes occur.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                  notificationsEnabled ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-700'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
                    notificationsEnabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="submit"
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm shadow-md shadow-blue-500/20 transition-all duration-200 cursor-pointer active:scale-98"
          >
            <Save size={16} />
            <span>Save Preferences</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default SettingsPage;
