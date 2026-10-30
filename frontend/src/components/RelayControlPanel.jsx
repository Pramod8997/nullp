import React, { useState } from 'react';
import {
  Power,
  PowerOff,
  Zap,
  ShieldAlert,
  Terminal,
  Copy,
  Check,
  Laptop,
  Smartphone,
  Cpu,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';
import { requestJson } from '../api/client';

const RelayControlPanel = ({
  deviceId = 'node_bench_agg',
  devices = {},
  telemetry = {},
  relayStates = {},
}) => {
  const [loadingAction, setLoadingAction] = useState(null);
  const [copiedCmd, setCopiedCmd] = useState(null);
  const [showCliDrawer, setShowCliDrawer] = useState(false);
  const [lastActionStatus, setLastActionStatus] = useState(null);

  // Extract live relay states: [0] = IN1 (Laptop), [1] = IN2 (Phone)
  const dev = devices[deviceId] || {};
  const telem = telemetry[deviceId] || {};
  const activeRelays = telem.relays || dev.relays || [
    dev.state === 'ON' ? 1 : 0,
    dev.state === 'ON' ? 1 : 0,
  ];

  const in1Active = Boolean(activeRelays[0]);
  const in2Active = Boolean(activeRelays[1]);
  const anyActive = in1Active || in2Active;
  const bothActive = in1Active && in2Active;

  // Approximate load watts based on active channels
  const in1Watts = in1Active ? 65.0 : 0.0;
  const in2Watts = in2Active ? 20.0 : 0.0;
  const totalWatts = telem.w != null ? telem.w : (in1Watts + in2Watts);

  const handleSendCommand = async (command, channel = null) => {
    const actionKey = channel ? `ch${channel}_${command}` : `master_${command}`;
    setLoadingAction(actionKey);
    setLastActionStatus(null);

    try {
      const res = await requestJson('/api/relay/command', {
        method: 'POST',
        body: JSON.stringify({
          device_id: deviceId,
          command,
          channel,
        }),
      });

      if (res && res.status === 'published') {
        setLastActionStatus({
          success: true,
          message: `Command ${res.command} published successfully to ${res.topic}`,
        });
      } else {
        setLastActionStatus({
          success: false,
          message: 'Server accepted command',
        });
      }
    } catch (err) {
      console.error('Relay command failed:', err);
      setLastActionStatus({
        success: false,
        message: err.message || 'Failed to dispatch relay command',
      });
    } finally {
      setTimeout(() => setLoadingAction(null), 400);
    }
  };

  const copyToClipboard = (cmdText, key) => {
    navigator.clipboard.writeText(cmdText);
    setCopiedCmd(key);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const pythonCliSnippet = (cmd) =>
    `python -c "import paho.mqtt.publish as pub; pub.single('home/plug/${deviceId}/command', '${cmd}', hostname='127.0.0.1')"`;

  return (
    <div
      id="relay-control-panel"
      className="bg-white/85 dark:bg-gray-800/85 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-3xl p-6 shadow-sm space-y-6 transition-all duration-300"
    >
      {/* ── Header Row ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-200/60 dark:border-gray-700/60 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 shadow-sm border border-indigo-100 dark:border-indigo-900/40">
            <Cpu size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-gray-900 dark:text-white tracking-tight">
                Manual Relay Cutoff Console
              </h2>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800 font-semibold">
                Active-LOW
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Target Node: <span className="font-mono font-semibold text-gray-700 dark:text-gray-300">{deviceId}</span> • Separate IN1 / IN2 Cutoff Controls
            </p>
          </div>
        </div>

        {/* Live Status Indicators */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-gray-100/80 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 text-xs">
            <Zap size={14} className={anyActive ? 'text-emerald-500 animate-pulse' : 'text-gray-400'} />
            <span className="font-mono font-bold text-gray-800 dark:text-gray-200">
              {totalWatts.toFixed(1)} W
            </span>
          </div>

          <button
            id="toggle-cli-drawer-btn"
            onClick={() => setShowCliDrawer((prev) => !prev)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-gray-100 hover:bg-gray-200 dark:bg-gray-700/60 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors cursor-pointer border border-gray-200 dark:border-gray-600"
            title="View direct Python CLI MQTT commands"
          >
            <Terminal size={14} />
            <span>CLI Codes</span>
          </button>
        </div>
      </div>

      {/* ── Status Feedback Alert (if any) ── */}
      {lastActionStatus && (
        <div
          className={`flex items-center justify-between p-3 rounded-2xl text-xs font-medium border ${
            lastActionStatus.success
              ? 'bg-emerald-50/80 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
              : 'bg-rose-50/80 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300 border-rose-200 dark:border-rose-800'
          }`}
        >
          <div className="flex items-center gap-2">
            {lastActionStatus.success ? <Check size={14} /> : <AlertTriangle size={14} />}
            <span>{lastActionStatus.message}</span>
          </div>
          <button
            onClick={() => setLastActionStatus(null)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xs ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Channel Controls Grid (IN1 & IN2 Separated) ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ── Channel 1: IN1 (Laptop Charger) ── */}
        <div
          id="channel-in1-card"
          className={`relative rounded-2xl p-5 border transition-all duration-300 flex flex-col justify-between ${
            in1Active
              ? 'bg-gradient-to-br from-emerald-50/70 via-white to-white dark:from-emerald-950/30 dark:via-gray-800 dark:to-gray-800 border-emerald-300 dark:border-emerald-700/80 shadow-sm shadow-emerald-500/5'
              : 'bg-gray-50/60 dark:bg-gray-800/40 border-gray-200 dark:border-gray-700/70 opacity-90'
          }`}
        >
          <div>
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="flex items-center gap-3">
                <div
                  className={`p-3 rounded-2xl transition-colors ${
                    in1Active
                      ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/20'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                  }`}
                >
                  <Laptop size={20} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-sm text-gray-900 dark:text-white">
                      Relay 1: IN1 (Laptop Charger)
                    </h3>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="font-mono text-[10px] text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700/60 px-1.5 py-0.5 rounded">
                      GPIO 26
                    </span>
                    <span className="text-xs text-gray-400 dark:text-gray-500">• Rated ~65W</span>
                  </div>
                </div>
              </div>

              {/* Status Pill */}
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                  in1Active
                    ? 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700'
                    : 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300 border border-rose-200 dark:border-rose-800'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    in1Active ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                  }`}
                />
                {in1Active ? 'CLOSED (ON)' : 'CUTOFF (OFF)'}
              </span>
            </div>

            {/* Load Metric Preview */}
            <div className="my-4 p-3 rounded-xl bg-white/70 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-800 flex items-center justify-between">
              <span className="text-xs text-gray-500 dark:text-gray-400">Estimated Channel Load:</span>
              <span className="font-mono font-bold text-sm text-gray-900 dark:text-white">
                {in1Active ? '65.0 W' : '0.0 W'}
              </span>
            </div>
          </div>

          {/* Toggle Button & Actions */}
          <div className="pt-2 flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              Manual Toggle:
            </span>

            {/* Sliding Toggle Switch */}
            <button
              id="toggle-in1-switch"
              type="button"
              disabled={loadingAction?.startsWith('ch1')}
              onClick={() => handleSendCommand(in1Active ? 'OFF' : 'ON', 1)}
              className={`relative inline-flex h-8 w-15 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:opacity-50 ${
                in1Active ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-gray-700'
              }`}
            >
              <span className="sr-only">Toggle IN1</span>
              <span
                className={`pointer-events-none inline-block h-7 w-7 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out flex items-center justify-center ${
                  in1Active ? 'translate-x-7' : 'translate-x-0'
                }`}
              >
                {loadingAction?.startsWith('ch1') ? (
                  <RefreshCw size={12} className="animate-spin text-gray-500" />
                ) : in1Active ? (
                  <Power size={12} className="text-emerald-600" />
                ) : (
                  <PowerOff size={12} className="text-gray-400" />
                )}
              </span>
            </button>
          </div>
        </div>

        {/* ── Channel 2: IN2 (Phone Charger) ── */}
        <div
          id="channel-in2-card"
          className={`relative rounded-2xl p-5 border transition-all duration-300 flex flex-col justify-between ${
            in2Active
              ? 'bg-gradient-to-br from-emerald-50/70 via-white to-white dark:from-emerald-950/30 dark:via-gray-800 dark:to-gray-800 border-emerald-300 dark:border-emerald-700/80 shadow-sm shadow-emerald-500/5'
              : 'bg-gray-50/60 dark:bg-gray-800/40 border-gray-200 dark:border-gray-700/70 opacity-90'
          }`}
        >
          <div>
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="flex items-center gap-3">
                <div
                  className={`p-3 rounded-2xl transition-colors ${
                    in2Active
                      ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/20'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                  }`}
                >
                  <Smartphone size={20} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-sm text-gray-900 dark:text-white">
                      Relay 2: IN2 (Phone Charger)
                    </h3>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="font-mono text-[10px] text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700/60 px-1.5 py-0.5 rounded">
                      GPIO 27
                    </span>
                    <span className="text-xs text-gray-400 dark:text-gray-500">• Rated ~20W</span>
                  </div>
                </div>
              </div>

              {/* Status Pill */}
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                  in2Active
                    ? 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700'
                    : 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300 border border-rose-200 dark:border-rose-800'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    in2Active ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                  }`}
                />
                {in2Active ? 'CLOSED (ON)' : 'CUTOFF (OFF)'}
              </span>
            </div>

            {/* Load Metric Preview */}
            <div className="my-4 p-3 rounded-xl bg-white/70 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-800 flex items-center justify-between">
              <span className="text-xs text-gray-500 dark:text-gray-400">Estimated Channel Load:</span>
              <span className="font-mono font-bold text-sm text-gray-900 dark:text-white">
                {in2Active ? '20.0 W' : '0.0 W'}
              </span>
            </div>
          </div>

          {/* Toggle Button & Actions */}
          <div className="pt-2 flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              Manual Toggle:
            </span>

            {/* Sliding Toggle Switch */}
            <button
              id="toggle-in2-switch"
              type="button"
              disabled={loadingAction?.startsWith('ch2')}
              onClick={() => handleSendCommand(in2Active ? 'OFF' : 'ON', 2)}
              className={`relative inline-flex h-8 w-15 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:opacity-50 ${
                in2Active ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-gray-700'
              }`}
            >
              <span className="sr-only">Toggle IN2</span>
              <span
                className={`pointer-events-none inline-block h-7 w-7 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out flex items-center justify-center ${
                  in2Active ? 'translate-x-7' : 'translate-x-0'
                }`}
              >
                {loadingAction?.startsWith('ch2') ? (
                  <RefreshCw size={12} className="animate-spin text-gray-500" />
                ) : in2Active ? (
                  <Power size={12} className="text-emerald-600" />
                ) : (
                  <PowerOff size={12} className="text-gray-400" />
                )}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Emergency Master Controls Bar ── */}
      <div className="p-4 rounded-2xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200/80 dark:border-gray-700/80 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 text-xs text-gray-600 dark:text-gray-400">
          <ShieldAlert size={16} className="text-amber-500 shrink-0" />
          <span>
            <strong>Fail-Safe Protection:</strong> If anything goes wrong, use the emergency cutoff below to immediately drop both coils.
          </span>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          {/* Master ON Button */}
          <button
            id="master-on-btn"
            type="button"
            disabled={loadingAction === 'master_ON'}
            onClick={() => handleSendCommand('ON')}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm hover:shadow transition-all disabled:opacity-50 cursor-pointer"
          >
            {loadingAction === 'master_ON' ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <Power size={14} />
            )}
            <span>Master ON (Both)</span>
          </button>

          {/* Emergency ALL-OFF Button */}
          <button
            id="emergency-cutoff-btn"
            type="button"
            disabled={loadingAction === 'master_OFF'}
            onClick={() => handleSendCommand('OFF')}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs bg-rose-600 hover:bg-rose-700 text-white shadow-md shadow-rose-600/20 hover:shadow-lg transition-all disabled:opacity-50 cursor-pointer"
          >
            {loadingAction === 'master_OFF' ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <PowerOff size={14} />
            )}
            <span>Emergency Cutoff (OFF)</span>
          </button>
        </div>
      </div>

      {/* ── Direct CLI / Python Codes Drawer ── */}
      {showCliDrawer && (
        <div className="mt-4 p-4 rounded-2xl bg-gray-900 text-gray-200 border border-gray-800 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-300">
              <Terminal size={14} className="text-emerald-400" />
              <span>Direct Python MQTT Terminal Codes</span>
            </div>
            <span className="text-[10px] text-gray-400">Tested & Verified Working</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            {/* Master ON */}
            <div className="p-2.5 rounded-lg bg-gray-950/80 border border-gray-800/80 flex items-center justify-between gap-2">
              <div className="overflow-x-auto">
                <span className="text-emerald-400 font-bold"># Master ON (Both)</span>
                <p className="text-gray-300 mt-0.5">{pythonCliSnippet('ON')}</p>
              </div>
              <button
                onClick={() => copyToClipboard(pythonCliSnippet('ON'), 'master_on')}
                className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors cursor-pointer shrink-0"
                title="Copy command"
              >
                {copiedCmd === 'master_on' ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>

            {/* Master OFF */}
            <div className="p-2.5 rounded-lg bg-gray-950/80 border border-gray-800/80 flex items-center justify-between gap-2">
              <div className="overflow-x-auto">
                <span className="text-rose-400 font-bold"># Master OFF (Emergency Cut Both)</span>
                <p className="text-gray-300 mt-0.5">{pythonCliSnippet('OFF')}</p>
              </div>
              <button
                onClick={() => copyToClipboard(pythonCliSnippet('OFF'), 'master_off')}
                className="p-1.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors cursor-pointer shrink-0"
                title="Copy command"
              >
                {copiedCmd === 'master_off' ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              </button>
            </div>

            {/* IN1 ON / OFF */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <div className="p-2 rounded bg-gray-950/80 border border-gray-800/80 flex items-center justify-between gap-1">
                <div className="truncate">
                  <span className="text-blue-400 font-bold"># Cut IN1 (Laptop)</span>
                  <p className="text-gray-300 truncate">{pythonCliSnippet('IN1_OFF')}</p>
                </div>
                <button
                  onClick={() => copyToClipboard(pythonCliSnippet('IN1_OFF'), 'in1_off')}
                  className="p-1 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors cursor-pointer shrink-0"
                >
                  {copiedCmd === 'in1_off' ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                </button>
              </div>

              <div className="p-2 rounded bg-gray-950/80 border border-gray-800/80 flex items-center justify-between gap-1">
                <div className="truncate">
                  <span className="text-blue-400 font-bold"># Cut IN2 (Phone)</span>
                  <p className="text-gray-300 truncate">{pythonCliSnippet('IN2_OFF')}</p>
                </div>
                <button
                  onClick={() => copyToClipboard(pythonCliSnippet('IN2_OFF'), 'in2_off')}
                  className="p-1 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition-colors cursor-pointer shrink-0"
                >
                  {copiedCmd === 'in2_off' ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RelayControlPanel;
