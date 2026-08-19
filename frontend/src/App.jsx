import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';
import Sidebar from './components/Sidebar/Sidebar';
import OverviewPage from './pages/OverviewPage/OverviewPage';
import AppliancesPage from './pages/AppliancesPage/AppliancesPage';
import AnalyticsPage from './pages/AnalyticsPage/AnalyticsPage';
import DigitalTwinPage from './pages/DigitalTwinPage/DigitalTwinPage';
import SchedulePage from './pages/SchedulePage/SchedulePage';
import AlertsPage from './pages/AlertsPage/AlertsPage';
import SettingsPage from './pages/SettingsPage/SettingsPage';
import { Sun, Moon, Zap } from 'lucide-react';

const WS_URL = `ws://${window.location.hostname}:8000/ws`;
const MAX_RECONNECT_DELAY = 10000;

function Dashboard() {
  const { toggleTheme, isDark } = useTheme();

  // ── Navigation State ──
  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ── Data State ──
  const [devices, setDevices] = useState({});
  const [powerHistory, setPowerHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [twinEvents, setTwinEvents] = useState([]);
  const [phantomData, setPhantomData] = useState({ loads: {}, total: 0 });
  const [pmvScore, setPmvScore] = useState(0);
  const [analytics, setAnalytics] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [pipelineStatus, setPipelineStatus] = useState('initializing');
  const [pendingUnknowns, setPendingUnknowns] = useState([]);
  const [latencyStats, setLatencyStats] = useState({ avg_ms: 0, max_ms: 0, p95_ms: 0 });
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [isArcFaultActive, setIsArcFaultActive] = useState(false);

  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);
  const arcFaultTimer = useRef(null);

  // ── Arc-Fault trigger helper ──
  const triggerArcFault = useCallback(() => {
    setIsArcFaultActive(true);
    if (arcFaultTimer.current) clearTimeout(arcFaultTimer.current);
    arcFaultTimer.current = setTimeout(() => setIsArcFaultActive(false), 4000);
  }, []);

  // ── Message Router ──
  const handleMessage = useCallback(
    (data) => {
      const { type } = data;

      switch (type) {
        case 'init_state':
          if (data.devices) setDevices(data.devices);
          if (data.pmv_score) setPmvScore(data.pmv_score);
          if (data.phantom_loads) setPhantomData((prev) => ({ ...prev, loads: data.phantom_loads }));
          if (data.pipeline_status) setPipelineStatus(data.pipeline_status);
          break;

        case 'heartbeat':
          setPipelineStatus(data.status || 'connected');
          break;

        case 'power_reading':
          setPowerHistory((prev) => {
            const next = [
              ...prev,
              {
                time: new Date().toLocaleTimeString('en-US', { hour12: false }),
                [data.device_id]: data.power,
                device_id: data.device_id,
              },
            ];
            return next.length > 120 ? next.slice(-120) : next;
          });
          setDevices((prev) => ({
            ...prev,
            [data.device_id]: {
              ...prev[data.device_id],
              power: data.power,
            },
          }));
          break;

        case 'power_batch': {
          const readings = data.readings || {};
          const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
          const entry = { time: ts };
          Object.entries(readings).forEach(([devId, watts]) => {
            entry[devId] = watts;
          });
          setPowerHistory((prev) => {
            const next = [...prev, entry];
            return next.length > 120 ? next.slice(-120) : next;
          });
          setDevices((prev) => {
            const updated = { ...prev };
            Object.entries(readings).forEach(([devId, watts]) => {
              updated[devId] = { ...updated[devId], power: watts };
            });
            return updated;
          });
          break;
        }

        case 'DEVICE_STATUS':
          setDevices((prev) => ({
            ...prev,
            [data.device_id]: {
              power: data.power,
              state: data.state,
              classification: data.classification,
              pmv: data.pmv,
              last_seen: data.timestamp,
            },
          }));
          if (data.pmv !== undefined) setPmvScore(data.pmv);
          break;

        case 'safety_alert':
        case 'SAFETY_CUTOFF':
          setAlerts((prev) =>
            [
              {
                id: Date.now() + Math.random(),
                severity: data.severity || 'critical',
                device_id: data.device_id || '',
                message: data.message,
                timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
              },
              ...prev,
            ].slice(0, 50)
          );
          if (
            data.message &&
            (data.message.includes('ARC') ||
              data.message.includes('RoC') ||
              data.message.includes('OVERCURRENT'))
          ) {
            triggerArcFault();
          }
          break;

        case 'SOFT_ANOMALY':
          setAlerts((prev) =>
            [
              {
                id: Date.now() + Math.random(),
                severity: 'warning',
                device_id: data.device_id,
                message: data.message,
                timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
              },
              ...prev,
            ].slice(0, 50)
          );
          break;

        case 'RL_ACTION':
        case 'EMPATHY_BLOCK':
        case 'EMPATHY_ACTION':
        case 'UNKNOWN_DEVICE':
          setTwinEvents((prev) => [data, ...prev].slice(0, 30));
          break;

        case 'LABEL_REQUEST':
          setTwinEvents((prev) => [data, ...prev].slice(0, 30));
          setPendingUnknowns((prev) => {
            const filtered = prev.filter((u) => u.device_id !== data.device_id);
            return [data, ...filtered].slice(0, 20);
          });
          break;

        case 'LOW_CONFIDENCE':
          setTwinEvents((prev) => [data, ...prev].slice(0, 30));
          break;

        case 'SAFETY_WARNING':
          setAlerts((prev) =>
            [
              {
                id: Date.now() + Math.random(),
                severity: 'warning',
                device_id: data.device_id || '',
                message: data.message,
                timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
              },
              ...prev,
            ].slice(0, 50)
          );
          if (data.message && (data.message.includes('ARC') || data.message.includes('RoC'))) {
            triggerArcFault();
          }
          break;

        case 'PHANTOM_LOAD':
          setPhantomData({
            loads: data.loads || {},
            total: data.total || 0,
            offenders: data.offenders || [],
          });
          break;

        case 'ANALYTICS_UPDATE':
          setAnalytics(data.summary || {});
          break;

        case 'PMV_UPDATE':
          setPmvScore(data.pmv || 0);
          break;

        case 'LATENCY_STATS':
          setLatencyStats({
            avg_ms: data.avg_ms || 0,
            max_ms: data.max_ms || 0,
            p95_ms: data.p95_ms || 0,
          });
          setLatencyHistory((prev) => {
            const next = [
              ...prev,
              {
                time: new Date().toLocaleTimeString('en-US', {
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                }),
                avg: data.avg_ms || 0,
                p95: data.p95_ms || 0,
                max: data.max_ms || 0,
              },
            ];
            return next.length > 20 ? next.slice(-20) : next;
          });
          break;

        default:
          break;
      }
    },
    [triggerArcFault]
  );

  // ── Connection forward declaration reference ──
  const connectRef = useRef(null);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = setTimeout(() => {
      console.log(`[EMS] Reconnecting in ${reconnectDelay.current}ms...`);
      if (connectRef.current) connectRef.current();
      reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, MAX_RECONNECT_DELAY);
    }, reconnectDelay.current);
  }, []);

  // ── WebSocket with auto-reconnect ──
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[EMS] WebSocket connected');
      setConnectionStatus('connected');
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (err) {
        console.error('[EMS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('[EMS] WebSocket disconnected');
      setConnectionStatus('reconnecting');
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [handleMessage, scheduleReconnect]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (arcFaultTimer.current) clearTimeout(arcFaultTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  // ── Page Renderer ──
  const renderPage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage devices={devices} powerHistory={powerHistory} />;

      case 'appliances':
      case 'devices':
        return <AppliancesPage devices={devices} />;

      case 'analytics':
        return (
          <AnalyticsPage
            devices={devices}
            powerHistory={powerHistory}
            connectionStatus={connectionStatus}
            pipelineStatus={pipelineStatus}
            analytics={analytics}
            latencyStats={latencyStats}
            latencyHistory={latencyHistory}
          />
        );

      case 'digital-twin':
        return (
          <DigitalTwinPage
            devices={devices}
            powerHistory={powerHistory}
            twinEvents={twinEvents}
            pmvScore={pmvScore}
            phantomData={phantomData}
            pendingUnknowns={pendingUnknowns}
          />
        );

      case 'schedule':
        return <SchedulePage />;

      case 'alerts':
        return <AlertsPage alerts={alerts} />;

      case 'settings':
        return <SettingsPage />;

      default:
        return <OverviewPage devices={devices} powerHistory={powerHistory} />;
    }
  };

  const getTabTitle = (tab) => {
    if (tab === 'appliances' || tab === 'devices') return 'Appliance Fleet & Sub-Metering';
    if (tab === 'digital-twin') return 'Digital Twin 3D Floor Plan';
    return tab.replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100 transition-colors duration-300 flex overflow-hidden font-sans antialiased selection:bg-blue-500 selection:text-white">
      {/* ── Arc-Fault Emergency Fullscreen Overlay ── */}
      {isArcFaultActive && (
        <div
          id="arc-fault-overlay"
          className="fixed inset-0 z-50 flex items-center justify-center bg-rose-950/80 backdrop-blur-md animate-pulse p-4"
        >
          <div className="bg-gradient-to-br from-rose-600 to-red-700 text-white rounded-3xl p-8 max-w-lg w-full shadow-2xl border-2 border-rose-400 flex flex-col items-center text-center gap-4">
            <div className="p-4 rounded-full bg-white/20 text-white animate-bounce">
              <Zap size={48} className="fill-current" />
            </div>
            <div>
              <h2 className="text-xl font-black uppercase tracking-wider">
                ARC FAULT DETECTED
              </h2>
              <p className="text-sm font-semibold text-rose-100 mt-1">
                HARDWARE RELAY CUTOFF AUTOMATICALLY TRIPPED
              </p>
            </div>
            <p className="text-xs text-rose-200 bg-rose-900/50 p-3 rounded-xl border border-rose-500/40">
              High-frequency arc signature isolated. Breakers open. Inspect appliance nodes immediately.
            </p>
          </div>
        </div>
      )}

      {/* ── Sidebar Navigation ── */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        alertCount={alerts.length}
      />

      {/* ── Main Content Area ── */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        {/* Header Bar */}
        <header className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200/80 dark:border-gray-800/80 h-20 shrink-0">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-gray-900 dark:text-white capitalize">
              {getTabTitle(activeTab)}
            </h1>
            <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              EMS Edge Controller • Zone 01
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Connection Status Badge */}
            <div
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border ${
                connectionStatus === 'connected'
                  ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
                  : connectionStatus === 'reconnecting'
                  ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border-amber-200 dark:border-amber-800'
                  : 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border-rose-200 dark:border-rose-800'
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  connectionStatus === 'connected'
                    ? 'bg-emerald-500 animate-ping'
                    : connectionStatus === 'reconnecting'
                    ? 'bg-amber-500 animate-pulse'
                    : 'bg-rose-500'
                }`}
              />
              <span className="hidden sm:inline">
                {connectionStatus === 'connected'
                  ? 'LIVE 1Hz Stream'
                  : connectionStatus === 'reconnecting'
                  ? 'Reconnecting...'
                  : 'Disconnected'}
              </span>
            </div>

            {/* Quick Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              title={`Switch to ${isDark ? 'Light' : 'Dark'} mode`}
              className="p-2.5 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700 cursor-pointer"
            >
              {isDark ? <Sun size={17} className="text-amber-400" /> : <Moon size={17} className="text-indigo-600" />}
            </button>
          </div>
        </header>

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 bg-gray-50/50 dark:bg-gray-950/50">
          <div className="max-w-7xl mx-auto pb-12">
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Dashboard />
    </ThemeProvider>
  );
}

export default App;
