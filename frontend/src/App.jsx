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
import { Sun, Moon } from 'lucide-react';

const WS_URL = `ws://${window.location.hostname}:8000/ws`;
const MAX_RECONNECT_DELAY = 10000;

function replayMetadata(data) {
  const metadata = Object.fromEntries(
    ['source', 'generated', 'presentation_only', 'recording_replay', 'pzem_connected', 'replay', 'demo']
      .filter((key) => data[key] !== undefined)
      .map((key) => [key, data[key]])
  );
  if (data.pipeline_status === 'SAMPLE_DATA_REPLAY' || data.status === 'SAMPLE_DATA_REPLAY' ||
      metadata.source === 'generated_sample_replay' || metadata.generated === true || metadata.replay?.generated === true) {
    Object.assign(metadata, { generated: true, presentation_only: true, recording_replay: false, pzem_connected: false });
  } else if (data.pipeline_status === 'RECORDED_DATA_REPLAY' || data.status === 'RECORDED_DATA_REPLAY') {
    Object.assign(metadata, { recording_replay: true, presentation_only: true });
  }
  return metadata;
}

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
  const [latencyStats, setLatencyStats] = useState({ avg_ms: 0, max_ms: 0, p95_ms: 0 });
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [telemetry, setTelemetry] = useState({});
  const [sessionMetadata, setSessionMetadata] = useState({});

  const sessionMetadataRef = useRef({});
  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);
  // ── Message Router ──
  const handleMessage = useCallback(
    (data) => {
      const { type } = data;
      const metadata = replayMetadata({ ...sessionMetadataRef.current, ...replayMetadata(data) });

      switch (type) {
        case 'init_state': {
          const summary = data.analytics?.summary || data.analytics;
          const initialMetadata = replayMetadata({ ...summary, ...data });
          sessionMetadataRef.current = initialMetadata;
          setSessionMetadata(initialMetadata);
          if (data.devices) {
            setDevices(Object.fromEntries(Object.entries(data.devices).map(([id, device]) => [
              id, { ...initialMetadata, ...device, ...replayMetadata({ ...initialMetadata, ...device }) },
            ])));
          }
          if (data.pmv_score !== undefined) setPmvScore(data.pmv_score);
          const presentationMode = initialMetadata.presentation_only || ['USB_PRESENTATION_MODE', 'RECORDED_DATA_REPLAY', 'SAMPLE_DATA_REPLAY'].includes(data.pipeline_status);
          if (Array.isArray(data.power_history)) setPowerHistory(data.power_history.slice(-120));
          else if (presentationMode) setPowerHistory([]);
          if (summary) setAnalytics({ ...summary, ...initialMetadata });
          else if (presentationMode) setAnalytics({ ...initialMetadata });
          if (data.telemetry) setTelemetry(data.telemetry);
          else if (presentationMode) setTelemetry({});
          if (data.phantom_loads) setPhantomData((prev) => ({ ...prev, loads: data.phantom_loads }));
          else if (presentationMode) setPhantomData({ loads: {}, total: 0 });
          if (data.pipeline_status) setPipelineStatus(data.pipeline_status);
          break;
        }

        case 'heartbeat':
          setPipelineStatus(data.status || 'connected');
          sessionMetadataRef.current = metadata;
          setSessionMetadata(metadata);
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
              ...metadata,
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
              updated[devId] = { ...updated[devId], ...metadata, power: watts };
            });
            return updated;
          });
          break;
        }

        case 'DEVICE_STATUS':
          setDevices((prev) => ({
            ...prev,
            [data.device_id]: {
              ...prev[data.device_id],
              ...metadata,
              power: data.power,
              state: data.state,
              classification: data.classification,
              confidence: data.confidence,
              pmv: data.pmv,
              last_seen: data.timestamp,
              ...(data.label !== undefined ? { label: data.label } : {}),
              ...(data.rated !== undefined ? { rated: data.rated } : {}),
            },
          }));
          if (data.pmv !== undefined) setPmvScore(data.pmv);
          break;

        // Real PZEM electrical telemetry (voltage / current / power factor)
        case 'TELEMETRY':
          if (data.device_id) {
            setTelemetry((prev) => ({
              ...prev,
              [data.device_id]: { v: data.v, i: data.i, pf: data.pf, ts: Date.now() },
            }));
          }
          break;

        case 'safety_alert':
        case 'SAFETY_CUTOFF': {
          setAlerts((prev) =>
            [
              {
                ...data,
                ...metadata,
                id: Date.now() + Math.random(),
                type: data.type,
                severity: data.severity || 'critical',
                device_id: data.device_id || '',
                message: data.message,
                timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
              },
              ...prev,
            ].slice(0, 50)
          );
          break;
        }

        case 'SOFT_ANOMALY':
          setAlerts((prev) =>
            [
              {
                ...data,
                ...metadata,
                id: Date.now() + Math.random(),
                type: data.type,
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
          break;

        case 'LOW_CONFIDENCE':
          setTwinEvents((prev) => [data, ...prev].slice(0, 30));
          break;

        case 'SAFETY_WARNING': {
          setAlerts((prev) =>
            [
              {
                ...data,
                ...metadata,
                id: Date.now() + Math.random(),
                type: data.type,
                severity: 'warning',
                device_id: data.device_id || '',
                message: data.message,
                timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
              },
              ...prev,
            ].slice(0, 50)
          );
          break;
        }

        case 'PHANTOM_LOAD':
          setPhantomData({
            loads: data.loads || {},
            total: data.total || 0,
            offenders: data.offenders || [],
          });
          break;

        case 'ANALYTICS_UPDATE': {
          const updatedMetadata = replayMetadata({ ...metadata, ...replayMetadata(data.summary || {}) });
          sessionMetadataRef.current = updatedMetadata;
          setSessionMetadata(updatedMetadata);
          setAnalytics((prev) => ({ ...prev, ...data.summary, ...updatedMetadata }));
          break;
        }

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
    []
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
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  // ── Page Renderer ──
  const renderPage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage devices={devices} powerHistory={powerHistory} telemetry={telemetry} analytics={analytics} />;

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
            telemetry={telemetry}
          />
        );

      case 'schedule':
        return <SchedulePage />;

      case 'alerts':
        return <AlertsPage alerts={alerts} replayContext={sessionMetadata} />;

      case 'settings':
        return <SettingsPage />;

      default:
         return <OverviewPage devices={devices} powerHistory={powerHistory} telemetry={telemetry} analytics={analytics} />;
    }
  };

  const getTabTitle = (tab) => {
    if (tab === 'appliances' || tab === 'devices') return 'Appliance Fleet & Sub-Metering';
    if (tab === 'digital-twin') return 'Digital Twin 3D Floor Plan';
    return tab.replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const isSampleReplay = pipelineStatus === 'SAMPLE_DATA_REPLAY' || sessionMetadata.generated === true;
  const isReplayMode = !isSampleReplay && (pipelineStatus === 'RECORDED_DATA_REPLAY' || sessionMetadata.recording_replay === true);
  const isUsbCollectionMode = pipelineStatus === 'USB_COLLECTION_MODE';
  const isPresentationMode = pipelineStatus === 'USB_PRESENTATION_MODE' || isReplayMode || isSampleReplay || isUsbCollectionMode;
  const replay = sessionMetadata.replay;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100 transition-colors duration-300 flex overflow-hidden font-sans antialiased selection:bg-blue-500 selection:text-white">
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
                connectionStatus === 'connected' && isPresentationMode
                  ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border-blue-200 dark:border-blue-800'
                  : connectionStatus === 'connected'
                  ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800'
                  : connectionStatus === 'reconnecting'
                  ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border-amber-200 dark:border-amber-800'
                  : 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border-rose-200 dark:border-rose-800'
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  connectionStatus === 'connected' && isPresentationMode
                    ? 'bg-blue-500 animate-pulse'
                    : connectionStatus === 'connected'
                    ? 'bg-emerald-500 animate-ping'
                    : connectionStatus === 'reconnecting'
                    ? 'bg-amber-500 animate-pulse'
                    : 'bg-rose-500'
                }`}
              />
              <span>
                {connectionStatus === 'reconnecting'
                  ? 'Reconnecting...'
                  : connectionStatus !== 'connected'
                  ? 'Disconnected'
                  : isSampleReplay
                  ? 'Sample-data replay'
                  : isReplayMode
                  ? 'RECORDED DATA REPLAY'
                  : isUsbCollectionMode
                  ? 'USB COLLECTION MODE'
                  : isPresentationMode
                  ? 'USB PRESENTATION MODE'
                  : 'LIVE 1Hz Stream'}
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

        {(isSampleReplay || isReplayMode) && (
          <section aria-label="Replay source" className="shrink-0 border-b border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950/40 px-4 sm:px-6 py-2 text-blue-900 dark:text-blue-200">
            <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-sm">
              <strong>{isSampleReplay ? 'Sample-data replay' : 'Recorded data replay'}</strong>
              {replay?.cycle != null && replay?.total_cycles != null && (
                <span className="font-mono text-xs">Replay {replay.cycle} / {replay.total_cycles}</span>
              )}
              {replay?.sample_index != null && replay?.total_samples != null && (
                <span className="font-mono text-xs">Sample {replay.sample_index} / {replay.total_samples}</span>
              )}
              {connectionStatus !== 'connected' && (
                <span className="text-xs font-semibold text-amber-700 dark:text-amber-300">
                  {connectionStatus === 'reconnecting' ? 'Reconnecting — showing last received data' : 'Disconnected — showing last received data'}
                </span>
              )}
            </div>
            <p className="text-xs mt-0.5">
              {isSampleReplay
                ? 'Source: generated samples · No hardware recording · No relay action'
                : 'Source: hardware recording · Recorded data replay · No relay action'}
            </p>
            {isSampleReplay && Number.isFinite(analytics.playback_speed) && (
              <p className="text-xs mt-0.5">
                {analytics.playback_speed}× scenario time · Energy and cost cover this three-cycle group and reset for the next group.
              </p>
            )}
          </section>
        )}

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
