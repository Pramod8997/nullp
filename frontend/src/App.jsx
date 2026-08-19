import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar/Sidebar';
import OverviewPage from './pages/OverviewPage/OverviewPage';
import AppliancesPage from './pages/AppliancesPage/AppliancesPage';
import AnalyticsPage from './pages/AnalyticsPage/AnalyticsPage';
import DigitalTwinPage from './pages/DigitalTwinPage/DigitalTwinPage';
import SchedulePage from './pages/SchedulePage/SchedulePage';
import AlertsPage from './pages/AlertsPage/AlertsPage';
import SettingsPage from './pages/SettingsPage/SettingsPage';
import './App.css';

const WS_URL = `ws://${window.location.hostname}:8000/ws`;
const MAX_RECONNECT_DELAY = 10000;

function App() {
  // ── Navigation State ──
  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ── Data State ──
  const [devices, setDevices]             = useState({});
  const [powerHistory, setPowerHistory]     = useState([]);
  const [alerts, setAlerts]                 = useState([]);
  const [twinEvents, setTwinEvents]         = useState([]);
  const [phantomData, setPhantomData]       = useState({ loads: {}, total: 0 });
  const [pmvScore, setPmvScore]             = useState(0);
  const [analytics, setAnalytics]           = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [pipelineStatus, setPipelineStatus]     = useState('initializing');
  const [pendingUnknowns, setPendingUnknowns]   = useState([]);
  const [latencyStats, setLatencyStats]           = useState({ avg_ms: 0, max_ms: 0, p95_ms: 0 });
  const [latencyHistory, setLatencyHistory]       = useState([]);
  const [isArcFaultActive, setIsArcFaultActive]   = useState(false);

  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);
  const arcFaultTimer = useRef(null);

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
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = setTimeout(() => {
      console.log(`[EMS] Reconnecting in ${reconnectDelay.current}ms...`);
      connect();
      reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, MAX_RECONNECT_DELAY);
    }, reconnectDelay.current);
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (arcFaultTimer.current) clearTimeout(arcFaultTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  // ── Arc-Fault trigger helper ──
  const triggerArcFault = useCallback(() => {
    setIsArcFaultActive(true);
    if (arcFaultTimer.current) clearTimeout(arcFaultTimer.current);
    arcFaultTimer.current = setTimeout(() => setIsArcFaultActive(false), 4000);
  }, []);

  // ── Message Router ──
  const handleMessage = useCallback((data) => {
    const { type } = data;

    switch (type) {
      case 'init_state':
        if (data.devices) setDevices(data.devices);
        if (data.pmv_score) setPmvScore(data.pmv_score);
        if (data.phantom_loads) setPhantomData(prev => ({ ...prev, loads: data.phantom_loads }));
        if (data.pipeline_status) setPipelineStatus(data.pipeline_status);
        break;

      case 'heartbeat':
        setPipelineStatus(data.status || 'connected');
        break;

      case 'power_reading':
        setPowerHistory(prev => {
          const next = [...prev, {
            time: new Date().toLocaleTimeString('en-US', { hour12: false }),
            [data.device_id]: data.power,
            device_id: data.device_id,
          }];
          return next.length > 120 ? next.slice(-120) : next;
        });
        setDevices(prev => ({
          ...prev,
          [data.device_id]: {
            ...prev[data.device_id],
            power: data.power,
          },
        }));
        break;

      case 'power_batch': {
        // Batched power readings from WS aggregation (§4.3.3)
        const readings = data.readings || {};
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        const entry = { time: ts };
        Object.entries(readings).forEach(([devId, watts]) => {
          entry[devId] = watts;
        });
        setPowerHistory(prev => {
          const next = [...prev, entry];
          return next.length > 120 ? next.slice(-120) : next;
        });
        setDevices(prev => {
          const updated = { ...prev };
          Object.entries(readings).forEach(([devId, watts]) => {
            updated[devId] = { ...updated[devId], power: watts };
          });
          return updated;
        });
        break;
      }

      case 'DEVICE_STATUS':
        setDevices(prev => ({
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
        setAlerts(prev => [
          {
            id: Date.now() + Math.random(),
            severity: data.severity || 'critical',
            device_id: data.device_id || '',
            message: data.message,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          },
          ...prev,
        ].slice(0, 50));
        // Trigger arc-fault overlay for critical safety events
        if (data.message && (data.message.includes('ARC') || data.message.includes('RoC') || data.message.includes('OVERCURRENT'))) {
          triggerArcFault();
        }
        break;

      case 'SOFT_ANOMALY':
        setAlerts(prev => [
          {
            id: Date.now() + Math.random(),
            severity: 'warning',
            device_id: data.device_id,
            message: data.message,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          },
          ...prev,
        ].slice(0, 50));
        break;

      case 'RL_ACTION':
      case 'EMPATHY_BLOCK':
      case 'EMPATHY_ACTION':
      case 'UNKNOWN_DEVICE':
        setTwinEvents(prev => [data, ...prev].slice(0, 30));
        break;

      // ── Phase-1 new event types ──────────────────────────────────────────
      case 'LABEL_REQUEST':
        // Stable unknown device — needs user label (GAP 9)
        setTwinEvents(prev => [data, ...prev].slice(0, 30));
        setPendingUnknowns(prev => {
          // Dedupe by device_id — keep the latest
          const filtered = prev.filter(u => u.device_id !== data.device_id);
          return [data, ...filtered].slice(0, 20);
        });
        break;

      case 'LOW_CONFIDENCE':
        // Classification uncertain — show in twin event log
        setTwinEvents(prev => [data, ...prev].slice(0, 30));
        break;

      case 'SAFETY_WARNING':
        setAlerts(prev => [
          {
            id: Date.now() + Math.random(),
            severity: 'warning',
            device_id: data.device_id || '',
            message: data.message,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          },
          ...prev,
        ].slice(0, 50));
        // Trigger arc-fault overlay for Rate-of-Change warnings
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
        // Append to rolling latency history (last 20 data points = ~10 min at 30s intervals)
        setLatencyHistory(prev => {
          const next = [...prev, {
            time: new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            avg: data.avg_ms || 0,
            p95: data.p95_ms || 0,
            max: data.max_ms || 0,
          }];
          return next.length > 20 ? next.slice(-20) : next;
        });
        break;

      default:
        break;
    }
  }, [triggerArcFault]);

  // ── Page Renderer ──
  const renderPage = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewPage
            devices={devices}
            powerHistory={powerHistory}
          />
        );

      case 'appliances':
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
        return (
          <OverviewPage
            devices={devices}
            powerHistory={powerHistory}
          />
        );
    }
  };

  return (
    <div className={`app-layout ${isArcFaultActive ? 'arc-fault-active' : ''}`}>
      {/* Arc-Fault Emergency Overlay */}
      {isArcFaultActive && (
        <div className="arc-fault-overlay" id="arc-fault-overlay">
          <div className="arc-fault-overlay__inner">
            <span className="arc-fault-overlay__icon">⚡</span>
            <span className="arc-fault-overlay__text">ARC FAULT DETECTED — RELAY CUTOFF ACTIVE</span>
          </div>
        </div>
      )}

      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(prev => !prev)}
      />

      {/* Main Content Area */}
      <main
        className={`app-main ${sidebarCollapsed ? 'app-main--sidebar-collapsed' : ''}`}
      >
        {/* Header Bar */}
        <header className="app-header">
          <div className="app-header__title">
            <h1>{activeTab.replace('-', ' ').replace(/\b\w/g, c => c.toUpperCase())}</h1>
          </div>
          <div className={`status-badge ${connectionStatus === 'connected' ? 'online' : connectionStatus === 'reconnecting' ? 'reconnecting' : 'offline'}`}>
            <span className="status-dot" />
            {connectionStatus === 'connected' ? 'LIVE 1Hz Stream' : connectionStatus === 'reconnecting' ? 'Reconnecting...' : 'Disconnected'}
          </div>
        </header>

        {/* Page Content */}
        <div className="app-content">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}

export default App;
