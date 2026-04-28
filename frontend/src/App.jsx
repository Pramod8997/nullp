import React, { useState, useEffect, useCallback, useRef } from 'react';
import RealTimeChart from './components/RealTimeChart';
import SafetyAlerts from './components/SafetyAlerts';
import DigitalTwin from './components/DigitalTwin';
import DeviceCards from './components/DeviceCards';
import PhantomTracker from './components/PhantomTracker';
import SystemStatus from './components/SystemStatus';
import './App.css';

const WS_URL = 'ws://localhost:8000/ws';
const MAX_RECONNECT_DELAY = 10000;

function App() {
  const [devices, setDevices]             = useState({});
  const [powerHistory, setPowerHistory]     = useState([]);
  const [alerts, setAlerts]                 = useState([]);
  const [twinEvents, setTwinEvents]         = useState([]);
  const [phantomData, setPhantomData]       = useState({ loads: {}, total: 0 });
  const [pmvScore, setPmvScore]             = useState(0);
  const [analytics, setAnalytics]           = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [pipelineStatus, setPipelineStatus]     = useState('initializing');
  const [pendingUnknowns, setPendingUnknowns]   = useState([]);  // LABEL_REQUEST events

  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);

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
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

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

      default:
        break;
    }
  }, []);

  return (
    <div className="dashboard">
      {/* ── Header ── */}
      <header className="dashboard-header">
        <div className="dashboard-header__title">
          <div>
            <h1>⚡ EMS Digital Twin</h1>
            <span className="subtitle">Confidence-Aware Energy Management System</span>
          </div>
        </div>
        <div className={`status-badge ${connectionStatus === 'connected' ? 'online' : connectionStatus === 'reconnecting' ? 'reconnecting' : 'offline'}`}>
          <span className="status-dot" />
          {connectionStatus === 'connected' ? 'LIVE 1Hz Stream' : connectionStatus === 'reconnecting' ? 'Reconnecting...' : 'Disconnected'}
        </div>
      </header>

      {/* ── Dashboard Grid ── */}
      <div className="dashboard-grid">
        {/* Row 1: Device Status Cards */}
        <div className="panel panel-devices">
          <DeviceCards devices={devices} />
        </div>

        {/* Row 2: Chart + Safety Alerts */}
        <div className="panel panel-chart">
          <div className="panel-header">
            <h2>📊 Real-Time Power Monitor</h2>
            <span className="panel-badge">{Object.keys(devices).length} devices</span>
          </div>
          <RealTimeChart data={powerHistory} devices={devices} />
        </div>

        <div className="panel panel-safety">
          <div className="panel-header">
            <h2>🛡️ Safety Layer</h2>
            <span className="panel-badge">{alerts.length} events</span>
          </div>
          <SafetyAlerts alerts={alerts} />
        </div>

        {/* Row 3: Digital Twin + Phantom + System Status */}
        <div className="panel panel-twin">
          <DigitalTwin events={twinEvents} pmvScore={pmvScore} />
        </div>

        <div className="panel panel-phantom">
          <PhantomTracker data={phantomData} />
        </div>

        <div className="panel panel-status">
          <SystemStatus
            connectionStatus={connectionStatus}
            pipelineStatus={pipelineStatus}
            analytics={analytics}
            deviceCount={Object.keys(devices).length}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
