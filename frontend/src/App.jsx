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
import { Sun, Moon, Wifi, ExternalLink } from 'lucide-react';
import { requestJson, WS_URL } from './api/client';

const MAX_RECONNECT_DELAY = 10000;
const MAX_POWER = 1000000;

const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const finiteNumber = (value) => {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
};

const getReceivedAt = () => Date.now();

const getSourceTimestamp = (value) => {
  const numeric = finiteNumber(value);
  if (numeric !== null && numeric > 0) return numeric > 1e12 ? numeric : numeric * 1000;
  if (typeof value === 'string') {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const normalizeDevice = (device, receivedAt, provenance = 'BACKEND_SNAPSHOT') => {
  if (!isRecord(device)) return null;
  const power = finiteNumber(device.power);
  const sourceTimestamp = getSourceTimestamp(device.source_timestamp ?? device.timestamp ?? device.last_seen, receivedAt);
  const ageMs = sourceTimestamp === null ? 0 : Math.max(0, receivedAt - sourceTimestamp);
  const isOffline = device.state === 'OFFLINE' || (sourceTimestamp !== null && ageMs > 8000);
  return {
    ...device,
    ...(power !== null ? { power: isOffline ? 0 : Math.max(0, Math.min(power, MAX_POWER)) } : {}),
    _receivedAt: receivedAt,
    source_timestamp: sourceTimestamp,
    provenance: device.provenance || device.source || provenance,
    state: isOffline ? 'OFFLINE' : (device.state || 'ON'),
    freshness: isOffline ? 'offline' : 'live',
  };
};

const normalizeDeviceMap = (devices, receivedAt, provenance) => {
  if (!isRecord(devices)) return {};
  return Object.entries(devices).reduce((result, [deviceId, device]) => {
    if (!deviceId) return result;
    const normalized = normalizeDevice(device, receivedAt, provenance);
    if (normalized) result[deviceId] = normalized;
    return result;
  }, {});
};

const asFiniteReadings = (readings) => {
  if (!isRecord(readings)) return {};
  return Object.entries(readings).reduce((result, [deviceId, watts]) => {
    const power = finiteNumber(watts);
    if (deviceId && power !== null && power >= 0 && power <= MAX_POWER) result[deviceId] = power;
    return result;
  }, {});
};

const offlineDevices = (devices) => Object.keys(devices || {}).reduce((result, deviceId) => {
  result[deviceId] = {
    ...devices[deviceId],
    power: 0,
    state: 'OFF',
    _receivedAt: Date.now(),
    source_timestamp: null,
    provenance: 'OFFLINE',
    freshness: 'offline',
  };
  return result;
}, {});

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
  const [relayStates, setRelayStates] = useState({});
  const wsRef = useRef(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef(null);
  const mountedRef = useRef(false);
  const connectionGeneration = useRef(0);
  const snapshotRequestRef = useRef(null);

  const syncSnapshot = useCallback(async (signal, generation) => {
    const responses = await Promise.allSettled([
      requestJson('/api/devices', { signal }),
      requestJson('/api/analytics', { signal }),
      requestJson('/api/phantom', { signal }),
      requestJson('/api/status', { signal }),
      requestJson('/api/safety-warnings', { signal }),
    ]);

    if (!mountedRef.current || signal.aborted || generation !== connectionGeneration.current) return;

    const [devicesResponse, analyticsResponse, phantomResponse, statusResponse, warningsResponse] = responses;
    const receivedAt = getReceivedAt();
    if (devicesResponse.status === 'fulfilled') {
      const body = devicesResponse.value;
      const devices = isRecord(body) ? body.devices : null;
      if (isRecord(devices)) setDevices(normalizeDeviceMap(devices, receivedAt, 'BACKEND_SNAPSHOT'));
    }
    if (analyticsResponse.status === 'fulfilled') {
      const body = analyticsResponse.value;
      const summary = isRecord(body) ? body.analytics : null;
      if (isRecord(summary)) setAnalytics(summary);
    }
    if (phantomResponse.status === 'fulfilled') {
      const body = phantomResponse.value;
      if (isRecord(body)) {
        setPhantomData((previous) => ({
          ...previous,
          loads: isRecord(body.phantom_loads) ? body.phantom_loads : previous.loads,
          total: finiteNumber(body.total_phantom_watts) ?? previous.total,
        }));
      }
    }
    if (statusResponse.status === 'fulfilled') {
      const body = statusResponse.value;
      if (isRecord(body) && typeof body.pipeline_status === 'string') {
        setPipelineStatus(body.pipeline_status);
      }
    }
    if (warningsResponse.status === 'fulfilled') {
      const body = warningsResponse.value;
      const warnings = isRecord(body) && Array.isArray(body.safety_warnings) ? body.safety_warnings : null;
      if (warnings) {
        setAlerts(warnings.slice(-50).reverse().map((warning, index) => ({
          ...warning,
          id: warning.id ?? `snapshot-${index}-${warning.timestamp ?? receivedAt}`,
          type: warning.type || 'SAFETY_WARNING',
          timestamp: warning.timestamp || receivedAt,
        })));
      }
    }
  }, []);

  const startSnapshotSync = useCallback((generation) => {
    snapshotRequestRef.current?.abort();
    const controller = new AbortController();
    snapshotRequestRef.current = controller;
    syncSnapshot(controller.signal, generation).catch((error) => {
      if (error?.name !== 'AbortError' && mountedRef.current) {
        console.warn('[EMS] Snapshot sync failed:', error);
      }
    });
  }, [syncSnapshot]);

  // ── Message Router ──
  const handleMessage = useCallback(
    (data) => {
      if (!isRecord(data) || typeof data.type !== 'string') return;
      const { type } = data;
      const receivedAt = getReceivedAt();

      switch (type) {
        case 'init_state': {
          const summary = data.analytics?.summary || data.analytics;
          if (isRecord(data.devices)) {
            setDevices(normalizeDeviceMap(data.devices, receivedAt, 'BACKEND_SNAPSHOT'));
          }
          const pmv = finiteNumber(data.pmv_score);
          if (pmv !== null) setPmvScore(pmv);
          if (Array.isArray(data.power_history)) setPowerHistory(data.power_history.slice(-120));
          if (isRecord(summary)) setAnalytics(summary);
          if (isRecord(data.telemetry)) setTelemetry(data.telemetry);
          if (isRecord(data.phantom_loads)) {
            setPhantomData((prev) => ({
              ...prev,
              loads: data.phantom_loads,
              total: finiteNumber(data.total_phantom_watts ?? data.phantom_total) ?? prev.total,
            }));
          }
          if (typeof data.pipeline_status === 'string') setPipelineStatus(data.pipeline_status);
          if (isRecord(data.relay_states)) setRelayStates(data.relay_states);
          break;
        }

        case 'heartbeat':
          if (typeof data.status === 'string') setPipelineStatus(data.status);
          break;

        case 'power_reading':
          if (!data.device_id) break;
          {
            const power = finiteNumber(data.power);
            if (power === null || power < 0 || power > MAX_POWER) break;
          setPowerHistory((prev) => {
            const next = [
              ...prev,
              {
                time: new Date(receivedAt).toLocaleTimeString('en-US', { hour12: false }),
                [data.device_id]: power,
                device_id: data.device_id,
                timestamp: receivedAt,
              },
            ];
            return next.length > 120 ? next.slice(-120) : next;
          });
          setDevices((prev) => ({
            ...prev,
            [data.device_id]: {
               ...prev[data.device_id],
               power,
               _receivedAt: receivedAt,
               source_timestamp: getSourceTimestamp(data.timestamp, receivedAt),
               provenance: data.provenance || 'LIVE_HARDWARE',
               freshness: 'live',
               state: prev[data.device_id]?.state === 'OFFLINE' ? 'ON' : (prev[data.device_id]?.state || 'ON'),
             },
           }));
          }
          break;

        case 'power_batch': {
          const readings = asFiniteReadings(data.readings);
          const sourceTimestamp = getSourceTimestamp(data.timestamp, receivedAt);
          const ts = new Date(sourceTimestamp || receivedAt).toLocaleTimeString('en-US', { hour12: false });
          const entry = { time: ts };
          Object.entries(readings).forEach(([devId, watts]) => {
            entry[devId] = watts;
          });
          entry.timestamp = sourceTimestamp || receivedAt;
          setPowerHistory((prev) => {
            const next = [...prev, entry];
            return next.length > 120 ? next.slice(-120) : next;
          });
          setDevices((prev) => {
            const updated = { ...prev };
            Object.entries(readings).forEach(([devId, watts]) => {
              updated[devId] = {
                ...updated[devId],
                power: watts,
                _receivedAt: receivedAt,
                source_timestamp: sourceTimestamp,
                provenance: data.provenance || 'LIVE_HARDWARE',
                freshness: 'live',
                state: updated[devId]?.state === 'OFFLINE' ? 'ON' : (updated[devId]?.state || 'ON'),
              };
            });
            return updated;
          });
          break;
        }

        case 'DEVICE_STATUS':
          if (!data.device_id) break;
          {
            const isDevOffline = data.state === 'OFFLINE';
            setDevices((prev) => ({
              ...prev,
              [data.device_id]: {
                ...prev[data.device_id],
                ...(finiteNumber(data.power) !== null ? { power: Math.max(0, finiteNumber(data.power)) } : {}),
                ...(typeof data.state === 'string' ? { state: data.state } : {}),
                ...(typeof data.classification === 'string' ? { classification: data.classification } : {}),
                ...(finiteNumber(data.confidence) !== null ? { confidence: finiteNumber(data.confidence) } : {}),
                ...(finiteNumber(data.pmv) !== null ? { pmv: finiteNumber(data.pmv) } : {}),
                last_seen: data.timestamp,
                _receivedAt: receivedAt,
                source_timestamp: getSourceTimestamp(data.timestamp, receivedAt),
                provenance: data.provenance || data.source || 'LIVE_HARDWARE',
                freshness: isDevOffline ? 'offline' : 'live',
                ...(data.label !== undefined ? { label: data.label } : {}),
                ...(data.rated !== undefined ? { rated: data.rated } : {}),
              },
            }));
            if (finiteNumber(data.pmv) !== null) setPmvScore(finiteNumber(data.pmv));
            if (isRecord(data.relay)) setRelayStates((prev) => ({ ...prev, [data.device_id]: data.relay }));
          }
          break;

        // Real PZEM electrical telemetry (voltage / current / power factor / power / relays)
        case 'TELEMETRY':
          if (data.device_id) {
            setTelemetry((prev) => ({
              ...prev,
              [data.device_id]: {
                v: finiteNumber(data.v),
                i: finiteNumber(data.i),
                pf: finiteNumber(data.pf),
                w: finiteNumber(data.w),
                relays: Array.isArray(data.relays) ? data.relays : prev[data.device_id]?.relays,
                ts: receivedAt,
                source_timestamp: getSourceTimestamp(data.timestamp, receivedAt),
                provenance: data.provenance || 'LIVE_HARDWARE',
              },
            }));
            setDevices((prev) => prev[data.device_id] ? ({
              ...prev,
              [data.device_id]: {
                ...prev[data.device_id],
                freshness: 'live',
                _receivedAt: receivedAt,
                ...(Array.isArray(data.relays) ? { relays: data.relays } : {}),
              },
            }) : prev);
          }
          break;

        case 'safety_alert':
        case 'SAFETY_CUTOFF': {
          setAlerts((prev) =>
            [
              {
                ...data,
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

        case 'HARDWARE_ACK': {
          if (!data.device_id || typeof data.message !== 'string') break;
          const ack = data.ack || data.command || data.message.match(/\b(ON_CONFIRMED|OFF_CONFIRMED|CH1_ON_CONFIRMED|CH1_OFF_CONFIRMED|CH2_ON_CONFIRMED|CH2_OFF_CONFIRMED|LOCKOUT_NACK)\b/i)?.[1] || '';
          const isCh1 = ack.includes('CH1') || ack.includes('IN1');
          const isCh2 = ack.includes('CH2') || ack.includes('IN2');
          const isConfirmed = ack.includes('CONFIRMED');
          const isOn = ack.includes('ON');
          setDevices((prev) => {
            const currentDev = prev[data.device_id] || {};
            const curRelays = Array.isArray(data.relays)
              ? [...data.relays]
              : [...(currentDev.relays || [0, 0])];
            if (isCh1) curRelays[0] = isOn ? 1 : 0;
            if (isCh2) curRelays[1] = isOn ? 1 : 0;
            if (ack === 'ON_CONFIRMED') { curRelays[0] = 1; curRelays[1] = 1; }
            if (ack === 'OFF_CONFIRMED') { curRelays[0] = 0; curRelays[1] = 0; }
            const anyOn = curRelays[0] === 1 || curRelays[1] === 1 || ack === 'ON_CONFIRMED';
            return {
              ...prev,
              [data.device_id]: {
                ...currentDev,
                state: anyOn ? 'ON' : 'OFF',
                relays: curRelays,
                confirmed: isConfirmed,
                ackTimestamp: receivedAt,
                provenance: 'EDGE_ACK',
              },
            };
          });
          setRelayStates((prev) => ({
            ...prev,
            [data.device_id]: {
              ...prev[data.device_id],
              ack: ack || 'UNKNOWN',
              state: ack.includes('ON') ? 'ON' : ack.includes('OFF') ? 'OFF' : prev[data.device_id]?.state,
              confirmed: isConfirmed,
              ackTimestamp: receivedAt,
              provenance: 'EDGE_ACK',
            },
          }));
          if (Array.isArray(data.relays) || isCh1 || isCh2 || isConfirmed) {
            setTelemetry((prev) => {
              const currentTelem = prev[data.device_id] || {};
              const curRelays = Array.isArray(data.relays)
                ? [...data.relays]
                : [...(currentTelem.relays || [0, 0])];
              if (isCh1) curRelays[0] = isOn ? 1 : 0;
              if (isCh2) curRelays[1] = isOn ? 1 : 0;
              if (ack === 'ON_CONFIRMED') { curRelays[0] = 1; curRelays[1] = 1; }
              if (ack === 'OFF_CONFIRMED') { curRelays[0] = 0; curRelays[1] = 0; }
              return {
                ...prev,
                [data.device_id]: {
                  ...currentTelem,
                  relays: curRelays,
                },
              };
            });
          }
          setAlerts((prev) => [{
            id: `ack-${data.device_id}-${receivedAt}`,
            type: 'HARDWARE_ACK',
            severity: ack === 'LOCKOUT_NACK' ? 'warning' : 'info',
            device_id: data.device_id,
            message: data.message,
            timestamp: receivedAt,
            provenance: 'EDGE_ACK',
          }, ...prev].slice(0, 50));
          break;
        }

        case 'SAFETY_EVENT': {
          if (!data.device_id) break;
          const isCutoff = /OFFLINE|FAULT|CUTOFF|LOCKED|OVERCURRENT|ARC_FAULT/i.test(data.message || '');
          const isOfflineMsg = /OFFLINE/i.test(data.message || '');
          const isOnlineMsg = /ONLINE/i.test(data.message || '');
          if (isOfflineMsg) {
            setDevices((prev) => prev[data.device_id] ? ({
              ...prev,
              [data.device_id]: {
                ...prev[data.device_id],
                power: 0,
                state: 'OFFLINE',
                freshness: 'offline',
                _receivedAt: receivedAt,
              },
            }) : prev);
          } else if (isOnlineMsg) {
            setDevices((prev) => prev[data.device_id] ? ({
              ...prev,
              [data.device_id]: {
                ...prev[data.device_id],
                state: 'ON',
                freshness: 'live',
                _receivedAt: receivedAt,
              },
            }) : prev);
          }
          setRelayStates((prev) => ({
            ...prev,
            [data.device_id]: {
              ...prev[data.device_id],
              state: isCutoff ? 'OFF' : prev[data.device_id]?.state,
              safety: data.message,
              safetyTimestamp: receivedAt,
              provenance: 'EDGE_STATUS',
            },
          }));
          if (isCutoff) {
            setAlerts((prev) => [
              {
                id: `safety-${data.device_id}-${receivedAt}`,
                type: 'SAFETY_EVENT',
                severity: 'critical',
                device_id: data.device_id,
                message: data.message,
                timestamp: new Date(receivedAt).toLocaleTimeString('en-US', { hour12: false }),
                provenance: 'EDGE_STATUS',
              },
              ...prev,
            ].slice(0, 50));
          }
          break;
        }

        case 'PHANTOM_LOAD':
          setPhantomData({
            loads: isRecord(data.loads) ? data.loads : {},
            total: finiteNumber(data.total) ?? 0,
            offenders: Array.isArray(data.offenders) ? data.offenders : [],
          });
          break;

        case 'ANALYTICS_UPDATE': {
          if (isRecord(data.summary)) setAnalytics((prev) => ({ ...prev, ...data.summary }));
          break;
        }

        case 'PMV_UPDATE':
          if (finiteNumber(data.pmv) !== null) setPmvScore(finiteNumber(data.pmv));
          break;

        case 'LATENCY_STATS':
          setLatencyStats({
            avg_ms: finiteNumber(data.avg_ms) ?? 0,
            max_ms: finiteNumber(data.max_ms) ?? 0,
            p95_ms: finiteNumber(data.p95_ms) ?? 0,
          });
          setLatencyHistory((prev) => {
            const next = [
              ...prev,
              {
                 time: new Date(receivedAt).toLocaleTimeString('en-US', {
                  hour12: false,
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                }),
                 avg: finiteNumber(data.avg_ms) ?? 0,
                 p95: finiteNumber(data.p95_ms) ?? 0,
                 max: finiteNumber(data.max_ms) ?? 0,
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
    if (!mountedRef.current) return;
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = setTimeout(() => {
      if (!mountedRef.current) return;
      console.log(`[EMS] Reconnecting in ${reconnectDelay.current}ms...`);
      if (connectRef.current) connectRef.current();
      reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, MAX_RECONNECT_DELAY);
    }, reconnectDelay.current);
  }, []);

  // ── WebSocket with auto-reconnect ──
  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    const generation = ++connectionGeneration.current;

    ws.onopen = () => {
      if (!mountedRef.current || wsRef.current !== ws || generation !== connectionGeneration.current) return;
      console.log('[EMS] WebSocket connected');
      setConnectionStatus('connected');
      reconnectDelay.current = 1000;
      startSnapshotSync(generation);
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current || wsRef.current !== ws || generation !== connectionGeneration.current) return;
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (err) {
        console.error('[EMS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws || generation !== connectionGeneration.current) return;
      console.log('[EMS] WebSocket disconnected');
      wsRef.current = null;
      if (mountedRef.current) {
        setConnectionStatus('reconnecting');
        setDevices((prev) => offlineDevices(prev));
        setTelemetry({});
      }
      scheduleReconnect();
    };

    ws.onerror = () => {
      if (!mountedRef.current || wsRef.current !== ws || generation !== connectionGeneration.current) return;
      ws.close();
    };
  }, [handleMessage, scheduleReconnect, startSnapshotSync]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      connectionGeneration.current += 1;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      snapshotRequestRef.current?.abort();
      if (wsRef.current) {
        const socket = wsRef.current;
        wsRef.current = null;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
    };
  }, [connect]);

  useEffect(() => {
    const freshnessTimer = setInterval(() => {
      const now = Date.now();
      setDevices((previous) => Object.entries(previous).reduce((next, [deviceId, device]) => {
        if (device?.freshness === 'offline' || device?.state === 'OFFLINE') {
          next[deviceId] = device;
          return next;
        }
        const age = now - (device?._receivedAt || 0);
        if (age <= 3500) {
          next[deviceId] = device;
          return next;
        }
        next[deviceId] = {
          ...device,
          power: 0,
          state: 'OFFLINE',
          freshness: 'offline',
          provenance: 'OFFLINE',
        };
        return next;
      }, {}));
    }, 1000);
    return () => clearInterval(freshnessTimer);
  }, []);

  // ── Page Renderer ──
  const renderPage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage devices={devices} powerHistory={powerHistory} telemetry={telemetry} analytics={analytics} relayStates={relayStates} />;

      case 'appliances':
      case 'devices':
        return <AppliancesPage devices={devices} telemetry={telemetry} relayStates={relayStates} />;

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
        return <AlertsPage alerts={alerts} relayStates={relayStates} />;

      case 'settings':
        return <SettingsPage />;

      default:
         return <OverviewPage devices={devices} powerHistory={powerHistory} telemetry={telemetry} analytics={analytics} relayStates={relayStates} />;
    }
  };

  const getTabTitle = (tab) => {
    if (tab === 'appliances' || tab === 'devices') return 'Appliance Fleet & Sub-Metering';
    if (tab === 'digital-twin') return 'Digital Twin 3D Floor Plan';
    return tab.replace('-', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-[#060a12] dark:text-gray-100 transition-colors duration-300 flex overflow-hidden font-sans antialiased selection:bg-blue-500 selection:text-white">
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
        <header className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 bg-white/90 dark:bg-[#070c16]/90 backdrop-blur-xl border-b border-gray-200/80 dark:border-[#172338] h-20 shrink-0">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-gray-900 dark:text-white capitalize">
              {getTabTitle(activeTab)}
            </h1>
            <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              EMS Edge Controller • Zone 01
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Live ESP32 Wi-Fi Hardware Node Link & Badge */}
            <a
              id="esp32-node-header-badge"
              href="http://172.20.10.10"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 dark:hover:bg-emerald-900/50 transition-colors shadow-sm cursor-pointer"
              title="Open ESP32 On-Chip Web Dashboard (172.20.10.10)"
            >
              <Wifi size={13} className="text-emerald-500 animate-pulse" />
              <span className="font-mono">ESP32: 172.20.10.10</span>
              <ExternalLink size={11} className="opacity-70" />
            </a>

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
              <span>
                {connectionStatus === 'reconnecting'
                  ? 'Reconnecting...'
                  : connectionStatus !== 'connected'
                  ? 'Disconnected'
                  : 'LIVE 1Hz Stream'}
              </span>
            </div>

            {/* Quick Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              title={`Switch to ${isDark ? 'Light' : 'Dark'} mode`}
              className="p-2.5 rounded-full bg-gray-100 dark:bg-[#101826] text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#182338] transition-colors border border-gray-200 dark:border-[#1e2c44] cursor-pointer shadow-sm"
            >
              {isDark ? <Sun size={17} className="text-amber-400" /> : <Moon size={17} className="text-indigo-600" />}
            </button>
          </div>
        </header>

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 bg-gray-50/50 dark:bg-[#060a12]">
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
