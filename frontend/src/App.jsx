import React, { useState, useEffect } from 'react';
import RealTimeChart from './components/RealTimeChart';
import SafetyAlerts from './components/SafetyAlerts';
import DigitalTwin from './components/DigitalTwin';
import './App.css';

function App() {
  const [powerData, setPowerData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [twinEvents, setTwinEvents] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to FastAPI WebSocket endpoint
    // In production, this would use a dynamic hostname or environment variable
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('Connected to EMS WebSocket');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'power_reading') {
          setPowerData((prevData) => {
            const newData = [...prevData, { 
              time: new Date().toLocaleTimeString(), 
              power: data.power, 
              device_id: data.device_id 
            }];
            // Keep only the last 60 seconds (1Hz = 60 points per device approx)
            if (newData.length > 60) newData.shift();
            return newData;
          });
        } else if (data.type === 'safety_alert') {
          setAlerts((prevAlerts) => [
            ...prevAlerts,
            { id: Date.now(), message: data.message, timestamp: new Date().toLocaleTimeString() }
          ]);
        } else if (data.type === 'UNKNOWN_DEVICE' || data.type === 'RL_ACTION') {
          setTwinEvents((prev) => [...prev, data]);
        }
      } catch (err) {
        console.error("Failed to parse websocket message", err);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from EMS WebSocket');
      setIsConnected(false);
    };

    return () => ws.close();
  }, []);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>EMS Digital Twin Dashboard</h1>
        <div className={`status-indicator ${isConnected ? 'online' : 'offline'}`}>
          {isConnected ? 'LIVE 1Hz Stream' : 'Disconnected'}
        </div>
      </header>
      
      <main className="dashboard-main">
        <section className="chart-section" style={{ gridColumn: '1 / -1' }}>
          <h2>Active Power Ground Truth</h2>
          <RealTimeChart data={powerData} />
        </section>

        <section className="alerts-section">
          <h2>Safety Layer Monitoring</h2>
          <SafetyAlerts alerts={alerts} />
        </section>

        <section className="twin-section">
          <DigitalTwin events={twinEvents} />
        </section>
      </main>
    </div>
  );
}

export default App;
