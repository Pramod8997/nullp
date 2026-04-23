import React from 'react';
import { Cpu, Power, PowerOff } from 'lucide-react';

const formatDeviceName = (id) => {
  return id.replace('esp32_', '').replace(/_/g, ' ').toUpperCase();
};

const DeviceCards = ({ devices }) => {
  const deviceIds = Object.keys(devices);

  if (deviceIds.length === 0) {
    return (
      <div>
        <div className="panel-header">
          <h2><Cpu size={18} color="#3b82f6" /> Device Fleet</h2>
          <span className="panel-badge">Waiting...</span>
        </div>
        <div className="device-cards">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="device-card" style={{ opacity: 0.4 }}>
              <div className="device-card__name">—</div>
              <div className="device-card__power">— <span className="unit">W</span></div>
              <div className="device-card__meta">
                <span className="device-card__state off">OFFLINE</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="panel-header">
        <h2><Cpu size={18} color="#3b82f6" /> Device Fleet</h2>
        <span className="panel-badge">{deviceIds.length} active</span>
      </div>
      <div className="device-cards">
        {deviceIds.map(id => {
          const dev = devices[id];
          const power = dev?.power ?? 0;
          const state = dev?.state || (power > 10 ? 'ON' : 'OFF');
          const classification = dev?.classification || 'pending';
          const isOn = state === 'ON';

          return (
            <div key={id} className={`device-card ${isOn ? 'state-on' : 'state-off'}`}>
              <div className="device-card__name">
                {isOn ? <Power size={12} style={{ marginRight: 4, color: '#10b981' }} /> : <PowerOff size={12} style={{ marginRight: 4, color: '#64748b' }} />}
                {formatDeviceName(id)}
              </div>
              <div className="device-card__power">
                {power.toFixed(1)} <span className="unit">W</span>
              </div>
              <div className="device-card__meta">
                <span className={`device-card__state ${isOn ? 'on' : 'off'}`}>
                  {state}
                </span>
                <span className="device-card__class">
                  {classification.startsWith('known:') ? `✓ ${classification.split(':')[1]}` : classification === 'unknown' ? '? Unknown' : classification}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DeviceCards;
