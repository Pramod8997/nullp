import React from 'react';
import './DigitalTwinVisualization.css';

const formatDeviceName = (id) => {
  return id
    .replace('esp32_', '')
    .replace('node_', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const DigitalTwinVisualization = ({ devices = {} }) => {
  const deviceList = Object.entries(devices);

  // Position devices around the house blueprint
  const devicePositions = [
    { top: '12%', right: '12%', side: 'right' },
    { top: '12%', left: '12%', side: 'left' },
    { top: '45%', right: '5%', side: 'right' },
    { bottom: '15%', left: '10%', side: 'left' },
    { bottom: '15%', right: '15%', side: 'right' },
    { top: '45%', left: '5%', side: 'left' },
    { bottom: '35%', left: '35%', side: 'left' },
    { top: '30%', right: '35%', side: 'right' },
  ];

  return (
    <div className="dt-viz">
      <h2 className="dt-viz__title">DIGITAL TWIN VISUALIZATION</h2>

      <div className="dt-viz__canvas">
        {/* House Blueprint SVG */}
        <svg className="dt-viz__blueprint" viewBox="0 0 500 400" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Outer walls */}
          <rect x="80" y="60" width="340" height="280" rx="4" stroke="#1e3a5f" strokeWidth="2" fill="none" />

          {/* Inner rooms */}
          {/* Living Room (top-left) */}
          <rect x="80" y="60" width="180" height="150" stroke="#1e3a5f" strokeWidth="1.5" fill="rgba(14, 30, 60, 0.4)" />

          {/* Kitchen (top-right) */}
          <rect x="260" y="60" width="160" height="150" stroke="#1e3a5f" strokeWidth="1.5" fill="rgba(14, 30, 60, 0.4)" />

          {/* Bedroom (bottom-left) */}
          <rect x="80" y="210" width="200" height="130" stroke="#1e3a5f" strokeWidth="1.5" fill="rgba(14, 30, 60, 0.4)" />

          {/* Bathroom (bottom-right) */}
          <rect x="280" y="210" width="140" height="130" stroke="#1e3a5f" strokeWidth="1.5" fill="rgba(14, 30, 60, 0.4)" />

          {/* Hallway */}
          <line x1="260" y1="170" x2="260" y2="210" stroke="#1e3a5f" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="280" y1="170" x2="280" y2="210" stroke="#1e3a5f" strokeWidth="1" strokeDasharray="4 4" />

          {/* Doors */}
          <line x1="150" y1="210" x2="190" y2="210" stroke="#3b82f6" strokeWidth="2" />
          <line x1="310" y1="210" x2="350" y2="210" stroke="#3b82f6" strokeWidth="2" />
          <line x1="260" y1="110" x2="260" y2="150" stroke="#3b82f6" strokeWidth="2" />

          {/* Room Labels */}
          <text x="145" y="140" fill="#334155" fontSize="11" fontWeight="600" textAnchor="middle">Living Room</text>
          <text x="340" y="140" fill="#334155" fontSize="11" fontWeight="600" textAnchor="middle">Kitchen</text>
          <text x="165" y="280" fill="#334155" fontSize="11" fontWeight="600" textAnchor="middle">Bedroom</text>
          <text x="350" y="280" fill="#334155" fontSize="11" fontWeight="600" textAnchor="middle">Bathroom</text>
        </svg>

        {/* Device Labels positioned around the blueprint */}
        {deviceList.map(([id, dev], index) => {
          const power = dev?.power || 0;
          const isOn = (dev?.state || (power > 10 ? 'ON' : 'OFF')) === 'ON';
          const pos = devicePositions[index % devicePositions.length];

          return (
            <div
              key={id}
              className={`dt-viz__device ${isOn ? 'dt-viz__device--active' : ''}`}
              style={{
                position: 'absolute',
                ...pos,
              }}
            >
              <span className="dt-viz__device-name">
                {dev?.label || formatDeviceName(id)}
              </span>
              <span className="dt-viz__device-power">
                {isOn ? `${power.toFixed(0)} W` : '0 W'}
              </span>
            </div>
          );
        })}

        {/* Glow lines connecting devices to house (decorative) */}
        <div className="dt-viz__grid-overlay" />
      </div>
    </div>
  );
};

export default DigitalTwinVisualization;
