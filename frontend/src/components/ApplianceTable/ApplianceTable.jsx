import React from 'react';
import './ApplianceTable.css';

const getConfidenceInfo = (confidence) => {
  if (confidence === undefined || confidence === null) return { label: '—', cls: 'none' };
  const pct = Math.round(confidence * 100);
  if (pct >= 80) return { label: `${pct}%`, cls: 'high' };
  if (pct >= 40) return { label: `${pct}%`, cls: 'medium' };
  return { label: `${pct}%`, cls: 'low' };
};

const formatDeviceName = (id) => {
  return id
    .replace('esp32_', '')
    .replace('node_', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const ApplianceTable = ({ devices = {} }) => {
  const deviceEntries = Object.entries(devices);

  if (deviceEntries.length === 0) {
    return (
      <div className="appliance-table-panel">
        <div className="appliance-table__header">
          <h3>REAL-TIME APPLIANCE STATUS</h3>
        </div>
        <div className="appliance-table__empty">
          <p>No appliances detected</p>
        </div>
      </div>
    );
  }

  return (
    <div className="appliance-table-panel">
      <div className="appliance-table__header">
        <h3>REAL-TIME APPLIANCE STATUS</h3>
      </div>
      <div className="appliance-table__wrapper">
        <table className="appliance-table">
          <thead>
            <tr>
              <th>Appliance</th>
              <th>Status</th>
              <th>Power (W)</th>
              <th>Energy (kWh)</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {deviceEntries.map(([id, dev]) => {
              const power = dev?.power || 0;
              const state = dev?.state || (power > 10 ? 'ON' : 'OFF');
              const isOn = state === 'ON';
              const energy = (power / 1000) * 0.5; // simplified kWh estimate
              const confidence = dev?.confidence;
              const confInfo = getConfidenceInfo(confidence);

              return (
                <tr key={id}>
                  <td className="appliance-table__name">
                    {dev?.label || formatDeviceName(id)}
                  </td>
                  <td>
                    <span className={`appliance-table__status ${isOn ? 'on' : 'off'}`}>
                      {state}
                    </span>
                  </td>
                  <td className="appliance-table__power">
                    {isOn ? power.toFixed(0) : '0'}
                  </td>
                  <td className="appliance-table__energy">
                    {isOn ? energy.toFixed(2) : '0.00'}
                  </td>
                  <td>
                    <span className={`appliance-table__confidence ${confInfo.cls}`}>
                      {confInfo.label}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="appliance-table__legend">
        <span className="appliance-table__legend-item">
          <span className="legend-dot legend-dot--high" /> High Confidence ({'>'} 80%)
        </span>
        <span className="appliance-table__legend-item">
          <span className="legend-dot legend-dot--medium" /> Low Confidence (40% - 80%)
        </span>
        <span className="appliance-table__legend-item">
          <span className="legend-dot legend-dot--low" /> Unknown ({'<'} 40%)
        </span>
      </div>
    </div>
  );
};

export default ApplianceTable;
