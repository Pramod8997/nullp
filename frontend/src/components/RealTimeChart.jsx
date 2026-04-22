import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ backgroundColor: '#0f172a', padding: '10px', border: '1px solid #334155', borderRadius: '5px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)' }}>
        <p style={{ color: '#94a3b8', margin: '0 0 5px 0', fontSize: '0.875rem' }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color || '#3b82f6', fontWeight: 'bold', margin: '0', fontSize: '1rem' }}>
            {entry.payload.device_id ? `${entry.payload.device_id}: ` : ''}{entry.value.toFixed(2)} W
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const RealTimeChart = ({ data }) => {
  return (
    <div style={{ width: '100%', height: '450px' }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis 
            dataKey="time" 
            stroke="#94a3b8" 
            tick={{ fill: '#94a3b8', fontSize: 12 }} 
            tickMargin={10} 
            minTickGap={30}
          />
          <YAxis 
            stroke="#94a3b8" 
            tick={{ fill: '#94a3b8', fontSize: 12 }} 
            unit="W" 
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
          <Line 
            type="monotone" 
            dataKey="power" 
            stroke="#3b82f6" 
            strokeWidth={3} 
            dot={false} 
            activeDot={{ r: 6, fill: '#3b82f6', stroke: '#0f172a', strokeWidth: 2 }}
            isAnimationActive={false} // Disabled animation for real-time 1Hz updates to prevent jitter
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RealTimeChart;
