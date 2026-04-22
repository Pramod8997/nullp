import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

const SafetyAlerts = ({ alerts }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '300px', color: '#10b981', gap: '1rem' }}>
        <ShieldCheck size={64} style={{ opacity: 0.8 }} />
        <p style={{ fontWeight: '600', fontSize: '1.1rem', margin: 0 }}>System Operating Normally</p>
        <p style={{ color: '#94a3b8', fontSize: '0.875rem', margin: 0 }}>No safety thresholds breached.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: '450px', overflowY: 'auto', paddingRight: '0.5rem' }}>
      {alerts.slice().reverse().map((alert) => (
        <div key={alert.id} style={{
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderLeft: '4px solid #ef4444',
          padding: '1.25rem',
          borderRadius: '0 0.5rem 0.5rem 0',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1rem',
          animation: 'slideIn 0.3s ease-out'
        }}>
          <AlertTriangle color="#ef4444" size={24} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <div style={{ color: '#ef4444', fontWeight: 'bold', fontSize: '1rem', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              CRITICAL CUTOFF
            </div>
            <div style={{ color: '#f8fafc', fontSize: '0.95rem', lineHeight: '1.4', marginBottom: '0.5rem' }}>
              {alert.message}
            </div>
            <div style={{ color: '#94a3b8', fontSize: '0.8rem', fontWeight: '500' }}>
              {alert.timestamp}
            </div>
          </div>
        </div>
      ))}
      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(10px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .alerts-section ::-webkit-scrollbar {
          width: 6px;
        }
        .alerts-section ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }
        .alerts-section ::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

export default SafetyAlerts;
