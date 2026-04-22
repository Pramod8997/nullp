import React from 'react';
import { BrainCircuit, HelpCircle, Zap } from 'lucide-react';

const DigitalTwin = ({ events }) => {
  const handleAcknowledge = (event) => {
    alert(`Labeling flow for ${event.device_id} initiated. (Database update pending)`);
  };

  return (
    <div style={{ backgroundColor: '#1e293b', padding: '1.5rem', borderRadius: '1rem', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
      <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: 0 }}>
        <BrainCircuit size={24} color="#a78bfa" /> Digital Twin Brain
      </h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxHeight: '400px', overflowY: 'auto', paddingRight: '0.5rem' }}>
        {events.length === 0 ? (
          <div style={{ color: '#94a3b8', textAlign: 'center', padding: '2rem' }}>
            Agent Monitoring Nominal. No anomalies detected.
          </div>
        ) : (
          events.slice().reverse().map((event, i) => (
            <div key={i} style={{ padding: '1rem', borderRadius: '0.5rem', backgroundColor: 'rgba(15, 23, 42, 0.5)', animation: 'slideIn 0.3s ease-out' }}>
              
              {event.type === 'UNKNOWN_DEVICE' && (
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                  <HelpCircle color="#eab308" size={24} style={{ flexShrink: 0 }} />
                  <div>
                    <strong style={{ color: '#eab308', display: 'block', marginBottom: '0.25rem' }}>
                      New Signature ({event.device_id})
                    </strong>
                    <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.9rem', color: '#f8fafc' }}>
                      ProtoNet confidence low for {event.power}W draw. What device is this?
                    </p>
                    <button 
                      onClick={() => handleAcknowledge(event)}
                      style={{ background: '#eab308', color: '#0f172a', border: 'none', padding: '0.25rem 0.75rem', borderRadius: '0.25rem', fontWeight: 'bold', cursor: 'pointer' }}
                    >
                      Label Device
                    </button>
                  </div>
                </div>
              )}
              
              {event.type === 'RL_ACTION' && (
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                  <Zap color="#a78bfa" size={24} style={{ flexShrink: 0 }} />
                  <div>
                    <strong style={{ color: '#a78bfa', display: 'block', marginBottom: '0.25rem' }}>Policy Optimization</strong>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: '#f8fafc' }}>{event.message}</p>
                  </div>
                </div>
              )}

            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default DigitalTwin;
