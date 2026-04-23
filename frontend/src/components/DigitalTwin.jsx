import React from 'react';
import { BrainCircuit, Zap, HelpCircle, ShieldOff, Thermometer } from 'lucide-react';

const getPmvLabel = (pmv) => {
  if (pmv <= -2) return { label: 'Cold', cls: 'cold' };
  if (pmv <= -1) return { label: 'Cool', cls: 'cold' };
  if (pmv <= 1)  return { label: 'Comfortable', cls: 'comfort' };
  if (pmv <= 2)  return { label: 'Warm', cls: 'warm' };
  return { label: 'Hot', cls: 'hot' };
};

const DigitalTwin = ({ events, pmvScore }) => {
  const pmv = getPmvLabel(pmvScore);

  return (
    <div>
      <div className="panel-header">
        <h2><BrainCircuit size={18} color="#a78bfa" /> Digital Twin</h2>
      </div>

      {/* PMV Gauge */}
      <div className="pmv-gauge">
        <Thermometer size={24} color="#8b5cf6" />
        <div className={`pmv-gauge__value ${pmv.cls}`}>{pmvScore.toFixed(2)}</div>
        <div className="pmv-gauge__info">
          <span className="pmv-gauge__label">PMV Index</span>
          <span className="pmv-gauge__desc">{pmv.label}</span>
        </div>
      </div>

      {/* Event Log */}
      <div className="event-log">
        {events.length === 0 ? (
          <div className="empty-state" style={{ minHeight: 120 }}>
            <BrainCircuit size={32} color="#64748b" />
            <p>Agent monitoring nominal</p>
          </div>
        ) : (
          events.slice(0, 15).map((event, i) => {
            if (event.type === 'RL_ACTION') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(139,92,246,0.15)' }}>
                    <Zap size={16} color="#a78bfa" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#a78bfa' }}>RL Optimization</div>
                    <div className="event-item__desc">{event.message}</div>
                  </div>
                </div>
              );
            }
            if (event.type === 'EMPATHY_BLOCK') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(239,68,68,0.15)' }}>
                    <ShieldOff size={16} color="#ef4444" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#ef4444' }}>Empathy Gate</div>
                    <div className="event-item__desc">{event.message}</div>
                  </div>
                </div>
              );
            }
            if (event.type === 'UNKNOWN_DEVICE') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(245,158,11,0.15)' }}>
                    <HelpCircle size={16} color="#f59e0b" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#f59e0b' }}>Unknown Signature</div>
                    <div className="event-item__desc">{event.message}</div>
                    {event.stable && (
                      <button style={{
                        marginTop: '0.4rem',
                        background: '#f59e0b',
                        color: '#0f172a',
                        border: 'none',
                        padding: '0.2rem 0.6rem',
                        borderRadius: '4px',
                        fontWeight: 700,
                        fontSize: '0.7rem',
                        cursor: 'pointer',
                      }}>
                        Label Device
                      </button>
                    )}
                  </div>
                </div>
              );
            }
            return null;
          })
        )}
      </div>
    </div>
  );
};

export default DigitalTwin;
