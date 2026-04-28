import React, { useState } from 'react';
import { BrainCircuit, Zap, HelpCircle, ShieldOff, Thermometer, Tag, AlertTriangle } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const getPmvLabel = (pmv) => {
  if (pmv <= -2) return { label: 'Cold',        cls: 'cold' };
  if (pmv <= -1) return { label: 'Cool',        cls: 'cold' };
  if (pmv <= 1)  return { label: 'Comfortable', cls: 'comfort' };
  if (pmv <= 2)  return { label: 'Warm',        cls: 'warm' };
  return         { label: 'Hot',         cls: 'hot' };
};

/* ── Label Request Card ────────────────────────────────────────────────────
   Shown when the DeltaStabilityAnalyzer flags a STABLE UNKNOWN device.
   Lets the user type a class name and submits to POST /api/label_device.
   The payload includes the embedding stored in the event as `segments`.
*/
const LabelRequestCard = ({ event, onLabeled }) => {
  const [label, setLabel]     = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [done, setDone]       = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = label.trim();
    if (!trimmed) { setError('Class name is required'); return; }

    setLoading(true);
    setError('');

    try {
      // Build (K, 128) segments from the embedding stored in the LABEL_REQUEST event.
      // The pipeline sends cluster_mean as `embedding` (128,); wrap it as [[...]] = (1, 128).
      const segments = event.embedding && event.embedding.length === 128
        ? [event.embedding]
        : null;

      if (!segments) {
        setError('No embedding data in event. Retrying next detection cycle.');
        setLoading(false);
        return;
      }

      const res = await fetch(`${API_BASE}/api/label_device`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ class_name: trimmed, segments }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      setDone(true);
      onLabeled && onLabeled(trimmed);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div style={{
        background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.4)',
        borderRadius: '8px', padding: '0.6rem 0.8rem', marginTop: '0.4rem',
        fontSize: '0.75rem', color: '#86efac', fontWeight: 600,
      }}>
        ✅ "{label}" added to Prototype Registry
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '0.5rem' }}>
      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <input
          id={`label-input-${event.device_id}`}
          type="text"
          placeholder="Enter appliance name…"
          value={label}
          onChange={e => setLabel(e.target.value)}
          disabled={loading}
          style={{
            flex: 1, minWidth: 120,
            background: 'rgba(255,255,255,0.07)',
            border: '1px solid rgba(245,158,11,0.5)',
            borderRadius: '6px', padding: '0.3rem 0.5rem',
            color: '#fef3c7', fontSize: '0.75rem', outline: 'none',
          }}
        />
        <button
          id={`label-submit-${event.device_id}`}
          type="submit"
          disabled={loading || !label.trim()}
          style={{
            background: loading ? '#6b7280' : '#f59e0b',
            color: '#0f172a', border: 'none',
            padding: '0.3rem 0.75rem', borderRadius: '6px',
            fontWeight: 700, fontSize: '0.73rem',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s',
          }}
        >
          {loading ? 'Saving…' : 'Label Device'}
        </button>
      </div>
      {error && (
        <div style={{ color: '#f87171', fontSize: '0.7rem', marginTop: '0.25rem' }}>{error}</div>
      )}
    </form>
  );
};

/* ── Main DigitalTwin component ────────────────────────────────────────── */
const DigitalTwin = ({ events, pmvScore }) => {
  const pmv = getPmvLabel(pmvScore);
  const [labeled, setLabeled] = useState({});   // track which device_ids have been labeled

  const handleLabeled = (deviceId, className) => {
    setLabeled(prev => ({ ...prev, [deviceId]: className }));
  };

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
            /* ── RL Action ── */
            if (event.type === 'RL_ACTION') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(139,92,246,0.15)' }}>
                    <Zap size={16} color="#a78bfa" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#a78bfa' }}>RL Optimization</div>
                    <div className="event-item__desc">{event.message}</div>
                    {event.confidence !== undefined && (
                      <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                        conf={event.confidence?.toFixed(3)} | PMV={event.pmv?.toFixed(2)} | ToU=${event.tou_rate?.toFixed(2)}/kWh
                      </div>
                    )}
                  </div>
                </div>
              );
            }

            /* ── Empathy Block / Empathy Action ── */
            if (event.type === 'EMPATHY_BLOCK' || event.type === 'EMPATHY_ACTION') {
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

            /* ── LABEL_REQUEST (stable unknown — needs user label) ── */
            if (event.type === 'LABEL_REQUEST') {
              const alreadyLabeled = labeled[event.device_id];
              return (
                <div key={i} className="event-item" style={{
                  borderLeft: '3px solid #f59e0b',
                  paddingLeft: '0.5rem',
                }}>
                  <div className="event-item__icon" style={{ background: 'rgba(245,158,11,0.15)' }}>
                    <Tag size={16} color="#f59e0b" />
                  </div>
                  <div className="event-item__body" style={{ width: '100%' }}>
                    <div className="event-item__label" style={{ color: '#f59e0b' }}>
                      Unknown Device — Label Required
                    </div>
                    <div className="event-item__desc">{event.message}</div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                      Device: <b>{event.device_id}</b> | Power: {event.power}W
                    </div>
                    {alreadyLabeled ? (
                      <div style={{ color: '#86efac', fontSize: '0.72rem', marginTop: '0.3rem', fontWeight: 600 }}>
                        ✅ Labeled as "{alreadyLabeled}"
                      </div>
                    ) : (
                      <LabelRequestCard
                        event={event}
                        onLabeled={(cls) => handleLabeled(event.device_id, cls)}
                      />
                    )}
                  </div>
                </div>
              );
            }

            /* ── UNKNOWN_DEVICE (legacy event type) ── */
            if (event.type === 'UNKNOWN_DEVICE') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(245,158,11,0.15)' }}>
                    <HelpCircle size={16} color="#f59e0b" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#f59e0b' }}>Unknown Signature</div>
                    <div className="event-item__desc">{event.message}</div>
                  </div>
                </div>
              );
            }

            /* ── LOW_CONFIDENCE ── */
            if (event.type === 'LOW_CONFIDENCE') {
              return (
                <div key={i} className="event-item">
                  <div className="event-item__icon" style={{ background: 'rgba(251,191,36,0.12)' }}>
                    <AlertTriangle size={16} color="#fbbf24" />
                  </div>
                  <div className="event-item__body">
                    <div className="event-item__label" style={{ color: '#fbbf24' }}>Low Confidence</div>
                    <div className="event-item__desc">
                      {event.classified_as} — conf {event.confidence?.toFixed(3)} &lt; {event.threshold}
                    </div>
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
