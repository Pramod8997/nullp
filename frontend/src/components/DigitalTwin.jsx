import React, { useState } from 'react';
import { BrainCircuit, Zap, HelpCircle, ShieldOff, Thermometer, Tag, AlertTriangle, CheckCircle2 } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

const getPmvLabel = (pmv) => {
  if (pmv <= -2) return { label: 'Cold', cls: 'cold text-blue-500 bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800' };
  if (pmv <= -1) return { label: 'Cool', cls: 'cold text-cyan-500 bg-cyan-50 dark:bg-cyan-950/50 border-cyan-200 dark:border-cyan-800' };
  if (pmv <= 1) return { label: 'Comfortable (Neutral)', cls: 'comfort neutral text-emerald-500 bg-emerald-50 dark:bg-emerald-950/50 border-emerald-200 dark:border-emerald-800' };
  if (pmv <= 2) return { label: 'Warm', cls: 'warm text-amber-500 bg-amber-50 dark:bg-amber-950/50 border-amber-200 dark:border-amber-800' };
  return { label: 'Hot', cls: 'hot text-rose-500 bg-rose-50 dark:bg-rose-950/50 border-rose-200 dark:border-rose-800' };
};

const LabelRequestCard = ({ event, onLabeled }) => {
  const [label, setLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = label.trim();
    if (!trimmed) { setError('Class name is required'); return; }

    setLoading(true);
    setError('');

    try {
      const segments = event.embedding && event.embedding.length === 128
        ? [event.embedding]
        : null;

      if (!segments) {
        setError('No embedding data in event. Retrying next cycle.');
        setLoading(false);
        return;
      }

      const res = await fetch(`${API_BASE}/api/submit-label`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': import.meta.env.VITE_API_KEY || 'changeme-ems-prod-key'
        },
        body: JSON.stringify({ device_id: event.device_id, label: trimmed, segments }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      setDone(true);
      if (onLabeled) onLabeled(trimmed);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div className="flex items-center gap-1.5 p-3 rounded-xl bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 text-xs font-semibold mt-2">
        <CheckCircle2 size={14} />
        <span>"{label}" successfully enrolled into Prototype Registry</span>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="mt-2.5">
      <div className="flex items-center gap-2 flex-wrap">
        <input
          id={`label-input-${event.device_id}`}
          type="text"
          placeholder="Enter appliance name (e.g. Microwave)..."
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          disabled={loading}
          className="flex-1 min-w-[140px] px-3 py-1.5 bg-white dark:bg-gray-800 border border-amber-300 dark:border-amber-700 rounded-lg text-xs text-gray-900 dark:text-amber-100 focus:outline-none focus:ring-2 focus:ring-amber-500"
        />
        <button
          id={`label-submit-${event.device_id}`}
          type="submit"
          disabled={loading || !label.trim()}
          className="px-3.5 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-gray-950 font-bold text-xs shadow-sm transition-all cursor-pointer"
        >
          {loading ? 'Saving...' : 'Label Device'}
        </button>
      </div>
      {error && (
        <p className="text-rose-500 text-[11px] mt-1 font-medium">{error}</p>
      )}
    </form>
  );
};

const DigitalTwin = ({
  events = [],
  pmvScore,
  pmv,
  rlLog = [],
  unknownDevices = [],
  onLabel,
  ...rest
}) => {
  const currentPmv = typeof pmv === 'number' ? pmv : (typeof pmvScore === 'number' ? pmvScore : 0);
  const pmvInfo = getPmvLabel(currentPmv);
  const [labeled, setLabeled] = useState({});
  const [unknownInputs, setUnknownInputs] = useState({});
  const [dismissedUnknowns, setDismissedUnknowns] = useState({});

  const handleLabeled = (deviceId, className) => {
    setLabeled((prev) => ({ ...prev, [deviceId]: className }));
  };

  const handleUnknownSubmit = (reqId, labelVal) => {
    if (onLabel) {
      onLabel(reqId, labelVal);
    }
    setDismissedUnknowns((prev) => ({ ...prev, [reqId]: true }));
  };

  const combinedEvents = events.length > 0 ? events : rlLog;

  return (
    <div className="w-full space-y-5" data-ppd={rest.ppd}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-200/80 dark:border-gray-700/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-purple-50 dark:bg-purple-950/60 text-purple-600 dark:text-purple-400">
            <BrainCircuit size={18} />
          </div>
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">
              AI Digital Twin & RL Agent
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Autonomous energy dispatch & thermal comfort
            </p>
          </div>
        </div>
      </div>

      {/* PMV Thermal Comfort Gauge Card */}
      <div
        data-testid="pmv-gauge"
        data-pmv={String(currentPmv)}
        className={`pmv-gauge ${pmvInfo.cls} ${currentPmv === 0 ? 'neutral comfort' : ''} p-4 rounded-2xl border flex items-center justify-between transition-all duration-300`}
      >
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-100 dark:bg-purple-900/50 text-purple-600 dark:text-purple-300">
            <Thermometer size={24} />
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
              Predicted Mean Vote (PMV)
            </span>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {pmvInfo.label}
            </div>
          </div>
        </div>

        <div className="text-2xl font-mono font-extrabold px-3.5 py-1 rounded-xl bg-white/80 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700 shadow-sm">
          {currentPmv.toFixed(2)}
        </div>
      </div>

      {/* Unknown Devices Label Requests */}
      {unknownDevices.map((dev) => {
        const reqId = dev.requestId || dev.id;
        if (dismissedUnknowns[reqId]) return null;
        const currentVal = unknownInputs[reqId] || '';

        return (
          <div
            key={reqId}
            data-testid={`label-request-${reqId}`}
            className="label-request-card p-4 rounded-2xl bg-amber-50/80 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800/60 shadow-sm transition-all"
          >
            <div className="flex items-center gap-2 text-xs font-bold text-amber-800 dark:text-amber-300 mb-2">
              <HelpCircle size={15} />
              <span>Unknown Device Detected: {dev.id}</span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                role="textbox"
                placeholder="Enter appliance name (e.g. Dishwasher)..."
                value={currentVal}
                onChange={(e) => {
                  const val = e.target.value;
                  setUnknownInputs((prev) => ({ ...prev, [reqId]: val }));
                }}
                className="flex-1 px-3 py-2 bg-white dark:bg-gray-800 border border-amber-300 dark:border-amber-700 rounded-xl text-xs text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
              />
              <button
                type="button"
                onClick={() => handleUnknownSubmit(reqId, currentVal)}
                className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-gray-950 font-bold text-xs shadow-sm transition-all cursor-pointer"
              >
                Submit
              </button>
            </div>
          </div>
        );
      })}

      {/* Agent Event Log */}
      <div className="space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">
          Reinforcement Learning Log
        </h3>

        <div className="event-log space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
          {combinedEvents.length === 0 ? (
            <div className="flex flex-col items-center justify-center p-8 bg-gray-50/50 dark:bg-gray-800/40 rounded-2xl border border-dashed border-gray-200 dark:border-gray-700 text-center">
              <BrainCircuit size={28} className="text-gray-400 dark:text-gray-500 mb-2 opacity-60" />
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Agent monitoring nominal</p>
            </div>
          ) : (
            combinedEvents.slice(0, 15).map((event, i) => {
              if (event.type === 'RL_ACTION') {
                return (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-purple-50/60 dark:bg-purple-950/20 border border-purple-200/60 dark:border-purple-800/40">
                    <div className="p-2 rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400 shrink-0">
                      <Zap size={15} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold text-purple-700 dark:text-purple-300">RL Optimization</div>
                      <div className="text-xs text-gray-700 dark:text-gray-300 mt-0.5">{event.message}</div>
                      {event.confidence !== undefined && (
                        <div className="text-[11px] font-mono text-gray-500 dark:text-gray-400 mt-1">
                          conf={event.confidence?.toFixed(3)} | PMV={event.pmv?.toFixed(2)} | ToU=${event.tou_rate?.toFixed(2)}/kWh
                        </div>
                      )}
                    </div>
                  </div>
                );
              }

              if (event.type === 'EMPATHY_BLOCK' || event.type === 'EMPATHY_ACTION') {
                return (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-rose-50/60 dark:bg-rose-950/20 border border-rose-200/60 dark:border-rose-800/40">
                    <div className="p-2 rounded-lg bg-rose-500/10 text-rose-600 dark:text-rose-400 shrink-0">
                      <ShieldOff size={15} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold text-rose-700 dark:text-rose-300">Empathy Gate Intervention</div>
                      <div className="text-xs text-gray-700 dark:text-gray-300 mt-0.5">{event.message}</div>
                    </div>
                  </div>
                );
              }

              if (event.type === 'LABEL_REQUEST') {
                const alreadyLabeled = labeled[event.device_id];
                return (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-amber-50/70 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800/60">
                    <div className="p-2 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 shrink-0">
                      <Tag size={15} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold text-amber-800 dark:text-amber-300">
                        Unknown Device — Label Required
                      </div>
                      <div className="text-xs text-gray-700 dark:text-gray-300 mt-0.5">{event.message}</div>
                      <div className="text-[11px] font-mono text-gray-500 dark:text-gray-400 mt-1">
                        Device: <b>{event.device_id}</b> | Power: {event.power}W
                      </div>
                      {alreadyLabeled ? (
                        <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mt-2">
                          ✓ Enrolled as "{alreadyLabeled}"
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

              if (event.type === 'LOW_CONFIDENCE') {
                return (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200/50 dark:border-amber-800/40">
                    <div className="p-2 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 shrink-0">
                      <AlertTriangle size={15} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold text-amber-700 dark:text-amber-300">Low Confidence Detection</div>
                      <div className="text-xs text-gray-700 dark:text-gray-300 mt-0.5">
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
    </div>
  );
};

export default DigitalTwin;
