import React from 'react';
import { Ghost, Zap } from 'lucide-react';

const PhantomTracker = ({ data }) => {
  const { loads = {}, total = 0, offenders = [] } = data;
  const loadEntries = Object.entries(loads).filter(([, v]) => v > 0);

  return (
    <div>
      <div className="panel-header">
        <h2><Ghost size={18} color="#f59e0b" /> Phantom Loads</h2>
        <span className="panel-badge">{loadEntries.length} detected</span>
      </div>

      {/* Total Meter */}
      <div className="phantom-meter">
        <Zap size={28} color="#f59e0b" />
        <div>
          <div className="phantom-meter__value">{total.toFixed(2)}</div>
          <div className="phantom-meter__label">Total Vampire Load (W)</div>
        </div>
      </div>

      {/* Per-device list */}
      {loadEntries.length === 0 ? (
        <div className="empty-state" style={{ minHeight: 100 }}>
          <p>No phantom loads detected</p>
        </div>
      ) : (
        <div className="phantom-list">
          {loadEntries
            .sort(([, a], [, b]) => b - a)
            .map(([deviceId, watts]) => (
              <div key={deviceId} className="phantom-list__item">
                <span className="phantom-list__name">
                  {deviceId.replace('esp32_', '').toUpperCase()}
                </span>
                <span className="phantom-list__watts">{watts.toFixed(3)} W</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};

export default PhantomTracker;
