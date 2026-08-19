import React from 'react';
import DigitalTwin from '../../components/DigitalTwin';
import DigitalTwinVisualization from '../../components/DigitalTwinVisualization/DigitalTwinVisualization';
import RealTimeChart from '../../components/RealTimeChart';
import DeviceCards from '../../components/DeviceCards';
import PhantomTracker from '../../components/PhantomTracker';
import './DigitalTwinPage.css';

const DigitalTwinPage = ({
  devices,
  powerHistory,
  twinEvents,
  pmvScore,
  phantomData,
  pendingUnknowns,
}) => {
  return (
    <div className="dt-page">
      {/* Top: Full-width Digital Twin 3D Visualization */}
      <div className="dt-page__visualization">
        <DigitalTwinVisualization devices={devices} />
      </div>

      {/* Middle Row: Power Monitor + Event Log */}
      <div className="dt-page__row">
        <div className="dt-page__col dt-page__col--chart">
          <div className="panel">
            <div className="panel-header">
              <h2>⚡ Real-Time Power Monitor</h2>
              <span className="panel-badge">{Object.keys(devices).length} devices</span>
            </div>
            <RealTimeChart data={powerHistory} devices={devices} />
          </div>
        </div>

        <div className="dt-page__col dt-page__col--twin">
          <div className="panel">
            <DigitalTwin events={twinEvents} pmvScore={pmvScore} />
          </div>
        </div>
      </div>

      {/* Bottom Row: Devices + Phantom Loads */}
      <div className="dt-page__row">
        <div className="dt-page__col dt-page__col--devices">
          <div className="panel">
            <DeviceCards devices={devices} />
          </div>
        </div>

        <div className="dt-page__col dt-page__col--phantom">
          <div className="panel">
            <PhantomTracker data={phantomData} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DigitalTwinPage;
