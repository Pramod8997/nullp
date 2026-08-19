import React from 'react';
import SummaryCards from '../../components/SummaryCards/SummaryCards';
import ApplianceTable from '../../components/ApplianceTable/ApplianceTable';
import EnergyChart from '../../components/EnergyChart/EnergyChart';
import DigitalTwinVisualization from '../../components/DigitalTwinVisualization/DigitalTwinVisualization';
import './OverviewPage.css';

const OverviewPage = ({ devices, powerHistory }) => {
  return (
    <div className="overview-page">
      {/* Row 1: Summary KPI Cards */}
      <SummaryCards devices={devices} powerHistory={powerHistory} />

      {/* Row 2: Appliance Table + Energy Chart */}
      <div className="overview-page__row">
        <div className="overview-page__col overview-page__col--table">
          <ApplianceTable devices={devices} />
        </div>
        <div className="overview-page__col overview-page__col--chart">
          <EnergyChart powerHistory={powerHistory} />
        </div>
      </div>

      {/* Row 3: Digital Twin Visualization (right-side preview) */}
      <div className="overview-page__twin-preview">
        <DigitalTwinVisualization devices={devices} />
      </div>
    </div>
  );
};

export default OverviewPage;
