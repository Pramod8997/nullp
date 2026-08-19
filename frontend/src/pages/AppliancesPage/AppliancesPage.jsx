import React from 'react';
import DeviceCards from '../../components/DeviceCards';
import './AppliancesPage.css';

const AppliancesPage = ({ devices }) => {
  return (
    <div className="appliances-page">
      <div className="appliances-page__header">
        <h2>Appliance Management</h2>
        <p className="appliances-page__subtitle">
          Monitor and control all connected appliances in your smart home.
        </p>
      </div>
      <div className="panel">
        <DeviceCards devices={devices} />
      </div>
    </div>
  );
};

export default AppliancesPage;
