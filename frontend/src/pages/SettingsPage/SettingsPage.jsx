import React from 'react';
import { Settings as SettingsIcon } from 'lucide-react';
import '../PlaceholderPage.css';

const SettingsPage = () => {
  return (
    <div className="placeholder-page">
      <div className="placeholder-page__icon">
        <SettingsIcon size={48} />
      </div>
      <h2>Settings</h2>
      <p>System configuration, device management, and notification preferences.</p>
      <span className="placeholder-page__badge">Coming Soon</span>
    </div>
  );
};

export default SettingsPage;
