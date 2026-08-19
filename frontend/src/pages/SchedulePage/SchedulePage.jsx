import React from 'react';
import { Calendar } from 'lucide-react';
import '../PlaceholderPage.css';

const SchedulePage = () => {
  return (
    <div className="placeholder-page">
      <div className="placeholder-page__icon">
        <Calendar size={48} />
      </div>
      <h2>Schedule</h2>
      <p>Automated appliance scheduling and energy optimization rules.</p>
      <span className="placeholder-page__badge">Coming Soon</span>
    </div>
  );
};

export default SchedulePage;
