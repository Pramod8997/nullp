import React from 'react';
import {
  Home,
  Cpu,
  BarChart3,
  Box,
  Calendar,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
  { id: 'overview', label: 'Overview', icon: Home },
  { id: 'appliances', label: 'Appliances', icon: Cpu },
  { id: 'analytics', label: 'Energy Analytics', icon: BarChart3 },
  { id: 'digital-twin', label: 'Digital Twin', icon: Box },
  { id: 'schedule', label: 'Schedule', icon: Calendar },
  { id: 'alerts', label: 'Alerts', icon: Bell },
  { id: 'settings', label: 'Settings', icon: Settings },
];

const Sidebar = ({ activeTab, onTabChange, collapsed, onToggleCollapse }) => {
  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      {/* Brand */}
      <div className="sidebar__brand">
        <div className="sidebar__brand-icon">
          <Home size={22} />
        </div>
        {!collapsed && (
          <span className="sidebar__brand-text">Smart Home Energy Manager</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="sidebar__nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`sidebar__nav-item ${isActive ? 'sidebar__nav-item--active' : ''}`}
              onClick={() => onTabChange(item.id)}
              title={collapsed ? item.label : undefined}
            >
              <Icon size={20} />
              {!collapsed && <span className="sidebar__nav-label">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Collapse Toggle */}
      <button
        className="sidebar__toggle"
        onClick={onToggleCollapse}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
      </button>
    </aside>
  );
};

export default Sidebar;
