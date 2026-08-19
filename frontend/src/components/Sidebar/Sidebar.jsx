import React from 'react';
import {
  LayoutDashboard,
  Cpu,
  BarChart3,
  Box,
  Calendar,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  Activity,
} from 'lucide-react';

const navItems = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'appliances', label: 'Devices', icon: Cpu },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'digital-twin', label: 'Digital Twin', icon: Box },
  { id: 'schedule', label: 'Schedule', icon: Calendar },
  { id: 'alerts', label: 'Alerts', icon: Bell },
  { id: 'settings', label: 'Settings', icon: Settings },
];

const Sidebar = ({ activeTab, onTabChange, collapsed, onToggleCollapse, alertCount = 0 }) => {
  return (
    <aside
      className={`relative flex flex-col h-screen border-r border-gray-200/80 dark:border-gray-800/80 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl transition-all duration-300 z-30 ${
        collapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-gray-200/60 dark:border-gray-800/60 h-20">
        <div className="flex items-center justify-center h-11 w-11 rounded-xl bg-gradient-to-br from-blue-500 via-indigo-500 to-cyan-500 text-white shadow-lg shadow-blue-500/20 shrink-0">
          <Zap className="h-6 w-6 fill-current animate-pulse" />
        </div>
        {!collapsed && (
          <div className="flex flex-col overflow-hidden transition-opacity duration-200">
            <span className="text-base font-bold tracking-tight text-gray-900 dark:text-white truncate">
              Smart Energy
            </span>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400 flex items-center gap-1.5 truncate">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
              IoT Digital Twin
            </span>
          </div>
        )}
      </div>

      {/* Navigation List */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id || (item.id === 'appliances' && activeTab === 'devices');
          
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              title={collapsed ? item.label : undefined}
              className={`group flex items-center w-full px-3 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400 font-semibold shadow-sm border border-blue-200/60 dark:border-blue-800/50'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100/80 dark:hover:bg-gray-800/60 hover:text-gray-900 dark:hover:text-gray-100'
              } ${collapsed ? 'justify-center' : 'justify-start gap-3'}`}
            >
              <div className={`transition-transform duration-200 group-hover:scale-110 ${isActive ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500 dark:text-gray-400'}`}>
                <Icon size={20} />
              </div>
              
              {!collapsed && (
                <span className="truncate flex-1 text-left">{item.label}</span>
              )}

              {!collapsed && item.id === 'alerts' && alertCount > 0 && (
                <span className="px-2 py-0.5 text-xs font-bold text-red-600 bg-red-100 dark:text-red-300 dark:bg-red-950/80 rounded-full border border-red-200 dark:border-red-800">
                  {alertCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer / Toggle Button */}
      <div className="p-3 border-t border-gray-200/60 dark:border-gray-800/60 flex items-center justify-between">
        {!collapsed && (
          <div className="flex items-center gap-2 px-2 text-xs text-gray-500 dark:text-gray-400 font-mono">
            <Activity size={14} className="text-emerald-500" />
            <span>v2.4.0 • LIVE</span>
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className={`p-2 rounded-xl text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors ${
            collapsed ? 'mx-auto' : ''
          }`}
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
