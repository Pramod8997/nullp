import React from 'react';
import { Zap, Battery, IndianRupee, Leaf, Activity } from 'lucide-react';

const SummaryCards = ({ devices = {}, analytics = {} }) => {
  // Calculate total current power
  const totalPower = Object.values(devices).reduce(
    (sum, dev) => sum + (dev?.power || 0),
    0
  );

  const totalKwh = Number(analytics?.total_kwh);
  const estimatedCost = Number(analytics?.estimated_cost_inr);
  const energySaved = Number(analytics?.energy_saved_kwh);
  const hasTotalKwh = Number.isFinite(totalKwh);
  const hasEstimatedCost = Number.isFinite(estimatedCost);
  const hasEnergySaved = Number.isFinite(energySaved);

  const cards = [
    {
      id: 'total-power',
      label: 'Total Active Power',
      value: (totalPower / 1000).toFixed(2),
      unit: 'kW',
      icon: Zap,
      accentColor: 'text-blue-600 dark:text-blue-400',
      iconBg: 'bg-blue-500/10 text-blue-600 dark:bg-blue-500/20 dark:text-blue-400 border border-blue-500/20',
      badge: 'Live 1Hz',
      badgeType: 'live',
    },
    {
      id: 'today-energy',
      label: "Today's Consumption",
      value: hasTotalKwh ? totalKwh.toFixed(3) : 'Unavailable',
      unit: hasTotalKwh ? 'kWh' : '',
      icon: Battery,
      accentColor: 'text-emerald-600 dark:text-emerald-400',
      iconBg: 'bg-emerald-500/10 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-400 border border-emerald-500/20',
      badge: hasTotalKwh ? 'Backend analytics' : 'Unavailable',
      badgeType: hasTotalKwh ? 'analytics' : 'unavailable',
    },
    {
      id: 'est-cost',
      label: 'Estimated Cost',
      value: hasEstimatedCost ? `₹ ${estimatedCost.toFixed(2)}` : 'Unavailable',
      unit: '',
      icon: IndianRupee,
      accentColor: 'text-amber-600 dark:text-amber-400',
      iconBg: 'bg-amber-500/10 text-amber-600 dark:bg-amber-500/20 dark:text-amber-400 border border-amber-500/20',
      badge: hasEstimatedCost ? 'Backend analytics' : 'Unavailable',
      badgeType: hasEstimatedCost ? 'analytics' : 'unavailable',
    },
    {
      id: 'energy-saved',
      label: 'Eco Efficiency & Saved',
      value: hasEnergySaved ? energySaved.toFixed(3) : 'Unavailable',
      unit: hasEnergySaved ? 'kWh' : '',
      icon: Leaf,
      accentColor: 'text-cyan-600 dark:text-cyan-400',
      iconBg: 'bg-cyan-500/10 text-cyan-600 dark:bg-cyan-500/20 dark:text-cyan-400 border border-cyan-500/20',
      badge: hasEnergySaved ? 'Backend baseline' : 'Unavailable',
      badgeType: hasEnergySaved ? 'eco' : 'unavailable',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
      {cards.map((card) => {
        const Icon = card.icon;

        return (
          <div
            key={card.id}
            className="relative bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border border-gray-200/80 dark:border-gray-700/80 rounded-2xl p-5 shadow-sm hover:shadow-md hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-300 flex flex-col justify-between"
          >
            {/* Header */}
            <div className="flex items-center justify-between gap-2 mb-3">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 tracking-wide uppercase">
                {card.label}
              </span>
              <div className={`p-2.5 rounded-xl ${card.iconBg}`}>
                <Icon size={18} />
              </div>
            </div>

            {/* Value Display */}
            <div className="my-2">
              <div className="text-2xl lg:text-3xl font-extrabold font-mono tracking-tight text-gray-900 dark:text-white flex items-baseline gap-1.5">
                <span>{card.value}</span>
                {card.unit && (
                  <span className="text-sm font-semibold text-gray-500 dark:text-gray-400 font-sans">
                    {card.unit}
                  </span>
                )}
              </div>
            </div>

            {/* Footer / Badge / Trend */}
            <div className="flex items-center gap-2 pt-2 text-xs">
              {card.badgeType === 'live' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
                  {card.badge}
                </span>
              )}

              {card.badgeType === 'eco' && (
                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold bg-cyan-50 text-cyan-700 dark:bg-cyan-950/60 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-800">
                  <Activity size={12} />
                  {card.badge}
                </span>
              )}

              {card.badgeType === 'analytics' && (
                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
                  <Activity size={12} />
                  {card.badge}
                </span>
              )}

              {card.badgeType === 'unavailable' && (
                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold bg-gray-100 text-gray-600 dark:bg-gray-900/60 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                  {card.badge}
                </span>
              )}

            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SummaryCards;
