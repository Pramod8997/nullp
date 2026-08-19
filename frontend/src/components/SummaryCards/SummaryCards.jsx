import React from 'react';
import { Zap, Battery, IndianRupee, Leaf } from 'lucide-react';
import './SummaryCards.css';

const SummaryCards = ({ devices = {}, powerHistory = [] }) => {
  // Calculate total current power
  const totalPower = Object.values(devices).reduce(
    (sum, dev) => sum + (dev?.power || 0),
    0
  );

  // Estimate today's energy (simplified: total power * hours assumed)
  const todayEnergy = powerHistory.length > 0
    ? (totalPower * powerHistory.length) / (3600 * 1000) // rough kWh
    : (totalPower / 1000) * 0.5; // fallback rough estimate

  // Estimated cost at ₹8/kWh
  const estimatedCost = (todayEnergy > 0 ? todayEnergy : totalPower * 0.012) * 8;

  // Energy saved (mock: 15% of consumption)
  const energySaved = todayEnergy * 0.15 || totalPower * 0.002;

  const cards = [
    {
      id: 'total-power',
      label: 'Total Power',
      value: (totalPower / 1000).toFixed(2),
      unit: 'kW',
      icon: Zap,
      color: '#3b82f6',
      bgColor: 'rgba(59, 130, 246, 0.12)',
      borderColor: 'rgba(59, 130, 246, 0.3)',
      badge: 'Live',
      badgeColor: '#10b981',
    },
    {
      id: 'today-energy',
      label: "Today's Energy",
      value: todayEnergy > 0 ? todayEnergy.toFixed(1) : (totalPower * 0.012).toFixed(1),
      unit: 'kWh',
      icon: Battery,
      color: '#10b981',
      bgColor: 'rgba(16, 185, 129, 0.12)',
      borderColor: 'rgba(16, 185, 129, 0.3)',
      change: '↑ 8.3% vs yesterday',
      changePositive: false,
    },
    {
      id: 'est-cost',
      label: 'Estimated Cost',
      value: `₹ ${estimatedCost > 0 ? estimatedCost.toFixed(1) : (totalPower * 0.096).toFixed(1)}`,
      unit: '',
      icon: IndianRupee,
      color: '#f59e0b',
      bgColor: 'rgba(245, 158, 11, 0.12)',
      borderColor: 'rgba(245, 158, 11, 0.3)',
      change: '↓ 6.1% vs yesterday',
      changePositive: true,
    },
    {
      id: 'energy-saved',
      label: 'Energy Saved',
      value: energySaved > 0 ? energySaved.toFixed(1) : (totalPower * 0.002).toFixed(1),
      unit: 'kWh',
      icon: Leaf,
      color: '#06b6d4',
      bgColor: 'rgba(6, 182, 212, 0.12)',
      borderColor: 'rgba(6, 182, 212, 0.3)',
      badge: 'Today',
      badgeColor: '#06b6d4',
    },
  ];

  return (
    <div className="summary-cards">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div
            key={card.id}
            className="summary-card"
            style={{
              background: card.bgColor,
              borderColor: card.borderColor,
            }}
          >
            <div className="summary-card__header">
              <span className="summary-card__label">{card.label}</span>
              {card.badge && (
                <span
                  className="summary-card__badge"
                  style={{ color: card.badgeColor }}
                >
                  <span className="summary-card__badge-dot" style={{ background: card.badgeColor }} />
                  {card.badge}
                </span>
              )}
            </div>
            <div className="summary-card__value" style={{ color: card.color }}>
              {card.value}
              {card.unit && <span className="summary-card__unit">{card.unit}</span>}
            </div>
            {card.change && (
              <div
                className="summary-card__change"
                style={{ color: card.changePositive ? '#10b981' : '#94a3b8' }}
              >
                {card.change}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default SummaryCards;
