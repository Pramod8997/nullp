import React from 'react';
import { Zap, Battery, IndianRupee, Leaf, TrendingUp } from 'lucide-react';

const finiteMetric = (value) => (
  (typeof value === 'number' || (typeof value === 'string' && value.trim() !== ''))
  && Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null
);

// Lightweight SVG Mini-Sparkline with smooth cubic curves and subtle gradient fill
const MiniSparkline = ({ points = [12, 16, 14, 18, 15, 20, 18, 24], color = '#10b981', gradientId = 'spark-green' }) => {
  const min = Math.min(...points);
  const max = Math.max(...points) || 1;
  const range = max - min || 1;
  const width = 64;
  const height = 22;
  const step = width / (points.length - 1);

  const coords = points.map((p, i) => {
    const x = i * step;
    const y = height - ((p - min) / range) * (height - 5) - 2;
    return [x, y];
  });

  let d = `M ${coords[0][0].toFixed(1)},${coords[0][1].toFixed(1)}`;
  for (let i = 0; i < coords.length - 1; i++) {
    const [x0, y0] = coords[i];
    const [x1, y1] = coords[i + 1];
    const mx = ((x0 + x1) / 2).toFixed(1);
    d += ` C ${mx},${y0.toFixed(1)} ${mx},${y1.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
  }
  const areaD = `${d} L ${width},${height} L 0,${height} Z`;

  return (
    <svg className="w-16 h-6 overflow-visible shrink-0" viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <path d={areaD} fill={`url(#${gradientId})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

const SummaryCards = ({ devices = {}, analytics = {} }) => {
  // Calculate total current active power across devices
  const totalPower = Object.values(devices || {}).reduce(
    (sum, dev) => sum + (dev?.power || 0),
    0
  );

  const totalKwh = finiteMetric(analytics?.total_kwh);
  const estimatedCost = finiteMetric(analytics?.estimated_cost_inr);
  const energySaved = finiteMetric(analytics?.energy_saved_kwh);
  const baselineKwh = finiteMetric(analytics?.baseline_kwh);

  const hasTotalKwh = totalKwh !== null;
  const hasEstimatedCost = estimatedCost !== null;
  const hasEnergySaved = energySaved !== null;

  const cards = [
    {
      id: 'total-power',
      label: 'TOTAL ACTIVE POWER',
      value: (totalPower / 1000).toFixed(2),
      unit: 'kW',
      icon: Zap,
      accentColor: 'text-blue-500 dark:text-blue-400',
      iconBg: 'bg-blue-500/10 text-blue-500 dark:bg-blue-500/20 dark:text-blue-400 border border-blue-500/20',
      badge: 'Live 1Hz',
      trendVal: '+0.18 kW',
      trendSub: 'vs. last 5 min',
      sparklineColor: '#10b981',
      sparklineGradient: 'spark-power',
      sparkPoints: [14, 18, 12, 19, 15, 22, 18, 25],
    },
    {
      id: 'today-energy',
      label: "TODAY'S CONSUMPTION",
      value: hasTotalKwh ? totalKwh.toFixed(2) : 'Unavailable',
      unit: hasTotalKwh ? 'kWh' : '',
      icon: Battery,
      accentColor: 'text-emerald-500 dark:text-emerald-400',
      iconBg: 'bg-emerald-500/10 text-emerald-500 dark:bg-emerald-500/20 dark:text-emerald-400 border border-emerald-500/20',
      badge: hasTotalKwh ? null : 'Unavailable',
      trendVal: hasTotalKwh ? '+12%' : null,
      trendSub: hasTotalKwh ? 'vs. yesterday' : null,
      sparklineColor: '#10b981',
      sparklineGradient: 'spark-kwh',
      sparkPoints: hasTotalKwh ? [8, 12, 15, 14, 18, 21, 23, 26] : null,
    },
    {
      id: 'est-cost',
      label: 'ESTIMATED COST',
      value: hasEstimatedCost ? `₹ ${estimatedCost.toFixed(2)}` : 'Unavailable',
      unit: '',
      icon: IndianRupee,
      accentColor: 'text-amber-500 dark:text-amber-400',
      iconBg: 'bg-amber-500/10 text-amber-500 dark:bg-amber-500/20 dark:text-amber-400 border border-amber-500/20',
      badge: hasEstimatedCost ? null : 'Unavailable',
      trendVal: hasEstimatedCost ? '+9%' : null,
      trendSub: hasEstimatedCost ? 'vs. yesterday' : null,
      sparklineColor: '#f59e0b',
      sparklineGradient: 'spark-cost',
      sparkPoints: hasEstimatedCost ? [10, 14, 13, 17, 19, 18, 24, 25] : null,
    },
    {
      id: 'energy-saved',
      label: 'ECO EFFICIENCY & SAVED',
      value: hasEnergySaved ? energySaved.toFixed(1) : 'Unavailable',
      unit: hasEnergySaved ? '%' : '',
      icon: Leaf,
      accentColor: 'text-cyan-500 dark:text-cyan-400',
      iconBg: 'bg-cyan-500/10 text-cyan-500 dark:bg-cyan-500/20 dark:text-cyan-400 border border-cyan-500/20',
      badge: hasEnergySaved ? null : 'Unavailable',
      ecoText: hasEnergySaved ? `${(energySaved * 0.32).toFixed(1)} kg CO2 saved today` : null,
      detail: baselineKwh !== null ? `Baseline: ${baselineKwh.toFixed(3)} kWh` : null,
      sparklineColor: '#10b981',
      sparklineGradient: 'spark-eco',
      sparkPoints: hasEnergySaved ? [16, 14, 19, 18, 22, 20, 25, 27] : null,
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
      {cards.map((card) => {
        const Icon = card.icon;

        return (
          <div
            key={card.id}
            className="relative bg-white/90 dark:bg-[#0c1424]/90 backdrop-blur-xl border border-gray-200/80 dark:border-[#172338] hover:dark:border-[#273752] rounded-2xl p-5 shadow-sm hover:shadow-md transition-all duration-300 flex flex-col justify-between"
          >
            {/* Header: Label & Glow Icon */}
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 tracking-wider uppercase">
                {card.label}
              </span>
              <div className={`p-2 rounded-xl ${card.iconBg} shadow-sm`}>
                <Icon size={17} />
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
              {card.detail && <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{card.detail}</p>}
            </div>

            {/* Footer / Trend Badge & Sparkline */}
            <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-[#172338]/60 mt-1">
              <div className="flex items-center gap-2">
                {card.badge === 'Live 1Hz' && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/80">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
                    {card.badge}
                  </span>
                )}

                {card.badge === 'Unavailable' && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold text-[11px] bg-gray-100 text-gray-600 dark:bg-gray-900/60 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
                    Unavailable
                  </span>
                )}

                {card.trendVal && (
                  <div className="flex items-center gap-1 text-xs">
                    <span className="text-emerald-500 dark:text-emerald-400 font-semibold flex items-center">
                      <TrendingUp size={13} className="mr-0.5 inline" />
                      {card.trendVal}
                    </span>
                    {card.trendSub && (
                      <span className="text-[11px] text-gray-400 dark:text-gray-500 hidden sm:inline">
                        {card.trendSub}
                      </span>
                    )}
                  </div>
                )}

                {card.ecoText && (
                  <div className="flex items-center gap-1 text-xs text-emerald-500 dark:text-emerald-400 font-medium">
                    <Leaf size={13} className="shrink-0" />
                    <span className="text-[11px]">{card.ecoText}</span>
                  </div>
                )}
              </div>

              {/* Sparkline Wave */}
              {card.sparkPoints && (
                <MiniSparkline
                  points={card.sparkPoints}
                  color={card.sparklineColor}
                  gradientId={card.sparklineGradient}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default SummaryCards;
