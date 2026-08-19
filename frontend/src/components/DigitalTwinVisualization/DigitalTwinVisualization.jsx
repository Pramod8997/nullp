import React from 'react';
import {
  Tv,
  Refrigerator,
  Coffee,
  Wind,
  WashingMachine,
  Flame,
  Zap,
  Power,
  Sun,
  Bed,
  Bath,
  UtensilsCrossed,
  Sofa,
  Layers,
} from 'lucide-react';

const formatDeviceName = (id) => {
  return id
    .replace('esp32_', '')
    .replace('node_', '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const getDeviceIcon = (id) => {
  const lower = id.toLowerCase();
  if (lower.includes('tv')) return Tv;
  if (lower.includes('fridge') || lower.includes('refrigerator')) return Refrigerator;
  if (lower.includes('kettle') || lower.includes('coffee')) return Coffee;
  if (lower.includes('hvac') || lower.includes('ac') || lower.includes('fan')) return Wind;
  if (lower.includes('washer') || lower.includes('dryer')) return WashingMachine;
  if (lower.includes('oven') || lower.includes('stove') || lower.includes('microwave')) return Flame;
  if (lower.includes('light') || lower.includes('lamp')) return Sun;
  return Zap;
};

const assignRoom = (id) => {
  const lower = id.toLowerCase();
  if (lower.includes('fridge') || lower.includes('kettle') || lower.includes('microwave') || lower.includes('oven') || lower.includes('dishwasher') || lower.includes('cooker')) {
    return 'kitchen';
  }
  if (lower.includes('washer') || lower.includes('dryer') || lower.includes('bath') || lower.includes('water_heater') || lower.includes('geyser')) {
    return 'bathroom';
  }
  if (lower.includes('bed') || lower.includes('lamp') || lower.includes('desk') || lower.includes('nightstand')) {
    return 'bedroom';
  }
  // Default to living room (tv, hvac, lighting, general nodes)
  return 'living_room';
};

const roomsConfig = [
  {
    id: 'living_room',
    name: 'Living Room',
    icon: Sofa,
    description: 'Zone A • Main Living Area',
  },
  {
    id: 'kitchen',
    name: 'Kitchen',
    icon: UtensilsCrossed,
    description: 'Zone B • Culinary & Heavy Appliances',
  },
  {
    id: 'bedroom',
    name: 'Master Bedroom',
    icon: Bed,
    description: 'Zone C • Comfort & Climate',
  },
  {
    id: 'bathroom',
    name: 'Utility & Bathroom',
    icon: Bath,
    description: 'Zone D • Laundry & Water Systems',
  },
];

const DigitalTwinVisualization = ({ devices = {} }) => {
  const deviceList = Object.entries(devices);

  // Group devices by room
  const roomDevices = {
    living_room: [],
    kitchen: [],
    bedroom: [],
    bathroom: [],
  };

  // If there are devices that are not assigned specifically, distribute them cleanly
  deviceList.forEach(([id, dev]) => {
    const room = assignRoom(id);
    if (roomDevices[room]) {
      roomDevices[room].push([id, dev]);
    } else {
      roomDevices.living_room.push([id, dev]);
    }
  });

  // Calculate total power
  const totalPower = deviceList.reduce((acc, [, dev]) => acc + (dev?.power || 0), 0);
  const activeCount = deviceList.filter(([, dev]) => (dev?.power || 0) > 10 || dev?.state === 'ON').length;

  return (
    <div className="w-full bg-white/70 dark:bg-gray-900/70 backdrop-blur-xl border border-gray-200/80 dark:border-gray-800/80 rounded-3xl p-5 md:p-7 shadow-sm transition-all duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-gray-200/60 dark:border-gray-800/60">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-indigo-500 to-purple-600 text-white shadow-md shadow-indigo-500/20">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-tight text-gray-900 dark:text-white">
              Floor Plan Digital Twin
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Interactive 4-Zone Spatial Energy Telemetry
            </p>
          </div>
        </div>

        {/* Global Summary Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-emerald-700 dark:text-emerald-400 text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
            <span>{activeCount} Active Nodes</span>
          </div>
          <div className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800/60 text-blue-700 dark:text-blue-400 text-xs font-bold font-mono">
            <Zap className="h-3.5 w-3.5" />
            <span>{(totalPower / 1000).toFixed(2)} kW Total</span>
          </div>
        </div>
      </div>

      {/* 4-Room Floor Plan Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-6">
        {roomsConfig.map((room) => {
          const RoomIcon = room.icon;
          const devicesInRoom = roomDevices[room.id] || [];
          const roomPower = devicesInRoom.reduce((sum, [, dev]) => sum + (dev?.power || 0), 0);
          const hasActiveDevices = devicesInRoom.some(
            ([, dev]) => (dev?.power || 0) > 10 || dev?.state === 'ON'
          );

          return (
            <div
              key={room.id}
              className={`relative border-2 border-dashed rounded-2xl p-5 transition-all duration-300 ${
                hasActiveDevices
                  ? 'border-indigo-300 dark:border-indigo-800/60 bg-gradient-to-br from-indigo-50/30 via-white/50 to-purple-50/20 dark:from-indigo-950/20 dark:via-gray-800/40 dark:to-purple-950/10'
                  : 'border-gray-200 dark:border-gray-800 bg-white/40 dark:bg-gray-800/30'
              } backdrop-blur-sm hover:border-indigo-400 dark:hover:border-indigo-600/80 group/room`}
            >
              {/* Room Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className={`p-2 rounded-xl ${hasActiveDevices ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900/60 dark:text-indigo-300' : 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400'}`}>
                    <RoomIcon size={18} />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                      {room.name}
                    </h3>
                    <p className="text-[11px] text-gray-500 dark:text-gray-400">
                      {room.description}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 px-2.5 py-1 rounded-lg border border-gray-200 dark:border-gray-700">
                    {roomPower.toFixed(0)} W
                  </span>
                </div>
              </div>

              {/* Devices in Room */}
              <div className="min-h-[110px] flex flex-wrap gap-2.5 content-start pt-1">
                {devicesInRoom.length === 0 ? (
                  <div className="w-full h-24 flex items-center justify-center text-xs text-gray-400 dark:text-gray-500 italic">
                    No connected appliances in this zone
                  </div>
                ) : (
                  devicesInRoom.map(([id, dev]) => {
                    const power = dev?.power || 0;
                    const isOn = (dev?.state || (power > 10 ? 'ON' : 'OFF')) === 'ON';
                    const DevIcon = getDeviceIcon(id);
                    const formattedName = dev?.label || formatDeviceName(id);

                    return (
                      <div
                        key={id}
                        className={`group relative flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all duration-300 cursor-pointer ${
                          isOn
                            ? 'bg-emerald-50/90 text-emerald-900 dark:bg-emerald-950/60 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-700/60 shadow-[0_0_12px_rgba(16,185,129,0.2)] hover:shadow-[0_0_18px_rgba(16,185,129,0.35)]'
                            : 'bg-gray-100/70 text-gray-600 dark:bg-gray-800/60 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                        }`}
                      >
                        {/* Device Icon */}
                        <DevIcon size={14} className={isOn ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-400 dark:text-gray-500'} />
                        
                        {/* Device Label */}
                        <span className="font-semibold">{formattedName}</span>

                        {/* Power text / Status pill */}
                        <span className={`font-mono text-[11px] px-1.5 py-0.5 rounded-md ${
                          isOn
                            ? 'bg-emerald-200/60 text-emerald-950 dark:bg-emerald-900/80 dark:text-emerald-200 font-bold'
                            : 'bg-gray-200/60 text-gray-600 dark:bg-gray-700/60 dark:text-gray-400'
                        }`}>
                          {isOn ? `${power.toFixed(0)}W` : '0W'}
                        </span>

                        {/* Pulsing indicator if active */}
                        {isOn && (
                          <span className="relative flex h-2 w-2 ml-0.5">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                          </span>
                        )}

                        {/* ── Interactive Hover Tooltip ── */}
                        <div className="opacity-0 scale-95 pointer-events-none group-hover:opacity-100 group-hover:scale-100 group-hover:pointer-events-auto transition-all duration-200 absolute -top-16 left-1/2 -translate-x-1/2 z-50 whitespace-nowrap bg-gray-950/95 dark:bg-gray-900/95 backdrop-blur-md text-white text-xs px-3.5 py-2 rounded-xl shadow-2xl border border-gray-700/80 flex flex-col items-center gap-0.5">
                          <span className="font-bold text-gray-100">{formattedName}</span>
                          <div className="flex items-center gap-2 text-[11px]">
                            <span className={isOn ? 'text-emerald-400 font-bold' : 'text-gray-400'}>
                              {isOn ? 'ACTIVE (ON)' : 'STANDBY (OFF)'}
                            </span>
                            <span className="text-gray-600">•</span>
                            <span className="font-mono text-cyan-300 font-bold">{power.toFixed(1)} W</span>
                          </div>
                          {/* Tooltip caret */}
                          <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-950/95 border-r border-b border-gray-700/80 transform rotate-45" />
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DigitalTwinVisualization;
