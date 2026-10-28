const TIME_ONLY_PATTERN = /^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?$/;
const NUMERIC_TIMESTAMP_LIMIT = 1e11;

/**
 * Convert a supported backend/browser timestamp to milliseconds since epoch.
 * Time-only values intentionally use a fixed UTC day: they have no timezone
 * information and must not be shifted by the browser's local timezone.
 */
export const parseTimestamp = (value) => {
  if (value instanceof Date) {
    const milliseconds = value.getTime();
    return Number.isFinite(milliseconds) ? milliseconds : null;
  }

  if (typeof value === 'number' || (typeof value === 'string' && value.trim() !== '')) {
    const stringValue = typeof value === 'string' ? value.trim() : value;
    const numericValue = Number(stringValue);
    if (Number.isFinite(numericValue)) {
      const milliseconds = Math.abs(numericValue) < NUMERIC_TIMESTAMP_LIMIT
        ? numericValue * 1000
        : numericValue;
      return Number.isFinite(milliseconds) ? milliseconds : null;
    }
  }

  if (typeof value !== 'string') return null;
  const text = value.trim();
  const timeOnly = text.match(TIME_ONLY_PATTERN);
  if (timeOnly) {
    const hours = Number(timeOnly[1]);
    const minutes = Number(timeOnly[2]);
    const seconds = Number(timeOnly[3] || 0);
    const milliseconds = Number((timeOnly[4] || '').padEnd(3, '0') || 0);
    if (hours > 23 || minutes > 59 || seconds > 59) return null;
    return Date.UTC(1970, 0, 1, hours, minutes, seconds, milliseconds);
  }

  const parsed = Date.parse(text);
  return Number.isFinite(parsed) ? parsed : null;
};

export const finiteNonNegative = (value) => {
  if (typeof value !== 'number' && (typeof value !== 'string' || value.trim() === '')) {
    return null;
  }
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue >= 0 ? numericValue : null;
};

export const getTimestampValue = (point) => {
  if (!point || typeof point !== 'object' || Array.isArray(point)) return null;
  return parseTimestamp(point.timestamp) ?? parseTimestamp(point.time);
};

export const getTimeLabel = (point) => {
  if (!point || typeof point !== 'object' || Array.isArray(point)) return '';
  const candidates = [point.time, point.timestamp];
  const validCandidate = candidates.find((candidate) => (
    typeof candidate === 'string'
      ? candidate.trim() !== '' && parseTimestamp(candidate) !== null
      : parseTimestamp(candidate) !== null
  ));

  if (validCandidate === undefined) return '';
  if (typeof validCandidate === 'string') return validCandidate.trim();

  const milliseconds = parseTimestamp(validCandidate);
  return new Date(milliseconds).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

export const MAX_REALTIME_POINTS = 60;
export const MAX_ENERGY_POINTS = 120;

const RESERVED_POWER_POINT_KEYS = new Set([
  'time',
  'timestamp',
  'device_id',
  'power',
  'watts',
  'unit',
  'source',
  'method',
]);

const normalizePowerPoint = (point, sequence) => {
  if (!point || typeof point !== 'object' || Array.isArray(point)) return null;
  const sortKey = getTimestampValue(point);
  const time = getTimeLabel(point);
  if (sortKey === null || !time) return null;

  const values = {};
  const deviceId = typeof point.device_id === 'string' ? point.device_id.trim() : '';
  const directPower = finiteNonNegative(point.power ?? point.watts);
  if (deviceId && directPower !== null) values[deviceId] = directPower;
  if (!deviceId && directPower !== null) values.power = directPower;

  Object.entries(point).forEach(([key, value]) => {
    if (RESERVED_POWER_POINT_KEYS.has(key)) return;
    const watts = finiteNonNegative(value);
    if (watts !== null) values[key] = watts;
  });

  if (Object.keys(values).length === 0) return null;
  return { sortKey, time, values, sequence };
};

export const buildRealTimeChartData = (data, maxPoints = MAX_REALTIME_POINTS) => {
  if (!Array.isArray(data) || maxPoints <= 0) return [];

  const timeMap = new Map();
  data.forEach((point, sequence) => {
    const normalized = normalizePowerPoint(point, sequence);
    if (!normalized) return;

    const existing = timeMap.get(normalized.sortKey);
    if (existing) {
      Object.assign(existing, normalized.values);
      return;
    }
    timeMap.set(normalized.sortKey, {
      time: normalized.time,
      sortKey: normalized.sortKey,
      sequence: normalized.sequence,
      ...normalized.values,
    });
  });

  return [...timeMap.values()]
    .sort((a, b) => a.sortKey - b.sortKey || a.sequence - b.sequence)
    .slice(-maxPoints)
    .map((point) => {
      const result = { ...point };
      delete result.sortKey;
      delete result.sequence;
      return result;
    });
};

const normalizeEnergyRow = (row, sequence) => {
  if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
  const sortKey = getTimestampValue(row);
  const time = getTimeLabel(row);
  if (sortKey === null || !time) return null;
  const yesterday = finiteNonNegative(row.yesterday);
  const today = finiteNonNegative(row.today);
  if (yesterday === null && today === null) return null;
  return { sortKey, time, yesterday, today, sequence };
};

export const buildEnergyChartData = (history, maxPoints = MAX_ENERGY_POINTS) => {
  if (!Array.isArray(history) || maxPoints <= 0) return [];

  const timeMap = new Map();
  history.forEach((row, sequence) => {
    const normalized = normalizeEnergyRow(row, sequence);
    if (!normalized) return;

    const existing = timeMap.get(normalized.sortKey);
    if (existing) {
      if (normalized.yesterday !== null) existing.yesterday = normalized.yesterday;
      if (normalized.today !== null) existing.today = normalized.today;
      return;
    }
    timeMap.set(normalized.sortKey, normalized);
  });

  return [...timeMap.values()]
    .sort((a, b) => a.sortKey - b.sortKey || a.sequence - b.sequence)
    .slice(-maxPoints)
    .map((row) => {
      const result = { ...row };
      delete result.sortKey;
      delete result.sequence;
      return result;
    });
};
