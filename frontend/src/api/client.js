const trimTrailingSlashes = (value) => value.replace(/\/+$/, '');

const getHttpOrigin = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return trimTrailingSlashes(import.meta.env.VITE_API_BASE_URL);
  }

  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${window.location.hostname}:8000`;
};

export const API_BASE = getHttpOrigin();

export const WS_URL = import.meta.env.VITE_WS_URL
  ? import.meta.env.VITE_WS_URL
  : `${API_BASE.replace(/^http/, 'ws')}/ws`;

export const getApiKey = () => import.meta.env.VITE_API_KEY?.trim() || '';

const readResponseBody = async (response) => {
  const contentType = response.headers?.get?.('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  try {
    return await response.text();
  } catch {
    return null;
  }
};

const getErrorMessage = (body, status) => {
  if (body && typeof body === 'object') {
    if (typeof body.detail === 'string') return body.detail;
    if (body.detail && typeof body.detail.message === 'string') return body.detail.message;
    if (typeof body.message === 'string') return body.message;
  }
  if (typeof body === 'string' && body.trim()) return body;
  return `Request failed (${status})`;
};

export const requestJson = async (path, options = {}) => {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const apiKey = getApiKey();
  if (apiKey && !headers.has('X-API-Key')) {
    headers.set('X-API-Key', apiKey);
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  const body = await readResponseBody(response);

  if (!response.ok) {
    const error = new Error(getErrorMessage(body, response.status));
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return body;
};

export const getApiUrl = (path) => `${API_BASE}${path}`;
