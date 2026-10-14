/**
 * Typed HTTP API client, query cache with targeted invalidation, and timezone formatter.
 */

import {
  AppSettings,
  Audit,
  AuditDetailBundle,
  AuditEvent,
  AuditPreviewResponse,
  AuditProgress,
  Calibration,
  CapabilityReport,
  ProbeRevision,
  Profile,
  SampleResponse,
  SystemCapabilities,
} from '../types/api';

export class ApiClientError extends Error {
  status: number;
  errorCode: string;
  detail: string;

  constructor(status: number, errorCode: string, detail: string) {
    super(detail);
    this.status = status;
    this.errorCode = errorCode;
    this.detail = detail;
  }
}

const queryCache: Map<string, { data: any; timestamp: number }> = new Map();
const cacheSubscribers: Set<() => void> = new Set();

export function subscribeCache(cb: () => void): () => void {
  cacheSubscribers.add(cb);
  return () => cacheSubscribers.delete(cb);
}

export function invalidateCache(prefix?: string): void {
  if (!prefix) {
    queryCache.clear();
  } else {
    for (const k of Array.from(queryCache.keys())) {
      if (k.startsWith(prefix)) {
        queryCache.delete(k);
      }
    }
  }
  cacheSubscribers.forEach((cb) => cb());
}

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
  useCacheKey?: string
): Promise<T> {
  if (useCacheKey && queryCache.has(useCacheKey)) {
    const cached = queryCache.get(useCacheKey)!;
    if (Date.now() - cached.timestamp < 5000) {
      return cached.data as T;
    }
  }

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.headers as Record<string, string> | undefined),
  };
  if (options.body && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  let resp: Response;
  try {
    resp = await fetch(path, { ...options, headers });
  } catch (err: any) {
    throw new ApiClientError(
      0,
      'NETWORK_OFFLINE',
      `Unable to reach localhost backend (${err?.message || 'network offline'}). Verify the ModelTrace server is running.`
    );
  }

  const rawText = await resp.text();
  let parsed: any = null;
  try {
    parsed = rawText ? JSON.parse(rawText) : {};
  } catch (_e) {
    parsed = { detail: rawText };
  }

  if (!resp.ok) {
    const code =
      parsed?.error_code ||
      (resp.status === 401 || resp.status === 403
        ? 'AUTH_OR_ORIGIN_ERROR'
        : resp.status === 409
        ? 'CONFLICT_STATE'
        : resp.status === 429
        ? 'RATE_LIMIT_429'
        : resp.status === 422
        ? 'SCHEMA_VALIDATION_ERROR'
        : `HTTP_${resp.status}`);
    let detail = '';
    if (typeof parsed?.detail === 'string') {
      detail = parsed.detail;
    } else if (Array.isArray(parsed?.detail)) {
      detail = parsed.detail
        .map((d: any) => `${(d.loc || []).join('.')}: ${d.msg}`)
        .join('; ');
    } else {
      detail = rawText || `Request failed with status ${resp.status}`;
    }
    throw new ApiClientError(resp.status, code, detail);
  }

  if (useCacheKey) {
    queryCache.set(useCacheKey, { data: parsed, timestamp: Date.now() });
  }
  return parsed as T;
}

export const apiClient = {
  getHealth: () => requestJson<{ status: string; app_name: string; tagline: string; version: string; timestamp_utc: string }>('/api/health'),

  getCapabilities: (bypassCache = false) =>
    requestJson<SystemCapabilities>(
      '/api/capabilities',
      {},
      bypassCache ? undefined : 'capabilities'
    ),

  listProfiles: (bypassCache = false) =>
    requestJson<{ profiles: Profile[] }>(
      '/api/profiles',
      {},
      bypassCache ? undefined : 'profiles'
    ),

  createProfile: async (payload: Partial<Profile>) => {
    const res = await requestJson<Profile>('/api/profiles', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('profiles');
    return res;
  },

  updateProfile: async (id: string, payload: Partial<Profile>) => {
    const res = await requestJson<Profile>(`/api/profiles/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    invalidateCache('profiles');
    return res;
  },

  deleteProfile: async (id: string) => {
    const res = await requestJson<{ deleted: boolean; profile_id: string }>(
      `/api/profiles/${encodeURIComponent(id)}`,
      { method: 'DELETE' }
    );
    invalidateCache('profiles');
    return res;
  },

  testProfileConnection: async (id: string, executeNetworkProbe = true) => {
    const res = await requestJson<CapabilityReport>(
      `/api/profiles/${encodeURIComponent(id)}/test-connection?execute_network_probe=${executeNetworkProbe}`,
      { method: 'POST', body: JSON.stringify({}) }
    );
    invalidateCache('profiles');
    return res;
  },

  getDefaultProbes: () => requestJson<ProbeRevision>('/api/probes', {}, 'probes:default'),

  listProbeRevisions: (includePrompts = true, bypassCache = false) =>
    requestJson<{ revisions: ProbeRevision[] }>(
      `/api/probes/revisions?include_prompts=${includePrompts}`,
      {},
      bypassCache ? undefined : `probes:list:${includePrompts}`
    ),

  getProbeRevision: (revId: string) =>
    requestJson<ProbeRevision>(`/api/probes/revisions/${encodeURIComponent(revId)}`),

  createProbeRevision: async (payload: {
    name: string;
    description: string;
    parent_revision_id?: string | null;
    prompts: any[];
  }) => {
    const res = await requestJson<ProbeRevision>('/api/probes/revisions', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('probes');
    return res;
  },

  importProbes: async (payload: {
    name: string;
    description: string;
    format: 'json' | 'jsonl';
    raw_content: string;
  }) => {
    const res = await requestJson<ProbeRevision>('/api/probes/import', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('probes');
    return res;
  },

  listCalibrations: (bypassCache = false) =>
    requestJson<{ calibrations: Calibration[] }>(
      '/api/calibrations',
      {},
      bypassCache ? undefined : 'calibrations'
    ),

  importCalibration: async (payload: any) => {
    const res = await requestJson<Calibration>('/api/calibrations/import', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('calibrations');
    return res;
  },

  runDemoCalibration: async (payload: {
    name: string;
    probe_revision_id?: string | null;
    score_mode: string;
    n_prompts: number;
    m_reference: number;
    max_output_tokens: number;
    alpha: number;
    n_trials: number;
  }) => {
    const res = await requestJson<Calibration>('/api/calibrations/run-demo', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('calibrations');
    return res;
  },

  previewAudit: (payload: any) =>
    requestJson<AuditPreviewResponse>('/api/audits/preview', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createAudit: async (payload: any) => {
    const res = await requestJson<Audit>('/api/audits', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    invalidateCache('audits');
    return res;
  },

  startAudit: async (auditId: string, synchronous = false) => {
    const res = await requestJson<Audit>(
      `/api/audits/${encodeURIComponent(auditId)}/start?synchronous=${synchronous}`,
      { method: 'POST', body: JSON.stringify({}) }
    );
    invalidateCache('audits');
    return res;
  },

  listAudits: (bypassCache = false) =>
    requestJson<{ audits: Audit[] }>(
      '/api/audits',
      {},
      bypassCache ? undefined : 'audits:list'
    ),

  getAuditDetail: (auditId: string, bypassCache = false) =>
    requestJson<AuditDetailBundle>(
      `/api/audits/${encodeURIComponent(auditId)}`,
      {},
      bypassCache ? undefined : `audits:detail:${auditId}`
    ),

  getAuditResponses: (auditId: string) =>
    requestJson<{ audit_id: string; response_count: number; responses: SampleResponse[] }>(
      `/api/audits/${encodeURIComponent(auditId)}/responses`
    ),

  getAuditProgress: (auditId: string) =>
    requestJson<AuditProgress>(`/api/audits/${encodeURIComponent(auditId)}/progress`),

  getAuditEvents: (auditId: string) =>
    requestJson<{ audit_id: string; events: AuditEvent[] }>(
      `/api/audits/${encodeURIComponent(auditId)}/events`
    ),

  cancelAudit: async (auditId: string) => {
    const res = await requestJson<Audit>(
      `/api/audits/${encodeURIComponent(auditId)}/cancel`,
      { method: 'POST', body: JSON.stringify({}) }
    );
    invalidateCache('audits');
    return res;
  },

  resumeAudit: async (auditId: string, synchronous = false) => {
    const res = await requestJson<Audit>(
      `/api/audits/${encodeURIComponent(auditId)}/resume?synchronous=${synchronous}`,
      { method: 'POST', body: JSON.stringify({}) }
    );
    invalidateCache('audits');
    return res;
  },

  purgeAuditEvidence: async (auditId: string) => {
    const res = await requestJson<{
      audit_id: string;
      purged_response_count: number;
      manifest_preserved: boolean;
    }>(`/api/audits/${encodeURIComponent(auditId)}/purge-evidence`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
    invalidateCache('audits');
    return res;
  },

  deleteAudit: async (auditId: string) => {
    const res = await requestJson<{ deleted: boolean; audit_id: string }>(
      `/api/audits/${encodeURIComponent(auditId)}`,
      { method: 'DELETE' }
    );
    invalidateCache('audits');
    return res;
  },

  getSettings: (bypassCache = false) =>
    requestJson<AppSettings>(
      '/api/settings',
      {},
      bypassCache ? undefined : 'settings'
    ),

  updateSettings: async (payload: Partial<AppSettings>) => {
    const res = await requestJson<AppSettings>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    invalidateCache('settings');
    invalidateCache('capabilities');
    return res;
  },

  runRetentionCleanup: async (olderThanDays?: number, preserveManifests = true) => {
    const q = new URLSearchParams();
    if (olderThanDays !== undefined) q.set('older_than_days', String(olderThanDays));
    q.set('preserve_manifests', String(preserveManifests));
    const res = await requestJson<any>(`/api/settings/retention/run?${q.toString()}`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
    invalidateCache('audits');
    return res;
  },
};

export function formatTimestamp(isoUtc: string | null | undefined, tz = 'UTC'): string {
  if (!isoUtc) return '—';
  try {
    const d = new Date(isoUtc);
    if (isNaN(d.getTime())) return isoUtc;
    return new Intl.DateTimeFormat('en-US', {
      timeZone: tz || 'UTC',
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
      timeZoneName: 'short',
    }).format(d);
  } catch (_e) {
    return isoUtc;
  }
}
