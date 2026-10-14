/**
 * Screen 6: Audit History & Detail (queued/running/completed/failed/cancelled/interrupted states,
 *           live polling/events, cooperative cancellation, explicit conservative resume)
 * Screen 7: Evidence Explorer (stored target/reference responses subject to retention,
 *           true scores/rank/observed-distribution charts, prompt-family counts, assumptions, calibration coverage)
 * Screen 8: Reports & Settings (actual JSON/CSV/HTML downloads, print stylesheet for Save as PDF,
 *           persistent defaults, retention, display timezone, operator network guidance, confirmed evidence purge)
 */

import React, { useEffect, useState } from 'react';
import { apiClient, formatTimestamp } from '../api/client';
import {
  DemoBadge,
  EmpiricalCdfChart,
  ExecutionStateBadge,
  PromptScoreShiftChart,
  ScoreScopeBadge,
  VerdictBadge,
} from '../components/BrandAndUI';
import {
  AppSettings,
  Audit,
  AuditDetailBundle,
  AuditEvent,
  AuditProgress,
  SystemCapabilities,
} from '../types/api';

export function AuditsScreen(props: {
  audits: Audit[];
  selectedAuditId: string | null;
  settings: AppSettings | null;
  onSelectAudit: (auditId: string) => void;
  onNavigate: (screen: string, auditId?: string) => void;
  onRefresh: () => void;
}) {
  const { audits, selectedAuditId, settings, onSelectAudit, onNavigate, onRefresh } = props;
  const tz = settings?.display_timezone || 'UTC';

  const activeId = selectedAuditId || audits[0]?.id || null;
  const [stateFilter, setStateFilter] = useState('ALL');
  const [detail, setDetail] = useState<AuditDetailBundle | null>(null);
  const [progress, setProgress] = useState<AuditProgress | null>(null);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  const loadSelectedAudit = async (id: string) => {
    try {
      const [det, prog, ev] = await Promise.all([
        apiClient.getAuditDetail(id, true),
        apiClient.getAuditProgress(id),
        apiClient.getAuditEvents(id),
      ]);
      setDetail(det);
      setProgress(prog);
      setEvents(ev.events || []);
      const listMatch = audits.find((a) => a.id === id);
      if (
        listMatch &&
        (listMatch.execution_state !== det.audit.execution_state ||
          listMatch.verdict !== det.audit.verdict)
      ) {
        onRefresh();
      }
    } catch (err: any) {
      setActionError(err.detail || err.message);
    }
  };

  useEffect(() => {
    if (!activeId) return;
    loadSelectedAudit(activeId);
    const timer = setInterval(() => {
      loadSelectedAudit(activeId);
    }, 1500);
    return () => clearInterval(timer);
  }, [activeId]);

  const handleStart = async (id: string) => {
    setActionError(null);
    setActionNotice(null);
    try {
      await apiClient.startAudit(id, false);
      setActionNotice(`Started audit ${id}.`);
      await loadSelectedAudit(id);
      onRefresh();
    } catch (err: any) {
      setActionError(err.detail || err.message);
    }
  };

  const handleCancel = async (id: string) => {
    setActionError(null);
    setActionNotice(null);
    try {
      await apiClient.cancelAudit(id);
      setActionNotice(`Requested cancellation for audit ${id}.`);
      await loadSelectedAudit(id);
      onRefresh();
    } catch (err: any) {
      setActionError(err.detail || err.message);
    }
  };

  const handleResume = async (id: string) => {
    setActionError(null);
    setActionNotice(null);
    try {
      await apiClient.resumeAudit(id, false);
      setActionNotice(
        `Explicitly resumed interrupted audit ${id}. Completed responses are reused; uncertain prior attempts remain counted conservatively.`
      );
      await loadSelectedAudit(id);
      onRefresh();
    } catch (err: any) {
      setActionError(err.detail || err.message);
    }
  };

  const filteredAudits = audits.filter(
    (a) => stateFilter === 'ALL' || a.execution_state === stateFilter
  );
  const activeAudit = detail?.audit || audits.find((a) => a.id === activeId) || null;

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">AUDIT LIFECYCLE &middot; PROGRESS POLLING &amp; RECOVERY</div>
          <h1 className="editorial-title">Audit History &amp; Execution Detail</h1>
          <p className="editorial-subtitle">
            Monitor queued, running, completed, failed, cancelled, and interrupted jobs. Failed or incomplete execution resolves to <code>INCONCLUSIVE</code>—never fabricated model inconsistency.
          </p>
        </div>
      </div>

      {actionError && (
        <div className="notice-banner notice-banner-error" role="alert">
          <div>{actionError}</div>
          <button type="button" className="btn btn-sm" onClick={() => setActionError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {actionNotice && (
        <div className="notice-banner notice-banner-info" role="status" data-testid="audit-action-notice">
          <div>{actionNotice}</div>
          <button type="button" className="btn btn-sm" onClick={() => setActionNotice(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Left: Audit History List */}
        <div className="bento-card span-5">
          <div className="card-label">
            <span>PERSISTED AUDIT JOBS ({filteredAudits.length})</span>
            <span className="mono">RELOAD-SAFE</span>
          </div>
          <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '12px' }}>
            {['ALL', 'CREATED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'].map(
              (st) => (
                <button
                  key={st}
                  type="button"
                  className={`btn btn-sm ${stateFilter === st ? 'btn-primary' : ''}`}
                  onClick={() => setStateFilter(st)}
                >
                  {st}
                </button>
              )
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {filteredAudits.map((a) => {
              const isSelected = a.id === activeId;
              const effectiveState =
                isSelected && detail?.audit ? detail.audit.execution_state : a.execution_state;
              const effectiveVerdict =
                isSelected && detail?.audit ? detail.audit.verdict : a.verdict;
              return (
                <div
                  key={a.id}
                  onClick={() => onSelectAudit(a.id)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '6px',
                    border: isSelected ? '1px solid #111111' : '1px solid #EAEAEA',
                    backgroundColor: isSelected ? '#F7F6F3' : '#FFFFFF',
                    cursor: 'pointer',
                  }}
                  data-testid={`audit-list-item-${a.id}`}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginBottom: '4px' }}>
                    <strong style={{ color: '#111111', fontSize: '13.5px' }}>{a.name}</strong>
                    <span className="mono" style={{ color: '#787774', fontSize: '11.5px' }}>
                      {a.id}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '4px' }}>
                    <ExecutionStateBadge state={effectiveState} />
                    <VerdictBadge verdict={effectiveVerdict} />
                    <DemoBadge isDemo={a.is_demo} />
                  </div>
                  <div className="mono" style={{ fontSize: '11px', color: '#787774' }}>
                    {formatTimestamp(a.created_at, tz)} &middot; n={a.n_prompts}, m={a.m_reference}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Selected Audit Inspector, Progress & Event Log */}
        <div className="bento-card span-7">
          {!activeAudit ? (
            <div style={{ color: '#787774' }}>Select an audit job to inspect details.</div>
          ) : (
            <div>
              <div className="card-label">
                <span>AUDIT INSPECTOR &middot; {activeAudit.id}</span>
                <span className="mono">MANIFEST SHA-256: {activeAudit.manifest_hash.slice(0, 14)}...</span>
              </div>

              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
                <VerdictBadge verdict={activeAudit.verdict} />
                <ExecutionStateBadge state={activeAudit.execution_state} />
                <DemoBadge isDemo={activeAudit.is_demo} />
                <ScoreScopeBadge scoreMode={activeAudit.score_mode} />
              </div>

              <h2 className="card-title">{activeAudit.name}</h2>
              <p style={{ fontSize: '13.5px', color: '#2F3437', marginBottom: '14px' }} data-testid="audit-detail-verdict-reason">
                {activeAudit.verdict_reason}
              </p>

              {/* Lifecycle Control Buttons */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '18px' }}>
                {activeAudit.execution_state === 'CREATED' && (
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => handleStart(activeAudit.id)}
                    data-testid="audit-detail-start-btn"
                  >
                    Deliberate Start Audit
                  </button>
                )}
                {(activeAudit.execution_state === 'CREATED' ||
                  activeAudit.execution_state === 'RUNNING') && (
                  <button
                    type="button"
                    className="btn btn-danger btn-sm"
                    onClick={() => handleCancel(activeAudit.id)}
                    data-testid="audit-detail-cancel-btn"
                  >
                    Cancel Audit
                  </button>
                )}
                {activeAudit.execution_state === 'INTERRUPTED' && (
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => handleResume(activeAudit.id)}
                    data-testid="audit-detail-resume-btn"
                  >
                    Explicit Resume (Conservative Accounting)
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate('evidence', activeAudit.id)}
                >
                  Open Evidence Explorer
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate('reports', activeAudit.id)}
                >
                  Export Report
                </button>
              </div>

              {/* Live Progress Bar & Attempt Counters */}
              {progress && (
                <div
                  className="bento-card bento-card-muted"
                  style={{ padding: '14px', marginBottom: '16px' }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span className="mono" style={{ fontWeight: 600 }}>
                      Progress: {progress.completed_samples} / {progress.nominal_total_samples} samples (
                      {progress.completed_prompts}/{progress.total_prompts} prompts)
                    </span>
                    <span className="mono">{(progress.progress_fraction * 100).toFixed(0)}%</span>
                  </div>
                  <div className="progress-track" style={{ marginBottom: '10px' }}>
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.min(100, progress.progress_fraction * 100)}%` }}
                    />
                  </div>
                  <div className="mono" style={{ fontSize: '12px', color: '#787774' }}>
                    Attempts Used: Ref {progress.attempts.reference_attempts_used}/
                    {progress.attempts.max_reference_attempts} &middot; Target{' '}
                    {progress.attempts.target_attempts_used}/
                    {progress.attempts.max_target_attempts} &middot; Total{' '}
                    {progress.attempts.total_attempts_used}/
                    {progress.attempts.max_total_attempts} &middot; Failed:{' '}
                    {progress.attempts.failed_attempts} &middot; Uncertain Prior:{' '}
                    {progress.uncertain_prior_attempts} &middot; Resumes: {progress.resume_count}
                  </div>
                </div>
              )}

              {/* Redacted Event Log */}
              <div className="card-label" style={{ marginTop: '14px' }}>
                <span>APPEND-ONLY REDACTED EVENT LOG ({events.length})</span>
                <span className="mono">UTC STORED</span>
              </div>
              <div className="table-scroll" style={{ maxHeight: '260px' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Event Type</th>
                      <th>Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((ev) => (
                      <tr key={ev.id}>
                        <td className="mono" style={{ whiteSpace: 'nowrap', fontSize: '11.5px' }}>
                          {formatTimestamp(ev.created_at, tz)}
                        </td>
                        <td>
                          <span
                            className={`badge ${
                              ev.level === 'ERROR'
                                ? 'badge-red'
                                : ev.level === 'WARNING'
                                ? 'badge-apricot'
                                : 'badge-neutral'
                            }`}
                          >
                            {ev.event_type}
                          </span>
                        </td>
                        <td style={{ fontSize: '12.5px' }}>{ev.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function EvidenceScreen(props: {
  audits: Audit[];
  selectedAuditId: string | null;
  settings: AppSettings | null;
  onSelectAudit: (auditId: string) => void;
}) {
  const { audits, selectedAuditId, onSelectAudit } = props;
  const activeId = selectedAuditId || audits[0]?.id || null;
  const [bundle, setBundle] = useState<AuditDetailBundle | null>(null);
  const [selectedPromptId, setSelectedPromptId] = useState<string>('ALL');

  useEffect(() => {
    if (!activeId) return;
    apiClient.getAuditDetail(activeId, true).then((b) => {
      setBundle(b);
    });
  }, [activeId]);

  const audit = bundle?.audit || null;
  const scores = bundle?.prompt_scores || [];
  const responses = bundle?.responses || [];
  const summary = audit?.summary || null;
  const primary = summary?.primary_test || null;
  const uncertainty = summary?.uncertainty || null;
  const calEval = summary?.calibration_evaluation || null;
  const byFamily = uncertainty?.by_family_descriptive || {};
  const holmTests = summary?.holm_corrected_tests || [];

  const filteredResponses = responses.filter(
    (r) => selectedPromptId === 'ALL' || r.prompt_id === selectedPromptId
  );

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">EMPIRICAL EVIDENCE &middot; STORED RESPONSES &amp; RANK DISTRIBUTIONS</div>
          <h1 className="editorial-title">Evidence &amp; Diagnostic Explorer</h1>
          <p className="editorial-subtitle">
            Inspect stored reference and target responses, exact finite-sample rank transformations <code>z_i = (a_i + J_i + U_i) / (m + 1)</code>, prompt-family breakdowns, and confounder assumptions. Charts render exclusively from persisted audit data.
          </p>
        </div>
        <div>
          <label className="sr-only" htmlFor="evidence-audit-select">
            Select Audit
          </label>
          <select
            id="evidence-audit-select"
            className="select mono"
            value={activeId || ''}
            onChange={(e: any) => onSelectAudit(e.target.value)}
            data-testid="evidence-audit-select"
          >
            {audits.map((a) => {
              const vLabel =
                a.id === audit?.id ? audit.verdict : a.verdict;
              return (
                <option key={a.id} value={a.id}>
                  {a.name} ({a.id} &middot; {vLabel})
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {!audit ? (
        <div className="bento-card">No audit selected or available.</div>
      ) : (
        <div>
          <div className="bento-grid">
            {/* Empirical CDF Chart */}
            <div className="bento-card span-6">
              <div className="card-label">
                <span>OBSERVED RANK DISTRIBUTION F_n(z) VS. UNIFORM(0,1)</span>
                <DemoBadge isDemo={audit.is_demo} />
              </div>
              <EmpiricalCdfChart
                promptScores={scores}
                cvmStatistic={primary?.cvm_statistic}
                pValue={primary?.p_value}
              />
            </div>

            {/* Per-Prompt Score Shift Chart */}
            <div className="bento-card span-6">
              <div className="card-label">
                <span>PER-PROMPT SCORE SHIFT (TARGET &minus; MEAN REFERENCE)</span>
                <ScoreScopeBadge scoreMode={audit.score_mode} />
              </div>
              <PromptScoreShiftChart promptScores={scores} />
            </div>

            {/* Prompt-Family Descriptive Breakdown */}
            <div className="bento-card span-6">
              <div className="card-label">
                <span>PROMPT-FAMILY BREAKDOWN</span>
                <span className="badge badge-neutral">DESCRIPTIVE_ONLY</span>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Task Family</th>
                      <th className="num-col">Prompts</th>
                      <th className="num-col">Mean Shift</th>
                      <th className="num-col">Median Shift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(byFamily).map(([fam, info]: [string, any]) => (
                      <tr key={fam}>
                        <td>
                          <span className="badge badge-neutral">{fam}</span>
                        </td>
                        <td className="num-col">{info.n_prompts}</td>
                        <td className="num-col">{Number(info.mean_score_shift).toFixed(4)}</td>
                        <td className="num-col">{Number(info.median_score_shift).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {uncertainty && (
                <div className="mono" style={{ fontSize: '12px', color: '#787774', marginTop: '10px' }}>
                  Prompt Bootstrap 95% CI: [{Number(uncertainty.prompt_bootstrap_ci_95?.[0] || 0).toFixed(4)},{' '}
                  {Number(uncertainty.prompt_bootstrap_ci_95?.[1] || 0).toFixed(4)}] &middot; Family Cluster 95% CI: [
                  {Number(uncertainty.family_bootstrap_ci_95?.[0] || 0).toFixed(4)},{' '}
                  {Number(uncertainty.family_bootstrap_ci_95?.[1] || 0).toFixed(4)}]
                </div>
              )}
            </div>

            {/* Calibration Coverage & Confounders */}
            <div className="bento-card span-6">
              <div className="card-label">
                <span>CALIBRATION COVERAGE &amp; ASSUMPTIONS</span>
                <span className={calEval?.compatible ? 'badge badge-sage' : 'badge badge-apricot'}>
                  {calEval?.compatible ? 'CALIBRATED' : 'UNCALIBRATED'}
                </span>
              </div>
              <p style={{ fontSize: '13.5px', color: '#2F3437', marginBottom: '10px' }}>
                {calEval?.reason || 'No calibration attached.'}
              </p>
              <div className="mono" style={{ fontSize: '12px', color: '#787774', lineHeight: 1.65 }}>
                <div>
                  &bull; Declared Model Headers Observed (Untrusted):{' '}
                  <strong>
                    {(summary?.descriptive_diagnostics?.observed_declared_models_untrusted || []).join(', ') || 'None'}
                  </strong>
                </div>
                <div>
                  &bull; Null Hypothesis: Prompt-conditional exchangeability between target and reference scores.
                </div>
                <div>
                  &bull; Confounders: Chat template differences, hidden system prompts, moderation filters, or GPU non-determinism can shift ranks without weight substitution.
                </div>
              </div>
              {holmTests.length > 0 && (
                <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #EAEAEA' }}>
                  <div className="mono" style={{ fontSize: '11.5px', fontWeight: 600 }}>
                    Holm Step-Down Corrected Inferential Tests:
                  </div>
                  {holmTests.map((ht: any) => (
                    <div key={ht.name} className="mono" style={{ fontSize: '11.5px', color: '#2F3437' }}>
                      {ht.name}: raw p={Number(ht.raw_p_value).toFixed(5)}, Holm adj p=
                      {Number(ht.holm_adjusted_p_value).toFixed(5)} (
                      {ht.rejected_at_alpha ? 'REJECTED' : 'NOT REJECTED'})
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Per-Prompt Rank Table */}
            <div className="bento-card span-12">
              <div className="card-label">
                <span>FINITE-SAMPLE RANK COORDINATES z_i = (a_i + J_i + U_i) / (m + 1)</span>
                <span className="mono">{scores.length} PROMPTS</span>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Prompt ID</th>
                      <th>Family</th>
                      <th className="num-col">Target Score</th>
                      <th className="num-col">Mean Ref Score</th>
                      <th className="num-col">Shift</th>
                      <th className="num-col">a_below / b_ties</th>
                      <th className="num-col">J_draw / U_draw</th>
                      <th className="num-col">Uniform z_i</th>
                      <th className="num-col">Norm LL (Descriptive)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scores.map((s) => {
                      const meanRef =
                        s.reference_scores.reduce((a, b) => a + b, 0) /
                        Math.max(1, s.reference_scores.length);
                      return (
                        <tr key={s.id}>
                          <td className="mono">
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => setSelectedPromptId(s.prompt_id)}
                            >
                              {s.prompt_id}
                            </button>
                          </td>
                          <td>
                            <span className="badge badge-neutral">{s.family}</span>
                          </td>
                          <td className="num-col">{Number(s.target_score).toFixed(4)}</td>
                          <td className="num-col">{meanRef.toFixed(4)}</td>
                          <td className="num-col">
                            {Number(s.score_shift) >= 0 ? '+' : ''}
                            {Number(s.score_shift).toFixed(4)}
                          </td>
                          <td className="num-col">
                            {s.a_below} / {s.b_ties}
                          </td>
                          <td className="num-col">
                            {s.j_draw} / {Number(s.u_draw).toFixed(3)}
                          </td>
                          <td className="num-col" style={{ fontWeight: 600 }}>
                            {Number(s.z_uniform).toFixed(4)}
                          </td>
                          <td className="num-col">
                            {s.target_norm_ll != null ? Number(s.target_norm_ll).toFixed(4) : 'N/A (Unsupported)'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Stored Target & Reference Responses */}
            <div className="bento-card span-12">
              <div className="card-label">
                <span>
                  STORED RAW RESPONSES ({filteredResponses.length} SHOWN / {responses.length} TOTAL)
                </span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  {selectedPromptId !== 'ALL' && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setSelectedPromptId('ALL')}
                    >
                      Show All Prompts
                    </button>
                  )}
                  <span className="mono">SUBJECT TO RETENTION POLICY</span>
                </div>
              </div>
              {responses.length === 0 ? (
                <div style={{ color: '#787774', fontSize: '13.5px' }}>
                  No raw response text stored for this audit (either purged under retention policy while preserving the manifest/scores, or execution did not complete).
                </div>
              ) : (
                <div className="table-scroll" style={{ maxHeight: '340px' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Prompt &amp; Role</th>
                        <th>Response Text Preview</th>
                        <th className="num-col">Tokens / Unicode Len</th>
                        <th className="num-col">Score</th>
                        <th>Integrity SHA-256</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredResponses.slice(0, 45).map((r) => (
                        <tr key={r.id}>
                          <td className="mono" style={{ whiteSpace: 'nowrap' }}>
                            <strong>{r.prompt_id}</strong> &middot;{' '}
                            <span
                              className={`badge ${
                                r.role === 'target' ? 'badge-blue' : 'badge-neutral'
                              }`}
                            >
                              {r.role} #{r.sample_index}
                            </span>
                          </td>
                          <td className="mono" style={{ fontSize: '12px', maxWidth: '480px' }}>
                            {r.response_text}
                          </td>
                          <td className="num-col">
                            {r.token_count} tok / {r.unicode_length} chars
                          </td>
                          <td className="num-col">{Number(r.raw_score).toFixed(4)}</td>
                          <td className="mono" style={{ fontSize: '11px', color: '#787774' }}>
                            {r.response_hash.slice(0, 14)}...
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function ReportsAndSettingsScreen(props: {
  audits: Audit[];
  selectedAuditId: string | null;
  settings: AppSettings | null;
  capabilities: SystemCapabilities | null;
  onSelectAudit: (auditId: string) => void;
  onRefresh: () => void;
}) {
  const { audits, selectedAuditId, settings, onSelectAudit, onRefresh } = props;
  const activeId = selectedAuditId || audits[0]?.id || null;
  const activeAudit = audits.find((a) => a.id === activeId) || null;

  const [outboundOrigins, setOutboundOrigins] = useState(
    (settings?.allowed_outbound_origins || []).join(', ')
  );
  const [localOrigins, setLocalOrigins] = useState(
    (settings?.allowed_local_inference_origins || []).join(', ')
  );
  const [retentionDays, setRetentionDays] = useState(String(settings?.retention_days || 90));
  const [displayTz, setDisplayTz] = useState(settings?.display_timezone || 'UTC');
  const [defN, setDefN] = useState(String(settings?.default_n_prompts || 20));
  const [defM, setDefM] = useState(String(settings?.default_m_reference || 8));
  const [defAlpha, setDefAlpha] = useState(String(settings?.default_alpha || 0.01));
  const [refRate, setRefRate] = useState(
    settings?.pricing_assumptions?.reference_cost_per_1k_tokens_usd != null
      ? String(settings.pricing_assumptions.reference_cost_per_1k_tokens_usd)
      : ''
  );
  const [tgtRate, setTgtRate] = useState(
    settings?.pricing_assumptions?.target_cost_per_1k_tokens_usd != null
      ? String(settings.pricing_assumptions.target_cost_per_1k_tokens_usd)
      : ''
  );

  const [confirmPurgeModal, setConfirmPurgeModal] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!settings) return;
    setOutboundOrigins((settings.allowed_outbound_origins || []).join(', '));
    setLocalOrigins((settings.allowed_local_inference_origins || []).join(', '));
    setRetentionDays(String(settings.retention_days));
    setDisplayTz(settings.display_timezone || 'UTC');
    setDefN(String(settings.default_n_prompts || 20));
    setDefM(String(settings.default_m_reference || 8));
    setDefAlpha(String(settings.default_alpha || 0.01));
  }, [settings]);

  const handleSaveSettings = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setErrorMsg(null);
    setStatusMsg(null);
    try {
      await apiClient.updateSettings({
        allowed_outbound_origins: outboundOrigins
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        allowed_local_inference_origins: localOrigins
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        retention_days: Number(retentionDays) || 90,
        display_timezone: displayTz.trim() || 'UTC',
        default_n_prompts: Number(defN) || 20,
        default_m_reference: Number(defM) || 8,
        default_alpha: Number(defAlpha) || 0.01,
        pricing_assumptions: {
          reference_cost_per_1k_tokens_usd: refRate.trim() !== '' ? Number(refRate) : null,
          target_cost_per_1k_tokens_usd: tgtRate.trim() !== '' ? Number(tgtRate) : null,
          estimated_prompt_tokens_per_call: 64,
        },
      });
      setStatusMsg('Saved persistent operator settings, allowlists, display timezone, and pricing assumptions.');
      onRefresh();
    } catch (err: any) {
      setErrorMsg(err.detail || err.message);
    }
  };

  const handleConfirmedEvidencePurge = async () => {
    if (!activeId) return;
    setErrorMsg(null);
    setStatusMsg(null);
    try {
      const res = await apiClient.purgeAuditEvidence(activeId);
      setConfirmPurgeModal(false);
      setStatusMsg(
        `Purged ${res.purged_response_count} stored raw response(s) for audit ${res.audit_id}. Historical manifest and statistical summary are preserved.`
      );
      onRefresh();
    } catch (err: any) {
      setConfirmPurgeModal(false);
      setErrorMsg(err.detail || err.message);
    }
  };

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">REPRODUCIBLE EXPORTS &amp; OPERATOR SETTINGS</div>
          <h1 className="editorial-title">Reports, Network Allowlists &amp; Retention</h1>
          <p className="editorial-subtitle">
            Download SHA-256 integrity-hashed JSON, CSV, and self-contained HTML reports (or print to PDF via browser print stylesheet), configure exact-origin outbound allowlists, and manage response retention.
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="notice-banner notice-banner-error" role="alert">
          <div>{errorMsg}</div>
          <button type="button" className="btn btn-sm" onClick={() => setErrorMsg(null)}>
            Dismiss
          </button>
        </div>
      )}
      {statusMsg && (
        <div className="notice-banner notice-banner-info" role="status" data-testid="settings-status-banner">
          <div>{statusMsg}</div>
          <button type="button" className="btn btn-sm" onClick={() => setStatusMsg(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Report Exports Card */}
        <div className="bento-card span-6">
          <div className="card-label">
            <span>REPRODUCIBLE AUDIT REPORTS</span>
            <span className="mono">SECRETS REDACTED</span>
          </div>
          <div className="form-field" style={{ marginBottom: '14px' }}>
            <label className="form-label" htmlFor="rep-audit-select">
              Select Audit to Export
            </label>
            <select
              id="rep-audit-select"
              className="select mono"
              value={activeId || ''}
              onChange={(e: any) => onSelectAudit(e.target.value)}
            >
              {audits.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} ({a.id} &middot; {a.verdict})
                </option>
              ))}
            </select>
          </div>

          {activeAudit && (
            <div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
                <VerdictBadge verdict={activeAudit.verdict} />
                <ExecutionStateBadge state={activeAudit.execution_state} />
                <DemoBadge isDemo={activeAudit.is_demo} />
                <ScoreScopeBadge scoreMode={activeAudit.score_mode} />
              </div>
              <p style={{ fontSize: '13px', color: '#787774', marginBottom: '14px' }}>
                Exports reproduce stored database records verbatim, include DEMO and score-scope banners, record a SHA-256 integrity hash, and scrub all secrets.
              </p>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                <a
                  className="btn btn-primary"
                  href={`/api/audits/${encodeURIComponent(activeAudit.id)}/export?format=html`}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-testid="export-html-link"
                >
                  Download Self-Contained HTML
                </a>
                <a
                  className="btn"
                  href={`/api/audits/${encodeURIComponent(activeAudit.id)}/export?format=json`}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-testid="export-json-link"
                >
                  Download JSON Bundle
                </a>
                <a
                  className="btn"
                  href={`/api/audits/${encodeURIComponent(activeAudit.id)}/export?format=csv`}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-testid="export-csv-link"
                >
                  Download Prompt CSV
                </a>
                <button
                  type="button"
                  className="btn"
                  onClick={() => window.print()}
                  data-testid="print-pdf-btn"
                >
                  Print / Save as PDF
                </button>
              </div>

              <div
                style={{
                  borderTop: '1px solid #EAEAEA',
                  paddingTop: '14px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '10px',
                }}
              >
                <div style={{ fontSize: '12.5px', color: '#787774' }}>
                  <strong>Evidence Retention Control:</strong> Purge stored response text for this audit while preserving its immutable manifest &amp; rank summary.
                </div>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={() => setConfirmPurgeModal(true)}
                  data-testid="open-purge-modal-btn"
                >
                  Purge Stored Evidence...
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Persistent Operator Settings */}
        <div className="bento-card span-6">
          <div className="card-label">
            <span>OPERATOR SETTINGS, ALLOWLISTS &amp; PRICING</span>
            <span className="mono">PERSISTED IN SQLITE</span>
          </div>
          <form onSubmit={handleSaveSettings} className="form-grid">
            <div className="form-field span-12">
              <label className="form-label" htmlFor="set-outbound">
                Allowed Public Outbound Origins (exact <code>scheme://host:port</code>, comma-separated)
              </label>
              <input
                id="set-outbound"
                className="input mono"
                value={outboundOrigins}
                onChange={(e: any) => setOutboundOrigins(e.target.value)}
                data-testid="settings-outbound-input"
              />
            </div>

            <div className="form-field span-12">
              <label className="form-label" htmlFor="set-local">
                Allowed Local Inference Origins (explicit operator override for localhost/private inference servers)
              </label>
              <input
                id="set-local"
                className="input mono"
                value={localOrigins}
                onChange={(e: any) => setLocalOrigins(e.target.value)}
                data-testid="settings-local-origins-input"
              />
              <span className="form-hint">
                Cloud metadata (169.254.169.254) is unconditionally blocked even if listed here.
              </span>
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="set-tz">
                Display Timezone
              </label>
              <select
                id="set-tz"
                className="select mono"
                value={displayTz}
                onChange={(e: any) => setDisplayTz(e.target.value)}
                data-testid="settings-timezone-select"
              >
                <option value="UTC">UTC</option>
                <option value="Asia/Kolkata">Asia/Kolkata (IST)</option>
                <option value="America/New_York">America/New_York</option>
                <option value="America/Los_Angeles">America/Los_Angeles</option>
                <option value="Europe/London">Europe/London</option>
              </select>
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="set-ret">
                Retention Days
              </label>
              <input
                id="set-ret"
                type="number"
                min="1"
                max="3650"
                className="input mono"
                value={retentionDays}
                onChange={(e: any) => setRetentionDays(e.target.value)}
              />
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="set-def-n">
                Default n Prompts
              </label>
              <input
                id="set-def-n"
                type="number"
                min="1"
                max="200"
                className="input mono"
                value={defN}
                onChange={(e: any) => setDefN(e.target.value)}
              />
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="set-ref-rate">
                Reference Cost ($ / 1k tokens, optional)
              </label>
              <input
                id="set-ref-rate"
                type="number"
                step="0.0001"
                min="0"
                className="input mono"
                value={refRate}
                onChange={(e: any) => setRefRate(e.target.value)}
                placeholder="Leave blank if unknown"
                data-testid="settings-ref-rate-input"
              />
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="set-tgt-rate">
                Target Cost ($ / 1k tokens, optional)
              </label>
              <input
                id="set-tgt-rate"
                type="number"
                step="0.0001"
                min="0"
                className="input mono"
                value={tgtRate}
                onChange={(e: any) => setTgtRate(e.target.value)}
                placeholder="Leave blank if unknown"
                data-testid="settings-tgt-rate-input"
              />
            </div>

            <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="submit"
                className="btn btn-primary"
                data-testid="save-settings-btn"
              >
                Save Operator Settings
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Confirmation Modal for Purging Evidence */}
      {confirmPurgeModal && activeAudit && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="purge-modal-title">
          <div className="modal-card">
            <div className="eyebrow">CONFIRM EVIDENCE DELETION</div>
            <h2 id="purge-modal-title" className="card-title">
              Purge Stored Raw Responses for {activeAudit.id}?
            </h2>
            <p style={{ fontSize: '13.5px', color: '#2F3437', margin: '10px 0 18px' }}>
              This action permanently deletes stored raw target and reference response strings for{' '}
              <strong>{activeAudit.name}</strong> while preserving the immutable audit manifest, per-prompt rank coordinates (<code>z_i</code>), and statistical verdict summary.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button
                type="button"
                className="btn"
                onClick={() => setConfirmPurgeModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleConfirmedEvidencePurge}
                data-testid="confirm-purge-evidence-btn"
              >
                Confirm Purge Evidence
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
