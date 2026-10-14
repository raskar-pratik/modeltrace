/**
 * Screen 1: Overview
 * Displays actual latest persisted audit result, execution state vs statistical verdict distinction,
 * method/scope, valid p-value & CvM W^2, sample/attempt counts, real-data SVG rank charts,
 * recent audit history, meaningful empty state, and visible DEMO badges.
 */

import React from 'react';
import { formatTimestamp } from '../api/client';
import {
  DemoBadge,
  EmpiricalCdfChart,
  EpistemicNoticeBar,
  ExecutionStateBadge,
  PromptScoreShiftChart,
  ScoreScopeBadge,
  VerdictBadge,
} from '../components/BrandAndUI';
import {
  AppSettings,
  Audit,
  AuditDetailBundle,
  SystemCapabilities,
} from '../types/api';

export function OverviewScreen(props: {
  audits: Audit[];
  latestDetail: AuditDetailBundle | null;
  capabilities: SystemCapabilities | null;
  settings: AppSettings | null;
  onNavigate: (screen: string, auditId?: string) => void;
  onRefresh: () => void;
}) {
  const { audits, latestDetail, capabilities, settings, onNavigate, onRefresh } = props;
  const tz = settings?.display_timezone || 'UTC';
  const latest = latestDetail?.audit || audits[0] || null;
  const summary = latest?.summary || null;
  const primary = summary?.primary_test || null;
  const uncertainty = summary?.uncertainty || null;
  const attemptAccounting = summary?.attempt_accounting || null;
  const promptScores = latestDetail?.prompt_scores || [];

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">WORKBENCH OVERVIEW &middot; STATISTICAL VERIFICATION</div>
          <h1 className="editorial-title">Know what your API is serving.</h1>
          <p className="editorial-subtitle">
            Black-box and gray-box statistical auditing of open-weight model APIs against a trusted reference under frozen prompt suites and sampling configurations.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn"
            onClick={onRefresh}
            data-testid="overview-refresh-btn"
          >
            Refresh State
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onNavigate('new-audit')}
            data-testid="overview-new-audit-btn"
          >
            Configure New Audit
          </button>
        </div>
      </div>

      <EpistemicNoticeBar />

      {!latest ? (
        <div className="bento-card" style={{ padding: '40px 28px', textAlign: 'center', marginBottom: '24px' }}>
          <div className="eyebrow">NO PERSISTED AUDITS YET</div>
          <h2 className="card-title" style={{ marginBottom: '10px' }}>
            Start with a Deliberate Audit or Seeded Synthetic DEMO
          </h2>
          <p style={{ color: '#787774', maxWidth: '580px', margin: '0 auto 20px' }}>
            ModelTrace never triggers paid inference calls on page load. Configure an audit in the New Audit screen to preview attempt budgets and freeze a manifest before execution.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onNavigate('new-audit')}
            >
              Open New Audit Builder
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => onNavigate('profiles')}
            >
              Inspect Endpoint Profiles
            </button>
          </div>
        </div>
      ) : (
        <div className="bento-grid">
          {/* Hero Latest Audit Result Card */}
          <div className="bento-card span-8">
            <div className="card-label">
              <span>LATEST PERSISTED AUDIT RESULT</span>
              <span className="mono">{latest.id}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
              <VerdictBadge verdict={latest.verdict} />
              <ExecutionStateBadge state={latest.execution_state} />
              <DemoBadge isDemo={latest.is_demo} />
              <ScoreScopeBadge scoreMode={latest.score_mode} />
            </div>
            <h2 className="card-title" data-testid="overview-latest-audit-name">
              {latest.name}
            </h2>
            <p style={{ fontSize: '14px', color: '#2F3437', marginBottom: '16px' }} data-testid="overview-verdict-reason">
              {latest.verdict_reason}
            </p>
            <div
              style={{
                borderTop: '1px solid #EAEAEA',
                paddingTop: '14px',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '18px',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div className="mono" style={{ fontSize: '12px', color: '#787774' }}>
                Method: {primary?.method || 'cramer_von_mises_randomized_rank_mc'} &middot; Created:{' '}
                {formatTimestamp(latest.created_at, tz)}
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate('audits', latest.id)}
                  data-testid="overview-inspect-audit-btn"
                >
                  Audit Detail &amp; Events
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate('evidence', latest.id)}
                  data-testid="overview-inspect-evidence-btn"
                >
                  Inspect Evidence
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onNavigate('reports', latest.id)}
                >
                  Export Report
                </button>
              </div>
            </div>
          </div>

          {/* Statistical Summary Card */}
          <div className="bento-card span-4">
            <div className="card-label">
              <span>CRAM&Eacute;R&ndash;VON MISES NULL TEST</span>
              <span className="mono">&alpha; = {latest.alpha.toFixed(4)}</span>
            </div>
            <div className="metric-value" data-testid="overview-latest-pvalue">
              {primary?.p_value != null ? `p = ${Number(primary.p_value).toFixed(5)}` : 'p = N/A (Incomplete)'}
            </div>
            <div className="mono" style={{ color: '#787774', fontSize: '12.5px', marginBottom: '12px' }}>
              W&sup2; Statistic: {primary?.cvm_statistic != null ? Number(primary.cvm_statistic).toFixed(4) : '—'}
              <br />
              MC Simulations: B = {latest.mc_simulations} (Exceedances: {primary?.exceedances ?? '—'})
              <br />
              Mean Score Shift:{' '}
              {uncertainty?.mean_score_shift != null
                ? `${Number(uncertainty.mean_score_shift) >= 0 ? '+' : ''}${Number(
                    uncertainty.mean_score_shift
                  ).toFixed(4)}`
                : '—'}
            </div>
            <div
              style={{
                borderTop: '1px solid #EAEAEA',
                paddingTop: '10px',
                fontSize: '12px',
                color: '#787774',
              }}
            >
              <strong>Execution vs. Verdict:</strong> Execution state ({latest.execution_state}) tracks job completion; verdict ({latest.verdict}) reflects finite-sample inference.
            </div>
          </div>

          {/* Sample & Attempt Budget Accounting */}
          <div className="bento-card span-4">
            <div className="card-label">
              <span>SAMPLE &amp; ATTEMPT ACCOUNTING</span>
              <span className="mono">n={latest.n_prompts}, m={latest.m_reference}</span>
            </div>
            <div className="mono" style={{ fontSize: '13px', lineHeight: 1.75 }}>
              <div>
                Reference Attempts:{' '}
                <strong>
                  {attemptAccounting?.reference_attempts_used ?? latest.n_prompts * latest.m_reference}
                </strong>{' '}
                / {latest.max_reference_attempts} cap
              </div>
              <div>
                Target Attempts:{' '}
                <strong>{attemptAccounting?.target_attempts_used ?? latest.n_prompts}</strong> /{' '}
                {latest.max_target_attempts} cap
              </div>
              <div>
                Total Attempts Used:{' '}
                <strong>
                  {attemptAccounting?.total_attempts_used ??
                    latest.n_prompts * (latest.m_reference + 1)}
                </strong>{' '}
                / {latest.max_total_attempts} cap
              </div>
              <div>
                Uncertain Prior Attempts: <strong>{latest.uncertain_prior_attempts}</strong>
              </div>
              <div>
                Local Scoring Compute:{' '}
                <strong>{Number(latest.local_scoring_compute_ms || 0).toFixed(2)} ms</strong>
              </div>
            </div>
          </div>

          {/* Real-data Chart 1: Empirical CDF of z_i */}
          <div className="bento-card span-4">
            <div className="card-label">
              <span>EMPIRICAL CDF OF RANKS z_i</span>
              <span className="mono">UNIFORM(0,1) NULL</span>
            </div>
            <EmpiricalCdfChart
              promptScores={promptScores}
              cvmStatistic={primary?.cvm_statistic}
              pValue={primary?.p_value}
            />
          </div>

          {/* Real-data Chart 2: Per-Prompt Score Shift */}
          <div className="bento-card span-4">
            <div className="card-label">
              <span>PER-PROMPT SCORE SHIFT</span>
              <span className="mono">{latest.score_mode}</span>
            </div>
            <PromptScoreShiftChart promptScores={promptScores} />
          </div>
        </div>
      )}

      {/* System Capability & Recent History Row */}
      <div className="bento-grid">
        <div className="bento-card span-8">
          <div className="card-label">
            <span>RECENT PERSISTED AUDITS ({audits.length})</span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onNavigate('audits')}
            >
              View Full History
            </button>
          </div>
          {audits.length === 0 ? (
            <div style={{ color: '#787774', fontSize: '13.5px' }}>
              No audits recorded yet.
            </div>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Audit</th>
                    <th>Mode &amp; Scope</th>
                    <th>Execution State</th>
                    <th>Verdict</th>
                    <th className="num-col">CvM p-value</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {audits.slice(0, 6).map((a) => {
                    const pVal = a.summary?.primary_test?.p_value;
                    return (
                      <tr key={a.id}>
                        <td>
                          <div style={{ fontWeight: 600, color: '#111111' }}>{a.name}</div>
                          <div className="mono" style={{ color: '#787774', fontSize: '11.5px' }}>
                            {a.id} &middot; n={a.n_prompts}, m={a.m_reference}
                          </div>
                        </td>
                        <td>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'flex-start' }}>
                            <DemoBadge isDemo={a.is_demo} />
                            <ScoreScopeBadge scoreMode={a.score_mode} />
                          </div>
                        </td>
                        <td>
                          <ExecutionStateBadge state={a.execution_state} />
                        </td>
                        <td>
                          <VerdictBadge verdict={a.verdict} />
                        </td>
                        <td className="num-col">
                          {pVal != null ? Number(pVal).toFixed(5) : '—'}
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '6px' }}>
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => onNavigate('audits', a.id)}
                            >
                              Detail
                            </button>
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => onNavigate('evidence', a.id)}
                            >
                              Evidence
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="bento-card span-4">
          <div className="card-label">
            <span>RUNTIME &amp; ADAPTER CAPABILITIES</span>
            <span className="badge badge-sage">LOCALHOST</span>
          </div>
          <div className="mono" style={{ fontSize: '12.5px', lineHeight: 1.75 }}>
            <div>
              Python: <strong>{capabilities?.runtime.python_version || '3.11'}</strong> &middot; SQLite:{' '}
              <strong>{capabilities?.runtime.sqlite_version || '3.40'}</strong>
            </div>
            <div>
              Adapter A (Synthetic DEMO):{' '}
              <span className="badge badge-sage">AVAILABLE</span>
            </div>
            <div>
              Adapter B (OpenAI HTTP):{' '}
              <span className="badge badge-sage">AVAILABLE</span>
            </div>
            <div>
              Adapter C (Local Transformers):{' '}
              {capabilities?.adapters.local_transformers.available ? (
                <span className="badge badge-sage">AVAILABLE</span>
              ) : (
                <span className="badge badge-apricot">
                  UNAVAILABLE (Missing: {(capabilities?.adapters.local_transformers.missing_dependencies || []).join(', ') || 'weights'})
                </span>
              )}
            </div>
            <div style={{ marginTop: '8px', color: '#787774' }}>
              Secret Env Prefix: <code>MODELTRACE_PROVIDER_*</code>
              <br />
              Display Timezone: <strong>{tz}</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
