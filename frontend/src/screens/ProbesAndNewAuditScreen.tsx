/**
 * Screen 4: Probe Library (search, family filter tags, JSON/JSONL import with validation,
 *           editing prompts creates new immutable revisions)
 * Screen 5: New Audit (choose profiles/probe revision, freeze settings, preview attempt budget,
 *           honest cost estimate where unknown cost != $0.00, separate clearly marked DEMO action)
 */

import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { DemoBadge, ScoreScopeBadge } from '../components/BrandAndUI';
import {
  AppSettings,
  AuditPreviewResponse,
  Calibration,
  ProbePromptItem,
  ProbeRevision,
  Profile,
  ScoreMode,
} from '../types/api';

export function ProbesScreen(props: {
  revisions: ProbeRevision[];
  onRefresh: () => void;
}) {
  const { revisions, onRefresh } = props;
  const [selectedRevId, setSelectedRevId] = useState<string>(
    revisions[0]?.id || 'rev_builtin_screening_v1'
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFamily, setSelectedFamily] = useState('ALL');

  // Import state
  const [importName, setImportName] = useState('');
  const [importFormat, setImportFormat] = useState<'json' | 'jsonl'>('jsonl');
  const [importContent, setImportContent] = useState('');

  // Edit-into-new-revision state
  const [editingPromptId, setEditingPromptId] = useState<string | null>(null);
  const [editPromptTitle, setEditPromptTitle] = useState('');
  const [editPromptText, setEditPromptText] = useState('');
  const [newRevisionName, setNewRevisionName] = useState('');

  const [probeError, setProbeError] = useState<string | null>(null);
  const [probeSuccess, setProbeSuccess] = useState<string | null>(null);

  const activeRev =
    revisions.find((r) => r.id === selectedRevId) || revisions[0] || null;
  const prompts = activeRev?.prompts || [];
  const families = ['ALL', ...(activeRev?.families || [])];

  const filteredPrompts = prompts.filter((p) => {
    if (selectedFamily !== 'ALL' && p.family !== selectedFamily) return false;
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      p.prompt_id.toLowerCase().includes(q) ||
      p.title.toLowerCase().includes(q) ||
      p.prompt_text.toLowerCase().includes(q)
    );
  });

  const handleImportProbes = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setProbeError(null);
    setProbeSuccess(null);
    try {
      const created = await apiClient.importProbes({
        name: importName.trim() || 'Imported Probe Revision',
        description: `Imported via UI (${importFormat.toUpperCase()})`,
        format: importFormat,
        raw_content: importContent,
      });
      setProbeSuccess(
        `Imported ${created.prompt_count} prompts into immutable revision '${created.name}' (${created.id}, SHA-256: ${created.content_hash.slice(0, 12)}...).`
      );
      setSelectedRevId(created.id);
      setImportName('');
      setImportContent('');
      onRefresh();
    } catch (err: any) {
      setProbeError(`Import validation error: ${err.detail || err.message}`);
    }
  };

  const beginEditPrompt = (p: ProbePromptItem) => {
    setEditingPromptId(p.prompt_id);
    setEditPromptTitle(p.title);
    setEditPromptText(p.prompt_text);
    setNewRevisionName(
      `${activeRev?.name || 'Suite'} (Rev ${(activeRev?.revision_number || 1) + 1})`
    );
    setProbeError(null);
    setProbeSuccess(null);
  };

  const handleSaveNewRevision = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!activeRev || !editingPromptId) return;
    setProbeError(null);
    setProbeSuccess(null);
    try {
      const updatedPrompts = (activeRev.prompts || []).map((item) =>
        item.prompt_id === editingPromptId
          ? {
              ...item,
              title: editPromptTitle.trim(),
              prompt_text: editPromptText.trim(),
            }
          : item
      );
      const created = await apiClient.createProbeRevision({
        name: newRevisionName.trim() || `${activeRev.name} (Edited)`,
        description: `Derived from immutable revision ${activeRev.id} (edited ${editingPromptId})`,
        parent_revision_id: activeRev.id,
        prompts: updatedPrompts,
      });
      setProbeSuccess(
        `Created new immutable revision #${created.revision_number} ('${created.name}', ID: ${created.id}). Prior revision ${activeRev.id} remains unchanged.`
      );
      setEditingPromptId(null);
      setSelectedRevId(created.id);
      onRefresh();
    } catch (err: any) {
      setProbeError(err.detail || err.message);
    }
  };

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">IMMUTABLE PROBE LIBRARY &middot; CONTENT-HASHED REVISIONS</div>
          <h1 className="editorial-title">Screening Probe Suites &amp; Revisions</h1>
          <p className="editorial-subtitle">
            Browse, filter, import, or edit multi-family evaluation prompts. All built-in and imported prompts are labeled <code>UNVALIDATED_FOR_IDENTITY_DISCRIMINATION</code>, and every edit creates a new immutable revision without rewriting audit history.
          </p>
        </div>
      </div>

      {probeError && (
        <div className="notice-banner notice-banner-error" role="alert" data-testid="probe-error-banner">
          <div>{probeError}</div>
          <button type="button" className="btn btn-sm" onClick={() => setProbeError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {probeSuccess && (
        <div className="notice-banner notice-banner-info" role="status" data-testid="probe-success-banner">
          <div>{probeSuccess}</div>
          <button type="button" className="btn btn-sm" onClick={() => setProbeSuccess(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Revision Selector & Search / Filter Controls */}
        <div className="bento-card span-8">
          <div className="card-label">
            <span>ACTIVE PROBE REVISION</span>
            <span className="badge badge-apricot">
              {activeRev?.validation_label || 'UNVALIDATED_FOR_IDENTITY_DISCRIMINATION'}
            </span>
          </div>

          <div className="form-grid" style={{ marginBottom: '16px' }}>
            <div className="form-field span-6">
              <label className="form-label" htmlFor="probe-rev-select">
                Select Immutable Revision ({revisions.length} stored)
              </label>
              <select
                id="probe-rev-select"
                className="select mono"
                value={activeRev?.id || ''}
                onChange={(e: any) => setSelectedRevId(e.target.value)}
                data-testid="probe-revision-select"
              >
                {revisions.map((r) => (
                  <option key={r.id} value={r.id}>
                    [Rev #{r.revision_number}] {r.name} ({r.prompt_count} prompts &middot;{' '}
                    {r.content_hash.slice(0, 8)})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="probe-search">
                Search Prompts (ID, Title, Text)
              </label>
              <input
                id="probe-search"
                className="input"
                value={searchQuery}
                onChange={(e: any) => setSearchQuery(e.target.value)}
                placeholder="Filter by keyword or prompt_id..."
                data-testid="probe-search-input"
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '16px' }}>
            {families.map((fam) => (
              <button
                key={fam}
                type="button"
                className={`btn btn-sm ${selectedFamily === fam ? 'btn-primary' : ''}`}
                onClick={() => setSelectedFamily(fam)}
              >
                {fam.toUpperCase()}
              </button>
            ))}
          </div>

          {activeRev && (
            <div className="mono" style={{ fontSize: '12px', color: '#787774', marginBottom: '12px' }}>
              Revision ID: <strong>{activeRev.id}</strong> &middot; Suite: {activeRev.suite_id} (Rev #
              {activeRev.revision_number}) &middot; SHA-256: <code>{activeRev.content_hash}</code>
            </div>
          )}

          {editingPromptId && (
            <form
              onSubmit={handleSaveNewRevision}
              className="bento-card bento-card-muted"
              style={{ marginBottom: '16px' }}
            >
              <div className="card-label">
                <span>EDIT PROMPT '{editingPromptId}' INTO NEW IMMUTABLE REVISION</span>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setEditingPromptId(null)}
                >
                  Cancel
                </button>
              </div>
              <div className="form-grid">
                <div className="form-field span-6">
                  <label className="form-label" htmlFor="edit-rev-name">
                    New Revision Name
                  </label>
                  <input
                    id="edit-rev-name"
                    className="input"
                    value={newRevisionName}
                    onChange={(e: any) => setNewRevisionName(e.target.value)}
                    required
                  />
                </div>
                <div className="form-field span-6">
                  <label className="form-label" htmlFor="edit-prompt-title">
                    Prompt Title
                  </label>
                  <input
                    id="edit-prompt-title"
                    className="input"
                    value={editPromptTitle}
                    onChange={(e: any) => setEditPromptTitle(e.target.value)}
                    required
                  />
                </div>
                <div className="form-field span-12">
                  <label className="form-label" htmlFor="edit-prompt-text">
                    Prompt Text
                  </label>
                  <textarea
                    id="edit-prompt-text"
                    className="textarea"
                    value={editPromptText}
                    onChange={(e: any) => setEditPromptText(e.target.value)}
                    required
                    data-testid="edit-prompt-textarea"
                  />
                </div>
                <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    data-testid="save-new-revision-btn"
                  >
                    Create New Immutable Revision
                  </button>
                </div>
              </div>
            </form>
          )}

          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID &amp; Family</th>
                  <th>Prompt Title &amp; Text</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredPrompts.map((p) => (
                  <tr key={p.prompt_id}>
                    <td style={{ whiteSpace: 'nowrap' }}>
                      <div className="mono" style={{ fontWeight: 600, color: '#111111' }}>
                        {p.prompt_id}
                      </div>
                      <span className="badge badge-neutral">{p.family}</span>
                    </td>
                    <td>
                      <div style={{ fontWeight: 600, color: '#111111', marginBottom: '3px' }}>
                        {p.title}
                      </div>
                      <div style={{ fontSize: '13px', color: '#2F3437' }}>{p.prompt_text}</div>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => beginEditPrompt(p)}
                        data-testid={`edit-prompt-btn-${p.prompt_id}`}
                      >
                        Edit into New Rev
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Import JSON / JSONL Probe Suite */}
        <div className="bento-card span-4">
          <div className="card-label">
            <span>IMPORT PROBE SUITE</span>
            <span className="mono">BOUNDED &le; 200 PROMPTS</span>
          </div>
          <h2 className="card-title">JSON / JSONL Probe Importer</h2>
          <p style={{ fontSize: '13px', color: '#787774', marginBottom: '12px' }}>
            Validates schema, enforces unique <code>prompt_id</code> values, and saves a new immutable revision.
          </p>
          <form onSubmit={handleImportProbes} className="form-grid">
            <div className="form-field span-8">
              <label className="form-label" htmlFor="imp-probe-name">
                Suite Name
              </label>
              <input
                id="imp-probe-name"
                className="input"
                value={importName}
                onChange={(e: any) => setImportName(e.target.value)}
                placeholder="Custom Domain Suite v1"
                required
                data-testid="probe-import-name-input"
              />
            </div>
            <div className="form-field span-4">
              <label className="form-label" htmlFor="imp-probe-fmt">
                Format
              </label>
              <select
                id="imp-probe-fmt"
                className="select"
                value={importFormat}
                onChange={(e: any) => setImportFormat(e.target.value as 'json' | 'jsonl')}
                data-testid="probe-import-format-select"
              >
                <option value="jsonl">JSONL</option>
                <option value="json">JSON</option>
              </select>
            </div>
            <div className="form-field span-12">
              <label className="form-label" htmlFor="imp-probe-content">
                Payload Content
              </label>
              <textarea
                id="imp-probe-content"
                className="textarea"
                value={importContent}
                onChange={(e: any) => setImportContent(e.target.value)}
                placeholder={
                  '{"prompt_id": "cust-01", "family": "reasoning", "title": "Check", "prompt_text": "Compute 19 * 23."}'
                }
                required
                data-testid="probe-import-content-textarea"
              />
            </div>
            <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="submit"
                className="btn btn-primary"
                data-testid="probe-import-submit-btn"
              >
                Validate &amp; Import Probes
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export function NewAuditScreen(props: {
  profiles: Profile[];
  probeRevisions: ProbeRevision[];
  calibrations: Calibration[];
  settings: AppSettings | null;
  onAuditCreated: (auditId: string) => void;
}) {
  const { profiles, probeRevisions, calibrations, settings, onAuditCreated } = props;

  const [auditName, setAuditName] = useState('Screening Audit Run');
  const [refProfileId, setRefProfileId] = useState(
    profiles.find((p) => p.id === 'prof_demo_reference')?.id || profiles[0]?.id || ''
  );
  const [tgtProfileId, setTgtProfileId] = useState(
    profiles.find((p) => p.id === 'prof_demo_target_substituted')?.id ||
      profiles[1]?.id ||
      profiles[0]?.id ||
      ''
  );
  const [probeRevId, setProbeRevId] = useState(
    probeRevisions[0]?.id || 'rev_builtin_screening_v1'
  );
  const [calibrationId, setCalibrationId] = useState<string>('');
  const [scoreMode, setScoreMode] = useState<ScoreMode>('local_logit_rank');
  const [nPrompts, setNPrompts] = useState(String(settings?.default_n_prompts || 20));
  const [mReference, setMReference] = useState(String(settings?.default_m_reference || 8));
  const [maxOutputTokens, setMaxOutputTokens] = useState('128');
  const [alpha, setAlpha] = useState(String(settings?.default_alpha || 0.01));
  const [temperature, setTemperature] = useState('0.7');
  const [topP, setTopP] = useState('0.95');
  const [maxTargetAttempts, setMaxTargetAttempts] = useState('');
  const [maxRefAttempts, setMaxRefAttempts] = useState('');
  const [maxTotalAttempts, setMaxTotalAttempts] = useState('');
  const [maxDurationSeconds, setMaxDurationSeconds] = useState(
    String(settings?.default_max_duration_seconds || 600)
  );
  const [masterSeed, setMasterSeed] = useState('424242');
  const [includeSecondary, setIncludeSecondary] = useState(false);

  const [preview, setPreview] = useState<AuditPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const buildRequestPayload = () => {
    const selectedRev = probeRevisions.find((r) => r.id === probeRevId);
    const maxAvail = selectedRev?.prompt_count || 20;
    const clampedN = Math.min(Math.max(1, Number(nPrompts) || 20), maxAvail);
    return {
      name: auditName.trim() || 'ModelTrace Audit',
      reference_profile_id: refProfileId,
      target_profile_id: tgtProfileId,
      probe_revision_id: probeRevId,
      calibration_id: calibrationId || null,
      score_mode: scoreMode,
      n_prompts: clampedN,
      m_reference: Math.max(1, Number(mReference) || 8),
      max_output_tokens: Math.min(128, Math.max(1, Number(maxOutputTokens) || 128)),
      alpha: Number(alpha) || 0.01,
      temperature: Number(temperature) || 0.7,
      top_p: Number(topP) || 0.95,
      mc_simulations: 9999,
      max_target_attempts: maxTargetAttempts ? Number(maxTargetAttempts) : null,
      max_reference_attempts: maxRefAttempts ? Number(maxRefAttempts) : null,
      max_total_attempts: maxTotalAttempts ? Number(maxTotalAttempts) : null,
      max_duration_seconds: Number(maxDurationSeconds) || 600,
      master_seed: Number(masterSeed) || 424242,
      include_secondary_tests: includeSecondary,
    };
  };

  const refreshPreview = async () => {
    if (!refProfileId || !tgtProfileId) return;
    setPreviewError(null);
    try {
      const res = await apiClient.previewAudit(buildRequestPayload());
      setPreview(res);
    } catch (err: any) {
      setPreview(null);
      setPreviewError(err.detail || err.message);
    }
  };

  useEffect(() => {
    refreshPreview();
  }, [
    refProfileId,
    tgtProfileId,
    probeRevId,
    calibrationId,
    scoreMode,
    nPrompts,
    mReference,
    maxOutputTokens,
    alpha,
    maxTargetAttempts,
    maxRefAttempts,
    maxTotalAttempts,
    maxDurationSeconds,
    masterSeed,
  ]);

  const handleCreateAndStart = async (startImmediately: boolean) => {
    setActionError(null);
    setSubmitting(true);
    try {
      const payload = {
        ...buildRequestPayload(),
        start_immediately: startImmediately,
      };
      const created = await apiClient.createAudit(payload);
      onAuditCreated(created.id);
    } catch (err: any) {
      setActionError(`[${err.errorCode || 'AUDIT_START_BLOCKED'}] ${err.detail || err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleQuickDemoRun = async (scenario: 'genuine' | 'substituted' | 'mixed_routing') => {
    setActionError(null);
    setSubmitting(true);
    try {
      const tgtMap = {
        genuine: 'prof_demo_target_genuine',
        substituted: 'prof_demo_target_substituted',
        mixed_routing: 'prof_demo_target_mixed',
      };
      const selectedRev = probeRevisions.find((r) => r.id === probeRevId);
      const nToUse = Math.min(Number(nPrompts) || 20, selectedRev?.prompt_count || 20);
      const created = await apiClient.createAudit({
        name: `[DEMO] Synthetic Audit (${scenario})`,
        reference_profile_id: 'prof_demo_reference',
        target_profile_id: tgtMap[scenario],
        probe_revision_id: probeRevId,
        calibration_id: calibrationId || null,
        score_mode: scoreMode,
        n_prompts: nToUse,
        m_reference: Number(mReference) || 8,
        max_output_tokens: 128,
        alpha: Number(alpha) || 0.01,
        master_seed: Number(masterSeed) || 424242,
        start_immediately: true,
        include_secondary_tests: includeSecondary,
      });
      onAuditCreated(created.id);
    } catch (err: any) {
      setActionError(err.detail || err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">PRE-REGISTERED AUDIT MANIFEST &middot; DELIBERATE EXECUTION</div>
          <h1 className="editorial-title">Configure &amp; Freeze New Audit</h1>
          <p className="editorial-subtitle">
            Freeze the probe revision, score mode, sampling parameters, significance level &alpha;, attempt caps, and derived RNG streams before sending a single query. Paid work never starts automatically.
          </p>
        </div>
      </div>

      {actionError && (
        <div className="notice-banner notice-banner-error" role="alert" data-testid="new-audit-error-banner">
          <div>{actionError}</div>
          <button type="button" className="btn btn-sm" onClick={() => setActionError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Configuration Form */}
        <div className="bento-card span-7">
          <div className="card-label">
            <span>1. SELECT PROFILES, PROBE REVISION &amp; SCORING MODE</span>
            <span className="mono">FROZEN BEFORE START</span>
          </div>

          <div className="form-grid">
            <div className="form-field span-12">
              <label className="form-label" htmlFor="aud-name">
                Audit Run Name
              </label>
              <input
                id="aud-name"
                className="input"
                value={auditName}
                onChange={(e: any) => setAuditName(e.target.value)}
                data-testid="new-audit-name-input"
              />
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="aud-ref">
                Trusted Reference Profile
              </label>
              <select
                id="aud-ref"
                className="select"
                value={refProfileId}
                onChange={(e: any) => setRefProfileId(e.target.value)}
                data-testid="new-audit-ref-select"
              >
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.adapter_type})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="aud-tgt">
                Audited Target Profile
              </label>
              <select
                id="aud-tgt"
                className="select"
                value={tgtProfileId}
                onChange={(e: any) => setTgtProfileId(e.target.value)}
                data-testid="new-audit-tgt-select"
              >
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.adapter_type})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="aud-probe">
                Frozen Probe Revision
              </label>
              <select
                id="aud-probe"
                className="select"
                value={probeRevId}
                onChange={(e: any) => setProbeRevId(e.target.value)}
                data-testid="new-audit-probe-select"
              >
                {probeRevisions.map((r) => (
                  <option key={r.id} value={r.id}>
                    [Rev #{r.revision_number}] {r.name} ({r.prompt_count} prompts)
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="aud-mode">
                Primary Score Mode
              </label>
              <select
                id="aud-mode"
                className="select"
                value={scoreMode}
                onChange={(e: any) => setScoreMode(e.target.value as ScoreMode)}
                data-testid="new-audit-score-mode-select"
              >
                <option value="local_logit_rank">
                  local_logit_rank (Teacher-Forcing Log Vocab Rank)
                </option>
                <option value="length_distribution_only">
                  length_distribution_only (Response-Only Unicode Length Fallback)
                </option>
              </select>
            </div>

            <div className="form-field span-12">
              <label className="form-label" htmlFor="aud-cal">
                Power Calibration Artifact (Required for CONSISTENT_WITHIN_TESTED_LIMITS)
              </label>
              <select
                id="aud-cal"
                className="select"
                value={calibrationId}
                onChange={(e: any) => setCalibrationId(e.target.value)}
              >
                <option value="">
                  None attached (Non-rejection will resolve to INCONCLUSIVE)
                </option>
                {calibrations.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} (mode={c.score_mode}, n={c.n_prompts}, m={c.m_reference}, minPower=
                    {(c.min_empirical_power * 100).toFixed(0)}%)
                  </option>
                ))}
              </select>
            </div>

            <div className="form-field span-3">
              <label className="form-label" htmlFor="aud-n">
                n Prompts
              </label>
              <input
                id="aud-n"
                type="number"
                min="1"
                max="200"
                className="input mono"
                value={nPrompts}
                onChange={(e: any) => setNPrompts(e.target.value)}
                data-testid="new-audit-nprompts-input"
              />
            </div>

            <div className="form-field span-3">
              <label className="form-label" htmlFor="aud-m">
                m Ref / Prompt
              </label>
              <input
                id="aud-m"
                type="number"
                min="1"
                max="50"
                className="input mono"
                value={mReference}
                onChange={(e: any) => setMReference(e.target.value)}
                data-testid="new-audit-mref-input"
              />
            </div>

            <div className="form-field span-3">
              <label className="form-label" htmlFor="aud-tok">
                Max Tokens (&le;128)
              </label>
              <input
                id="aud-tok"
                type="number"
                min="1"
                max="128"
                className="input mono"
                value={maxOutputTokens}
                onChange={(e: any) => setMaxOutputTokens(e.target.value)}
              />
            </div>

            <div className="form-field span-3">
              <label className="form-label" htmlFor="aud-alpha">
                Alpha (&alpha;)
              </label>
              <input
                id="aud-alpha"
                type="number"
                step="0.005"
                min="0.001"
                max="0.25"
                className="input mono"
                value={alpha}
                onChange={(e: any) => setAlpha(e.target.value)}
              />
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="aud-cap-tgt">
                Target Attempt Cap
              </label>
              <input
                id="aud-cap-tgt"
                type="number"
                className="input mono"
                value={maxTargetAttempts}
                onChange={(e: any) => setMaxTargetAttempts(e.target.value)}
                placeholder="Auto (+retry buffer)"
              />
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="aud-cap-ref">
                Reference Attempt Cap
              </label>
              <input
                id="aud-cap-ref"
                type="number"
                className="input mono"
                value={maxRefAttempts}
                onChange={(e: any) => setMaxRefAttempts(e.target.value)}
                placeholder="Auto (+retry buffer)"
              />
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="aud-seed">
                Master Seed
              </label>
              <input
                id="aud-seed"
                type="number"
                className="input mono"
                value={masterSeed}
                onChange={(e: any) => setMasterSeed(e.target.value)}
              />
            </div>

            <div className="form-field span-12" style={{ flexDirection: 'row', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <input
                id="aud-secondary"
                type="checkbox"
                checked={includeSecondary}
                onChange={(e: any) => setIncludeSecondary(Boolean(e.target.checked))}
              />
              <label htmlFor="aud-secondary" style={{ fontSize: '13px', color: '#2F3437' }}>
                Include secondary inferential tests (Wilcoxon signed-rank &amp; KS) with mandatory Holm step-down correction
              </label>
            </div>
          </div>
        </div>

        {/* Live Pre-Flight Budget, Cost & RNG Preview */}
        <div className="bento-card span-5">
          <div className="card-label">
            <span>2. PRE-FLIGHT BUDGET &amp; COST PREVIEW</span>
            {preview?.ready_to_start ? (
              <span className="badge badge-sage">READY TO START</span>
            ) : (
              <span className="badge badge-red">BLOCKED</span>
            )}
          </div>

          {previewError && (
            <div className="notice-banner notice-banner-error">{previewError}</div>
          )}

          {preview && (
            <div>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                <DemoBadge isDemo={preview.is_demo} />
                <ScoreScopeBadge scoreMode={preview.score_mode} />
              </div>

              <div
                className="bento-card bento-card-muted"
                style={{ padding: '14px', marginBottom: '14px' }}
              >
                <div className="mono" style={{ fontSize: '12.5px', lineHeight: 1.75 }}>
                  <div>
                    Nominal Reference Calls: <strong>{preview.budget.nominal_reference_calls}</strong> (Cap:{' '}
                    {preview.budget.max_reference_attempts})
                  </div>
                  <div>
                    Nominal Target Calls: <strong>{preview.budget.nominal_target_calls}</strong> (Cap:{' '}
                    {preview.budget.max_target_attempts})
                  </div>
                  <div>
                    Total Calls Planned: <strong>{preview.budget.nominal_total_calls}</strong> (Hard Total Cap:{' '}
                    <strong>{preview.budget.max_total_attempts}</strong>)
                  </div>
                  <div>
                    Max Output Token Bound:{' '}
                    <strong>{preview.budget.estimated_max_output_tokens_total} tokens</strong>
                  </div>
                  <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px solid #EAEAEA' }}>
                    Cost Estimate:{' '}
                    <strong data-testid="preview-cost-display">
                      {preview.budget.cost_estimate.display_label}
                    </strong>
                  </div>
                  <div style={{ fontSize: '11px', color: '#787774' }}>
                    {preview.budget.cost_estimate.assumptions}
                  </div>
                </div>
              </div>

              {preview.blockers.length > 0 && (
                <div className="notice-banner notice-banner-error" style={{ flexDirection: 'column', gap: '4px' }}>
                  <strong>Pre-Flight Blockers:</strong>
                  {preview.blockers.map((b, i) => (
                    <div key={i} style={{ fontSize: '12.5px' }}>
                      &bull; {b}
                    </div>
                  ))}
                </div>
              )}

              {preview.warnings.length > 0 && (
                <div className="notice-banner notice-banner-warn" style={{ flexDirection: 'column', gap: '4px' }}>
                  <strong>Scope &amp; Calibration Notices:</strong>
                  {preview.warnings.map((w, i) => (
                    <div key={i} style={{ fontSize: '12px' }}>
                      &bull; {w}
                    </div>
                  ))}
                </div>
              )}

              <div className="mono" style={{ fontSize: '11.5px', color: '#787774', marginBottom: '16px' }}>
                Derived Seeds (SeedSequence): ref={preview.statistical_plan.derived_seeds.reference_seed}, tgt=
                {preview.statistical_plan.derived_seeds.target_seed}, tie=
                {preview.statistical_plan.derived_seeds.tie_break_seed}, mc=
                {preview.statistical_plan.derived_seeds.mc_null_seed}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!preview.ready_to_start || submitting}
                  onClick={() => handleCreateAndStart(true)}
                  data-testid="start-configured-audit-btn"
                >
                  {submitting ? 'Starting Audit...' : 'Freeze Manifest & Start Audit Now'}
                </button>
                <button
                  type="button"
                  className="btn"
                  disabled={!preview.ready_to_start || submitting}
                  onClick={() => handleCreateAndStart(false)}
                  data-testid="freeze-only-audit-btn"
                >
                  Freeze Manifest Only (Keep in CREATED State)
                </button>
              </div>
            </div>
          )}

          {/* Clearly Separated Synthetic DEMO Actions */}
          <div
            style={{
              marginTop: '20px',
              paddingTop: '16px',
              borderTop: '1px solid #EAEAEA',
            }}
          >
            <div className="card-label">
              <span>SEPARATE SYNTHETIC DEMO ACTIONS (ZERO API COST)</span>
              <span className="badge badge-blue">DEMO</span>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-sm"
                disabled={submitting}
                onClick={() => handleQuickDemoRun('substituted')}
                data-testid="quick-demo-substituted-btn"
              >
                [DEMO] Run Substituted Scenario
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={submitting}
                onClick={() => handleQuickDemoRun('mixed_routing')}
                data-testid="quick-demo-mixed-btn"
              >
                [DEMO] Run Mixed-Routing Scenario
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={submitting}
                onClick={() => handleQuickDemoRun('genuine')}
                data-testid="quick-demo-genuine-btn"
              >
                [DEMO] Run Genuine Scenario
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
