/**
 * Screen 2: Endpoint Profiles (working create/edit/delete, claimed model, URL, settings,
 *           MODELTRACE_PROVIDER_* env-key reference, bounded connection tests & errors)
 * Screen 3: References & Calibrations (synthetic/API/local Transformers configurations,
 *           provenance/device/precision, capability checks, missing prerequisites, calibration import & runner)
 */

import React, { useState } from 'react';
import { apiClient } from '../api/client';
import { DemoBadge, ScoreScopeBadge } from '../components/BrandAndUI';
import {
  AdapterType,
  AppSettings,
  Calibration,
  CapabilityReport,
  ProbeRevision,
  Profile,
  SyntheticScenario,
  SystemCapabilities,
} from '../types/api';

export function ProfilesScreen(props: {
  profiles: Profile[];
  settings: AppSettings | null;
  onRefresh: () => void;
}) {
  const { profiles, onRefresh } = props;

  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [adapterType, setAdapterType] = useState<AdapterType>('openai_compatible');
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:8080/v1');
  const [apiKeyEnvVar, setApiKeyEnvVar] = useState('MODELTRACE_PROVIDER_TARGET_KEY');
  const [modelIdentifier, setModelIdentifier] = useState('meta-llama/Llama-3-8B-Instruct');
  const [syntheticScenario, setSyntheticScenario] = useState<SyntheticScenario>('genuine');
  const [mixedBadFraction, setMixedBadFraction] = useState('0.45');
  const [checkpointPath, setCheckpointPath] = useState('');
  const [precision, setPrecision] = useState('float32');
  const [device, setDevice] = useState('cpu');
  const [timeoutSeconds, setTimeoutSeconds] = useState('15');
  const [maxRetries, setMaxRetries] = useState('2');

  const [formError, setFormError] = useState<string | null>(null);
  const [formSuccess, setFormSuccess] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [testingProfileId, setTestingProfileId] = useState<string | null>(null);
  const [testResultMap, setTestResultMap] = useState<Record<string, CapabilityReport>>({});

  const startEdit = (p: Profile) => {
    setEditingId(p.id);
    setName(p.name);
    setDescription(p.description || '');
    setAdapterType(p.adapter_type);
    setBaseUrl(p.base_url || '');
    setApiKeyEnvVar(p.api_key_env_var || '');
    setModelIdentifier(p.model_identifier);
    setSyntheticScenario((p.synthetic_scenario as SyntheticScenario) || 'genuine');
    setMixedBadFraction(String(p.mixed_routing_bad_fraction ?? 0.45));
    setCheckpointPath(p.checkpoint_path || '');
    setPrecision(p.precision || 'float32');
    setDevice(p.device || 'cpu');
    setTimeoutSeconds(String(p.timeout_seconds ?? 15));
    setMaxRetries(String(p.max_retries ?? 2));
    setFormError(null);
    setFormSuccess(null);
  };

  const resetForm = () => {
    setEditingId(null);
    setName('');
    setDescription('');
    setAdapterType('openai_compatible');
    setBaseUrl('http://127.0.0.1:8080/v1');
    setApiKeyEnvVar('MODELTRACE_PROVIDER_TARGET_KEY');
    setModelIdentifier('meta-llama/Llama-3-8B-Instruct');
    setSyntheticScenario('genuine');
    setMixedBadFraction('0.45');
    setCheckpointPath('');
    setPrecision('float32');
    setDevice('cpu');
    setTimeoutSeconds('15');
    setMaxRetries('2');
  };

  const handleSaveProfile = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setFormError(null);
    setFormSuccess(null);
    setSubmitting(true);
    try {
      const payload: any = {
        name: name.trim(),
        description: description.trim(),
        adapter_type: adapterType,
        model_identifier: modelIdentifier.trim(),
        timeout_seconds: Number(timeoutSeconds) || 15,
        max_retries: Number(maxRetries) || 0,
        precision,
        device,
      };
      if (adapterType === 'openai_compatible') {
        payload.base_url = baseUrl.trim() || null;
        payload.api_key_env_var = apiKeyEnvVar.trim() || null;
      } else if (adapterType === 'synthetic_demo') {
        payload.synthetic_scenario = syntheticScenario;
        payload.mixed_routing_bad_fraction = Number(mixedBadFraction) || 0.45;
      } else if (adapterType === 'local_transformers') {
        payload.checkpoint_path = checkpointPath.trim() || null;
      }

      if (editingId) {
        await apiClient.updateProfile(editingId, payload);
        setFormSuccess(`Updated profile '${payload.name}'.`);
      } else {
        const created = await apiClient.createProfile(payload);
        setFormSuccess(`Created profile '${created.name}' (${created.id}).`);
        resetForm();
      }
      onRefresh();
    } catch (err: any) {
      setFormError(`[${err.errorCode || 'ERROR'}] ${err.detail || err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (p: Profile) => {
    setFormError(null);
    setFormSuccess(null);
    try {
      await apiClient.deleteProfile(p.id);
      setFormSuccess(`Deleted profile '${p.name}'.`);
      onRefresh();
    } catch (err: any) {
      setFormError(`[${err.errorCode || 'DELETE_BLOCKED'}] ${err.detail || err.message}`);
    }
  };

  const handleTestConnection = async (p: Profile, executeNet: boolean) => {
    setTestingProfileId(p.id);
    setFormError(null);
    try {
      const rep = await apiClient.testProfileConnection(p.id, executeNet);
      setTestResultMap((prev) => ({ ...prev, [p.id]: rep }));
      onRefresh();
    } catch (err: any) {
      setFormError(`Connection test error for '${p.name}': ${err.detail || err.message}`);
    } finally {
      setTestingProfileId(null);
    }
  };

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">ENDPOINT PROFILES &middot; ZERO BROWSER SECRETS</div>
          <h1 className="editorial-title">Target &amp; Reference Endpoint Profiles</h1>
          <p className="editorial-subtitle">
            Configure OpenAI-compatible HTTP endpoints, local HuggingFace checkpoints, or seeded synthetic DEMO targets. Provider API keys are referenced strictly by backend environment variable names (<code>MODELTRACE_PROVIDER_*</code>) and never enter the browser.
          </p>
        </div>
      </div>

      {formError && (
        <div className="notice-banner notice-banner-error" role="alert" data-testid="profile-error-banner">
          <div>
            <strong>Profile Action Blocked:</strong> {formError}
          </div>
          <button type="button" className="btn btn-sm" onClick={() => setFormError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {formSuccess && (
        <div className="notice-banner notice-banner-info" role="status" data-testid="profile-success-banner">
          <div>{formSuccess}</div>
          <button type="button" className="btn btn-sm" onClick={() => setFormSuccess(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Profile Create/Edit Form */}
        <div className="bento-card span-12">
          <div className="card-label">
            <span>{editingId ? `EDIT PROFILE (${editingId})` : 'CREATE ENDPOINT PROFILE'}</span>
            {editingId && (
              <button type="button" className="btn btn-sm" onClick={resetForm}>
                Cancel Edit
              </button>
            )}
          </div>
          <form onSubmit={handleSaveProfile} className="form-grid">
            <div className="form-field span-4">
              <label className="form-label" htmlFor="prof-name">
                Profile Name
              </label>
              <input
                id="prof-name"
                className="input"
                value={name}
                onChange={(e: any) => setName(e.target.value)}
                placeholder="e.g., Production Together-AI Llama-3-8B"
                required
                data-testid="profile-name-input"
              />
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="prof-adapter">
                Adapter Type
              </label>
              <select
                id="prof-adapter"
                className="select"
                value={adapterType}
                onChange={(e: any) => setAdapterType(e.target.value as AdapterType)}
                disabled={Boolean(editingId)}
                data-testid="profile-adapter-select"
              >
                <option value="openai_compatible">openai_compatible (Live HTTP)</option>
                <option value="synthetic_demo">synthetic_demo (Seeded DEMO)</option>
                <option value="local_transformers">local_transformers (Local HF)</option>
              </select>
            </div>

            <div className="form-field span-4">
              <label className="form-label" htmlFor="prof-model">
                Claimed Model Identifier
              </label>
              <input
                id="prof-model"
                className="input mono"
                value={modelIdentifier}
                onChange={(e: any) => setModelIdentifier(e.target.value)}
                required
                data-testid="profile-model-input"
              />
              <span className="form-hint">Untrusted provider claim; verified via behavior.</span>
            </div>

            {adapterType === 'openai_compatible' && (
              <div className="form-field span-6">
                <label className="form-label" htmlFor="prof-url">
                  Endpoint Base URL
                </label>
                <input
                  id="prof-url"
                  className="input mono"
                  value={baseUrl}
                  onChange={(e: any) => setBaseUrl(e.target.value)}
                  placeholder="https://api.openai.com/v1 or http://127.0.0.1:8080/v1"
                  data-testid="profile-url-input"
                />
                <span className="form-hint">
                  Exact origin must be permitted by Settings allowlists.
                </span>
              </div>
            )}

            {adapterType === 'openai_compatible' && (
              <div className="form-field span-6">
                <label className="form-label" htmlFor="prof-env">
                  Backend Secret Env Reference (<code>MODELTRACE_PROVIDER_*</code>)
                </label>
                <input
                  id="prof-env"
                  className="input mono"
                  value={apiKeyEnvVar}
                  onChange={(e: any) => setApiKeyEnvVar(e.target.value)}
                  placeholder="MODELTRACE_PROVIDER_TARGET_KEY"
                  data-testid="profile-env-input"
                />
                <span className="form-hint">
                  Enter only the env variable name. Raw API keys are rejected.
                </span>
              </div>
            )}

            {adapterType === 'synthetic_demo' && (
              <div className="form-field span-4">
                <label className="form-label" htmlFor="prof-scenario">
                  Synthetic Scenario
                </label>
                <select
                  id="prof-scenario"
                  className="select"
                  value={syntheticScenario}
                  onChange={(e: any) => setSyntheticScenario(e.target.value as SyntheticScenario)}
                  data-testid="profile-scenario-select"
                >
                  <option value="genuine">genuine</option>
                  <option value="substituted">substituted</option>
                  <option value="mixed_routing">mixed_routing</option>
                </select>
              </div>
            )}

            {adapterType === 'synthetic_demo' && syntheticScenario === 'mixed_routing' && (
              <div className="form-field span-4">
                <label className="form-label" htmlFor="prof-bad-frac">
                  Substitute Fraction (0..1)
                </label>
                <input
                  id="prof-bad-frac"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="input mono"
                  value={mixedBadFraction}
                  onChange={(e: any) => setMixedBadFraction(e.target.value)}
                />
              </div>
            )}

            {adapterType === 'local_transformers' && (
              <div className="form-field span-6">
                <label className="form-label" htmlFor="prof-ckpt">
                  Local Checkpoint Subdirectory
                </label>
                <input
                  id="prof-ckpt"
                  className="input mono"
                  value={checkpointPath}
                  onChange={(e: any) => setCheckpointPath(e.target.value)}
                  placeholder="llama-3-8b-hf"
                />
              </div>
            )}

            <div className="form-field span-3">
              <label className="form-label" htmlFor="prof-timeout">
                Timeout (s)
              </label>
              <input
                id="prof-timeout"
                type="number"
                step="0.5"
                min="0.5"
                max="120"
                className="input mono"
                value={timeoutSeconds}
                onChange={(e: any) => setTimeoutSeconds(e.target.value)}
              />
            </div>

            <div className="form-field span-3">
              <label className="form-label" htmlFor="prof-retries">
                Max Retries (0..5)
              </label>
              <input
                id="prof-retries"
                type="number"
                min="0"
                max="5"
                className="input mono"
                value={maxRetries}
                onChange={(e: any) => setMaxRetries(e.target.value)}
              />
            </div>

            <div className="form-field span-6">
              <label className="form-label" htmlFor="prof-desc">
                Provenance Notes
              </label>
              <input
                id="prof-desc"
                className="input"
                value={description}
                onChange={(e: any) => setDescription(e.target.value)}
                placeholder="Optional endpoint provenance notes"
              />
            </div>

            <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '4px' }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitting}
                data-testid="profile-submit-btn"
              >
                {submitting ? 'Saving...' : editingId ? 'Save Profile Changes' : 'Create Profile'}
              </button>
            </div>
          </form>
        </div>

        {/* Profiles Inventory Table */}
        <div className="bento-card span-12">
          <div className="card-label">
            <span>REGISTERED PROFILES ({profiles.length})</span>
            <span className="mono">BOUNDED CONNECTION TESTS</span>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Profile &amp; Claimed Model</th>
                  <th>Adapter &amp; Origin / Env Ref</th>
                  <th>Capability / Last Probe</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {profiles.map((p) => {
                  const cap = testResultMap[p.id] || p.last_capability;
                  return (
                    <tr key={p.id} data-testid={`profile-row-${p.id}`}>
                      <td>
                        <div style={{ fontWeight: 600, color: '#111111' }}>{p.name}</div>
                        <div className="mono" style={{ fontSize: '11.5px', color: '#787774' }}>
                          {p.id} &middot; {p.model_identifier}
                        </div>
                        <div style={{ marginTop: '4px' }}>
                          <DemoBadge isDemo={p.is_demo} />
                        </div>
                      </td>
                      <td className="mono" style={{ fontSize: '11.5px' }}>
                        <div>{p.adapter_type}</div>
                        {p.base_url && <div style={{ color: '#787774' }}>{p.base_url}</div>}
                        {p.api_key_env_var && (
                          <div style={{ color: '#1F6C9F' }}>env: {p.api_key_env_var}</div>
                        )}
                        {p.adapter_type === 'synthetic_demo' && (
                          <div style={{ color: '#787774' }}>
                            scenario: {p.synthetic_scenario}
                          </div>
                        )}
                        <div style={{ color: '#787774' }}>
                          timeout={p.timeout_seconds}s, retries={p.max_retries}
                        </div>
                      </td>
                      <td>
                        {!cap ? (
                          <span className="badge badge-neutral">NOT TESTED YET</span>
                        ) : cap.status === 'available' ? (
                          <div>
                            <span className="badge badge-sage">AVAILABLE</span>
                            {cap.connection_test && (
                              <div className="mono" style={{ fontSize: '11px', color: '#787774', marginTop: '4px' }}>
                                HTTP {cap.connection_test.http_status} &middot;{' '}
                                {cap.connection_test.latency_ms}ms
                                <br />
                                Declared (untrusted): {cap.connection_test.declared_model}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div>
                            <span className="badge badge-red">
                              UNAVAILABLE: {cap.reason_code || 'ERROR'}
                            </span>
                            <div className="mono" style={{ fontSize: '11px', color: '#9F2F2D', marginTop: '4px' }}>
                              {cap.detail}
                            </div>
                          </div>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                          <button
                            type="button"
                            className="btn btn-sm"
                            disabled={testingProfileId === p.id}
                            onClick={() => handleTestConnection(p, true)}
                            data-testid={`test-conn-btn-${p.id}`}
                          >
                            {testingProfileId === p.id ? 'Testing...' : 'Test Connection'}
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm"
                            onClick={() => startEdit(p)}
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-danger"
                            onClick={() => handleDelete(p)}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ReferencesScreen(props: {
  profiles: Profile[];
  calibrations: Calibration[];
  probeRevisions: ProbeRevision[];
  capabilities: SystemCapabilities | null;
  settings: AppSettings | null;
  onRefresh: () => void;
}) {
  const { calibrations, probeRevisions, capabilities, onRefresh } = props;

  const [demoCalName, setDemoCalName] = useState('DEMO Screening Power Calibration (n=20, m=8)');
  const [demoScoreMode, setDemoScoreMode] = useState<'local_logit_rank' | 'length_distribution_only'>('local_logit_rank');
  const [demoNPrompts, setDemoNPrompts] = useState('20');
  const [demoMRef, setDemoMRef] = useState('8');
  const [demoAlpha, setDemoAlpha] = useState('0.01');
  const [demoTrials, setDemoTrials] = useState('20');
  const [runningDemoCal, setRunningDemoCal] = useState(false);

  const [importJsonText, setImportJsonText] = useState('');
  const [calMessage, setCalMessage] = useState<string | null>(null);
  const [calError, setCalError] = useState<string | null>(null);

  const handleRunDemoCalibration = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setCalError(null);
    setCalMessage(null);
    setRunningDemoCal(true);
    try {
      const cal = await apiClient.runDemoCalibration({
        name: demoCalName.trim() || 'DEMO Screening Calibration',
        probe_revision_id: probeRevisions[0]?.id || 'rev_builtin_screening_v1',
        score_mode: demoScoreMode,
        n_prompts: Number(demoNPrompts) || 20,
        m_reference: Number(demoMRef) || 8,
        max_output_tokens: 128,
        alpha: Number(demoAlpha) || 0.01,
        n_trials: Number(demoTrials) || 20,
      });
      setCalMessage(
        `Generated & verified DEMO calibration '${cal.name}' (${cal.id}) — min sensitivity ${(
          cal.min_empirical_power * 100
        ).toFixed(1)}% across ${cal.min_trials} trials.`
      );
      onRefresh();
    } catch (err: any) {
      setCalError(err.detail || err.message);
    } finally {
      setRunningDemoCal(false);
    }
  };

  const handleImportCalibrationJson = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setCalError(null);
    setCalMessage(null);
    try {
      const parsed = JSON.parse(importJsonText);
      const created = await apiClient.importCalibration(parsed);
      setCalMessage(`Imported calibration artifact '${created.name}' (${created.id}).`);
      setImportJsonText('');
      onRefresh();
    } catch (err: any) {
      setCalError(`Calibration import failed: ${err.detail || err.message}`);
    }
  };

  const hfInfo = capabilities?.adapters.local_transformers;

  return (
    <div>
      <div className="screen-header">
        <div>
          <div className="eyebrow">TRUSTED REFERENCES &amp; POWER CALIBRATION</div>
          <h1 className="editorial-title">Reference Configurations &amp; Sensitivity</h1>
          <p className="editorial-subtitle">
            Inspect local Transformers teacher-forcing prerequisites, reference endpoint capabilities, and empirical power calibration artifacts required to claim <code>CONSISTENT_WITHIN_TESTED_LIMITS</code>.
          </p>
        </div>
      </div>

      {calError && (
        <div className="notice-banner notice-banner-error" role="alert">
          <div>{calError}</div>
          <button type="button" className="btn btn-sm" onClick={() => setCalError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {calMessage && (
        <div className="notice-banner notice-banner-info" role="status" data-testid="calibration-success-banner">
          <div>{calMessage}</div>
          <button type="button" className="btn btn-sm" onClick={() => setCalMessage(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bento-grid">
        {/* Local Transformers Capability & Prerequisite Diagnostics */}
        <div className="bento-card span-6">
          <div className="card-label">
            <span>ADAPTER C: LOCAL TRANSFORMERS TEACHER-FORCING</span>
            {hfInfo?.available ? (
              <span className="badge badge-sage">AVAILABLE</span>
            ) : (
              <span className="badge badge-apricot">PREREQUISITES MISSING</span>
            )}
          </div>
          <h2 className="card-title">Full-Vocabulary Logit-Rank Scoring Engine</h2>
          <p style={{ fontSize: '13.5px', color: '#2F3437', marginBottom: '12px' }}>
            Computes mean log vocabulary rank of response tokens under reference logits with next-token alignment, chat prefix formatting, prompt/padding exclusion, and public-text retokenization.
          </p>
          <div className="mono" style={{ fontSize: '12.5px', lineHeight: 1.75, background: '#F9F9F8', padding: '12px', borderRadius: '6px', border: '1px solid #EAEAEA' }}>
            <div>
              Security Policy: <code>trust_remote_code=False</code>, <code>local_files_only=True</code>
            </div>
            <div>
              Allowed Checkpoint Base Dir: <code>{capabilities?.runtime.local_models_dir || './data/local_models'}</code>
            </div>
            <div>
              Missing Optional Packages:{' '}
              {hfInfo?.missing_dependencies && hfInfo.missing_dependencies.length > 0 ? (
                <strong style={{ color: '#956400' }}>{hfInfo.missing_dependencies.join(', ')}</strong>
              ) : (
                <strong>None</strong>
              )}
            </div>
            <div style={{ color: '#787774', marginTop: '6px' }}>
              No unsolicited model downloads or silent checkpoint substitutions are ever performed. When local weights are unavailable, use Synthetic DEMO or explicitly scoped <code>length_distribution_only</code> fallback.
            </div>
          </div>
        </div>

        {/* Generate Seeded DEMO Power Calibration */}
        <div className="bento-card span-6">
          <div className="card-label">
            <span>EMPIRICAL SENSITIVITY CALIBRATOR</span>
            <span className="badge badge-blue">DEMO SIMULATION</span>
          </div>
          <h2 className="card-title">Generate Compatible DEMO Power Calibration</h2>
          <p style={{ fontSize: '13.5px', color: '#787774', marginBottom: '12px' }}>
            Non-rejection (p &gt; &alpha;) without power calibration yields <code>INCONCLUSIVE</code>. Run a seeded synthetic power study across named alternatives (<code>Substituted-INT4</code> and <code>Mixed-Routing-65pct</code>) to establish sensitivity (&ge; 80% empirical power and 95% Wilson lower confidence bound &ge; 80% across &ge; 20 trials).
          </p>
          <form onSubmit={handleRunDemoCalibration} className="form-grid">
            <div className="form-field span-8">
              <label className="form-label" htmlFor="cal-name">
                Calibration Artifact Name
              </label>
              <input
                id="cal-name"
                className="input"
                value={demoCalName}
                onChange={(e: any) => setDemoCalName(e.target.value)}
              />
            </div>
            <div className="form-field span-4">
              <label className="form-label" htmlFor="cal-mode">
                Score Mode
              </label>
              <select
                id="cal-mode"
                className="select"
                value={demoScoreMode}
                onChange={(e: any) => setDemoScoreMode(e.target.value)}
              >
                <option value="local_logit_rank">local_logit_rank</option>
                <option value="length_distribution_only">length_distribution_only</option>
              </select>
            </div>
            <div className="form-field span-3">
              <label className="form-label" htmlFor="cal-n">
                n Prompts
              </label>
              <input
                id="cal-n"
                type="number"
                className="input mono"
                value={demoNPrompts}
                onChange={(e: any) => setDemoNPrompts(e.target.value)}
              />
            </div>
            <div className="form-field span-3">
              <label className="form-label" htmlFor="cal-m">
                m Ref / Prompt
              </label>
              <input
                id="cal-m"
                type="number"
                className="input mono"
                value={demoMRef}
                onChange={(e: any) => setDemoMRef(e.target.value)}
              />
            </div>
            <div className="form-field span-3">
              <label className="form-label" htmlFor="cal-alpha">
                Alpha (&alpha;)
              </label>
              <input
                id="cal-alpha"
                type="number"
                step="0.005"
                className="input mono"
                value={demoAlpha}
                onChange={(e: any) => setDemoAlpha(e.target.value)}
              />
            </div>
            <div className="form-field span-3">
              <label className="form-label" htmlFor="cal-trials">
                Trials (&ge;20)
              </label>
              <input
                id="cal-trials"
                type="number"
                min="20"
                max="100"
                className="input mono"
                value={demoTrials}
                onChange={(e: any) => setDemoTrials(e.target.value)}
              />
            </div>
            <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={runningDemoCal}
                data-testid="run-demo-calibration-btn"
              >
                {runningDemoCal ? 'Simulating Trials...' : 'Run Seeded DEMO Power Calibration'}
              </button>
            </div>
          </form>
        </div>

        {/* Persisted Calibration Artifacts Table */}
        <div className="bento-card span-8">
          <div className="card-label">
            <span>STORED CALIBRATION ARTIFACTS ({calibrations.length})</span>
            <span className="mono">PROVENANCE &amp; SENSITIVITY</span>
          </div>
          {calibrations.length === 0 ? (
            <div style={{ color: '#787774', fontSize: '13.5px' }}>
              No calibration artifacts stored yet. Generate a DEMO calibration above or import a JSON calibration artifact.
            </div>
          ) : (
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Calibration Artifact</th>
                    <th>Budget &amp; Mode</th>
                    <th>Named Alternatives &amp; Power</th>
                    <th className="num-col">Min Power / Trials</th>
                  </tr>
                </thead>
                <tbody>
                  {calibrations.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <div style={{ fontWeight: 600, color: '#111111' }}>{c.name}</div>
                        <div className="mono" style={{ fontSize: '11.5px', color: '#787774' }}>
                          {c.id} &middot; Probe Hash: {c.probe_revision_hash.slice(0, 12)}...
                        </div>
                        <div style={{ marginTop: '4px' }}>
                          <DemoBadge isDemo={c.is_demo} />
                        </div>
                      </td>
                      <td className="mono" style={{ fontSize: '12px' }}>
                        <ScoreScopeBadge scoreMode={c.score_mode} />
                        <div style={{ marginTop: '4px' }}>
                          n={c.n_prompts}, m={c.m_reference}, &alpha;={c.alpha}
                        </div>
                      </td>
                      <td className="mono" style={{ fontSize: '12px' }}>
                        {c.alternatives.map((alt) => (
                          <div key={alt.alternative_name}>
                            {alt.alternative_name}: {(alt.empirical_power * 100).toFixed(1)}% (
                            {alt.n_rejected}/{alt.n_trials}, Wilson95&ge;
                            {(alt.wilson_lower_95 * 100).toFixed(1)}%)
                          </div>
                        ))}
                      </td>
                      <td className="num-col">
                        <span
                          className={
                            c.min_empirical_power >= 0.8 && c.min_trials >= 20 && c.alternatives.every((a: any) => a.wilson_lower_95 >= 0.8)
                              ? 'badge badge-sage'
                              : 'badge badge-apricot'
                          }
                        >
                          {(c.min_empirical_power * 100).toFixed(1)}% ({c.min_trials} trials)
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Import External Calibration JSON */}
        <div className="bento-card span-4">
          <div className="card-label">
            <span>IMPORT CALIBRATION ARTIFACT</span>
            <span className="mono">HASH VALIDATED</span>
          </div>
          <form onSubmit={handleImportCalibrationJson} className="form-grid">
            <div className="form-field span-12">
              <label className="form-label" htmlFor="cal-import-json">
                Calibration JSON Payload
              </label>
              <textarea
                id="cal-import-json"
                className="textarea"
                value={importJsonText}
                onChange={(e: any) => setImportJsonText(e.target.value)}
                placeholder='{"name": "...", "score_mode": "local_logit_rank", "probe_revision_id": "rev_builtin_screening_v1", "probe_revision_hash": "...", ...}'
              />
              <span className="form-hint">
                Incompatible probe revision hashes or mismatched empirical_power ratios are rejected.
              </span>
            </div>
            <div className="span-12" style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="submit"
                className="btn"
                disabled={!importJsonText.trim()}
              >
                Validate &amp; Import Calibration
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
