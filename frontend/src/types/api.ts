/**
 * TypeScript interfaces aligned 1-to-1 with ModelTrace FastAPI OpenAPI schemas.
 */

export type AdapterType = 'synthetic_demo' | 'openai_compatible' | 'local_transformers';
export type SyntheticScenario = 'genuine' | 'substituted' | 'mixed_routing';
export type ScoreMode = 'local_logit_rank' | 'length_distribution_only';
export type ExecutionState =
  | 'CREATED'
  | 'RUNNING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'INTERRUPTED';
export type AuditVerdict =
  | 'INCONSISTENT'
  | 'INCONCLUSIVE'
  | 'CONSISTENT_WITHIN_TESTED_LIMITS';

export interface CapabilityReport {
  status: 'available' | 'unavailable';
  adapter_type: AdapterType;
  is_demo: boolean;
  reason_code?: string;
  detail?: string;
  missing_packages?: string[];
  checkpoint_path?: string;
  endpoint_origin?: string;
  resolved_ips?: string[];
  supports_generation: boolean;
  supports_local_logit_scoring: boolean;
  connection_test?: {
    http_status: number;
    latency_ms: number;
    declared_model?: string;
    finish_reason?: string;
    sample_preview?: string;
  };
}

export interface Profile {
  id: string;
  name: string;
  description: string;
  adapter_type: AdapterType;
  base_url?: string | null;
  api_key_env_var?: string | null;
  model_identifier: string;
  checkpoint_path?: string | null;
  checkpoint_revision?: string | null;
  tokenizer_path?: string | null;
  chat_template?: string | null;
  precision: string;
  device: string;
  synthetic_scenario?: SyntheticScenario | null;
  mixed_routing_bad_fraction: number;
  timeout_seconds: number;
  max_retries: number;
  extra_config: Record<string, any>;
  is_demo: boolean;
  last_capability?: CapabilityReport | null;
  created_at: string;
  updated_at: string;
}

export interface ProbePromptItem {
  prompt_id: string;
  family: string;
  title: string;
  prompt_text: string;
  system_hint?: string | null;
  expected_constraint?: string | null;
  validation_status: string;
}

export interface ProbeRevision {
  id: string;
  suite_id: string;
  revision_number: number;
  parent_revision_id?: string | null;
  name: string;
  description: string;
  prompt_count: number;
  families: string[];
  validation_label: string;
  content_hash: string;
  created_at: string;
  prompts?: ProbePromptItem[];
}

export interface CalibrationAlternative {
  alternative_name: string;
  alternative_description: string;
  n_trials: number;
  n_rejected: number;
  empirical_power: number;
  wilson_lower_95: number;
}

export interface Calibration {
  id: string;
  name: string;
  description: string;
  score_mode: ScoreMode;
  probe_revision_id: string;
  probe_revision_hash: string;
  n_prompts: number;
  m_reference: number;
  max_output_tokens: number;
  alpha: number;
  temperature: number;
  top_p: number;
  reference_model_descriptor: string;
  is_demo: boolean;
  provenance_notes: string;
  alternatives: CalibrationAlternative[];
  min_empirical_power: number;
  min_trials: number;
  calibration_seed: number;
  artifact_hash: string;
  created_at: string;
}

export interface CostEstimate {
  status: string;
  is_known: boolean;
  nominal_usd: number | null;
  max_cap_usd: number | null;
  display_label: string;
  assumptions: string;
}

export interface AuditPreviewResponse {
  ready_to_start: boolean;
  is_demo: boolean;
  score_mode: ScoreMode;
  score_scope_banner: string;
  probe_revision: {
    id: string;
    name: string;
    revision_number: number;
    content_hash: string;
    validation_label: string;
  };
  budget: {
    n_prompts: number;
    m_reference_per_prompt: number;
    nominal_reference_calls: number;
    nominal_target_calls: number;
    nominal_total_calls: number;
    max_output_tokens: number;
    max_target_attempts: number;
    max_reference_attempts: number;
    max_total_attempts: number;
    max_duration_seconds: number;
    estimated_max_output_tokens_total: number;
    cost_estimate: CostEstimate;
  };
  statistical_plan: {
    method: string;
    version: string;
    alpha: number;
    mc_simulations: number;
    derived_seeds: Record<string, number>;
    screening_budget_disclaimer: string;
  };
  calibration_compatibility: {
    compatible: boolean;
    calibration_id?: string;
    calibration_name?: string;
    is_demo_calibration?: boolean;
    min_empirical_power?: number;
    min_trials?: number;
    issues?: string[];
    reason: string;
  };
  warnings: string[];
  blockers: string[];
}

export interface Audit {
  id: string;
  name: string;
  notes: string;
  reference_profile_id: string;
  target_profile_id: string;
  scoring_reference_profile_id?: string | null;
  probe_revision_id: string;
  probe_revision_hash: string;
  calibration_id?: string | null;
  score_mode: ScoreMode;
  is_demo: boolean;
  n_prompts: number;
  m_reference: number;
  max_output_tokens: number;
  alpha: number;
  temperature: number;
  top_p: number;
  mc_simulations: number;
  include_secondary_tests: boolean;
  max_target_attempts: number;
  max_reference_attempts: number;
  max_total_attempts: number;
  max_duration_seconds: number;
  master_seed: number;
  seeds: Record<string, number>;
  manifest: Record<string, any>;
  manifest_hash: string;
  execution_state: ExecutionState;
  verdict: AuditVerdict;
  verdict_reason: string;
  cancel_requested: boolean;
  resume_count: number;
  uncertain_prior_attempts: number;
  local_scoring_compute_ms: number;
  summary?: Record<string, any> | null;
  integrity_hash?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at: string;
}

export interface PromptScore {
  id: string;
  audit_id: string;
  prompt_id: string;
  prompt_index: number;
  family: string;
  score_mode: ScoreMode;
  m_reference: number;
  reference_scores: number[];
  target_score: number;
  a_below: number;
  b_ties: number;
  j_draw: number;
  u_draw: number;
  z_uniform: number;
  score_shift: number;
  target_norm_ll?: number | null;
  ref_mean_norm_ll?: number | null;
  local_scoring_compute_ms: number;
  created_at: string;
}

export interface SampleAttempt {
  id: string;
  audit_id: string;
  prompt_id: string;
  prompt_index: number;
  role: 'reference' | 'target';
  sample_index: number;
  attempt_number: number;
  profile_id: string;
  status: string;
  http_status?: number | null;
  error_code?: string | null;
  error_detail?: string | null;
  latency_ms?: number | null;
  retry_count: number;
  started_at: string;
  completed_at?: string | null;
}

export interface SampleResponse {
  id: string;
  attempt_id: string;
  audit_id: string;
  prompt_id: string;
  role: 'reference' | 'target';
  sample_index: number;
  response_text: string;
  token_count: number;
  unicode_length: number;
  finish_reason?: string | null;
  declared_model?: string | null;
  observable_meta: Record<string, any>;
  raw_score: number;
  normalized_log_likelihood?: number | null;
  scoring_compute_ms: number;
  response_hash: string;
  created_at: string;
}

export interface AuditEvent {
  id: string;
  audit_id: string;
  event_type: string;
  level: string;
  message: string;
  details: Record<string, any>;
  created_at: string;
}

export interface ExportRecord {
  id: string;
  audit_id: string;
  export_format: string;
  file_path?: string | null;
  content_hash: string;
  byte_size: number;
  created_at: string;
}

export interface AuditDetailBundle {
  audit: Audit;
  prompt_scores: PromptScore[];
  attempts: SampleAttempt[];
  responses: SampleResponse[];
  exports: ExportRecord[];
}

export interface AuditProgress {
  audit_id: string;
  name: string;
  is_demo: boolean;
  score_mode: ScoreMode;
  execution_state: ExecutionState;
  verdict: AuditVerdict;
  verdict_reason: string;
  completed_prompts: number;
  total_prompts: number;
  completed_samples: number;
  nominal_total_samples: number;
  progress_fraction: number;
  attempts: {
    reference_attempts_used: number;
    target_attempts_used: number;
    total_attempts_used: number;
    completed_reference_samples: number;
    completed_target_samples: number;
    failed_attempts: number;
    uncertain_prior_attempts: number;
    max_reference_attempts: number;
    max_target_attempts: number;
    max_total_attempts: number;
  };
  local_scoring_compute_ms: number;
  resume_count: number;
  uncertain_prior_attempts: number;
  latest_event?: AuditEvent | null;
  updated_at: string;
}

export interface AppSettings {
  allowed_outbound_origins: string[];
  allowed_local_inference_origins: string[];
  retention_days: number;
  default_max_duration_seconds: number;
  display_timezone: string;
  default_n_prompts: number;
  default_m_reference: number;
  default_alpha: number;
  pricing_assumptions: {
    reference_cost_per_1k_tokens_usd?: number | null;
    target_cost_per_1k_tokens_usd?: number | null;
    estimated_prompt_tokens_per_call: number;
  };
  updated_at: string;
}

export interface SystemCapabilities {
  app_name: string;
  tagline: string;
  version: string;
  runtime: {
    python_version: string;
    sqlite_version: string;
    fastapi_version: string;
    pydantic_version: string;
    httpx_version: string;
    numpy_version: string;
    scipy_version: string;
    local_models_dir: string;
  };
  adapters: {
    synthetic_demo: { available: boolean; supports_local_logit_scoring: boolean; is_demo: boolean };
    openai_compatible: { available: boolean; supports_local_logit_scoring: boolean; is_demo: boolean };
    local_transformers: {
      available: boolean;
      missing_dependencies: string[];
      installed_versions: Record<string, string>;
      trust_remote_code: boolean;
      local_files_only: boolean;
    };
  };
  security: {
    secret_env_prefix: string;
    configured_provider_env_names: string[];
    allowed_outbound_origins: string[];
    allowed_local_inference_origins: string[];
    allowed_frontend_origins: string[];
    redirects_disabled: boolean;
    tls_verification_enforced: boolean;
  };
}
