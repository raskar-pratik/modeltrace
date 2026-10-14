"""Pydantic schemas and domain enumerations for ModelTrace."""

import math
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

from modeltrace.utils.security import validate_secret_env_var_name


class AdapterType(str, Enum):
    SYNTHETIC_DEMO = "synthetic_demo"
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL_TRANSFORMERS = "local_transformers"


class SyntheticScenario(str, Enum):
    GENUINE = "genuine"
    SUBSTITUTED = "substituted"
    MIXED_ROUTING = "mixed_routing"


class ScoreMode(str, Enum):
    LOCAL_LOGIT_RANK = "local_logit_rank"
    LENGTH_DISTRIBUTION_ONLY = "length_distribution_only"


class ExecutionState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class AuditVerdict(str, Enum):
    INCONSISTENT = "INCONSISTENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONSISTENT_WITHIN_TESTED_LIMITS = "CONSISTENT_WITHIN_TESTED_LIMITS"


class AttemptStatus(str, Enum):
    PLANNED = "PLANNED"
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNCERTAIN_PRIOR_ATTEMPT = "UNCERTAIN_PRIOR_ATTEMPT"
    CANCELLED = "CANCELLED"


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    adapter_type: AdapterType
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    model_identifier: str = Field(..., min_length=1, max_length=255)
    checkpoint_path: Optional[str] = None
    checkpoint_revision: Optional[str] = Field(default="main", max_length=120)
    tokenizer_path: Optional[str] = None
    chat_template: Optional[str] = None
    precision: str = Field(default="float32")
    device: str = Field(default="cpu")
    synthetic_scenario: Optional[SyntheticScenario] = SyntheticScenario.GENUINE
    mixed_routing_bad_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=15.0, ge=0.5, le=120.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    extra_config: Dict[str, Any] = Field(default_factory=dict)

    @validator("api_key_env_var")
    def _check_secret_env_var(cls, v: Optional[str]) -> Optional[str]:
        return validate_secret_env_var_name(v)

    @validator("precision")
    def _check_precision(cls, v: str) -> str:
        allowed = {"float32", "float16", "bfloat16"}
        if v not in allowed:
            raise ValueError(f"precision must be one of {sorted(allowed)}")
        return v

    @validator("device")
    def _check_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}")
        return v


class ProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    model_identifier: Optional[str] = Field(default=None, min_length=1, max_length=255)
    checkpoint_path: Optional[str] = None
    checkpoint_revision: Optional[str] = Field(default=None, max_length=120)
    tokenizer_path: Optional[str] = None
    chat_template: Optional[str] = None
    precision: Optional[str] = None
    device: Optional[str] = None
    synthetic_scenario: Optional[SyntheticScenario] = None
    mixed_routing_bad_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timeout_seconds: Optional[float] = Field(default=None, ge=0.5, le=120.0)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)
    extra_config: Optional[Dict[str, Any]] = None

    @validator("api_key_env_var")
    def _check_secret_env_var(cls, v: Optional[str]) -> Optional[str]:
        return validate_secret_env_var_name(v)


class ProbePromptItem(BaseModel):
    prompt_id: str = Field(..., min_length=1, max_length=80)
    family: str = Field(..., min_length=1, max_length=60)
    title: str = Field(..., min_length=1, max_length=160)
    prompt_text: str = Field(..., min_length=1, max_length=4000)
    system_hint: Optional[str] = Field(default=None, max_length=1000)
    expected_constraint: Optional[str] = Field(default=None, max_length=500)
    validation_status: str = Field(default="UNVALIDATED_FOR_IDENTITY_DISCRIMINATION")


class ProbeRevisionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    parent_revision_id: Optional[str] = None
    prompts: List[ProbePromptItem] = Field(..., min_items=1, max_items=200)

    @validator("prompts")
    def _check_unique_prompt_ids(cls, prompts: List[ProbePromptItem]) -> List[ProbePromptItem]:
        seen = set()
        for item in prompts:
            if item.prompt_id in seen:
                raise ValueError(f"Duplicate prompt_id '{item.prompt_id}' in probe revision.")
            seen.add(item.prompt_id)
        return prompts


class ProbeImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="Imported probe suite", max_length=500)
    format: str = Field(default="json")
    raw_content: str = Field(..., min_length=2, max_length=500000)


class CalibrationAlternativeResult(BaseModel):
    alternative_name: str = Field(..., min_length=1, max_length=140)
    alternative_description: str = Field(default="", max_length=400)
    n_trials: int = Field(..., ge=1, le=10000)
    n_rejected: int = Field(..., ge=0, le=10000)
    empirical_power: Optional[float] = Field(default=None)
    wilson_lower_95: Optional[float] = Field(default=None)
    clopper_pearson_lower_95: Optional[float] = Field(default=None)

    @validator("empirical_power", "wilson_lower_95", "clopper_pearson_lower_95", pre=True)
    def _check_finite_optional_float(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        val = float(v)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"Calibration proportion/bound must be a finite float in [0, 1]; got {v!r}.")
        return val


class CalibrationImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=140)
    description: str = Field(default="", max_length=500)
    score_mode: ScoreMode
    probe_revision_id: str
    probe_revision_hash: str
    n_prompts: int = Field(..., ge=1, le=200)
    m_reference: int = Field(..., ge=1, le=100)
    max_output_tokens: int = Field(default=128, ge=1, le=128)
    alpha: float = Field(..., gt=0.0, lt=0.5)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    reference_model_descriptor: str = Field(..., min_length=1, max_length=255)
    reference_profile_id: Optional[str] = None
    scoring_reference_profile_id: Optional[str] = None
    reference_identity: Optional[Dict[str, Any]] = None
    stat_engine_method: str = Field(default="cramer_von_mises_randomized_rank_mc")
    stat_engine_version: str = Field(default="1.0.0")
    is_demo: bool = Field(default=False)
    provenance_notes: str = Field(default="", max_length=1000)
    alternatives: List[CalibrationAlternativeResult] = Field(..., min_items=1, max_items=50)
    calibration_seed: int = Field(default=104729)


class DemoCalibrationCreateRequest(BaseModel):
    name: str = Field(default="DEMO Synthetic Screening Calibration")
    probe_revision_id: Optional[str] = None
    reference_profile_id: str = Field(default="prof_demo_reference")
    scoring_reference_profile_id: Optional[str] = None
    score_mode: ScoreMode = Field(default=ScoreMode.LOCAL_LOGIT_RANK)
    n_prompts: int = Field(default=20, ge=1, le=100)
    m_reference: int = Field(default=8, ge=1, le=50)
    max_output_tokens: int = Field(default=128, ge=1, le=128)
    alpha: float = Field(default=0.01, gt=0.0, lt=0.5)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    n_trials: int = Field(default=24, ge=20, le=200)
    calibration_seed: int = Field(default=20261014)


class AuditPreviewRequest(BaseModel):
    reference_profile_id: str
    target_profile_id: str
    probe_revision_id: Optional[str] = None
    calibration_id: Optional[str] = None
    score_mode: ScoreMode = Field(default=ScoreMode.LOCAL_LOGIT_RANK)
    scoring_reference_profile_id: Optional[str] = None
    n_prompts: int = Field(default=20, ge=1, le=200)
    m_reference: int = Field(default=8, ge=1, le=50)
    max_output_tokens: int = Field(default=128, ge=1, le=128)
    alpha: float = Field(default=0.01, gt=0.0, lt=0.5)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    mc_simulations: int = Field(default=9999, ge=9999, le=50000)
    max_target_attempts: Optional[int] = Field(default=None, ge=1, le=5000)
    max_reference_attempts: Optional[int] = Field(default=None, ge=1, le=10000)
    max_total_attempts: Optional[int] = Field(default=None, ge=1, le=15000)
    max_duration_seconds: int = Field(default=600, ge=5, le=7200)
    master_seed: int = Field(default=424242)
    reference_cost_per_1k_tokens_usd: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    target_cost_per_1k_tokens_usd: Optional[float] = Field(default=None, ge=0.0, le=500.0)


class AuditCreateRequest(AuditPreviewRequest):
    name: str = Field(default="ModelTrace Screening Audit", min_length=1, max_length=160)
    notes: str = Field(default="", max_length=1000)
    start_immediately: bool = Field(default=False)
    include_secondary_tests: bool = Field(default=False)


class PricingAssumptions(BaseModel):
    reference_cost_per_1k_tokens_usd: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    target_cost_per_1k_tokens_usd: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    estimated_prompt_tokens_per_call: int = Field(default=64, ge=1, le=4096)


class SettingsUpdateRequest(BaseModel):
    allowed_outbound_origins: Optional[List[str]] = None
    allowed_local_inference_origins: Optional[List[str]] = None
    retention_days: Optional[int] = Field(default=None, ge=1, le=3650)
    default_max_duration_seconds: Optional[int] = Field(default=None, ge=10, le=7200)
    display_timezone: Optional[str] = Field(default=None, min_length=1, max_length=64)
    default_n_prompts: Optional[int] = Field(default=None, ge=1, le=200)
    default_m_reference: Optional[int] = Field(default=None, ge=1, le=50)
    default_alpha: Optional[float] = Field(default=None, gt=0.0, lt=0.5)
    pricing_assumptions: Optional[PricingAssumptions] = None
