"""Adapter interfaces and data structures for ModelTrace."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    """Normalized output from a single reference or target generation call."""

    response_text: str
    token_count: int
    finish_reason: str = "stop"
    declared_model: Optional[str] = None
    latency_ms: float = 0.0
    http_status: int = 200
    retry_count: int = 0
    observable_meta: Dict[str, Any] = field(default_factory=dict)
    precomputed_logit_rank: Optional[float] = None
    precomputed_norm_ll: Optional[float] = None
    scoring_compute_ms: float = 0.0


@dataclass
class ScoringResult:
    """Output from scoring a response text under reference full-vocabulary logits or length fallback."""

    raw_score: float
    normalized_log_likelihood: Optional[float]
    token_count: int
    scoring_compute_ms: float
    score_mode: str
    meta: Dict[str, Any] = field(default_factory=dict)


class AdapterError(RuntimeError):
    """Structured error raised by an adapter during generation, scoring, or connection check."""

    def __init__(
        self,
        error_code: str,
        message: str,
        http_status: Optional[int] = None,
        retryable: bool = False,
        retry_count: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.retry_count = retry_count
        self.latency_ms = latency_ms
