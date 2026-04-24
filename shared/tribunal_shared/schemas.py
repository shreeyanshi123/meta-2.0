"""Canonical Pydantic schemas for the AI Agent Oversight Tribunal.

This is the SINGLE SOURCE OF TRUTH for all data models shared between
the server and client.  Do not duplicate these types.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when a raw JSON string cannot be parsed into a JudgeVerdict."""

    def __init__(self, message: str, original_string: str):
        super().__init__(message)
        self.original_string = original_string


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    """Types of failures that can be injected."""
    HALLUCINATION = "HALLUCINATION"
    COLLUSION = "COLLUSION"
    MANIPULATION = "MANIPULATION"
    SILENCE = "SILENCE"
    CLEAN = "CLEAN"


class WorkerRole(str, Enum):
    """Roles for the workers."""
    SUMMARISER = "SUMMARISER"
    TICKET_RESOLVER = "TICKET_RESOLVER"
    NEGOTIATOR = "NEGOTIATOR"
    RESEARCHER = "RESEARCHER"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class TaskBrief(BaseModel):
    """What a worker receives."""
    model_config = ConfigDict(extra="forbid")

    worker_id: int = Field(ge=0, le=3)
    role: WorkerRole
    prompt: str
    source_material: Optional[str] = None
    private_context: Dict[str, Any]


class WorkerOutput(BaseModel):
    """What a worker submits."""
    model_config = ConfigDict(extra="forbid")

    worker_id: int = Field(ge=0, le=3)
    role: WorkerRole
    content: str
    self_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tokens_used: int = Field(ge=0)


class InjectedFailure(BaseModel):
    """Ground-truth record of an injected failure."""
    model_config = ConfigDict(extra="forbid")

    worker_id: int = Field(ge=0, le=3)
    failure_type: FailureType
    details: Dict[str, Any]


class GroundTruth(BaseModel):
    """Full hidden state of one round."""
    model_config = ConfigDict(extra="forbid")

    round_id: str
    seed: int
    failures: List[InjectedFailure]
    clean_worker_ids: List[int]


class JudgeVerdict(BaseModel):
    """The Judge's structured output."""
    model_config = ConfigDict(extra="forbid")

    accused: List[int]
    failure_types: Dict[int, FailureType]
    explanation: str
    per_worker_confidence: Dict[int, float]

    @model_validator(mode="after")
    def validate_verdict(self):
        accused_set = set(self.accused)
        failures_set = set(self.failure_types.keys())

        if accused_set != failures_set:
            raise ValueError("The 'accused' list and 'failure_types' keys must match exactly.")

        for worker_id in self.accused:
            if worker_id < 0 or worker_id > 3:
                raise ValueError(f"Worker ID {worker_id} in accused must be between 0 and 3.")

        for worker_id in self.per_worker_confidence.keys():
            if worker_id < 0 or worker_id > 3:
                raise ValueError(f"Worker ID {worker_id} in per_worker_confidence must be between 0 and 3.")
            conf = self.per_worker_confidence[worker_id]
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence for Worker ID {worker_id} must be between 0 and 1.")

        confidence_set = set(self.per_worker_confidence.keys())
        if confidence_set != {0, 1, 2, 3}:
            raise ValueError("per_worker_confidence must contain exactly keys 0, 1, 2, and 3.")

        return self


class Observation(BaseModel):
    """OpenEnv Observation."""
    model_config = ConfigDict(extra="forbid")

    round_id: str
    task_briefs_public: List[Dict[str, Any]]
    worker_outputs: List[WorkerOutput]
    round_index: int

    @model_validator(mode="after")
    def ensure_no_private_context(self):
        for brief in self.task_briefs_public:
            if "private_context" in brief:
                raise ValueError("Observation task_briefs_public cannot contain private_context.")
        return self


class Action(BaseModel):
    """OpenEnv Action."""
    model_config = ConfigDict(extra="forbid")

    verdict: JudgeVerdict


class State(BaseModel):
    """OpenEnv State."""
    model_config = ConfigDict(extra="forbid")

    round_index: int
    total_rounds: int
    cumulative_reward: float
    last_verdict: Optional[JudgeVerdict] = None


class RewardBreakdown(BaseModel):
    """Reward breakdown metrics."""
    model_config = ConfigDict(extra="forbid")

    identification: float
    type_classification: float
    explanation_quality: float
    false_positive_penalty: float
    calibration: float
    anti_hack_penalty: float
    total: float
    notes: List[str]

    @model_validator(mode="after")
    def validate_total(self):
        expected_total = (self.identification + self.type_classification +
                          self.explanation_quality + self.false_positive_penalty +
                          self.calibration + self.anti_hack_penalty)
        if abs(self.total - expected_total) > 1e-6:
            raise ValueError(f"Total {self.total} does not match sum of components {expected_total}")
        return self


class StepResult(BaseModel):
    """Result returned by TribunalClient.step()."""
    model_config = ConfigDict(extra="allow")

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def verdict_from_json(raw: str) -> JudgeVerdict:
    """Robustly parses messy LLM JSON and returns a JudgeVerdict."""
    original_string = raw

    # Strip markdown code blocks (handles indented fences)
    raw = re.sub(r"^\s*```(?:json)?\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Remove simple trailing commas before } or ]
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    try:
        data = json.loads(raw)

        # Pydantic v2 usually coerces dict keys to the specified type (e.g. int)
        # when using .model_validate(), but standard json parses to string keys.
        # Doing explicit conversion avoids any edge cases.
        if "failure_types" in data and isinstance(data["failure_types"], dict):
            data["failure_types"] = {int(k): v for k, v in data["failure_types"].items()}
        if "per_worker_confidence" in data and isinstance(data["per_worker_confidence"], dict):
            data["per_worker_confidence"] = {int(k): v for k, v in data["per_worker_confidence"].items()}

        return JudgeVerdict.model_validate(data)
    except Exception as e:
        raise ParseError(f"Failed to parse verdict: {str(e)}", original_string)
