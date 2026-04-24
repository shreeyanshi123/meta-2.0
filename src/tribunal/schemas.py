"""Re-export shim — all schemas live in ``tribunal_shared.schemas``.

This module re-exports every name so that existing ``from tribunal.schemas
import X`` statements continue to work unchanged.
"""

# Re-export everything from the shared canonical source.
# Existing imports like `from tribunal.schemas import FailureType` still work.
from tribunal_shared.schemas import (  # noqa: F401
    Action,
    FailureType,
    GroundTruth,
    InjectedFailure,
    JudgeVerdict,
    Observation,
    ParseError,
    RewardBreakdown,
    State,
    StepResult,
    TaskBrief,
    WorkerOutput,
    WorkerRole,
    verdict_from_json,
)

__all__ = [
    "Action",
    "FailureType",
    "GroundTruth",
    "InjectedFailure",
    "JudgeVerdict",
    "Observation",
    "ParseError",
    "RewardBreakdown",
    "State",
    "StepResult",
    "TaskBrief",
    "WorkerOutput",
    "WorkerRole",
    "verdict_from_json",
]
