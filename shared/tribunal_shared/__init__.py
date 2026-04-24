"""Tribunal Shared — Pydantic schemas used by both server and client.

This package is the single source of truth for all data models in the
AI Agent Oversight Tribunal.  Neither the server nor the client should
define their own schema types — they import from here.
"""

from tribunal_shared.schemas import (
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
