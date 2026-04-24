"""AI Agent Oversight Tribunal — OpenEnv environment for Multi-Agent Interactions.

A multi-agent environment where four specialised AI workers complete tasks,
a hidden failure injector corrupts some outputs, and a Judge agent must
identify, classify, and explain the failures.
"""

__version__ = "0.1.0"

from tribunal.env import TribunalEnvironment
from tribunal.schemas import (
    Action,
    FailureType,
    GroundTruth,
    JudgeVerdict,
    Observation,
    RewardBreakdown,
    State,
    WorkerRole,
)

__all__ = [
    "TribunalEnvironment",
    "Action",
    "FailureType",
    "GroundTruth",
    "JudgeVerdict",
    "Observation",
    "RewardBreakdown",
    "State",
    "WorkerRole",
]
