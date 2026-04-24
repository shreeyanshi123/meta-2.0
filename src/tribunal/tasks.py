"""TaskDispatcher: generates a fresh TaskBrief for each of the 4 workers every round."""

import hashlib
import json
import random
from pathlib import Path
from typing import List

from tribunal.schemas import TaskBrief, WorkerRole

# Path to the JSON template files
_DATA_DIR = Path(__file__).parent / "data" / "tasks"

# Mapping from WorkerRole to JSON filename
_ROLE_TO_FILE = {
    WorkerRole.SUMMARISER: "summariser.json",
    WorkerRole.TICKET_RESOLVER: "ticket_resolver.json",
    WorkerRole.NEGOTIATOR: "negotiator.json",
    WorkerRole.RESEARCHER: "researcher.json",
}

# Fixed mapping from WorkerRole to worker_id for determinism
_ROLE_TO_WORKER_ID = {
    WorkerRole.SUMMARISER: 0,
    WorkerRole.TICKET_RESOLVER: 1,
    WorkerRole.NEGOTIATOR: 2,
    WorkerRole.RESEARCHER: 3,
}


def _load_templates(role: WorkerRole) -> list[dict]:
    """Load task templates for a given role from the JSON data file."""
    filepath = _DATA_DIR / _ROLE_TO_FILE[role]
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_seed(base_seed: int, round_id: str) -> int:
    """Derive a deterministic per-round seed from the base seed and round_id."""
    h = hashlib.sha256(f"{base_seed}:{round_id}".encode()).hexdigest()
    return int(h[:8], 16)


def get_public_brief(brief: TaskBrief) -> dict:
    """Return a redacted version of the TaskBrief for the Observation.

    Drops ``private_context`` so that the Judge never sees hidden information.
    """
    data = brief.model_dump()
    data.pop("private_context", None)
    return data


class TaskDispatcher:
    """Generates one TaskBrief per WorkerRole for each round.

    Parameters
    ----------
    seed : int
        Base seed for reproducible template selection.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        # Pre-load all templates once
        self._templates: dict[WorkerRole, list[dict]] = {
            role: _load_templates(role) for role in WorkerRole
        }

    def dispatch(self, round_id: str) -> List[TaskBrief]:
        """Dispatch exactly 4 TaskBriefs (one per WorkerRole) for a round.

        Selection is deterministic given ``(self.seed, round_id)``.
        """
        round_seed = _derive_seed(self.seed, round_id)
        rng = random.Random(round_seed)

        briefs: list[TaskBrief] = []
        for role in WorkerRole:
            templates = self._templates[role]
            template = rng.choice(templates)
            brief = TaskBrief(
                worker_id=_ROLE_TO_WORKER_ID[role],
                role=role,
                prompt=template["prompt"],
                source_material=template.get("source_material"),
                private_context=template.get("private_context", {}),
            )
            briefs.append(brief)

        return briefs
