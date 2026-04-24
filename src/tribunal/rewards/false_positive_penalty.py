"""FalsePositivePenalty — negative signal proportional to wrongly accused clean workers.

penalty = −1.0 × (FP / 4) ∈ [−1, 0].

Strong disincentive against paranoia: a Judge that accuses all 4 workers
when only 2 are guilty gets penalty = −0.5 (2 FPs / 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from tribunal.schemas import GroundTruth, JudgeVerdict


@dataclass
class FalsePositivePenaltyResult:
    """Detailed false-positive penalty result."""
    false_positives: int
    score: float
    notes: List[str]


class FalsePositivePenalty:
    """Penalise the Judge for accusing clean workers.

    Returns
    -------
    FalsePositivePenaltyResult
        score ∈ [-1, 0], with notes listing which clean workers were wrongly accused.
    """

    def score(
        self,
        verdict: JudgeVerdict,
        ground_truth: GroundTruth,
    ) -> FalsePositivePenaltyResult:
        predicted: Set[int] = set(verdict.accused)
        actual: Set[int] = {f.worker_id for f in ground_truth.failures}

        fps = sorted(predicted - actual)
        fp_count = len(fps)

        penalty = -1.0 * (fp_count / 4.0)
        # Clamp to [-1, 0]
        penalty = max(-1.0, min(0.0, penalty))

        notes: List[str] = []
        if fps:
            notes.append(f"False positives: workers {fps} are clean but were accused")
            notes.append(f"FP penalty: -{fp_count}/4 = {penalty:.2f}")
        else:
            notes.append("No false positives — no penalty applied")

        return FalsePositivePenaltyResult(
            false_positives=fp_count,
            score=penalty,
            notes=notes,
        )
