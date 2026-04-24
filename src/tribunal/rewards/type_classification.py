"""TypeClassificationReward — did the Judge predict the correct FailureType for each TP?

Score = (# correct types among TPs) / max(1, # true failures) ∈ [0, 1].

This denominator (true failures, not TPs) keeps the signal informative even
when F1 is imperfect: a Judge that detects 1 of 2 failures and gets its type
right scores 0.5 here, not 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from tribunal.schemas import FailureType, GroundTruth, JudgeVerdict


@dataclass
class TypeClassificationResult:
    """Detailed type-classification result."""
    correct_types: int
    total_true_failures: int
    score: float
    notes: List[str]


class TypeClassificationReward:
    """Score the Judge on failure-type accuracy for correctly identified workers.

    For each true-positive (correctly accused worker), check whether
    ``verdict.failure_types[worker_id]`` matches the ground-truth
    ``InjectedFailure.failure_type``.

    Returns
    -------
    TypeClassificationResult
        Contains the count of correct types, total true failures, the score,
        and diagnostic notes.
    """

    def score(
        self,
        verdict: JudgeVerdict,
        ground_truth: GroundTruth,
    ) -> TypeClassificationResult:
        predicted: Set[int] = set(verdict.accused)
        actual_map = {f.worker_id: f.failure_type for f in ground_truth.failures}
        actual_ids: Set[int] = set(actual_map.keys())

        tps = predicted & actual_ids
        total_true = len(ground_truth.failures)

        correct = 0
        notes: List[str] = []

        for wid in sorted(tps):
            predicted_type = verdict.failure_types.get(wid)
            true_type = actual_map[wid]
            if predicted_type == true_type:
                correct += 1
                notes.append(f"Worker {wid}: correct type {true_type.value}")
            else:
                notes.append(
                    f"Worker {wid}: predicted {predicted_type.value if predicted_type else 'None'}, "
                    f"actual {true_type.value}"
                )

        denominator = max(1, total_true)
        score = correct / denominator

        if total_true == 0:
            notes.append("No true failures — type classification is vacuously perfect.")
            score = 1.0

        return TypeClassificationResult(
            correct_types=correct,
            total_true_failures=total_true,
            score=score,
            notes=notes,
        )
