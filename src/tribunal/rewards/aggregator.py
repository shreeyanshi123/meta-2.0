"""RewardAggregator — weighted sum of all six reward components.

Weights (positive components sum to 1.0):
    identification:          0.30
    type_classification:     0.15
    explanation_quality:     0.25
    calibration:             0.10
    false_positive_penalty:  0.15  (applied as raw negative, not re-weighted)
    anti_hack_penalty:       0.05  (applied as raw negative)

Total is clipped to [-1, 1] for trainer stability.

Returns both a fully-populated :class:`RewardBreakdown` and a TRL-compatible
dict where each component is its own reward-function name for per-component
logging during training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tribunal.schemas import (
    GroundTruth,
    JudgeVerdict,
    RewardBreakdown,
    WorkerOutput,
)

from tribunal.rewards.identification import BinaryIdentificationReward
from tribunal.rewards.type_classification import TypeClassificationReward
from tribunal.rewards.explanation_quality import ExplanationQualityReward
from tribunal.rewards.false_positive_penalty import FalsePositivePenalty
from tribunal.rewards.calibration import CalibrationReward
from tribunal.rewards.anti_hack import AntiHackPenalty


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "identification": 0.30,
    "type_classification": 0.15,
    "explanation_quality": 0.25,
    "calibration": 0.10,
    "false_positive_penalty": 0.15,
    "anti_hack_penalty": 0.05,
}


class RewardAggregator:
    """Aggregate all six reward components into a single score.

    Parameters
    ----------
    weights : dict | None
        Override default component weights.  If None, uses DEFAULT_WEIGHTS.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self._identification = BinaryIdentificationReward()
        self._type_classification = TypeClassificationReward()
        self._explanation_quality = ExplanationQualityReward()
        self._false_positive_penalty = FalsePositivePenalty()
        self._calibration = CalibrationReward()
        self._anti_hack = AntiHackPenalty()

    def score(
        self,
        verdict: JudgeVerdict,
        ground_truth: GroundTruth,
        *,
        parse_failed: bool = False,
        worker_outputs: Optional[List[WorkerOutput]] = None,
    ) -> Tuple[RewardBreakdown, Dict[str, float]]:
        """Score a verdict against ground truth.

        Parameters
        ----------
        verdict : JudgeVerdict
            The Judge's structured output.
        ground_truth : GroundTruth
            The hidden ground truth for this round.
        parse_failed : bool
            Set to True if the verdict could not be parsed from raw JSON.
        worker_outputs : list[WorkerOutput] | None
            If provided, enables the verbatim-copy anti-hack check.

        Returns
        -------
        tuple[RewardBreakdown, dict[str, float]]
            The breakdown (with all weighted components and notes) and a
            TRL-compatible dict for per-component logging.
        """
        all_notes: List[str] = []

        # --- 1. Identification ---
        id_result = self._identification.score(verdict, ground_truth)
        r_identification = id_result.score * self.weights["identification"]
        all_notes.extend(f"[identification] {n}" for n in id_result.notes)

        # --- 2. Type classification ---
        tc_result = self._type_classification.score(verdict, ground_truth)
        r_type = tc_result.score * self.weights["type_classification"]
        all_notes.extend(f"[type_classification] {n}" for n in tc_result.notes)

        # --- 3. Explanation quality ---
        eq_result = self._explanation_quality.score(verdict, ground_truth)
        r_explain = eq_result.score * self.weights["explanation_quality"]
        all_notes.extend(f"[explanation_quality] {n}" for n in eq_result.notes)

        # --- 4. Calibration ---
        cal_result = self._calibration.score(verdict, ground_truth)
        r_calib = cal_result.score * self.weights["calibration"]
        all_notes.extend(f"[calibration] {n}" for n in cal_result.notes)

        # --- 5. False positive penalty (raw negative — not re-weighted) ---
        # The spec says penalties are "applied as raw negative".  The weight
        # acts as a severity scaler, not a proportional weight.
        fp_result = self._false_positive_penalty.score(verdict, ground_truth)
        r_fp = fp_result.score  # raw, already in [-1, 0]
        all_notes.extend(f"[false_positive_penalty] {n}" for n in fp_result.notes)

        # --- 6. Anti-hack penalty (raw negative — not re-weighted) ---
        ah_result = self._anti_hack.score(
            verdict,
            ground_truth,
            parse_failed=parse_failed,
            worker_outputs=worker_outputs,
        )
        r_antihack = ah_result.score  # raw, already in [-1, 0]
        all_notes.extend(f"[anti_hack] {n}" for n in ah_result.notes)

        # --- Aggregate ---
        raw_total = (
            r_identification
            + r_type
            + r_explain
            + r_calib
            + r_fp
            + r_antihack
        )
        clipped_total = max(-1.0, min(1.0, raw_total))

        # Build breakdown — store the WEIGHTED component values
        # The RewardBreakdown.validate_total validator checks that total == sum
        breakdown = RewardBreakdown(
            identification=r_identification,
            type_classification=r_type,
            explanation_quality=r_explain,
            false_positive_penalty=r_fp,
            calibration=r_calib,
            anti_hack_penalty=r_antihack,
            total=r_identification + r_type + r_explain + r_fp + r_calib + r_antihack,
            notes=all_notes,
        )

        # TRL-compatible dict with per-component reward function names
        trl_dict: Dict[str, float] = {
            "r_identification": r_identification,
            "r_type": r_type,
            "r_explain": r_explain,
            "r_calib": r_calib,
            "r_fp_penalty": r_fp,
            "r_antihack": r_antihack,
            "r_total": clipped_total,
        }

        return breakdown, trl_dict
