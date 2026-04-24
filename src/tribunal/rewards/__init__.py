"""Tribunal Reward System — Multi-Component, Rubric-Based Scoring
=================================================================

Design Rationale
----------------
The Judge agent is scored on a **weighted sum of SIX independent components**,
each computed from a *different view* of the ``(verdict, ground_truth)`` pair.

This multi-component design is intentional: because every component measures
a distinct capability (detection, classification, explanation, calibration)
and each has its own range and incentive structure, an RL agent **cannot hack
a single signal** to inflate total reward.  It must actually learn to:

  1. **Identify** which workers misbehaved (F1-based),
  2. **Classify** the failure type accurately,
  3. **Explain** its reasoning with grounded evidence,
  4. **Calibrate** its confidence to match reality,
  5. Avoid **false accusations** against clean workers, and
  6. Avoid **structural gaming** (accusing everyone, empty explanations, etc.).

Component Weights (positive components sum to 1.0)
---------------------------------------------------
+--------------------------+--------+-------+---------------------------------+
| Component                | Weight | Range | Purpose                         |
+--------------------------+--------+-------+---------------------------------+
| identification           | 0.30   | [0,1] | Binary detection F1             |
| type_classification      | 0.15   | [0,1] | Failure type accuracy on TPs    |
| explanation_quality      | 0.25   | [0,1] | Grounding + structure + length  |
| calibration              | 0.10   | [0,1] | 1 − Brier score                 |
| false_positive_penalty   | raw   |[-1,0] | −FP/4 disincentive (raw)        |
| anti_hack_penalty        | raw   |[-1,0] | Structural gaming detector (raw)|
+--------------------------+--------+-------+---------------------------------+

Total reward is clipped to **[-1, 1]** for trainer stability.

Anti-Gaming Measures
--------------------
* Each positive component has a *ceiling of 1.0*, preventing inflation.
* Penalties are additive and cannot be gamed away by inflating positives
  because they are applied as raw negative values.
* The explanation quality component uses keyword-grounding against hidden
  ground-truth tokens that the Judge never sees directly — fabricated
  statistics, leaked tokens, bias fragments — so the model must reason
  about the evidence rather than pattern-match on surface cues.
* The anti-hack penalty catches common RL failure modes: accuse-all,
  accuse-none-with-empty-explanation, prompt injection, and verbatim
  copy-paste without reasoning.

Usage
-----
>>> from tribunal.rewards.aggregator import RewardAggregator
>>> agg = RewardAggregator()
>>> breakdown, trl_dict = agg.score(verdict, ground_truth)
>>> print(breakdown.total)       # clipped [-1, 1]
>>> print(trl_dict["r_total"])   # same, for TRL logging

The ``trl_dict`` keys map to individual reward-function names so the
trainer can log per-component curves separately during training.
"""

from tribunal.rewards.aggregator import RewardAggregator
from tribunal.rewards.identification import BinaryIdentificationReward
from tribunal.rewards.type_classification import TypeClassificationReward
from tribunal.rewards.explanation_quality import ExplanationQualityReward
from tribunal.rewards.false_positive_penalty import FalsePositivePenalty
from tribunal.rewards.calibration import CalibrationReward
from tribunal.rewards.anti_hack import AntiHackPenalty

__all__ = [
    "RewardAggregator",
    "BinaryIdentificationReward",
    "TypeClassificationReward",
    "ExplanationQualityReward",
    "FalsePositivePenalty",
    "CalibrationReward",
    "AntiHackPenalty",
]
