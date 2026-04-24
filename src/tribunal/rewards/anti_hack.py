"""AntiHackPenalty — detects and penalises common RL failure modes.

Multiple checks; each trigger adds a penalty that accumulates (floor at −1.0).

Triggers
--------
1. Verdict accuses all 4 workers when the round has ≥1 clean worker: −0.5
2. Verdict accuses 0 workers when round has ≥1 failure AND explanation
   is very short (<30 chars): −0.3  (silent-shrug strategy)
3. Explanation contains prompt-injection tokens: −0.5
4. Explanation copies large verbatim chunks (>100 chars) from worker
   outputs without adding reasoning: −0.5
   (Not implemented in scoring since we don't pass worker outputs here;
    left as a hook for the aggregator to call separately.)

The ``parse_failed`` flag can be set by the caller if the verdict could not
be parsed from raw JSON — this adds −0.5.

Total ∈ [−1, 0].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from tribunal.schemas import GroundTruth, JudgeVerdict, WorkerOutput


# Forbidden tokens — prompt injection / special tokens
_FORBIDDEN_PATTERNS = [
    re.compile(r"IGNORE\s+PREVIOUS", re.IGNORECASE),
    re.compile(r"IGNORE\s+ALL", re.IGNORECASE),
    re.compile(r"DISREGARD", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>"),
    re.compile(r"<\|im_start\|>"),
    re.compile(r"<\|im_end\|>"),
    re.compile(r"\[INST\]"),
    re.compile(r"\[/INST\]"),
    re.compile(r"<<SYS>>"),
    re.compile(r"</s>"),
    re.compile(r"SYSTEM\s*:", re.IGNORECASE),
]


@dataclass
class AntiHackResult:
    """Detailed anti-hack penalty result."""
    triggers: List[str]
    score: float
    notes: List[str]


class AntiHackPenalty:
    """Detect and penalise structural gaming attempts.

    Parameters
    ----------
    worker_outputs : list[WorkerOutput] | None
        If provided, enables the verbatim-copy check.

    Returns
    -------
    AntiHackResult
        score ∈ [-1, 0], with triggers and notes listing what was detected.
    """

    def score(
        self,
        verdict: JudgeVerdict,
        ground_truth: GroundTruth,
        *,
        parse_failed: bool = False,
        worker_outputs: Optional[List[WorkerOutput]] = None,
    ) -> AntiHackResult:
        penalty = 0.0
        triggers: List[str] = []
        notes: List[str] = []

        # --- 1. Parse failure ---
        if parse_failed:
            penalty -= 0.5
            triggers.append("PARSE_FAILED")
            notes.append("Verdict did not parse as valid JSON: -0.5")

        # --- 2. Accuse-all when clean workers exist ---
        clean_count = len(ground_truth.clean_worker_ids)
        if len(verdict.accused) == 4 and clean_count > 0:
            penalty -= 0.5
            triggers.append("ACCUSE_ALL")
            notes.append(
                f"Accused all 4 workers but {clean_count} are clean: -0.5"
            )

        # --- 3. Silent shrug: accuse none + short explanation when failures exist ---
        true_failure_count = len(ground_truth.failures)
        if (
            len(verdict.accused) == 0
            and true_failure_count > 0
            and len(verdict.explanation) < 30
        ):
            penalty -= 0.3
            triggers.append("SILENT_SHRUG")
            notes.append(
                "Accused 0 workers with <30-char explanation but failures exist: -0.3"
            )

        # --- 4. Prompt-injection / reserved tokens ---
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(verdict.explanation):
                penalty -= 0.5
                triggers.append("PROMPT_INJECTION")
                notes.append(
                    f"Explanation contains forbidden token matching '{pattern.pattern}': -0.5"
                )
                break  # Only penalise once for prompt injection

        # --- 5. Verbatim copy from worker outputs ---
        if worker_outputs:
            explanation = verdict.explanation
            for wo in worker_outputs:
                content = wo.content
                # Check if any substring of length >100 from worker output
                # appears verbatim in the explanation
                if len(content) > 100:
                    # Sliding window check for efficiency
                    chunk_size = 100
                    for start in range(0, len(content) - chunk_size + 1, 50):
                        chunk = content[start : start + chunk_size]
                        if chunk in explanation:
                            penalty -= 0.5
                            triggers.append("VERBATIM_COPY")
                            notes.append(
                                f"Explanation copies >100 chars verbatim from "
                                f"worker {wo.worker_id}: -0.5"
                            )
                            break
                    else:
                        continue
                    break  # Only penalise once for verbatim copy

        # Floor at -1.0
        score = max(-1.0, penalty)

        if not triggers:
            notes.append("No anti-hack triggers detected — no penalty")

        return AntiHackResult(
            triggers=triggers,
            score=score,
            notes=notes,
        )
