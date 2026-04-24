"""TribunalEnvironment — the core OpenEnv environment for the AI Agent
Oversight Tribunal.

Orchestrates TaskDispatcher → WorkerPool → FailureInjector → Judge (external)
→ RewardAggregator across multiple rounds per episode.

Guarantee: Observations NEVER leak GroundTruth or any ``private_context``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from tribunal.failure_injector import FailureInjector
from tribunal.rewards.aggregator import RewardAggregator
from tribunal.schemas import (
    Action,
    GroundTruth,
    JudgeVerdict,
    Observation,
    RewardBreakdown,
    State,
    TaskBrief,
    WorkerOutput,
    verdict_from_json,
    ParseError,
)
from tribunal.tasks import TaskDispatcher, get_public_brief
from tribunal.workers import WorkerPool

logger = logging.getLogger(__name__)


class TribunalEnvironment:
    """OpenEnv-compatible environment for the AI Agent Oversight Tribunal.

    Parameters
    ----------
    seed : int
        Base seed for deterministic task generation, worker output, and
        failure injection.
    episodes_per_reset : int
        Number of rounds (step calls) before the episode terminates.
    failure_rate : float
        Probability of corruption per round (passed to FailureInjector).
    use_llm_workers : bool
        If True, workers use the LLM backend instead of templates.
    """

    def __init__(
        self,
        seed: int = 42,
        episodes_per_reset: int = 1,
        failure_rate: float = 0.6,
        use_llm_workers: bool = False,
    ) -> None:
        self._base_seed = seed
        self._episodes_per_reset = episodes_per_reset
        self._failure_rate = failure_rate
        self._use_llm_workers = use_llm_workers

        # Mutable state — populated on reset()
        self._dispatcher: Optional[TaskDispatcher] = None
        self._pool: Optional[WorkerPool] = None
        self._injector: Optional[FailureInjector] = None
        self._aggregator: Optional[RewardAggregator] = None

        self._round_index: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = True
        self._last_verdict: Optional[JudgeVerdict] = None

        # Current round state (hidden from the Judge)
        self._current_briefs: List[TaskBrief] = []
        self._current_outputs: List[WorkerOutput] = []
        self._current_ground_truth: Optional[GroundTruth] = None
        self._current_observation: Optional[Observation] = None

        # SSE event history for streaming
        self._round_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the first Observation.

        Parameters
        ----------
        seed : int | None
            Override the base seed for this episode.
        """
        if seed is not None:
            self._base_seed = seed

        self._dispatcher = TaskDispatcher(seed=self._base_seed)
        self._pool = WorkerPool(
            seed=self._base_seed,
            use_llm=self._use_llm_workers,
        )
        self._injector = FailureInjector(
            seed=self._base_seed,
            failure_rate=self._failure_rate,
        )
        self._aggregator = RewardAggregator()

        self._round_index = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_verdict = None
        self._round_events = []

        # Generate round 0
        obs = self._generate_round()
        logger.info("Environment reset. seed=%d, rounds=%d", self._base_seed, self._episodes_per_reset)
        return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Process the Judge's verdict and advance to the next round.

        Parameters
        ----------
        action : Action
            Contains the JudgeVerdict.

        Returns
        -------
        tuple[Observation, float, bool, dict]
            (next_observation_or_terminal, reward_total, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if self._current_ground_truth is None or self._aggregator is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        verdict = action.verdict
        self._last_verdict = verdict

        # Score the verdict against ground truth
        parse_failed = False
        breakdown, trl_dict = self._aggregator.score(
            verdict,
            self._current_ground_truth,
            parse_failed=parse_failed,
            worker_outputs=self._current_outputs,
        )

        reward = max(-1.0, min(1.0, breakdown.total))
        self._cumulative_reward += reward

        # Record the round event for SSE streaming
        self._round_events.append({
            "round_index": self._round_index,
            "round_id": self._current_ground_truth.round_id,
            "task_briefs": [get_public_brief(b) for b in self._current_briefs],
            "worker_outputs": [o.model_dump() for o in self._current_outputs],
            "ground_truth": self._current_ground_truth.model_dump(),
            "verdict": verdict.model_dump(),
            "reward_breakdown": breakdown.model_dump(),
        })

        # Build info dict
        info: Dict[str, Any] = {
            "breakdown": breakdown.model_dump(),
            "ground_truth": self._current_ground_truth.model_dump(),
            "per_reward": trl_dict,
        }

        # Advance
        self._round_index += 1

        if self._round_index >= self._episodes_per_reset:
            self._done = True
            # Return a terminal observation (repeat last)
            obs = self._current_observation
            assert obs is not None
            logger.info(
                "Episode done. rounds=%d, cumulative_reward=%.4f",
                self._round_index, self._cumulative_reward,
            )
            return obs, reward, True, info

        # Generate the next round
        obs = self._generate_round()
        return obs, reward, False, info

    def state(self) -> State:
        """Return a serializable snapshot of the environment state."""
        return State(
            round_index=self._round_index,
            total_rounds=self._episodes_per_reset,
            cumulative_reward=self._cumulative_reward,
            last_verdict=self._last_verdict,
        )

    def close(self) -> None:
        """Clean up resources."""
        logger.info("Environment closed.")

    @property
    def round_events(self) -> List[Dict[str, Any]]:
        """Access recorded round events for SSE streaming."""
        return self._round_events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_round(self) -> Observation:
        """Dispatch briefs, run workers, inject failures, build observation."""
        assert self._dispatcher is not None
        assert self._pool is not None
        assert self._injector is not None

        round_id = f"ep-{self._base_seed}-round-{self._round_index:04d}"

        # 1. Dispatch task briefs
        briefs = self._dispatcher.dispatch(round_id)
        self._current_briefs = briefs

        # 2. Run workers
        clean_outputs = self._pool.run_all(briefs)

        # 3. Inject failures
        corrupted_outputs, ground_truth = self._injector.inject(briefs, clean_outputs)
        self._current_outputs = corrupted_outputs
        self._current_ground_truth = ground_truth

        # 4. Build observation — NEVER expose private_context or ground_truth
        public_briefs = [get_public_brief(b) for b in briefs]
        obs = Observation(
            round_id=round_id,
            task_briefs_public=public_briefs,
            worker_outputs=corrupted_outputs,
            round_index=self._round_index,
        )
        self._current_observation = obs
        return obs
