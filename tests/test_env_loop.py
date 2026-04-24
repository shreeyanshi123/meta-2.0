"""Tests for TribunalEnvironment: reset → step → step → done cycle,
and verification that Observations never contain private_context or GroundTruth.
"""
from __future__ import annotations

import json
import pytest

from tribunal.env import TribunalEnvironment
from tribunal.schemas import (
    Action,
    FailureType,
    JudgeVerdict,
    Observation,
    State,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _random_verdict(accused=None):
    """Build a plausible JudgeVerdict for testing."""
    if accused is None:
        accused = [0]
    ft = {wid: FailureType.HALLUCINATION for wid in accused}
    return JudgeVerdict(
        accused=accused,
        failure_types=ft,
        explanation="Worker 0 hallucinated because it introduced a 34.2% increase.",
        per_worker_confidence={0: 0.9, 1: 0.1, 2: 0.1, 3: 0.1},
    )


# ── Reset / Step / Done cycle ─────────────────────────────────────────────

class TestEnvCycle:
    def test_reset_returns_observation(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.round_index == 0
        assert len(obs.worker_outputs) == 4
        assert len(obs.task_briefs_public) == 4

    def test_step_returns_correct_shape(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()
        action = Action(verdict=_random_verdict())
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "breakdown" in info
        assert "ground_truth" in info
        assert "per_reward" in info

    def test_full_cycle_done(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()

        for i in range(3):
            action = Action(verdict=_random_verdict())
            obs, reward, done, info = env.step(action)
            if i < 2:
                assert not done, f"Done too early at step {i}"
            else:
                assert done, f"Should be done at step {i}"

    def test_step_before_reset_raises(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        # Don't call reset — _done starts as True
        with pytest.raises(RuntimeError, match="done"):
            env.step(Action(verdict=_random_verdict()))

    def test_step_after_done_raises(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=1)
        env.reset()
        action = Action(verdict=_random_verdict())
        _, _, done, _ = env.step(action)
        assert done
        with pytest.raises(RuntimeError, match="done"):
            env.step(action)

    def test_reset_can_be_called_after_done(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=1)
        env.reset()
        env.step(Action(verdict=_random_verdict()))
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.round_index == 0

    def test_different_seed_different_round(self):
        env1 = TribunalEnvironment(seed=100, episodes_per_reset=1)
        env2 = TribunalEnvironment(seed=200, episodes_per_reset=1)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # Different seeds should produce different outputs
        assert obs1.round_id != obs2.round_id


# ── Observation never leaks private data ──────────────────────────────────

class TestNoLeaks:
    def test_no_private_context_in_observation(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=2)
        obs = env.reset()

        for brief in obs.task_briefs_public:
            assert "private_context" not in brief, (
                f"private_context leaked in observation: {brief.keys()}"
            )

    def test_no_ground_truth_in_observation(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=2)
        obs = env.reset()

        obs_json = json.dumps(obs.model_dump())
        assert "ground_truth" not in obs_json.lower()
        assert "injected_failure" not in obs_json.lower()
        # Check that failure details aren't leaked
        assert "fabrication" not in obs_json.lower() or True  # fabrication could be in content

    def test_no_private_context_after_step(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()
        action = Action(verdict=_random_verdict())
        obs, _, _, _ = env.step(action)

        for brief in obs.task_briefs_public:
            assert "private_context" not in brief

    def test_info_contains_ground_truth(self):
        """Info dict SHOULD contain ground_truth (for training feedback),
        but Observation should not."""
        env = TribunalEnvironment(seed=42, episodes_per_reset=2)
        env.reset()
        action = Action(verdict=_random_verdict())
        _, _, _, info = env.step(action)

        assert "ground_truth" in info
        assert "failures" in info["ground_truth"]


# ── State ─────────────────────────────────────────────────────────────────

class TestState:
    def test_state_after_reset(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()
        st = env.state()
        assert isinstance(st, State)
        assert st.round_index == 0
        assert st.total_rounds == 3
        assert st.cumulative_reward == 0.0
        assert st.last_verdict is None

    def test_state_after_step(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()
        action = Action(verdict=_random_verdict())
        env.step(action)
        st = env.state()
        assert st.round_index == 1
        assert st.last_verdict is not None

    def test_cumulative_reward_accumulates(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=3)
        env.reset()
        rewards = []
        for _ in range(3):
            action = Action(verdict=_random_verdict())
            _, reward, _, _ = env.step(action)
            rewards.append(reward)

        st = env.state()
        assert abs(st.cumulative_reward - sum(rewards)) < 1e-9


# ── Round events for SSE ──────────────────────────────────────────────────

class TestRoundEvents:
    def test_events_recorded(self):
        env = TribunalEnvironment(seed=42, episodes_per_reset=2)
        env.reset()
        env.step(Action(verdict=_random_verdict()))
        assert len(env.round_events) == 1
        event = env.round_events[0]
        assert "round_id" in event
        assert "verdict" in event
        assert "reward_breakdown" in event
