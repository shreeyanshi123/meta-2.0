"""Tests for the FastAPI server: TestClient drives /reset, /step, /state, /health,
asserts response shapes and status codes.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tribunal.server import app


@pytest.fixture
def client():
    return TestClient(app)


def _random_verdict_payload(accused=None):
    if accused is None:
        accused = [0]
    return {
        "verdict": {
            "accused": accused,
            "failure_types": {str(w): "HALLUCINATION" for w in accused},
            "explanation": "Worker 0 hallucinated because of fabricated data.",
            "per_worker_confidence": {"0": 0.9, "1": 0.1, "2": 0.1, "3": 0.1},
        }
    }


# ── Health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "version" in data
        assert "round_index" in data


# ── Info ──────────────────────────────────────────────────────────────────

class TestInfo:
    def test_info(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert data["theme"] == "multi-agent-interactions"
        assert "worker_roles" in data
        assert "failure_types" in data
        assert len(data["worker_roles"]) == 4


# ── Reset ─────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        obs = r.json()
        assert "round_id" in obs
        assert "task_briefs_public" in obs
        assert "worker_outputs" in obs
        assert "round_index" in obs
        assert len(obs["worker_outputs"]) == 4
        assert len(obs["task_briefs_public"]) == 4

    def test_reset_no_private_context(self, client):
        r = client.post("/reset")
        obs = r.json()
        for brief in obs["task_briefs_public"]:
            assert "private_context" not in brief

    def test_reset_with_seed(self, client):
        r = client.post("/reset?seed=123")
        assert r.status_code == 200


# ── Step ──────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_after_reset(self, client):
        client.post("/reset")
        r = client.post("/step", json=_random_verdict_payload())
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert isinstance(data["reward"], float)
        assert isinstance(data["done"], bool)

    def test_step_info_has_breakdown(self, client):
        client.post("/reset")
        r = client.post("/step", json=_random_verdict_payload())
        info = r.json()["info"]
        assert "breakdown" in info
        assert "ground_truth" in info
        assert "per_reward" in info

    def test_step_breakdown_shape(self, client):
        client.post("/reset")
        r = client.post("/step", json=_random_verdict_payload())
        bd = r.json()["info"]["breakdown"]
        for key in ["identification", "type_classification", "explanation_quality",
                     "false_positive_penalty", "calibration", "anti_hack_penalty", "total"]:
            assert key in bd, f"Missing key: {key}"

    def test_step_per_reward_keys(self, client):
        client.post("/reset")
        r = client.post("/step", json=_random_verdict_payload())
        pr = r.json()["info"]["per_reward"]
        expected = {"r_identification", "r_type", "r_explain", "r_calib",
                    "r_fp_penalty", "r_antihack", "r_total"}
        assert set(pr.keys()) == expected

    def test_invalid_verdict_422(self, client):
        client.post("/reset")
        r = client.post("/step", json={"verdict": {"bad": "data"}})
        assert r.status_code == 422

    def test_full_episode_cycle(self, client):
        """Reset → 5 steps → done"""
        client.post("/reset")  # default episodes_per_reset=5
        for i in range(5):
            r = client.post("/step", json=_random_verdict_payload())
            assert r.status_code == 200
            data = r.json()
            if i < 4:
                assert data["done"] is False
            else:
                assert data["done"] is True


# ── State ─────────────────────────────────────────────────────────────────

class TestState:
    def test_state_after_reset(self, client):
        client.post("/reset")
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "round_index" in data
        assert "total_rounds" in data
        assert "cumulative_reward" in data
        assert data["round_index"] == 0

    def test_state_after_step(self, client):
        client.post("/reset")
        client.post("/step", json=_random_verdict_payload())
        r = client.get("/state")
        data = r.json()
        assert data["round_index"] == 1
