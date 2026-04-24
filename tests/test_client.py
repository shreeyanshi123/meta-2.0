"""Tests for TribunalClient — boots the server via FastAPI TestClient,
runs 3 rounds, asserts shapes.

Uses httpx transport adapter to route client requests through FastAPI's
TestClient without needing a subprocess.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient as FastAPITestClient

from tribunal.server import app
from tribunal_shared.schemas import (
    FailureType,
    JudgeVerdict,
    Observation,
    State,
    StepResult,
)
from tribunal_client.client import TribunalClient


# ---------------------------------------------------------------------------
# Transport adapter: routes httpx requests through FastAPI's TestClient
# ---------------------------------------------------------------------------

class _TestTransport(httpx.BaseTransport):
    """httpx transport that delegates to FastAPI's TestClient."""

    def __init__(self, app):
        self._client = FastAPITestClient(app)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Build the relative URL
        url = str(request.url)
        # FastAPI TestClient needs relative paths
        path = request.url.raw_path.decode("ascii")
        if request.url.query:
            path = f"{path}?{request.url.query.decode('ascii')}"

        # Read body
        body = request.read()

        resp = self._client.request(
            method=request.method,
            url=path,
            content=body,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=resp.content,
        )



@pytest.fixture
def client():
    """TribunalClient backed by in-process FastAPI TestClient."""
    transport = _TestTransport(app)
    http_client = httpx.Client(transport=transport, base_url="http://testserver")
    tc = TribunalClient.__new__(TribunalClient)
    tc._base_url = "http://testserver"
    tc._http = http_client
    yield tc
    http_client.close()


def _random_verdict(accused=None):
    if accused is None:
        accused = [0]
    return JudgeVerdict(
        accused=accused,
        failure_types={wid: FailureType.HALLUCINATION for wid in accused},
        explanation="Worker 0 hallucinated because of fabricated statistics.",
        per_worker_confidence={0: 0.9, 1: 0.1, 2: 0.1, 3: 0.1},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClientHealth:
    def test_health(self, client):
        h = client.health()
        assert h["ok"] is True
        assert "version" in h

    def test_info(self, client):
        info = client.info()
        assert info["theme"] == "multi-agent-interactions"


class TestClientReset:
    def test_reset_returns_observation(self, client):
        obs = client.reset()
        assert isinstance(obs, Observation)
        assert obs.round_index == 0
        assert len(obs.worker_outputs) == 4

    def test_reset_with_seed(self, client):
        obs = client.reset(seed=123)
        assert isinstance(obs, Observation)

    def test_no_private_context(self, client):
        obs = client.reset()
        for brief in obs.task_briefs_public:
            assert "private_context" not in brief


class TestClientStep:
    def test_step_returns_step_result(self, client):
        client.reset()
        result = client.step(_random_verdict())
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_info_has_breakdown(self, client):
        client.reset()
        result = client.step(_random_verdict())
        assert "breakdown" in result.info
        assert "per_reward" in result.info
        assert "ground_truth" in result.info

    def test_per_reward_keys(self, client):
        client.reset()
        result = client.step(_random_verdict())
        pr = result.info["per_reward"]
        expected = {"r_identification", "r_type", "r_explain", "r_calib",
                    "r_fp_penalty", "r_antihack", "r_total"}
        assert set(pr.keys()) == expected


class TestClient3RoundCycle:
    """Run 3 rounds and verify the full cycle."""

    def test_3_round_cycle(self, client):
        obs = client.reset(seed=42)
        assert obs.round_index == 0

        rewards = []
        for i in range(3):
            verdict = _random_verdict(accused=[i % 4])
            result = client.step(verdict)
            rewards.append(result.reward)

            assert isinstance(result.reward, float)
            assert -1.0 <= result.reward <= 1.0

        assert len(rewards) == 3

    def test_state_advances(self, client):
        client.reset()
        s0 = client.state()
        assert s0.round_index == 0

        client.step(_random_verdict())
        s1 = client.state()
        assert s1.round_index == 1

        client.step(_random_verdict())
        s2 = client.state()
        assert s2.round_index == 2


class TestClientImportIsolation:
    """Verify the client does NOT import server-side modules."""

    def test_no_env_import(self):
        import tribunal_client.client as mod
        source = open(mod.__file__).read()
        assert "tribunal.env" not in source
        assert "tribunal.server" not in source
        assert "tribunal.failure_injector" not in source
        assert "tribunal.rewards" not in source

    def test_only_shared_imports(self):
        import tribunal_client.client as mod
        source = open(mod.__file__).read()
        assert "tribunal_shared" in source
