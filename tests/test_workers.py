"""Tests for WorkerAgent and WorkerPool."""

import re

import pytest

from tribunal.schemas import TaskBrief, WorkerOutput, WorkerRole
from tribunal.tasks import TaskDispatcher
from tribunal.workers import WorkerAgent, WorkerPool, NullLLMBackend, _extract_numbers


@pytest.fixture
def briefs():
    """Dispatch a deterministic set of 4 briefs."""
    return TaskDispatcher(seed=42).dispatch("test-round-001")


@pytest.fixture
def summariser_brief(briefs):
    return [b for b in briefs if b.role == WorkerRole.SUMMARISER][0]


@pytest.fixture
def researcher_brief(briefs):
    return [b for b in briefs if b.role == WorkerRole.RESEARCHER][0]


@pytest.fixture
def ticket_brief(briefs):
    return [b for b in briefs if b.role == WorkerRole.TICKET_RESOLVER][0]


@pytest.fixture
def negotiator_brief(briefs):
    return [b for b in briefs if b.role == WorkerRole.NEGOTIATOR][0]


class TestWorkerAgent:
    """Tests for individual WorkerAgent."""

    def test_deterministic_output(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=99)
        out1 = agent.run(summariser_brief)
        out2 = agent.run(summariser_brief)
        assert out1.content == out2.content
        assert out1.self_confidence == out2.self_confidence

    def test_summariser_non_empty(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=42)
        out = agent.run(summariser_brief)
        assert len(out.content) > 0
        assert out.role == WorkerRole.SUMMARISER

    def test_summariser_contains_numeric_fact(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=42)
        out = agent.run(summariser_brief)
        source_nums = set(_extract_numbers(summariser_brief.source_material or ""))
        output_nums = set(_extract_numbers(out.content))
        # At least 1 number from source should appear in the summary
        assert len(source_nums & output_nums) >= 1, (
            f"No numeric overlap.\nSource nums: {source_nums}\nOutput nums: {output_nums}"
        )

    def test_ticket_resolver_non_empty(self, ticket_brief):
        agent = WorkerAgent(role=WorkerRole.TICKET_RESOLVER, seed=42)
        out = agent.run(ticket_brief)
        assert len(out.content) > 0
        assert "Step" in out.content

    def test_negotiator_non_empty(self, negotiator_brief):
        agent = WorkerAgent(role=WorkerRole.NEGOTIATOR, seed=42)
        out = agent.run(negotiator_brief)
        assert len(out.content) > 0
        assert out.role == WorkerRole.NEGOTIATOR

    def test_researcher_exactly_3_citations(self, researcher_brief):
        agent = WorkerAgent(role=WorkerRole.RESEARCHER, seed=42)
        out = agent.run(researcher_brief)
        citation_markers = re.findall(r"\[\d+\]", out.content)
        assert len(citation_markers) == 3, f"Expected 3 citations, got {len(citation_markers)}: {out.content}"

    def test_researcher_non_empty(self, researcher_brief):
        agent = WorkerAgent(role=WorkerRole.RESEARCHER, seed=42)
        out = agent.run(researcher_brief)
        assert len(out.content) > 0

    def test_output_is_workeroutput(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=42)
        out = agent.run(summariser_brief)
        assert isinstance(out, WorkerOutput)

    def test_tokens_used_positive(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=42)
        out = agent.run(summariser_brief)
        assert out.tokens_used > 0

    def test_null_llm_backend_raises(self, summariser_brief):
        agent = WorkerAgent(role=WorkerRole.SUMMARISER, seed=42, use_llm=True)
        with pytest.raises(RuntimeError, match="NullLLMBackend"):
            agent.run(summariser_brief)


class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_run_all_returns_4(self, briefs):
        pool = WorkerPool(seed=42)
        outputs = pool.run_all(briefs)
        assert len(outputs) == 4

    def test_run_all_covers_all_roles(self, briefs):
        pool = WorkerPool(seed=42)
        outputs = pool.run_all(briefs)
        roles = {o.role for o in outputs}
        assert roles == set(WorkerRole)

    def test_run_all_deterministic(self, briefs):
        pool1 = WorkerPool(seed=42)
        pool2 = WorkerPool(seed=42)
        out1 = pool1.run_all(briefs)
        out2 = pool2.run_all(briefs)
        for a, b in zip(out1, out2):
            assert a.content == b.content
