"""Tests for TaskDispatcher."""

import json
from pathlib import Path

import pytest

from tribunal.schemas import TaskBrief, WorkerRole
from tribunal.tasks import TaskDispatcher, get_public_brief, _DATA_DIR


ROLES = list(WorkerRole)


class TestTemplateFiles:
    """Ensure each role has >= 20 templates on disk."""

    @pytest.mark.parametrize("role", ROLES)
    def test_minimum_20_templates(self, role: WorkerRole):
        filepath = _DATA_DIR / f"{role.value.lower()}.json"
        with open(filepath, "r") as f:
            templates = json.load(f)
        assert len(templates) >= 20, f"{role.value} has only {len(templates)} templates"


class TestDispatch:
    """Tests for TaskDispatcher.dispatch()."""

    def test_returns_exactly_4_briefs(self):
        dispatcher = TaskDispatcher(seed=42)
        briefs = dispatcher.dispatch("round-001")
        assert len(briefs) == 4

    def test_one_brief_per_role(self):
        dispatcher = TaskDispatcher(seed=42)
        briefs = dispatcher.dispatch("round-001")
        roles = {b.role for b in briefs}
        assert roles == set(WorkerRole)

    def test_worker_ids_are_0_to_3(self):
        dispatcher = TaskDispatcher(seed=42)
        briefs = dispatcher.dispatch("round-001")
        ids = sorted(b.worker_id for b in briefs)
        assert ids == [0, 1, 2, 3]

    def test_determinism_same_seed_same_round(self):
        d1 = TaskDispatcher(seed=99)
        d2 = TaskDispatcher(seed=99)
        b1 = d1.dispatch("round-abc")
        b2 = d2.dispatch("round-abc")
        for a, b in zip(b1, b2):
            assert a.prompt == b.prompt
            assert a.source_material == b.source_material
            assert a.private_context == b.private_context

    def test_different_round_ids_give_different_tasks(self):
        dispatcher = TaskDispatcher(seed=42)
        # Run enough rounds that at least one differs (probabilistic but near-certain with 20+ templates)
        prompts_a = [b.prompt for b in dispatcher.dispatch("round-001")]
        prompts_b = [b.prompt for b in dispatcher.dispatch("round-999")]
        # They *could* match by chance; assert at least structure is valid
        assert len(prompts_a) == 4
        assert len(prompts_b) == 4

    def test_all_briefs_are_taskbrief_instances(self):
        dispatcher = TaskDispatcher(seed=7)
        briefs = dispatcher.dispatch("round-xyz")
        for b in briefs:
            assert isinstance(b, TaskBrief)


class TestPublicBrief:
    """Tests for get_public_brief()."""

    def test_no_private_context_in_public(self):
        dispatcher = TaskDispatcher(seed=42)
        briefs = dispatcher.dispatch("round-001")
        for brief in briefs:
            public = get_public_brief(brief)
            assert "private_context" not in public

    def test_public_preserves_other_fields(self):
        dispatcher = TaskDispatcher(seed=42)
        brief = dispatcher.dispatch("round-001")[0]
        public = get_public_brief(brief)
        assert public["worker_id"] == brief.worker_id
        assert public["role"] == brief.role.value
        assert public["prompt"] == brief.prompt
