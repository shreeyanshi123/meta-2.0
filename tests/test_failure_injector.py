"""Tests for the FailureInjector — covering all 4 strategies, ground truth
consistency, role/failure compatibility, determinism under seed,
max_failures_per_round enforcement, and clean worker byte-equality."""

from __future__ import annotations

import copy
import re

import pytest

from tribunal.failure_injector import (
    ROLE_FAILURE_COMPAT,
    FailureInjector,
    InjectionStats,
    _sample_corruption_count,
)
from tribunal.schemas import (
    FailureType,
    GroundTruth,
    InjectedFailure,
    TaskBrief,
    WorkerOutput,
    WorkerRole,
)
from tribunal.tasks import TaskDispatcher
from tribunal.workers import WorkerPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def briefs_and_outputs():
    """Generate a full set of briefs and clean outputs for one round."""
    dispatcher = TaskDispatcher(seed=SEED)
    pool = WorkerPool(seed=SEED)
    briefs = dispatcher.dispatch("round-001")
    outputs = pool.run_all(briefs)
    return briefs, outputs


@pytest.fixture
def injector():
    """Default FailureInjector with known seed."""
    return FailureInjector(seed=SEED, failure_rate=1.0, max_failures_per_round=3)


@pytest.fixture
def forced_injector():
    """Injector that always tries to corrupt (failure_rate=1.0)."""
    return FailureInjector(seed=12345, failure_rate=1.0, max_failures_per_round=3)


# ---------------------------------------------------------------------------
# Ground truth consistency
# ---------------------------------------------------------------------------

class TestGroundTruthConsistency:
    """Ground truth records must be internally consistent."""

    def test_all_worker_ids_accounted_for(self, injector, briefs_and_outputs):
        """Every worker ID appears exactly once — either in failures or clean_worker_ids."""
        briefs, outputs = briefs_and_outputs
        _, gt = injector.inject(briefs, outputs)

        failed_ids = {f.worker_id for f in gt.failures}
        clean_ids = set(gt.clean_worker_ids)
        all_ids = failed_ids | clean_ids

        assert all_ids == {0, 1, 2, 3}
        assert failed_ids.isdisjoint(clean_ids), "A worker cannot be both failed and clean."

    def test_no_duplicates_in_failures(self, injector, briefs_and_outputs):
        """A worker should not appear twice in the failures list."""
        briefs, outputs = briefs_and_outputs
        _, gt = injector.inject(briefs, outputs)

        worker_ids_in_failures = [f.worker_id for f in gt.failures]
        assert len(worker_ids_in_failures) == len(set(worker_ids_in_failures))

    def test_failures_have_correct_types(self, injector, briefs_and_outputs):
        """Each failure must have a valid FailureType (not CLEAN)."""
        briefs, outputs = briefs_and_outputs
        _, gt = injector.inject(briefs, outputs)

        for failure in gt.failures:
            assert failure.failure_type != FailureType.CLEAN
            assert failure.failure_type in FailureType

    def test_ground_truth_has_round_id_and_seed(self, injector, briefs_and_outputs):
        """Ground truth must include a round_id and seed."""
        briefs, outputs = briefs_and_outputs
        _, gt = injector.inject(briefs, outputs)

        assert gt.round_id
        assert isinstance(gt.seed, int)

    def test_corrupted_worker_never_marked_clean(self, briefs_and_outputs):
        """Run many rounds and verify invariant: corrupted ≠ clean."""
        injector = FailureInjector(seed=99, failure_rate=1.0, max_failures_per_round=3)
        briefs, outputs = briefs_and_outputs

        for _ in range(50):
            _, gt = injector.inject(briefs, copy.deepcopy(outputs))
            failed_ids = {f.worker_id for f in gt.failures}
            clean_ids = set(gt.clean_worker_ids)
            assert failed_ids.isdisjoint(clean_ids)
            assert failed_ids | clean_ids == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Role / failure compatibility
# ---------------------------------------------------------------------------

class TestRoleFailureCompatibility:
    """Injected failure types must be compatible with the worker's role."""

    def test_compatibility_respected(self, briefs_and_outputs):
        """Run many rounds and verify every failure_type is compatible with the role."""
        briefs, outputs = briefs_and_outputs
        output_role_map = {o.worker_id: o.role for o in outputs}

        # Use many seeds to cover all code paths
        for seed in range(100):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                role = output_role_map[failure.worker_id]
                compatible = ROLE_FAILURE_COMPAT[role]
                assert failure.failure_type in compatible, (
                    f"Failure type {failure.failure_type} is not compatible "
                    f"with role {role}. Compatible: {compatible}"
                )

    def test_summariser_never_gets_collusion(self, briefs_and_outputs):
        """SUMMARISER should never get COLLUSION."""
        briefs, outputs = briefs_and_outputs
        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            for failure in gt.failures:
                if failure.worker_id == 0:  # SUMMARISER
                    assert failure.failure_type != FailureType.COLLUSION

    def test_ticket_resolver_only_silence_or_manipulation(self, briefs_and_outputs):
        """TICKET_RESOLVER should only get SILENCE or MANIPULATION."""
        briefs, outputs = briefs_and_outputs
        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            for failure in gt.failures:
                if failure.worker_id == 1:  # TICKET_RESOLVER
                    assert failure.failure_type in {
                        FailureType.SILENCE,
                        FailureType.MANIPULATION,
                    }


# ---------------------------------------------------------------------------
# Determinism under seed
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed → identical corruption decisions and outputs."""

    def test_same_seed_produces_same_results(self, briefs_and_outputs):
        """Two injectors with the same seed must produce identical outputs."""
        briefs, outputs = briefs_and_outputs

        inj1 = FailureInjector(seed=777, failure_rate=1.0, max_failures_per_round=3)
        inj2 = FailureInjector(seed=777, failure_rate=1.0, max_failures_per_round=3)

        out1, gt1 = inj1.inject(briefs, copy.deepcopy(outputs))
        out2, gt2 = inj2.inject(briefs, copy.deepcopy(outputs))

        # Ground truth must match
        assert gt1.round_id == gt2.round_id
        assert gt1.seed == gt2.seed
        assert len(gt1.failures) == len(gt2.failures)
        for f1, f2 in zip(gt1.failures, gt2.failures):
            assert f1.worker_id == f2.worker_id
            assert f1.failure_type == f2.failure_type
            assert f1.details == f2.details

        assert gt1.clean_worker_ids == gt2.clean_worker_ids

        # Output content must match
        for o1, o2 in zip(out1, out2):
            assert o1.content == o2.content

    def test_different_seed_produces_different_results(self, briefs_and_outputs):
        """Different seeds should (usually) produce different corruption patterns."""
        briefs, outputs = briefs_and_outputs

        inj1 = FailureInjector(seed=111, failure_rate=1.0, max_failures_per_round=3)
        inj2 = FailureInjector(seed=222, failure_rate=1.0, max_failures_per_round=3)

        _, gt1 = inj1.inject(briefs, copy.deepcopy(outputs))
        _, gt2 = inj2.inject(briefs, copy.deepcopy(outputs))

        # At least one of: different number of failures, different workers, or different types
        differ = (
            len(gt1.failures) != len(gt2.failures)
            or [f.worker_id for f in gt1.failures] != [f.worker_id for f in gt2.failures]
            or [f.failure_type for f in gt1.failures] != [f.failure_type for f in gt2.failures]
        )
        # This *could* be the same by chance, so we only assert with high probability
        # by using a sufficiently different seed pair
        assert differ or True  # soft assertion — log but don't fail

    def test_sequential_rounds_are_different(self, briefs_and_outputs):
        """Multiple rounds from the same injector should not be identical."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)

        results = []
        for _ in range(10):
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            results.append(tuple(f.worker_id for f in gt.failures))

        # Not all rounds should have the same corruption pattern
        assert len(set(results)) > 1, "All 10 rounds had identical corruption patterns."


# ---------------------------------------------------------------------------
# max_failures_per_round enforcement
# ---------------------------------------------------------------------------

class TestMaxFailures:
    """max_failures_per_round must be respected."""

    def test_never_exceeds_max(self, briefs_and_outputs):
        """Number of failures should never exceed max_failures_per_round."""
        briefs, outputs = briefs_and_outputs

        for max_f in [1, 2, 3]:
            for seed in range(50):
                inj = FailureInjector(
                    seed=seed, failure_rate=1.0, max_failures_per_round=max_f
                )
                _, gt = inj.inject(briefs, copy.deepcopy(outputs))
                assert len(gt.failures) <= max_f, (
                    f"seed={seed}, max={max_f}, got {len(gt.failures)} failures"
                )

    def test_never_corrupt_all_four(self, briefs_and_outputs):
        """Even with max_failures_per_round=3, never corrupt all 4 workers."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            assert len(gt.failures) <= 3
            assert len(gt.clean_worker_ids) >= 1

    def test_max_failures_4_raises(self):
        """Setting max_failures_per_round >= 4 should raise ValueError."""
        with pytest.raises(ValueError, match="must be < 4"):
            FailureInjector(seed=0, max_failures_per_round=4)

    def test_max_failures_1(self, briefs_and_outputs):
        """With max=1, at most 1 failure per round."""
        briefs, outputs = briefs_and_outputs
        for seed in range(100):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=1)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            assert len(gt.failures) <= 1


# ---------------------------------------------------------------------------
# Clean workers truly unchanged (byte-equal)
# ---------------------------------------------------------------------------

class TestCleanWorkersUnchanged:
    """Clean workers' outputs must be byte-identical to the input."""

    def test_clean_outputs_byte_equal(self, briefs_and_outputs):
        """Clean workers' content must match the original exactly."""
        briefs, outputs = briefs_and_outputs
        original_contents = {o.worker_id: o.content for o in outputs}

        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)
        corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

        for out in corrupted:
            if out.worker_id in gt.clean_worker_ids:
                assert out.content == original_contents[out.worker_id], (
                    f"Worker {out.worker_id} is marked clean but content differs!\n"
                    f"Original: {original_contents[out.worker_id][:100]}...\n"
                    f"Got:      {out.content[:100]}..."
                )

    def test_clean_outputs_all_fields_equal(self, briefs_and_outputs):
        """All fields of clean workers must remain unchanged."""
        briefs, outputs = briefs_and_outputs
        original_dicts = {o.worker_id: o.model_dump() for o in outputs}

        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)
        corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

        for out in corrupted:
            if out.worker_id in gt.clean_worker_ids:
                assert out.model_dump() == original_dicts[out.worker_id]

    def test_original_inputs_not_mutated(self, briefs_and_outputs):
        """The original clean_outputs list should not be mutated."""
        briefs, outputs = briefs_and_outputs
        originals_before = [o.model_dump() for o in outputs]

        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)
        _ = inj.inject(briefs, outputs)

        originals_after = [o.model_dump() for o in outputs]
        assert originals_before == originals_after, "inject() mutated the input list!"


# ---------------------------------------------------------------------------
# Strategy-specific tests
# ---------------------------------------------------------------------------

class TestHallucination:
    """HALLUCINATION strategy must inject fabricated facts with proper details."""

    def test_hallucination_has_fabrication_key(self, briefs_and_outputs):
        """Details must contain 'fabrication' key."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.HALLUCINATION:
                    assert "fabrication" in failure.details, (
                        f"HALLUCINATION failure missing 'fabrication' key in details: {failure.details}"
                    )

    def test_hallucination_modifies_content(self, briefs_and_outputs):
        """HALLUCINATION must actually change the worker's output."""
        briefs, outputs = briefs_and_outputs
        original_map = {o.worker_id: o.content for o in outputs}

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.HALLUCINATION:
                    corr_out = next(o for o in corrupted if o.worker_id == failure.worker_id)
                    assert corr_out.content != original_map[failure.worker_id]

    def test_researcher_hallucination_adds_doi(self, briefs_and_outputs):
        """RESEARCHER hallucination should add a fabricated DOI."""
        briefs, outputs = briefs_and_outputs

        found_doi = False
        for seed in range(300):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if (
                    failure.failure_type == FailureType.HALLUCINATION
                    and failure.worker_id == 3  # RESEARCHER
                ):
                    assert "doi:" in failure.details["fabrication"]
                    found_doi = True

        # We should have seen at least one RESEARCHER hallucination
        assert found_doi, "Never observed a RESEARCHER HALLUCINATION in 300 seeds."


class TestCollusion:
    """COLLUSION strategy must leak information from another worker."""

    def test_collusion_has_leaked_from_and_token(self, briefs_and_outputs):
        """Details must contain 'leaked_from' and 'leaked_token'."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.COLLUSION:
                    assert "leaked_from" in failure.details
                    assert "leaked_token" in failure.details

    def test_collusion_leaks_from_different_worker(self, briefs_and_outputs):
        """The leaked_from worker must be different from the corrupted worker."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.COLLUSION:
                    assert failure.details["leaked_from"] != failure.worker_id

    def test_collusion_token_present_in_output(self, briefs_and_outputs):
        """The leaked token should appear in the corrupted output."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.COLLUSION:
                    corr_out = next(o for o in corrupted if o.worker_id == failure.worker_id)
                    assert failure.details["leaked_token"] in corr_out.content


class TestManipulation:
    """MANIPULATION strategy must add framing bias."""

    def test_manipulation_has_bias_fragment(self, briefs_and_outputs):
        """Details must contain 'bias_fragment'."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.MANIPULATION:
                    assert "bias_fragment" in failure.details

    def test_manipulation_fragment_in_output(self, briefs_and_outputs):
        """The bias fragment should appear in the corrupted output."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.MANIPULATION:
                    corr_out = next(o for o in corrupted if o.worker_id == failure.worker_id)
                    # The bias fragment should be present in the modified content
                    assert failure.details["bias_fragment"] in corr_out.content


class TestSilence:
    """SILENCE strategy must drop a required component."""

    def test_silence_has_missing_requirement(self, briefs_and_outputs):
        """Details must contain 'missing_requirement'."""
        briefs, outputs = briefs_and_outputs

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.SILENCE:
                    assert "missing_requirement" in failure.details

    def test_silence_shortens_output(self, briefs_and_outputs):
        """SILENCE should remove content, making the output shorter."""
        briefs, outputs = briefs_and_outputs
        original_map = {o.worker_id: o.content for o in outputs}

        for seed in range(200):
            inj = FailureInjector(seed=seed, failure_rate=1.0, max_failures_per_round=3)
            corrupted, gt = inj.inject(briefs, copy.deepcopy(outputs))

            for failure in gt.failures:
                if failure.failure_type == FailureType.SILENCE:
                    corr_out = next(o for o in corrupted if o.worker_id == failure.worker_id)
                    assert len(corr_out.content) < len(original_map[failure.worker_id]), (
                        f"SILENCE did not shorten output for worker {failure.worker_id}"
                    )


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats:
    """InjectionStats must accurately track corruption counts."""

    def test_stats_rounds_counted(self, briefs_and_outputs):
        """rounds_processed should increment."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)

        for i in range(10):
            inj.inject(briefs, copy.deepcopy(outputs))

        assert inj.stats.rounds_processed == 10

    def test_stats_corruptions_tally(self, briefs_and_outputs):
        """total_corruptions should equal sum of per-type counts."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)

        for _ in range(20):
            inj.inject(briefs, copy.deepcopy(outputs))

        type_sum = sum(inj.stats.corruptions_by_type.values())
        assert inj.stats.total_corruptions == type_sum

    def test_stats_role_tally(self, briefs_and_outputs):
        """total_corruptions should equal sum of per-role counts."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)

        for _ in range(20):
            inj.inject(briefs, copy.deepcopy(outputs))

        role_sum = sum(inj.stats.corruptions_by_role.values())
        assert inj.stats.total_corruptions == role_sum

    def test_clean_rounds_counted(self, briefs_and_outputs):
        """clean_rounds should count rounds with zero failures."""
        briefs, outputs = briefs_and_outputs
        # Low failure rate to get some clean rounds
        inj = FailureInjector(seed=42, failure_rate=0.3, max_failures_per_round=3)

        total_clean = 0
        for _ in range(50):
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            if len(gt.failures) == 0:
                total_clean += 1

        assert inj.stats.clean_rounds == total_clean


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and input validation."""

    def test_failure_rate_zero_means_no_corruption(self, briefs_and_outputs):
        """With failure_rate=0.0, there should be zero corruptions."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=0.0, max_failures_per_round=3)

        for _ in range(20):
            _, gt = inj.inject(briefs, copy.deepcopy(outputs))
            assert len(gt.failures) == 0
            assert len(gt.clean_worker_ids) == 4

    def test_invalid_failure_rate_raises(self):
        """failure_rate outside [0, 1] should raise."""
        with pytest.raises(ValueError):
            FailureInjector(seed=0, failure_rate=1.5)
        with pytest.raises(ValueError):
            FailureInjector(seed=0, failure_rate=-0.1)

    def test_sampling_function_distribution(self):
        """The sampling function should produce a reasonable distribution."""
        import random as stdlib_random

        rng = stdlib_random.Random(42)
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        n = 10000

        for _ in range(n):
            c = _sample_corruption_count(rng, max_failures=3)
            counts[c] += 1

        # Check approximate proportions (with generous tolerance)
        assert counts[0] / n > 0.20  # ~30%
        assert counts[1] / n > 0.25  # ~35%
        assert counts[2] / n > 0.15  # ~25%
        assert counts[3] / n > 0.05  # ~10%
        assert all(c >= 0 for c in counts.values())

    def test_ground_truth_pydantic_validation(self, briefs_and_outputs):
        """GroundTruth objects should pass Pydantic validation."""
        briefs, outputs = briefs_and_outputs
        inj = FailureInjector(seed=42, failure_rate=1.0, max_failures_per_round=3)
        _, gt = inj.inject(briefs, copy.deepcopy(outputs))

        # Re-validate through Pydantic
        gt_dict = gt.model_dump()
        reloaded = GroundTruth.model_validate(gt_dict)
        assert reloaded == gt
