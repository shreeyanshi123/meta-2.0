import pytest
from pydantic import ValidationError
from tribunal.schemas import (
    FailureType, WorkerRole, TaskBrief, WorkerOutput, InjectedFailure, 
    GroundTruth, JudgeVerdict, Observation, Action, State, RewardBreakdown,
    verdict_from_json, ParseError
)

def test_failure_type_enum():
    assert FailureType.HALLUCINATION == "HALLUCINATION"
    assert FailureType.CLEAN == "CLEAN"

def test_worker_role_enum():
    assert WorkerRole.SUMMARISER == "SUMMARISER"

def test_task_brief_happy_path():
    brief = TaskBrief(
        worker_id=2,
        role=WorkerRole.NEGOTIATOR,
        prompt="Negotiate price.",
        source_material="Price: 500",
        private_context={"limit": "400"}
    )
    assert brief.worker_id == 2
    assert brief.private_context["limit"] == "400"

def test_judge_verdict_happy_path():
    verdict = JudgeVerdict(
        accused=[1, 3],
        failure_types={1: FailureType.HALLUCINATION, 3: FailureType.SILENCE},
        explanation="Detected issues.",
        per_worker_confidence={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.9}
    )
    assert set(verdict.accused) == {1, 3}

def test_judge_verdict_validation_mismatch():
    with pytest.raises(ValidationError, match="must match exactly"):
        JudgeVerdict(
            accused=[1, 2],
            failure_types={1: FailureType.HALLUCINATION},
            explanation="Mismatch in lists.",
            per_worker_confidence={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.9}
        )

def test_judge_verdict_validation_invalid_accused_id():
    with pytest.raises(ValidationError, match="between 0 and 3"):
        JudgeVerdict(
            accused=[4],
            failure_types={4: FailureType.HALLUCINATION},
            explanation="Invalid ID.",
            per_worker_confidence={0: 0.9, 1: 0.8, 2: 0.7, 3: 0.9}
        )

def test_judge_verdict_validation_confidence_missing_keys():
    with pytest.raises(ValidationError, match="exactly keys 0, 1, 2, and 3"):
        JudgeVerdict(
            accused=[],
            failure_types={},
            explanation="Missing some confidences.",
            per_worker_confidence={0: 0.9, 1: 0.8, 2: 0.7}
        )

def test_observation_no_private_context():
    with pytest.raises(ValidationError, match="cannot contain private_context"):
        Observation(
            round_id="123",
            task_briefs_public=[{"worker_id": 0, "private_context": {"secret": "val"}}],
            worker_outputs=[],
            round_index=1
        )
        
    obs = Observation(
        round_id="456",
        task_briefs_public=[{"worker_id": 1, "prompt": "Hi"}],
        worker_outputs=[],
        round_index=2
    )
    assert obs.round_id == "456"

def test_reward_breakdown_total_validation():
    with pytest.raises(ValidationError, match="does not match sum of components"):
        RewardBreakdown(
            identification=1.0, type_classification=1.0, explanation_quality=1.0,
            false_positive_penalty=-0.5, calibration=0.5, anti_hack_penalty=-0.1,
            total=10.0, notes=[]
        )
    
    # Happy path: sum is 1.0 + 1.0 + 1.0 - 0.5 + 0.5 - 0.2 = 2.8
    rb = RewardBreakdown(
        identification=1.0, type_classification=1.0, explanation_quality=1.0,
        false_positive_penalty=-0.5, calibration=0.5, anti_hack_penalty=-0.2,
        total=2.8, notes=[]
    )
    assert abs(rb.total - 2.8) < 1e-6

def test_verdict_from_json_resilience():
    # Messy json with markdown, trailing commas, string keys
    messy_json = """
    ```json
    {
        "accused": [1],
        "failure_types": {"1": "HALLUCINATION",},
        "explanation": "Test trailing comma.",
        "per_worker_confidence": {
            "0": 0.8,
            "1": 0.9,
            "2": 0.7,
            "3": 0.8,
        }
    }
    ```
    """
    verdict = verdict_from_json(messy_json)
    assert verdict.accused == [1]
    assert verdict.failure_types[1] == FailureType.HALLUCINATION
    assert verdict.per_worker_confidence[3] == 0.8

def test_verdict_from_json_parse_error():
    invalid_json = "{ invalid json "
    with pytest.raises(ParseError) as exc_info:
        verdict_from_json(invalid_json)
    
    assert exc_info.value.original_string == invalid_json
