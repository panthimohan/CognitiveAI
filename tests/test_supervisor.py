
import pytest

from backend.supervisor.router import (
    TaskRequest,
    classify_task,
    assess_complexity,
    identify_capabilities,
    route_task,
)
from backend.supervisor.planner import CognitivePlanner
from backend.supervisor.evaluator import CognitiveEvaluator


def test_classifies_coding_task():
    assert classify_task("Write Python code") == "coding"


def test_classifies_research_task():
    assert classify_task("Research the latest AI developments") == "research"


def test_classifies_conversation():
    assert classify_task("Hello, how are you?") == "conversation"


def test_assesses_complexity():
    assert assess_complexity("Design the system architecture") == "medium"


def test_identifies_capabilities():
    capabilities = identify_capabilities(
        "Analyze privacy and long-term memory"
    )

    assert "reasoning" in capabilities
    assert "privacy" in capabilities
    assert "memory" in capabilities


def test_private_task_uses_local_provider():
    decision = route_task(
        TaskRequest(task="Write Python code", privacy="private")
    )

    assert decision.provider == "local"
    assert decision.execution_ready is False
    assert decision.privacy == "private"


def test_public_coding_task_selects_anthropic():
    decision = route_task(
        TaskRequest(task="Write Python code", privacy="public")
    )

    assert decision.provider == "anthropic"
    assert decision.task_type == "coding"


def test_invalid_privacy_is_rejected():
    with pytest.raises(ValueError):
        route_task(
            TaskRequest(task="Write Python code", privacy="unknown")
        )


def test_planner_creates_structured_plan():
    planner = CognitivePlanner()

    plan = planner.create_plan(
        "Build a Python application with privacy controls and long-term memory"
    )

    assert plan.total_steps == len(plan.subtasks)
    assert plan.total_steps >= 3

    capabilities = [
        item.required_capability
        for item in plan.subtasks
    ]

    assert "reasoning" in capabilities
    assert "coding" in capabilities
    assert "privacy" in capabilities
    assert "memory" in capabilities


def test_evaluator_accepts_good_scores():
    evaluator = CognitiveEvaluator()

    result = evaluator.evaluate(
        completeness=0.9,
        relevance=0.9,
        confidence=0.9
    )

    assert result.passed is True


def test_evaluator_rejects_low_scores():
    evaluator = CognitiveEvaluator()

    result = evaluator.evaluate(
        completeness=0.9,
        relevance=0.5,
        confidence=0.9
    )

    assert result.passed is False
