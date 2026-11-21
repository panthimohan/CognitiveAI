
import asyncio
import json

import pytest

from backend.models.providers import ModelResponse
from backend.supervisor.semantic_evaluator import (
    SemanticEvaluationError,
    SemanticEvaluator,
)


class FakeProvider:
    def __init__(self, content=None, success=True):
        self.content = content or json.dumps({
            "completeness": 0.95,
            "relevance": 0.95,
            "confidence": 0.95,
            "correctness": 0.95,
            "instruction_adherence": 0.95,
            "uncertainty": 0.05,
            "risk": 0.0,
            "unsupported_claims": [],
            "missing_information": [],
            "requires_clarification": False,
            "feedback": "The answer addresses the question clearly.",
        })
        self.success = success

    async def generate(self, request):
        return ModelResponse(
            provider="local",
            model="test-model",
            content=self.content,
            success=self.success,
            error=None if self.success else "Test provider failure",
        )

    def is_available(self):
        return True

    def get_provider_name(self):
        return "local"


def test_semantic_evaluator_accepts_good_answer():
    async def run_test():
        evaluator = SemanticEvaluator()

        result = await evaluator.evaluate_response(
            question="What is Python?",
            answer="Python is a general-purpose programming language.",
            provider=FakeProvider(),
        )

        assert result.evaluation.decision == "accept"
        assert result.assessment.completeness == 0.95
        assert result.evaluation_method == (
            "local_llm_judge_plus_deterministic_rules"
        )

    asyncio.run(run_test())


def test_semantic_evaluator_rejects_invalid_json():
    async def run_test():
        evaluator = SemanticEvaluator()

        with pytest.raises(SemanticEvaluationError):
            await evaluator.evaluate_response(
                question="What is Python?",
                answer="Python is a programming language.",
                provider=FakeProvider(content="This is not JSON"),
            )

    asyncio.run(run_test())


def test_semantic_evaluator_rejects_invalid_scores():
    async def run_test():
        evaluator = SemanticEvaluator()

        invalid_assessment = {
            "completeness": 1.5,
            "relevance": 0.9,
            "confidence": 0.9,
            "correctness": 0.9,
            "instruction_adherence": 0.9,
            "uncertainty": 0.1,
            "risk": 0.0,
        }

        with pytest.raises(SemanticEvaluationError):
            await evaluator.evaluate_response(
                question="What is Python?",
                answer="Python is a programming language.",
                provider=FakeProvider(
                    content=json.dumps(invalid_assessment)
                ),
            )

    asyncio.run(run_test())


def test_semantic_evaluator_reports_provider_failure():
    async def run_test():
        evaluator = SemanticEvaluator()

        with pytest.raises(SemanticEvaluationError):
            await evaluator.evaluate_response(
                question="What is Python?",
                answer="Python is a programming language.",
                provider=FakeProvider(success=False),
            )

    asyncio.run(run_test())
