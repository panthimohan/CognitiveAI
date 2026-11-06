"""
CognitiveAI - Response Evaluation Engine
"""

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    completeness: float
    relevance: float
    confidence: float
    passed: bool
    feedback: str


class CognitiveEvaluator:

    def evaluate(
        self,
        completeness: float,
        relevance: float,
        confidence: float
    ) -> EvaluationResult:

        passed = (
            completeness >= 0.7
            and relevance >= 0.7
            and confidence >= 0.7
        )

        if passed:
            feedback = "Response meets the evaluation thresholds."
        else:
            feedback = "Response requires further improvement."

        return EvaluationResult(
            completeness=completeness,
            relevance=relevance,
            confidence=confidence,
            passed=passed,
            feedback=feedback
        )