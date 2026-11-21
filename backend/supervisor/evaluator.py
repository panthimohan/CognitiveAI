
"""
CognitiveAI - Metacognitive Evaluation Engine

Evaluates response quality, reliability, uncertainty and risk.

The evaluator produces a transparent decision:
- accept: available quality checks pass
- revise: quality or reliability needs improvement
- clarify: essential information is missing

This is a deterministic evaluation layer. Scores are supplied by
the caller; semantic LLM-based evaluation will be integrated later.
"""

from typing import Literal

from pydantic import BaseModel, Field


EvaluationDecision = Literal["accept", "revise", "clarify"]


class EvaluationResult(BaseModel):
    # Original evaluation dimensions (kept for compatibility)
    completeness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Additional evaluation dimensions
    correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    instruction_adherence: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    risk: float | None = Field(default=None, ge=0.0, le=1.0)

    # Overall result
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    decision: EvaluationDecision

    # Transparent explanation
    feedback: str
    issues: list[str] = Field(default_factory=list)


class CognitiveEvaluator:
    """
    Deterministic metacognitive quality evaluator.

    Positive dimensions:
        completeness, relevance, confidence,
        correctness, instruction adherence.

    Negative dimensions:
        uncertainty, risk.
    """

    QUALITY_THRESHOLD = 0.7
    UNCERTAINTY_THRESHOLD = 0.4
    RISK_THRESHOLD = 0.5

    @staticmethod
    def _validate_score(name: str, value: float | None) -> None:
        if value is None:
            return

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number between 0 and 1")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    def evaluate(
        self,
        completeness: float,
        relevance: float,
        confidence: float,
        *,
        correctness: float | None = None,
        instruction_adherence: float | None = None,
        uncertainty: float | None = None,
        risk: float | None = None,
        missing_information: bool = False,
        unsupported_claims: bool = False,
        requires_clarification: bool = False,
    ) -> EvaluationResult:

        scores = {
            "completeness": completeness,
            "relevance": relevance,
            "confidence": confidence,
            "correctness": correctness,
            "instruction adherence": instruction_adherence,
            "uncertainty": uncertainty,
            "risk": risk,
        }

        for name, value in scores.items():
            self._validate_score(name, value)

        if not all(
            isinstance(flag, bool)
            for flag in (
                missing_information,
                unsupported_claims,
                requires_clarification,
            )
        ):
            raise ValueError("Evaluation flags must be boolean")

        issues = []

        # Check positive quality dimensions.
        positive_scores = {
            "Completeness": completeness,
            "Relevance": relevance,
            "Confidence": confidence,
        }

        if correctness is not None:
            positive_scores["Correctness"] = correctness

        if instruction_adherence is not None:
            positive_scores["Instruction adherence"] = (
                instruction_adherence
            )

        for name, score in positive_scores.items():
            if score < self.QUALITY_THRESHOLD:
                issues.append(
                    f"{name} is below the required threshold."
                )

        # Check negative dimensions.
        if uncertainty is not None:
            if uncertainty > self.UNCERTAINTY_THRESHOLD:
                issues.append(
                    "Response uncertainty is too high."
                )

        if risk is not None:
            if risk > self.RISK_THRESHOLD:
                issues.append(
                    "Response risk exceeds the acceptable threshold."
                )

        if unsupported_claims:
            issues.append(
                "The response contains unsupported claims."
            )

        if missing_information:
            issues.append(
                "Essential information is missing."
            )

        if requires_clarification:
            issues.append(
                "User clarification is required."
            )

        # Calculate a transparent average of available quality signals.
        normalized_scores = list(positive_scores.values())

        if uncertainty is not None:
            normalized_scores.append(1.0 - uncertainty)

        if risk is not None:
            normalized_scores.append(1.0 - risk)

        overall_score = sum(normalized_scores) / len(normalized_scores)

        # Determine the next action.
        if missing_information or requires_clarification:
            decision: EvaluationDecision = "clarify"

        elif issues:
            decision = "revise"

        else:
            decision = "accept"

        passed = decision == "accept"

        if decision == "accept":
            feedback = (
                "Response passed all available evaluation checks."
            )

        elif decision == "clarify":
            feedback = (
                "Additional information is required before "
                "a reliable response can be produced."
            )

        else:
            feedback = (
                "Response requires revision before it can be accepted."
            )

        return EvaluationResult(
            completeness=completeness,
            relevance=relevance,
            confidence=confidence,
            correctness=correctness,
            instruction_adherence=instruction_adherence,
            uncertainty=uncertainty,
            risk=risk,
            overall_score=round(overall_score, 4),
            passed=passed,
            decision=decision,
            feedback=feedback,
            issues=issues,
        )
