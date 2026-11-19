
"""
CognitiveAI - Learning Consolidation

Evaluates active-learning insights before they can become
permanent memory candidates.

Responsibilities:
- Validate supporting evidence.
- Evaluate confidence and support thresholds.
- Distinguish recurring patterns from possible changes.
- Preserve historical values.
- Recommend whether an insight should be accepted,
  deferred, or rejected.

This module does NOT write to permanent memory.
Memory promotion is a separate, explicitly controlled step.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.learning.active import (
    InsightType,
    LearningInsight,
)
from backend.learning.contracts import (
    LearningDomain,
    LearningStatus,
)
from backend.memory.models import MemoryKind


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ==========================================
# 1. CONSOLIDATION DECISION
# ==========================================


class ConsolidationDecision(BaseModel):
    """The evaluation result for one learning insight."""

    domain: LearningDomain
    owner: str | None = None

    claim_key: str
    insight_type: InsightType

    status: LearningStatus
    reason: str

    evidence_count: int = Field(ge=0)
    evidence_strength: float = Field(ge=0, le=1)

    # Values recommended for possible future memory promotion.
    # Historical values remain available in the original insight.
    recommended_values: list[str] = Field(default_factory=list)

    recommended_memory_kind: MemoryKind | None = None

    evaluated_at: datetime = Field(default_factory=utc_now)


# ==========================================
# 2. CONSOLIDATION ENGINE
# ==========================================


class LearningConsolidator:
    """
    Evaluates insights produced by ActiveLearningEngine.

    Default policy:
    - Recurring patterns require at least 2 supporting claims.
    - Possible changes require at least 2 claims supporting
      the newest value.
    - Evidence strength must meet the configured threshold.

    An accepted decision is a recommendation, not an automatic
    permanent-memory write.
    """

    def __init__(
        self,
        *,
        min_pattern_support: int = 2,
        min_change_support: int = 2,
        min_confidence: float = 0.70,
    ):
        if min_pattern_support < 2:
            raise ValueError(
                "Pattern support must be at least 2."
            )

        if min_change_support < 2:
            raise ValueError(
                "Change support must be at least 2."
            )

        if not 0 <= min_confidence <= 1:
            raise ValueError(
                "Minimum confidence must be between 0 and 1."
            )

        self.min_pattern_support = min_pattern_support
        self.min_change_support = min_change_support
        self.min_confidence = min_confidence

    # --------------------------------------
    # Public API
    # --------------------------------------

    def evaluate(
        self,
        insight: LearningInsight,
    ) -> ConsolidationDecision:
        """Evaluate one insight and return a decision."""

        invalid_reason = self._validate_insight(insight)

        if invalid_reason is not None:
            return self._decision(
                insight=insight,
                status=LearningStatus.REJECTED,
                reason=invalid_reason,
            )

        if insight.evidence_strength < self.min_confidence:
            return self._decision(
                insight=insight,
                status=LearningStatus.DEFERRED,
                reason=(
                    "Evidence confidence is below the "
                    "consolidation threshold."
                ),
            )

        if insight.insight_type == InsightType.RECURRING_PATTERN:
            return self._evaluate_pattern(insight)

        if insight.insight_type == InsightType.POSSIBLE_CHANGE:
            return self._evaluate_change(insight)

        return self._decision(
            insight=insight,
            status=LearningStatus.REJECTED,
            reason="Unsupported insight type.",
        )

    def evaluate_many(
        self,
        insights: list[LearningInsight],
    ) -> list[ConsolidationDecision]:
        """Evaluate a collection of insights independently."""

        return [
            self.evaluate(insight)
            for insight in insights
        ]

    # --------------------------------------
    # Pattern evaluation
    # --------------------------------------

    def _evaluate_pattern(
        self,
        insight: LearningInsight,
    ) -> ConsolidationDecision:

        if len(insight.values) != 1:
            return self._decision(
                insight=insight,
                status=LearningStatus.REJECTED,
                reason=(
                    "A recurring pattern must contain exactly "
                    "one consistently supported value."
                ),
            )

        value = insight.values[0]

        if value.support_count < self.min_pattern_support:
            return self._decision(
                insight=insight,
                status=LearningStatus.DEFERRED,
                reason=(
                    "The recurring pattern does not yet have "
                    "enough supporting evidence."
                ),
            )

        return self._decision(
            insight=insight,
            status=LearningStatus.ACCEPTED,
            reason=(
                "The recurring pattern meets the evidence "
                "and confidence requirements."
            ),
            recommended_values=[value.value],
        )

    # --------------------------------------
    # Change evaluation
    # --------------------------------------

    def _evaluate_change(
        self,
        insight: LearningInsight,
    ) -> ConsolidationDecision:

        if len(insight.values) < 2:
            return self._decision(
                insight=insight,
                status=LearningStatus.REJECTED,
                reason=(
                    "A possible change must contain at least "
                    "two distinct historical values."
                ),
            )

        # Find the value with the most recent supporting evidence.
        latest_value = max(
            insight.values,
            key=lambda item: max(
                reference.observed_at
                for reference in item.evidence
            ),
        )

        if latest_value.support_count < self.min_change_support:
            return self._decision(
                insight=insight,
                status=LearningStatus.DEFERRED,
                reason=(
                    "The newest value does not yet have enough "
                    "independent supporting experiences to "
                    "recommend a change."
                ),
            )

        return self._decision(
            insight=insight,
            status=LearningStatus.ACCEPTED,
            reason=(
                "The newer value has sufficient supporting "
                "evidence. It may be recommended as an update, "
                "while all historical values must be preserved."
            ),
            recommended_values=[latest_value.value],
        )

    # --------------------------------------
    # Evidence validation
    # --------------------------------------

    @staticmethod
    def _validate_insight(
        insight: LearningInsight,
    ) -> str | None:

        if insight.domain == LearningDomain.PRIVATE:
            if not insight.owner or not insight.owner.strip():
                return "Private insights must have an owner."

        elif insight.domain == LearningDomain.PUBLIC:
            if insight.owner is not None:
                return "Public insights cannot have a private owner."

        if not insight.claim_key.strip():
            return "Insight claim key cannot be empty."

        if not insight.values:
            return "Insight must contain supporting values."

        all_experience_ids = set()
        total_evidence = 0

        for value in insight.values:
            if not value.value.strip():
                return "Evidence value cannot be empty."

            if value.support_count != len(value.evidence):
                return (
                    "Evidence count does not match the number "
                    "of supporting references."
                )

            if not value.evidence:
                return "Every value must have supporting evidence."

            for reference in value.evidence:
                if not reference.experience_id.strip():
                    return "Evidence is missing an experience ID."

                if not reference.source_ref.strip():
                    return "Evidence is missing a source reference."

                if reference.experience_id in all_experience_ids:
                    return (
                        "An experience cannot be counted more "
                        "than once for the same insight."
                    )

                all_experience_ids.add(reference.experience_id)
                total_evidence += 1

        if total_evidence != insight.evidence_count:
            return (
                "Insight evidence count does not match its "
                "supporting references."
            )

        return None

    # --------------------------------------
    # Decision construction
    # --------------------------------------

    @staticmethod
    def _decision(
        *,
        insight: LearningInsight,
        status: LearningStatus,
        reason: str,
        recommended_values: list[str] | None = None,
    ) -> ConsolidationDecision:

        accepted = status == LearningStatus.ACCEPTED

        return ConsolidationDecision(
            domain=insight.domain,
            owner=insight.owner,
            claim_key=insight.claim_key,
            insight_type=insight.insight_type,
            status=status,
            reason=reason,
            evidence_count=insight.evidence_count,
            evidence_strength=insight.evidence_strength,
            recommended_values=(
                recommended_values or []
            ),
            recommended_memory_kind=(
                MemoryKind.SEMANTIC if accepted else None
            ),
        )
