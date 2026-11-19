
import pytest

from backend.learning.active import (
    ActiveLearningEngine,
    ClaimObservation,
)
from backend.learning.consolidation import LearningConsolidator
from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
    LearningExperience,
    LearningStatus,
)
from backend.memory.models import MemoryKind


def make_experience(
    source_ref,
    *,
    domain=LearningDomain.PRIVATE,
    owner="test-user",
):
    if domain == LearningDomain.PRIVATE:
        return LearningExperience(
            domain=domain,
            owner=owner,
            content="Private conversation.",
            source_type=ExperienceSource.PRIVATE_CONVERSATION,
            source_ref=source_ref,
            authorization_basis=AuthorizationBasis.PRIVATE_CONTEXT,
        )

    return LearningExperience(
        domain=domain,
        content="Trusted external information.",
        source_type=ExperienceSource.TRUSTED_EXTERNAL_SOURCE,
        source_ref=source_ref,
        authorization_basis=AuthorizationBasis.TRUSTED_SOURCE_POLICY,
    )


def make_insights(
    values,
    *,
    domain=LearningDomain.PRIVATE,
    owner="test-user",
    confidence=0.9,
):
    engine = ActiveLearningEngine()

    for index, value in enumerate(values):
        experience = make_experience(
            f"source-{index}",
            domain=domain,
            owner=owner,
        )

        engine.record_claim(
            ClaimObservation(
                experience=experience,
                key="learning_goal",
                value=value,
                confidence=confidence,
            )
        )

    return engine.analyze(
        domain=domain,
        owner=owner,
    )


def test_recurring_pattern_is_accepted():
    insights = make_insights([
        "Learn Python",
        "Learn Python",
    ])

    decision = LearningConsolidator().evaluate(insights[0])

    assert decision.status == LearningStatus.ACCEPTED
    assert decision.recommended_values == ["Learn Python"]
    assert decision.recommended_memory_kind == MemoryKind.SEMANTIC


def test_single_support_is_not_enough_for_pattern():
    insights = make_insights([
        "Learn Python",
        "Learn Python",
    ])

    insight = insights[0].model_copy(
        update={"values": [insights[0].values[0].model_copy(
            update={"support_count": 1, "evidence": insights[0].values[0].evidence[:1]}
        )],
        "evidence_count": 1}
    )

    decision = LearningConsolidator().evaluate(insight)

    assert decision.status == LearningStatus.DEFERRED


def test_low_confidence_pattern_is_deferred():
    insights = make_insights(
        ["Learn Python", "Learn Python"],
        confidence=0.4,
    )

    decision = LearningConsolidator().evaluate(insights[0])

    assert decision.status == LearningStatus.DEFERRED
    assert decision.recommended_values == []


def test_possible_change_is_deferred_without_repeated_new_value():
    insights = make_insights([
        "Learn Python",
        "Learn machine learning",
    ])

    decision = LearningConsolidator().evaluate(insights[0])

    assert decision.status == LearningStatus.DEFERRED
    assert decision.recommended_values == []


def test_possible_change_accepts_repeated_new_value():
    insights = make_insights([
        "Learn Python",
        "Learn machine learning",
        "Learn machine learning",
    ])

    decision = LearningConsolidator().evaluate(insights[0])

    assert decision.status == LearningStatus.ACCEPTED
    assert decision.recommended_values == ["Learn machine learning"]


def test_possible_change_does_not_delete_historical_values():
    engine = ActiveLearningEngine()

    values = [
        "Learn Python",
        "Learn machine learning",
        "Learn machine learning",
    ]

    for index, value in enumerate(values):
        engine.record_claim(
            ClaimObservation(
                experience=make_experience(f"source-{index}"),
                key="learning_goal",
                value=value,
            )
        )

    insight = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )[0]

    decision = LearningConsolidator().evaluate(insight)

    history = engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    assert decision.status == LearningStatus.ACCEPTED
    assert len(history) == 3
    assert history[0].value == "Learn Python"


def test_public_pattern_can_be_consolidated():
    insights = make_insights(
        ["Reliable information", "Reliable information"],
        domain=LearningDomain.PUBLIC,
        owner=None,
    )

    decision = LearningConsolidator().evaluate(insights[0])

    assert decision.status == LearningStatus.ACCEPTED
    assert decision.domain == LearningDomain.PUBLIC
    assert decision.owner is None


def test_private_insight_must_have_owner():
    insights = make_insights([
        "Learn Python",
        "Learn Python",
    ])

    invalid_insight = insights[0].model_copy(
        update={"owner": None}
    )

    decision = LearningConsolidator().evaluate(invalid_insight)

    assert decision.status == LearningStatus.REJECTED
    assert "owner" in decision.reason.lower()


def test_public_insight_cannot_have_owner():
    insights = make_insights(
        ["Public fact", "Public fact"],
        domain=LearningDomain.PUBLIC,
        owner=None,
    )

    invalid_insight = insights[0].model_copy(
        update={"owner": "test-user"}
    )

    decision = LearningConsolidator().evaluate(invalid_insight)

    assert decision.status == LearningStatus.REJECTED
    assert "owner" in decision.reason.lower()


def test_inconsistent_evidence_count_is_rejected():
    insights = make_insights([
        "Learn Python",
        "Learn Python",
    ])

    invalid_insight = insights[0].model_copy(
        update={"evidence_count": 10}
    )

    decision = LearningConsolidator().evaluate(invalid_insight)

    assert decision.status == LearningStatus.REJECTED
    assert "evidence count" in decision.reason.lower()


def test_evaluate_many_evaluates_each_insight():
    insights = make_insights([
        "Learn Python",
        "Learn Python",
    ])

    decisions = LearningConsolidator().evaluate_many(
        insights
    )

    assert len(decisions) == 1
    assert decisions[0].status == LearningStatus.ACCEPTED


def test_invalid_consolidation_policy_is_rejected():
    with pytest.raises(ValueError):
        LearningConsolidator(min_pattern_support=1)

    with pytest.raises(ValueError):
        LearningConsolidator(min_change_support=1)

    with pytest.raises(ValueError):
        LearningConsolidator(min_confidence=1.5)
