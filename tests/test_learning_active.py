
import pytest

from pydantic import ValidationError

from backend.learning.active import (
    ActiveLearningEngine,
    ClaimObservation,
    InsightType,
)
from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
    LearningExperience,
)


def make_private_experience(
    *,
    owner="test-user",
    source_ref="conversation-1",
):
    return LearningExperience(
        domain=LearningDomain.PRIVATE,
        owner=owner,
        content="A private conversation.",
        source_type=ExperienceSource.PRIVATE_CONVERSATION,
        source_ref=source_ref,
        authorization_basis=AuthorizationBasis.PRIVATE_CONTEXT,
    )


def make_public_experience(source_ref="source-1"):
    return LearningExperience(
        domain=LearningDomain.PUBLIC,
        content="A trusted external source.",
        source_type=ExperienceSource.TRUSTED_EXTERNAL_SOURCE,
        source_ref=source_ref,
        authorization_basis=AuthorizationBasis.TRUSTED_SOURCE_POLICY,
    )


def make_claim(
    *,
    value,
    key="learning_goal",
    experience=None,
    confidence=0.8,
):
    return ClaimObservation(
        experience=experience or make_private_experience(),
        key=key,
        value=value,
        confidence=confidence,
    )


def test_record_claim_accepts_valid_private_experience():
    engine = ActiveLearningEngine()

    recorded = engine.record_claim(
        make_claim(value="Learn Python")
    )

    assert recorded is True


def test_duplicate_claim_from_same_experience_is_ignored():
    engine = ActiveLearningEngine()

    claim = make_claim(
        value="Learn Python",
        experience=make_private_experience(),
    )

    assert engine.record_claim(claim) is True
    assert engine.record_claim(claim) is False


def test_repeated_claim_creates_recurring_pattern():
    engine = ActiveLearningEngine(min_pattern_support=2)

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                source_ref="conversation-1"
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                source_ref="conversation-2"
            ),
        )
    )

    insights = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(insights) == 1

    insight = insights[0]

    assert insight.insight_type == InsightType.RECURRING_PATTERN
    assert insight.evidence_count == 2
    assert insight.values[0].support_count == 2


def test_single_claim_does_not_create_pattern():
    engine = ActiveLearningEngine(min_pattern_support=2)

    engine.record_claim(
        make_claim(value="Learn Python")
    )

    insights = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert insights == []


def test_different_values_create_possible_change():
    engine = ActiveLearningEngine()

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                source_ref="conversation-1"
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn machine learning",
            experience=make_private_experience(
                source_ref="conversation-2"
            ),
        )
    )

    insights = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(insights) == 1

    insight = insights[0]

    assert insight.insight_type == InsightType.POSSIBLE_CHANGE
    assert insight.evidence_count == 2
    assert len(insight.values) == 2


def test_claim_history_preserves_all_values():
    engine = ActiveLearningEngine()

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                source_ref="conversation-1"
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn machine learning",
            experience=make_private_experience(
                source_ref="conversation-2"
            ),
        )
    )

    history = engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    assert len(history) == 2
    assert history[0].value == "Learn Python"
    assert history[1].value == "Learn machine learning"


def test_private_claims_are_isolated_by_owner():
    engine = ActiveLearningEngine()

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                owner="user-a",
                source_ref="conversation-a",
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn Java",
            experience=make_private_experience(
                owner="user-b",
                source_ref="conversation-b",
            ),
        )
    )

    user_a_insights = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
    )

    user_b_history = engine.history(
        domain=LearningDomain.PRIVATE,
        owner="user-b",
        key="learning_goal",
    )

    assert user_a_insights == []
    assert len(user_b_history) == 1
    assert user_b_history[0].value == "Learn Java"


def test_public_claims_are_separate_from_private_claims():
    engine = ActiveLearningEngine()

    engine.record_claim(
        make_claim(
            value="Private goal",
            experience=make_private_experience(),
        )
    )

    engine.record_claim(
        make_claim(
            value="Public knowledge",
            experience=make_public_experience(),
        )
    )

    private_history = engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    public_history = engine.history(
        domain=LearningDomain.PUBLIC,
        key="learning_goal",
    )

    assert len(private_history) == 1
    assert len(public_history) == 1

    assert private_history[0].value == "Private goal"
    assert public_history[0].value == "Public knowledge"


def test_private_analysis_requires_owner():
    engine = ActiveLearningEngine()

    with pytest.raises(ValueError):
        engine.analyze(
            domain=LearningDomain.PRIVATE,
        )


def test_public_analysis_rejects_owner():
    engine = ActiveLearningEngine()

    with pytest.raises(ValueError):
        engine.analyze(
            domain=LearningDomain.PUBLIC,
            owner="test-user",
        )


def test_claim_rejects_empty_key():
    with pytest.raises(ValidationError):
        make_claim(
            key="   ",
            value="Learn Python",
        )


def test_claim_rejects_empty_value():
    with pytest.raises(ValidationError):
        make_claim(value="   ")


def test_pattern_threshold_must_be_at_least_two():
    with pytest.raises(ValueError):
        ActiveLearningEngine(min_pattern_support=1)
