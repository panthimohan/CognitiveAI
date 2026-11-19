
import pytest

from backend.learning.active import (
    ActiveLearningEngine,
    ClaimObservation,
)
from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
    LearningExperience,
)
from backend.learning.consolidation import LearningConsolidator
from backend.learning.repository import LearningRepository


# ==========================================
# TEST HELPERS
# ==========================================


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


def make_repository(tmp_path):
    return LearningRepository(
        tmp_path / "learning_repository.sqlite3"
    )


def make_recurring_insight(repository):
    """Create two persisted observations and analyze them."""

    engine = ActiveLearningEngine(min_pattern_support=2)

    for number in range(1, 3):
        observation = make_claim(
            value="Learn Python",
            experience=make_private_experience(
                source_ref=f"conversation-{number}",
            ),
        )

        assert repository.record_claim(observation) is True
        engine.record_claim(observation)

    insights = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(insights) == 1

    return insights[0]


# ==========================================
# EXPERIENCE PERSISTENCE
# ==========================================


def test_experience_survives_repository_restart(tmp_path):
    database = tmp_path / "learning.sqlite3"

    experience = make_private_experience()

    first_repository = LearningRepository(database)

    assert first_repository.record_experience(experience) is True

    second_repository = LearningRepository(database)

    restored = second_repository.get_experience(
        experience.id,
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert restored is not None
    assert restored.id == experience.id
    assert restored.content == experience.content
    assert restored.source_ref == experience.source_ref


def test_duplicate_experience_is_ignored(tmp_path):
    repository = make_repository(tmp_path)

    experience = make_private_experience()

    assert repository.record_experience(experience) is True
    assert repository.record_experience(experience) is False


def test_experience_id_cannot_be_reused_with_different_data(
    tmp_path,
):
    repository = make_repository(tmp_path)

    experience = make_private_experience()

    repository.record_experience(experience)

    conflicting = experience.model_copy(
        update={"content": "Different content."}
    )

    with pytest.raises(ValueError):
        repository.record_experience(conflicting)


# ==========================================
# CLAIM PERSISTENCE
# ==========================================


def test_claim_survives_repository_restart(tmp_path):
    database = tmp_path / "learning.sqlite3"

    observation = make_claim(
        value="Learn Python",
    )

    first_repository = LearningRepository(database)

    assert first_repository.record_claim(observation) is True

    second_repository = LearningRepository(database)

    history = second_repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        claim_key="learning_goal",
    )

    assert len(history) == 1
    assert history[0].value == "Learn Python"
    assert history[0].experience.id == observation.experience.id


def test_duplicate_claim_is_ignored(tmp_path):
    repository = make_repository(tmp_path)

    observation = make_claim(
        value="Learn Python",
    )

    assert repository.record_claim(observation) is True
    assert repository.record_claim(observation) is False

    history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(history) == 1


def test_different_claim_values_preserve_history(tmp_path):
    repository = make_repository(tmp_path)

    first = make_claim(
        value="Learn Python",
        experience=make_private_experience(
            source_ref="conversation-1"
        ),
    )

    second = make_claim(
        value="Learn machine learning",
        experience=make_private_experience(
            source_ref="conversation-2"
        ),
    )

    repository.record_claim(first)
    repository.record_claim(second)

    history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        claim_key="learning_goal",
    )

    assert len(history) == 2
    assert {item.value for item in history} == {
        "Learn Python",
        "Learn machine learning",
    }


def test_normalized_duplicate_claim_is_ignored(tmp_path):
    repository = make_repository(tmp_path)

    first = make_claim(
        value="Learn Python",
        experience=make_private_experience(
            source_ref="conversation-1"
        ),
    )

    duplicate = make_claim(
        value=" learn   python ",
        key=" LEARNING_GOAL ",
        experience=first.experience,
    )

    assert repository.record_claim(first) is True
    assert repository.record_claim(duplicate) is False


# ==========================================
# PRIVATE OWNER ISOLATION
# ==========================================


def test_private_experiences_are_isolated_by_owner(tmp_path):
    repository = make_repository(tmp_path)

    user_a = make_private_experience(
        owner="user-a",
        source_ref="conversation-a",
    )

    user_b = make_private_experience(
        owner="user-b",
        source_ref="conversation-b",
    )

    repository.record_experience(user_a)
    repository.record_experience(user_b)

    assert repository.get_experience(
        user_a.id,
        domain=LearningDomain.PRIVATE,
        owner="user-b",
    ) is None

    user_a_items = repository.list_experiences(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
    )

    assert len(user_a_items) == 1
    assert user_a_items[0].id == user_a.id


def test_private_claims_are_isolated_by_owner(tmp_path):
    repository = make_repository(tmp_path)

    repository.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                owner="user-a",
                source_ref="conversation-a",
            ),
        )
    )

    repository.record_claim(
        make_claim(
            value="Learn Java",
            experience=make_private_experience(
                owner="user-b",
                source_ref="conversation-b",
            ),
        )
    )

    user_a_history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
    )

    user_b_history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="user-b",
    )

    assert len(user_a_history) == 1
    assert len(user_b_history) == 1

    assert user_a_history[0].value == "Learn Python"
    assert user_b_history[0].value == "Learn Java"


def test_private_repository_operations_require_owner(tmp_path):
    repository = make_repository(tmp_path)

    with pytest.raises(ValueError):
        repository.list_experiences(
            domain=LearningDomain.PRIVATE,
        )

    with pytest.raises(ValueError):
        repository.get_claim_history(
            domain=LearningDomain.PRIVATE,
        )


# ==========================================
# PUBLIC / PRIVATE SEPARATION
# ==========================================


def test_public_and_private_claims_are_separate(tmp_path):
    repository = make_repository(tmp_path)

    private_claim = make_claim(
        value="Private learning goal",
        experience=make_private_experience(),
    )

    public_claim = make_claim(
        value="Public knowledge",
        experience=make_public_experience(),
    )

    repository.record_claim(private_claim)
    repository.record_claim(public_claim)

    private_history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    public_history = repository.get_claim_history(
        domain=LearningDomain.PUBLIC,
    )

    assert len(private_history) == 1
    assert len(public_history) == 1

    assert private_history[0].value == "Private learning goal"
    assert public_history[0].value == "Public knowledge"


def test_public_repository_operations_reject_owner(tmp_path):
    repository = make_repository(tmp_path)

    with pytest.raises(ValueError):
        repository.list_experiences(
            domain=LearningDomain.PUBLIC,
            owner="test-user",
        )

    with pytest.raises(ValueError):
        repository.get_claim_history(
            domain=LearningDomain.PUBLIC,
            owner="test-user",
        )


# ==========================================
# INSIGHT PERSISTENCE
# ==========================================


def test_insight_survives_repository_restart(tmp_path):
    database = tmp_path / "learning.sqlite3"

    first_repository = LearningRepository(database)

    insight = make_recurring_insight(first_repository)

    insight_id = first_repository.save_insight(insight)

    second_repository = LearningRepository(database)

    restored = second_repository.get_insight(
        insight_id,
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert restored is not None
    assert restored.claim_key == insight.claim_key
    assert restored.evidence_count == 2
    assert restored.values[0].value == "Learn Python"


def test_insight_history_is_append_only(tmp_path):
    repository = make_repository(tmp_path)

    insight = make_recurring_insight(repository)

    first_id = repository.save_insight(insight)
    second_id = repository.save_insight(insight)

    assert first_id != second_id

    history = repository.list_insights(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(history) == 2


def test_insight_rejects_evidence_from_another_owner(tmp_path):
    repository = make_repository(tmp_path)

    engine = ActiveLearningEngine(min_pattern_support=2)

    for number in range(1, 3):
        observation = make_claim(
            value="Learn Python",
            experience=make_private_experience(
                owner="user-a",
                source_ref=f"conversation-{number}",
            ),
        )

        repository.record_claim(observation)
        engine.record_claim(observation)

    insight = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
    )[0]

    with pytest.raises(ValueError):
        repository.save_insight(
            insight.model_copy(
                update={"owner": "user-b"}
            )
        )


def test_insight_rejects_missing_evidence_experience(tmp_path):
    repository = make_repository(tmp_path)

    engine = ActiveLearningEngine(min_pattern_support=2)

    for number in range(1, 3):
        engine.record_claim(
            make_claim(
                value="Learn Python",
                experience=make_private_experience(
                    source_ref=f"conversation-{number}",
                ),
            )
        )

    insight = engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )[0]

    with pytest.raises(ValueError):
        repository.save_insight(insight)


# ==========================================
# CONSOLIDATION DECISION HISTORY
# ==========================================


def test_decision_survives_repository_restart(tmp_path):
    database = tmp_path / "learning.sqlite3"

    first_repository = LearningRepository(database)

    insight = make_recurring_insight(first_repository)

    decision = LearningConsolidator().evaluate(insight)

    decision_id = first_repository.save_decision(decision)

    second_repository = LearningRepository(database)

    history = second_repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(history) == 1
    assert history[0].status == decision.status
    assert history[0].claim_key == decision.claim_key


def test_decision_history_is_append_only(tmp_path):
    repository = make_repository(tmp_path)

    insight = make_recurring_insight(repository)

    consolidator = LearningConsolidator()

    first = consolidator.evaluate(insight)
    second = consolidator.evaluate(insight)

    first_id = repository.save_decision(first)
    second_id = repository.save_decision(second)

    assert first_id != second_id

    history = repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(history) == 2


def test_decisions_are_isolated_by_owner(tmp_path):
    repository = make_repository(tmp_path)

    insight = make_recurring_insight(repository)

    decision = LearningConsolidator().evaluate(insight)

    repository.save_decision(decision)

    assert repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="another-user",
    ) == []


# ==========================================
# STATISTICS
# ==========================================


def test_repository_statistics_are_scope_specific(tmp_path):
    repository = make_repository(tmp_path)

    repository.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_private_experience(
                owner="user-a",
                source_ref="conversation-a",
            ),
        )
    )

    repository.record_claim(
        make_claim(
            value="Learn Java",
            experience=make_private_experience(
                owner="user-b",
                source_ref="conversation-b",
            ),
        )
    )

    stats = repository.get_statistics(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
    )

    assert stats == {
        "experiences": 1,
        "claims": 1,
        "insights": 0,
        "decisions": 0,
    }
