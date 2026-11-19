
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
from backend.learning.repository import LearningRepository


def make_experience(
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


def make_claim(
    *,
    value,
    experience,
):
    return ClaimObservation(
        experience=experience,
        key="learning_goal",
        value=value,
        confidence=0.9,
    )


def test_engine_restores_claims_after_restart(tmp_path):
    database = tmp_path / "learning.sqlite3"

    repository = LearningRepository(database)

    first_engine = ActiveLearningEngine(
        repository=repository,
    )

    first_engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_experience(
                source_ref="conversation-1",
            ),
        )
    )

    first_engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_experience(
                source_ref="conversation-2",
            ),
        )
    )

    # Simulate application restart.
    second_repository = LearningRepository(database)

    second_engine = ActiveLearningEngine(
        repository=second_repository,
    )

    insights = second_engine.analyze(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(insights) == 1
    assert insights[0].insight_type == InsightType.RECURRING_PATTERN
    assert insights[0].evidence_count == 2


def test_engine_restores_complete_claim_history(tmp_path):
    database = tmp_path / "learning.sqlite3"

    repository = LearningRepository(database)

    engine = ActiveLearningEngine(repository=repository)

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_experience(
                source_ref="conversation-1",
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn machine learning",
            experience=make_experience(
                source_ref="conversation-2",
            ),
        )
    )

    restarted_engine = ActiveLearningEngine(
        repository=LearningRepository(database),
    )

    history = restarted_engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    assert len(history) == 2

    assert {
        observation.value
        for observation in history
    } == {
        "Learn Python",
        "Learn machine learning",
    }


def test_engine_persists_new_claims_automatically(tmp_path):
    repository = LearningRepository(
        tmp_path / "learning.sqlite3"
    )

    engine = ActiveLearningEngine(repository=repository)

    observation = make_claim(
        value="Learn Python",
        experience=make_experience(),
    )

    assert engine.record_claim(observation) is True

    history = repository.get_claim_history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
    )

    assert len(history) == 1
    assert history[0].value == "Learn Python"


def test_engine_ignores_duplicate_persisted_claims(tmp_path):
    database = tmp_path / "learning.sqlite3"

    first_engine = ActiveLearningEngine(
        repository=LearningRepository(database),
    )

    observation = make_claim(
        value="Learn Python",
        experience=make_experience(),
    )

    assert first_engine.record_claim(observation) is True

    second_engine = ActiveLearningEngine(
        repository=LearningRepository(database),
    )

    assert second_engine.record_claim(observation) is False

    history = second_engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    assert len(history) == 1


def test_engine_restores_private_scopes_independently(tmp_path):
    database = tmp_path / "learning.sqlite3"

    repository = LearningRepository(database)

    engine = ActiveLearningEngine(repository=repository)

    engine.record_claim(
        make_claim(
            value="Learn Python",
            experience=make_experience(
                owner="user-a",
                source_ref="conversation-a",
            ),
        )
    )

    engine.record_claim(
        make_claim(
            value="Learn Java",
            experience=make_experience(
                owner="user-b",
                source_ref="conversation-b",
            ),
        )
    )

    restarted_engine = ActiveLearningEngine(
        repository=LearningRepository(database),
    )

    user_a_history = restarted_engine.history(
        domain=LearningDomain.PRIVATE,
        owner="user-a",
        key="learning_goal",
    )

    user_b_history = restarted_engine.history(
        domain=LearningDomain.PRIVATE,
        owner="user-b",
        key="learning_goal",
    )

    assert len(user_a_history) == 1
    assert len(user_b_history) == 1

    assert user_a_history[0].value == "Learn Python"
    assert user_b_history[0].value == "Learn Java"


def test_engine_does_not_mix_public_and_private_learning(tmp_path):
    database = tmp_path / "learning.sqlite3"

    repository = LearningRepository(database)

    private_experience = make_experience()

    public_experience = LearningExperience(
        domain=LearningDomain.PUBLIC,
        content="A trusted public source.",
        source_type=ExperienceSource.TRUSTED_EXTERNAL_SOURCE,
        source_ref="public-source-1",
        authorization_basis=AuthorizationBasis.TRUSTED_SOURCE_POLICY,
    )

    engine = ActiveLearningEngine(repository=repository)

    engine.record_claim(
        make_claim(
            value="Private goal",
            experience=private_experience,
        )
    )

    engine.record_claim(
        make_claim(
            value="Public knowledge",
            experience=public_experience,
        )
    )

    restarted_engine = ActiveLearningEngine(
        repository=LearningRepository(database),
    )

    private_history = restarted_engine.history(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        key="learning_goal",
    )

    public_history = restarted_engine.history(
        domain=LearningDomain.PUBLIC,
        key="learning_goal",
    )

    assert len(private_history) == 1
    assert len(public_history) == 1

    assert private_history[0].value == "Private goal"
    assert public_history[0].value == "Public knowledge"
