
import pytest

from backend.learning.active import ActiveLearningEngine, ClaimObservation
from backend.learning.consolidation import LearningConsolidator
from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
    LearningExperience,
    LearningStatus,
)
from backend.learning.lifecycle import LearningLifecycle
from backend.learning.repository import LearningRepository


def make_lifecycle(database_path):
    repository = LearningRepository(database_path=database_path)
    engine = ActiveLearningEngine(
        repository=repository,
        min_pattern_support=2,
    )
    consolidator = LearningConsolidator(
        min_pattern_support=2,
        min_confidence=0.70,
    )
    lifecycle = LearningLifecycle(
        repository=repository,
        active_engine=engine,
        consolidator=consolidator,
    )
    return lifecycle, repository


def add_claims(engine, domain, owner, key, value):
    for index in range(2):
        private = domain == LearningDomain.PRIVATE

        experience = LearningExperience(
            domain=domain,
            owner=owner,
            content=f"Evidence {index}: {key} is {value}",
            source_type=(
                ExperienceSource.PRIVATE_CONVERSATION
                if private
                else ExperienceSource.TRUSTED_EXTERNAL_SOURCE
            ),
            source_ref=f"source-{domain.value}-{owner or 'public'}-{index}",
            authorization_basis=(
                AuthorizationBasis.PRIVATE_CONTEXT
                if private
                else AuthorizationBasis.TRUSTED_SOURCE_POLICY
            ),
        )

        observation = ClaimObservation(
            experience=experience,
            key=key,
            value=value,
            confidence=0.95,
        )
        assert engine.record_claim(observation) is True


def test_lifecycle_persists_insight_and_decision(tmp_path):
    lifecycle, repository = make_lifecycle(
        tmp_path / "learning.sqlite3"
    )
    add_claims(
        lifecycle.active_engine,
        LearningDomain.PRIVATE,
        "user-1",
        "preferred_language",
        "Python",
    )

    result = lifecycle.run(LearningDomain.PRIVATE, "user-1")

    insights = repository.list_insights(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )
    decisions = repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )

    assert result.status == LearningStatus.COMPLETED
    assert result.insights_created == 1
    assert result.memories_created == 0
    assert len(insights) == 1
    assert len(decisions) == 1


def test_repeated_run_does_not_duplicate_records(tmp_path):
    lifecycle, repository = make_lifecycle(
        tmp_path / "learning.sqlite3"
    )
    add_claims(
        lifecycle.active_engine,
        LearningDomain.PRIVATE,
        "user-1",
        "preferred_language",
        "Python",
    )

    first = lifecycle.run(LearningDomain.PRIVATE, "user-1")
    second = lifecycle.run(LearningDomain.PRIVATE, "user-1")

    insights = repository.list_insights(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )
    decisions = repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )

    assert first.insights_created == 1
    assert second.insights_created == 0
    assert len(insights) == 1
    assert len(decisions) == 1


def test_lifecycle_recovers_after_repository_restart(tmp_path):
    database_path = tmp_path / "learning.sqlite3"
    lifecycle, _ = make_lifecycle(database_path)

    add_claims(
        lifecycle.active_engine,
        LearningDomain.PRIVATE,
        "user-1",
        "preferred_language",
        "Python",
    )
    lifecycle.run(LearningDomain.PRIVATE, "user-1")

    restarted, repository = make_lifecycle(database_path)
    result = restarted.run(LearningDomain.PRIVATE, "user-1")

    insights = repository.list_insights(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )
    decisions = repository.list_decisions(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )

    assert result.insights_created == 0
    assert len(insights) == 1
    assert len(decisions) == 1


def test_private_and_public_learning_are_isolated(tmp_path):
    lifecycle, repository = make_lifecycle(
        tmp_path / "learning.sqlite3"
    )

    add_claims(
        lifecycle.active_engine,
        LearningDomain.PRIVATE,
        "user-1",
        "preferred_language",
        "Python",
    )
    add_claims(
        lifecycle.active_engine,
        LearningDomain.PUBLIC,
        None,
        "common_language",
        "Python",
    )

    lifecycle.run(LearningDomain.PRIVATE, "user-1")
    lifecycle.run(LearningDomain.PUBLIC, None)

    private_insights = repository.list_insights(
        domain=LearningDomain.PRIVATE,
        owner="user-1",
        claim_key=None,
    )
    public_insights = repository.list_insights(
        domain=LearningDomain.PUBLIC,
        owner=None,
        claim_key=None,
    )

    assert len(private_insights) == 1
    assert len(public_insights) == 1
    assert private_insights[0].owner == "user-1"
    assert public_insights[0].owner is None


def test_lifecycle_rejects_invalid_scope(tmp_path):
    lifecycle, _ = make_lifecycle(
        tmp_path / "learning.sqlite3"
    )

    with pytest.raises(ValueError):
        lifecycle.run(LearningDomain.PRIVATE, None)

    with pytest.raises(ValueError):
        lifecycle.run(LearningDomain.PUBLIC, "user-1")
