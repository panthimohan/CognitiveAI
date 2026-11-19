
import pytest
from pydantic import ValidationError

from backend.learning.contracts import (
    AuthorizationBasis,
    ExperienceSource,
    LearningDomain,
    LearningExperience,
    LearningResult,
    LearningStatus,
)


def test_private_learning_accepts_private_conversation():
    experience = LearningExperience(
        domain=LearningDomain.PRIVATE,
        owner="test-user",
        content="The user is learning Python.",
        source_type=ExperienceSource.PRIVATE_CONVERSATION,
        source_ref="conversation-1",
        authorization_basis=AuthorizationBasis.PRIVATE_CONTEXT,
    )

    assert experience.domain == LearningDomain.PRIVATE
    assert experience.owner == "test-user"


def test_private_learning_requires_owner():
    with pytest.raises(ValidationError):
        LearningExperience(
            domain=LearningDomain.PRIVATE,
            content="Private information.",
            source_type=ExperienceSource.PRIVATE_CONVERSATION,
            source_ref="conversation-1",
            authorization_basis=AuthorizationBasis.PRIVATE_CONTEXT,
        )


def test_public_learning_accepts_approved_contribution():
    experience = LearningExperience(
        domain=LearningDomain.PUBLIC,
        content="A user-approved shared contribution.",
        source_type=ExperienceSource.APPROVED_CONTRIBUTION,
        source_ref="approval-123",
        authorization_basis=AuthorizationBasis.EXPLICIT_USER_APPROVAL,
    )

    assert experience.domain == LearningDomain.PUBLIC
    assert experience.owner is None


def test_public_learning_accepts_trusted_external_source():
    experience = LearningExperience(
        domain=LearningDomain.PUBLIC,
        content="Knowledge from a trusted external source.",
        source_type=ExperienceSource.TRUSTED_EXTERNAL_SOURCE,
        source_ref="source-123",
        authorization_basis=AuthorizationBasis.TRUSTED_SOURCE_POLICY,
    )

    assert experience.domain == LearningDomain.PUBLIC


def test_private_conversation_cannot_enter_public_learning():
    with pytest.raises(ValidationError):
        LearningExperience(
            domain=LearningDomain.PUBLIC,
            content="Private conversation content.",
            source_type=ExperienceSource.PRIVATE_CONVERSATION,
            source_ref="conversation-1",
            authorization_basis=AuthorizationBasis.PRIVATE_CONTEXT,
        )


def test_public_learning_rejects_unapproved_contribution():
    with pytest.raises(ValidationError):
        LearningExperience(
            domain=LearningDomain.PUBLIC,
            content="Unapproved contribution.",
            source_type=ExperienceSource.APPROVED_CONTRIBUTION,
            source_ref="contribution-1",
            authorization_basis=AuthorizationBasis.TRUSTED_SOURCE_POLICY,
        )


def test_public_learning_cannot_have_private_owner():
    with pytest.raises(ValidationError):
        LearningExperience(
            domain=LearningDomain.PUBLIC,
            owner="test-user",
            content="Shared knowledge.",
            source_type=ExperienceSource.APPROVED_CONTRIBUTION,
            source_ref="approval-123",
            authorization_basis=AuthorizationBasis.EXPLICIT_USER_APPROVAL,
        )


def test_learning_result_tracks_processing():
    result = LearningResult(
        domain=LearningDomain.PRIVATE,
        status=LearningStatus.COMPLETED,
        experiences_processed=10,
        insights_created=3,
        memories_created=2,
    )

    assert result.experiences_processed == 10
    assert result.insights_created == 3
    assert result.memories_created == 2
