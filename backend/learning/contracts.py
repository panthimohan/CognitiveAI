
"""
CognitiveAI - Dual Learning Contracts

Defines the shared data contracts used by private and public learning.

Security boundaries:
- Private experiences always belong to an owner.
- Public experiences never contain a private owner.
- Private conversations cannot enter public learning directly.
- Public learning requires explicit authorization:
    1. Explicit user approval for shared contributions.
    2. Trusted-source authorization for external knowledge.
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LearningDomain(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class ExperienceSource(str, Enum):
    PRIVATE_CONVERSATION = "private_conversation"
    APPROVED_CONTRIBUTION = "approved_contribution"
    TRUSTED_EXTERNAL_SOURCE = "trusted_external_source"


class AuthorizationBasis(str, Enum):
    PRIVATE_CONTEXT = "private_context"
    EXPLICIT_USER_APPROVAL = "explicit_user_approval"
    TRUSTED_SOURCE_POLICY = "trusted_source_policy"


class LearningStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    FAILED = "failed"


class LearningExperience(BaseModel):
    """
    A unit of information that a learning engine is authorized to process.

    This is not necessarily a memory.
    It is an input to the learning process.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    domain: LearningDomain
    owner: str | None = None

    content: str = Field(min_length=1)

    source_type: ExperienceSource
    source_ref: str = Field(min_length=1)

    authorization_basis: AuthorizationBasis

    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_learning_boundary(self):
        if not self.content.strip():
            raise ValueError("Learning content cannot be empty.")

        if not self.source_ref.strip():
            raise ValueError("Source reference cannot be empty.")

        if self.domain == LearningDomain.PRIVATE:
            if not self.owner or not self.owner.strip():
                raise ValueError(
                    "Private learning requires an owner."
                )

            if self.source_type != ExperienceSource.PRIVATE_CONVERSATION:
                raise ValueError(
                    "Private learning must originate from a private "
                    "conversation."
                )

            if (
                self.authorization_basis
                != AuthorizationBasis.PRIVATE_CONTEXT
            ):
                raise ValueError(
                    "Private conversations require private-context "
                    "authorization."
                )

        elif self.domain == LearningDomain.PUBLIC:
            if self.owner is not None:
                raise ValueError(
                    "Public learning experiences cannot have a private owner."
                )

            if (
                self.source_type
                == ExperienceSource.PRIVATE_CONVERSATION
            ):
                raise ValueError(
                    "Private conversations cannot enter public learning."
                )

            if (
                self.source_type
                == ExperienceSource.APPROVED_CONTRIBUTION
                and self.authorization_basis
                != AuthorizationBasis.EXPLICIT_USER_APPROVAL
            ):
                raise ValueError(
                    "Shared contributions require explicit user approval."
                )

            if (
                self.source_type
                == ExperienceSource.TRUSTED_EXTERNAL_SOURCE
                and self.authorization_basis
                != AuthorizationBasis.TRUSTED_SOURCE_POLICY
            ):
                raise ValueError(
                    "External knowledge requires trusted-source "
                    "authorization."
                )

        return self


class LearningResult(BaseModel):
    """A structured result from a learning operation."""

    domain: LearningDomain
    status: LearningStatus

    experiences_processed: int = Field(default=0, ge=0)
    insights_created: int = Field(default=0, ge=0)
    memories_created: int = Field(default=0, ge=0)

    message: str = ""
    created_at: datetime = Field(default_factory=utc_now)
