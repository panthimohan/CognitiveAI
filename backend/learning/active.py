
"""
CognitiveAI - Active Learning Engine

Records structured claims, tracks their evidence, and identifies:
- Recurring patterns.
- Possible changes or conflicts.
- Historical support for each observation.

Responsibilities:
- Restore historical claims from persistent storage.
- Persist new claims when a repository is configured.
- Identify evidence-backed patterns and possible changes.
- Preserve historical observations.
- Enforce private/public learning boundaries.

This engine does not interpret raw conversation text.
Claims must first be extracted into a structured form.

It does not automatically promote observations into
permanent memory.
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from backend.learning.contracts import (
    LearningDomain,
    LearningExperience,
)


# ==========================================
# 1. MODELS
# ==========================================


class ClaimObservation(BaseModel):
    """A structured claim supported by an authorized experience."""

    experience: LearningExperience

    key: str = Field(min_length=1)
    value: str = Field(min_length=1)

    confidence: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def validate_claim(self):
        if not self.key.strip():
            raise ValueError("Claim key cannot be empty.")

        if not self.value.strip():
            raise ValueError("Claim value cannot be empty.")

        return self


class InsightType(str, Enum):
    RECURRING_PATTERN = "recurring_pattern"
    POSSIBLE_CHANGE = "possible_change"


class EvidenceReference(BaseModel):
    """A reference to the source supporting an observation."""

    experience_id: str
    source_ref: str
    observed_at: datetime
    confidence: float = Field(ge=0, le=1)


class ValueEvidence(BaseModel):
    """Evidence supporting one observed value."""

    value: str
    support_count: int = Field(ge=1)
    evidence: list[EvidenceReference]


class LearningInsight(BaseModel):
    """An evidence-backed observation awaiting evaluation."""

    domain: LearningDomain
    owner: str | None = None

    insight_type: InsightType
    claim_key: str

    values: list[ValueEvidence]

    evidence_count: int = Field(ge=1)
    evidence_strength: float = Field(ge=0, le=1)

    first_seen: datetime
    last_seen: datetime


# ==========================================
# 2. ACTIVE LEARNING ENGINE
# ==========================================


class ActiveLearningEngine:
    """
    Tracks structured claims and discovers evidence-backed insights.

    Claims are isolated by:
        - Learning domain.
        - Private owner, where applicable.
        - Claim key.

    When a repository is configured:
        - New observations are persisted.
        - Historical observations are restored on demand.
        - Duplicate observations are ignored.
        - Insights can be rebuilt after application restarts.

    Without a repository, the engine retains its original
    in-memory behavior.
    """

    def __init__(
        self,
        min_pattern_support: int = 2,
        *,
        repository: Any | None = None,
    ):
        if min_pattern_support < 2:
            raise ValueError(
                "Minimum pattern support must be at least 2."
            )

        self.min_pattern_support = min_pattern_support
        self.repository = repository

        self._claims: dict[
            tuple[LearningDomain, str | None, str],
            list[ClaimObservation],
        ] = defaultdict(list)

        self._seen_claims: set[
            tuple[
                LearningDomain,
                str | None,
                str,
                str,
                str,
            ]
        ] = set()

        # A scope is loaded only once per engine instance.
        self._loaded_scopes: set[
            tuple[LearningDomain, str | None]
        ] = set()

    # --------------------------------------
    # Recording
    # --------------------------------------

    def record_claim(
        self,
        observation: ClaimObservation,
    ) -> bool:
        """
        Record a structured claim.

        When persistence is enabled, the repository is the
        source of truth for duplicate detection.

        Returns:
            True: claim was recorded.
            False: identical claim from the same experience
                   was already recorded.
        """

        experience = observation.experience

        domain = experience.domain
        owner = experience.owner

        self._validate_scope(domain, owner)

        if self.repository is not None:
            # Persist first. A failed database write must not
            # leave an observation only in application memory.
            recorded = self.repository.record_claim(observation)

            if not recorded:
                return False

        return self._record_in_memory(observation)

    def _record_in_memory(
        self,
        observation: ClaimObservation,
    ) -> bool:
        """Add an observation to the current in-memory index."""

        experience = observation.experience

        domain = experience.domain
        owner = experience.owner

        normalized_key = self._normalize(observation.key)
        normalized_value = self._normalize(observation.value)

        storage_key = (
            domain,
            owner,
            normalized_key,
        )

        deduplication_key = (
            domain,
            owner,
            experience.id,
            normalized_key,
            normalized_value,
        )

        if deduplication_key in self._seen_claims:
            return False

        self._claims[storage_key].append(observation)
        self._seen_claims.add(deduplication_key)

        return True

    # --------------------------------------
    # Persistent restoration
    # --------------------------------------

    def _load_scope(
        self,
        *,
        domain: LearningDomain,
        owner: str | None,
    ) -> None:
        """
        Restore a learning scope from persistent storage.

        Loading is lazy: only the requested scope is restored.
        Other private owners and public learning are untouched.
        """

        self._validate_scope(domain, owner)

        scope_key = (domain, owner)

        if scope_key in self._loaded_scopes:
            return

        if self.repository is not None:
            observations = self.repository.get_claim_history(
                domain=domain,
                owner=owner,
            )

            for observation in observations:
                self._record_in_memory(observation)

        self._loaded_scopes.add(scope_key)

    # --------------------------------------
    # Insight discovery
    # --------------------------------------

    def analyze(
        self,
        *,
        domain: LearningDomain,
        owner: str | None = None,
    ) -> list[LearningInsight]:
        """
        Analyze claims belonging to exactly one learning scope.

        Historical claims are restored before analysis.

        Repeated values become recurring-pattern candidates.
        Different values become possible-change candidates.
        """

        self._validate_scope(domain, owner)

        self._load_scope(
            domain=domain,
            owner=owner,
        )

        insights = []

        for (claim_domain, claim_owner, claim_key), claims in (
            self._claims.items()
        ):
            if claim_domain != domain:
                continue

            if claim_owner != owner:
                continue

            insight = self._build_insight(
                domain=domain,
                owner=owner,
                claim_key=claim_key,
                claims=claims,
            )

            if insight is not None:
                insights.append(insight)

        return sorted(
            insights,
            key=lambda item: item.last_seen,
            reverse=True,
        )

    def _build_insight(
        self,
        *,
        domain: LearningDomain,
        owner: str | None,
        claim_key: str,
        claims: list[ClaimObservation],
    ) -> LearningInsight | None:

        grouped: dict[
            str,
            list[ClaimObservation],
        ] = defaultdict(list)

        for claim in claims:
            grouped[self._normalize(claim.value)].append(claim)

        if len(grouped) > 1:
            insight_type = InsightType.POSSIBLE_CHANGE

        elif len(grouped) == 1:
            only_group = next(iter(grouped.values()))

            if len(only_group) < self.min_pattern_support:
                return None

            insight_type = InsightType.RECURRING_PATTERN

        else:
            return None

        value_evidence = []

        all_claims = []

        for supporting_claims in grouped.values():
            supporting_claims.sort(
                key=lambda item: item.experience.created_at
            )

            original_value = supporting_claims[0].value

            references = [
                EvidenceReference(
                    experience_id=claim.experience.id,
                    source_ref=claim.experience.source_ref,
                    observed_at=claim.experience.created_at,
                    confidence=claim.confidence,
                )
                for claim in supporting_claims
            ]

            value_evidence.append(
                ValueEvidence(
                    value=original_value,
                    support_count=len(supporting_claims),
                    evidence=references,
                )
            )

            all_claims.extend(supporting_claims)

        all_claims.sort(
            key=lambda item: item.experience.created_at
        )

        average_confidence = (
            sum(claim.confidence for claim in all_claims)
            / len(all_claims)
        )

        return LearningInsight(
            domain=domain,
            owner=owner,
            insight_type=insight_type,
            claim_key=claim_key,
            values=value_evidence,
            evidence_count=len(all_claims),
            evidence_strength=round(average_confidence, 4),
            first_seen=all_claims[0].experience.created_at,
            last_seen=all_claims[-1].experience.created_at,
        )

    # --------------------------------------
    # Historical evidence
    # --------------------------------------

    def history(
        self,
        *,
        domain: LearningDomain,
        key: str,
        owner: str | None = None,
    ) -> list[ClaimObservation]:
        """Return the complete chronological history of a claim."""

        self._validate_scope(domain, owner)

        self._load_scope(
            domain=domain,
            owner=owner,
        )

        storage_key = (
            domain,
            owner,
            self._normalize(key),
        )

        return sorted(
            self._claims.get(storage_key, []),
            key=lambda item: item.experience.created_at,
        )

    # --------------------------------------
    # Privacy boundaries
    # --------------------------------------

    @staticmethod
    def _validate_scope(
        domain: LearningDomain,
        owner: str | None,
    ) -> None:

        if domain == LearningDomain.PRIVATE:
            if not owner or not owner.strip():
                raise ValueError(
                    "Private learning requires an owner."
                )

        elif domain == LearningDomain.PUBLIC:
            if owner is not None:
                raise ValueError(
                    "Public learning cannot have a private owner."
                )

        else:
            raise ValueError("Unsupported learning domain.")

    # --------------------------------------
    # Utilities
    # --------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())
