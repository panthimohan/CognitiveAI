
"""
CognitiveAI - Persistent Insight Lifecycle

Coordinates:
1. Active insight discovery.
2. Idempotent insight persistence.
3. Consolidation evaluation.
4. Idempotent decision persistence.

Important boundaries:
- Private and public learning remain isolated.
- Insight history is append-only.
- Consolidation decisions do not automatically create memories.
- Existing repository persistence methods are not modified.
"""

import hashlib
import json
from typing import Any

from backend.learning.active import ActiveLearningEngine
from backend.learning.consolidation import LearningConsolidator
from backend.learning.contracts import (
    LearningDomain,
    LearningResult,
    LearningStatus,
)
from backend.learning.repository import LearningRepository


class LearningLifecycle:
    """Coordinates persistent insight discovery and consolidation."""

    def __init__(
        self,
        repository: LearningRepository,
        active_engine: ActiveLearningEngine | None = None,
        consolidator: LearningConsolidator | None = None,
    ):
        self.repository = repository
        self.active_engine = active_engine or ActiveLearningEngine(
            repository=repository
        )
        self.consolidator = consolidator or LearningConsolidator()

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

    @staticmethod
    def _unwrap_record(record: Any) -> Any:
        """
        Accept repository records represented as models, dictionaries,
        or (id, model) tuples.
        """
        if isinstance(record, tuple) and len(record) >= 2:
            return record[-1]

        if isinstance(record, dict):
            for key in ("insight", "decision", "record"):
                if key in record:
                    return record[key]

        for attribute in ("insight", "decision", "record"):
            value = getattr(record, attribute, None)
            if value is not None:
                return value

        return record

    @staticmethod
    def _model_data(model: Any) -> dict:
        """Convert a Pydantic model or dictionary into JSON data."""
        if hasattr(model, "model_dump"):
            data = model.model_dump(mode="json")
        elif isinstance(model, dict):
            data = dict(model)
        else:
            raise TypeError(
                f"Unsupported repository record: {type(model).__name__}"
            )

        return data

    @classmethod
    def _fingerprint(
        cls,
        model: Any,
        *,
        ignored_fields: tuple[str, ...] = (),
    ) -> str:
        """
        Create a stable fingerprint for duplicate detection.

        Generated identifiers and evaluation timestamps do not define
        the identity of an insight or consolidation decision.
        """
        data = cls._model_data(model)

        for field in (
            "id",
            "created_at",
            "evaluated_at",
            *ignored_fields,
        ):
            data.pop(field, None)

        canonical = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def _existing_insight_fingerprints(
        self,
        domain: LearningDomain,
        owner: str | None,
    ) -> set[str]:
        records = self.repository.list_insights(
            domain=domain,
            owner=owner,
            claim_key=None,
            limit=100000,
        )

        fingerprints = set()

        for record in records:
            insight = self._unwrap_record(record)
            fingerprints.add(self._fingerprint(insight))

        return fingerprints

    def _existing_decision_fingerprints(
        self,
        domain: LearningDomain,
        owner: str | None,
    ) -> set[str]:
        records = self.repository.list_decisions(
            domain=domain,
            owner=owner,
            claim_key=None,
            limit=100000,
        )

        fingerprints = set()

        for record in records:
            decision = self._unwrap_record(record)
            fingerprints.add(self._fingerprint(decision))

        return fingerprints

    def run(
        self,
        domain: LearningDomain,
        owner: str | None = None,
    ) -> LearningResult:
        """
        Discover insights, persist new snapshots, and consolidate them.

        Existing insights are evaluated too. This allows a later run
        to finish decision persistence if an earlier run stopped
        after saving an insight.

        Returns a LearningResult. Permanent memory promotion is
        intentionally outside this lifecycle.
        """
        self._validate_scope(domain, owner)

        insights = self.active_engine.analyze(
            domain=domain,
            owner=owner,
        )

        known_insights = self._existing_insight_fingerprints(
            domain,
            owner,
        )

        known_decisions = self._existing_decision_fingerprints(
            domain,
            owner,
        )

        insights_created = 0
        decisions_created = 0
        evidence_processed = 0

        for insight in insights:
            if insight.domain != domain or insight.owner != owner:
                raise ValueError(
                    "Active learning returned an insight outside "
                    "the requested learning scope."
                )

            evidence_processed += insight.evidence_count

            insight_fingerprint = self._fingerprint(insight)

            if insight_fingerprint not in known_insights:
                self.repository.save_insight(insight)

                known_insights.add(insight_fingerprint)
                insights_created += 1

            decision = self.consolidator.evaluate(insight)

            if decision.domain != domain or decision.owner != owner:
                raise ValueError(
                    "Consolidation returned a decision outside "
                    "the requested learning scope."
                )

            decision_fingerprint = self._fingerprint(decision)

            if decision_fingerprint not in known_decisions:
                self.repository.save_decision(decision)

                known_decisions.add(decision_fingerprint)
                decisions_created += 1

        return LearningResult(
            domain=domain,
            status=LearningStatus.COMPLETED,
            experiences_processed=evidence_processed,
            insights_created=insights_created,
            memories_created=0,
            message=(
                f"Learning lifecycle completed. "
                f"{insights_created} new insight(s) persisted; "
                f"{decisions_created} new consolidation decision(s) "
                f"persisted. No permanent memories were created."
            ),
        )
