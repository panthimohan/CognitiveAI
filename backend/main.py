
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.supervisor.router import TaskRequest, route_task
from backend.supervisor.planner import CognitivePlanner
from backend.supervisor.evaluator import CognitiveEvaluator
from backend.supervisor.semantic_evaluator import (
    SemanticEvaluator,
)

from backend.models.factory import create_provider
from backend.models.providers import ModelRequest

from backend.memory.manager import MemoryManager
from backend.memory.processor import MemoryProcessor
from backend.memory.extractor import MemoryExtractor
from backend.conversations.archive import ConversationArchive

from backend.learning.contracts import LearningDomain
from backend.learning.queue import LearningQueue
from backend.learning.repository import LearningRepository
from backend.learning.active import ActiveLearningEngine
from backend.learning.consolidation import LearningConsolidator
from backend.learning.lifecycle import LearningLifecycle
from backend.learning.claim_extractor import ClaimExtractor
from backend.learning.background import PrivateLearningWorker

from backend.transparency.events import EventStatus, EventType
from backend.transparency.event_store import InMemoryEventStore
from backend.transparency.recorder import TransparencyRecorder
from backend.transparency.audit import TransparencyAudit

from config.settings import get_settings


logger = logging.getLogger("cognitiveai")


# ==========================================
# MEMORY AND ARCHIVE SERVICES
# ==========================================

memory_manager = MemoryManager(
    owner=os.getenv("COGNITIVEAI_OWNER", "local-user")
)

memory_processor = MemoryProcessor(memory_manager)

conversation_archive = ConversationArchive()


# ==========================================
# LEARNING SERVICES
# ==========================================

# Learning storage is independent of the conversation archive
# and the permanent-memory database.

learning_queue = LearningQueue()
learning_repository = LearningRepository()

active_learning_engine = ActiveLearningEngine(
    repository=learning_repository,
)

learning_consolidator = LearningConsolidator()

learning_lifecycle = LearningLifecycle(
    repository=learning_repository,
    active_engine=active_learning_engine,
    consolidator=learning_consolidator,
)

# Background extraction uses the configured general-purpose
# local model. It does not use a cloud provider.

settings = get_settings()

background_model = settings.local_model_for("conversation")

background_provider = create_provider(
    name="local",
    model=background_model,
)

background_memory_extractor = MemoryExtractor(
    provider=background_provider,
)

background_claim_extractor = ClaimExtractor(
    provider=background_provider,
)

private_learning_worker = PrivateLearningWorker(
    archive=conversation_archive,
    queue=learning_queue,
    extractor=background_memory_extractor,
    processor=memory_processor,
    claim_extractor=background_claim_extractor,
    active_engine=active_learning_engine,
    lifecycle=learning_lifecycle,
)


# ==========================================
# BACKGROUND WORKER CONFIGURATION
# ==========================================

def background_learning_enabled() -> bool:
    value = os.getenv(
        "COGNITIVEAI_BACKGROUND_LEARNING_ENABLED",
        "true",
    ).strip().lower()

    return value in {"1", "true", "yes", "on"}


async def background_learning_loop() -> None:
    """
    Continuously process durable private-learning jobs.

    The queue owns retry and checkpoint behavior.
    The loop only waits when there is no available work.
    """

    logger.info("Background learning worker started.")

    while True:
        try:
            result = await private_learning_worker.process_next()

            if result is None:
                await asyncio.sleep(3)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Unexpected background learning worker error."
            )

            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle.

    When enabled, recover archived conversations into the
    durable queue and start the background processing loop.
    """

    worker_task = None

    if background_learning_enabled():
        try:
            recovered = learning_queue.enqueue_archived_conversations(
                conversation_archive,
                owner=memory_manager.owner,
            )

            logger.info(
                "Background learning startup recovery completed: %s",
                recovered,
            )

        except Exception:
            logger.exception(
                "Could not recover archived conversations."
            )

        worker_task = asyncio.create_task(
            background_learning_loop()
        )

    else:
        logger.info(
            "Background learning is disabled. "
            "Set COGNITIVEAI_BACKGROUND_LEARNING_ENABLED=true to enable it."
        )

    yield

    if worker_task is not None:
        worker_task.cancel()

        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    logger.info("CognitiveAI application shutdown complete.")


# ==========================================
# APPLICATION
# ==========================================

app = FastAPI(
    title="CognitiveAI",
    description="Metacognitive Intelligence Platform",
    version="0.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

planner = CognitivePlanner()
evaluator = CognitiveEvaluator()

semantic_evaluator = SemanticEvaluator(
    deterministic_evaluator=evaluator
)


# ==========================================
# TRANSPARENCY SERVICES
# ==========================================

transparency_store = InMemoryEventStore()
transparency_recorder = TransparencyRecorder(transparency_store)
transparency_audit = TransparencyAudit(transparency_store)


# ==========================================
# REQUEST MODELS
# ==========================================

class EvaluationInput(BaseModel):
    completeness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    privacy: str = "private"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=1, le=10000)


# ==========================================
# TRANSPARENCY HELPERS
# ==========================================

def record_failure(
    interaction_id: str,
    stage: str,
    error: Exception,
) -> None:
    """Record a failure without copying sensitive error contents."""

    transparency_recorder.record(
        user_id=memory_manager.owner,
        interaction_id=interaction_id,
        event_type=EventType.INTERACTION_FAILED,
        source="supervisor_ask",
        status=EventStatus.FAILED,
        details={
            "stage": stage,
            "error_type": type(error).__name__,
        },
    )


# ==========================================
# APPLICATION ENDPOINTS
# ==========================================

@app.get("/")
async def home():
    return {
        "application": "CognitiveAI",
        "status": "running",
        "version": "0.4.1",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/learning/status")
async def learning_status():
    """
    Report the configuration and persistent learning statistics.

    This endpoint does not expose private conversation contents.
    """

    try:
        statistics = learning_repository.get_statistics(
            domain=LearningDomain.PRIVATE,
            owner=memory_manager.owner,
        )

        return {
            "background_learning_enabled": background_learning_enabled(),
            "worker": (
                "enabled"
                if background_learning_enabled()
                else "disabled"
            ),
            "owner_scope": "private",
            "repository": statistics,
        }

    except Exception as error:
        logger.exception("Could not retrieve learning status.")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve learning status.",
        ) from error


# ==========================================
# SUPERVISOR ROUTING
# ==========================================

@app.post("/supervisor/route")
async def supervisor_route(request: TaskRequest):
    try:
        return route_task(request).model_dump()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/supervisor/plan")
async def supervisor_plan(request: TaskRequest):
    return planner.create_plan(request.task).model_dump()


@app.post("/supervisor/evaluate")
async def supervisor_evaluate(request: EvaluationInput):
    result = evaluator.evaluate(
        completeness=request.completeness,
        relevance=request.relevance,
        confidence=request.confidence,
    )

    return result.model_dump()


@app.post("/supervisor/inspect")
async def supervisor_inspect(request: TaskRequest):
    try:
        routing = route_task(request)
        plan = planner.create_plan(request.task)

        return {
            "request": request.task,
            "supervisor": "CognitiveAI",
            "routing_decision": routing.model_dump(),
            "task_plan": plan.model_dump(),
            "execution_status": "planning_complete",
            "note": "No AI model was executed.",
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


# ==========================================
# MEMORY CONTEXT
# ==========================================

def build_memory_context(memories) -> str:
    if not memories:
        return "No relevant memories were retrieved."

    sections = []

    for memory in memories:
        sections.append(
            f"[{memory.kind.value}] {memory.content}"
        )

    return "\n".join(sections)


# ==========================================
# TRANSPARENCY API
# ==========================================

@app.get("/transparency/interactions/{interaction_id}")
async def get_interaction_transparency(interaction_id: str):
    report = transparency_audit.get_interaction(
        user_id=memory_manager.owner,
        interaction_id=interaction_id,
    )

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Transparency report not found.",
        )

    return report


@app.get("/transparency/events")
async def get_transparency_events(limit: int = 100):
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500.",
        )

    return {
        "events": transparency_audit.list_events(
            user_id=memory_manager.owner,
            limit=limit,
        )
    }


# ==========================================
# CONVERSATION ARCHIVE API
# ==========================================

@app.get("/conversations/{conversation_id}")
async def get_archived_conversation(
    conversation_id: str,
    privacy: str | None = None,
):
    """Return one owner's archived conversation in chronological order."""

    if privacy is not None and privacy not in {"private", "public"}:
        raise HTTPException(
            status_code=400,
            detail="Privacy must be private or public.",
        )

    messages = conversation_archive.get_conversation(
        owner=memory_manager.owner,
        conversation_id=conversation_id,
        privacy=privacy,
    )

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "id": message.id,
                "interaction_id": message.interaction_id,
                "role": message.role,
                "content": message.content,
                "privacy": message.privacy,
                "created_at": message.created_at,
            }
            for message in messages
        ],
        "count": len(messages),
    }


# ==========================================
# IMMEDIATE MEMORY EXTRACTION
# ==========================================

async def automatically_remember(
    question: str,
    provider,
    interaction_id: str,
    user_id: str,
) -> dict:
    """
    Extract useful information from the user's question
    and save accepted candidates to personal memory.

    Memory failures never interrupt the main answer.
    """

    result = {
        "memories_saved": 0,
        "candidates_found": 0,
        "extraction_status": "failed",
    }

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MEMORY_EXTRACTION_STARTED,
        source="memory_extractor",
        status=EventStatus.STARTED,
        details={
            "scope": "personal",
            "source": "user_question",
        },
    )

    try:
        extractor = MemoryExtractor(provider=provider)

        candidates = await extractor.extract(question)

        result["candidates_found"] = len(candidates)

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.MEMORY_EXTRACTION_COMPLETED,
            source="memory_extractor",
            status=EventStatus.COMPLETED,
            details={
                "candidates_found": len(candidates),
                "scope": "personal",
            },
        )

        processed_count = 0
        saved_count = 0

        for candidate in candidates:
            try:
                processed = memory_processor.process(candidate)
                processed_count += 1

                if processed.status == "saved":
                    saved_count += 1

            except Exception:
                continue

        result["memories_saved"] = saved_count
        result["extraction_status"] = "completed"

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.MEMORY_SAVED,
            source="memory_processor",
            status=EventStatus.RECORDED,
            details={
                "candidates_processed": processed_count,
                "memories_saved": saved_count,
                "scope": "personal",
            },
        )

    except Exception as error:
        result["extraction_status"] = "failed"

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.MEMORY_EXTRACTION_FAILED,
            source="memory_extractor",
            status=EventStatus.FAILED,
            details={
                "error_type": type(error).__name__,
                "scope": "personal",
            },
        )

    return result


# ==========================================
# BACKGROUND QUEUE ENQUEUE
# ==========================================

def enqueue_private_learning(
    user_id: str,
) -> dict:
    """
    Discover archived private conversations and enqueue work.

    Queue failures do not invalidate a completed user response.
    The queue performs duplicate and checkpoint checks.
    """

    if user_id != memory_manager.owner:
        return {
            "status": "skipped",
            "reason": "owner_scope_mismatch",
        }

    try:
        queued = learning_queue.enqueue_archived_conversations(
            conversation_archive,
            owner=user_id,
        )

        return {
            "status": "queued",
            "discovered": queued,
        }

    except Exception as error:
        logger.exception(
            "Could not enqueue private conversation for learning."
        )

        return {
            "status": "queue_error",
            "error_type": type(error).__name__,
        }


# ==========================================
# REAL QUESTION-ANSWER PIPELINE
# ==========================================

@app.post("/supervisor/ask")
async def supervisor_ask(request: AskRequest):

    # 1. Validate privacy.

    privacy = request.privacy.lower()

    if privacy not in {"private", "public"}:
        raise HTTPException(
            status_code=400,
            detail="Privacy must be private or public.",
        )

    interaction_id = str(uuid4())
    user_id = memory_manager.owner
    conversation_id = request.conversation_id or str(uuid4())

    # 2. Archive the original user message.

    try:
        conversation_archive.save_message(
            owner=user_id,
            conversation_id=conversation_id,
            interaction_id=interaction_id,
            role="user",
            content=request.question,
            privacy=privacy,
        )

    except Exception as error:
        record_failure(
            interaction_id,
            "archive_user_message",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail="Could not archive your message. The request was not processed.",
        )

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.INTERACTION_STARTED,
        source="supervisor_ask",
        status=EventStatus.STARTED,
        details={"privacy": privacy},
    )

    # 3. Supervisor classification and planning.

    task_request = TaskRequest(
        task=request.question,
        privacy=privacy,
    )

    try:
        routing = route_task(task_request)

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.TASK_CLASSIFIED,
            source="supervisor_router",
            status=EventStatus.COMPLETED,
            details={
                "task_type": str(routing.task_type),
                "complexity": str(routing.complexity),
                "decision_id": routing.decision_id,
            },
        )

        plan = planner.create_plan(request.question)

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.PLAN_CREATED,
            source="supervisor_planner",
            status=EventStatus.COMPLETED,
        )

    except ValueError as error:
        record_failure(
            interaction_id,
            "classification_or_planning",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # 4. Select the configured local model.

    try:
        settings = get_settings()
        local_model = settings.local_model_for(routing.task_type)

    except Exception as error:
        record_failure(
            interaction_id,
            "model_configuration",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Model configuration failed.",
        )

    # 5. Retrieve memories under the requested privacy boundary.

    try:
        memories = memory_manager.recall(
            query=request.question,
            privacy=privacy,
            limit=5,
        )

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.MEMORY_RETRIEVED,
            source="memory_manager",
            status=EventStatus.COMPLETED,
            details={
                "count": len(memories),
                "memory_ids": [item.id for item in memories],
                "memory_scopes": [
                    item.scope.value for item in memories
                ],
                "memory_kinds": [
                    item.kind.value for item in memories
                ],
            },
        )

    except (ValueError, OSError) as error:
        record_failure(
            interaction_id,
            "memory_retrieval",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Memory retrieval failed.",
        )

    memory_context = build_memory_context(memories)

    # 6. Build the original answer prompt.

    system_prompt = (
        "You are CognitiveAI, a thoughtful and transparent "
        "AI assistant. Use relevant memory as contextual "
        "information, not as unquestionable truth. "
        "If memory conflicts with the user's current statement, "
        "prioritize the current statement. "
        "Do not claim to have performed actions you did not perform."
    )

    prompt = (
        f"Relevant memory:\n{memory_context}\n\n"
        f"User question:\n{request.question}"
    )

    model_request = ModelRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # 7. Create the local provider.

    try:
        provider = create_provider(
            name="local",
            model=local_model,
        )

    except Exception as error:
        record_failure(
            interaction_id,
            "provider_initialization",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail="Local model provider initialization failed.",
        )

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MODEL_SELECTED,
        source="model_factory",
        status=EventStatus.COMPLETED,
        details={
            "provider": "local",
            "model": local_model,
            "destination": "local",
        },
    )

    # 8. Generate the initial answer locally.

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MODEL_EXECUTION_STARTED,
        source="model_provider",
        status=EventStatus.STARTED,
    )

    try:
        model_response = await provider.generate(model_request)

    except Exception as error:
        record_failure(
            interaction_id,
            "model_execution",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail="Local model execution failed.",
        )

    if not model_response.success:
        error = RuntimeError(
            "Model returned an unsuccessful response."
        )

        record_failure(
            interaction_id,
            "model_execution",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                model_response.error
                or "Local model failed to generate an answer."
            ),
        )

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MODEL_EXECUTION_COMPLETED,
        source="model_provider",
        status=EventStatus.COMPLETED,
        details={
            "provider": model_response.provider,
            "model": model_response.model,
        },
    )

    # 9. Evaluate the initial answer.
    #
    # Evaluation uses the same local provider.
    # If evaluation fails, retain the generated answer and report
    # that evaluation was unavailable rather than inventing a result.

    final_answer = model_response.content
    evaluation_result = None
    evaluation_status = "unavailable"
    evaluation_decision = None
    revision_attempts = 0
    initial_outcome = None

    try:
        initial_outcome = await semantic_evaluator.evaluate_response(
            question=request.question,
            answer=final_answer,
            provider=provider,
        )

        evaluation_result = initial_outcome.model_dump()
        evaluation_status = "completed"
        evaluation_decision = initial_outcome.evaluation.decision

    except Exception as error:
        record_failure(
            interaction_id,
            "semantic_evaluation",
            error,
        )

    # 10. Revise once if the evaluator requests revision.

    if (
        evaluation_status == "completed"
        and evaluation_decision == "revise"
        and initial_outcome is not None
    ):
        assessment = initial_outcome.assessment

        revision_prompt = (
            "Improve the following draft answer using the evaluator's "
            "feedback.\n\n"
            f"User question:\n{request.question}\n\n"
            f"Original draft:\n{final_answer}\n\n"
            f"Evaluator feedback:\n{assessment.feedback}\n\n"
            f"Unsupported claims to address:\n"
            f"{assessment.unsupported_claims}\n\n"
            f"Missing information to address:\n"
            f"{assessment.missing_information}\n\n"
            "Produce a better answer that directly addresses the user's "
            "question. Correct unsupported claims, address important "
            "omissions, and acknowledge uncertainty where appropriate. "
            "Do not mention this internal evaluation process."
        )

        revision_request = ModelRequest(
            prompt=revision_prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Count the revision attempt even if the provider fails.

        revision_attempts = 1

        try:
            revision_response = await provider.generate(
                revision_request
            )

            if not revision_response.success:
                raise RuntimeError(
                    revision_response.error
                    or "The local model could not revise the answer."
                )

            if not revision_response.content.strip():
                raise RuntimeError(
                    "The local model returned an empty revision."
                )

            # Evaluate the revised answer before adopting it.

            revision_outcome = (
                await semantic_evaluator.evaluate_response(
                    question=request.question,
                    answer=revision_response.content,
                    provider=provider,
                )
            )

            # Adopt the revision only after its evaluation succeeds.

            final_answer = revision_response.content
            evaluation_result = revision_outcome.model_dump()
            evaluation_decision = revision_outcome.evaluation.decision

        except Exception as error:
            record_failure(
                interaction_id,
                "answer_revision",
                error,
            )

            # Keep the original answer and original evaluation.
            # Never attempt a second revision.

    # 11. Archive only the final assistant answer.

    try:
        conversation_archive.save_message(
            owner=user_id,
            conversation_id=conversation_id,
            interaction_id=interaction_id,
            role="assistant",
            content=final_answer,
            privacy=privacy,
        )

    except Exception as error:
        record_failure(
            interaction_id,
            "archive_assistant_message",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The model generated an answer, but CognitiveAI could not "
                "archive it. The answer was not returned; please retry."
            ),
        )

    # 12. Immediate memory extraction.
    #
    # This existing feature remains separate from background learning.
    # Only private interactions enter personal memory extraction.

    if privacy == "private":
        memory_result = await automatically_remember(
            question=request.question,
            provider=provider,
            interaction_id=interaction_id,
            user_id=user_id,
        )

    else:
        memory_result = {
            "memories_saved": 0,
            "candidates_found": 0,
            "extraction_status": "skipped_public_request",
        }

    # 13. Enqueue private conversation for background learning.
    #
    # Public conversations are deliberately excluded.

    if privacy == "private":
        background_result = enqueue_private_learning(user_id)

    else:
        background_result = {
            "status": "skipped_public_request",
            "discovered": 0,
        }

    # 14. Record successful completion.

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.INTERACTION_COMPLETED,
        source="supervisor_ask",
        status=EventStatus.COMPLETED,
    )

    # 15. Return the final answer and processing status.

    return {
        "supervisor": "CognitiveAI",
        "interaction_id": interaction_id,
        "conversation_id": conversation_id,
        "archive_status": "saved",
        "transparency_url": (
            f"/transparency/interactions/{interaction_id}"
        ),
        "question": request.question,
        "answer": final_answer,
        "privacy": privacy,
        "evaluation": {
            "status": evaluation_status,
            "decision": evaluation_decision,
            "result": evaluation_result,
            "revision_attempts": revision_attempts,
            "max_revision_attempts": 1,
            "method": "local_llm_judge_plus_deterministic_rules",
            "scope": (
                "Heuristic answer-quality assessment; "
                "not independent factual verification."
            ),
        },
        "routing": {
            "task_type": routing.task_type,
            "complexity": routing.complexity,
            "provider": model_response.provider,
            "model": model_response.model,
            "execution_ready": True,
            "reason": (
                "Local-only execution enforced by "
                "the CognitiveAI ask pipeline."
            ),
            "decision_id": routing.decision_id,
        },
        "memory": {
            "memories_retrieved": len(memories),
            "memory_ids": [item.id for item in memories],
            "memory_scopes": [
                item.scope.value for item in memories
            ],
            "memories_saved": memory_result["memories_saved"],
            "candidates_found": memory_result["candidates_found"],
            "extraction_status": memory_result["extraction_status"],
        },
        "background_learning": background_result,
        "plan": plan.model_dump(),
        "execution_status": "completed",
    }
