
import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.supervisor.router import (
    TaskRequest,
    route_task
)
from backend.supervisor.planner import CognitivePlanner
from backend.supervisor.evaluator import CognitiveEvaluator

from backend.models.factory import create_provider
from backend.models.providers import ModelRequest

from backend.memory.manager import MemoryManager

from backend.transparency.events import EventStatus, EventType
from backend.transparency.event_store import InMemoryEventStore
from backend.transparency.recorder import TransparencyRecorder
from backend.transparency.audit import TransparencyAudit

from config.settings import get_settings


app = FastAPI(
    title="CognitiveAI",
    description="Metacognitive Intelligence Platform",
    version="0.4.1"
)

planner = CognitivePlanner()
evaluator = CognitiveEvaluator()

# Single-user development configuration.
# Production must derive this identity from authenticated user sessions.
memory_manager = MemoryManager(
    owner=os.getenv("COGNITIVEAI_OWNER", "local-user")
)

# Transparency services.
# Development only: events disappear when the process restarts.
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
    privacy: str = "private"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=1, le=10000)


# ==========================================
# TRANSPARENCY HELPERS
# ==========================================

def record_failure(
    interaction_id: str,
    stage: str,
    error: Exception
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
            "error_type": type(error).__name__
        }
    )


# ==========================================
# APPLICATION
# ==========================================

@app.get("/")
async def home():
    return {
        "application": "CognitiveAI",
        "status": "running",
        "version": "0.4.1"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
        confidence=request.confidence
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
            "note": "No AI model was executed."
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

    # The owner comes from server configuration.
    # The caller cannot select another user's identity.
    report = transparency_audit.get_interaction(
        user_id=memory_manager.owner,
        interaction_id=interaction_id
    )

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Transparency report not found."
        )

    return report


@app.get("/transparency/events")
async def get_transparency_events(limit: int = 100):

    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500."
        )

    return {
        "events": transparency_audit.list_events(
            user_id=memory_manager.owner,
            limit=limit
        )
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
            detail="Privacy must be private or public."
        )

    interaction_id = str(uuid4())
    user_id = memory_manager.owner

    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.INTERACTION_STARTED,
        source="supervisor_ask",
        status=EventStatus.STARTED,
        details={
            "privacy": privacy
        }
    )

    # 2. Supervisor classification and planning.
    task_request = TaskRequest(
        task=request.question,
        privacy=privacy
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
                "decision_id": routing.decision_id
            }
        )

        plan = planner.create_plan(request.question)

        transparency_recorder.record(
            user_id=user_id,
            interaction_id=interaction_id,
            event_type=EventType.PLAN_CREATED,
            source="supervisor_planner",
            status=EventStatus.COMPLETED
        )

    except ValueError as error:
        record_failure(interaction_id, "classification_or_planning", error)
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # 3. Select the configured local model.
    try:
        settings = get_settings()
        local_model = settings.local_model_for(routing.task_type)
    except Exception as error:
        record_failure(interaction_id, "model_configuration", error)
        raise HTTPException(
            status_code=500,
            detail="Model configuration failed."
        )

    # 4. Retrieve memories under the requested privacy boundary.
    try:
        memories = memory_manager.recall(
            query=request.question,
            privacy=privacy,
            limit=5
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
                ]
            }
        )

    except (ValueError, OSError) as error:
        record_failure(interaction_id, "memory_retrieval", error)
        raise HTTPException(
            status_code=500,
            detail="Memory retrieval failed."
        )

    memory_context = build_memory_context(memories)

    # 5. Build the model prompt.
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
        max_tokens=request.max_tokens
    )

    # 6. Create the provider and record model selection.
    try:
        provider = create_provider(
            name="local",
            model=local_model
        )
    except Exception as error:
        record_failure(interaction_id, "provider_initialization", error)
        raise HTTPException(
            status_code=503,
            detail="Local model provider initialization failed."
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
            "destination": "local"
        }
    )

    # 7. Execute locally.
    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MODEL_EXECUTION_STARTED,
        source="model_provider",
        status=EventStatus.STARTED
    )

    try:
        model_response = await provider.generate(model_request)

    except Exception as error:
        record_failure(interaction_id, "model_execution", error)
        raise HTTPException(
            status_code=503,
            detail="Local model execution failed."
        )

    if not model_response.success:
        error = RuntimeError("Model returned an unsuccessful response.")
        record_failure(interaction_id, "model_execution", error)

        raise HTTPException(
            status_code=503,
            detail=model_response.error or
                   "Local model failed to generate an answer."
        )

    # Record the provider/model reported by the completed execution.
    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.MODEL_EXECUTION_COMPLETED,
        source="model_provider",
        status=EventStatus.COMPLETED,
        details={
            "provider": model_response.provider,
            "model": model_response.model
        }
    )

    # 8. Record successful completion.
    transparency_recorder.record(
        user_id=user_id,
        interaction_id=interaction_id,
        event_type=EventType.INTERACTION_COMPLETED,
        source="supervisor_ask",
        status=EventStatus.COMPLETED
    )

    # 9. Return the answer and transparency reference.
    return {
        "supervisor": "CognitiveAI",
        "interaction_id": interaction_id,
        "transparency_url": (
            f"/transparency/interactions/{interaction_id}"
        ),
        "question": request.question,
        "answer": model_response.content,
        "privacy": privacy,
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
            "decision_id": routing.decision_id
        },
        "memory": {
            "memories_retrieved": len(memories),
            "memory_ids": [item.id for item in memories],
            "memory_scopes": [
                item.scope.value for item in memories
            ]
        },
        "plan": plan.model_dump(),
        "execution_status": "completed"
    }
