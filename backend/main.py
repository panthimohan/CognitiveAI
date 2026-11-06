from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.supervisor.router import (
    TaskRequest,
    route_task
)

from backend.supervisor.planner import CognitivePlanner

from backend.supervisor.evaluator import CognitiveEvaluator


app = FastAPI(
    title="CognitiveAI",
    description="Metacognitive Intelligence Platform",
    version="0.3.0"
)

planner = CognitivePlanner()
evaluator = CognitiveEvaluator()


@app.get("/")
async def home():
    return {
        "application": "CognitiveAI",
        "status": "running",
        "version": "0.3.0"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/supervisor/route")
async def supervisor_route(request: TaskRequest):
    try:
        return route_task(request).model_dump()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/supervisor/plan")
async def supervisor_plan(request: TaskRequest):
    return planner.create_plan(request.task).model_dump()


class EvaluationInput(BaseModel):
    completeness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


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
            "note": "No external AI model has been executed."
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))