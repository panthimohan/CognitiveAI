"""
CognitiveAI - Adaptive Cognitive Planner

Creates task-specific plans using shared complexity
assessment from the metacognitive router.
"""

from pydantic import BaseModel

from backend.supervisor.router import assess_complexity


# ==========================================
# 1. DATA MODELS
# ==========================================

class SubTask(BaseModel):
    task_id: int
    description: str
    objective: str
    priority: int
    required_capability: str


class TaskPlan(BaseModel):
    original_task: str
    complexity: str
    subtasks: list[SubTask]
    total_steps: int


# ==========================================
# 2. COGNITIVE PLANNER
# ==========================================

class CognitivePlanner:

    def create_plan(self, task: str) -> TaskPlan:

        text = task.lower()

        complexity = assess_complexity(task)

        subtasks = []

        # ----------------------------------
        # STEP 1: Understand user intent
        # ----------------------------------

        subtasks.append({
            "description": "Understand the request",
            "objective": "Identify intent, constraints and expected outcomes",
            "capability": "reasoning"
        })

        # ----------------------------------
        # STEP 2: Architecture planning
        # ----------------------------------

        if any(word in text for word in [
            "architecture", "system", "design",
            "application", "platform"
        ]):
            subtasks.append({
                "description": "Design system architecture",
                "objective": "Identify components, interfaces and dependencies",
                "capability": "architecture"
            })

        # ----------------------------------
        # STEP 3: Privacy planning
        # ----------------------------------

        if any(word in text for word in [
            "privacy", "private", "security",
            "confidential", "data protection"
        ]):
            subtasks.append({
                "description": "Design privacy controls",
                "objective": "Define consent, isolation and data protection requirements",
                "capability": "privacy"
            })

        # ----------------------------------
        # STEP 4: Memory planning
        # ----------------------------------

        if any(word in text for word in [
            "memory", "remember", "knowledge",
            "long-term", "working memory"
        ]):
            subtasks.append({
                "description": "Design cognitive memory",
                "objective": "Define working, episodic, semantic and procedural memory",
                "capability": "memory"
            })

        # ----------------------------------
        # STEP 5: Learning planning
        # ----------------------------------

        if any(word in text for word in [
            "learning", "learn", "continuous",
            "improve", "adapt", "training"
        ]):
            subtasks.append({
                "description": "Design learning mechanisms",
                "objective": "Define feedback, evaluation and knowledge consolidation",
                "capability": "learning"
            })

        # ----------------------------------
        # STEP 6: Multi-model planning
        # ----------------------------------

        if any(word in text for word in [
            "multiple models", "multi-model",
            "gpt", "claude", "gemini", "llama",
            "ai models", "model routing"
        ]):
            subtasks.append({
                "description": "Design model orchestration",
                "objective": "Define provider integration and capability-based routing",
                "capability": "model_orchestration"
            })

        # ----------------------------------
        # STEP 7: Coding tasks
        # ----------------------------------

        if any(word in text for word in [
            "code", "python", "program",
            "implement", "develop", "build"
        ]):
            subtasks.append({
                "description": "Develop implementation strategy",
                "objective": "Identify implementation steps and technical requirements",
                "capability": "coding"
            })

        # ----------------------------------
        # STEP 8: Evaluation
        # ----------------------------------

        subtasks.append({
            "description": "Evaluate proposed solution",
            "objective": "Check correctness, completeness and potential risks",
            "capability": "evaluation"
        })

        # ----------------------------------
        # STEP 9: Final integration
        # ----------------------------------

        if complexity in ["medium", "high"]:
            subtasks.append({
                "description": "Integrate and optimize",
                "objective": "Check component compatibility and system-wide consistency",
                "capability": "integration"
            })

        # ----------------------------------
        # STEP 10: Build structured plan
        # ----------------------------------

        structured_tasks = []

        for index, item in enumerate(subtasks, start=1):

            structured_tasks.append(
                SubTask(
                    task_id=index,
                    description=item["description"],
                    objective=item["objective"],
                    priority=index,
                    required_capability=item["capability"]
                )
            )

        return TaskPlan(
            original_task=task,
            complexity=complexity,
            subtasks=structured_tasks,
            total_steps=len(structured_tasks)
        )