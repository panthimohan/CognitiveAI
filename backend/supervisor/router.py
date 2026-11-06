"""
CognitiveAI - Metacognitive Supervisor Router

Classifies incoming tasks and selects an appropriate
AI provider based on task type, capabilities and privacy.
"""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


# ==========================================
# 1. REQUEST MODEL
# ==========================================

class TaskRequest(BaseModel):
    task: str = Field(min_length=1)
    privacy: str = "private"
    preferred_provider: str | None = None


# ==========================================
# 2. ROUTING DECISION MODEL
# ==========================================

class RoutingDecision(BaseModel):
    decision_id: str
    task_type: str
    complexity: str
    provider: str
    model: str
    capabilities: list[str]
    reason: str
    execution_ready: bool
    privacy: str
    created_at: str


# ==========================================
# 3. TASK CLASSIFICATION
# ==========================================

def classify_task(task: str) -> str:

    text = task.lower()

    categories = {
        "coding": [
            "code", "python", "program",
            "debug", "software", "develop",
            "build", "implement"
        ],

        "research": [
            "research", "latest", "news",
            "current", "search", "investigate",
            "information", "discover"
        ],

        "reasoning": [
            "analyze", "reason", "compare",
            "evaluate", "complex", "architecture",
            "design", "system", "strategy",
            "intelligence", "learning", "memory",
            "privacy", "optimize"
        ]
    }

    scores = {
        category: sum(
            1 for keyword in keywords if keyword in text
        )
        for category, keywords in categories.items()
    }

    best_category = max(scores, key=scores.get)

    if scores[best_category] == 0:
        return "conversation"

    return best_category


# ==========================================
# 4. COMPLEXITY ASSESSMENT
# ==========================================

def assess_complexity(task: str) -> str:

    text = task.lower()

    complexity_indicators = [
        "architecture",
        "multiple models",
        "continuous learning",
        "long-term memory",
        "privacy-preserving",
        "metacognitive",
        "integration",
        "distributed",
        "optimization",
        "reasoning",
        "learning system"
    ]

    score = sum(
        1 for indicator in complexity_indicators
        if indicator in text
    )

    if score >= 3:
        return "high"

    if score >= 1:
        return "medium"

    if len(text.split()) > 40:
        return "high"

    if len(text.split()) > 20:
        return "medium"

    return "low"


# ==========================================
# 5. CAPABILITY IDENTIFICATION
# ==========================================

def identify_capabilities(task: str) -> list[str]:

    text = task.lower()

    capability_keywords = {
        "reasoning": [
            "analyze", "reason", "evaluate",
            "compare", "complex", "strategy"
        ],

        "architecture": [
            "architecture", "design",
            "system", "infrastructure"
        ],

        "coding": [
            "code", "python", "program",
            "software", "develop", "implement"
        ],

        "research": [
            "research", "investigate",
            "latest", "discover", "search"
        ],

        "memory": [
            "memory", "remember",
            "long-term", "knowledge"
        ],

        "learning": [
            "learning", "learn",
            "continuous", "improve", "adapt"
        ],

        "privacy": [
            "privacy", "private",
            "security", "confidential"
        ]
    }

    capabilities = []

    for capability, keywords in capability_keywords.items():

        if any(keyword in text for keyword in keywords):
            capabilities.append(capability)

    if not capabilities:
        capabilities.append("conversation")

    return capabilities


# ==========================================
# 6. PROVIDER SELECTION
# ==========================================

def select_provider(
    task_type: str,
    preferred_provider: str | None = None
):

    providers = {
        "coding": ("anthropic", "Claude"),
        "research": ("google", "Gemini"),
        "reasoning": ("openai", "GPT"),
        "conversation": ("openai", "GPT")
    }

    if preferred_provider:

        allowed_providers = {
            "openai",
            "anthropic",
            "google",
            "local"
        }

        provider = preferred_provider.lower()

        if provider not in allowed_providers:
            raise ValueError(
                f"Unsupported provider: {provider}"
            )

        models = {
            "openai": "GPT",
            "anthropic": "Claude",
            "google": "Gemini",
            "local": "Local model"
        }

        return provider, models[provider]

    return providers[task_type]


# ==========================================
# 7. MAIN SUPERVISOR ROUTER
# ==========================================

def route_task(request: TaskRequest) -> RoutingDecision:

    privacy = request.privacy.lower()

    if privacy not in {"private", "public"}:
        raise ValueError(
            "Privacy must be private or public"
        )

    task_type = classify_task(request.task)

    complexity = assess_complexity(request.task)

    capabilities = identify_capabilities(request.task)

    if privacy == "private":

        provider = "local"
        model = "Local model (not connected)"

        reason = (
            "Private task: external transmission disabled. "
            "Local execution is required."
        )

    else:

        provider, model = select_provider(
            task_type,
            request.preferred_provider
        )

        reason = (
            f"Selected {provider} for {task_type} "
            f"with {complexity} complexity."
        )

    return RoutingDecision(
        decision_id=str(uuid4()),
        task_type=task_type,
        complexity=complexity,
        provider=provider,
        model=model,
        capabilities=capabilities,
        reason=reason,
        execution_ready=False,
        privacy=privacy,
        created_at=datetime.now(timezone.utc).isoformat()
    )