
from enum import Enum


class EventType(str, Enum):
    INTERACTION_STARTED = "interaction_started"
    TASK_CLASSIFIED = "task_classified"
    PLAN_CREATED = "plan_created"

    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_EXTRACTION_STARTED = "memory_extraction_started"
    MEMORY_EXTRACTION_COMPLETED = "memory_extraction_completed"
    MEMORY_SAVED = "memory_saved"
    MEMORY_EXTRACTION_FAILED = "memory_extraction_failed"

    MODEL_SELECTED = "model_selected"
    MODEL_EXECUTION_STARTED = "model_execution_started"
    MODEL_EXECUTION_COMPLETED = "model_execution_completed"

    INTERACTION_COMPLETED = "interaction_completed"
    INTERACTION_FAILED = "interaction_failed"


class EventStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    RECORDED = "recorded"
