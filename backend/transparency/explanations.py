
"""
CognitiveAI - User-Readable Transparency Explanations

Converts recorded system events into understandable explanations.

This module explains observable system decisions and recorded evidence.
It does not expose private model chain-of-thought.
"""

from backend.transparency.schemas import TransparencyEvent


def build_interaction_explanation(
    events: list[TransparencyEvent]
) -> dict:

    ordered = sorted(events, key=lambda event: event.sequence)

    if not ordered:
        return {
            "available": False,
            "message": "No recorded events were found."
        }

    # --------------------------------------
    # Collect recorded information
    # --------------------------------------

    interaction_started = None
    task_information = None
    model_information = None
    execution_information = None
    memory_information = None
    failure_information = None

    for event in ordered:

        event_type = event.event_type.value

        if event_type == "interaction_started":
            interaction_started = event.details

        elif event_type == "task_classified":
            task_information = event.details

        elif event_type == "model_selected":
            model_information = event.details

        elif event_type == "model_execution_completed":
            execution_information = event.details

        elif event_type == "memory_retrieved":
            memory_information = event.details

        elif event_type == "interaction_failed":
            failure_information = event.details

    # --------------------------------------
    # Determine interaction status
    # --------------------------------------

    completed = any(
        event.event_type.value == "interaction_completed"
        for event in ordered
    )

    failed = any(
        event.event_type.value == "interaction_failed"
        for event in ordered
    )

    status = (
        "failed" if failed
        else "completed" if completed
        else "incomplete"
    )

    # --------------------------------------
    # Explain task classification
    # --------------------------------------

    task_type = (
        task_information.get("task_type")
        if task_information else None
    )

    complexity = (
        task_information.get("complexity")
        if task_information else None
    )

    task_explanation = {
        "task_type": task_type,
        "complexity": complexity,
        "explanation": (
            f"The system classified this request as a "
            f"{task_type} task."
            if task_type
            else "Task classification was not recorded."
        )
    }

    # --------------------------------------
    # Explain model selection
    # --------------------------------------

    selected_model = (
        model_information.get("model")
        if model_information else None
    )

    provider = (
        model_information.get("provider")
        if model_information else None
    )

    destination = (
        model_information.get("destination")
        if model_information else None
    )

    model_explanation = {
        "provider": provider,
        "model": selected_model,
        "destination": destination,
        "reason": (
            f"The request was classified as {task_type}. "
            f"The configured local model for this task "
            f"was selected."
            if selected_model and task_type
            else "Model selection information is incomplete."
        )
    }

    # --------------------------------------
    # Explain memory usage
    # --------------------------------------

    memory_count = (
        memory_information.get("count", 0)
        if memory_information else 0
    )

    memory_scopes = (
        memory_information.get("memory_scopes", [])
        if memory_information else []
    )

    memory_explanation = {
        "count": memory_count,
        "memory_ids": (
            memory_information.get("memory_ids", [])
            if memory_information else []
        ),
        "memory_scopes": memory_scopes,
        "memory_kinds": (
            memory_information.get("memory_kinds", [])
            if memory_information else []
        ),
        "explanation": (
            f"{memory_count} stored memories were retrieved "
            f"and made available as context."
            if memory_count > 0
            else "No matching stored memories were retrieved. "
                 "The response was generated without retrieved "
                 "long-term memory."
            if memory_information
            else "Memory retrieval information was not recorded."
        )
    }

    # --------------------------------------
    # Explain privacy and data destination
    # --------------------------------------

    privacy = (
        interaction_started.get("privacy")
        if interaction_started else None
    )

    privacy_explanation = {
        "requested_privacy": privacy,
        "execution_destination": destination,
        "explanation": (
            "The request was processed using the local model "
            "provider. The recorded execution destination was local."
            if destination == "local"
            else "The execution destination was not confirmed "
                 "as local by the recorded events."
        )
    }

    # --------------------------------------
    # Execution result
    # --------------------------------------

    execution_explanation = {
        "status": (
            "completed" if execution_information
            else "failed" if failed
            else "not_completed"
        ),
        "provider": (
            execution_information.get("provider")
            if execution_information else None
        ),
        "model": (
            execution_information.get("model")
            if execution_information else None
        )
    }

    # --------------------------------------
    # User-readable decision summary
    # --------------------------------------

    summary_parts = []

    if task_type:
        summary_parts.append(
            f"The request was classified as {task_type}."
        )

    if selected_model:
        summary_parts.append(
            f"{selected_model} was selected."
        )

    if memory_information:
        summary_parts.append(
            f"{memory_count} stored memories were retrieved."
        )

    if completed:
        summary_parts.append(
            "The model completed execution and the interaction "
            "finished successfully."
        )
    elif failed:
        summary_parts.append(
            "The interaction encountered an error."
        )
    else:
        summary_parts.append(
            "The interaction has not yet completed."
        )

    # --------------------------------------
    # Timeline
    # --------------------------------------

    timeline = [
        {
            "sequence": event.sequence,
            "event_type": event.event_type.value,
            "source": event.source,
            "status": event.status.value,
            "timestamp": event.timestamp.isoformat()
        }
        for event in ordered
    ]

    # --------------------------------------
    # Transparency limitations
    # --------------------------------------

    limitations = [
        "This report explains recorded system events and decisions.",
        "It does not expose private model chain-of-thought.",
        "The report cannot establish that the generated answer "
        "is factually correct.",
        "The development event store is temporary and loses "
        "records when the application restarts."
    ]

    if failure_information:
        limitations.append(
            "Failure information is limited to the recorded "
            "failure stage and error type."
        )

    return {
        "available": True,
        "interaction_id": ordered[0].interaction_id,
        "status": status,
        "event_count": len(ordered),

        "summary": " ".join(summary_parts),

        "privacy": privacy_explanation,
        "task": task_explanation,
        "model": model_explanation,
        "memory": memory_explanation,
        "execution": execution_explanation,

        "failure": failure_information,

        "timeline": timeline,

        "limitations": limitations
    }
