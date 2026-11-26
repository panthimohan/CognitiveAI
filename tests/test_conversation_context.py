from types import SimpleNamespace

from backend.main import build_conversation_context


def msg(interaction_id, role, content):
    return SimpleNamespace(interaction_id=interaction_id, role=role, content=content)


def test_previous_turns_are_in_chronological_order_and_current_turn_is_excluded():
    messages = [
        msg("one", "user", "I am planning a move."),
        msg("one", "assistant", "Let's compare options."),
        msg("two", "user", "What about the cost?"),
    ]
    result = build_conversation_context(messages, current_interaction_id="two")
    assert result.index("User: I am planning") < result.index("CognitiveAI: Let's compare")
    assert "What about the cost?" not in result


def test_current_interaction_assistant_message_is_excluded_too():
    messages = [msg("old", "user", "Earlier"), msg("now", "user", "Current"), msg("now", "assistant", "Answer")]
    result = build_conversation_context(messages, current_interaction_id="now")
    assert "Earlier" in result
    assert "Current" not in result
    assert "Answer" not in result


def test_history_is_limited_to_latest_messages():
    messages = [msg(str(i), "user", f"Question {i}") for i in range(40)]
    result = build_conversation_context(messages, current_interaction_id="none", max_messages=5)
    assert "Question 39" in result and "Question 35" in result
    assert "Question 34" not in result


def test_empty_history_has_fallback():
    assert build_conversation_context([], current_interaction_id="new") == "This is the beginning of the conversation."
