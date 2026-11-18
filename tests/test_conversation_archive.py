
from backend.conversations.archive import ConversationArchive


def test_save_and_retrieve_conversation(tmp_path):
    archive = ConversationArchive(
        tmp_path / "conversations.sqlite3"
    )

    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="I am learning Python.",
    )

    archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="assistant",
        content="Python is a programming language.",
    )

    messages = archive.get_conversation(
        owner="test-user",
        conversation_id="conversation-1",
    )

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_messages_survive_archive_restart(tmp_path):
    database = tmp_path / "conversations.sqlite3"

    first_archive = ConversationArchive(database)

    first_archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="Remember this conversation.",
    )

    second_archive = ConversationArchive(database)

    messages = second_archive.get_conversation(
        owner="test-user",
        conversation_id="conversation-1",
    )

    assert len(messages) == 1
    assert messages[0].content == "Remember this conversation."


def test_duplicate_message_is_not_saved_twice(tmp_path):
    archive = ConversationArchive(
        tmp_path / "conversations.sqlite3"
    )

    first = archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="Hello.",
    )

    second = archive.save_message(
        owner="test-user",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="Hello.",
    )

    assert first.id == second.id

    assert archive.count_messages(
        owner="test-user"
    ) == 1


def test_private_and_public_history_are_separate(tmp_path):
    archive = ConversationArchive(
        tmp_path / "conversations.sqlite3"
    )

    archive.save_message(
        owner="test-user",
        conversation_id="private-1",
        interaction_id="interaction-1",
        role="user",
        content="Private message.",
        privacy="private",
    )

    archive.save_message(
        owner="test-user",
        conversation_id="public-1",
        interaction_id="interaction-2",
        role="user",
        content="Public message.",
        privacy="public",
    )

    private_messages = archive.list_messages(
        owner="test-user",
        privacy="private",
    )

    public_messages = archive.list_messages(
        owner="test-user",
        privacy="public",
    )

    assert len(private_messages) == 1
    assert len(public_messages) == 1
    assert private_messages[0].content == "Private message."
    assert public_messages[0].content == "Public message."


def test_different_owners_are_isolated(tmp_path):
    archive = ConversationArchive(
        tmp_path / "conversations.sqlite3"
    )

    archive.save_message(
        owner="user-a",
        conversation_id="conversation-1",
        interaction_id="interaction-1",
        role="user",
        content="User A message.",
    )

    assert archive.count_messages(owner="user-a") == 1
    assert archive.count_messages(owner="user-b") == 0
