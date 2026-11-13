from datetime import datetime, timedelta, timezone

import pytest

from backend.memory.manager import MemoryManager
from backend.memory.models import MemoryKind, MemoryScope


class FakeClock:

    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance_days(self, days):
        self.now += timedelta(days=days)


def manager(tmp_path, owner="alice", **kwargs):
    return MemoryManager(owner, base_dir=tmp_path, **kwargs)


# ---------- remember ----------

def test_remember_defaults_to_personal_foreground(tmp_path):
    m = manager(tmp_path)
    item = m.remember("my favourite colour is green")

    assert item.scope == MemoryScope.PERSONAL
    assert item.owner == "alice"
    assert item.id in m.personal_fg
    assert item.id not in m.personal_bg


def test_remember_public_has_no_owner(tmp_path):
    m = manager(tmp_path)
    item = m.remember("water boils at 100C", scope=MemoryScope.PUBLIC)

    assert item.owner is None
    assert item.id in m.public_fg


def test_eviction_moves_items_to_background(tmp_path):
    m = manager(tmp_path, foreground_capacity=2)

    items = [m.remember(f"note {i}", importance=0.5) for i in range(3)]

    in_fg = [i for i in items if i.id in m.personal_fg]
    in_bg = [i for i in items if i.id in m.personal_bg]

    assert len(in_fg) == 2
    assert len(in_bg) == 1
    assert {i.id for i in in_fg + in_bg} == {i.id for i in items}


# ---------- recall ----------

def test_recall_finds_foreground_items(tmp_path):
    m = manager(tmp_path)
    item = m.remember("python is my favourite language")

    assert m.recall("python") == [item]


def test_recall_pulls_background_item_into_foreground(tmp_path):
    first = manager(tmp_path)
    original = first.remember("project deadline is friday", importance=0.9)
    first.consolidate()

    second = manager(tmp_path)          # fresh session: empty foreground
    assert len(second.personal_fg) == 0

    results = second.recall("deadline")

    assert [r.content for r in results] == [original.content]
    assert original.id in second.personal_fg


def test_recall_does_not_duplicate_items_in_both_layers(tmp_path):
    m = manager(tmp_path)
    m.remember("python tips", importance=0.9)
    m.consolidate()                      # now in fg AND bg

    assert len(m.recall("python")) == 1


def test_recall_respects_limit_and_kind(tmp_path):
    m = manager(tmp_path)

    for i in range(4):
        m.remember(f"python fact {i}", kind=MemoryKind.SEMANTIC)
    m.remember("python install steps", kind=MemoryKind.PROCEDURAL)

    assert len(m.recall("python", limit=2)) == 2

    procedural = m.recall("python", kind=MemoryKind.PROCEDURAL)
    assert [i.content for i in procedural] == ["python install steps"]


def test_recall_no_match_returns_empty(tmp_path):
    m = manager(tmp_path)
    m.remember("python")

    assert m.recall("zebra") == []


def test_recall_rejects_invalid_privacy(tmp_path):
    with pytest.raises(ValueError):
        manager(tmp_path).recall("x", privacy="secret")


# ---------- privacy rules ----------

def test_public_recall_never_returns_personal_memory(tmp_path):
    m = manager(tmp_path, foreground_capacity=1)

    m.remember("my bank pin is 1234")          # personal, later evicted
    m.remember("my bank card is blue")         # personal, in foreground
    m.remember("a bank holds money", scope=MemoryScope.PUBLIC)

    results = m.recall("bank", privacy="public")

    assert [r.content for r in results] == ["a bank holds money"]
    assert all(r.scope == MemoryScope.PUBLIC for r in results)


def test_private_recall_includes_personal_and_public(tmp_path):
    m = manager(tmp_path)
    m.remember("my python notes")
    m.remember("python is a language", scope=MemoryScope.PUBLIC)

    scopes = {r.scope for r in m.recall("python", privacy="private")}

    assert scopes == {MemoryScope.PERSONAL, MemoryScope.PUBLIC}


def test_users_cannot_recall_each_others_memories(tmp_path):
    alice = manager(tmp_path, owner="alice")
    alice.remember("alice secret plan", importance=0.9)
    alice.consolidate()

    bob = manager(tmp_path, owner="bob")

    assert bob.recall("secret", privacy="private") == []


def test_public_memory_is_shared_between_users(tmp_path):
    alice = manager(tmp_path, owner="alice")
    alice.remember(
        "the sky is blue", scope=MemoryScope.PUBLIC, importance=0.9
    )
    alice.consolidate()

    bob = manager(tmp_path, owner="bob")

    assert [r.content for r in bob.recall("sky", privacy="public")] == [
        "the sky is blue"
    ]


# ---------- share ----------

def test_share_creates_ownerless_public_copy(tmp_path):
    m = manager(tmp_path)
    original = m.remember("tip: use virtual environments")

    shared = m.share(original.id)

    assert shared.id != original.id
    assert shared.scope == MemoryScope.PUBLIC
    assert shared.owner is None
    assert shared.content == original.content
    assert original.id in m.personal_fg           # original stays


def test_nothing_is_public_until_shared(tmp_path):
    m = manager(tmp_path)
    m.remember("private diary entry")

    assert m.recall("diary", privacy="public") == []


def test_share_finds_items_that_are_only_in_background(tmp_path):
    m = manager(tmp_path, foreground_capacity=1)
    first = m.remember("older note about pandas", importance=0.1)
    m.remember("newer note", importance=0.9)

    assert first.id in m.personal_bg and first.id not in m.personal_fg

    shared = m.share(first.id)

    assert shared.content == first.content


def test_share_unknown_id_raises(tmp_path):
    with pytest.raises(KeyError):
        manager(tmp_path).share("missing")


# ---------- consolidate ----------

def test_consolidate_persists_important_items(tmp_path):
    m = manager(tmp_path)
    important = m.remember("keep this", importance=0.9)
    trivial = m.remember("skip this", importance=0.1)

    report = m.consolidate()

    assert report.promoted == 1
    assert important.id in m.personal_bg
    assert trivial.id not in m.personal_bg
    assert important.id in manager(tmp_path).personal_bg


def test_consolidate_promotes_frequently_recalled_items(tmp_path):
    m = manager(tmp_path)
    item = m.remember("rarely important but often used", importance=0.1)

    m.recall("often")
    m.recall("often")
    m.consolidate()

    assert item.id in m.personal_bg


def test_consolidate_forgets_faded_memories(tmp_path):
    clock = FakeClock()
    m = manager(tmp_path, half_life_days=30, clock=clock)

    old = m.remember("ancient note", importance=0.7)
    m.consolidate()
    assert old.id in m.personal_bg

    m.personal_fg.clear()
    clock.advance_days(600)

    report = m.consolidate()

    assert report.forgotten == 1
    assert old.id not in m.personal_bg


# ---------- forget ----------

def test_forget_removes_from_both_layers(tmp_path):
    m = manager(tmp_path)
    item = m.remember("delete me", importance=0.9)
    m.consolidate()

    assert m.forget(item.id) is True
    assert item.id not in m.personal_fg
    assert item.id not in manager(tmp_path).personal_bg
    assert m.recall("delete") == []


def test_forget_unknown_id_returns_false(tmp_path):
    assert manager(tmp_path).forget("missing") is False
