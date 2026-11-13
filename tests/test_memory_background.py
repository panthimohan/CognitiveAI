import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.memory.background import BackgroundMemory
from backend.memory.models import MemoryItem, MemoryKind, MemoryScope


class FakeClock:

    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance_days(self, days):
        self.now += timedelta(days=days)


def personal(content="note", importance=0.5, owner="alice",
             kind=MemoryKind.EPISODIC):
    return MemoryItem(
        content=content, scope=MemoryScope.PERSONAL, owner=owner,
        importance=importance, kind=kind
    )


def public(content="fact", importance=0.5, kind=MemoryKind.SEMANTIC):
    return MemoryItem(
        content=content, scope=MemoryScope.PUBLIC,
        importance=importance, kind=kind
    )


def personal_store(tmp_path, owner="alice", **kwargs):
    return BackgroundMemory(
        MemoryScope.PERSONAL, owner=owner, base_dir=tmp_path, **kwargs
    )


def public_store(tmp_path, **kwargs):
    return BackgroundMemory(
        MemoryScope.PUBLIC, base_dir=tmp_path, **kwargs
    )


# ---------- construction and paths ----------

def test_personal_path_is_per_owner(tmp_path):
    store = personal_store(tmp_path, owner="alice")

    assert store.path == tmp_path / "personal" / "alice" / "memory.json"


def test_public_path_is_global(tmp_path):
    assert public_store(tmp_path).path == (
        tmp_path / "global" / "memory.json"
    )


def test_personal_requires_owner(tmp_path):
    with pytest.raises(ValueError):
        BackgroundMemory(MemoryScope.PERSONAL, base_dir=tmp_path)


def test_public_rejects_owner(tmp_path):
    with pytest.raises(ValueError):
        BackgroundMemory(
            MemoryScope.PUBLIC, owner="alice", base_dir=tmp_path
        )


@pytest.mark.parametrize(
    "owner", ["../bob", "a/b", "..", ".hidden", "a b", "x" * 101]
)
def test_unsafe_owner_names_are_rejected(tmp_path, owner):
    with pytest.raises(ValueError):
        personal_store(tmp_path, owner=owner)


def test_email_style_owner_is_allowed(tmp_path):
    personal_store(tmp_path, owner="mpanthi@lakeheadu.ca")


# ---------- privacy boundary ----------

def test_rejects_wrong_scope_and_owner(tmp_path):
    store = personal_store(tmp_path, owner="alice")

    with pytest.raises(ValueError):
        store.add(public())

    with pytest.raises(ValueError):
        store.add(personal(owner="bob"))


def test_users_are_isolated(tmp_path):
    alice = personal_store(tmp_path, owner="alice")
    bob = personal_store(tmp_path, owner="bob")

    alice.add(personal("alice secret", owner="alice"))

    assert len(bob) == 0
    assert bob.search("secret") == []


def test_add_many_is_all_or_nothing(tmp_path):
    store = personal_store(tmp_path, owner="alice")

    with pytest.raises(ValueError):
        store.add_many([personal("ok"), personal("bad", owner="bob")])

    assert len(store) == 0
    assert not store.path.exists()


# ---------- persistence ----------

def test_items_survive_reload(tmp_path):
    store = public_store(tmp_path)
    item = public("Paris is the capital of France", importance=0.8)
    store.add(item)

    reloaded = public_store(tmp_path)

    assert item.id in reloaded
    assert reloaded.get(item.id).content == item.content
    assert reloaded.get(item.id).importance == 0.8


def test_remove_is_persisted(tmp_path):
    store = public_store(tmp_path)
    item = public()
    store.add(item)

    assert store.remove(item.id) is item
    assert store.remove(item.id) is None
    assert len(public_store(tmp_path)) == 0


def test_corrupt_file_raises_clear_error(tmp_path):
    store = public_store(tmp_path)
    store.path.parent.mkdir(parents=True)
    store.path.write_text("not json")

    with pytest.raises(ValueError, match="corrupt"):
        public_store(tmp_path)


def test_file_with_wrong_scope_items_is_rejected(tmp_path):
    other = personal_store(tmp_path, owner="alice")
    other.add(personal("secret"))

    public_path = tmp_path / "global" / "memory.json"
    public_path.parent.mkdir(parents=True)
    public_path.write_text(other.path.read_text())

    with pytest.raises(ValueError):
        public_store(tmp_path)


def test_save_leaves_no_temp_file(tmp_path):
    store = public_store(tmp_path)
    store.add(public())

    assert [p.name for p in store.path.parent.iterdir()] == ["memory.json"]


# ---------- search ----------

def test_search_ranks_by_relevance(tmp_path):
    store = public_store(tmp_path)
    best = public("python web server with fastapi")
    partial = public("python data analysis")
    unrelated = public("cooking pasta")
    store.add_many([best, partial, unrelated])

    results = store.search("python fastapi server")

    assert results[0] is best
    assert partial in results
    assert unrelated not in results


def test_search_no_match_returns_empty(tmp_path):
    store = public_store(tmp_path)
    store.add(public("python"))

    assert store.search("zebra") == []
    assert store.search("") == []


def test_search_ignores_stopwords_but_falls_back(tmp_path):
    store = public_store(tmp_path)
    item = public("the cat sat")
    store.add(item)

    assert store.search("the") == [item]


def test_search_filters_by_kind(tmp_path):
    store = public_store(tmp_path)
    fact = public("python is a language", kind=MemoryKind.SEMANTIC)
    how = public("python install steps", kind=MemoryKind.PROCEDURAL)
    store.add_many([fact, how])

    assert store.search("python", kind=MemoryKind.PROCEDURAL) == [how]


def test_search_respects_limit(tmp_path):
    store = public_store(tmp_path)
    store.add_many([public(f"python tip {i}") for i in range(10)])

    assert len(store.search("python", limit=3)) == 3


def test_search_strengthens_recalled_items(tmp_path):
    store = public_store(tmp_path)
    item = public("python")
    store.add(item)

    store.search("python")

    assert item.access_count == 1
    assert public_store(tmp_path).get(item.id).access_count == 2


def test_search_touch_false_leaves_items_alone(tmp_path):
    store = public_store(tmp_path)
    item = public("python")
    store.add(item)

    store.search("python", touch=False)

    assert item.access_count == 0


# ---------- decay ----------

def test_strength_halves_after_one_half_life(tmp_path):
    clock = FakeClock()
    store = public_store(tmp_path, half_life_days=30, clock=clock)
    item = public(importance=0.4)
    item.last_accessed_at = clock()
    store.add(item)

    fresh = store.strength(item)
    clock.advance_days(30)

    assert store.strength(item) == pytest.approx(fresh / 2)


def test_use_makes_memory_stronger(tmp_path):
    store = public_store(tmp_path)
    used = public("a", importance=0.3)
    unused = public("b", importance=0.3)
    store.add_many([used, unused])

    for _ in range(3):
        used.touch()

    assert store.strength(used) > store.strength(unused)


def test_forget_weak_removes_faded_keeps_fresh(tmp_path):
    clock = FakeClock()
    store = public_store(tmp_path, half_life_days=30, clock=clock)

    old = public("old", importance=0.2)
    old.last_accessed_at = clock()
    store.add(old)

    clock.advance_days(300)

    fresh = public("fresh", importance=0.2)
    fresh.last_accessed_at = clock()
    store.add(fresh)

    removed = store.forget_weak(threshold=0.05)

    assert removed == [old]
    assert fresh.id in store
    assert old.id not in public_store(tmp_path, clock=clock)


def test_get_missing_returns_none(tmp_path):
    assert public_store(tmp_path).get("missing") is None
