from datetime import datetime, timedelta, timezone

import pytest

from backend.memory.models import MemoryItem, MemoryKind, MemoryScope
from backend.memory.working import ForegroundMemory


class FakeClock:

    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def personal(content="note", importance=0.5, owner="alice"):
    return MemoryItem(
        content=content,
        scope=MemoryScope.PERSONAL,
        owner=owner,
        importance=importance
    )


def public(content="fact", importance=0.5):
    return MemoryItem(
        content=content,
        scope=MemoryScope.PUBLIC,
        kind=MemoryKind.SEMANTIC,
        importance=importance
    )


# ---------- MemoryItem ----------

def test_personal_item_requires_owner():
    with pytest.raises(ValueError):
        MemoryItem(content="x", scope=MemoryScope.PERSONAL)


def test_public_item_rejects_owner():
    with pytest.raises(ValueError):
        MemoryItem(content="x", scope=MemoryScope.PUBLIC, owner="alice")


def test_importance_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        personal(importance=1.5)


def test_touch_updates_access_info():
    item = personal()
    item.touch()

    assert item.access_count == 1


# ---------- ForegroundMemory: construction ----------

def test_personal_foreground_requires_owner():
    with pytest.raises(ValueError):
        ForegroundMemory(MemoryScope.PERSONAL)


def test_public_foreground_rejects_owner():
    with pytest.raises(ValueError):
        ForegroundMemory(MemoryScope.PUBLIC, owner="alice")


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ForegroundMemory(MemoryScope.PUBLIC, capacity=0)


# ---------- ForegroundMemory: privacy boundary ----------

def test_rejects_personal_item_in_public_memory():
    memory = ForegroundMemory(MemoryScope.PUBLIC)

    with pytest.raises(ValueError):
        memory.add(personal())


def test_rejects_public_item_in_personal_memory():
    memory = ForegroundMemory(MemoryScope.PERSONAL, owner="alice")

    with pytest.raises(ValueError):
        memory.add(public())


def test_rejects_other_users_item():
    memory = ForegroundMemory(MemoryScope.PERSONAL, owner="alice")

    with pytest.raises(ValueError):
        memory.add(personal(owner="bob"))


# ---------- ForegroundMemory: capacity and eviction ----------

def test_add_within_capacity_evicts_nothing():
    memory = ForegroundMemory(MemoryScope.PUBLIC, capacity=3)

    assert memory.add(public("a")) == []
    assert len(memory) == 1


def test_evicts_least_important_when_full():
    clock = FakeClock()
    memory = ForegroundMemory(
        MemoryScope.PUBLIC, capacity=2, clock=clock
    )

    low = public("low", importance=0.1)
    high = public("high", importance=0.9)
    new = public("new", importance=0.5)

    memory.add(low)
    memory.add(high)
    evicted = memory.add(new)

    assert evicted == [low]
    assert low.id not in memory
    assert high.id in memory
    assert new.id in memory
    assert len(memory) == 2


def test_recency_breaks_importance_ties():
    clock = FakeClock()
    memory = ForegroundMemory(
        MemoryScope.PUBLIC, capacity=2, clock=clock
    )

    old = public("old", importance=0.5)
    memory.add(old)

    clock.advance(600)

    recent = public("recent", importance=0.5)
    memory.add(recent)

    evicted = memory.add(public("newest", importance=0.5))

    assert evicted == [old]


def test_new_item_is_never_evicted_immediately():
    memory = ForegroundMemory(MemoryScope.PUBLIC, capacity=1)

    memory.add(public("important", importance=1.0))
    new = public("unimportant", importance=0.0)
    evicted = memory.add(new)

    assert new.id in memory
    assert len(evicted) == 1
    assert len(memory) == 1


def test_adding_existing_item_counts_as_access_not_duplicate():
    memory = ForegroundMemory(MemoryScope.PUBLIC)
    item = public()

    memory.add(item)
    memory.add(item)

    assert len(memory) == 1
    assert item.access_count == 1


# ---------- ForegroundMemory: reading ----------

def test_get_returns_item_and_counts_access():
    memory = ForegroundMemory(MemoryScope.PUBLIC)
    item = public()
    memory.add(item)

    assert memory.get(item.id) is item
    assert item.access_count == 1


def test_get_missing_returns_none():
    memory = ForegroundMemory(MemoryScope.PUBLIC)

    assert memory.get("missing") is None


def test_items_sorted_by_salience():
    memory = ForegroundMemory(MemoryScope.PUBLIC)

    low = public("low", importance=0.1)
    high = public("high", importance=0.9)
    memory.add(low)
    memory.add(high)

    assert memory.items() == [high, low]


def test_recency_decays_over_time():
    clock = FakeClock()
    memory = ForegroundMemory(
        MemoryScope.PUBLIC, half_life_seconds=300, clock=clock
    )
    item = public(importance=0.0)
    memory.add(item)

    fresh = memory.salience(item)
    clock.advance(300)
    later = memory.salience(item)

    assert later == pytest.approx(fresh / 2)


# ---------- ForegroundMemory: removal ----------

def test_remove_and_clear():
    memory = ForegroundMemory(MemoryScope.PUBLIC)
    a, b = public("a"), public("b")
    memory.add(a)
    memory.add(b)

    assert memory.remove(a.id) is a
    assert memory.remove(a.id) is None
    assert memory.clear() == [b]
    assert len(memory) == 0


# ---------- ForegroundMemory: promotion ----------

def test_promotion_candidates_by_importance_or_use():
    memory = ForegroundMemory(MemoryScope.PUBLIC)

    important = public("important", importance=0.9)
    used = public("used", importance=0.1)
    ignored = public("ignored", importance=0.1)

    for item in (important, used, ignored):
        memory.add(item)

    memory.get(used.id)
    memory.get(used.id)

    candidates = memory.promotion_candidates()

    assert important in candidates
    assert used in candidates
    assert ignored not in candidates
