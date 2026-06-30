"""Tests for short-term and long-term memory."""
from __future__ import annotations

from open_agent.memory.long_term import LongTermMemory, MemoryEntry
from open_agent.memory.short_term import ShortTermMemory
from open_agent.models.base import Message


def test_short_term_add_and_get_history_preserves_order():
    mem = ShortTermMemory()
    m1 = Message(role="user", content="hi")
    m2 = Message(role="assistant", content="hello")
    mem.add(m1)
    mem.add(m2)

    history = mem.get_history()
    assert len(history) == 2
    assert history[0] is m1
    assert history[1] is m2


def test_short_term_max_messages_evicts_oldest():
    mem = ShortTermMemory(max_messages=3)
    for i in range(5):
        mem.add(Message(role="user", content=f"msg {i}"))

    history = mem.get_history()
    assert len(history) == 3
    assert len(mem) == 3
    # Oldest two messages evicted; the window keeps messages 2, 3, 4.
    assert history[0].content == "msg 2"
    assert history[-1].content == "msg 4"


def test_short_term_default_max_messages():
    mem = ShortTermMemory()
    assert mem.max_messages == 20


def test_short_term_clear():
    mem = ShortTermMemory()
    mem.add(Message(role="user", content="hi"))
    mem.clear()
    assert mem.get_history() == []
    assert len(mem) == 0


def test_short_term_get_history_returns_copy():
    mem = ShortTermMemory()
    mem.add(Message(role="user", content="hi"))
    first = mem.get_history()
    first.append(Message(role="user", content="extra"))
    second = mem.get_history()
    # Mutating the returned list must not affect internal state.
    assert len(second) == 1


def test_long_term_add_and_search():
    mem = LongTermMemory()
    mem.add("Python is a programming language.")
    mem.add("The cat sat on the mat.")
    mem.add("Python is great for data science.", metadata={"tag": "ds"})

    results = mem.search("Python")
    assert len(results) == 2
    texts = {r.text for r in results}
    assert "Python is a programming language." in texts
    assert "Python is great for data science." in texts


def test_long_term_search_case_insensitive():
    mem = LongTermMemory()
    mem.add("The Quick Brown Fox")
    results = mem.search("quick")
    assert len(results) == 1
    assert results[0].text == "The Quick Brown Fox"


def test_long_term_search_no_match():
    mem = LongTermMemory()
    mem.add("hello world")
    assert mem.search("missing") == []


def test_long_term_search_empty_query():
    mem = LongTermMemory()
    mem.add("hello world")
    assert mem.search("") == []


def test_long_term_search_ranks_by_occurrence_count():
    mem = LongTermMemory()
    mem.add("cat cat cat")
    mem.add("cat")
    results = mem.search("cat")
    assert len(results) == 2
    assert results[0].text == "cat cat cat"
    assert results[1].text == "cat"


def test_long_term_search_respects_k_limit():
    mem = LongTermMemory()
    for i in range(6):
        mem.add(f"item item {i}")
    results = mem.search("item", k=3)
    assert len(results) == 3


def test_long_term_add_with_metadata():
    mem = LongTermMemory()
    mem.add("note", metadata={"source": "test"})
    results = mem.search("note")
    assert len(results) == 1
    assert isinstance(results[0], MemoryEntry)
    assert results[0].metadata == {"source": "test"}


def test_long_term_all_entries_and_clear():
    mem = LongTermMemory()
    mem.add("a")
    mem.add("b")
    assert len(mem.all_entries()) == 2
    mem.clear()
    assert mem.all_entries() == []
