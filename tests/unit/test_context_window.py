"""Tests for the context-window management helpers."""
from __future__ import annotations

from open_agent.agent.context_window import (
    estimate_messages_tokens,
    estimate_tokens,
    truncate_messages,
)


def test_estimate_tokens_positive_and_proportional():
    assert estimate_tokens("") == 0
    short = estimate_tokens("hi")
    long = estimate_tokens("a" * 100)
    assert isinstance(short, int)
    assert short > 0
    assert long > short
    assert estimate_tokens("hello world") >= estimate_tokens("hello")


def test_estimate_tokens_cjk_heuristic_uses_smaller_divisor():
    latin = "a" * 60
    cjk = "中" * 60
    assert estimate_tokens(cjk) >= estimate_tokens(latin)


def test_estimate_messages_tokens_sums_with_overhead():
    msgs = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "defgh"},
    ]
    expected = estimate_tokens("abc") + 4 + estimate_tokens("defgh") + 4
    assert estimate_messages_tokens(msgs) == expected


def test_estimate_messages_tokens_handles_non_string_content():
    assert estimate_messages_tokens([{"role": "user", "content": 12345}]) > 0

def test_truncate_preserves_system_message():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    result = truncate_messages(msgs, max_tokens=10, preserve_system=True, preserve_recent=2)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are helpful."


def test_truncate_preserves_recent_n_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "old2"},
        {"role": "user", "content": "recent1"},
        {"role": "assistant", "content": "recent2"},
    ]
    result = truncate_messages(msgs, max_tokens=20, preserve_recent=2)
    contents = [m["content"] for m in result]
    assert "recent1" in contents
    assert "recent2" in contents
    assert result[-1]["content"] == "recent2"


def test_truncate_drops_middle_messages_when_over_limit():
    msgs = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 1000},
        {"role": "user", "content": "z" * 20},
        {"role": "assistant", "content": "w" * 20},
    ]
    result = truncate_messages(msgs, max_tokens=60, preserve_recent=2)
    contents = [m["content"] for m in result]
    assert len(result) == 3
    assert result[0]["content"] == "system"
    assert contents[-2] == "z" * 20
    assert contents[-1] == "w" * 20
    assert "x" * 1000 not in contents
    assert "y" * 1000 not in contents

def test_truncate_keeps_everything_when_under_limit():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    result = truncate_messages(msgs, max_tokens=10_000, preserve_recent=2)
    assert [m["content"] for m in result] == ["sys", "hello", "world"]


def test_truncate_returns_copy_and_does_not_mutate_input():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    result = truncate_messages(msgs, max_tokens=10_000, preserve_recent=2)
    assert result is not msgs
    result[0]["content"] = "MUTATED"
    result.append({"role": "user", "content": "extra"})
    assert msgs[0]["content"] == "sys"
    assert len(msgs) == 3


def test_truncate_empty_list():
    assert truncate_messages([], max_tokens=100) == []


def test_truncate_no_system_message_when_disabled():
    msgs = [
        {"role": "user", "content": "a" * 1000},
        {"role": "assistant", "content": "b" * 1000},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    result = truncate_messages(msgs, max_tokens=20, preserve_system=False, preserve_recent=2)
    assert result[-1]["content"] == "d"
    assert result[-2]["content"] == "c"


def test_truncate_shrinks_recent_content_when_system_plus_recent_exceed_limit():
    msgs = [
        {"role": "system", "content": "s" * 50},
        {"role": "user", "content": "r1-tail-marker"},
        {"role": "assistant", "content": "r2"},
    ]
    result = truncate_messages(msgs, max_tokens=30, preserve_recent=2)
    assert estimate_messages_tokens(result) <= 30
    assert result[0]["content"] == "s" * 50


def test_truncate_respects_preserve_recent_zero():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a" * 1000},
        {"role": "assistant", "content": "b" * 1000},
    ]
    result = truncate_messages(msgs, max_tokens=10, preserve_recent=0, preserve_system=True)
    assert result[0]["role"] == "system"
    assert estimate_messages_tokens(result) <= 10 or len(result) == 1
