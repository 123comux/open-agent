"""Context window management for the agent.

Provides lightweight helpers to estimate the token cost of messages and to
truncate a conversation so it fits within a model's context window. Token
estimation prefers :mod:`tiktoken` when available and falls back to a
length-based heuristic that accounts for CJK text being denser per char.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_encoding_cache: dict[str, Any] = {}

_CJK_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]"
)


def _count_cjk(text: str) -> int:
    """Return the number of CJK characters in ``text``."""
    return len(_CJK_RE.findall(text))


def _get_encoding(model: str) -> Any:
    """Return a cached tiktoken encoding appropriate for ``model``.

    Prefers ``tiktoken.encoding_for_model`` when ``model`` is provided and
    recognized by tiktoken, falling back to the ``cl100k_base`` encoding.
    Imports :mod:`tiktoken` lazily so the module remains importable when
    tiktoken is not installed; the resulting :class:`ImportError` propagates
    to the caller (typically caught by :func:`estimate_tokens`).
    """
    import tiktoken

    key = model or "default"
    if key in _encoding_cache:
        return _encoding_cache[key]
    enc = None
    if model:
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            pass
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")
    _encoding_cache[key] = enc
    return enc


def estimate_tokens(text: str, model: str = "") -> int:
    """Estimate the number of tokens ``text`` consumes.

    Uses :mod:`tiktoken` when available, selecting the encoding appropriate
    for ``model`` (cached per model). Otherwise falls back to a heuristic:
    CJK-heavy text uses one token per ~2 characters, other text one token per
    ~3 characters. Returns 0 for an empty string and a positive int otherwise.
    """
    if not text:
        return 0
    try:
        enc = _get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        logger.debug("tiktoken unavailable, using heuristic token estimation")
    if _count_cjk(text) / len(text) > 0.30:
        return max(1, len(text) // 2)
    return max(1, len(text) // 3)


def _estimate_single_message_tokens(msg: dict[str, Any], model: str) -> int:
    """Estimate the token cost of a single message.

    Each message contributes ``estimate_tokens(content) + 4`` (the 4 accounts
    for role/formatting overhead). Non-string content is coerced to ``str``.
    """
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    return estimate_tokens(content, model) + 4


def estimate_messages_tokens(messages: list[dict[str, Any]], model: str = "") -> int:
    """Estimate total tokens for a list of messages.

    Each message contributes ``estimate_tokens(content) + 4`` (the 4 accounts
    for role/formatting overhead). Non-string content is coerced to ``str``.
    """
    total = 0
    for msg in messages:
        total += _estimate_single_message_tokens(msg, model)
    return total


def _fit_tail(
    content: str,
    context_messages: list[dict[str, Any]],
    role: str,
    max_tokens: int,
    model: str,
) -> str:
    """Binary-search the longest tail of ``content`` that fits within ``max_tokens``.

    ``context_messages`` are the messages that will precede the truncated
    message. Returns the tail substring (keeping the most recent characters) or
    an empty string if nothing fits.
    """
    if not content:
        return ""
    full = context_messages + [{"role": role, "content": content}]
    if estimate_messages_tokens(full, model) <= max_tokens:
        return content
    lo, hi = 0, len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        tail = content[-mid:] if mid > 0 else ""
        test = context_messages + [{"role": role, "content": tail}]
        if estimate_messages_tokens(test, model) <= max_tokens:
            best = tail
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def truncate_messages(
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    preserve_system: bool = True,
    preserve_recent: int = 2,
    model: str = "",
) -> list[dict[str, Any]]:
    """Truncate a conversation so its total token cost is at most ``max_tokens``.

    Strategy (preserves the most relevant context):
    1. Always preserve the leading ``system`` message when ``preserve_system``.
    2. Preserve the trailing ``preserve_recent`` messages (most recent context).
    3. Drop middle messages from the oldest until under the limit.
    4. If still over, truncate each recent message content via binary search
       (keeping the tail), dropping it entirely if it still does not fit.
    5. If only the system message remains and it is still over the limit,
       truncate its content (keeping the head) so the result fits.

    Returns a *new* list of message dicts (copies); the input list is not
    mutated. Returns an empty list when ``messages`` is empty.
    """
    if not messages:
        return []
    if estimate_messages_tokens(messages, model) <= max_tokens:
        return [dict(m) for m in messages]
    system_msg: dict[str, Any] | None = None
    start = 0
    if preserve_system and messages and messages[0].get("role") == "system":
        system_msg = dict(messages[0])
        start = 1
    rest = messages[start:]
    recent_count = max(0, preserve_recent)
    if recent_count >= len(rest):
        middle: list[dict[str, Any]] = []
        recent: list[dict[str, Any]] = [dict(m) for m in rest]
    else:
        split = len(rest) - recent_count
        middle = [dict(m) for m in rest[:split]]
        recent = [dict(m) for m in rest[split:]]
    sys_list: list[dict[str, Any]] = [system_msg] if system_msg is not None else []

    # Cache per-message token counts so the truncation loops don't re-encode
    # the same message on every iteration (would otherwise be O(n^2)).
    token_cache: dict[int, int] = {}

    def _msg_tokens(msg: dict[str, Any]) -> int:
        key = id(msg)
        if key not in token_cache:
            token_cache[key] = _estimate_single_message_tokens(msg, model)
        return token_cache[key]

    total = sum(_msg_tokens(m) for m in sys_list)
    total += sum(_msg_tokens(m) for m in middle)
    total += sum(_msg_tokens(m) for m in recent)

    while middle and total > max_tokens:
        total -= _msg_tokens(middle.pop(0))
    if total <= max_tokens:
        return sys_list + middle + recent
    while recent and total > max_tokens:
        msg = recent[0]
        content = str(msg.get("content", ""))
        role = str(msg.get("role", "user"))
        context = sys_list + middle + recent[1:]
        new_content = _fit_tail(content, context, role, max_tokens, model)
        if new_content or not content:
            old = token_cache.pop(id(msg), 0)
            msg["content"] = new_content
            total += _msg_tokens(msg) - old
            if total <= max_tokens:
                break
        total -= _msg_tokens(recent.pop(0))

    # Final fallback: if only the system message remains and it is still over
    # the limit, truncate the system content (keeping the head) so the result
    # fits within max_tokens. This handles oversized system prompts that
    # contain e.g. large tool descriptions.
    if (
        system_msg is not None
        and not middle
        and not recent
        and total > max_tokens
    ):
        content = system_msg.get("content", "")
        if isinstance(content, str) and content:
            marker = "\n...(truncated)"
            budget = max(0, max_tokens - 4)  # reserve role overhead
            lo, hi, best = 0, len(content), ""
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = (
                    content[:mid] + marker if mid < len(content) else content
                )
                if estimate_tokens(candidate, model) <= budget:
                    best = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1
            system_msg["content"] = best
            token_cache.pop(id(system_msg), None)

    return sys_list + middle + recent
