"""Optional sandbox guards for tool execution.

When ``enable_tool_sandbox`` is enabled in settings, the ``check_*``
functions raise ``PermissionError`` if the user input matches a known
dangerous pattern. The guards are intentionally conservative: they block
anything that looks like it could escape the sandbox or damage the system.
"""
from __future__ import annotations

import os
import re
import shlex

from open_agent.config import get_settings

DEFAULT_PYTHON_BLOCKED_PATTERNS: list[str] = [
    r"__import__\s*\(",
    r"\bimportlib\b",
    r"\bimport\s+os\b",
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\bos\.popen\b",
    r"\bshutil\.rmtree\b",
    r"getattr\s*\(\s*__builtins__",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"open\s*\(\s*['\"]",
    r"__builtins__",
    r"\bglobals\s*\(\s*\)",
    r"\blocals\s*\(\s*\)",
    r"\bvars\s*\(\s*\)",
    r"\bbreakpoint\s*\(",
]

DEFAULT_SHELL_BLOCKED_PATTERNS: list[str] = [
    r"rm\s+-rf?\b",
    r"rmdir\s+/s\b",
    r"del\s+/[fq]\b",
    r"\bformat\b\s+[a-z]: ",
    r"\bmkfs\b",
    r"dd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r":\(\)\s*\{\s*:\|:\&\s*\}\s*;\s*:\}",
    r">\s*/dev/sd",
    r"chmod\s+777\b",
    r"curl\s+[^|]+\|\s*(sh|bash)",
    r"wget\s+[^|]+\|\s*(sh|bash)",
    r"\bnc\s+-l\b",
    r"crontab\s+-r\b",
    r"kill\s+-9\s+1\b",
    r"\bkillall\b",
    r"\bpkill\b",
    r"taskkill\s+/[fi]\b",
    r"reg\s+delete\b",
    r"reg\s+add\b.*\s+/f\b",
    r"takeown\s+/f\b",
    r"icacls\b.*\s+/grant\b",
    r"pip\s+uninstall\b",
    r"npm\s+uninstall\b",
    r"(rm|del|format)\s+.*(;\s*|&&\s*|\|\|\s*)",
]


def _sandbox_enabled() -> bool:
    """Return whether the tool sandbox is currently enabled.

    The setting is sourced from the cached :func:`get_settings` singleton
    (which parses ``OPEN_AGENT_ENABLE_TOOL_SANDBOX`` at load time) so callers
    and tests can patch either ``get_settings`` or ``sandbox_enabled``.
    """
    try:
        return bool(get_settings().enable_tool_sandbox)
    except Exception:
        return False


def sandbox_enabled() -> bool:
    """Public wrapper around :func:`_sandbox_enabled`."""
    return _sandbox_enabled()


def check_python(code: str) -> str | None:
    """Return a blocking message if ``code`` matches a dangerous pattern.

    Returns ``None`` when the code is allowed. A string (rather than a raised
    exception) is returned so :class:`PythonTool` can surface the message to
    the caller verbatim.
    """
    if not _sandbox_enabled():
        return None
    for pattern in DEFAULT_PYTHON_BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"Sandbox blocked Python code: matched pattern '{pattern}'"
    return None


def check_path(path: str) -> str | None:
    """Return a blocking message if ``path`` is unsafe.

    Blocks path traversal (``..``), sensitive system directories, and paths
    outside the configured ``sandbox_allowed_paths`` whitelist. Returns
    ``None`` when the path is allowed (or when the sandbox is disabled).
    """
    if not _sandbox_enabled():
        return None
    normalized = os.path.normpath(path)
    try:
        settings = get_settings()
        allowed = list(settings.sandbox_allowed_paths)
        blocked_paths = list(settings.sandbox_blocked_paths)
    except Exception:
        allowed = []
        blocked_paths = []
    for bp in blocked_paths:
        nbp = os.path.normpath(bp)
        if normalized == nbp or normalized.startswith(nbp + os.sep):
            return f"Sandbox blocked path: {path}"
    if allowed:
        norm_allowed = [os.path.normpath(a) for a in allowed]
        inside = any(
            normalized == a or normalized.startswith(a + os.sep)
            for a in norm_allowed
        )
        if not inside:
            return f"Sandbox blocked path (outside allowed directories): {path}"
    parts = normalized.split(os.sep)
    if ".." in parts:
        return f"Sandbox blocked path traversal: {path}"
    sensitive = ("/etc/", "/root/", "/var/log/")
    for prefix in sensitive:
        posix_norm = prefix.replace("/", os.sep)
        if (
            normalized == posix_norm.rstrip(os.sep)
            or normalized.startswith(posix_norm)
        ):
            return f"Sandbox blocked sensitive path: {path}"
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            return f"Sandbox blocked sensitive path: {path}"
    if normalized.lower().startswith("c:\\windows\\system32"):
        return f"Sandbox blocked sensitive path: {path}"
    return None


def check_shell(command: str) -> None:
    """Raise :class:`PermissionError` if ``command`` matches a dangerous pattern."""
    for pattern in DEFAULT_SHELL_BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise PermissionError(f"Sandbox blocked shell command: {command}")


def check_shell_safety(command: str, *, enabled: bool | None = None) -> None:
    """Raise :class:`PermissionError` if the sandbox is on and ``command`` is unsafe.

    When ``enabled`` is ``None`` the current sandbox state is read via
    :func:`sandbox_enabled`. Pass an explicit bool to override the check
    (useful for tests and callers that already know the desired state).
    """
    if enabled is None:
        enabled = sandbox_enabled()
    if not enabled:
        return
    check_shell(command)


def parse_shell_command(command: str) -> list[str]:
    """Split a shell command into tokens.

    Always uses POSIX-style splitting so that surrounding quotes are stripped
    from arguments (e.g. ``dir "C:\\Program Files"`` → ``['dir',
    'C:\\Program Files']``). This is the correct behavior for
    :func:`asyncio.create_subprocess_exec`, which takes an argv-style list and
    does not go through a shell — keeping the quotes in the token would make
    the subprocess look for a filename that literally contains quote
    characters.

    On a parse error (e.g. an unmatched quote) the original command is
    returned as the only token. An empty command returns ``[""]`` for a
    predictable non-empty list.
    """
    if not command:
        return [""]
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return [command]
