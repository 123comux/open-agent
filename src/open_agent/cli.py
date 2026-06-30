"""Command-line interface for open-agent (Typer + Rich).

Commands:
  chat   Start an interactive REPL session with the agent.
  ask    Ask the agent a single question and print the answer.
  serve  Start the FastAPI server.

Use --demo flag on chat/ask to try without an API key (uses a mock model).
"""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from open_agent.config import Settings, get_settings

# Fix Windows GBK encoding issues with Rich
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

app = typer.Typer(
    name="open-agent",
    help="A general-purpose Agentic RAG autonomous work assistant.",
    no_args_is_help=True,
)
console = Console(force_terminal=True)


class DemoModel:
    """A mock model that simulates tool calls for demo mode."""

    def __init__(self) -> None:
        from open_agent.models.base import Message, ModelResponse, ToolCall

        self._Message = Message
        self._ModelResponse = ModelResponse
        self._ToolCall = ToolCall
        self._call_count = 0

    async def chat(self, messages, tools=None):
        """Return canned responses that demonstrate the ReAct loop."""
        user_msg = ""
        for m in reversed(messages):
            if m.role == "user":
                user_msg = m.content.lower()
                break

        # First call: try to use a tool based on the question
        if self._call_count == 0:
            self._call_count += 1
            if "calculate" in user_msg or "compute" in user_msg or "math" in user_msg:
                return self._ModelResponse(
                    content="",
                    tool_calls=[
                        self._ToolCall(
                            name="python",
                            arguments={"code": "result = 2 ** 10\nprint(result)"},
                        )
                    ],
                )
            if "list" in user_msg and "file" in user_msg:
                return self._ModelResponse(
                    content="",
                    tool_calls=[self._ToolCall(name="file", arguments={"action": "list", "path": "."})],
                )
            if "run" in user_msg or "execute" in user_msg or "command" in user_msg:
                return self._ModelResponse(
                    content="",
                    tool_calls=[
                        self._ToolCall(name="shell", arguments={"command": "echo Hello from Open Agent!"})
                    ],
                )
            # Default: respond directly
            return self._ModelResponse(
                content=(
                    "Hello! I'm Open Agent running in **demo mode**.\n\n"
                    "I can use tools like `python`, `shell`, and `file` to help you.\n\n"
                    "Try asking me to:\n"
                    "- Calculate something (e.g. 'calculate 2 to the power of 10')\n"
                    "- List files in a directory\n"
                    "- Run a shell command\n\n"
                    "To use a real LLM, set `OPEN_AGENT_API_KEY` and run without `--demo`."
                ),
                tool_calls=[],
            )

        # Second call: summarize the tool result
        self._call_count = 0
        return self._ModelResponse(
            content="Done! The tool executed successfully. (Demo mode — connect a real LLM for intelligent responses.)",
            tool_calls=[],
        )


def _build_agent(settings: Settings, demo: bool = False):
    """Construct an Agent from settings (heavy imports kept lazy)."""
    from open_agent.agent.core import Agent
    from open_agent.models.base import ModelInterface
    from open_agent.tools.builtin import FileTool, PythonTool, ShellTool, WebSearchTool
    from open_agent.tools.registry import ToolRegistry

    if demo:
        model = DemoModel()
    else:
        provider = settings.model_provider
        model: ModelInterface
        if provider == "anthropic":
            from open_agent.models.anthropic_provider import AnthropicModel

            model = AnthropicModel(
                api_key=settings.api_key, model=settings.model_name, timeout=settings.request_timeout
            )
        elif provider == "ollama":
            from open_agent.models.ollama_provider import OllamaModel

            model = OllamaModel(
                base_url=settings.base_url, model=settings.model_name, timeout=settings.request_timeout
            )
        else:
            from open_agent.models.openai_provider import OpenAIModel

            model = OpenAIModel(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model_name,
                timeout=settings.request_timeout,
            )

    registry = ToolRegistry()
    for tool in (ShellTool(), PythonTool(), FileTool(), WebSearchTool()):
        registry.register(tool)

    return Agent(model=model, tool_registry=registry, max_steps=settings.max_steps)


@app.command()
def chat(demo: bool = typer.Option(False, "--demo", help="Run with a mock model, no API key needed.")) -> None:
    """Start an interactive REPL session with the agent."""
    settings = get_settings()
    agent = _build_agent(settings, demo=demo)
    mode_label = "demo" if demo else f"{settings.model_provider}/{settings.model_name}"
    console.print(
        Panel(
            f"Open Agent ready. Mode: {mode_label}. Type 'exit' to quit.",
            title="open-agent",
        )
    )
    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break
        if user_input.strip().lower() in {"exit", "quit", ":q"}:
            console.print("Goodbye.")
            break
        if not user_input.strip():
            continue
        try:
            output = asyncio.run(agent.run(user_input))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            continue
        console.print(Panel(Markdown(output.response), title="assistant"))
        if output.tool_calls_made:
            for tc in output.tool_calls_made:
                console.print(
                    f"  [dim]tool: {tc['name']}({tc['arguments']}) -> "
                    f"{tc['observation'][:80]}...[/dim]"
                )


@app.command()
def ask(
    question: str,
    demo: bool = typer.Option(False, "--demo", help="Run with a mock model, no API key needed."),
) -> None:
    """Ask the agent a single question and print the answer."""
    settings = get_settings()
    agent = _build_agent(settings, demo=demo)
    output = asyncio.run(agent.run(question))
    console.print(Panel(Markdown(output.response), title="assistant"))
    console.print(
        f"[dim]steps: {output.steps}, tool calls: {len(output.tool_calls_made)}[/dim]"
    )
    if output.tool_calls_made:
        for tc in output.tool_calls_made:
            console.print(
                f"  [dim]tool: {tc['name']}({tc['arguments']}) -> "
                f"{tc['observation'][:100]}[/dim]"
            )


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Start the FastAPI server."""
    settings = get_settings()
    from open_agent.server.api import main as run_server

    run_server(host=host or settings.server_host, port=port or settings.server_port)


if __name__ == "__main__":
    app()
