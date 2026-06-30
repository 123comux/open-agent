"""Command-line interface for open-agent (Typer + Rich).

Commands:
  chat   Start an interactive REPL session with the agent.
  ask    Ask the agent a single question and print the answer.
  serve  Start the FastAPI server.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from open_agent.config import Settings, get_settings

app = typer.Typer(
    name="open-agent",
    help="A general-purpose Agentic RAG autonomous work assistant.",
    no_args_is_help=True,
)
console = Console()


def _build_agent(settings: Settings):
    """Construct an Agent from settings (heavy imports kept lazy)."""
    from open_agent.agent.core import Agent
    from open_agent.models.base import ModelInterface
    from open_agent.tools.builtin import FileTool, PythonTool, ShellTool, WebSearchTool
    from open_agent.tools.registry import ToolRegistry

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
def chat() -> None:
    """Start an interactive REPL session with the agent."""
    settings = get_settings()
    agent = _build_agent(settings)
    console.print(
        Panel(
            f"Open Agent ready. Provider: {settings.model_provider}, "
            f"Model: {settings.model_name}. Type 'exit' to quit.",
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


@app.command()
def ask(question: str) -> None:
    """Ask the agent a single question and print the answer."""
    settings = get_settings()
    agent = _build_agent(settings)
    output = asyncio.run(agent.run(question))
    console.print(Panel(Markdown(output.response), title="assistant"))
    console.print(
        f"[dim]steps: {output.steps}, tool calls: {len(output.tool_calls_made)}[/dim]"
    )


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Start the FastAPI server."""
    settings = get_settings()
    from open_agent.server.api import main as run_server

    run_server(host=host or settings.server_host, port=port or settings.server_port)


if __name__ == "__main__":
    app()
