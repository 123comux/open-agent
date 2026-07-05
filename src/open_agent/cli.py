"""Command-line interface for open-agent (Typer + Rich).

Commands:
  chat      Start an interactive REPL session with the agent.
  ask       Ask the agent a single question and print the answer.
  serve     Start the FastAPI server.
  index     Index documents into a knowledge base for RAG.
  evaluate  Evaluate RAG quality on a set of test cases.

Use --demo flag on chat/ask to try without an API key (uses a mock model).
Use --langgraph flag to use LangGraph-based agent with thinking chain.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from open_agent.config import Settings, get_settings
from open_agent.observability.tracer import NoOpTracer, Tracer

if TYPE_CHECKING:
    from open_agent.models.base import ModelInterface

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
        self._last_content = ""

    async def chat(self, messages: list[Any], tools: list[Any] | None = None) -> Any:
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
                response = self._ModelResponse(
                    content="",
                    tool_calls=[
                        self._ToolCall(
                            name="python",
                            arguments={"code": "result = 2 ** 10\nprint(result)"},
                        )
                    ],
                )
            elif "list" in user_msg and "file" in user_msg:
                response = self._ModelResponse(
                    content="",
                    tool_calls=[
                        self._ToolCall(name="file", arguments={"action": "list", "path": "."})
                    ],
                )
            elif "run" in user_msg or "execute" in user_msg or "command" in user_msg:
                response = self._ModelResponse(
                    content="",
                    tool_calls=[
                        self._ToolCall(
                            name="shell",
                            arguments={"command": "echo Hello from Open Agent!"},
                        )
                    ],
                )
            else:
                # Default: respond directly
                response = self._ModelResponse(
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
        else:
            # Second call: summarize the tool result
            self._call_count = 0
            response = self._ModelResponse(
                content=(
                    "Done! The tool executed successfully. "
                    "(Demo mode -- connect a real LLM for intelligent responses.)"
                ),
                tool_calls=[],
            )

        self._last_content = response.content
        return response

    async def stream_chat(self, messages: list[Any], tools: list[Any] | None = None) -> Any:
        """Stream the cached response as a single chunk (demo mode).

        :meth:`Agent.run_stream` calls this right after :meth:`chat` returned a
        direct response, so we replay the cached content as one chunk.
        """
        if self._last_content:
            yield self._last_content


def _build_model(settings: Settings) -> ModelInterface:
    """Construct a model from settings (heavy imports kept lazy)."""
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
    elif provider == "zhipu":
        from open_agent.models.zhipu_provider import ZHIPU_DEFAULT_BASE_URL, ZhipuModel

        # Use Zhipu default base_url unless the user explicitly overrode it
        # away from the OpenAI default.
        base_url = (
            settings.base_url
            if settings.base_url != "https://api.openai.com/v1"
            else ZHIPU_DEFAULT_BASE_URL
        )
        model = ZhipuModel(
            api_key=settings.api_key,
            base_url=base_url,
            model=settings.model_name,
            timeout=settings.request_timeout,
        )
    else:
        from open_agent.models.openai_provider import OpenAIModel

        model = OpenAIModel(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model_name,
            timeout=settings.request_timeout,
        )
    return model


async def _build_agent(
    settings: Settings, demo: bool = False, use_langgraph: bool = False
) -> Any:
    """Construct an Agent from settings (heavy imports kept lazy)."""
    from open_agent.tools.builtin import (
        FileTool,
        KnowledgeBaseTool,
        PythonTool,
        ShellTool,
        WebSearchTool,
    )
    from open_agent.tools.registry import ToolRegistry

    if demo:
        model = cast("ModelInterface", DemoModel())
    else:
        model = _build_model(settings)

    registry = ToolRegistry()
    try:
        from open_agent.rag.kb_manager import KBManager

        kb_manager = KBManager(
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            split_unit=settings.split_unit,
            top_k=settings.rag_top_k,
            reranker_model=settings.reranker_model,
            rerank_k=settings.rerank_k,
        )
        kb_tool = KnowledgeBaseTool(kb_manager=kb_manager, top_k=settings.rag_top_k)
    except ImportError:
        kb_tool = KnowledgeBaseTool()
    for tool in (ShellTool(), PythonTool(), FileTool(), WebSearchTool(), kb_tool):
        registry.register(tool)

    # Load MCP servers if configured.
    mcp_client: Any = None
    if settings.mcp_servers_file:
        try:
            from open_agent.mcp import MCPClient, adapt_mcp_tools, load_mcp_servers

            servers = load_mcp_servers(settings.mcp_servers_file)
            if servers:
                mcp_client = MCPClient(servers)
                await mcp_client.connect()
                for mcp_tool in adapt_mcp_tools(mcp_client):
                    registry.register(mcp_tool)
        except Exception as exc:  # pragma: no cover - optional integration
            console.print(
                f"[yellow]Warning: failed to load MCP servers: {exc}[/yellow]"
            )

    tracer: Tracer = NoOpTracer()
    if settings.enable_observability:
        try:
            if settings.observability_provider == "langsmith":
                from open_agent.observability.tracer import LangSmithTracer

                tracer = LangSmithTracer(
                    api_key=settings.langsmith_api_key or None,
                    api_url=settings.langsmith_api_url or None,
                    project_name=settings.langsmith_project or None,
                )
            elif settings.observability_provider == "langfuse":
                from open_agent.observability.tracer import LangfuseTracer

                tracer = LangfuseTracer(
                    public_key=settings.langfuse_public_key or None,
                    secret_key=settings.langfuse_secret_key or None,
                    host=settings.langfuse_host or None,
                )
            else:
                from open_agent.observability.tracer import LocalJsonlTracer

                tracer = LocalJsonlTracer(settings.observability_output_dir)
        except Exception as exc:  # pragma: no cover
            console.print(
                f"[yellow]Warning: failed to initialize observability tracer: {exc}[/yellow]"
            )

    if use_langgraph:
        try:
            from open_agent.agent.langgraph_agent import LangGraphAgent

            lg_agent: Any = LangGraphAgent(
                model=model,
                tools=list(registry._tools.values()),
                max_steps=settings.max_steps,
            )
            lg_agent._mcp_client = mcp_client
            return lg_agent
        except ImportError as exc:
            console.print(
                f"[yellow]LangGraph not available ({exc}), "
                "falling back to core agent.[/yellow]"
            )

    long_term_memory = None
    if settings.enable_long_term_memory:
        try:
            from open_agent.memory.long_term import LongTermMemory

            long_term_memory = LongTermMemory(
                embedding_model=settings.embedding_model,
                storage_dir=settings.long_term_memory_dir,
                top_k=settings.long_term_memory_top_k,
            )
        except Exception as exc:  # pragma: no cover - optional integration
            console.print(
                f"[yellow]Warning: failed to initialize long-term memory: {exc}[/yellow]"
            )

    from open_agent.agent.core import Agent

    agent = Agent(
        model=model,
        tool_registry=registry,
        max_steps=settings.max_steps,
        long_term_memory=long_term_memory,
        tracer=tracer,
    )
    agent._mcp_client = mcp_client  # type: ignore[attr-defined]
    return agent


def _print_thinking_chain(output: Any) -> None:
    """Display the agent's thinking chain (Thought / Action / Observation)."""
    # Show intent classification if available
    if hasattr(output, "intent") and output.intent:
        console.print(
            Panel(
                f"[bold green]Intent:[/bold green] {output.intent}",
                title="Intent Classification",
            )
        )

    # Show sub-tasks if available
    if hasattr(output, "sub_tasks") and output.sub_tasks:
        tree = Tree("[bold blue]Task Decomposition[/bold blue]")
        for i, task in enumerate(output.sub_tasks, 1):
            tree.add(f"[dim]Sub-task {i}:[/dim] {task}")
        console.print(tree)

    if not hasattr(output, "thoughts") or not output.thoughts:
        return

    tree = Tree("[bold blue]Thinking Chain[/bold blue]")
    for i, thought in enumerate(output.thoughts, 1):
        tree.add(f"[dim]Step {i}:[/dim] {thought}")
    console.print(tree)

    # Show reflections if available
    if hasattr(output, "reflections") and output.reflections:
        tree = Tree("[bold magenta]Reflections[/bold magenta]")
        for i, refl in enumerate(output.reflections, 1):
            tree.add(f"[dim]Reflection {i}:[/dim] {refl}")
        console.print(tree)

    if output.tool_calls_made:
        table = Table(title="Tool Calls", show_header=True, header_style="bold magenta")
        table.add_column("Step", style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Arguments", style="green")
        table.add_column("Result", style="yellow")
        for tc in output.tool_calls_made:
            obs = tc.get("observation", "")
            if len(obs) > 100:
                obs = obs[:100] + "..."
            table.add_row(
                str(tc.get("step", "?")),
                tc.get("name", "?"),
                str(tc.get("arguments", {})),
                obs,
            )
        console.print(table)


async def _stream_agent_response(agent: Any, user_input: str) -> None:
    """Run the agent with streaming, printing events as they arrive.

    Falls back to the non-streaming :meth:`run` for agents without
    ``run_stream`` (e.g. :class:`LangGraphAgent`).
    """
    if not hasattr(agent, "run_stream"):
        # Fallback for agents without streaming.
        output = await agent.run(user_input)
        _print_thinking_chain(output)
        console.print(Panel(Markdown(output.response), title="assistant"))
        console.print(
            f"[dim]steps: {output.steps}, tool calls: {len(output.tool_calls_made)}[/dim]"
        )
        return

    tool_calls_made: list[dict[str, Any]] = []
    steps = 0
    console.print("[bold cyan]assistant>[/bold cyan] ", end="")
    async for event in agent.run_stream(user_input):
        etype = event.get("type")
        if etype == "tool_start":
            console.print(
                f"\n[dim cyan]→ calling {event.get('name', '?')}...[/dim cyan]"
            )
            tool_calls_made.append(
                {
                    "name": event.get("name", "?"),
                    "arguments": event.get("arguments", {}),
                    "observation": "",
                    "is_error": False,
                }
            )
        elif etype == "tool_end":
            obs = event.get("observation", "")
            if len(obs) > 200:
                obs = obs[:200] + "..."
            color = "dim red" if event.get("is_error") else "dim yellow"
            console.print(f"[{color}]  ← {obs}[/{color}]")
            if tool_calls_made:
                tool_calls_made[-1]["observation"] = event.get("observation", "")
                tool_calls_made[-1]["is_error"] = event.get("is_error", False)
        elif etype == "token":
            console.print(event.get("content", ""), end="", style="green")
        elif etype == "done":
            steps = event.get("steps", 0)
            tool_calls_made = event.get("tool_calls_made", tool_calls_made)
    console.print()  # newline after streaming tokens
    console.print(
        f"[dim]steps: {steps}, tool calls: {len(tool_calls_made)}[/dim]"
    )


async def _close_resources(agent: Any) -> None:
    """Close resources (model httpx client, MCP client) held by ``agent``.

    Runs on session exit so the underlying httpx ``AsyncClient`` and MCP stdio
    subprocesses are not left bound to a closed event loop.
    """
    mcp_client = getattr(agent, "_mcp_client", None)
    if mcp_client is not None:
        close = getattr(mcp_client, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass
    model = getattr(agent, "model", None)
    if model is not None:
        aclose = getattr(model, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


async def _chat_main(
    settings: Settings, demo: bool = False, use_langgraph: bool = False
) -> None:
    """Run the interactive REPL inside a single event loop.

    Building the agent and streaming every response share one loop so the
    model's ``httpx.AsyncClient`` is never reused across a closed loop.
    """
    agent = await _build_agent(settings, demo=demo, use_langgraph=use_langgraph)
    mode_label = "demo" if demo else f"{settings.model_provider}/{settings.model_name}"
    if use_langgraph:
        mode_label += " (LangGraph)"
    console.print(
        Panel(
            f"Open Agent ready. Mode: {mode_label}. Type 'exit' to quit.",
            title="open-agent",
        )
    )
    try:
        while True:
            try:
                user_input = await asyncio.to_thread(
                    console.input, "[bold cyan]you>[/bold cyan] "
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                break
            if user_input.strip().lower() in {"exit", "quit", ":q"}:
                console.print("Goodbye.")
                break
            if not user_input.strip():
                continue
            try:
                await _stream_agent_response(agent, user_input)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Error:[/red] {exc}")
                continue
    finally:
        await _close_resources(agent)


async def _ask_main(
    settings: Settings,
    question: str,
    demo: bool = False,
    use_langgraph: bool = False,
) -> None:
    """Ask a single question within a single event loop."""
    agent = await _build_agent(settings, demo=demo, use_langgraph=use_langgraph)
    try:
        await _stream_agent_response(agent, question)
    finally:
        await _close_resources(agent)


@app.command()
def chat(
    demo: bool = typer.Option(
        False, "--demo", help="Run with a mock model, no API key needed."
    ),
    langgraph: bool = typer.Option(
        False, "--langgraph", help="Use LangGraph agent with thinking chain."
    ),
) -> None:
    """Start an interactive REPL session with the agent."""
    settings = get_settings()
    asyncio.run(_chat_main(settings, demo=demo, use_langgraph=langgraph))


@app.command()
def ask(
    question: str,
    demo: bool = typer.Option(
        False, "--demo", help="Run with a mock model, no API key needed."
    ),
    langgraph: bool = typer.Option(
        False, "--langgraph", help="Use LangGraph agent with thinking chain."
    ),
) -> None:
    """Ask the agent a single question and print the answer."""
    settings = get_settings()
    asyncio.run(_ask_main(settings, question, demo=demo, use_langgraph=langgraph))


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory or file path to index."),
    kb_name: str = typer.Option("default", "--kb", help="Knowledge base name."),
    description: str = typer.Option("", "--desc", help="Knowledge base description."),
) -> None:
    """Index documents into a knowledge base for RAG retrieval."""
    settings = get_settings()
    console.print(f"[cyan]Indexing '{path}' into KB '{kb_name}'...[/cyan]")
    try:
        from open_agent.rag.kb_manager import KBManager

        manager = KBManager(
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            split_unit=settings.split_unit,
            top_k=settings.rag_top_k,
            reranker_model=settings.reranker_model,
            rerank_k=settings.rerank_k,
        )
        p = Path(path)
        if p.is_dir():
            count = asyncio.run(manager.index_directory(str(p), kb_name, description))
        elif p.is_file():
            count = asyncio.run(manager.index_file(str(p), kb_name))
        else:
            console.print(f"[red]Error: '{path}' is not a valid file or directory.[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Indexed {count} chunks into KB '{kb_name}'.[/green]")
    except ImportError as exc:
        console.print(f"[red]RAG dependencies not installed: {exc}[/red]")
        console.print("[dim]Install with: pip install 'open-agent[rag]'[/dim]")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def serve(host: str | None = None, port: int | None = None) -> None:
    """Start the FastAPI server."""
    settings = get_settings()
    from open_agent.server.api import main as run_server

    run_server(host=host or settings.server_host, port=port or settings.server_port)


@app.command()
def evaluate(
    qa_file: str = typer.Argument(..., help="JSON file with test cases."),
    demo: bool = typer.Option(False, "--demo", help="Use heuristic metrics without LLM."),
) -> None:
    """Evaluate RAG quality on a set of test cases."""
    import json

    from open_agent.rag.evaluation import RAGEvaluator, RAGTestCase

    test_data = json.loads(Path(qa_file).read_text(encoding="utf-8"))
    test_cases = [
        RAGTestCase(
            question=tc["question"],
            expected_answer=tc.get("expected_answer", ""),
            retrieved_contexts=tc.get("retrieved_contexts", []),
            generated_answer=tc.get("generated_answer", ""),
            ground_truth_contexts=tc.get("ground_truth_contexts", []),
        )
        for tc in test_data["test_cases"]
    ]

    if demo:
        evaluator = RAGEvaluator(model=None)
        results = asyncio.run(evaluator.evaluate_batch(test_cases))
    else:
        settings = get_settings()
        model = _build_model(settings)
        evaluator = RAGEvaluator(model=model)

        async def _run_eval() -> Any:
            try:
                return await evaluator.evaluate_batch(test_cases)
            finally:
                aclose = getattr(model, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass

        results = asyncio.run(_run_eval())

    # Print results table
    table = Table(title="RAG Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("Faithfulness", style="green")
    table.add_column("Relevance", style="yellow")
    table.add_column("Recall", style="blue")
    table.add_column("Precision", style="magenta")
    table.add_column("Overall", style="bold red")

    for tc, result in zip(test_cases, results):
        table.add_row(
            tc.question[:40] + "..." if len(tc.question) > 40 else tc.question,
            f"{result.faithfulness:.2f}",
            f"{result.answer_relevance:.2f}",
            f"{result.context_recall:.2f}",
            f"{result.context_precision:.2f}",
            f"{result.overall_score:.2f}",
        )

    console.print(table)

    # Print average
    def _avg(key: str) -> float:
        return float(sum(getattr(r, key) for r in results) / len(results))

    console.print("\n[bold]Average Scores:[/bold]")
    console.print(f"  Faithfulness: {_avg('faithfulness'):.3f}")
    console.print(f"  Relevance:    {_avg('answer_relevance'):.3f}")
    console.print(f"  Recall:       {_avg('context_recall'):.3f}")
    console.print(f"  Precision:    {_avg('context_precision'):.3f}")
    console.print(f"  Overall:      {_avg('overall_score'):.3f}")


if __name__ == "__main__":
    app()
