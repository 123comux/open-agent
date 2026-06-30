"""Full Agentic RAG demo: knowledge base + web search + code execution.

This demo shows the complete Agentic RAG pipeline:
1. Index sample documents into a knowledge base
2. Ask questions that require different tool combinations:
   - Knowledge base query (static knowledge)
   - Web search (real-time information)
   - Code execution (computation)
   - Multi-step complex task (all tools combined)

Requirements:
    pip install open-agent[all]
    Set environment variables:
        OPEN_AGENT_API_KEY=your-key
        OPEN_AGENT_BASE_URL=https://open.bigmodel.cn/api/paas/v4
        OPEN_AGENT_MODEL_NAME=glm-4-flash
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Fix Windows GBK encoding issues for Chinese output.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from open_agent.agent.core import Agent, AgentOutput
from open_agent.config import Settings, get_settings
from open_agent.models.base import (
    Message,
    ModelInterface,
    ModelResponse,
    ToolCall,
    ToolSchema,
)
from open_agent.tools.builtin import FileTool, PythonTool, ShellTool, WebSearchTool
from open_agent.tools.builtin.knowledge_base import KnowledgeBaseTool
from open_agent.tools.registry import ToolRegistry

# Optional: Rich for colored output.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    _CONSOLE = Console(force_terminal=True)
    HAS_RICH = True
except ImportError:
    _CONSOLE = None
    HAS_RICH = False


# --------------------------------------------------------------------------- #
# Sample knowledge base documents
# --------------------------------------------------------------------------- #

COMPANY_SALES = """\
公司2024年销售数据报告

2024年第一季度（Q1）销售额：1,200万元
2024年第二季度（Q2）销售额：1,420万元
2024年第三季度（Q3）销售额：1,580万元
2024年第四季度（Q4）预计销售额：1,750万元

各产品线Q3销售占比：
- 智能终端：620万元
- 软件服务：480万元
- 云解决方案：320万元
- 技术咨询：160万元

同比增长分析：
2023年Q3销售额为1,350万元，2024年Q3同比增长17.04%。
"""

PRODUCT_MANUAL = """\
OpenAgent Pro 产品手册 v3.0

产品概述：
OpenAgent Pro 是一款企业级智能助手平台，支持多模态交互、工具调用和知识库检索。

核心功能：
1. ReAct推理循环：通过"思考-行动-观察"模式解决复杂任务
2. 工具集成：支持Shell、Python、文件操作、网页搜索等内置工具
3. 知识库RAG：支持文档索引和语义检索
4. 多模型支持：兼容OpenAI、Anthropic、Ollama等模型提供商

版本信息：
- v3.0（2024年9月）：新增LangGraph编排引擎
- v2.5（2024年6月）：新增混合检索（向量+BM25）
- v2.0（2024年3月）：新增多知识库路由
"""

HR_POLICY = """\
公司人力资源政策手册

一、考勤制度
- 标准工作时间：周一至周五 9:00-18:00
- 弹性工作制：核心工作时间 10:00-16:00
- 远程办公：每周最多2天，需提前申请

二、年假政策
- 入职满1年：5天年假
- 入职满3年：10天年假
- 入职满5年：15天年假

三、绩效考核
- 考核周期：每季度一次
- 等级划分：S（卓越）、A（优秀）、B（达标）、C（待改进）
- 季度奖金：S级3倍月薪，A级2倍月薪，B级1倍月薪
"""

SAMPLE_DOCS = {
    "company_sales.txt": COMPANY_SALES,
    "product_manual.txt": PRODUCT_MANUAL,
    "hr_policy.txt": HR_POLICY,
}


# --------------------------------------------------------------------------- #
# Demo model -- simulates tool calls when no API key is set
# --------------------------------------------------------------------------- #


class RagDemoModel(ModelInterface):
    """Mock model that simulates the ReAct loop for demo mode.

    Inspects the current user question and plays back a pre-defined script
    of :class:`ModelResponse` objects (tool calls followed by a final answer).
    The tools are executed for real -- only the model reasoning is canned.
    """

    def __init__(self) -> None:
        self._step = 0
        self._question: str | None = None

    @staticmethod
    def _detect_question(messages: list[Message]) -> str:
        """Return the most recent non-observation user message."""
        for m in reversed(messages):
            if m.role == "user" and not m.content.startswith("Observation:"):
                return m.content
        return ""

    def _get_script(self, question: str) -> list[ModelResponse]:
        q = question
        # Scenario d: multi-tool (check before a -- both mention sales)
        if "环比" in q or "增长率" in q:
            return [
                ModelResponse(
                    content="首先，我从知识库获取Q2和Q3的销售数据。",
                    tool_calls=[ToolCall(
                        name="knowledge_base",
                        arguments={"query": "Q2 Q3 销售额"},
                    )],
                ),
                ModelResponse(
                    content="已获取数据：Q2=1,420万元，Q3=1,580万元。"
                           "现在用Python计算环比增长率。",
                    tool_calls=[ToolCall(
                        name="python",
                        arguments={
                            "code": "q2=1420; q3=1580; "
                                    "growth=(q3-q2)/q2*100; "
                                    "print(f'{growth:.2f}%')",
                        },
                    )],
                ),
                ModelResponse(
                    content="增长率计算完成。现在搜索行业最新行情分析。",
                    tool_calls=[ToolCall(
                        name="web_search",
                        arguments={
                            "query": "2024年行业行情分析 销售增长趋势",
                            "max_results": 3,
                        },
                    )],
                ),
                ModelResponse(
                    content=(
                        "综合分析结果：\n\n"
                        "1. **销售数据**：Q2销售额1,420万元，Q3销售额1,580万元。\n"
                        "2. **环比增长率**：Q3环比Q2增长 **11.27%**，表现强劲。\n"
                        "3. **行业行情**：根据搜索结果，行业整体保持增长态势，"
                        "AI驱动的数字化转型是主要增长动力。"
                    ),
                ),
            ]
        # Scenario c: code execution
        if "1234" in q and "5678" in q:
            return [
                ModelResponse(
                    content="我将使用Python工具精确计算这个乘法。",
                    tool_calls=[ToolCall(
                        name="python",
                        arguments={"code": "print(1234 * 5678)"},
                    )],
                ),
                ModelResponse(content="1234 × 5678 = **7,006,652**。"),
            ]
        # Scenario b: web search
        if "AI" in q.upper() or "趋势" in q:
            return [
                ModelResponse(
                    content="这个问题需要最新信息，我将使用网页搜索工具。",
                    tool_calls=[ToolCall(
                        name="web_search",
                        arguments={"query": "2024年AI行业趋势", "max_results": 5},
                    )],
                ),
                ModelResponse(
                    content=(
                        "根据搜索结果，2024年AI行业的主要趋势包括：\n\n"
                        "1. **大语言模型**持续进化，多模态能力增强\n"
                        "2. **AI Agent**成为主流应用形态\n"
                        "3. **端侧AI**部署加速\n"
                        "4. **AI安全与治理**受到广泛关注\n"
                        "5. 开源生态蓬勃发展"
                    ),
                ),
            ]
        # Scenario a: knowledge base query
        if "销售额" in q or "销售" in q:
            return [
                ModelResponse(
                    content="我需要查询公司销售知识库来回答这个问题。",
                    tool_calls=[ToolCall(
                        name="knowledge_base",
                        arguments={"query": "2024年Q3销售额"},
                    )],
                ),
                ModelResponse(
                    content="根据公司销售文档，2024年第三季度（Q3）的"
                           "销售额为 **1,580万元**。"
                ),
            ]
        return [ModelResponse(content="(demo mode) 我暂时无法处理这个问题。")]

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        question = self._detect_question(messages)
        if question != self._question:
            self._question = question
            self._step = 0
        script = self._get_script(question)
        if self._step < len(script):
            resp = script[self._step]
        else:
            resp = ModelResponse(content="Done.")
        self._step += 1
        return resp


# --------------------------------------------------------------------------- #
# Output helpers (Rich or plain text)
# --------------------------------------------------------------------------- #


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _print_banner(text: str) -> None:
    if HAS_RICH:
        _CONSOLE.print(Panel(text, title="open-agent", border_style="cyan"))
    else:
        print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


def _print_scenario(index: int, total: int, label: str, question: str) -> None:
    title = f"Scenario {index}/{total}: {label}"
    if HAS_RICH:
        _CONSOLE.print(Panel(question, title=title, border_style="magenta"))
    else:
        print(f"\n{'─' * 60}\n{title}\nQ: {question}\n{'─' * 60}")


def _print_result(output: AgentOutput, elapsed: float, demo: bool) -> None:
    # --- Tool calls table ---
    if output.tool_calls_made:
        if HAS_RICH:
            table = Table(
                title="Tool Calls", show_header=True, header_style="bold magenta"
            )
            table.add_column("Step", style="dim", width=6)
            table.add_column("Tool", style="cyan", width=18)
            table.add_column("Arguments", style="green", width=36)
            table.add_column("Observation", style="yellow", width=48)
            for tc in output.tool_calls_made:
                table.add_row(
                    str(tc.get("step", "?")),
                    tc.get("name", "?"),
                    _truncate(str(tc.get("arguments", {})), 120),
                    _truncate(tc.get("observation", ""), 200),
                )
            _CONSOLE.print(table)
        else:
            print("\nTool Calls:")
            for tc in output.tool_calls_made:
                print(f"  Step {tc.get('step', '?')}: "
                      f"{tc.get('name', '?')}({tc.get('arguments', {})})")
                print(f"    Observation: "
                      f"{_truncate(tc.get('observation', ''), 200)}")

    # --- Thinking chain ---
    if HAS_RICH:
        tree = Tree("[bold blue]Thinking Chain[/bold blue]")
        for tc in output.tool_calls_made:
            node = tree.add(
                f"[cyan]Step {tc.get('step', '?')}: "
                f"Action → {tc.get('name', '?')}[/cyan]"
            )
            node.add(f"[green]Arguments:[/green] {tc.get('arguments', {})}")
            node.add(f"[yellow]Observation:[/yellow] "
                     f"{_truncate(tc.get('observation', ''), 150)}")
        tree.add(f"[bold green]Final Answer (step {output.steps})[/bold green]")
        _CONSOLE.print(tree)
    else:
        print("\nThinking Chain:")
        for tc in output.tool_calls_made:
            print(f"  [Step {tc.get('step', '?')}] "
                  f"Action: {tc.get('name', '?')}({tc.get('arguments', {})})")
            print(f"    → {_truncate(tc.get('observation', ''), 150)}")
        print(f"  [Final Answer @ step {output.steps}]")

    # --- Final answer ---
    if HAS_RICH:
        _CONSOLE.print(
            Panel(output.response, title="Final Answer", border_style="green")
        )
    else:
        print(f"\nFinal Answer:\n  {output.response}")

    # --- Stats ---
    mode = "demo" if demo else "real"
    stats = (f"mode={mode} | steps={output.steps} | "
             f"tool_calls={len(output.tool_calls_made)} | "
             f"time={elapsed:.2f}s")
    if HAS_RICH:
        _CONSOLE.print(f"[dim]{stats}[/dim]\n")
    else:
        print(f"  [{stats}]\n")


# --------------------------------------------------------------------------- #
# KB indexing
# --------------------------------------------------------------------------- #


async def build_knowledge_base(docs_dir: str):
    """Index sample documents into a KBManager.

    Returns ``(kb_manager, chunk_count)`` or ``(None, 0)`` if RAG
    dependencies (faiss / sentence-transformers) are not installed.
    """
    try:
        from open_agent.rag.kb_manager import KBManager
    except ImportError as exc:
        print(f"[skip] KB dependencies not installed: {exc}")
        print("       Install with: pip install 'open-agent[rag]'")
        return None, 0

    try:
        manager = KBManager()
        count = await manager.index_directory(
            docs_dir, kb_name="company_docs", description="公司文档知识库"
        )
        return manager, count
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] KB indexing failed: {exc}")
        print("       The demo will continue without a knowledge base.")
        return None, 0


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #


def build_model(settings: Settings, demo: bool) -> ModelInterface:
    """Construct the model -- mock for demo mode, real provider otherwise."""
    if demo:
        return RagDemoModel()

    provider = settings.model_provider
    if provider == "anthropic":
        from open_agent.models.anthropic_provider import AnthropicModel

        return AnthropicModel(
            api_key=settings.api_key,
            model=settings.model_name,
            timeout=settings.request_timeout,
        )
    if provider == "ollama":
        from open_agent.models.ollama_provider import OllamaModel

        return OllamaModel(
            base_url=settings.base_url,
            model=settings.model_name,
            timeout=settings.request_timeout,
        )
    from open_agent.models.openai_provider import OpenAIModel

    return OpenAIModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        timeout=settings.request_timeout,
    )


def build_agent(settings: Settings, demo: bool, kb_manager) -> Agent:
    """Build an agent with all builtin tools plus KnowledgeBaseTool."""
    model = build_model(settings, demo)

    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(PythonTool())
    registry.register(FileTool())
    registry.register(WebSearchTool())
    registry.register(KnowledgeBaseTool(kb_manager=kb_manager))

    return Agent(model=model, tool_registry=registry, max_steps=settings.max_steps)


# --------------------------------------------------------------------------- #
# Scenario runner
# --------------------------------------------------------------------------- #


async def run_scenario(
    agent: Agent,
    index: int,
    total: int,
    label: str,
    question: str,
    demo: bool,
) -> None:
    _print_scenario(index, total, label, question)
    start = time.perf_counter()
    try:
        output: AgentOutput = await agent.run(question)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        msg = f"Error: {exc}"
        if HAS_RICH:
            _CONSOLE.print(f"[red]{msg}[/red]")
        else:
            print(f"  {msg}")
        print(f"  [time={elapsed:.2f}s]\n")
        return
    elapsed = time.perf_counter() - start
    _print_result(output, elapsed, demo)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

SCENARIOS: list[tuple[str, str]] = [
    ("知识库查询", "公司2024年Q3销售额是多少？"),
    ("网页搜索", "2024年最新的AI行业趋势是什么？"),
    ("代码执行", "计算1234乘以5678的结果"),
    ("多工具协同",
     "结合公司销售文档，计算Q3环比Q2的增长率，并搜索行业最新行情分析"),
]


async def main() -> None:
    settings = get_settings()
    demo = not settings.api_key

    mode_label = (
        "DEMO (no API key)" if demo
        else f"{settings.model_provider}/{settings.model_name}"
    )
    _print_banner(f"Open Agent — Full Agentic RAG Demo\nMode: {mode_label}")

    if demo:
        print("No OPEN_AGENT_API_KEY found — running in demo mode with a mock model.")
        print("Tools execute for real; only the model reasoning is canned.")
        print("Set OPEN_AGENT_API_KEY to use a real LLM.\n")
    else:
        print(f"Using real model: {settings.model_name} @ {settings.base_url}\n")

    # 1. Create sample documents in a temp directory.
    tmp_dir = tempfile.mkdtemp(prefix="open_agent_kb_")
    print(f"Created temp directory: {tmp_dir}")
    for filename, content in SAMPLE_DOCS.items():
        path = Path(tmp_dir) / filename
        path.write_text(content, encoding="utf-8")
        print(f"  Wrote {filename} ({len(content)} chars)")

    # 2. Index documents into a knowledge base.
    print("\nIndexing documents into knowledge base...")
    kb_manager, chunk_count = await build_knowledge_base(tmp_dir)
    if kb_manager is not None:
        print(f"Indexed {chunk_count} chunks | KBs: {kb_manager.list_kbs()}")
    else:
        print("Knowledge base not available — "
              "KnowledgeBaseTool will return a notice.")

    # 3. Build agent with all tools.
    print("\nBuilding agent with tools: "
          "shell, python, file, web_search, knowledge_base")
    agent = build_agent(settings, demo, kb_manager)

    # 4. Run scenarios.
    total = len(SCENARIOS)
    print(f"\nRunning {total} demo scenarios...\n")
    for i, (label, question) in enumerate(SCENARIOS, 1):
        await run_scenario(agent, i, total, label, question, demo)

    # 5. Cleanup.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Cleaned up temp directory: {tmp_dir}")
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
