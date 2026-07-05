"""Agentic RAG: index a document with KBManager and query it.

This example shows how to:
  * Write a sample ``.txt`` document to a temp directory.
  * Index it into a knowledge base with :class:`KBManager`.
  * Run a RAG query (route -> retrieve -> context) and feed the retrieved
    context into an :class:`Agent` to answer a question.

KBManager uses a sentence-transformers embedding model, which is downloaded on
first use. If the embedding backend is unavailable, the example falls back to
the lightweight :class:`Indexer`/:class:`Retriever` (keyword overlap) so it
still runs end-to-end. A fake model is used for the final answer so no API key
is needed.

Run with:  python examples/rag_indexing.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import Message, ModelInterface, ModelResponse, ToolSchema
from open_agent.tools.registry import ToolRegistry

SAMPLE_DOCUMENT = """\
Open Agent is a general-purpose autonomous work assistant.

It follows a ReAct loop: it reasons, calls tools, observes results, then repeats
until it can answer the user directly.

The agent ships with builtin tools such as shell, python, and file operations.

RAG is supported through a KBManager that chunks documents and a Retriever that
finds the most relevant chunks by semantic similarity.
"""


class FakeModel(ModelInterface):
    """Programmable fake model that replays canned textual replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ModelResponse:
        if self._index < len(self._replies):
            text = self._replies[self._index]
        else:
            text = "Done."
        self._index += 1
        return ModelResponse(content=text)


async def index_and_query_with_kbmanager(sample_path: Path) -> str:
    """Index ``sample_path`` with KBManager and return the retrieved context.

    Raises if the embedding backend cannot be loaded; the caller falls back to
    the lightweight indexer.
    """
    from open_agent.rag.kb_manager import KBManager

    with tempfile.TemporaryDirectory() as storage_dir:
        kb_manager = KBManager(storage_dir=storage_dir)
        await kb_manager.create_kb("docs", description="sample documents")
        chunks = await kb_manager.index_file(str(sample_path), "docs")
        print(f"Indexed {chunks} chunks into KB 'docs' with KBManager.")

        query = "How does the agent's reasoning loop work?"
        result = await kb_manager.query(query, top_k=2)
        print(f"Routed KBs: {result['routed_kbs']}")
        print(f"Retrieved {len(result['chunks'])} chunks.")
        return result["context_text"]

def index_and_query_with_indexer(sample_path: Path) -> str:
    """Lightweight fallback: keyword-overlap retrieval with Indexer/Retriever."""
    from open_agent.rag.indexer import Indexer
    from open_agent.rag.retriever import Retriever

    indexer = Indexer(chunk_size=1, chunk_overlap=0)
    chunks = indexer.index_text(
        doc_id="sample",
        text=sample_path.read_text(encoding="utf-8"),
        metadata={"source": str(sample_path)},
    )
    print(f"Indexed {len(chunks)} chunks with the lightweight Indexer.")
    retriever = Retriever(indexer=indexer, top_k=2)
    relevant = retriever.retrieve("How does the agent's reasoning loop work?")
    print(f"Top {len(relevant)} chunks retrieved.")
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(relevant, 1))


async def main() -> None:
    # 1. Write a sample .txt document to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(SAMPLE_DOCUMENT)
        sample_path = Path(tmp.name)
    print(f"Sample document written to: {sample_path}")

    # 2. Index and retrieve. Try KBManager (semantic) first, fall back to the
    #    lightweight Indexer if the embedding backend is unavailable.
    try:
        context = await index_and_query_with_kbmanager(sample_path)
    except Exception as exc:  # noqa: BLE001 - fall back gracefully
        print(f"KBManager path unavailable ({type(exc).__name__}: {exc}).")
        print("Falling back to the lightweight Indexer/Retriever.")
        context = index_and_query_with_indexer(sample_path)
    finally:
        sample_path.unlink(missing_ok=True)

    # 3. Feed the retrieved context into the agent and answer.
    query = "How does the agent's reasoning loop work?"
    augmented_prompt = (
        "Use the following retrieved context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    registry = ToolRegistry()
    model = FakeModel(
        [
            "Based on the retrieved context, Open Agent follows a ReAct loop: "
            "it reasons, calls tools, observes the results, and repeats until "
            "it can answer the user directly.",
        ]
    )
    agent = Agent(model=model, tool_registry=registry, max_steps=5)
    output: AgentOutput = await agent.run(augmented_prompt)

    print("\n=== Agent Response ===")
    print(output.response)
    print(f"\nSteps taken: {output.steps}")


if __name__ == "__main__":
    asyncio.run(main())
