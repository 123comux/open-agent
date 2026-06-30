"""Agentic RAG: index a document, retrieve relevant chunks, feed them to an agent.

This example shows how to:
  * Chunk and index a document with :class:`Indexer`.
  * Retrieve the most relevant chunks for a query with :class:`Retriever`.
  * Inject the retrieved context into the agent's user prompt and answer.

A fake model is used so the example runs with no API key.

Run with:  python examples/rag_indexing.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the local ``open_agent`` package importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_agent.agent.core import Agent, AgentOutput
from open_agent.models.base import Message, ModelInterface, ModelResponse, ToolSchema
from open_agent.rag.indexer import Indexer
from open_agent.rag.retriever import Retriever
from open_agent.tools.registry import ToolRegistry


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


# A small multi-paragraph document used as the knowledge base for this example.
SAMPLE_DOCUMENT = """\
Open Agent is a general-purpose autonomous work assistant.

It follows a ReAct loop: it reasons, calls tools, observes results, then repeats
until it can answer the user directly.

The agent ships with builtin tools such as shell, python, and file operations.

RAG is supported through an Indexer that chunks documents and a Retriever that
finds the most relevant chunks by keyword overlap.
"""


async def main() -> None:
    # 1. Index the document. With chunk_size=1 each paragraph becomes its own
    #    chunk. ``index_text`` is a convenience wrapper around ``index``.
    indexer = Indexer(chunk_size=1, chunk_overlap=0)
    chunks = indexer.index_text(
        doc_id="readme", text=SAMPLE_DOCUMENT, metadata={"source": "intro"}
    )
    print(f"Indexed {len(chunks)} chunks from the document.")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk.text!r}")

    # 2. Retrieve the chunks most relevant to a user query.
    retriever = Retriever(indexer=indexer, top_k=2)
    query = "How does the agent's reasoning loop work?"
    relevant = retriever.retrieve(query)
    print(f"\nTop {len(relevant)} chunks for query: {query!r}")
    for i, chunk in enumerate(relevant, 1):
        print(f"  [{i}] (doc={chunk.document_id}) {chunk.text!r}")

    # 3. Integrate the RAG results into the agent's context. We build an
    #    augmented prompt that includes the retrieved evidence and run the agent.
    context = "\n\n".join(
        f"[{i}] {c.text}" for i, c in enumerate(relevant, 1)
    )
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
