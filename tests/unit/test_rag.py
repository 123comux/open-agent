"""Tests for the RAG indexer and retriever."""
from __future__ import annotations

from open_agent.rag.indexer import Chunk, Document, Indexer
from open_agent.rag.retriever import Retriever


def test_indexer_splits_paragraphs():
    indexer = Indexer(chunk_size=1, chunk_overlap=0)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = indexer.index_text("doc1", text)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph."
    assert chunks[2].text == "Third paragraph."
    assert all(c.document_id == "doc1" for c in chunks)


def test_indexer_chunk_size_groups_paragraphs():
    indexer = Indexer(chunk_size=2, chunk_overlap=0)
    text = "P1.\n\nP2.\n\nP3."
    chunks = indexer.index_text("doc1", text)

    # step = chunk_size - overlap = 2; windows: [P1, P2], then [P3]
    assert len(chunks) == 2
    assert chunks[0].text == "P1.\n\nP2."
    assert chunks[1].text == "P3."


def test_indexer_overlap_creates_overlapping_chunks():
    indexer = Indexer(chunk_size=2, chunk_overlap=1)
    text = "P1.\n\nP2.\n\nP3."
    chunks = indexer.index_text("doc1", text)

    # step = chunk_size - overlap = 1; windows: [P1, P2], [P2, P3], [P3]
    assert len(chunks) == 3
    assert chunks[0].text == "P1.\n\nP2."
    assert chunks[1].text == "P2.\n\nP3."
    assert chunks[2].text == "P3."


def test_indexer_records_metadata_and_paragraph_start():
    indexer = Indexer()
    chunks = indexer.index_text("doc1", "Hello.\n\nWorld.", metadata={"src": "test"})
    assert chunks[0].metadata["src"] == "test"
    assert chunks[0].metadata["paragraph_start"] == 0
    assert chunks[1].metadata["paragraph_start"] == 1


def test_indexer_index_multiple_documents():
    indexer = Indexer()
    docs = [
        Document(id="a", text="alpha"),
        Document(id="b", text="beta"),
    ]
    chunks = indexer.index(docs)
    assert len(chunks) == 2
    assert {c.document_id for c in chunks} == {"a", "b"}


def test_indexer_get_chunks_and_clear():
    indexer = Indexer()
    indexer.index_text("d1", "a.\n\nb.")
    assert len(indexer.get_chunks()) == 2
    indexer.clear()
    assert indexer.get_chunks() == []


def test_indexer_empty_text_yields_no_chunks():
    indexer = Indexer()
    chunks = indexer.index_text("d1", "")
    assert chunks == []


def test_retriever_keyword_matching():
    indexer = Indexer()
    indexer.index_text("d1", "Python is a programming language.")
    indexer.index_text("d2", "The cat sat on the mat.")
    indexer.index_text("d3", "Python is great for data science.")
    retriever = Retriever(indexer, top_k=5)

    results = retriever.retrieve("Python programming")
    ids = {c.document_id for c in results}
    assert "d1" in ids
    assert "d3" in ids
    assert "d2" not in ids
    # d1 matches both tokens (overlap 2), d3 matches one (overlap 1).
    assert results[0].document_id == "d1"


def test_retriever_top_k_limit():
    indexer = Indexer()
    indexer.index_text("d1", "apple apple")
    indexer.index_text("d2", "apple")
    indexer.index_text("d3", "apple")
    retriever = Retriever(indexer, top_k=2)

    results = retriever.retrieve("apple")
    assert len(results) == 2


def test_retriever_no_overlap_returns_empty():
    indexer = Indexer()
    indexer.index_text("d1", "hello world")
    retriever = Retriever(indexer)
    assert retriever.retrieve("missing") == []


def test_retriever_empty_query_returns_empty():
    indexer = Indexer()
    indexer.index_text("d1", "hello world")
    retriever = Retriever(indexer)
    assert retriever.retrieve("") == []


def test_retriever_explicit_top_k_overrides_default():
    indexer = Indexer()
    indexer.index_text("d1", "apple apple")
    indexer.index_text("d2", "apple")
    indexer.index_text("d3", "apple")
    retriever = Retriever(indexer, top_k=5)
    results = retriever.retrieve("apple", top_k=1)
    assert len(results) == 1


async def test_retriever_aretrieve():
    indexer = Indexer()
    indexer.index_text("d1", "Python is fun.")
    retriever = Retriever(indexer)
    results = await retriever.aretrieve("Python")
    assert len(results) == 1
    assert results[0].document_id == "d1"
