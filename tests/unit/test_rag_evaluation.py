"""Tests for the RAG evaluation metrics."""
from __future__ import annotations

import pytest

from open_agent.models.base import Message, ModelResponse
from open_agent.rag.evaluation import EvaluationResult, RAGEvaluator, RAGTestCase


def test_heuristic_faithfulness_perfect() -> None:
    """An answer copied from context should score high on faithfulness."""
    tc = RAGTestCase(
        question="What is the capital of France?",
        retrieved_contexts=["The capital of France is Paris."],
        generated_answer="The capital of France is Paris.",
    )
    evaluator = RAGEvaluator(model=None)
    result = evaluator._heuristic_faithfulness(tc)
    assert result >= 0.9


def test_heuristic_faithfulness_unrelated() -> None:
    """An answer unrelated to context should score low."""
    tc = RAGTestCase(
        question="What is the capital of France?",
        retrieved_contexts=["The capital of France is Paris."],
        generated_answer="The speed of light is approximately 299,792 km/s.",
    )
    evaluator = RAGEvaluator(model=None)
    result = evaluator._heuristic_faithfulness(tc)
    assert result < 0.5


def test_heuristic_answer_relevance() -> None:
    """Answer relevance uses keyword overlap with the question."""
    tc = RAGTestCase(
        question="capital France",
        generated_answer="The capital of France is Paris.",
    )
    evaluator = RAGEvaluator(model=None)
    result = evaluator._heuristic_answer_relevance(tc)
    assert result > 0.5


def test_heuristic_context_recall_with_ground_truth() -> None:
    """Recall measures ground-truth coverage."""
    tc = RAGTestCase(
        question="What is Python?",
        retrieved_contexts=["Python is a programming language created by Guido van Rossum."],
        ground_truth_contexts=["Python is a programming language."],
    )
    evaluator = RAGEvaluator(model=None)
    result = evaluator._heuristic_context_recall(tc)
    assert result >= 0.5


def test_heuristic_context_recall_no_ground_truth() -> None:
    """Without ground truth recall defaults to 1.0."""
    tc = RAGTestCase(
        question="What is Python?",
        retrieved_contexts=["Python is a programming language."],
    )
    evaluator = RAGEvaluator(model=None)
    assert evaluator._heuristic_context_recall(tc) == 1.0


def test_heuristic_context_precision() -> None:
    """Precision measures fraction of relevant retrieved chunks."""
    tc = RAGTestCase(
        question="capital France",
        retrieved_contexts=[
            "The capital of France is Paris.",
            "Dogs are popular pets.",
        ],
    )
    evaluator = RAGEvaluator(model=None)
    result = evaluator._heuristic_context_precision(tc)
    assert 0.4 < result < 0.6


def test_heuristic_context_precision_empty() -> None:
    """Precision is zero when nothing is retrieved."""
    tc = RAGTestCase(
        question="capital France",
        retrieved_contexts=[],
    )
    evaluator = RAGEvaluator(model=None)
    assert evaluator._heuristic_context_precision(tc) == 0.0


@pytest.mark.asyncio
async def test_evaluate_returns_evaluation_result() -> None:
    """The full evaluate() call returns an EvaluationResult with all fields."""
    tc = RAGTestCase(
        question="What is the capital of France?",
        expected_answer="Paris",
        retrieved_contexts=["The capital of France is Paris."],
        generated_answer="Paris",
    )
    evaluator = RAGEvaluator(model=None)
    result = await evaluator.evaluate(tc)
    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevance <= 1.0
    assert 0.0 <= result.context_recall <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.overall_score <= 1.0
    assert result.explanation
    assert result.details["method"] == "heuristic"


@pytest.mark.asyncio
async def test_evaluate_batch() -> None:
    """Batch evaluation returns one result per test case."""
    cases = [
        RAGTestCase(
            question="Q1",
            retrieved_contexts=["A1 context"],
            generated_answer="A1",
        ),
        RAGTestCase(
            question="Q2",
            retrieved_contexts=["A2 context"],
            generated_answer="A2",
        ),
    ]
    evaluator = RAGEvaluator(model=None)
    results = await evaluator.evaluate_batch(cases)
    assert len(results) == 2
    assert all(isinstance(r, EvaluationResult) for r in results)


class _MockModel:
    """Mock ModelInterface that always returns a fixed score string."""

    def __init__(self, content: str) -> None:
        self.content = content

    async def chat(self, messages: list[Message], tools: list | None = None) -> ModelResponse:
        return ModelResponse(content=self.content, tool_calls=[])

    async def stream_chat(self, messages: list[Message], tools: list | None = None):
        yield self.content


@pytest.mark.asyncio
async def test_llm_score_parses_numeric_response() -> None:
    """_llm_score extracts the first number from the LLM response."""
    evaluator = RAGEvaluator(model=_MockModel("0.85"))
    score = await evaluator._llm_score("any prompt")
    assert score == 0.85


@pytest.mark.asyncio
async def test_llm_score_clamps_out_of_range() -> None:
    """Scores outside [0, 1] are clamped."""
    evaluator = RAGEvaluator(model=_MockModel("1.5"))
    assert await evaluator._llm_score("prompt") == 1.0

    # The regex extracts the first positive number; a value below 0 is clamped.
    evaluator = RAGEvaluator(model=_MockModel("0.05"))
    assert await evaluator._llm_score("prompt") == 0.05


@pytest.mark.asyncio
async def test_llm_score_defaults_on_parse_failure() -> None:
    """When no number is found, default to 0.5."""
    evaluator = RAGEvaluator(model=_MockModel("not a number"))
    assert await evaluator._llm_score("prompt") == 0.5


@pytest.mark.asyncio
async def test_llm_metrics_use_model() -> None:
    """When a model is provided, the LLM-based path is used."""
    tc = RAGTestCase(
        question="What is the capital of France?",
        retrieved_contexts=["The capital of France is Paris."],
        generated_answer="Paris",
        ground_truth_contexts=["The capital of France is Paris."],
    )
    evaluator = RAGEvaluator(model=_MockModel("0.9"))
    result = await evaluator.evaluate(tc)
    assert result.details["method"] == "llm"
    assert result.faithfulness == 0.9
    assert result.answer_relevance == 0.9
    assert result.context_recall == 0.9
    assert result.context_precision == 0.9


def test_generate_explanation_flags_low_scores() -> None:
    """Explanation highlights problematic metrics."""
    evaluator = RAGEvaluator(model=None)
    explanation = evaluator._generate_explanation(0.3, 0.3, 0.3, 0.3)
    assert "hallucinations" in explanation
    assert "relevance" in explanation
    assert "recall" in explanation
    assert "precision" in explanation
