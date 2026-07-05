"""RAG evaluation metrics: Faithfulness, Answer Relevance, Context Recall.

These metrics help assess the quality of a RAG pipeline by measuring:
- **Faithfulness**: Is the answer grounded in the retrieved context? (no hallucination)
- **Answer Relevance**: Does the answer actually address the question?
- **Context Recall**: Did the retrieval find all relevant information?
- **Context Precision**: Were the retrieved chunks relevant (not noise)?

Each metric returns a score in [0.0, 1.0] and an optional explanation.

The evaluation can work in two modes:
1. **LLM-based** (default): Uses a language model to judge quality. Requires a ModelInterface.
2. **Heuristic** (fallback): Uses simple text overlap/keyword matching when no LLM is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Result of evaluating a single RAG response."""

    faithfulness: float = Field(
        ge=0.0, le=1.0, description="Is the answer grounded in the context?"
    )
    answer_relevance: float = Field(
        ge=0.0, le=1.0, description="Does the answer address the question?"
    )
    context_recall: float = Field(
        ge=0.0, le=1.0, description="Did retrieval find all relevant info?"
    )
    context_precision: float = Field(
        ge=0.0, le=1.0, description="Were retrieved chunks relevant?"
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Harmonic mean of the above.")
    explanation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RAGTestCase:
    """A single test case for RAG evaluation."""
    question: str
    expected_answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    generated_answer: str = ""
    ground_truth_contexts: list[str] = field(default_factory=list)


class RAGEvaluator:
    """Evaluates RAG pipeline quality using LLM-based or heuristic metrics.

    Args:
        model: Optional ModelInterface for LLM-based evaluation.
            When None, falls back to heuristic (text overlap) metrics.
    """

    def __init__(self, model: Any = None) -> None:
        self.model = model

    async def evaluate(self, test_case: RAGTestCase) -> EvaluationResult:
        """Evaluate a single RAG test case across all metrics."""
        if self.model:
            faithfulness = await self._llm_faithfulness(test_case)
            relevance = await self._llm_answer_relevance(test_case)
            recall = await self._llm_context_recall(test_case)
            precision = await self._llm_context_precision(test_case)
        else:
            faithfulness = self._heuristic_faithfulness(test_case)
            relevance = self._heuristic_answer_relevance(test_case)
            recall = self._heuristic_context_recall(test_case)
            precision = self._heuristic_context_precision(test_case)

        # Harmonic mean for overall score
        scores = [faithfulness, relevance, recall, precision]
        non_zero = [s for s in scores if s > 0]
        overall = 4.0 / sum(1.0 / s for s in non_zero) if non_zero else 0.0

        return EvaluationResult(
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(relevance, 4),
            context_recall=round(recall, 4),
            context_precision=round(precision, 4),
            overall_score=round(min(overall, 1.0), 4),
            explanation=self._generate_explanation(faithfulness, relevance, recall, precision),
            details={
                "method": "llm" if self.model else "heuristic",
                "num_contexts": len(test_case.retrieved_contexts),
            },
        )

    async def evaluate_batch(self, test_cases: list[RAGTestCase]) -> list[EvaluationResult]:
        """Evaluate multiple test cases."""
        results = []
        for tc in test_cases:
            results.append(await self.evaluate(tc))
        return results

    # ---- LLM-based metrics ----

    async def _llm_faithfulness(self, tc: RAGTestCase) -> float:
        """Check if the answer is grounded in the retrieved context."""
        context = "\n\n".join(tc.retrieved_contexts)
        prompt = (
            f"Given the following context and answer, determine if the answer is "
            f"fully supported by the context. Score from 0.0 to 1.0.\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Answer:\n{tc.generated_answer}\n\n"
            f"Reply with ONLY a number between 0.0 and 1.0."
        )
        score = await self._llm_score(prompt)
        return score

    async def _llm_answer_relevance(self, tc: RAGTestCase) -> float:
        """Check if the answer addresses the question."""
        prompt = (
            f"Given the following question and answer, determine how relevant "
            f"the answer is to the question. Score from 0.0 to 1.0.\n\n"
            f"Question:\n{tc.question}\n\n"
            f"Answer:\n{tc.generated_answer}\n\n"
            f"Reply with ONLY a number between 0.0 and 1.0."
        )
        score = await self._llm_score(prompt)
        return score

    async def _llm_context_recall(self, tc: RAGTestCase) -> float:
        """Check if retrieval found all relevant information."""
        if not tc.ground_truth_contexts:
            return 1.0  # No ground truth to compare against
        retrieved = "\n\n".join(tc.retrieved_contexts)
        ground_truth = "\n\n".join(tc.ground_truth_contexts)
        prompt = (
            f"Given the ground truth contexts and retrieved contexts, determine "
            f"what fraction of the ground truth information was retrieved. "
            f"Score from 0.0 to 1.0.\n\n"
            f"Ground Truth:\n{ground_truth[:1000]}\n\n"
            f"Retrieved:\n{retrieved[:1000]}\n\n"
            f"Reply with ONLY a number between 0.0 and 1.0."
        )
        score = await self._llm_score(prompt)
        return score

    async def _llm_context_precision(self, tc: RAGTestCase) -> float:
        """Check if retrieved chunks are relevant (not noise)."""
        if not tc.retrieved_contexts:
            return 0.0
        context_list = "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(tc.retrieved_contexts))
        prompt = (
            f"Given the following question and retrieved context chunks, "
            f"determine what fraction of the chunks are relevant to the question. "
            f"Score from 0.0 to 1.0.\n\n"
            f"Question:\n{tc.question}\n\n"
            f"Chunks:\n{context_list}\n\n"
            f"Reply with ONLY a number between 0.0 and 1.0."
        )
        score = await self._llm_score(prompt)
        return score

    async def _llm_score(self, prompt: str) -> float:
        """Call the LLM and parse a numeric score from the response."""
        from open_agent.models.base import Message
        try:
            response = await self.model.chat(
                [Message(role="user", content=prompt)]
            )
            text = response.content.strip()
            # Extract first number from the response
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5  # Default if parsing fails
        except Exception:
            return 0.5

    # ---- Heuristic metrics (no LLM required) ----

    def _heuristic_faithfulness(self, tc: RAGTestCase) -> float:
        """Check answer grounding using sentence-level overlap with context."""
        if not tc.retrieved_contexts or not tc.generated_answer:
            return 0.0
        context = " ".join(tc.retrieved_contexts).lower()
        answer_sentences = re.split(r"[.!?]+", tc.generated_answer)
        answer_sentences = [s.strip().lower() for s in answer_sentences if len(s.strip()) > 10]
        if not answer_sentences:
            return 0.0
        supported = 0
        for sent in answer_sentences:
            words = set(sent.split())
            if len(words) == 0:
                continue
            context_words = set(context.split())
            overlap = len(words & context_words) / len(words)
            if overlap > 0.5:
                supported += 1
        return supported / len(answer_sentences)

    def _heuristic_answer_relevance(self, tc: RAGTestCase) -> float:
        """Check answer relevance using keyword overlap."""
        if not tc.generated_answer:
            return 0.0
        q_words = set(tc.question.lower().split())
        a_words = set(tc.generated_answer.lower().split())
        if not q_words:
            return 0.0
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
            "when", "where", "who", "do", "does", "did", "can", "could", "would",
            "should", "will", "to", "of", "in", "on", "at", "for", "with", "and",
            "or", "not",
        }
        q_words -= stop_words
        a_words -= stop_words
        if not q_words:
            return 0.5
        overlap = len(q_words & a_words) / len(q_words)
        return overlap

    def _heuristic_context_recall(self, tc: RAGTestCase) -> float:
        """Check retrieval coverage using keyword overlap."""
        if not tc.ground_truth_contexts:
            return 1.0
        gt_text = " ".join(tc.ground_truth_contexts).lower()
        retrieved_text = " ".join(tc.retrieved_contexts).lower()
        gt_words = set(gt_text.split())
        retrieved_words = set(retrieved_text.split())
        if not gt_words:
            return 1.0
        return len(gt_words & retrieved_words) / len(gt_words)

    def _heuristic_context_precision(self, tc: RAGTestCase) -> float:
        """Check if retrieved chunks are relevant using question keyword overlap."""
        if not tc.retrieved_contexts:
            return 0.0
        q_words = set(tc.question.lower().split())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
            "when", "where", "who", "do", "does", "did", "can", "could", "would",
            "should", "will", "to", "of", "in", "on", "at", "for", "with", "and",
            "or", "not",
        }
        q_words -= stop_words
        if not q_words:
            return 0.5
        relevant = 0
        for ctx in tc.retrieved_contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(q_words & ctx_words) / len(q_words)
            if overlap > 0.2:
                relevant += 1
        return relevant / len(tc.retrieved_contexts)

    def _generate_explanation(self, faith: float, rel: float, recall: float, prec: float) -> str:
        """Generate a human-readable explanation of the scores."""
        parts = []
        if faith < 0.5:
            parts.append("Answer may contain hallucinations (low faithfulness)")
        elif faith > 0.8:
            parts.append("Answer is well-grounded in context (high faithfulness)")
        if rel < 0.5:
            parts.append("Answer doesn't fully address the question (low relevance)")
        elif rel > 0.8:
            parts.append("Answer directly addresses the question (high relevance)")
        if recall < 0.5:
            parts.append("Retrieval missed important information (low recall)")
        if prec < 0.5:
            parts.append("Retrieval returned irrelevant chunks (low precision)")
        if not parts:
            parts.append("All metrics are at acceptable levels")
        return "; ".join(parts) + "."
