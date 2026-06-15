"""Evaluator — cosine similarity scoring (TF-IDF) and optional RAGAS metrics."""

from __future__ import annotations

from typing import List, Optional

from .models import EvalResult, EvalSummary, GoldenEntry


class Evaluator:
    """Evaluates LLM answers against golden entries using TF-IDF cosine similarity."""

    def __init__(self) -> None:
        self._vectorizer = None

    def _get_vectorizer(self):
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer()
        return self._vectorizer

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text_a, text_b])
        score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return max(0.0, min(1.0, score))

    def evaluate_entry(self, entry: GoldenEntry, actual_answer: str) -> EvalResult:
        sim = self.semantic_similarity(entry.answer, actual_answer)
        return EvalResult(
            entry_id=entry.id,
            question=entry.question,
            expected_answer=entry.answer,
            actual_answer=actual_answer,
            semantic_similarity=sim,
            overall_score=sim,
        )

    def evaluate_dataset(
        self,
        entries: List[GoldenEntry],
        answers: List[str],
        dataset_name: str = "unknown",
        version: str = "unknown",
    ) -> EvalSummary:
        if len(entries) != len(answers):
            raise ValueError(
                f"Mismatch: {len(entries)} entries but {len(answers)} answers provided."
            )

        results = [
            self.evaluate_entry(entry, answer)
            for entry, answer in zip(entries, answers)
        ]

        sims = [r.semantic_similarity for r in results if r.semantic_similarity is not None]
        avg_sim: Optional[float] = (sum(sims) / len(sims)) if sims else None

        return EvalSummary(
            dataset_name=dataset_name,
            version=version,
            total_entries=len(entries),
            avg_semantic_similarity=avg_sim,
            results=results,
        )

    def ragas_evaluate(self, entries: List[GoldenEntry], answers: List[str]) -> dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "RAGAS evaluation requires additional dependencies.\n"
                "Install them with:  pip install ragas datasets"
            ) from exc

        dataset = Dataset.from_dict(
            {
                "question": [e.question for e in entries],
                "answer": answers,
                "contexts": [e.contexts for e in entries],
                "ground_truth": [e.effective_ground_truth() for e in entries],
            }
        )
        return evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
