import json
from dataclasses import dataclass
from pathlib import Path

from .models import SearchResult
from .retrieval import HybridRetriever


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    relevant_document_ids: set[str]


@dataclass(frozen=True)
class EvaluationReport:
    cases: int
    recall_at_k: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float
    citation_coverage: float
    precision_at_k: float


class EvaluationRunner:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def run(self, cases: list[EvaluationCase], k: int = 5) -> EvaluationReport:
        if not cases:
            raise ValueError("Evaluation requires at least one case")
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        citation_coverage: list[float] = []
        precisions: list[float] = []
        recall_5: list[float] = []
        recall_10: list[float] = []
        for case in cases:
            results: list[SearchResult] = self.retriever.search(case.question, max(k, 10))
            top_k_results = results[:k]
            relevant = [result for result in top_k_results if result.chunk.document_id in case.relevant_document_ids]
            recall_5.append(float(any(result.chunk.document_id in case.relevant_document_ids for result in results[:5])))
            recall_10.append(float(any(result.chunk.document_id in case.relevant_document_ids for result in results[:10])))
            recalls.append(1.0 if relevant else 0.0)
            reciprocal_ranks.append(next((1 / (index + 1) for index, result in enumerate(top_k_results) if result.chunk.document_id in case.relevant_document_ids), 0.0))
            citation_coverage.append(len(relevant) / max(len(top_k_results), 1))
            precisions.append(len(relevant) / max(len(top_k_results), 1))
        return EvaluationReport(
            len(cases),
            round(sum(recalls) / len(recalls), 3),
            round(sum(recall_5) / len(recall_5), 3),
            round(sum(recall_10) / len(recall_10), 3),
            round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
            round(sum(citation_coverage) / len(citation_coverage), 3),
            round(sum(precisions) / len(precisions), 3),
        )


def load_cases(path: str | Path, limit: int | None = None) -> list[EvaluationCase]:
    """Load fixed question/document labels without tuning them to results."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = raw if limit is None else raw[:limit]
    return [EvaluationCase(item["question"], set(item["relevant_document_ids"])) for item in selected]


def starter_cases() -> list[EvaluationCase]:
    """Small deterministic fallback benchmark for unit tests."""
    return [
        EvaluationCase("How does retrieval augmented generation use external knowledge?", {"retrieval-001"}),
        EvaluationCase("Which metrics separate retrieval quality from answer quality?", {"retrieval-001"}),
        EvaluationCase("Why does a production ML system need observability?", {"systems-002"}),
        EvaluationCase("How do data contracts prevent upstream breakage?", {"systems-002"}),
        EvaluationCase("Why should automated evaluation include human review?", {"fairness-003"}),
    ]
