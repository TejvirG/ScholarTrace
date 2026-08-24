from dataclasses import dataclass

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
    mean_reciprocal_rank: float
    citation_coverage: float
    precision_at_k: float


class EvaluationRunner:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def run(self, cases: list[EvaluationCase], k: int = 3) -> EvaluationReport:
        if not cases:
            raise ValueError("Evaluation requires at least one case")
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        citation_coverage: list[float] = []
        precisions: list[float] = []
        for case in cases:
            results: list[SearchResult] = self.retriever.search(case.question, k)
            relevant = [result for result in results if result.chunk.document_id in case.relevant_document_ids]
            recalls.append(1.0 if relevant else 0.0)
            reciprocal_ranks.append(next((1 / (index + 1) for index, result in enumerate(results) if result.chunk.document_id in case.relevant_document_ids), 0.0))
            citation_coverage.append(len(relevant) / max(len(results), 1))
            precisions.append(len(relevant) / max(len(results), 1))
        return EvaluationReport(
            len(cases),
            round(sum(recalls) / len(recalls), 3),
            round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
            round(sum(citation_coverage) / len(citation_coverage), 3),
            round(sum(precisions) / len(precisions), 3),
        )


def starter_cases() -> list[EvaluationCase]:
    """Evaluation cases using real arXiv document IDs from the corpus."""
    return [
        EvaluationCase("What are the benefits of retrieval-augmented generation?", {"arxiv-2608-21252v1"}),
        EvaluationCase("How can RAG improve language model trustworthiness?", {"arxiv-2608-21095v1"}),
        EvaluationCase("What role does memory play in reasoning systems?", {"arxiv-2608-21265v1"}),
        EvaluationCase("How should we evaluate LLM-based systems?", {"arxiv-2608-21057v1"}),
        EvaluationCase("What are challenges in agent-based systems?", {"arxiv-2608-21156v1"}),
    ]
