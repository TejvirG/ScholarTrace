import time
from dataclasses import dataclass
from pathlib import Path

from .evaluation import EvaluationCase, EvaluationReport, EvaluationRunner
from .generation import DemoAnswerGenerator
from .ingestion import Chunker, DocumentLoader
from .models import Answer, Chunk, SearchResult
from .retrieval import HybridRetriever


@dataclass(frozen=True)
class QueryExecution:
    answer: Answer
    retrieved: list[SearchResult]
    latency_ms: float


class ResearchPipeline:
    """Application service composing the complete ScholarTrace workflow."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.retriever = HybridRetriever(chunks)
        self.generator = DemoAnswerGenerator()
        self.chunks = chunks

    @classmethod
    def from_path(cls, path: str | Path, max_words: int = 120, overlap_words: int = 20) -> "ResearchPipeline":
        loader = DocumentLoader()
        source = Path(path)
        records = loader.load_json(source) if source.is_file() else loader.load_directory(source)
        chunks = Chunker(max_words=max_words, overlap_words=overlap_words).chunk(records)
        return cls(chunks)

    def query(self, question: str, top_k: int = 5) -> QueryExecution:
        started = time.perf_counter()
        retrieved = self.retriever.search(question, top_k)
        answer = self.generator.answer(question, retrieved)
        return QueryExecution(answer, retrieved, round((time.perf_counter() - started) * 1000, 2))

    def evaluate(self, cases: list[EvaluationCase], k: int = 3) -> EvaluationReport:
        return EvaluationRunner(self.retriever).run(cases, k)
