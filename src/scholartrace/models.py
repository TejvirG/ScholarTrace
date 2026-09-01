from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    section: str
    page: int
    text: str
    source_url: str
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    source_pdf: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    lexical_score: float
    overlap_score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: list[SearchResult]
    confidence: float
    abstained: bool
