import math
import re
from collections import Counter

from .models import Chunk, SearchResult

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{2,}")
_STOP_WORDS = {"the", "and", "that", "with", "from", "this", "what", "are", "for", "into", "does"}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if token in {"evaluate", "evaluated", "evaluating", "evaluation"}:
            token = "evaluate"
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        tokens.append(token)
    return tokens


class HybridRetriever:
    """Small, explainable TF-IDF plus token-overlap retrieval baseline."""

    def __init__(self, chunks: list[Chunk], lexical_weight: float = 0.45, bm25_weight: float = 0.35, overlap_weight: float = 0.2) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one chunk")
        if min(lexical_weight, bm25_weight, overlap_weight) < 0 or abs(lexical_weight + bm25_weight + overlap_weight - 1) > 1e-6:
            raise ValueError("retrieval weights must be non-negative and sum to 1")
        self.chunks = chunks
        self.lexical_weight = lexical_weight
        self.bm25_weight = bm25_weight
        self.overlap_weight = overlap_weight
        self._tokens = [Counter(tokenize(f"{chunk.title} {chunk.section} {chunk.text}")) for chunk in chunks]
        self._average_length = sum(sum(tokens.values()) for tokens in self._tokens) / len(chunks)
        document_frequency = Counter()
        for tokens in self._tokens:
            document_frequency.update(tokens.keys())
        self._idf = {
            term: math.log((1 + len(chunks)) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._vector(tokens) for tokens in self._tokens]
        self._norms = [self._norm(vector) for vector in self._vectors]

    def _bm25(self, query_terms: set[str], index: int) -> float:
        """Score term saturation and document rarity using Okapi BM25."""
        k1, b = 1.5, 0.75
        length = sum(self._tokens[index].values()) or 1
        score = 0.0
        for term in query_terms:
            frequency = self._tokens[index].get(term, 0)
            if not frequency:
                continue
            idf = self._idf.get(term, 1.0)
            denominator = frequency + k1 * (1 - b + b * length / self._average_length)
            score += idf * frequency * (k1 + 1) / denominator
        return score

    def _vector(self, tokens: Counter[str]) -> dict[str, float]:
        return {term: count * self._idf.get(term, 1.0) for term, count in tokens.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values())) or 1.0

    def _cosine(self, query: dict[str, float], query_norm: float, index: int) -> float:
        product = sum(value * self._vectors[index].get(term, 0.0) for term, value in query.items())
        return product / (query_norm * self._norms[index])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        query_vector = self._vector(query_tokens)
        query_norm = self._norm(query_vector)
        query_terms = set(query_tokens)
        scored: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            chunk_terms = set(self._tokens[index])
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            lexical = self._cosine(query_vector, query_norm, index)
            bm25 = self._bm25(query_terms, index)
            normalized_bm25 = bm25 / (bm25 + 1) if bm25 else 0.0
            score = (self.lexical_weight * lexical) + (self.bm25_weight * normalized_bm25) + (self.overlap_weight * overlap)
            scored.append(SearchResult(chunk, score, lexical, overlap, bm25))
        scored.sort(key=lambda result: (-result.score, result.chunk.id))
        return self._diverse_rerank(scored, max(1, top_k))

    @staticmethod
    def _diverse_rerank(results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Avoid spending every result slot on the same document."""
        selected: list[SearchResult] = []
        remaining = list(results)
        while remaining and len(selected) < top_k:
            candidate = max(
                remaining,
                key=lambda result: result.score - (0.12 if any(result.chunk.document_id == item.chunk.document_id for item in selected) else 0),
            )
            selected.append(candidate)
            remaining.remove(candidate)
        return selected
