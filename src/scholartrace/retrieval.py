import math
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Explainable lexical retrieval with optional semantic dense similarity."""

    def __init__(
        self,
        chunks: list[Chunk],
        lexical_weight: float = 0.45,
        bm25_weight: float = 0.35,
        overlap_weight: float = 0.2,
        semantic_weight: float = 0.0,
    ) -> None:
        weights = [lexical_weight, bm25_weight, overlap_weight, semantic_weight]
        if min(weights) < 0 or abs(sum(weights) - 1) > 1e-6:
            raise ValueError("retrieval weights must be non-negative and sum to 1")
        self.chunks = chunks
        self.lexical_weight = lexical_weight
        self.bm25_weight = bm25_weight
        self.overlap_weight = overlap_weight
        self.semantic_weight = semantic_weight
        self._semantic_vectorizer = None
        self._semantic_matrix = None
        self._semantic_model = None
        if not chunks:
            self._tokens = []
            self._average_length = 0
            self._idf = {}
            self._vectors = []
            self._norms = []
        else:
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
            self._init_semantic_index()

    def _init_semantic_index(self) -> None:
        if not self.chunks or self.semantic_weight == 0:
            return
        texts = [f"{chunk.title} {chunk.section} {chunk.text}" for chunk in self.chunks]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._semantic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
            self._semantic_matrix = self._semantic_vectorizer.fit_transform(texts)
            return
        try:
            self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._semantic_matrix = self._semantic_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            self._semantic_model = None
            self._semantic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
            self._semantic_matrix = self._semantic_vectorizer.fit_transform(texts)

    def _bm25(self, query_terms: set[str], index: int) -> float:
        """Score term saturation and document rarity using Okapi BM25."""
        if not self._average_length:
            return 0.0
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

    def _semantic_score(self, query: str, index: int) -> float:
        if self.semantic_weight == 0 or self._semantic_matrix is None:
            return 0.0
        if self._semantic_model is not None:
            encoded = self._semantic_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            sims = cosine_similarity(encoded, self._semantic_matrix).ravel()
            return float(sims[index]) if 0 <= index < len(sims) else 0.0
        if self._semantic_vectorizer is not None:
            query_vector = self._semantic_vectorizer.transform([query])
            sims = cosine_similarity(query_vector, self._semantic_matrix).ravel()
            return float(sims[index]) if 0 <= index < len(sims) else 0.0
        return 0.0

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 50) -> list[SearchResult]:
        """Retrieve final results from a wider hybrid candidate pool."""
        if top_k <= 0 or candidate_pool <= 0:
            raise ValueError("top_k and candidate_pool must be positive")
        if not query.strip():
            return []
        if not self.chunks:
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
            semantic = self._semantic_score(query, index)
            normalized_bm25 = bm25 / (bm25 + 1) if bm25 else 0.0
            score = (self.lexical_weight * lexical) + (self.bm25_weight * normalized_bm25) + (self.overlap_weight * overlap) + (self.semantic_weight * semantic)
            scored.append(SearchResult(chunk, score, lexical, overlap, bm25, semantic))
        scored.sort(key=lambda result: (-result.score, result.chunk.id))
        vector_ranked = sorted(scored, key=lambda result: (-result.lexical_score, result.chunk.id))
        bm25_ranked = sorted(scored, key=lambda result: (-result.bm25_score, result.chunk.id))
        semantic_ranked = sorted(scored, key=lambda result: (-result.semantic_score, result.chunk.id))
        pool = {result.chunk.id: result for result in vector_ranked[:candidate_pool]}
        pool.update({result.chunk.id: result for result in bm25_ranked[:candidate_pool]})
        pool.update({result.chunk.id: result for result in semantic_ranked[:candidate_pool]})
        candidates = sorted(pool.values(), key=lambda result: (-result.score, result.chunk.id))
        return self._diverse_rerank(candidates, min(top_k, len(candidates)))

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
