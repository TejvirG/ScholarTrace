from .models import Answer, SearchResult
from .retrieval import tokenize
import re


class CitationValidator:
    """Require evidence to share meaningful terms with the user's question."""

    def validate(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        question_terms = set(tokenize(question))
        return [
            result for result in results
            if result.overlap_score > 0 and result.score >= 0.05 and question_terms
        ]

    def citation_coverage(self, question: str, results: list[SearchResult]) -> float:
        terms = set(tokenize(question))
        if not terms:
            return 0.0
        covered = set().union(*(set(tokenize(result.chunk.text)) for result in results)) if results else set()
        return round(len(terms & covered) / len(terms), 3)


class DemoAnswerGenerator:
    """Transparent extractive generator used for local demos and tests."""

    def __init__(self, validator: CitationValidator | None = None) -> None:
        self.validator = validator or CitationValidator()

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        supported = self.validator.validate(question, results)
        if not supported:
            return Answer(question, "I could not find enough evidence in the research collection.", [], 0.0, True)
        selected = supported[:3]
        sentences = []
        question_terms = set(tokenize(question))
        for index, result in enumerate(selected, start=1):
            candidates = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            best = max(candidates, key=lambda sentence: len(question_terms & set(tokenize(sentence))))
            sentences.append(f"{best} [{index}]")
        confidence = min(0.99, max(result.score for result in selected) * 1.15)
        return Answer(question, " ".join(sentences), selected, round(confidence, 3), False)
