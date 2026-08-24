from pathlib import Path

from scholartrace.evaluation import EvaluationRunner, starter_cases
from scholartrace.generation import DemoAnswerGenerator
from scholartrace.ingestion import Chunker, DocumentLoader
from scholartrace.retrieval import HybridRetriever

ROOT = Path(__file__).parents[1]


def build_retriever():
    records = DocumentLoader().load_json(ROOT / "data" / "sample_corpus.json")
    return HybridRetriever(Chunker().chunk(records))


def test_retrieval_returns_relevant_evidence():
    results = build_retriever().search("How should RAG retrieval be evaluated?", top_k=2)
    assert results
    assert results[0].chunk.document_id == "retrieval-001"
    assert results[0].lexical_score > 0
    assert build_retriever().search("the and what") == []


def test_generator_abstains_without_evidence():
    answer = DemoAnswerGenerator().answer("What is quantum entanglement?", build_retriever().search("quantum entanglement"))
    assert answer.abstained is True
    assert answer.citations == []


def test_starter_benchmark_is_measurable():
    report = EvaluationRunner(build_retriever()).run(starter_cases())
    assert report.cases == 5
    assert report.recall_at_k >= 0.8
    assert 0 <= report.mean_reciprocal_rank <= 1
    assert 0 <= report.citation_coverage <= 1


def test_directory_loader_supports_markdown_and_configurable_chunks(tmp_path):
    note = tmp_path / "observability.md"
    note.write_text("Logging makes failures diagnosable. " * 30, encoding="utf-8")
    records = DocumentLoader().load_directory(tmp_path)
    chunks = Chunker(max_words=10, overlap_words=2).chunk(records)
    assert records[0]["title"] == "Observability"
    assert len(chunks) > 1
    assert all(chunk.source_url.startswith("file://") for chunk in chunks)


def test_precision_uses_actual_returned_results():
    report = EvaluationRunner(build_retriever()).run(starter_cases(), k=20)
    assert report.precision_at_k <= 1
