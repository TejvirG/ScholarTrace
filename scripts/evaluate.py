import json
from pathlib import Path

from scholartrace.evaluation import EvaluationRunner, load_cases, starter_cases
from scholartrace.ingestion import Chunker, DocumentLoader
from scholartrace.retrieval import HybridRetriever

root = Path(__file__).resolve().parents[1]
corpus = root / "data" / "arxiv_corpus.json"
if not corpus.exists():
    corpus = root / "data" / "sample_corpus.json"
records = DocumentLoader().load_json(corpus)
retriever = HybridRetriever(Chunker().chunk(records))
case_path = root / "data" / "arxiv_evaluation_cases.json"
cases = load_cases(case_path, limit=50) if case_path.exists() else starter_cases()
report = EvaluationRunner(retriever).run(cases, k=5)
print(json.dumps(report.__dict__, indent=2))
