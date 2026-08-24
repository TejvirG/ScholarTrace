import json
from pathlib import Path

from scholartrace.evaluation import EvaluationRunner, starter_cases
from scholartrace.ingestion import Chunker, DocumentLoader
from scholartrace.retrieval import HybridRetriever

root = Path(__file__).resolve().parents[1]
records = DocumentLoader().load_json(root / "data" / "sample_corpus.json")
retriever = HybridRetriever(Chunker().chunk(records))
report = EvaluationRunner(retriever).run(starter_cases())
print(json.dumps(report.__dict__, indent=2))
