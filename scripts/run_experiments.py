"""Run reproducible retrieval ablations and write a comparison report."""

import json
from pathlib import Path

from scholartrace.evaluation import EvaluationCase, EvaluationRunner
from scholartrace.ingestion import Chunker, DocumentLoader
from scholartrace.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def load_cases() -> list[EvaluationCase]:
    case_path = ROOT / "data" / "hotpot_evaluation_cases.json"
    if not case_path.exists():
        case_path = ROOT / "data" / "arxiv_evaluation_cases.json"
    if not case_path.exists():
        case_path = ROOT / "data" / "evaluation_cases.json"
    raw = json.loads(case_path.read_text(encoding="utf-8"))
    return [EvaluationCase(item["question"], set(item["relevant_document_ids"])) for item in raw]


def main() -> None:
    corpus = ROOT / "data" / "hotpot_corpus.json"
    if not corpus.exists():
        corpus = ROOT / "data" / "arxiv_corpus.json"
    if not corpus.exists():
        corpus = ROOT / "data" / "sample_corpus.json"
    records = DocumentLoader().load_json(corpus)
    chunks = Chunker().chunk(records)
    configurations = {
        "tfidf": (1.0, 0.0, 0.0),
        "bm25": (0.0, 1.0, 0.0),
        "hybrid": (0.45, 0.35, 0.2),
    }
    report = {name: EvaluationRunner(HybridRetriever(chunks, *weights)).run(load_cases()).__dict__ for name, weights in configurations.items()}
    output = ROOT / "experiments" / "retrieval_ablation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()