# ScholarTrace

ScholarTrace is an evidence-grounded research assistant built as a complete RAG system rather than a notebook demo. It ingests structured research notes, retrieves supporting passages, produces a cited answer, and measures retrieval quality.

## Why this is a strong portfolio project


`Retrieval supplies evidence that can be inspected and updated without retraining the generator.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn scholartrace.api:app --reload
```

Open `http://127.0.0.1:8000`. The API also exposes interactive docs at `/docs`.

Run tests:

```powershell
pytest
```

Run the evaluation report:

```powershell
python scripts\evaluate.py
```

Build the real-paper benchmark and compare retrievers:

```powershell
python scripts\collect_arxiv.py --limit 50
python scripts\build_arxiv_cases.py
python scripts\run_experiments.py
```

Read the resulting study in `experiments/REPORT.md`. The checked-in arXiv snapshot is metadata and abstract text from the public API; the experiment explicitly labels its title-derived cases as silver labels rather than human evaluation.

## Architecture

```text
Research notes -> DocumentLoader -> Chunker -> InvertedIndex
                                             |
Question -> HybridRetriever -> CitationValidator -> AnswerGenerator -> API/UI
                                             |
                                  EvaluationRunner -> metrics.json
```

The starter corpus lives in `data/sample_corpus.json`. Each record preserves a document title, section, page, and source URL. The retrieval implementation combines TF-IDF cosine similarity, BM25 term saturation, token overlap, and a diversity reranker. The `ResearchPipeline` service owns the complete workflow and reports per-query latency.

The loader also accepts a directory containing JSON, Markdown, text, or PDF files. Install PDF support with `pip install -e "[dev,pdf]"`. Local files are converted into traceable records with stable source references.

The current benchmark reports Recall@K, MRR, precision@K, and citation coverage. These are retrieval and evidence metrics, not claims about general language-model quality; a serious extension should add a human-labeled faithfulness set and compare multiple retrievers.

## Repository map

- `src/scholartrace/ingestion.py`: validated document loading and chunking.
- `src/scholartrace/retrieval.py`: deterministic hybrid retrieval index.
- `src/scholartrace/generation.py`: cited local answer generation contract.
- `src/scholartrace/evaluation.py`: benchmark cases and retrieval metrics.
- `src/scholartrace/pipeline.py`: reusable application service for API and experiments.
- `src/scholartrace/api.py`: FastAPI service and static frontend serving.
- `frontend/`: intentionally small vanilla interface for quick demos.
- `tests/`: unit and API coverage for the main behavior.

## Research extensions

1. Add dense embeddings and compare them with the lexical baseline.
2. Add a cross-encoder reranker and report latency versus Recall@K.
3. Build a human-labeled faithfulness set from 50 questions.
4. Add answer-level claim extraction and entailment checks.
5. Track experiments with MLflow or a lightweight JSONL run log.

## Limitations

The demo answer generator is extractive and intentionally avoids pretending to be a general-purpose LLM. It is a transparent baseline for measuring retrieval and citation behavior. Production use would need stronger PDF parsing, access control, prompt-injection defenses, and a reviewed model provider.
