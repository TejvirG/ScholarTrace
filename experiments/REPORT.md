# ScholarTrace Retrieval Study

## Research question

Does combining TF-IDF cosine similarity, BM25, and token overlap improve retrieval over either lexical method alone on an academic-paper corpus?

## Dataset

- Source: arXiv API query for retrieval-augmented generation papers
- Snapshot: generated locally on 2026-08-25
- Documents: 50 abstracts
- Labels: 50 transparent silver labels generated from paper-title queries
- Important limitation: title-derived labels measure document identification, not semantic answer faithfulness

## Protocol

Each query is ranked with `top_k=3`. The experiment compares TF-IDF-only, BM25-only, and the ScholarTrace hybrid weighting. Reported metrics are Recall@3, mean reciprocal rank, precision@3, and citation coverage.

## Results

| Retriever | Recall@3 | MRR | Precision@3 | Citation coverage |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 1.000 | 1.000 | 0.800 | 0.800 |
| BM25 | 1.000 | 1.000 | 0.660 | 0.660 |
| Hybrid | 1.000 | 1.000 | 0.800 | 0.800 |

## Interpretation

The hybrid configuration matches TF-IDF on this title-retrieval task and outperforms BM25 on precision and citation coverage. Because all methods achieve perfect Recall@3 and MRR, this corpus and label construction are not sufficient to claim a general retrieval improvement.

## Next experiment

Create 50 to 100 human-written questions with multiple relevant passages per question. Split questions into development and held-out test sets, add dense embeddings and a cross-encoder reranker, measure confidence calibration and latency, and have two reviewers score answer faithfulness and citation correctness.

The machine-readable source of these results is `experiments/retrieval_ablation.json`.
