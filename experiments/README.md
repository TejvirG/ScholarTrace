# Experiment Log

`run_experiments.py` compares three retrieval configurations on the labeled cases in `data/evaluation_cases.json`:

- `tfidf`: cosine similarity only
- `bm25`: Okapi BM25 only
- `hybrid`: TF-IDF, BM25, and token overlap

The script writes the machine-readable result to `experiments/retrieval_ablation.json`. Run `scripts/collect_hotpotqa.py --limit 100` to use human-authored questions and gold supporting paragraphs from HotpotQA. The HotpotQA dataset is distributed under CC BY-SA 4.0; retain its attribution when publishing results. The arXiv title-derived set remains available as a silver-label fallback. Do not compare scores across corpus versions without recording the query, date, source count, and configuration.