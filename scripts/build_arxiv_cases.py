"""Create a transparent metadata-retrieval benchmark for the collected corpus."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    corpus = json.loads((ROOT / "data" / "arxiv_corpus.json").read_text(encoding="utf-8"))
    cases = [
        {
            "question": f"What is discussed in the paper {record['title']}?",
            "relevant_document_ids": [record["id"]],
            "label_source": "silver-title-match",
        }
        for record in corpus
    ]
    output = ROOT / "data" / "arxiv_evaluation_cases.json"
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} silver-labeled cases to {output}")


if __name__ == "__main__":
    main()