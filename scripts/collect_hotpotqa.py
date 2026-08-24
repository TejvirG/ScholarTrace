"""Import a reproducible human-authored HotpotQA evaluation slice."""

import argparse
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import quote

URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
ROOT = Path(__file__).resolve().parents[1]


def slug(title: str) -> str:
    return "hotpot-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def collect(limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    request = urllib.request.Request(URL, headers={"User-Agent": "ScholarTrace/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        examples = json.loads(response.read())[:limit]
    paragraphs: dict[str, dict[str, object]] = {}
    cases = []
    for example in examples:
        supporting_titles = {title for title, _ in example["supporting_facts"]}
        for title, sentences in example["context"]:
            identifier = slug(title)
            paragraphs.setdefault(identifier, {
                "id": identifier,
                "title": title,
                "section": "HotpotQA context",
                "page": 1,
                "text": " ".join(sentences),
                "source_url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            })
        cases.append({
            "id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "relevant_document_ids": [slug(title) for title in supporting_titles],
            "label_source": "HotpotQA human supporting facts",
        })
    return list(paragraphs.values()), cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    corpus, cases = collect(args.limit)
    (ROOT / "data" / "hotpot_corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    (ROOT / "data" / "hotpot_evaluation_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(corpus)} paragraphs and {len(cases)} human-authored cases")


if __name__ == "__main__":
    main()