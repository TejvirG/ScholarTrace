"""Collect public arXiv metadata and abstracts for a reproducible benchmark.

This intentionally stores metadata and abstracts, not PDF files. The query and
limit are explicit so an experiment can be recreated and audited later.
"""

import argparse
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ATOM = "http://www.w3.org/2005/Atom"


def collect(query: str, limit: int) -> list[dict[str, object]]:
    params = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": limit, "sortBy": "submittedDate", "sortOrder": "descending"})
    request = urllib.request.Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "ScholarTrace/0.1 (academic benchmark collector)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    records = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        identifier = entry.findtext(f"{{{ATOM}}}id", "").rsplit("/", 1)[-1]
        title = " ".join(entry.findtext(f"{{{ATOM}}}title", "").split())
        abstract = " ".join(entry.findtext(f"{{{ATOM}}}summary", "").split())
        published = entry.findtext(f"{{{ATOM}}}published", "")[:10]
        records.append({
            "id": f"arxiv-{identifier.replace('.', '-')}",
            "title": title,
            "section": "Abstract",
            "page": 1,
            "text": abstract,
            "source_url": f"https://arxiv.org/abs/{identifier}",
            "published": published,
            "authors": [author.findtext(f"{{{ATOM}}}name", "") for author in entry.findall(f"{{{ATOM}}}author")],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="all:%22retrieval augmented generation%22")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_corpus.json"))
    args = parser.parse_args()
    records = collect(args.query, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()