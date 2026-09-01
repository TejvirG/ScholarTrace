"""Download arXiv papers and build a page-aware ScholarTrace corpus.

The resulting directory retains the original PDFs. Extracted pages are written
as JSON records with stable metadata and are indexed by the same loader and
chunker used for user uploads.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from scholartrace.ingestion import DocumentLoader

ATOM = "http://www.w3.org/2005/Atom"


def fetch_entries(query: str, limit: int, batch_size: int = 100) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for start in range(0, limit, batch_size):
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": start,
            "max_results": min(batch_size, limit - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{params}",
            headers={"User-Agent": "ScholarTrace/0.1 (academic corpus collector)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            root = ET.fromstring(response.read())
        batch = root.findall(f"{{{ATOM}}}entry")
        for entry in batch:
            identifier = entry.findtext(f"{{{ATOM}}}id", "").rsplit("/", 1)[-1]
            published = entry.findtext(f"{{{ATOM}}}published", "")[:10]
            entries.append({
                "id": f"arxiv-{identifier.replace('.', '-')}",
                "title": " ".join(entry.findtext(f"{{{ATOM}}}title", "").split()),
                "authors": [author.findtext(f"{{{ATOM}}}name", "") for author in entry.findall(f"{{{ATOM}}}author")],
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "source_url": f"https://arxiv.org/abs/{identifier}",
                "pdf_url": f"https://arxiv.org/pdf/{identifier}",
                "published": published,
            })
        if len(batch) < min(batch_size, limit - start):
            break
        time.sleep(3)
    return entries[:limit]


def download_pdf(entry: dict[str, object], pdf_dir: Path) -> Path:
    target = pdf_dir / f"{entry['id']}.pdf"
    if not target.exists():
        request = urllib.request.Request(str(entry["pdf_url"]), headers={"User-Agent": "ScholarTrace/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        time.sleep(1)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="all:%22retrieval augmented generation%22")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_pdf_corpus.json"))
    args = parser.parse_args()
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, entry in enumerate(fetch_entries(args.query, args.limit), start=1):
        pdf_path = download_pdf(entry, args.pdf_dir)
        pages = DocumentLoader().load_file(pdf_path)
        for page in pages:
            page.update({
                "id": f"{entry['id']}-page-{page['page']}",
                "title": entry["title"],
                "authors": entry["authors"],
                "year": entry["year"],
                "source_url": entry["source_url"],
                "source_pdf": str(pdf_path),
            })
        records.extend(pages)
        if index % 25 == 0:
            print(f"Processed {index} papers / {len(records)} pages")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} extracted pages from {index if records else 0} papers to {args.output}")


if __name__ == "__main__":
    main()
