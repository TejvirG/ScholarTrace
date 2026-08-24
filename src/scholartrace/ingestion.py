import json
from pathlib import Path
from typing import Any

from .models import Chunk


class DocumentLoader:
    """Load validated document records from the project corpus format."""

    def load_json(self, path: str | Path) -> list[dict[str, Any]]:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Corpus must contain a JSON list")
        required = {"id", "title", "section", "page", "text", "source_url"}
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"Record {index} is missing required fields")
        return records

    def load_directory(self, directory: str | Path) -> list[dict[str, Any]]:
        """Load JSON records and plain research files from a directory."""
        root = Path(directory)
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                records.extend(self.load_json(path))
            elif path.suffix.lower() in {".md", ".txt"}:
                records.append(self._text_record(path))
            elif path.suffix.lower() == ".pdf":
                records.extend(self._pdf_records(path))
        return records

    def load_file(self, path: str | Path) -> list[dict[str, Any]]:
        """Load one supported file without scanning sibling files."""
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".json":
            return self.load_json(source)
        if suffix in {".md", ".txt"}:
            return [self._text_record(source)]
        if suffix == ".pdf":
            return self._pdf_records(source)
        raise ValueError(f"Unsupported file type: {suffix or 'none'}")

    @staticmethod
    def _text_record(path: Path) -> dict[str, Any]:
        return {
            "id": path.stem,
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
            "section": "Document",
            "page": 1,
            "text": path.read_text(encoding="utf-8"),
            "source_url": f"file://{path.resolve()}",
        }

    @staticmethod
    def _pdf_records(path: Path) -> list[dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF ingestion requires the optional 'pdf' dependency") from error
        title = path.stem.replace("_", " ").replace("-", " ").title()
        source_url = f"file://{path.resolve()}"
        return [{
            "id": f"{path.stem}-page-{page_number}",
            "title": title,
            "section": "PDF",
            "page": page_number,
            "text": page.extract_text() or "",
            "source_url": source_url,
        } for page_number, page in enumerate(PdfReader(str(path)).pages, start=1)]


class Chunker:
    """Convert source records into stable, traceable retrieval chunks."""

    def __init__(self, max_words: int = 120, overlap_words: int = 20) -> None:
        if max_words <= 0 or not 0 <= overlap_words < max_words:
            raise ValueError("overlap_words must be non-negative and smaller than max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, records: list[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for record in records:
            words = str(record["text"]).split()
            step = self.max_words - self.overlap_words
            for chunk_index, start in enumerate(range(0, max(len(words), 1), step)):
                text = " ".join(words[start : start + self.max_words]).strip()
                if not text:
                    continue
                chunks.append(Chunk(
                    id=f"{record['id']}-chunk-{chunk_index}",
                    document_id=str(record["id"]),
                    title=str(record["title"]),
                    section=str(record["section"]),
                    page=int(record["page"]),
                    text=text,
                    source_url=str(record["source_url"]),
                ))
        return chunks
