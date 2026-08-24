from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .evaluation import starter_cases
from .pipeline import ResearchPipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "sample_corpus.json"
FRONTEND_PATH = ROOT / "frontend"
UPLOAD_PATH = ROOT / "data" / "uploads"
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
ALLOWED_UPLOADS = {".json", ".md", ".txt", ".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

pipeline = ResearchPipeline.from_path(DATA_PATH)

app = FastAPI(title="ScholarTrace", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


def serialize_result(result):
    return {
        "id": result.chunk.id,
        "title": result.chunk.title,
        "section": result.chunk.section,
        "page": result.chunk.page,
        "text": result.chunk.text,
        "source_url": result.chunk.source_url,
        "score": round(result.score, 3),
        "lexical_score": round(result.lexical_score, 3),
        "overlap_score": round(result.overlap_score, 3),
        "bm25_score": round(result.bm25_score, 3),
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_PATH / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks), "documents": len({chunk.document_id for chunk in pipeline.chunks}), "uploaded_documents": len(list(UPLOAD_PATH.iterdir()))}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """Persist one bounded research file and add it to the active index."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOADS:
        raise HTTPException(status_code=415, detail="Supported files: .json, .md, .txt, .pdf")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File must be 10 MB or smaller")
    target = UPLOAD_PATH / f"{uuid4().hex}{suffix}"
    target.write_bytes(content)
    try:
        added_chunks = pipeline.add_path(target)
    except (ValueError, RuntimeError, UnicodeDecodeError) as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"filename": file.filename, "chunks_added": added_chunks, "total_chunks": len(pipeline.chunks)}


@app.post("/api/query")
def query(request: QueryRequest):
    execution = pipeline.query(request.question, request.top_k)
    answer = execution.answer
    results = execution.retrieved
    return {
        "question": answer.question,
        "answer": answer.text,
        "confidence": answer.confidence,
        "abstained": answer.abstained,
        "citations": [serialize_result(result) for result in answer.citations],
        "retrieval": [serialize_result(result) for result in results],
        "latency_ms": execution.latency_ms,
        "retrieval_count": len(results),
    }


@app.get("/api/evaluation")
def evaluation():
    report = pipeline.evaluate(starter_cases())
    return report.__dict__
