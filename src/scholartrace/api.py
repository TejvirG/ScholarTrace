from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .evaluation import load_cases, starter_cases
from .pipeline import ResearchPipeline
from .retrieval import HybridRetriever

LAST_UPLOADED_TEXT = ""
LAST_UPLOADED_CHUNK_IDS: list[str] = []


def _resolve_project_root() -> Path:
    current = Path.cwd().resolve()
    module_file = Path(__file__).resolve()
    candidates = [current, *current.parents, module_file.parent, *module_file.parents]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "frontend").exists() and (candidate / "src" / "scholartrace").exists():
            return candidate
    return current


ROOT = _resolve_project_root()
ARXIV_PATH = ROOT / "data" / "arxiv_corpus.json"
# The app uses the real research corpus by default. The demo/sample corpus is excluded from runtime.
DATA_PATH = ARXIV_PATH if ARXIV_PATH.exists() else ROOT / "data"
FRONTEND_PATH = ROOT / "frontend"
UPLOAD_PATH = ROOT / "data" / "uploads"
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
ALLOWED_UPLOADS = {".json", ".md", ".txt", ".pdf"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

pipeline = ResearchPipeline.from_path(DATA_PATH if DATA_PATH.exists() else ROOT / "data")
EVALUATION_PATH = ROOT / "data" / "arxiv_evaluation_cases.json"

app = FastAPI(title="ScholarTrace", version="0.1.0")
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


class QueryRequest(BaseModel):
    question: str = Field(default="", max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


def serialize_result(result):
    return {
        "id": result.chunk.id,
        "title": result.chunk.title,
        "section": result.chunk.section,
        "page": result.chunk.page,
        "text": result.chunk.text,
        "source_url": result.chunk.source_url,
        "authors": result.chunk.authors,
        "year": result.chunk.year,
        "doi": result.chunk.doi,
        "source_pdf": result.chunk.source_pdf,
        "score": round(result.score, 3),
        "lexical_score": round(result.lexical_score, 3),
        "overlap_score": round(result.overlap_score, 3),
        "bm25_score": round(result.bm25_score, 3),
    }


@app.get("/", include_in_schema=False)
def home():
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "message": "API is running; no frontend bundle was found in this environment."}


@app.get("/api/health")
def health():
    summary = pipeline.index_summary()
    return {
        "status": "ok",
        "index_ready": bool(pipeline.chunks),
        "retriever_type": type(pipeline.retriever).__name__,
        "data_source": summary["data_source"],
        "chunks": summary["chunks"],
        "documents": summary["documents"],
        "uploaded_documents": len(list(UPLOAD_PATH.iterdir())),
        "chunking_config": summary["chunking_config"],
    }


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """Persist one bounded research file and add it to the active index."""
    global LAST_UPLOADED_TEXT, LAST_UPLOADED_CHUNK_IDS
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

    LAST_UPLOADED_CHUNK_IDS = [chunk.id for chunk in pipeline.chunks[-added_chunks:]]
    if suffix in {".md", ".txt", ".json"}:
        try:
            LAST_UPLOADED_TEXT = content.decode("utf-8", errors="ignore")
        except Exception:
            LAST_UPLOADED_TEXT = ""
    else:
        LAST_UPLOADED_TEXT = ""

    return {"filename": file.filename, "chunks_added": added_chunks, "total_chunks": len(pipeline.chunks), "uploaded_text_length": len(LAST_UPLOADED_TEXT)}


@app.post("/api/query")
def query(request: QueryRequest):
    global LAST_UPLOADED_TEXT, LAST_UPLOADED_CHUNK_IDS
    question = request.question.strip()
    if not question:
        question = LAST_UPLOADED_TEXT.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Provide a question or upload a file to search from.")

    if LAST_UPLOADED_CHUNK_IDS:
        scoped_chunks = [chunk for chunk in pipeline.chunks if chunk.id in LAST_UPLOADED_CHUNK_IDS]
        retriever = HybridRetriever(scoped_chunks)
        results = retriever.search(question, request.top_k)
        answer = pipeline.generator.answer(question, results, allow_fallback=True)
        execution = type("QueryExecution", (), {"answer": answer, "retrieved": results, "latency_ms": 0})()
        answer_payload = execution.answer
        results_payload = execution.retrieved
        return {
            "question": answer_payload.question,
            "answer": answer_payload.text,
            "confidence": answer_payload.confidence,
            "abstained": answer_payload.abstained,
            "citations": [serialize_result(result) for result in answer_payload.citations],
            "retrieval": [serialize_result(result) for result in results_payload],
            "latency_ms": execution.latency_ms,
            "retrieval_count": len(results_payload),
        }

    execution = pipeline.query(question, request.top_k)
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
    cases = load_cases(EVALUATION_PATH, limit=50) if EVALUATION_PATH.exists() else starter_cases()
    report = pipeline.evaluate(cases, k=5)
    return {**report.__dict__, "recall_at_k": report.recall_at_5}
