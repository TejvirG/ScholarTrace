from fastapi.testclient import TestClient

from scholartrace.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["chunks"] == 5


def test_query_endpoint_exposes_citations():
    response = client.post("/api/query", json={"question": "What is a data contract?"})
    body = response.json()
    assert response.status_code == 200
    assert body["citations"]
    assert "evidence" in body["answer"].lower() or "schema" in body["answer"].lower()
    assert body["latency_ms"] >= 0
    assert body["retrieval_count"] == 3


def test_document_upload_indexes_text_file():
    response = client.post("/api/documents", files={"file": ("field-notes.md", b"A unique field note about reproducible deployment.", "text/markdown")})
    assert response.status_code == 200
    assert response.json()["chunks_added"] == 1


def test_document_upload_rejects_unsupported_type():
    response = client.post("/api/documents", files={"file": ("notes.exe", b"not research", "application/octet-stream")})
    assert response.status_code == 415
