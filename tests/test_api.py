from pathlib import Path

from fastapi.testclient import TestClient

import scholartrace.api as api

client = TestClient(api.app)


def test_project_root_resolves_from_installed_package_layout(monkeypatch, tmp_path):
    root = tmp_path / "project"
    (root / "frontend").mkdir(parents=True)
    (root / "src" / "scholartrace").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "__file__", str(root / "site-packages" / "scholartrace" / "api.py"))

    assert api._resolve_project_root() == root


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"] > 0
    assert payload["documents"] > 0
    assert payload["index_ready"] is True
    assert payload["retriever_type"] == "HybridRetriever"
    assert "data_source" in payload


def test_query_endpoint_exposes_citations():
    response = client.post("/api/query", json={"question": "What is retrieval augmented generation?", "top_k": 3})
    body = response.json()
    assert response.status_code == 200
    assert body["citations"]
    answer = body["answer"].lower()
    assert "retrieval" in answer and "generation" in answer
    assert body["latency_ms"] >= 0
    assert body["retrieval_count"] == 3


def test_query_supports_uploaded_file_without_manual_prompt():
    upload = client.post("/api/documents", files={"file": ("research-notes.md", b"A unique field note about reproducible deployment.", "text/markdown")})
    assert upload.status_code == 200

    response = client.post("/api/query", json={"top_k": 4})
    body = response.json()
    assert response.status_code == 200
    assert 1 <= body["retrieval_count"] <= 4
    assert body["answer"]


def test_uploaded_file_becomes_active_evidence_source_for_queries():
    upload = client.post("/api/documents", files={"file": ("deployment-notes.md", b"This document describes reproducible deployment and release evidence for a service. It includes rollback checks and monitoring signals.", "text/markdown")})
    assert upload.status_code == 200

    response = client.post("/api/query", json={"question": "What is the capital of France?", "top_k": 4})
    body = response.json()
    assert response.status_code == 200
    answer = body["answer"].lower()
    assert "deployment" in answer or "reproducible" in answer or "rollback" in answer or "monitoring" in answer


def test_document_upload_indexes_text_file():
    response = client.post("/api/documents", files={"file": ("field-notes.md", b"A unique field note about reproducible deployment.", "text/markdown")})
    assert response.status_code == 200
    assert response.json()["chunks_added"] == 1


def test_document_upload_rejects_unsupported_type():
    response = client.post("/api/documents", files={"file": ("notes.exe", b"not research", "application/octet-stream")})
    assert response.status_code == 415
