from unittest.mock import AsyncMock

from ocrapi.config import settings
from ocrapi.graph.builder import get_compiled_graph
from ocrapi.main import app
from ocrapi.ocr.sglang_client import SGLangTimeoutError, SGLangUnavailableError


def _override_graph(ainvoke_result=None, ainvoke_side_effect=None):
    stub_graph = type("StubGraph", (), {})()
    stub_graph.ainvoke = AsyncMock(return_value=ainvoke_result, side_effect=ainvoke_side_effect)
    app.dependency_overrides[get_compiled_graph] = lambda: stub_graph
    return stub_graph


def test_ocr_route_happy_path(api_client, sample_png_bytes):
    _override_graph(
        ainvoke_result={
            "filename": "doc.png",
            "pages": [{"page": 1, "text": "hello"}],
            "full_text": "hello",
        }
    )

    response = api_client.post(
        "/api/v1/ocr", files={"file": ("doc.png", sample_png_bytes, "image/png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "doc.png"
    assert body["page_count"] == 1
    assert body["full_text"] == "hello"
    assert body["pages"] == [{"page": 1, "text": "hello"}]


def test_ocr_route_rejects_unsupported_content_type(api_client):
    _override_graph()
    response = api_client.post(
        "/api/v1/ocr", files={"file": ("doc.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400


def test_ocr_route_rejects_oversized_upload(api_client, sample_png_bytes, monkeypatch):
    _override_graph()
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)
    response = api_client.post(
        "/api/v1/ocr", files={"file": ("doc.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 413


def test_ocr_route_maps_unavailable_error(api_client, sample_png_bytes):
    _override_graph(ainvoke_side_effect=SGLangUnavailableError("unreachable"))
    response = api_client.post(
        "/api/v1/ocr", files={"file": ("doc.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 502


def test_ocr_route_maps_timeout_error(api_client, sample_png_bytes):
    _override_graph(ainvoke_side_effect=SGLangTimeoutError("timed out"))
    response = api_client.post(
        "/api/v1/ocr", files={"file": ("doc.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 504


def test_health_route(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
