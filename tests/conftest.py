import io

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocrapi.main import app
from ocrapi.ocr.sglang_client import SGLangUnavailableError


@pytest.fixture
def sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), "hello world")
    try:
        return doc.tobytes()
    finally:
        doc.close()


class FakeSGLangClient:
    """Records every extract_text call and returns incrementing canned text."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def extract_text(
        self, image_bytes: bytes, *, media_type: str = "image/png", prompt: str = ""
    ) -> str:
        self.calls.append((image_bytes, media_type))
        return f"page {len(self.calls)} text"


class FailingSGLangClient:
    async def extract_text(
        self, image_bytes: bytes, *, media_type: str = "image/png", prompt: str = ""
    ) -> str:
        raise SGLangUnavailableError(
            "Could not reach SGLang endpoint at http://fake:30000/v1: connection refused"
        )


@pytest.fixture
def fake_sglang_client() -> FakeSGLangClient:
    return FakeSGLangClient()


@pytest.fixture
def failing_sglang_client() -> FailingSGLangClient:
    return FailingSGLangClient()


@pytest.fixture
def api_client():
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
