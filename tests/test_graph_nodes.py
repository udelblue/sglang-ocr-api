import io

import pytest
from PIL import Image

from ocrapi.graph.builder import build_graph
from ocrapi.graph.nodes import aggregate_node, ingest_node, ocr_node
from ocrapi.ocr.sglang_client import SGLangUnavailableError


def test_ingest_node_passes_through_image(sample_png_bytes):
    state = {"content_type": "image/png", "file_bytes": sample_png_bytes}
    result = ingest_node(state)
    assert result["page_images"] == [("image/png", sample_png_bytes)]


def test_ingest_node_rasterizes_pdf_pages(sample_pdf_bytes):
    state = {"content_type": "application/pdf", "file_bytes": sample_pdf_bytes}
    result = ingest_node(state)
    assert len(result["page_images"]) == 2
    for media_type, image_bytes in result["page_images"]:
        assert media_type == "image/png"
        Image.open(io.BytesIO(image_bytes)).verify()


def test_ingest_node_rejects_unsupported_type():
    state = {"content_type": "text/plain", "file_bytes": b"not an image"}
    with pytest.raises(ValueError):
        ingest_node(state)


@pytest.mark.asyncio
async def test_ocr_node_calls_client_per_page_in_order(fake_sglang_client):
    state = {"page_images": [("image/png", b"a"), ("image/png", b"b")]}
    result = await ocr_node(state, client=fake_sglang_client)
    assert result["page_texts"] == ["page 1 text", "page 2 text"]
    assert [img for img, _ in fake_sglang_client.calls] == [b"a", b"b"]


@pytest.mark.asyncio
async def test_ocr_node_propagates_client_errors(failing_sglang_client):
    state = {"page_images": [("image/png", b"a")]}
    with pytest.raises(SGLangUnavailableError):
        await ocr_node(state, client=failing_sglang_client)


def test_aggregate_node_joins_pages():
    state = {"page_texts": ["first", "second"]}
    result = aggregate_node(state)
    assert result["pages"] == [{"page": 1, "text": "first"}, {"page": 2, "text": "second"}]
    assert result["full_text"] == "first\n\nsecond"


@pytest.mark.asyncio
async def test_full_graph_end_to_end(sample_pdf_bytes, fake_sglang_client):
    graph = build_graph(client=fake_sglang_client)
    initial_state = {
        "filename": "doc.pdf",
        "content_type": "application/pdf",
        "file_bytes": sample_pdf_bytes,
    }
    result = await graph.ainvoke(initial_state)
    assert len(result["pages"]) == 2
    assert result["full_text"] == "page 1 text\n\npage 2 text"
