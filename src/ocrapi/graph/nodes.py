from ocrapi.graph.state import OCRState
from ocrapi.ocr.pdf_utils import pdf_bytes_to_page_pngs
from ocrapi.ocr.sglang_client import SGLangOCRClient


def ingest_node(state: OCRState) -> dict:
    content_type = state["content_type"]
    file_bytes = state["file_bytes"]

    if content_type == "application/pdf":
        page_images = [("image/png", png) for png in pdf_bytes_to_page_pngs(file_bytes)]
    elif content_type in ("image/png", "image/jpeg", "image/webp"):
        page_images = [(content_type, file_bytes)]
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    return {"page_images": page_images}


async def ocr_node(state: OCRState, client: SGLangOCRClient) -> dict:
    page_texts = [
        await client.extract_text(image_bytes, media_type=media_type)
        for media_type, image_bytes in state["page_images"]
    ]
    return {"page_texts": page_texts}


def aggregate_node(state: OCRState) -> dict:
    page_texts = state["page_texts"]
    pages = [{"page": i + 1, "text": text} for i, text in enumerate(page_texts)]
    full_text = "\n\n".join(page_texts)
    return {"pages": pages, "full_text": full_text}
