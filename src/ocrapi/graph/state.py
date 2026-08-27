from typing import Any, TypedDict


class OCRState(TypedDict, total=False):
    filename: str
    content_type: str
    file_bytes: bytes
    page_images: list[tuple[str, bytes]]  # (media_type, image_bytes) per page, in order
    page_texts: list[str]  # OCR text per page, aligned with page_images
    pages: list[dict[str, Any]]  # [{"page": 1, "text": ...}, ...]
    full_text: str
