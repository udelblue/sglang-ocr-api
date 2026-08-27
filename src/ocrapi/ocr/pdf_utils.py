import fitz  # PyMuPDF

from ocrapi.config import settings


def pdf_bytes_to_page_pngs(pdf_bytes: bytes, dpi: int | None = None) -> list[bytes]:
    """Rasterize each page of a PDF to PNG bytes.

    Uses PyMuPDF rather than pdf2image/poppler so the runtime image needs no
    system binaries - see the sibling doc-qa-agent-pipeline project's CLAUDE.md
    for the real-world bug that decision avoids.
    """
    dpi = dpi or settings.pdf_rasterize_dpi
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        return [page.get_pixmap(matrix=matrix).tobytes("png") for page in doc]
    finally:
        doc.close()
