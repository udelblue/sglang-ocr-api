from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ocrapi.api.schemas import OCRPageResult, OCRResponse
from ocrapi.config import settings
from ocrapi.graph.builder import get_compiled_graph
from ocrapi.graph.state import OCRState
from ocrapi.ocr.sglang_client import SGLangTimeoutError, SGLangUnavailableError

router = APIRouter(prefix="/api/v1", tags=["ocr"])


@router.post("/ocr", response_model=OCRResponse)
async def run_ocr(
    file: UploadFile = File(...),
    graph=Depends(get_compiled_graph),
) -> OCRResponse:
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(413, f"File exceeds MAX_UPLOAD_SIZE_MB ({settings.max_upload_size_mb} MB)")

    initial_state: OCRState = {
        "filename": file.filename or "upload",
        "content_type": file.content_type,
        "file_bytes": file_bytes,
    }

    try:
        result = await graph.ainvoke(initial_state)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SGLangUnavailableError as exc:
        raise HTTPException(502, str(exc)) from exc
    except SGLangTimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc

    return OCRResponse(
        filename=result["filename"],
        page_count=len(result["pages"]),
        full_text=result["full_text"],
        pages=[OCRPageResult(**p) for p in result["pages"]],
    )
