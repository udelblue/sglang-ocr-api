from pydantic import BaseModel


class OCRPageResult(BaseModel):
    page: int
    text: str


class OCRResponse(BaseModel):
    filename: str
    page_count: int
    full_text: str
    pages: list[OCRPageResult]
