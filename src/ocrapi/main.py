from fastapi import FastAPI

from ocrapi.api.routes import router

app = FastAPI(title="SGLang OCR API")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
