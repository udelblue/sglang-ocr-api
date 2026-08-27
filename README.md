# sglang-ocr-api

A FastAPI + LangGraph scaffold that extracts text from an uploaded PDF/image by
delegating inference to a vision-language model served by
[SGLang](https://github.com/sgl-project/sglang) over its OpenAI-compatible HTTP API.

SGLang is an inference-*serving* runtime, not a model — this app is a thin client
against whatever SGLang endpoint you point it at (`SGLANG_BASE_URL`). It never runs
model inference in-process.

## Architecture

```
POST /api/v1/ocr (multipart file upload)
        │
        ▼
  LangGraph: START -> ingest -> ocr -> aggregate -> END
        │            │          │
        │      PDF->PNG pages   one request per page to SGLang's
        │      (PyMuPDF)        /v1/chat/completions (vision message)
        ▼
  OCRResponse { filename, page_count, full_text, pages[] }
```

## Requirements

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/) for dependency management.
- An SGLang server reachable over HTTP, serving a vision-capable model
  (default assumes `Qwen/Qwen2.5-VL-7B-Instruct`).

**This dev machine has no NVIDIA GPU**, so SGLang cannot run locally here at usable
speed. Point `SGLANG_BASE_URL` at a remote GPU host or cloud instance instead — see
"Running SGLang" below.

## Setup

```bash
uv sync --dev
cp .env.example .env   # edit SGLANG_BASE_URL to point at your SGLang endpoint
```

## Run

```bash
uv run uvicorn ocrapi.main:app --reload --port 8000
```

```bash
curl -F "file=@sample.png" http://localhost:8000/api/v1/ocr
```

## Test & lint

```bash
uv run pytest
uv run ruff check .
```

All tests mock the SGLang/OpenAI client boundary — no test makes a real network call
or requires a GPU.

## Running SGLang

On a Linux host with an NVIDIA GPU + NVIDIA Container Toolkit:

```bash
docker run --gpus all -p 30000:30000 lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 30000
```

or, from this repo, on that GPU host:

```bash
docker compose --profile gpu up sglang
```

Then set `SGLANG_BASE_URL=http://<that-host>:30000/v1` in `.env` on this machine.

## Docker (this app only)

```bash
docker compose up app
```

The `sglang` compose service is defined but gated behind `profiles: ["gpu"]`, so this
command never tries to start it here. `SGLANG_BASE_URL` defaults to
`http://sglang:30000/v1` (assuming the compose-network `sglang` service), which won't
resolve on this machine — override it to point at a remote endpoint via `.env` or
`SGLANG_BASE_URL=... docker compose up app`. Hitting `/api/v1/ocr` with an unreachable
endpoint returns a clean `502`, not a hang or a raw traceback.

## Configuration

See `.env.example` for all environment variables (`SGLANG_BASE_URL`, `SGLANG_MODEL`,
`SGLANG_API_KEY`, `SGLANG_TIMEOUT_SECONDS`, `SGLANG_MAX_TOKENS`, `MAX_UPLOAD_SIZE_MB`,
`PDF_RASTERIZE_DPI`), loaded via `pydantic-settings` (`src/ocrapi/config.py`).

This intentionally differs from the sibling `doc-qa-agent-pipeline` project, which
uses a plain `@dataclass` + `python-dotenv` `Settings` — `pydantic-settings` pairs
more naturally with FastAPI/pydantic here, since pydantic is already a transitive
dependency.

## Scope

This is a minimal scaffold: no auth, no rate limiting, no job queue, no persistence,
no retry logic (add retries inside `SGLangOCRClient.extract_text` if needed later,
not as a graph node — see `src/ocrapi/graph/nodes.py` for why a retry/validation node
was left out of the graph itself).
