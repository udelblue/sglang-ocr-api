import base64

import openai

from ocrapi.config import settings

DEFAULT_OCR_PROMPT = (
    "Extract all text from this image exactly as it appears, preserving "
    "reading order and layout. Return only the extracted text, no commentary."
)


class SGLangUnavailableError(RuntimeError):
    """The SGLang endpoint could not be reached (connection refused, DNS failure, etc.)."""


class SGLangTimeoutError(RuntimeError):
    """The SGLang endpoint did not respond within the configured timeout."""


class SGLangOCRClient:
    """Thin wrapper over an OpenAI-compatible client pointed at an SGLang server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model or settings.sglang_model
        self._base_url = base_url or settings.sglang_base_url
        self._client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key or settings.sglang_api_key,
            timeout=timeout or settings.sglang_timeout_seconds,
        )

    async def extract_text(
        self,
        image_bytes: bytes,
        *,
        media_type: str = "image/png",
        prompt: str = DEFAULT_OCR_PROMPT,
    ) -> str:
        data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=settings.sglang_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
        except openai.APITimeoutError as exc:
            raise SGLangTimeoutError(
                f"SGLang endpoint at {self._base_url} timed out after "
                f"{settings.sglang_timeout_seconds}s"
            ) from exc
        except openai.APIConnectionError as exc:
            raise SGLangUnavailableError(
                f"Could not reach SGLang endpoint at {self._base_url}: {exc}"
            ) from exc

        return response.choices[0].message.content or ""
