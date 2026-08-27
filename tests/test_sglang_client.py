import base64
from unittest.mock import AsyncMock

import openai
import pytest

from ocrapi.ocr.sglang_client import SGLangOCRClient, SGLangTimeoutError, SGLangUnavailableError


def _make_response(text: str = "extracted text"):
    message = type("Message", (), {"content": text})
    choice = type("Choice", (), {"message": message})
    return type("Response", (), {"choices": [choice]})()


@pytest.mark.asyncio
async def test_extract_text_builds_expected_payload():
    client = SGLangOCRClient(base_url="http://fake:30000/v1", model="test-model")
    client._client.chat.completions.create = AsyncMock(return_value=_make_response("hello"))

    result = await client.extract_text(b"png-bytes", media_type="image/png")

    assert result == "hello"
    _, kwargs = client._client.chat.completions.create.call_args
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    expected_prefix = f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')[:10]}"
    assert content[1]["image_url"]["url"].startswith(expected_prefix)


@pytest.mark.asyncio
async def test_connection_error_wrapped():
    client = SGLangOCRClient(base_url="http://fake:30000/v1")
    client._client.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=None)
    )

    with pytest.raises(SGLangUnavailableError):
        await client.extract_text(b"bytes")


@pytest.mark.asyncio
async def test_timeout_error_wrapped():
    client = SGLangOCRClient(base_url="http://fake:30000/v1")
    client._client.chat.completions.create = AsyncMock(
        side_effect=openai.APITimeoutError(request=None)
    )

    with pytest.raises(SGLangTimeoutError):
        await client.extract_text(b"bytes")
