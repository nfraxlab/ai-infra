"""Regression tests for Google image generation timeouts."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.imagegen.imagegen import _GOOGLE_IMAGE_REQUEST_TIMEOUT_MS, ImageGen
from ai_infra.imagegen.models import ImageGenProvider


def _google_genai_modules(client_ctor: MagicMock) -> dict[str, object]:
    fake_types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    fake_genai = SimpleNamespace(Client=client_ctor, types=fake_types)
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai
    return {"google": fake_google, "google.genai": fake_genai}


def test_google_gemini_generation_uses_extended_request_timeout() -> None:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=b"generated-image"))
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response
    client_ctor = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, _google_genai_modules(client_ctor)):
        imagegen = ImageGen(
            provider="google",
            model="gemini-2.5-flash-image",
            api_key="test-google-key",
        )
        images = imagegen.generate("Generate a note cover")

    client_ctor.assert_called_once_with(
        api_key="test-google-key",
        http_options={"timeout": _GOOGLE_IMAGE_REQUEST_TIMEOUT_MS},
    )
    fake_client.models.generate_content.assert_called_once()
    assert images[0].provider == ImageGenProvider.GOOGLE
    assert images[0].data == b"generated-image"


@pytest.mark.asyncio
async def test_async_google_gemini_generation_uses_extended_request_timeout() -> None:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=b"generated-image"))
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    fake_client = MagicMock()
    fake_client.aio = SimpleNamespace(
        models=SimpleNamespace(generate_content=AsyncMock(return_value=response))
    )
    client_ctor = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, _google_genai_modules(client_ctor)):
        imagegen = ImageGen(
            provider="google",
            model="gemini-2.5-flash-image",
            api_key="test-google-key",
        )
        images = await imagegen.agenerate("Generate a note cover")

    client_ctor.assert_called_once_with(
        api_key="test-google-key",
        http_options={"timeout": _GOOGLE_IMAGE_REQUEST_TIMEOUT_MS},
    )
    fake_client.aio.models.generate_content.assert_awaited_once()
    assert images[0].provider == ImageGenProvider.GOOGLE
    assert images[0].data == b"generated-image"
