"""Unit tests for the ImageGen module.

Tests cover:
- Provider detection and initialization
- Generate method (mocked)
- Edit and variations methods
- Async methods
"""

from __future__ import annotations

import base64
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.imagegen import GeneratedImage, ImageGen, ImageGenProvider
from ai_infra.imagegen.models import AVAILABLE_MODELS, DEFAULT_MODELS

# Check if google.genai is available
try:
    from google import genai  # noqa: F401

    HAS_GOOGLE_GENAI = True
except ImportError:  # pragma: no cover
    HAS_GOOGLE_GENAI = False
# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_env_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_google(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Google key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_xai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only xAI key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Stability key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("STABILITY_API_KEY", "test-stability-key")
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


@pytest.fixture
def mock_env_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with only Replicate key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-replicate-token")


@pytest.fixture
def mock_env_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with no API keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)


# =============================================================================
# Provider Detection Tests
# =============================================================================


class TestProviderDetection:
    """Tests for automatic provider detection."""

    def test_detects_openai(self, mock_env_openai: None) -> None:
        """Test that OpenAI is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.OPENAI
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.OPENAI]

    def test_detects_google(self, mock_env_google: None) -> None:
        """Test that Google is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.GOOGLE
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.GOOGLE]

    def test_detects_stability(self, mock_env_stability: None) -> None:
        """Test that Stability is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.STABILITY
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.STABILITY]

    def test_detects_xai(self, mock_env_xai: None) -> None:
        """Test that xAI is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.XAI
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.XAI]

    def test_detects_replicate(self, mock_env_replicate: None) -> None:
        """Test that Replicate is detected from env var."""
        gen = ImageGen()
        assert gen.provider == ImageGenProvider.REPLICATE
        assert gen.model == DEFAULT_MODELS[ImageGenProvider.REPLICATE]

    def test_raises_without_api_key(self, mock_env_none: None) -> None:
        """Test that error is raised when no API key is found."""
        with pytest.raises(ValueError, match="No API key found"):
            ImageGen()

    def test_explicit_provider(self, mock_env_openai: None) -> None:
        """Test explicit provider selection."""
        gen = ImageGen(provider="openai")
        assert gen.provider == ImageGenProvider.OPENAI

    def test_explicit_provider_with_api_key(self, mock_env_none: None) -> None:
        """Test explicit provider with explicit API key."""
        gen = ImageGen(provider="openai", api_key="sk-explicit-key")
        assert gen.provider == ImageGenProvider.OPENAI

    def test_explicit_model(self, mock_env_openai: None) -> None:
        """Test explicit model selection."""
        gen = ImageGen(model="dall-e-2")
        assert gen.model == "dall-e-2"


# =============================================================================
# Generate Tests (Mocked)
# =============================================================================


class TestGenerateOpenAI:
    """Tests for OpenAI image generation."""

    def test_generate_returns_images(self, mock_env_openai: None) -> None:
        """Test that generate returns GeneratedImage objects."""
        gen = ImageGen()

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(url="https://example.com/image1.png", revised_prompt="A cat"),
            MagicMock(url="https://example.com/image2.png", revised_prompt="A cat"),
        ]

        with patch.object(gen, "_get_openai_client") as mock_client:
            mock_client.return_value.images.generate.return_value = mock_response

            images = gen.generate("A cat wearing a hat", n=2)

            assert len(images) == 2
            assert all(isinstance(img, GeneratedImage) for img in images)
            assert images[0].url == "https://example.com/image1.png"
            assert images[0].provider == ImageGenProvider.OPENAI

    def test_generate_passes_correct_params(self, mock_env_openai: None) -> None:
        """Test that generate passes correct parameters to OpenAI."""
        gen = ImageGen(model="dall-e-3")

        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="https://example.com/image.png")]

        with patch.object(gen, "_get_openai_client") as mock_client:
            mock_client.return_value.images.generate.return_value = mock_response

            gen.generate(
                "A sunset",
                size="1792x1024",
                n=1,
                quality="hd",
                style="vivid",
            )

            mock_client.return_value.images.generate.assert_called_once()
            call_kwargs = mock_client.return_value.images.generate.call_args[1]
            assert call_kwargs["model"] == "dall-e-3"
            assert call_kwargs["prompt"] == "A sunset"
            assert call_kwargs["size"] == "1792x1024"
            assert call_kwargs["n"] == 1
            assert call_kwargs["quality"] == "hd"
            assert call_kwargs["style"] == "vivid"

    def test_generate_retries_transient_timeout_errors(self, mock_env_openai: None) -> None:
        """Test that transient OpenAI image timeouts are retried."""
        gen = ImageGen()

        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="https://example.com/image.png")]

        with (
            patch.object(gen, "_get_openai_client") as mock_client,
            patch("time.sleep") as mock_sleep,
        ):
            mock_client.return_value.images.generate.side_effect = [
                RuntimeError("Operation did not complete within 15s"),
                mock_response,
            ]

            images = gen.generate("A cat wearing a hat")

            assert len(images) == 1
            assert images[0].url == "https://example.com/image.png"
            assert mock_client.return_value.images.generate.call_count == 2
            mock_sleep.assert_called_once_with(1.0)

    def test_generate_does_not_retry_invalid_request_errors(self, mock_env_openai: None) -> None:
        """Test that permanent OpenAI request errors are surfaced immediately."""
        gen = ImageGen()

        with (
            patch.object(gen, "_get_openai_client") as mock_client,
            patch("time.sleep") as mock_sleep,
        ):
            mock_client.return_value.images.generate.side_effect = RuntimeError(
                "Invalid 'prompt': unsupported content"
            )

            with pytest.raises(RuntimeError, match="Invalid 'prompt'"):
                gen.generate("A cat wearing a hat")

            assert mock_client.return_value.images.generate.call_count == 1
            mock_sleep.assert_not_called()

    def test_generate_gpt_image_model_decodes_b64_json(self, mock_env_openai: None) -> None:
        """Test GPT image models return decoded image bytes instead of URLs."""
        gen = ImageGen(model="gpt-image-1-mini")
        raw_image = b"\x89PNG\r\n\x1a\ngpt-image"

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(
                b64_json=base64.b64encode(raw_image).decode("ascii"),
                revised_prompt=None,
                url=None,
            )
        ]

        with patch.object(gen, "_get_openai_client") as mock_client:
            mock_client.return_value.images.generate.return_value = mock_response

            images = gen.generate("A tiny editorial cover", size="1024x1024", quality="low")

            call_kwargs = mock_client.return_value.images.generate.call_args[1]
            assert call_kwargs["model"] == "gpt-image-1-mini"
            assert "response_format" not in call_kwargs
            assert call_kwargs["quality"] == "low"
            assert images[0].data == raw_image
            assert images[0].url is None
            assert images[0].provider == ImageGenProvider.OPENAI


@pytest.mark.skipif(not HAS_GOOGLE_GENAI, reason="google-genai not installed")
class TestGenerateGoogle:
    """Tests for Google Imagen generation."""

    def test_generate_returns_images(self, mock_env_google: None) -> None:
        """Test that Google generate returns images."""
        gen = ImageGen(provider="google")
        raw_image = b"\x89PNG\r\n\x1a\nimagedata"

        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = base64.b64encode(raw_image).decode("ascii")

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]

        with patch.object(gen, "_get_google_client") as mock_client:
            mock_client.return_value.models.generate_images.return_value = mock_response

            images = gen.generate("A mountain landscape")

            mock_client.return_value.models.generate_images.assert_called_once()
            call_kwargs = mock_client.return_value.models.generate_images.call_args[1]
            assert call_kwargs["model"] == "imagen-4.0-fast-generate-001"
            assert len(images) == 1
            assert images[0].data == raw_image
            assert images[0].provider == ImageGenProvider.GOOGLE

    def test_generate_imagen_config_omits_image_size_by_default(
        self, mock_env_google: None
    ) -> None:
        """Test that Imagen config does not force sampleImageSize for wide outputs."""
        gen = ImageGen(provider="google")

        config = gen._build_google_generate_images_config("1536x864", 1)

        image_size = (
            config.get("image_size")
            if isinstance(config, dict)
            else getattr(config, "image_size", None)
        )
        aspect_ratio = (
            config.get("aspect_ratio")
            if isinstance(config, dict)
            else getattr(config, "aspect_ratio", None)
        )

        assert image_size is None
        assert aspect_ratio == "16:9"

    def test_generate_imagen_retries_without_negative_prompt_when_gemini_api_rejects_it(
        self,
        mock_env_google: None,
    ) -> None:
        """Retry Imagen generation without negative_prompt when Gemini API rejects it."""
        gen = ImageGen(provider="google")
        raw_image = b"\x89PNG\r\n\x1a\nretry-image"

        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = raw_image

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]

        with patch.object(gen, "_get_google_client") as mock_client:
            mock_client.return_value.models.generate_images.side_effect = [
                RuntimeError("negative_prompt parameter is not supported in Gemini API."),
                mock_response,
            ]

            images = gen.generate(
                "A mountain landscape",
                negative_prompt="readable text",
            )

            assert len(images) == 1
            assert images[0].data == raw_image
            assert mock_client.return_value.models.generate_images.call_count == 2

            first_config = mock_client.return_value.models.generate_images.call_args_list[0].kwargs[
                "config"
            ]
            second_config = mock_client.return_value.models.generate_images.call_args_list[
                1
            ].kwargs["config"]
            assert getattr(first_config, "negative_prompt", None) == "readable text"
            assert getattr(second_config, "negative_prompt", None) is None

    def test_generate_gemini_ignores_thought_images(self, mock_env_google: None) -> None:
        """Test Gemini image generation skips interim thought images."""
        gen = ImageGen(provider="google", model="gemini-3.1-flash-image-preview")
        final_image = b"\x89PNG\r\n\x1a\nfinal-image"

        thought_part = MagicMock()
        thought_part.thought = True
        thought_part.inline_data = MagicMock(data=b"\x89PNG\r\n\x1a\nthought-image")

        final_part = MagicMock()
        final_part.thought = False
        final_part.inline_data = MagicMock(data=final_image)

        mock_response = MagicMock()
        mock_response.parts = [thought_part, final_part]

        with patch.object(gen, "_get_google_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            images = gen.generate("A crisp infographic cover")

            assert len(images) == 1
            assert images[0].data == final_image
            assert images[0].provider == ImageGenProvider.GOOGLE


class TestGenerateXAI:
    """Tests for xAI image generation."""

    def test_generate_returns_binary_image_by_default(self, mock_env_xai: None) -> None:
        """Test xAI generation defaults to base64 output for better DX."""
        gen = ImageGen(provider="xai")
        raw_image = b"\x89PNG\r\n\x1a\nxai-image"

        mock_response = MagicMock(
            image=raw_image,
            model="grok-imagine-image",
            respect_moderation=True,
            url=None,
        )
        mock_client = MagicMock()
        mock_client.image.sample.return_value = mock_response
        mock_xai_sdk = MagicMock()
        mock_xai_sdk.Client.return_value = mock_client

        with patch.dict(sys.modules, {"xai_sdk": mock_xai_sdk}):
            images = gen.generate("A product hero shot", size="1024x1024")

        call_kwargs = mock_client.image.sample.call_args[1]
        assert call_kwargs["image_format"] == "base64"
        assert call_kwargs["aspect_ratio"] == "1:1"
        assert call_kwargs["resolution"] == "1k"
        assert len(images) == 1
        assert images[0].data == raw_image
        assert images[0].provider == ImageGenProvider.XAI
        assert images[0].metadata["respect_moderation"] is True

    def test_generate_ignores_url_property_when_base64_image_is_present(
        self,
        mock_env_xai: None,
    ) -> None:
        """Test xAI base64 responses do not fail when the URL property is unavailable."""
        gen = ImageGen(provider="xai")
        raw_image = b"\x89PNG\r\n\x1a\nbase64-xai-image"

        class _Response:
            image = raw_image
            model = "grok-imagine-image"
            respect_moderation = True

            @property
            def url(self) -> str:
                raise ValueError("Image was not returned via URL and cannot be fetched.")

        mock_client = MagicMock()
        mock_client.image.sample.return_value = _Response()
        mock_xai_sdk = MagicMock()
        mock_xai_sdk.Client.return_value = mock_client

        with patch.dict(sys.modules, {"xai_sdk": mock_xai_sdk}):
            images = gen.generate("A product hero shot", size="1024x1024")

        assert len(images) == 1
        assert images[0].data == raw_image
        assert images[0].url is None
        assert images[0].provider == ImageGenProvider.XAI

    def test_edit_uses_data_uri_input(self, mock_env_xai: None) -> None:
        """Test xAI edits are sent as JSON-friendly data URIs."""
        gen = ImageGen(provider="xai")
        source_image = b"\x89PNG\r\n\x1a\nsource-image"

        mock_response = MagicMock(
            image=b"\x89PNG\r\n\x1a\nedited-image", model="grok-imagine-image"
        )
        mock_client = MagicMock()
        mock_client.image.sample.return_value = mock_response
        mock_xai_sdk = MagicMock()
        mock_xai_sdk.Client.return_value = mock_client

        with patch.dict(sys.modules, {"xai_sdk": mock_xai_sdk}):
            images = gen.edit(source_image, "Turn this into a poster", size="1024x1536")

        call_kwargs = mock_client.image.sample.call_args[1]
        assert call_kwargs["image_url"].startswith("data:image/png;base64,")
        assert call_kwargs["image_format"] == "base64"
        assert call_kwargs["aspect_ratio"] == "2:3"
        assert images[0].data == b"\x89PNG\r\n\x1a\nedited-image"
        assert images[0].provider == ImageGenProvider.XAI

    def test_generate_preserves_raw_binary_image_bytes(self, mock_env_google: None) -> None:
        """Test that Google generate does not corrupt binary image payloads."""
        gen = ImageGen(provider="google")
        raw_image = b"\x89PNG\r\n\x1a\nraw-binary-image"

        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = raw_image

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]

        with patch.object(gen, "_get_google_client") as mock_client:
            mock_client.return_value.models.generate_images.return_value = mock_response

            images = gen.generate("A mountain landscape")

            assert len(images) == 1
            assert images[0].data == raw_image
            assert images[0].provider == ImageGenProvider.GOOGLE

    @pytest.mark.asyncio
    async def test_agenerate_preserves_raw_binary_image_bytes(self, mock_env_google: None) -> None:
        """Test that async Google generate preserves binary image payloads."""
        gen = ImageGen(provider="google")
        raw_image = b"\x89PNG\r\n\x1a\nasync-binary-image"

        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = raw_image

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        images = await gen._agenerate_google_imagen(
            mock_client,
            "A mountain landscape",
            size="1024x1024",
            n=1,
        )

        assert len(images) == 1
        assert images[0].data == raw_image
        assert images[0].provider == ImageGenProvider.GOOGLE

    @pytest.mark.asyncio
    async def test_agenerate_imagen_retries_without_negative_prompt_when_gemini_api_rejects_it(
        self,
        mock_env_google: None,
    ) -> None:
        """Async Imagen generation retries without negative_prompt on Gemini API."""
        gen = ImageGen(provider="google")
        raw_image = b"\x89PNG\r\n\x1a\nasync-retry-image"

        mock_generated_image = MagicMock()
        mock_generated_image.image.image_bytes = raw_image

        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(
            side_effect=[
                RuntimeError("negative_prompt parameter is not supported in Gemini API."),
                mock_response,
            ]
        )

        images = await gen._agenerate_google_imagen(
            mock_client,
            "A mountain landscape",
            size="1024x1024",
            n=1,
            negative_prompt="readable text",
        )

        assert len(images) == 1
        assert images[0].data == raw_image
        assert mock_client.aio.models.generate_images.call_count == 2

        first_config = mock_client.aio.models.generate_images.call_args_list[0].kwargs["config"]
        second_config = mock_client.aio.models.generate_images.call_args_list[1].kwargs["config"]
        assert getattr(first_config, "negative_prompt", None) == "readable text"
        assert getattr(second_config, "negative_prompt", None) is None


# =============================================================================
# GeneratedImage Tests
# =============================================================================


class TestGeneratedImage:
    """Tests for GeneratedImage dataclass."""

    def test_save_requires_data(self, tmp_path: Any) -> None:
        """Test that save raises error without data."""
        img = GeneratedImage(url="https://example.com/image.png")

        with pytest.raises(ValueError, match="No image data available"):
            img.save(str(tmp_path / "output.png"))

    def test_save_writes_data(self, tmp_path: Any) -> None:
        """Test that save writes image data to file."""
        img = GeneratedImage(data=b"fake image data")
        output_path = tmp_path / "output.png"

        img.save(str(output_path))

        assert output_path.read_bytes() == b"fake image data"

    @pytest.mark.asyncio
    async def test_fetch_from_url(self) -> None:
        """Test fetching image data from URL."""
        img = GeneratedImage(url="https://example.com/image.png")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.content = b"downloaded image data"
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            data = await img.fetch()

            assert data == b"downloaded image data"
            assert img.data == b"downloaded image data"

    @pytest.mark.asyncio
    async def test_fetch_returns_cached_data(self) -> None:
        """Test that fetch returns cached data if already loaded."""
        img = GeneratedImage(data=b"cached data", url="https://example.com/image.png")

        data = await img.fetch()

        assert data == b"cached data"

    @pytest.mark.asyncio
    async def test_fetch_raises_without_url(self) -> None:
        """Test that fetch raises error without URL."""
        img = GeneratedImage()

        with pytest.raises(ValueError, match="No URL available"):
            await img.fetch()


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_list_providers(self) -> None:
        """Test listing available providers."""
        providers = ImageGen.list_providers()

        assert "openai" in providers
        assert "google" in providers
        assert "xai" in providers
        assert "stability" in providers
        assert "replicate" in providers

    @patch("ai_infra.imagegen.discovery.list_models")
    def test_list_models(self, mock_list_models: MagicMock) -> None:
        """Test live model discovery for a provider."""
        mock_list_models.return_value = ["gpt-image-1.5", "gpt-image-1-mini"]

        models = ImageGen.list_models("openai", refresh=True, timeout=12.5)

        assert models == ["gpt-image-1.5", "gpt-image-1-mini"]
        mock_list_models.assert_called_once_with("openai", refresh=True, timeout=12.5)

    def test_list_known_models(self) -> None:
        """Test listing known fallback models for a provider."""
        openai_models = ImageGen.list_known_models("openai")

        assert "gpt-image-1.5" in openai_models
        assert "gpt-image-1-mini" in openai_models
        assert "dall-e-2" in openai_models
        assert "dall-e-3" in openai_models

    def test_list_known_models_google(self) -> None:
        """Test listing known Google fallback models."""
        google_models = ImageGen.list_known_models("google")

        assert "imagen-4.0-ultra-generate-001" in google_models
        assert "imagen-3.0-generate-002" in google_models
        assert "imagen-3.0-fast-generate-001" in google_models
        assert "imagen-4.0-generate-001" in google_models

    def test_list_known_models_xai(self) -> None:
        """Test listing known xAI fallback models."""
        xai_models = ImageGen.list_known_models("xai")

        assert xai_models == ["grok-imagine-image"]

    @patch("ai_infra.imagegen.discovery.list_known_models")
    def test_list_known_models_delegates(self, mock_list_known_models: MagicMock) -> None:
        """Test known model catalog helper delegates to discovery."""
        mock_list_known_models.return_value = ["grok-imagine-image"]

        models = ImageGen.list_known_models("xai")

        assert models == ["grok-imagine-image"]
        mock_list_known_models.assert_called_once_with("xai")

    @patch("ai_infra.imagegen.discovery.list_available_models")
    def test_list_available_models(self, mock_list_available_models: MagicMock) -> None:
        """Test live model discovery for a provider."""
        mock_list_available_models.return_value = ["gpt-image-1.5", "gpt-image-1-mini"]

        models = ImageGen.list_available_models("openai", refresh=True, timeout=12.5)

        assert models == ["gpt-image-1.5", "gpt-image-1-mini"]
        mock_list_available_models.assert_called_once_with(
            "openai",
            refresh=True,
            timeout=12.5,
        )

    @patch("ai_infra.imagegen.discovery.list_all_models")
    def test_list_all_models(self, mock_list_all_models: MagicMock) -> None:
        """Test live model discovery for all providers."""
        mock_list_all_models.return_value = {
            "openai": ["gpt-image-1-mini"],
            "google": ["imagen-4.0-fast-generate-001"],
        }

        models = ImageGen.list_all_models(refresh=True, timeout=8.0)

        assert models == {
            "openai": ["gpt-image-1-mini"],
            "google": ["imagen-4.0-fast-generate-001"],
        }
        mock_list_all_models.assert_called_once_with(refresh=True, timeout=8.0)

    @patch("ai_infra.imagegen.discovery.list_configured_providers")
    def test_list_configured_providers(self, mock_list_configured_providers: MagicMock) -> None:
        """Test listing configured providers."""
        mock_list_configured_providers.return_value = ["openai", "xai"]

        providers = ImageGen.list_configured_providers()

        assert providers == ["openai", "xai"]
        mock_list_configured_providers.assert_called_once_with()

    @patch("ai_infra.imagegen.discovery.is_provider_configured")
    def test_is_provider_configured(self, mock_is_provider_configured: MagicMock) -> None:
        """Test provider configuration helper."""
        mock_is_provider_configured.return_value = True

        configured = ImageGen.is_provider_configured("google")

        assert configured is True
        mock_is_provider_configured.assert_called_once_with("google")


# =============================================================================
# Edit and Variations Tests
# =============================================================================


class TestEditAndVariations:
    """Tests for edit and variations methods."""

    def test_edit_rejects_google_default_imagen_model(self, mock_env_google: None) -> None:
        """Test that Google editing requires a Gemini image model."""
        gen = ImageGen()

        with pytest.raises(NotImplementedError, match="requires a Gemini image model"):
            gen.edit(b"image data", "Make it blue")

    def test_variations_only_openai(self, mock_env_google: None) -> None:
        """Test that variations raises error for non-OpenAI providers."""
        gen = ImageGen()

        with pytest.raises(NotImplementedError, match="not supported"):
            gen.variations(b"image data")


# =============================================================================
# Model Constants Tests
# =============================================================================


class TestModelConstants:
    """Tests for model constants."""

    def test_all_providers_have_defaults(self) -> None:
        """Test that all providers have default models."""
        for provider in ImageGenProvider:
            assert provider in DEFAULT_MODELS
            assert DEFAULT_MODELS[provider] is not None

    def test_all_providers_have_available_models(self) -> None:
        """Test that all providers have available models list."""
        for provider in ImageGenProvider:
            assert provider in AVAILABLE_MODELS
            assert len(AVAILABLE_MODELS[provider]) > 0

    def test_default_model_in_available(self) -> None:
        """Test that default model is in available models."""
        for provider in ImageGenProvider:
            default = DEFAULT_MODELS[provider]
            available = AVAILABLE_MODELS[provider]
            assert default in available
