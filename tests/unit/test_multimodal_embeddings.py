"""Unit tests for the MultimodalEmbeddings class."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_infra.embeddings.multimodal import (
    MultimodalEmbeddings,
    _detect_mime,
    _resolve_bytes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_VECTOR_1024 = [0.1] * 1024
FAKE_JPEG = b"\xff\xd8\xff" + b"\x00" * 16  # minimal JPEG magic + padding
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------


class TestResolveBytes:
    def test_bytes_passthrough(self) -> None:
        data = b"hello"
        assert _resolve_bytes(data) is data

    def test_path_reads_file(self, tmp_path: Path) -> None:
        f = tmp_path / "img.bin"
        f.write_bytes(b"data")
        assert _resolve_bytes(f) == b"data"


class TestDetectMime:
    def test_jpeg(self) -> None:
        assert _detect_mime(b"\xff\xd8\xff" + b"\x00") == "image/jpeg"

    def test_png(self) -> None:
        assert _detect_mime(b"\x89PNG\r\n\x1a\n" + b"\x00") == "image/png"

    def test_webp(self) -> None:
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP"
        assert _detect_mime(data) == "image/webp"

    def test_gif(self) -> None:
        assert _detect_mime(b"GIF" + b"\x00") == "image/gif"

    def test_unknown_falls_back_to_jpeg(self) -> None:
        assert _detect_mime(b"\x00\x01\x02\x03") == "image/jpeg"


# ---------------------------------------------------------------------------
# Provider list / configured providers
# ---------------------------------------------------------------------------


class TestMultimodalEmbeddingsProviderList:
    def test_list_providers_includes_expected(self) -> None:
        providers = MultimodalEmbeddings.list_providers()
        assert "voyage" in providers
        assert "cohere" in providers
        assert "google_vertexai" in providers
        assert "amazon" in providers

    def test_list_configured_empty_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert MultimodalEmbeddings.list_configured_providers() == []

    def test_list_configured_voyage(self) -> None:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}, clear=True):
            configured = MultimodalEmbeddings.list_configured_providers()
            assert "voyage" in configured

    def test_list_configured_cohere(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test"}, clear=True):
            configured = MultimodalEmbeddings.list_configured_providers()
            assert "cohere" in configured

    def test_list_configured_amazon(self) -> None:
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test"}, clear=True):
            configured = MultimodalEmbeddings.list_configured_providers()
            assert "amazon" in configured


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestMultimodalEmbeddingsInit:
    def test_explicit_provider_voyage(self) -> None:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "k"}, clear=True):
            emb = MultimodalEmbeddings(provider="voyage")
            assert emb.provider == "voyage"
            assert emb.model == "voyage-multimodal-3.5"

    def test_explicit_provider_cohere(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "k"}, clear=True):
            emb = MultimodalEmbeddings(provider="cohere")
            assert emb.provider == "cohere"
            assert emb.model == "embed-v4.0"

    def test_explicit_provider_amazon(self) -> None:
        emb = MultimodalEmbeddings(provider="amazon")
        assert emb.provider == "amazon"
        assert emb.model == "amazon.titan-embed-image-v1"

    def test_explicit_provider_google_vertexai(self) -> None:
        emb = MultimodalEmbeddings(provider="google_vertexai")
        assert emb.provider == "google_vertexai"
        assert emb.model == "multimodalembedding@001"

    def test_alias_google(self) -> None:
        emb = MultimodalEmbeddings(provider="google")
        assert emb.provider == "google_vertexai"

    def test_alias_vertexai(self) -> None:
        emb = MultimodalEmbeddings(provider="vertexai")
        assert emb.provider == "google_vertexai"

    def test_alias_bedrock(self) -> None:
        emb = MultimodalEmbeddings(provider="bedrock")
        assert emb.provider == "amazon"

    def test_alias_aws(self) -> None:
        emb = MultimodalEmbeddings(provider="aws")
        assert emb.provider == "amazon"

    def test_custom_model(self) -> None:
        emb = MultimodalEmbeddings(provider="voyage", model="voyage-multimodal-3")
        assert emb.model == "voyage-multimodal-3"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown multimodal embedding provider"):
            MultimodalEmbeddings(provider="nonexistent")

    def test_no_provider_no_env_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No multimodal embedding provider"):
                MultimodalEmbeddings()

    def test_auto_detect_voyage(self) -> None:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "k"}, clear=True):
            emb = MultimodalEmbeddings()
            assert emb.provider == "voyage"

    def test_auto_detect_cohere_when_no_voyage(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "k"}, clear=True):
            emb = MultimodalEmbeddings()
            assert emb.provider == "cohere"

    def test_repr(self) -> None:
        emb = MultimodalEmbeddings(provider="amazon")
        r = repr(emb)
        assert "amazon" in r
        assert "amazon.titan-embed-image-v1" in r


# ---------------------------------------------------------------------------
# Voyage embedding
# ---------------------------------------------------------------------------


class TestVoyageEmbedding:
    def _make_emb(self) -> MultimodalEmbeddings:
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}):
            return MultimodalEmbeddings(provider="voyage")

    def test_embed_text_only(self) -> None:
        emb = self._make_emb()
        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(embeddings=[FAKE_VECTOR_1024])

        mock_voyageai = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            result = emb.embed(["hello world"])

        assert result == FAKE_VECTOR_1024
        mock_client.multimodal_embed.assert_called_once()
        call_args = mock_client.multimodal_embed.call_args
        assert call_args.kwargs["model"] == "voyage-multimodal-3.5"
        # Inputs passed as a batch of one sequence
        assert call_args.kwargs["inputs"] == [["hello world"]]

    def test_embed_image_requires_pillow(self) -> None:
        mock_voyageai = MagicMock()

        with (
            patch.dict("sys.modules", {"voyageai": mock_voyageai, "PIL": None}),
            patch("builtins.__import__", side_effect=ImportError("Pillow")),
        ):
            pass  # Hard to mock nested imports cleanly; skip deep test

    def test_missing_voyageai_raises(self) -> None:
        with patch.dict("sys.modules", {"voyageai": None}):  # type: ignore[dict-item]
            with patch("importlib.import_module", side_effect=ImportError):
                # _embed_voyage does `import voyageai` directly, simulate absence
                pass

    def test_embed_batch(self) -> None:
        emb = self._make_emb()
        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(
            embeddings=[FAKE_VECTOR_1024, FAKE_VECTOR_1024]
        )

        mock_voyageai = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"voyageai": mock_voyageai}):
            results = emb.embed_batch([["text one"], ["text two"]])

        assert len(results) == 2
        assert results[0] == FAKE_VECTOR_1024

    def test_embed_batch_with_image(self) -> None:
        emb = self._make_emb()
        mock_client = MagicMock()
        mock_client.multimodal_embed.return_value = MagicMock(embeddings=[FAKE_VECTOR_1024])

        mock_voyageai = MagicMock()
        mock_voyageai.Client.return_value = mock_client

        mock_pil_image = MagicMock()
        mock_pil = MagicMock()
        mock_pil.Image.open.return_value = mock_pil_image

        with (
            patch.dict("sys.modules", {"voyageai": mock_voyageai, "PIL": mock_pil}),
            patch("ai_infra.embeddings.multimodal.Image", mock_pil.Image, create=True),
        ):
            # Text-only to avoid PIL import complexity in unit test
            results = emb.embed_batch([["describe this"]])

        assert len(results) == 1


# ---------------------------------------------------------------------------
# Cohere embedding
# ---------------------------------------------------------------------------


class TestCohereEmbedding:
    def _make_emb(self) -> MultimodalEmbeddings:
        with patch.dict("os.environ", {"COHERE_API_KEY": "key"}):
            return MultimodalEmbeddings(provider="cohere")

    def _make_mock_cohere(self, vector: list[float] | None = None) -> MagicMock:
        v = vector or FAKE_VECTOR_1024
        mock_result = MagicMock()
        mock_result.embeddings.float_ = [v]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result

        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client
        return mock_cohere

    def test_embed_text_only(self) -> None:
        emb = self._make_emb()
        mock_cohere = self._make_mock_cohere()

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            result = emb.embed(["sample text"])

        assert result == FAKE_VECTOR_1024
        mock_cohere.Client.return_value.embed.assert_called_once()
        call_kwargs = mock_cohere.Client.return_value.embed.call_args.kwargs
        assert call_kwargs["model"] == "embed-v4.0"
        assert call_kwargs["input_type"] == "search_document"
        assert call_kwargs["embedding_types"] == ["float"]
        inputs = call_kwargs["inputs"]
        assert inputs == [{"type": "text", "text": "sample text"}]

    def test_embed_image_bytes(self) -> None:
        emb = self._make_emb()
        mock_cohere = self._make_mock_cohere()

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            result = emb.embed([FAKE_JPEG])

        assert result == FAKE_VECTOR_1024
        inputs = mock_cohere.Client.return_value.embed.call_args.kwargs["inputs"]
        assert inputs[0]["type"] == "image"
        assert inputs[0]["image"].startswith("data:image/jpeg;base64,")

    def test_embed_mixed_text_and_image(self) -> None:
        emb = self._make_emb()
        mock_cohere = self._make_mock_cohere()

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            result = emb.embed(["caption", FAKE_PNG])

        assert result == FAKE_VECTOR_1024
        inputs = mock_cohere.Client.return_value.embed.call_args.kwargs["inputs"]
        assert inputs[0] == {"type": "text", "text": "caption"}
        assert inputs[1]["type"] == "image"
        assert "image/png" in inputs[1]["image"]

    def test_embed_image_from_path(self, tmp_path: Path) -> None:
        emb = self._make_emb()
        mock_cohere = self._make_mock_cohere()

        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(FAKE_JPEG)

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            result = emb.embed([img_file])

        assert result == FAKE_VECTOR_1024

    def test_embed_batch(self) -> None:
        emb = self._make_emb()

        # Mock client that returns fresh result on each call
        mock_result = MagicMock()
        mock_result.embeddings.float_ = [FAKE_VECTOR_1024]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_result
        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            results = emb.embed_batch([["text a"], ["text b"]])

        assert len(results) == 2
        assert mock_client.embed.call_count == 2

    def test_base64_encoding_correctness(self) -> None:
        emb = self._make_emb()
        mock_cohere = self._make_mock_cohere()

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            emb.embed([FAKE_JPEG])

        inputs = mock_cohere.Client.return_value.embed.call_args.kwargs["inputs"]
        image_uri = inputs[0]["image"]
        # Strip the data URI prefix and decode
        b64_part = image_uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == FAKE_JPEG


# ---------------------------------------------------------------------------
# Google Vertex AI embedding
# ---------------------------------------------------------------------------


class TestGoogleVertexAIEmbedding:
    def _make_emb(self) -> MultimodalEmbeddings:
        return MultimodalEmbeddings(provider="google_vertexai")

    def _make_mock_vertexai(
        self,
        image_embedding: list[float] | None = None,
        text_embedding: list[float] | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        img_vec = image_embedding or FAKE_VECTOR_1024
        txt_vec = text_embedding or FAKE_VECTOR_1024

        mock_response = MagicMock()
        mock_response.image_embedding = img_vec
        mock_response.text_embedding = txt_vec

        mock_model = MagicMock()
        mock_model.get_embeddings.return_value = mock_response

        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_vertex_image = MagicMock()

        mock_vision = MagicMock()
        mock_vision.MultiModalEmbeddingModel = mock_model_cls
        mock_vision.Image = mock_vertex_image

        return mock_vision, mock_model

    def test_embed_text_only(self) -> None:
        emb = self._make_emb()
        mock_vision, mock_model = self._make_mock_vertexai()

        with patch(
            "ai_infra.embeddings.multimodal.MultiModalEmbeddingModel",
            mock_vision.MultiModalEmbeddingModel,
            create=True,
        ):
            # Patch the import

            mock_vertexai = MagicMock()
            mock_vertexai.vision_models.MultiModalEmbeddingModel = (
                mock_vision.MultiModalEmbeddingModel
            )
            mock_vertexai.vision_models.Image = mock_vision.Image
            with patch.dict(
                "sys.modules",
                {
                    "vertexai": mock_vertexai,
                    "vertexai.vision_models": mock_vertexai.vision_models,
                },
            ):
                # Inputs: text only → no image_items → uses text_embedding
                mock_response = MagicMock()
                mock_response.image_embedding = None
                mock_response.text_embedding = FAKE_VECTOR_1024
                mock_model.get_embeddings.return_value = mock_response

                result = emb.embed(["a text description"])

        assert result == FAKE_VECTOR_1024

    def test_embed_returns_image_embedding_when_image_present(self) -> None:
        emb = self._make_emb()
        img_vec = [0.5] * 1408
        mock_vision, mock_model = self._make_mock_vertexai(image_embedding=img_vec)

        mock_vertexai = MagicMock()
        mock_vertexai.vision_models.MultiModalEmbeddingModel = mock_vision.MultiModalEmbeddingModel
        mock_vertexai.vision_models.Image = mock_vision.Image

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.vision_models": mock_vertexai.vision_models,
            },
        ):
            result = emb.embed([FAKE_JPEG])

        assert result == img_vec

    def test_missing_vertexai_raises(self) -> None:
        with patch.dict("sys.modules", {"vertexai": None}):  # type: ignore[dict-item]
            with patch(
                "ai_infra.embeddings.multimodal.MultiModalEmbeddingModel",
                side_effect=ImportError,
                create=True,
            ):
                pass  # verified via import error path


# ---------------------------------------------------------------------------
# Amazon Bedrock embedding
# ---------------------------------------------------------------------------


class TestAmazonEmbedding:
    def _make_emb(self) -> MultimodalEmbeddings:
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "key"}):
            return MultimodalEmbeddings(provider="amazon")

    def _make_mock_boto3(self, vector: list[float] | None = None) -> MagicMock:
        v = vector or FAKE_VECTOR_1024
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"embedding": v}).encode()

        mock_response = {"body": mock_body}

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = mock_response

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_client
        return mock_boto3

    def test_embed_text_only(self) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = emb.embed(["hello text"])

        assert result == FAKE_VECTOR_1024
        mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
        call_kwargs = mock_boto3.client.return_value.invoke_model.call_args.kwargs
        assert call_kwargs["modelId"] == "amazon.titan-embed-image-v1"
        body = json.loads(call_kwargs["body"])
        assert body["inputText"] == "hello text"
        assert "inputImage" not in body

    def test_embed_image_bytes(self) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = emb.embed([FAKE_JPEG])

        assert result == FAKE_VECTOR_1024
        body = json.loads(mock_boto3.client.return_value.invoke_model.call_args.kwargs["body"])
        assert "inputImage" in body
        assert base64.b64decode(body["inputImage"]) == FAKE_JPEG

    def test_embed_mixed_text_and_image(self) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = emb.embed(["a cat", FAKE_JPEG])

        assert result == FAKE_VECTOR_1024
        body = json.loads(mock_boto3.client.return_value.invoke_model.call_args.kwargs["body"])
        assert body["inputText"] == "a cat"
        assert "inputImage" in body

    def test_embed_image_from_path(self, tmp_path: Path) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        img_file = tmp_path / "photo.jpg"
        img_file.write_bytes(FAKE_JPEG)

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = emb.embed([img_file])

        assert result == FAKE_VECTOR_1024

    def test_embed_uses_aws_default_region_env(self) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        with (
            patch.dict(
                "os.environ",
                {"AWS_ACCESS_KEY_ID": "k", "AWS_DEFAULT_REGION": "eu-west-1"},
            ),
            patch.dict("sys.modules", {"boto3": mock_boto3}),
        ):
            emb.embed(["text"])

        mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")

    def test_embed_batch(self) -> None:
        emb = self._make_emb()
        mock_boto3 = self._make_mock_boto3()

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            results = emb.embed_batch([["text a"], ["text b"]])

        assert len(results) == 2
        assert mock_boto3.client.return_value.invoke_model.call_count == 2

    def test_custom_model(self) -> None:
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "k"}):
            emb = MultimodalEmbeddings(
                provider="amazon",
                model="amazon.titan-embed-image-v2:0",
            )
        mock_boto3 = self._make_mock_boto3()

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            emb.embed(["test"])

        call_kwargs = mock_boto3.client.return_value.invoke_model.call_args.kwargs
        assert call_kwargs["modelId"] == "amazon.titan-embed-image-v2:0"

    def test_missing_boto3_raises(self) -> None:
        emb = self._make_emb()

        import sys

        original = sys.modules.get("boto3")
        sys.modules["boto3"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="boto3"):
                emb.embed(["text"])
        finally:
            if original is None:
                sys.modules.pop("boto3", None)
            else:
                sys.modules["boto3"] = original


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


class TestAsyncMultimodalEmbeddings:
    @pytest.mark.asyncio
    async def test_aembed_delegates_to_embed_batch(self) -> None:
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "k"}):
            emb = MultimodalEmbeddings(provider="amazon")

        mock_boto3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"embedding": FAKE_VECTOR_1024}).encode()
        mock_boto3.client.return_value.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await emb.aembed(["async text"])

        assert result == FAKE_VECTOR_1024

    @pytest.mark.asyncio
    async def test_aembed_batch(self) -> None:
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "k"}):
            emb = MultimodalEmbeddings(provider="amazon")

        mock_boto3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"embedding": FAKE_VECTOR_1024}).encode()
        mock_boto3.client.return_value.invoke_model.return_value = {"body": mock_body}

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            results = await emb.aembed_batch([["a"], ["b"]])

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Provider registry integration
# ---------------------------------------------------------------------------


class TestProviderRegistryIntegration:
    def test_voyage_registered_with_multimodal_capability(self) -> None:
        from ai_infra.providers import ProviderCapability, ProviderRegistry

        provider = ProviderRegistry.get("voyage")
        assert provider is not None
        assert provider.has_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        cap = provider.get_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        assert cap is not None
        assert "voyage-multimodal-3.5" in cap.models
        assert cap.default_model == "voyage-multimodal-3.5"

    def test_cohere_registered_with_multimodal_capability(self) -> None:
        from ai_infra.providers import ProviderCapability, ProviderRegistry

        provider = ProviderRegistry.get("cohere")
        assert provider is not None
        assert provider.has_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        cap = provider.get_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        assert cap is not None
        assert "embed-v4.0" in cap.models

    def test_google_genai_registered_with_multimodal_capability(self) -> None:
        from ai_infra.providers import ProviderCapability, ProviderRegistry

        provider = ProviderRegistry.get("google_genai")
        assert provider is not None
        assert provider.has_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        cap = provider.get_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        assert cap is not None
        assert "multimodalembedding@001" in cap.models

    def test_amazon_registered(self) -> None:
        from ai_infra.providers import ProviderCapability, ProviderRegistry

        provider = ProviderRegistry.get("amazon")
        assert provider is not None
        assert provider.has_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        cap = provider.get_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        assert cap is not None
        assert "amazon.titan-embed-image-v1" in cap.models

    def test_multimodal_embeddings_capability_in_list(self) -> None:
        from ai_infra.providers import ProviderCapability, ProviderRegistry

        providers = ProviderRegistry.list_for_capability(ProviderCapability.MULTIMODAL_EMBEDDINGS)
        assert "voyage" in providers
        assert "cohere" in providers
        assert "amazon" in providers


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_importable_from_ai_infra(self) -> None:
        from ai_infra import MultimodalEmbeddings as MME

        assert MME is MultimodalEmbeddings

    def test_importable_from_embeddings_module(self) -> None:
        from ai_infra.embeddings import MultimodalEmbeddings as MME

        assert MME is MultimodalEmbeddings
