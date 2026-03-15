"""Provider-agnostic multimodal embeddings (text + images).

Supports Voyage AI, Cohere, Google Vertex AI, and Amazon Bedrock.
Users never need to import provider SDKs directly.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

# A single item in a multimodal sequence: str for text, bytes/Path for an image
MultimodalItem = str | bytes | Path

# An ordered sequence of text and/or image items to embed as one vector
MultimodalInput = list[MultimodalItem]

# Provider-specific metadata (env key, default model, etc.)
_PROVIDER_INFO: dict[str, dict[str, Any]] = {
    "voyage": {
        "env_key": "VOYAGE_API_KEY",
        "default_model": "voyage-multimodal-3.5",
        "install_hint": "pip install voyageai Pillow",
    },
    "cohere": {
        "env_key": "COHERE_API_KEY",
        "default_model": "embed-v4.0",
        "install_hint": "pip install cohere",
    },
    "google_vertexai": {
        "env_key": "GOOGLE_APPLICATION_CREDENTIALS",
        "default_model": "multimodalembedding@001",
        "install_hint": "pip install google-cloud-aiplatform",
    },
    "amazon": {
        "env_key": "AWS_ACCESS_KEY_ID",
        "default_model": "amazon.titan-embed-image-v1",
        "install_hint": "pip install boto3",
    },
}

_PROVIDER_PRIORITY = ["voyage", "cohere", "google_vertexai", "amazon"]

_PROVIDER_ALIASES: dict[str, str] = {
    "google": "google_vertexai",
    "vertexai": "google_vertexai",
    "bedrock": "amazon",
    "aws": "amazon",
}


def _resolve_bytes(item: bytes | Path) -> bytes:
    """Return raw bytes for an image item (reads file if Path)."""
    if isinstance(item, Path):
        return item.read_bytes()
    return item


def _detect_mime(data: bytes) -> str:
    """Infer image MIME type from magic bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] in (b"GIF",):
        return "image/gif"
    return "image/jpeg"  # safe fallback


class MultimodalEmbeddings:
    """Provider-agnostic embeddings for mixed text and image inputs.

    Generates a single embedding vector from an ordered sequence of text
    strings and/or images. Supports interleaved content (e.g. caption +
    image + follow-up text) where the provider supports it.

    Supported providers:
        - voyage: Voyage AI voyage-multimodal-3.5 (single-backbone, best RAG)
        - cohere: Cohere embed-v4.0 (128K context, multilingual)
        - google_vertexai: Google multimodalembedding@001 (Vertex AI)
        - amazon: Amazon Titan image embeddings (AWS Bedrock)

    Requires at least one of: VOYAGE_API_KEY, COHERE_API_KEY,
    GOOGLE_APPLICATION_CREDENTIALS, or AWS_ACCESS_KEY_ID.

    Example:
        ```python
        from pathlib import Path
        from ai_infra import MultimodalEmbeddings

        emb = MultimodalEmbeddings()  # auto-detects provider

        # Embed a single image
        vector = emb.embed([Path("photo.jpg")])

        # Embed image + caption together
        vector = emb.embed([Path("photo.jpg"), "a picture of a mountain"])

        # Batch embedding
        vectors = emb.embed_batch([
            [Path("img1.jpg"), "caption one"],
            [Path("img2.png"), "caption two"],
        ])

        # Async
        vector = await emb.aembed([Path("photo.jpg")])
        ```

    Providers:
        - voyage / voyage_ai: Voyage AI (VOYAGE_API_KEY)
        - cohere: Cohere (COHERE_API_KEY)
        - google / google_vertexai / vertexai: Google Vertex AI
          (GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT)
        - amazon / bedrock / aws: Amazon Bedrock (AWS_ACCESS_KEY_ID)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize multimodal embeddings.

        Args:
            provider: Provider name. Auto-detects from environment if not given.
            model: Model name. Uses provider default if not specified.
            **kwargs: Additional provider-specific options passed through.

        Raises:
            ValueError: If no provider is available or an unknown provider is
                specified.

        Example:
            ```python
            # Auto-detect
            emb = MultimodalEmbeddings()

            # Explicit provider and model
            emb = MultimodalEmbeddings(
                provider="voyage",
                model="voyage-multimodal-3.5",
            )
            ```
        """
        if provider is None:
            provider = self._auto_detect()

        provider = provider.lower()
        provider = _PROVIDER_ALIASES.get(provider, provider)

        if provider not in _PROVIDER_INFO:
            raise ValueError(
                f"Unknown multimodal embedding provider: {provider!r}. "
                f"Available: {', '.join(_PROVIDER_INFO)}"
            )

        info = _PROVIDER_INFO[provider]
        self._provider = provider
        self._model = model or info["default_model"]
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """Get the active provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """Get the active model name."""
        return self._model

    def embed(self, inputs: MultimodalInput) -> list[float]:
        """Embed a sequence of text strings and/or images into one vector.

        Args:
            inputs: Ordered sequence of str (text) and bytes/Path (image).

        Returns:
            Embedding vector as a list of floats.

        Example:
            ```python
            from pathlib import Path
            vector = emb.embed([Path("image.jpg"), "a sunset photo"])
            ```
        """
        return self.embed_batch([inputs])[0]

    def embed_batch(self, batch: list[MultimodalInput]) -> list[list[float]]:
        """Embed a batch of multimodal input sequences.

        Args:
            batch: List of input sequences; each sequence becomes one vector.

        Returns:
            List of embedding vectors, one per input sequence.

        Example:
            ```python
            vectors = emb.embed_batch([
                [Path("cat.jpg"), "a cat"],
                [Path("dog.jpg"), "a dog"],
            ])
            ```
        """
        dispatch = {
            "voyage": self._embed_voyage,
            "cohere": self._embed_cohere,
            "google_vertexai": self._embed_google_vertexai,
            "amazon": self._embed_amazon,
        }
        return dispatch[self._provider](batch)

    async def aembed(self, inputs: MultimodalInput) -> list[float]:
        """Async embed a sequence of text strings and/or images.

        Args:
            inputs: Ordered sequence of str (text) and bytes/Path (image).

        Returns:
            Embedding vector as a list of floats.
        """
        return (await self.aembed_batch([inputs]))[0]

    async def aembed_batch(self, batch: list[MultimodalInput]) -> list[list[float]]:
        """Async embed a batch of multimodal input sequences.

        Runs the synchronous embed_batch in a thread pool to avoid blocking
        the event loop.

        Args:
            batch: List of input sequences; each sequence becomes one vector.

        Returns:
            List of embedding vectors, one per input sequence.
        """
        return await asyncio.to_thread(self.embed_batch, batch)

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all supported multimodal embedding providers.

        Returns:
            List of provider names.
        """
        return list(_PROVIDER_INFO.keys())

    @classmethod
    def list_configured_providers(cls) -> list[str]:
        """List multimodal embedding providers with credentials configured.

        Returns:
            List of provider names that have their required env var set.
        """
        return [p for p in _PROVIDER_PRIORITY if os.environ.get(_PROVIDER_INFO[p]["env_key"])]

    def __repr__(self) -> str:
        return f"MultimodalEmbeddings(provider={self._provider!r}, model={self._model!r})"

    # ------------------------------------------------------------------
    # Internal: provider dispatch
    # ------------------------------------------------------------------

    def _auto_detect(self) -> str:
        """Return the first provider whose required env var is set."""
        for p in _PROVIDER_PRIORITY:
            if os.environ.get(_PROVIDER_INFO[p]["env_key"]):
                return p
        required = ", ".join(_PROVIDER_INFO[p]["env_key"] for p in _PROVIDER_PRIORITY)
        raise ValueError(f"No multimodal embedding provider available. Set one of: {required}")

    # ------------------------------------------------------------------
    # Voyage AI  (voyage-multimodal-3.5)
    # Uses voyageai.Client.multimodal_embed with PIL Images for image items.
    # ------------------------------------------------------------------

    def _embed_voyage(self, batch: list[MultimodalInput]) -> list[list[float]]:
        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Voyage multimodal embeddings require: pip install voyageai") from exc

        api_key = os.environ.get("VOYAGE_API_KEY")
        client = voyageai.Client(api_key=api_key)

        voyage_batch: list[list[Any]] = []
        for inputs in batch:
            sequence: list[Any] = []
            for item in inputs:
                if isinstance(item, str):
                    sequence.append(item)
                else:
                    img_bytes = _resolve_bytes(item)
                    try:
                        import io

                        from PIL import Image  # type: ignore[import-untyped]

                        sequence.append(Image.open(io.BytesIO(img_bytes)))
                    except ImportError as exc:
                        raise ImportError(
                            "Voyage image embedding requires: pip install Pillow"
                        ) from exc
            voyage_batch.append(sequence)

        result = client.multimodal_embed(inputs=voyage_batch, model=self._model, **self._kwargs)
        return [list(v) for v in result.embeddings]

    # ------------------------------------------------------------------
    # Cohere  (embed-v4.0)
    # Uses the inputs=[{"type": "text"|"image", ...}] API.
    # ------------------------------------------------------------------

    def _embed_cohere(self, batch: list[MultimodalInput]) -> list[list[float]]:
        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Cohere multimodal embeddings require: pip install cohere") from exc

        api_key = os.environ.get("COHERE_API_KEY")
        co = cohere.Client(api_key=api_key)

        all_embeddings: list[list[float]] = []
        for inputs in batch:
            cohere_inputs: list[dict[str, str]] = []
            for item in inputs:
                if isinstance(item, str):
                    cohere_inputs.append({"type": "text", "text": item})
                else:
                    img_bytes = _resolve_bytes(item)
                    mime = _detect_mime(img_bytes)
                    b64 = base64.b64encode(img_bytes).decode()
                    cohere_inputs.append(
                        {
                            "type": "image",
                            "image": f"data:{mime};base64,{b64}",
                        }
                    )

            result = co.embed(
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
                inputs=cohere_inputs,
                **self._kwargs,
            )
            all_embeddings.append(list(result.embeddings.float_[0]))

        return all_embeddings

    # ------------------------------------------------------------------
    # Google Vertex AI  (multimodalembedding@001)
    # Returns separate text/image embeddings; we return image_embedding when
    # an image is present, text_embedding otherwise.
    # ------------------------------------------------------------------

    def _embed_google_vertexai(self, batch: list[MultimodalInput]) -> list[list[float]]:
        try:
            from vertexai.vision_models import (  # type: ignore[import-untyped]
                Image as VertexImage,
            )
            from vertexai.vision_models import (
                MultiModalEmbeddingModel,
            )
        except ImportError as exc:
            raise ImportError(
                "Google Vertex AI multimodal embeddings require: "
                "pip install google-cloud-aiplatform"
            ) from exc

        model = MultiModalEmbeddingModel.from_pretrained(self._model)

        all_embeddings: list[list[float]] = []
        for inputs in batch:
            text_items = [item for item in inputs if isinstance(item, str)]
            image_items = [item for item in inputs if not isinstance(item, str)]

            call_kwargs: dict[str, Any] = {}
            if text_items:
                call_kwargs["contextual_text"] = " ".join(text_items)
            if image_items:
                img_bytes = _resolve_bytes(image_items[0])
                call_kwargs["image"] = VertexImage(image_bytes=img_bytes)

            response = model.get_embeddings(**call_kwargs, **self._kwargs)

            # Prefer image embedding when an image was provided
            if image_items and response.image_embedding:
                vector: list[float] = list(response.image_embedding)
            else:
                vector = list(response.text_embedding)

            all_embeddings.append(vector)

        return all_embeddings

    # ------------------------------------------------------------------
    # Amazon Bedrock  (amazon.titan-embed-image-v1)
    # ------------------------------------------------------------------

    def _embed_amazon(self, batch: list[MultimodalInput]) -> list[list[float]]:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Amazon Bedrock multimodal embeddings require: pip install boto3"
            ) from exc

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        all_embeddings: list[list[float]] = []
        for inputs in batch:
            text_items = [item for item in inputs if isinstance(item, str)]
            image_items = [item for item in inputs if not isinstance(item, str)]

            body: dict[str, Any] = {}
            if text_items:
                body["inputText"] = " ".join(text_items)
            if image_items:
                img_bytes = _resolve_bytes(image_items[0])
                body["inputImage"] = base64.b64encode(img_bytes).decode()

            response = client.invoke_model(
                modelId=self._model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            all_embeddings.append(list(result["embedding"]))

        return all_embeddings
