"""Cohere provider configuration.

Cohere supports:
- Embeddings: Multilingual embeddings for search and RAG
- Multimodal Embeddings: Text + image embeddings (embed-v4.0)

Known for enterprise-focused NLP and embeddings.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

COHERE = ProviderConfig(
    name="cohere",
    display_name="Cohere",
    env_var="COHERE_API_KEY",
    capabilities={
        ProviderCapability.EMBEDDINGS: CapabilityConfig(
            models=[
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
                "embed-english-v2.0",
                "embed-multilingual-v2.0",
            ],
            default_model="embed-english-v3.0",
            features=["multilingual", "input_type", "truncation"],
            extra={
                "dimensions": {
                    "embed-english-v3.0": 1024,
                    "embed-multilingual-v3.0": 1024,
                    "embed-english-light-v3.0": 384,
                    "embed-multilingual-light-v3.0": 384,
                },
                "input_types": [
                    "search_document",
                    "search_query",
                    "classification",
                    "clustering",
                ],
            },
        ),
        ProviderCapability.MULTIMODAL_EMBEDDINGS: CapabilityConfig(
            models=[
                "embed-v4.0",
                "embed-multilingual-v4.0",
            ],
            default_model="embed-v4.0",
            features=["multilingual", "mixed_inputs", "128k_context"],
            extra={
                "dimensions": {
                    "embed-v4.0": 1024,
                    "embed-multilingual-v4.0": 1024,
                },
                "input_types": ["text", "image"],
                "max_context_tokens": 128000,
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(COHERE)
