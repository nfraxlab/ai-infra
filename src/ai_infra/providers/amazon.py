"""Amazon Web Services provider configuration.

Amazon supports:
- Multimodal Embeddings: Amazon Titan image/text embeddings via Bedrock

Requires AWS credentials: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY,
or an IAM role when running on AWS infrastructure.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

AMAZON = ProviderConfig(
    name="amazon",
    display_name="Amazon Web Services",
    env_var="AWS_ACCESS_KEY_ID",
    alt_env_vars=["AWS_SECRET_ACCESS_KEY"],
    capabilities={
        ProviderCapability.MULTIMODAL_EMBEDDINGS: CapabilityConfig(
            models=[
                "amazon.titan-embed-image-v1",
                "amazon.titan-embed-image-v2:0",
            ],
            default_model="amazon.titan-embed-image-v1",
            features=["text_and_image", "batch"],
            extra={
                "dimensions": {
                    "amazon.titan-embed-image-v1": 1024,
                    "amazon.titan-embed-image-v2:0": 1024,
                },
                "input_types": ["text", "image"],
                "default_region": "us-east-1",
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(AMAZON)
