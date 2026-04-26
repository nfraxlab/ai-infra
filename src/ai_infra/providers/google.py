"""Google/Gemini provider configuration.

Google supports:
- Chat: Gemini 3, 2.0, 1.5 Pro/Flash models
- Embeddings: text-embedding-004
- Multimodal Embeddings: multimodalembedding@001 (Vertex AI)
- TTS: Google Cloud TTS (standard, neural2, studio)
- STT: Google Cloud Speech-to-Text
- ImageGen: Gemini native image models + Imagen
- Realtime: Gemini Live API

Note: TTS/STT may require GOOGLE_APPLICATION_CREDENTIALS for service account.
Note: Multimodal embeddings use Vertex AI (multimodalembedding@001) and require
      GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT.
"""

from ai_infra.providers.base import CapabilityConfig, ProviderCapability, ProviderConfig
from ai_infra.providers.registry import ProviderRegistry

GOOGLE = ProviderConfig(
    name="google_genai",
    display_name="Google Gemini",
    env_var="GEMINI_API_KEY",
    alt_env_vars=["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"],
    capabilities={
        ProviderCapability.CHAT: CapabilityConfig(
            models=[
                "gemini-2.0-flash",
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-thinking-exp",
                "gemini-1.5-pro",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-1.0-pro",
            ],
            default_model="gemini-3-pro-preview",
            features=[
                "streaming",
                "function_calling",
                "vision",
                "audio_input",
                "grounding",
            ],
            extra={
                "context_window": {
                    "gemini-2.0-flash": 1048576,
                    "gemini-1.5-pro": 2097152,
                    "gemini-1.5-flash": 1048576,
                },
            },
        ),
        ProviderCapability.EMBEDDINGS: CapabilityConfig(
            models=[
                "models/text-embedding-004",
                "models/embedding-001",
            ],
            default_model="models/text-embedding-004",
            extra={
                "dimensions": {
                    "models/text-embedding-004": 768,
                    "models/embedding-001": 768,
                }
            },
        ),
        ProviderCapability.MULTIMODAL_EMBEDDINGS: CapabilityConfig(
            models=[
                "multimodalembedding@001",
            ],
            default_model="multimodalembedding@001",
            features=["text_embedding", "image_embedding", "video_embedding"],
            extra={
                "dimensions": {
                    "multimodalembedding@001": 1408,
                },
                "available_dimensions": [128, 256, 512, 1408],
                "requires_vertex_ai": True,
                "env_var_override": "GOOGLE_APPLICATION_CREDENTIALS",
            },
        ),
        ProviderCapability.TTS: CapabilityConfig(
            models=["standard", "neural2", "studio"],
            default_model="neural2",
            voices=[
                # Neural2 voices (US English)
                "en-US-Neural2-A",
                "en-US-Neural2-C",
                "en-US-Neural2-D",
                "en-US-Neural2-E",
                "en-US-Neural2-F",
                "en-US-Neural2-G",
                "en-US-Neural2-H",
                "en-US-Neural2-I",
                "en-US-Neural2-J",
            ],
            default_voice="en-US-Neural2-C",
            features=["ssml", "multiple_languages"],
            extra={
                "env_var_override": "GOOGLE_APPLICATION_CREDENTIALS",
            },
        ),
        ProviderCapability.STT: CapabilityConfig(
            models=["default", "latest_long", "latest_short"],
            default_model="default",
            features=[
                "streaming",
                "timestamps",
                "speaker_diarization",
                "automatic_punctuation",
            ],
            extra={
                "env_var_override": "GOOGLE_APPLICATION_CREDENTIALS",
            },
        ),
        ProviderCapability.IMAGEGEN: CapabilityConfig(
            models=[
                # Gemini multimodal (uses generate_content API)
                "gemini-3.1-flash-image-preview",
                "gemini-3-pro-image-preview",
                "gemini-2.5-flash-image",
                "gemini-2.0-flash-exp-image-generation",
                # Imagen models (uses generate_images API, may require billing)
                "imagen-4.0-ultra-generate-001",
                "imagen-3.0-generate-002",
                "imagen-3.0-generate-001",
                "imagen-3.0-fast-generate-001",
                "imagen-4.0-generate-001",
                "imagen-4.0-fast-generate-001",
            ],
            default_model="imagen-4.0-fast-generate-001",
            features=["edit", "upscale", "reference_images", "search_grounding", "4k_output"],
        ),
        ProviderCapability.REALTIME: CapabilityConfig(
            models=[
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-thinking-exp",
            ],
            default_model="gemini-2.0-flash-exp",
            voices=["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
            default_voice="Puck",
            features=["vad", "function_calling", "video_input", "affective_dialog"],
            extra={
                "audio_format": "pcm16",
                "input_sample_rate": 16000,
                "output_sample_rate": 24000,
                "max_session_duration": 900,  # 15 minutes for audio
            },
        ),
    },
)

# Register with the central registry
ProviderRegistry.register(GOOGLE)
