"""Multimodal support for LLM - Vision, TTS, STT, Audio.

This module extends LLM capabilities with multimodal support:
- Vision: Image input to LLMs (GPT-4o, Claude 3, Gemini) - provider-agnostic
- TTS: Text-to-Speech (OpenAI, Google, ElevenLabs)
- STT: Speech-to-Text (OpenAI Whisper, Google, Deepgram)

Example - Vision:
    ```python
    from ai_infra.llm import LLM

    llm = LLM()

    # Simple: Just pass images - works with any provider!
    response = llm.chat(
        "What's in this image?",
        images=["https://example.com/photo.jpg"]
    )

    # Multiple images, mixed sources
    response = llm.chat(
        "Compare these images",
        images=[
            "https://example.com/cat.jpg",  # URL
            "local_photo.png",               # Local file
            screenshot_bytes,                # Raw bytes
        ]
    )
    ```

Example - TTS:
    ```python
    from ai_infra.llm.multimodal import TTS

    tts = TTS()  # Auto-detect provider
    audio = tts.speak("Hello, world!")
    tts.speak_to_file("Hello!", "greeting.mp3")

    # Streaming for real-time playback
    for chunk in tts.stream("Long text..."):
        play_audio(chunk)
    ```

Example - STT:
    ```python
    from ai_infra.llm.multimodal import STT

    stt = STT()
    result = stt.transcribe("audio.mp3")
    print(result.text)

    # With timestamps
    result = stt.transcribe("audio.mp3", timestamps=True)
    for segment in result.segments:
        print(f"{segment.start:.2f}s: {segment.text}")
    ```
"""

from ai_infra.llm.multimodal.audio import AudioInput as AudioInputType
from ai_infra.llm.multimodal.audio import (
    build_audio_content,
    create_audio_message,
    encode_audio,
    encode_audio_for_google,
    encode_audio_for_openai,
)
from ai_infra.llm.multimodal.audio_output import (
    AUDIO_MODELS,
    AudioOutput,
    AudioOutputFormat,
    AudioResponse,
    AudioVoice,
    get_audio_model,
    is_audio_model,
    list_audio_voices,
    parse_audio_response,
)
from ai_infra.llm.multimodal.documents import (
    DocumentInput,
    build_document_content,
    create_document_message,
    encode_document,
)
from ai_infra.llm.multimodal.models import (
    AudioFormat,
    AudioSegment,
    TranscriptionResult,
    TranscriptionSegment,
    Voice,
)
from ai_infra.llm.multimodal.stt import STT
from ai_infra.llm.multimodal.tts import TTS
from ai_infra.llm.multimodal.vision import (  # Backwards compatibility (deprecated)
    ImageInput,
    build_vision_content,
    create_vision_message,
    encode_image,
    encode_image_for_anthropic,
    encode_image_for_google,
    encode_image_for_openai,
    make_vision_message,
)
from ai_infra.llm.multimodal.voice import (
    AudioPlayer,
    Microphone,
    RecordingConfig,
    VoiceChat,
    VoiceChatConfig,
)

__all__ = [
    # Documents - New API
    "create_document_message",
    "build_document_content",
    "encode_document",
    "DocumentInput",
    # Vision - New API (recommended)
    "create_vision_message",
    "build_vision_content",
    "encode_image",
    "ImageInput",
    # Vision - Backwards compatibility (deprecated)
    "encode_image_for_openai",
    "encode_image_for_anthropic",
    "encode_image_for_google",
    "make_vision_message",
    # Audio Input - New API
    "create_audio_message",
    "build_audio_content",
    "encode_audio",
    "encode_audio_for_openai",
    "encode_audio_for_google",
    "AudioInputType",
    # Audio Output - New API
    "AudioOutput",
    "AudioResponse",
    "AudioVoice",
    "AudioOutputFormat",
    "parse_audio_response",
    "list_audio_voices",
    "get_audio_model",
    "is_audio_model",
    "AUDIO_MODELS",
    # TTS
    "TTS",
    # STT
    "STT",
    # Voice (Microphone, Playback, VoiceChat)
    "Microphone",
    "AudioPlayer",
    "VoiceChat",
    "RecordingConfig",
    "VoiceChatConfig",
    # Models
    "AudioFormat",
    "AudioSegment",
    "TranscriptionResult",
    "TranscriptionSegment",
    "Voice",
]
