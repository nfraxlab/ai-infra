# Provider Registry

> Centralized configuration for all AI providers and their capabilities.

## Overview

The Provider Registry is the single source of truth for all provider configurations in ai-infra. It manages:

- **10 providers**: OpenAI, Anthropic, Google, xAI, ElevenLabs, Deepgram, Stability, Replicate, Voyage, Cohere
- **6 capabilities**: Chat, Embeddings, TTS, STT, ImageGen, Realtime

---

## Quick Start

```python
from ai_infra import (
    ProviderRegistry,
    ProviderCapability,
    list_providers,
    list_providers_for_capability,
    get_provider,
    is_provider_configured,
)

# List all registered providers
providers = list_providers()
# ['openai', 'anthropic', 'google_genai', 'xai', 'elevenlabs', ...]

# List providers for a specific capability
chat_providers = list_providers_for_capability(ProviderCapability.CHAT)
# ['openai', 'anthropic', 'google_genai', 'xai']

# Get provider configuration
openai = get_provider("openai")
print(openai.display_name)  # "OpenAI"
print(openai.env_var)       # "OPENAI_API_KEY"

# Check if provider is configured (API key set)
if is_provider_configured("openai"):
    print("OpenAI is ready!")
```

---

## Supported Providers

| Provider | Name | Env Var | Capabilities |
|----------|------|---------|--------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | Chat, Embeddings, TTS, STT, ImageGen, Realtime |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Chat |
| Google | `google_genai` | `GEMINI_API_KEY` | Chat, Embeddings, TTS, STT, ImageGen, Realtime |
| xAI | `xai` | `XAI_API_KEY` | Chat, ImageGen |
| ElevenLabs | `elevenlabs` | `ELEVENLABS_API_KEY` | TTS |
| Deepgram | `deepgram` | `DEEPGRAM_API_KEY` | STT |
| Stability | `stability` | `STABILITY_API_KEY` | ImageGen |
| Replicate | `replicate` | `REPLICATE_API_TOKEN` | ImageGen |
| Voyage | `voyage` | `VOYAGE_API_KEY` | Embeddings |
| Cohere | `cohere` | `COHERE_API_KEY` | Embeddings |

---

## Capabilities

```python
from ai_infra import ProviderCapability

# Available capabilities
ProviderCapability.CHAT        # Text chat/completion
ProviderCapability.EMBEDDINGS  # Text embeddings
ProviderCapability.TTS         # Text-to-speech
ProviderCapability.STT         # Speech-to-text
ProviderCapability.IMAGEGEN    # Image generation
ProviderCapability.REALTIME    # Real-time voice
```

### Query by Capability

```python
from ai_infra import list_providers_for_capability, ProviderCapability

# Chat providers
chat = list_providers_for_capability(ProviderCapability.CHAT)
# ['openai', 'anthropic', 'google_genai', 'xai']

# Embedding providers
embed = list_providers_for_capability(ProviderCapability.EMBEDDINGS)
# ['openai', 'google_genai', 'voyage', 'cohere']

# TTS providers
tts = list_providers_for_capability(ProviderCapability.TTS)
# ['openai', 'google_genai', 'elevenlabs']

# STT providers
stt = list_providers_for_capability(ProviderCapability.STT)
# ['openai', 'google_genai', 'deepgram']

# Image generation providers
imagegen = list_providers_for_capability(ProviderCapability.IMAGEGEN)
# ['openai', 'google_genai', 'stability', 'replicate']

# Realtime voice providers
realtime = list_providers_for_capability(ProviderCapability.REALTIME)
# ['openai', 'google_genai']
```

---

## Provider Configuration

Get detailed configuration for a provider:

```python
from ai_infra import get_provider, ProviderCapability

openai = get_provider("openai")

# Basic info
print(openai.name)          # "openai"
print(openai.display_name)  # "OpenAI"
print(openai.env_var)       # "OPENAI_API_KEY"
print(openai.alt_env_vars)  # []

# Check configuration
print(openai.capabilities)  # [CHAT, EMBEDDINGS, TTS, STT, IMAGEGEN, REALTIME]

# Get capability-specific config
chat_config = openai.get_capability(ProviderCapability.CHAT)
print(chat_config.models)        # ['gpt-4o', 'gpt-4o-mini', ...]
print(chat_config.default_model) # 'gpt-4o-mini'

tts_config = openai.get_capability(ProviderCapability.TTS)
print(tts_config.voices)         # ['alloy', 'echo', 'fable', ...]
print(tts_config.default_voice)  # 'alloy'
```

---

## Check Provider Status

```python
from ai_infra import is_provider_configured, ProviderRegistry

# Check if API key is set
if is_provider_configured("openai"):
    print("OpenAI ready")

# Get API key (if set)
api_key = ProviderRegistry.get_api_key("openai")

# List all configured providers
configured = [
    p for p in list_providers()
    if is_provider_configured(p)
]
```

---

## Environment Variables

### Primary Variables

Each provider has a primary environment variable:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=...
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
STABILITY_API_KEY=...
REPLICATE_API_TOKEN=...
VOYAGE_API_KEY=...
COHERE_API_KEY=...
```

### Alternative Variables

Some providers support alternative variable names:

| Provider | Primary | Alternatives |
|----------|---------|--------------|
| Google | `GEMINI_API_KEY` | `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY` |
| ElevenLabs | `ELEVENLABS_API_KEY` | `ELEVEN_API_KEY` |
| Cohere | `COHERE_API_KEY` | `CO_API_KEY` |

---

## Models and Voices

### Get Models for a Provider

```python
from ai_infra import get_provider, ProviderCapability

openai = get_provider("openai")

# Chat models
chat = openai.get_capability(ProviderCapability.CHAT)
print(chat.models)
# ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1', 'o1-mini', ...]
print(chat.default_model)
# 'gpt-4o-mini'

# Embedding models
embed = openai.get_capability(ProviderCapability.EMBEDDINGS)
print(embed.models)
# ['text-embedding-3-small', 'text-embedding-3-large']
```

### Get Voices for TTS

```python
openai = get_provider("openai")
tts = openai.get_capability(ProviderCapability.TTS)

print(tts.voices)
# ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
print(tts.default_voice)
# 'alloy'

# ElevenLabs
elevenlabs = get_provider("elevenlabs")
tts = elevenlabs.get_capability(ProviderCapability.TTS)
print(tts.voices)
# ['Rachel', 'Domi', 'Bella', 'Antoni', ...]
```

---

## Direct Registry Access

For advanced use cases:

```python
from ai_infra import ProviderRegistry, ProviderCapability

# List all providers
all_providers = ProviderRegistry.list_all()

# Get provider by name
config = ProviderRegistry.get("openai")

# List for capability
chat_providers = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)

# Check if configured
is_ready = ProviderRegistry.is_configured("openai")

# Get API key
key = ProviderRegistry.get_api_key("openai")
```

---

## Provider Support Matrix

| Provider | Chat | Embed | TTS | STT | Image | Realtime |
|----------|:----:|:-----:|:---:|:---:|:-----:|:--------:|
| OpenAI | [OK] | [OK] | [OK] | [OK] | [OK] | [OK] |
| Anthropic | [OK] | - | - | - | - | - |
| Google | [OK] | [OK] | [OK] | [OK] | [OK] | [OK] |
| xAI | [OK] | - | - | - | - | - |
| ElevenLabs | - | - | [OK] | - | - | - |
| Deepgram | - | - | - | [OK] | - | - |
| Stability | - | - | - | - | [OK] | - |
| Replicate | - | - | - | - | [OK] | - |
| Voyage | - | [OK] | - | - | - | - |
| Cohere | - | [OK] | - | - | - | - |

---

## See Also

- [LLM](llm.md) - Chat with providers
- [TTS](../multimodal/tts.md) - Text-to-speech providers
- [STT](../multimodal/stt.md) - Speech-to-text providers
- [Embeddings](../embeddings/embeddings.md) - Embedding providers
- [ImageGen](../imagegen/imagegen.md) - Image generation providers
- [Realtime](../multimodal/realtime.md) - Realtime voice providers
