# Image Generation

> Generate images with OpenAI, Google, xAI, Stability AI, and Replicate.

## Quick Start

```python
from ai_infra import ImageGen

gen = ImageGen()  # Auto-detects the first configured image provider
images = gen.generate("A serene mountain landscape at sunset")

images[0].save("landscape.png")
```

`generate()` and `agenerate()` always return `list[GeneratedImage]`, even when `n=1`.

## Overview

ai-infra wraps provider-specific image APIs behind a single `ImageGen` interface:

- **OpenAI**: GPT Image 1.5/1/mini plus legacy DALL-E 2/3
- **Google**: Imagen 3/4 and Gemini native image models
- **xAI**: Grok Imagine (`grok-imagine-image`)
- **Stability AI**: SDXL and other text-to-image engines
- **Replicate**: Curated popular image models such as Flux and SDXL

Auto-detection priority is:

1. `OPENAI_API_KEY`
2. `GOOGLE_API_KEY` / `GEMINI_API_KEY`
3. `XAI_API_KEY`
4. `STABILITY_API_KEY`
5. `REPLICATE_API_TOKEN`

## Basic Usage

### Explicit Provider Selection

```python
from ai_infra import ImageGen

openai_gen = ImageGen(provider="openai")
google_gen = ImageGen(provider="google")
xai_gen = ImageGen(provider="xai")
stability_gen = ImageGen(provider="stability")
replicate_gen = ImageGen(provider="replicate")
```

`provider="google_genai"` is also accepted for backwards compatibility, but `google` is the preferred user-facing name.

### Async Generation

```python
from ai_infra import ImageGen

gen = ImageGen(provider="openai")
images = await gen.agenerate("A sunset over mountains")

images[0].save("sunset.png")
```

### Multiple Images

```python
gen = ImageGen(provider="openai", model="gpt-image-1.5")
images = gen.generate("A magical forest", n=3)

for index, image in enumerate(images):
    image.save(f"forest_{index}.png")
```

Provider limits vary. OpenAI legacy DALL-E 3 remains more restrictive than GPT Image, Google Imagen, or xAI batch generation.

## Provider Notes

### OpenAI

```python
gen = ImageGen(provider="openai", model="gpt-image-1.5")
images = gen.generate(
    "A polished ecommerce photo of a ceramic mug",
    size="1536x1024",
    quality="high",
    background="transparent",
    output_format="webp",
)
```

- Default model: `gpt-image-1.5`
- GPT Image models return binary image data directly
- Legacy `dall-e-2` and `dall-e-3` are still supported
- Provider-native kwargs such as `background`, `output_format`, `output_compression`, `moderation`, and `input_fidelity` pass through to the SDK

### Google

Use Imagen when you want the dedicated image-generation API, and Gemini image models when you want native editing, reference images, or Gemini-specific image workflows.

```python
gen = ImageGen(provider="google", model="imagen-4.0-fast-generate-001")
images = gen.generate(
    "A clean editorial product shot of a notebook",
    aspect_ratio="16:9",
    image_size="2K",
)
```

```python
gen = ImageGen(provider="google", model="gemini-3.1-flash-image-preview")
images = gen.generate(
    "Create a crisp infographic explaining photosynthesis",
    aspect_ratio="16:9",
    image_size="2K",
)
```

- Default model: `imagen-4.0-fast-generate-001`
- Gemini image models are used through `generate_content`
- Imagen models are used through `generate_images`
- For Gemini models, ai-infra skips interim thought images and returns only final image outputs
- For Google image generation, `aspect_ratio` and `image_size` are a better fit than raw `size`, though `size` is still used to derive sensible defaults when possible

### xAI

```python
gen = ImageGen(provider="xai")
images = gen.generate(
    "A collage of London landmarks in a stenciled street-art style",
    aspect_ratio="3:2",
    resolution="2k",
)

images[0].save("london.png")
```

- Default model: `grok-imagine-image`
- ai-infra requests base64 output by default so `save()` works without an additional fetch step
- Supported provider-native kwargs include `aspect_ratio`, `resolution`, and `image_format`
- If you prefer temporary URLs instead of binary data, pass `image_format="url"`

### Stability AI

```python
gen = ImageGen(provider="stability")
images = gen.generate(
    "A photorealistic portrait",
    size="1024x1024",
    steps=30,
    cfg_scale=7.0,
    seed=12345,
)
```

### Replicate

```python
gen = ImageGen(provider="replicate", model="black-forest-labs/flux-schnell")
images = gen.generate(
    "A futuristic city skyline at night",
    size="1024x1024",
    num_outputs=2,
)
```

Replicate commonly returns URLs rather than image bytes. Use `await images[0].fetch()` if you need local bytes before saving.

## Editing Images

### OpenAI Inpainting

```python
gen = ImageGen(provider="openai", model="gpt-image-1.5")
images = gen.edit(
    image="input.png",
    mask="mask.png",
    prompt="Add a rainbow in the sky",
    size="1024x1024",
)
```

### Google Gemini Editing

```python
gen = ImageGen(provider="google", model="gemini-3.1-flash-image-preview")
images = gen.edit(
    image="input.png",
    prompt="Turn this into a magazine cover with bold serif typography",
    aspect_ratio="16:9",
)
```

Google editing requires a Gemini image model. The default Imagen model does not support the `edit()` path.

### xAI Editing

```python
gen = ImageGen(provider="xai")
images = gen.edit(
    image="input.png",
    prompt="Render this as a pencil sketch with detailed shading",
    aspect_ratio="3:2",
)
```

### Variations

```python
gen = ImageGen(provider="openai", model="dall-e-2")
variations = gen.variations(image="input.png", n=3)
```

Image variations are still OpenAI DALL-E 2 only.

## Working With Results

Each item returned by `generate()`, `agenerate()`, `edit()`, or `variations()` is a `GeneratedImage`.

```python
images = ImageGen(provider="openai").generate("A cute robot")
image = images[0]

print(image.provider)
print(image.model)
print(image.revised_prompt)

if image.data is not None:
    image.save("robot.png")
elif image.url is not None:
    await image.fetch()
    image.save("robot.png")
```

`GeneratedImage` fields:

- `data`: Image bytes, when the provider returned binary output
- `url`: Provider-hosted URL, when the provider returned a URL
- `revised_prompt`: Revised prompt text, when the provider exposes it
- `metadata`: Provider-specific metadata such as moderation or seed data

## Discovery

```python
from ai_infra.imagegen import (
    ImageGen,
    list_all_models,
    list_configured_providers,
    list_known_models,
    list_models,
    list_providers,
)

print(list_providers())
print(list_configured_providers())
print(list_known_models("xai"))
print(list_models("xai"))
print(list_all_models())

# Class-level convenience helpers
print(ImageGen.list_models("google", refresh=True))
print(ImageGen.list_known_models("google"))
print(ImageGen.list_all_models())
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...
STABILITY_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
```

## See Also

- [Providers](../core/providers.md) - Provider configuration
- [Vision](../multimodal/vision.md) - Image understanding
- [Validation](../infrastructure/validation.md) - Input validation
