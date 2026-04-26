#!/usr/bin/env python
"""Basic Image Generation Examples.

This example demonstrates:
- Zero-config image generation
- Multiple providers (OpenAI, Google, xAI, Stability, Replicate)
- Different image sizes and quality options
- Saving and handling generated images
- Multiple image generation
- Image variations (DALL-E 2)
- Provider discovery

ai-infra provides a unified interface for image generation
across OpenAI GPT Image / DALL-E, Google Imagen/Gemini, xAI, Stability AI, and Replicate.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from ai_infra import ImageGen

# =============================================================================
# Example 1: Zero-Config Generation
# =============================================================================


def zero_config():
    """Generate an image with automatic provider detection."""
    print("=" * 60)
    print("1. Zero-Config Image Generation")
    print("=" * 60)

    # Auto-detects provider from environment variables
    # Priority: OPENAI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, STABILITY_API_KEY,
    # REPLICATE_API_TOKEN
    gen = ImageGen()

    print(f"\n  Provider: {gen.provider.value}")
    print(f"  Model: {gen.model}")

    # Generate an image
    print("\n  Generating image...")

    try:
        images = gen.generate("A serene mountain landscape at sunset")

        print(f"\n  [OK] Generated {len(images)} image(s)")
        if images:
            img = images[0]
            print(f"    URL: {img.url[:50]}..." if img.url else f"    Data: {len(img.data)} bytes")
            if img.revised_prompt:
                print(f"    Revised prompt: {img.revised_prompt[:60]}...")
    except Exception as e:
        print(f"\n  [!] Generation failed: {e}")
        print("    Make sure you have an API key configured")


# =============================================================================
# Example 2: Explicit Provider Selection
# =============================================================================


def explicit_provider():
    """Select a specific provider."""
    print("\n" + "=" * 60)
    print("2. Explicit Provider Selection")
    print("=" * 60)

    providers = [
        ("openai", "OPENAI_API_KEY", "gpt-image-1.5"),
        ("google", "GOOGLE_API_KEY", "imagen-4.0-fast-generate-001"),
        ("xai", "XAI_API_KEY", "grok-imagine-image"),
        ("stability", "STABILITY_API_KEY", "stable-diffusion-xl"),
        ("replicate", "REPLICATE_API_TOKEN", "flux-schnell"),
    ]

    for provider, env_var, model in providers:
        configured = bool(os.getenv(env_var))
        status = "[OK]" if configured else "[X]"
        print(f"\n  {status} {provider}: {model}")
        if not configured:
            print(f"    Set {env_var} to use")

    # Example with explicit provider
    if os.getenv("OPENAI_API_KEY"):
        print("\n  Using OpenAI GPT Image 1.5:")
        gen = ImageGen(provider="openai", model="gpt-image-1.5")
        print(f"    Provider: {gen.provider.value}")
        print(f"    Model: {gen.model}")


# =============================================================================
# Example 3: Image Size and Quality Options
# =============================================================================


def size_and_quality():
    """Configure image size and quality."""
    print("\n" + "=" * 60)
    print("3. Size and Quality Options")
    print("=" * 60)

    print("\nOpenAI GPT Image options:")
    options = {
        "Sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "Quality": ["auto", "low", "medium", "high"],
        "Background": ["auto", "opaque", "transparent"],
    }

    for option, values in options.items():
        print(f"  {option}: {', '.join(values)}")

    print("\nExample usage:")
    print("""
    gen = ImageGen(provider="openai", model="gpt-image-1.5")
    images = gen.generate(
        "A futuristic cityscape",
        size="1536x1024",    # Landscape
        quality="high",      # High quality
        background="transparent",
    )
""")

    print("\nGoogle image options:")
    print("  Imagen models: imagen-4.0-fast-generate-001, imagen-4.0-generate-001")
    print("  Gemini models: gemini-3.1-flash-image-preview, gemini-3-pro-image-preview")

    print("\nxAI options:")
    print("  Model: grok-imagine-image")
    print("  Provider-native kwargs: aspect_ratio, resolution, image_format")

    print("\nStability AI options:")
    print("  Sizes: 1024x1024")
    print("  Additional: steps, cfg_scale, seed")


# =============================================================================
# Example 4: Saving Generated Images
# =============================================================================


def saving_images():
    """Save generated images to disk."""
    print("\n" + "=" * 60)
    print("4. Saving Generated Images")
    print("=" * 60)

    # ImageGen() would be used in production - examples shown below
    _ = ImageGen  # Reference to show the class exists

    print("\n  Methods for saving images:")
    print("""
    # Method 1: Direct save (handles URL or bytes)
    images = gen.generate("A cute robot")
    images[0].save("robot.png")

    # Method 2: Save multiple images
    for i, image in enumerate(images):
        image.save(f"robot_{i}.png")

    # Method 3: Get raw bytes
    image_bytes = images[0].data
    if image_bytes is None and images[0].url:
        import httpx
        image_bytes = httpx.get(images[0].url).content

    # Method 4: Display in notebook
    from IPython.display import Image, display
    display(Image(url=images[0].url))
""")

    # Demonstrate with temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "demo.png"
        print(f"\n  Demo output path: {output_path}")
        print("  (Actual save skipped - set API key to generate)")


# =============================================================================
# Example 5: Multiple Image Generation
# =============================================================================


def multiple_images():
    """Generate multiple image variations."""
    print("\n" + "=" * 60)
    print("5. Multiple Image Generation")
    print("=" * 60)

    print("\n  Generate multiple variations:")
    print("""
    gen = ImageGen(provider="openai")
    images = gen.generate(
        "A magical forest",
        n=4,  # Generate 4 variations
    )

    for i, image in enumerate(images):
        image.save(f"forest_{i}.png")
        print(f"Image {i}: {image.url[:50]}...")
""")

    print("\n  Note: GPT Image models support multiple images")
    print("        DALL-E 3 supports n=1 only")
    print("        DALL-E 2 supports n=1 to n=10")
    print("        Google Imagen supports multiple images")


# =============================================================================
# Example 6: Async Generation
# =============================================================================


async def async_generation():
    """Generate images asynchronously."""
    print("\n" + "=" * 60)
    print("6. Async Image Generation")
    print("=" * 60)

    print("\n  Use agenerate() for async operations:")
    print("""
    gen = ImageGen()

    # Single async generation
    images = await gen.agenerate("A sunset over the ocean")

    # Concurrent generation
    prompts = ["A cat", "A dog", "A bird"]
    tasks = [gen.agenerate(p) for p in prompts]
    all_images = await asyncio.gather(*tasks)
""")

    # Demo (if API key available)
    gen = ImageGen()
    print(f"\n  Provider: {gen.provider.value}")
    print("  (Actual async generation skipped - demonstrates pattern)")


# =============================================================================
# Example 7: Image Variations (DALL-E 2)
# =============================================================================


def image_variations():
    """Generate variations of an existing image."""
    print("\n" + "=" * 60)
    print("7. Image Variations (DALL-E 2 only)")
    print("=" * 60)

    print("\n  Create variations of an existing image:")
    print("""
    gen = ImageGen(provider="openai", model="dall-e-2")

    # From file path
    variations = gen.variations(
        image="input.png",
        n=3,  # Generate 3 variations
        size="1024x1024",
    )

    # From bytes
    with open("input.png", "rb") as f:
        image_bytes = f.read()
    variations = gen.variations(image=image_bytes, n=2)
""")

    print("\n  Note: Only DALL-E 2 supports variations")
    print("        The model must match the original image size")


# =============================================================================
# Example 8: Provider Discovery
# =============================================================================


def provider_discovery():
    """Discover available providers and models."""
    print("\n" + "=" * 60)
    print("8. Provider Discovery")
    print("=" * 60)

    # List providers
    providers = ImageGen.list_providers()
    print(f"\n  Available providers: {providers}")

    # List built-in fallback models per provider
    print("\n  Known fallback models by provider:")
    for provider in providers:
        try:
            models = ImageGen.list_known_models(provider)
            print(f"    {provider}: {models}")
        except Exception:
            print(f"    {provider}: (error listing models)")

    # Live discovery for configured providers
    print("\n  Live models by configured provider:")
    try:
        configured_providers = ImageGen.list_configured_providers()
        if not configured_providers:
            print("    (no configured providers)")
        for provider in configured_providers:
            models = ImageGen.list_models(provider)
            print(f"    {provider}: {models[:5]}{' ...' if len(models) > 5 else ''}")
    except Exception:
        print("    (error fetching live models)")

    # Check which are configured
    print("\n  Configured providers:")
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "stability": "STABILITY_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
    }

    for provider, env_var in env_vars.items():
        configured = bool(os.getenv(env_var))
        print(f"    {provider}: {'[OK]' if configured else '[X]'}")


# =============================================================================
# Example 9: Error Handling
# =============================================================================


def error_handling():
    """Handle common errors gracefully."""
    print("\n" + "=" * 60)
    print("9. Error Handling")
    print("=" * 60)

    print("\n  Common errors and handling:")
    print("""
    from ai_infra.imagegen import (
        ImageGenError,
        ContentFilterError,
        RateLimitError,
    )

    try:
        images = gen.generate("A beautiful landscape")
    except ContentFilterError:
        print("Prompt rejected by content filter")
    except RateLimitError:
        print("Rate limit hit - try again later")
    except ImageGenError as e:
        print(f"Generation failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
""")

    print("\n  Best practices:")
    print("    - Always wrap generate() in try/except")
    print("    - Handle rate limits with exponential backoff")
    print("    - Check content filter for user-provided prompts")
    print("    - Validate API keys before generation")


# =============================================================================
# Example 10: Provider Comparison
# =============================================================================


def provider_comparison():
    """Compare features across providers."""
    print("\n" + "=" * 60)
    print("10. Provider Comparison")
    print("=" * 60)

    comparison = {
        "OpenAI DALL-E 3": {
            "Quality": "Excellent",
            "Speed": "Fast (~5s)",
            "Cost": "$$",
            "Features": "Style control, HD quality",
            "Best for": "High-quality commercial images",
        },
        "Google Gemini Image": {
            "Quality": "Excellent",
            "Speed": "Fast (~3-5s)",
            "Cost": "$",
            "Features": "Free tier, multimodal",
            "Best for": "Development, free tier usage",
        },
        "Stability AI": {
            "Quality": "Good",
            "Speed": "Medium (~10s)",
            "Cost": "$",
            "Features": "Fine control, seeds",
            "Best for": "Reproducible generations",
        },
        "Replicate (Flux)": {
            "Quality": "Excellent",
            "Speed": "Fast",
            "Cost": "$",
            "Features": "Open models, flexibility",
            "Best for": "Open-source alternatives",
        },
    }

    for provider, features in comparison.items():
        print(f"\n  {provider}:")
        for key, value in features.items():
            print(f"    {key}: {value}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Basic Image Generation Examples")
    print("=" * 60)
    print("\nai-infra provides unified image generation across providers.\n")

    zero_config()
    explicit_provider()
    size_and_quality()
    saving_images()
    multiple_images()
    await async_generation()
    image_variations()
    provider_discovery()
    error_handling()
    provider_comparison()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. ImageGen() auto-detects provider from env vars")
    print("  2. Explicit provider: ImageGen(provider='openai')")
    print("  3. Use generate() for sync, agenerate() for async")
    print("  4. save() handles both URL and bytes automatically")
    print("  5. Check list_providers() for available options")


if __name__ == "__main__":
    asyncio.run(main())
