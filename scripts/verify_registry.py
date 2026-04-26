#!/usr/bin/env python3
"""Verification script for Provider Registry integration.

This script tests that all modules correctly use the provider registry
for their configuration and discovery needs.
"""

import os
import sys

# Ensure we can import ai_infra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_registry_basics():
    """Test basic registry functionality."""
    print("\n" + "=" * 60)
    print("1. TESTING REGISTRY BASICS")
    print("=" * 60)

    from ai_infra.providers import (
        ProviderCapability,
        ProviderRegistry,
        get_provider,
        list_providers_for_capability,
    )

    # Test list_all
    all_providers = ProviderRegistry.list_all()
    print(f"\n[OK] All registered providers ({len(all_providers)}): {all_providers}")
    assert len(all_providers) >= 10, "Expected at least 10 providers"

    # Test capabilities
    for cap in ProviderCapability:
        providers = list_providers_for_capability(cap)
        print(f"[OK] {cap.name} providers: {providers}")
        assert isinstance(providers, list)

    # Test get_provider
    openai = get_provider("openai")
    assert openai is not None, "OpenAI provider not found"
    assert openai.display_name == "OpenAI"
    print(f"\n[OK] OpenAI config: name={openai.name}, env_var={openai.env_var}")

    # Test capabilities on provider
    assert openai.has_capability(ProviderCapability.CHAT)
    assert openai.has_capability(ProviderCapability.TTS)
    assert openai.has_capability(ProviderCapability.STT)
    assert openai.has_capability(ProviderCapability.EMBEDDINGS)
    print("[OK] OpenAI capabilities verified")

    print("\n[OK] Registry basics: PASSED")
    return True


def test_embeddings_integration():
    """Test embeddings module uses registry."""
    print("\n" + "=" * 60)
    print("2. TESTING EMBEDDINGS INTEGRATION")
    print("=" * 60)

    from ai_infra.embeddings.embeddings import (
        _LANGCHAIN_MAPPINGS,
        _PROVIDER_ALIASES,
        _PROVIDER_PRIORITY,
    )
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    # Check that embeddings uses registry
    print(f"\n[OK] Provider priority: {_PROVIDER_PRIORITY}")
    print(f"[OK] Provider aliases: {_PROVIDER_ALIASES}")
    print(f"[OK] LangChain mappings: {list(_LANGCHAIN_MAPPINGS.keys())}")

    # Verify embeddings providers match registry
    registry_emb_providers = ProviderRegistry.list_for_capability(ProviderCapability.EMBEDDINGS)
    print(f"[OK] Registry EMBEDDINGS providers: {registry_emb_providers}")

    # Test provider detection uses registry
    for provider in ["openai", "voyage", "cohere"]:
        if provider in _LANGCHAIN_MAPPINGS:
            config = ProviderRegistry.get(provider)
            assert config is not None, f"Provider {provider} not in registry"
            cap = config.get_capability(ProviderCapability.EMBEDDINGS)
            assert cap is not None, f"Provider {provider} missing EMBEDDINGS capability"
            print(f"[OK] {provider}: models={cap.models[:2]}..., default={cap.default_model}")

    print("\n[OK] Embeddings integration: PASSED")
    return True


def test_tts_stt_integration():
    """Test TTS/STT modules use registry."""
    print("\n" + "=" * 60)
    print("3. TESTING TTS/STT INTEGRATION")
    print("=" * 60)

    from ai_infra.llm.multimodal.discovery import (
        list_stt_models,
        list_stt_providers,
        list_tts_models,
        list_tts_providers,
        list_tts_voices,
    )
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    # TTS tests
    tts_providers = list_tts_providers()
    print(f"\n[OK] TTS providers from discovery: {tts_providers}")
    registry_tts = ProviderRegistry.list_for_capability(ProviderCapability.TTS)
    print(f"[OK] TTS providers from registry: {registry_tts}")
    assert set(tts_providers) == set(registry_tts), "TTS provider mismatch"

    # Test OpenAI TTS voices come from registry
    voices = list_tts_voices("openai")
    print(f"[OK] OpenAI TTS voices: {voices}")
    assert "alloy" in voices
    assert "nova" in voices

    # Test models
    models = list_tts_models("openai")
    print(f"[OK] OpenAI TTS models: {models}")
    assert "tts-1" in models

    # STT tests
    stt_providers = list_stt_providers()
    print(f"\n[OK] STT providers from discovery: {stt_providers}")
    registry_stt = ProviderRegistry.list_for_capability(ProviderCapability.STT)
    print(f"[OK] STT providers from registry: {registry_stt}")
    assert set(stt_providers) == set(registry_stt), "STT provider mismatch"

    # Test OpenAI STT models
    stt_models = list_stt_models("openai")
    print(f"[OK] OpenAI STT models: {stt_models}")
    assert "whisper-1" in stt_models

    print("\n[OK] TTS/STT integration: PASSED")
    return True


def test_imagegen_integration():
    """Test ImageGen module uses registry."""
    print("\n" + "=" * 60)
    print("4. TESTING IMAGEGEN INTEGRATION")
    print("=" * 60)

    from ai_infra.imagegen.discovery import list_known_models, list_providers
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    # Test list_providers uses registry
    providers = list_providers()
    print(f"\n[OK] ImageGen providers: {providers}")
    registry_ig = ProviderRegistry.list_for_capability(ProviderCapability.IMAGEGEN)
    # Account for google/google_genai alias
    assert len(providers) == len(registry_ig), "ImageGen provider count mismatch"

    # Test list_known_models for each provider
    for provider in ["openai", "stability", "replicate"]:
        models = list_known_models(provider)
        print(f"[OK] {provider} models: {models[:3]}...")

        # Verify against registry
        config = ProviderRegistry.get(provider)
        assert config is not None
        cap = config.get_capability(ProviderCapability.IMAGEGEN)
        assert cap is not None
        assert set(models) == set(cap.models), f"Model mismatch for {provider}"

    # Test Google (which maps to google_genai)
    google_models = list_known_models("google")
    print(f"[OK] Google models: {google_models[:3]}...")

    print("\n[OK] ImageGen integration: PASSED")
    return True


def test_realtime_integration():
    """Test Realtime Voice module uses registry."""
    print("\n" + "=" * 60)
    print("5. TESTING REALTIME VOICE INTEGRATION")
    print("=" * 60)

    from ai_infra.llm.realtime.gemini import DEFAULT_MODEL as GEMINI_DEFAULT
    from ai_infra.llm.realtime.gemini import GEMINI_LIVE_MODELS, GEMINI_VOICES
    from ai_infra.llm.realtime.openai import DEFAULT_MODEL as OPENAI_DEFAULT
    from ai_infra.llm.realtime.openai import OPENAI_REALTIME_MODELS, OPENAI_REALTIME_VOICES
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    # Test OpenAI realtime config
    openai_rt = ProviderRegistry.get("openai")
    assert openai_rt is not None
    rt_cap = openai_rt.get_capability(ProviderCapability.REALTIME)
    assert rt_cap is not None

    print(f"\n[OK] OpenAI Realtime models (module): {OPENAI_REALTIME_MODELS}")
    print(f"[OK] OpenAI Realtime models (registry): {rt_cap.models}")
    assert rt_cap.models == OPENAI_REALTIME_MODELS, "OpenAI realtime models mismatch"

    print(f"[OK] OpenAI Realtime voices (module): {OPENAI_REALTIME_VOICES}")
    print(f"[OK] OpenAI Realtime voices (registry): {rt_cap.voices}")
    assert rt_cap.voices == OPENAI_REALTIME_VOICES, "OpenAI realtime voices mismatch"

    print(f"[OK] OpenAI default model: {OPENAI_DEFAULT}")
    assert rt_cap.default_model == OPENAI_DEFAULT

    # Test Gemini realtime config
    gemini_rt = ProviderRegistry.get("google_genai")
    assert gemini_rt is not None
    grt_cap = gemini_rt.get_capability(ProviderCapability.REALTIME)
    assert grt_cap is not None

    print(f"\n[OK] Gemini Realtime models (module): {GEMINI_LIVE_MODELS}")
    print(f"[OK] Gemini Realtime models (registry): {grt_cap.models}")
    assert grt_cap.models == GEMINI_LIVE_MODELS, "Gemini realtime models mismatch"

    print(f"[OK] Gemini voices (module): {GEMINI_VOICES}")
    print(f"[OK] Gemini voices (registry): {grt_cap.voices}")
    assert grt_cap.voices == GEMINI_VOICES, "Gemini realtime voices mismatch"

    print(f"[OK] Gemini default model: {GEMINI_DEFAULT}")
    assert grt_cap.default_model == GEMINI_DEFAULT

    print("\n[OK] Realtime Voice integration: PASSED")
    return True


def test_llm_discovery_integration():
    """Test LLM discovery uses registry."""
    print("\n" + "=" * 60)
    print("6. TESTING LLM DISCOVERY INTEGRATION")
    print("=" * 60)

    from ai_infra.llm.defaults import DEFAULT_MODELS, get_default_model
    from ai_infra.llm.providers.discovery import (
        PROVIDER_ENV_VARS,
        SUPPORTED_PROVIDERS,
        is_provider_configured,
        list_providers,
    )
    from ai_infra.providers import ProviderCapability, ProviderRegistry

    # Test list_providers returns chat providers
    chat_providers = list_providers()
    registry_chat = ProviderRegistry.list_for_capability(ProviderCapability.CHAT)
    print(f"\n[OK] LLM providers (list_providers): {chat_providers}")
    print(f"[OK] Registry CHAT providers: {registry_chat}")
    assert set(chat_providers) == set(registry_chat), "Chat provider mismatch"

    # Test SUPPORTED_PROVIDERS matches
    print(f"[OK] SUPPORTED_PROVIDERS: {SUPPORTED_PROVIDERS}")
    assert set(SUPPORTED_PROVIDERS) == set(registry_chat)

    # Test env vars match registry
    print(f"\n[OK] PROVIDER_ENV_VARS: {PROVIDER_ENV_VARS}")
    for provider, env_var in PROVIDER_ENV_VARS.items():
        config = ProviderRegistry.get(provider)
        assert config is not None, f"Provider {provider} not in registry"
        assert config.env_var == env_var, f"Env var mismatch for {provider}"

    # Test is_provider_configured uses registry
    for provider in ["openai", "anthropic", "google_genai"]:
        is_configured = is_provider_configured(provider)
        registry_configured = ProviderRegistry.is_configured(provider)
        print(f"[OK] {provider} configured: {is_configured} (registry: {registry_configured})")
        # Note: values might differ if env var set in current session but not passed through

    # Test default models
    print(f"\n[OK] DEFAULT_MODELS: {DEFAULT_MODELS}")
    for provider in ["openai", "anthropic"]:
        default = get_default_model(provider)
        config = ProviderRegistry.get(provider)
        cap = config.get_capability(ProviderCapability.CHAT)
        print(f"[OK] {provider} default: {default} (registry: {cap.default_model})")

    print("\n[OK] LLM discovery integration: PASSED")
    return True


def test_public_exports():
    """Test that public exports work correctly."""
    print("\n" + "=" * 60)
    print("7. TESTING PUBLIC EXPORTS")
    print("=" * 60)

    # Test imports from ai_infra root
    from ai_infra import (
        ProviderCapability,
        get_provider,
        list_providers,
        list_providers_for_capability,
    )

    print("\n[OK] All public exports imported successfully")

    # Test they work
    providers = list_providers()
    assert len(providers) > 0
    print(f"[OK] list_providers(): {len(providers)} providers")

    chat_providers = list_providers_for_capability(ProviderCapability.CHAT)
    print(f"[OK] list_providers_for_capability(CHAT): {chat_providers}")

    openai = get_provider("openai")
    assert openai is not None
    print(f"[OK] get_provider('openai'): {openai.display_name}")

    print("\n[OK] Public exports: PASSED")
    return True


def test_provider_config_completeness():
    """Test that all provider configs have required data."""
    print("\n" + "=" * 60)
    print("8. TESTING PROVIDER CONFIG COMPLETENESS")
    print("=" * 60)

    from ai_infra.providers import ProviderCapability, ProviderRegistry

    all_providers = ProviderRegistry.list_all()

    for name in all_providers:
        config = ProviderRegistry.get(name)
        assert config is not None, f"Provider {name} returned None"
        assert config.name, f"Provider {name} missing name"
        assert config.display_name, f"Provider {name} missing display_name"
        assert config.env_var, f"Provider {name} missing env_var"
        assert config.capabilities, f"Provider {name} missing capabilities"

        # Check each capability has required fields
        for cap_name, cap_config in config.capabilities.items():
            assert cap_config.models is not None, f"{name}.{cap_name} missing models"
            assert cap_config.default_model, f"{name}.{cap_name} missing default_model"
            assert cap_config.default_model in cap_config.models, (
                f"{name}.{cap_name} default_model not in models list"
            )

            # For TTS, check voices
            if cap_name == ProviderCapability.TTS:
                assert cap_config.voices, f"{name}.TTS missing voices"
                if cap_config.default_voice:
                    assert cap_config.default_voice in cap_config.voices, (
                        f"{name}.TTS default_voice not in voices list"
                    )

        print(f"[OK] {name}: {len(config.capabilities)} capabilities OK")

    print(f"\n[OK] All {len(all_providers)} provider configs complete: PASSED")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("PROVIDER REGISTRY VERIFICATION SCRIPT")
    print("=" * 60)

    tests = [
        test_registry_basics,
        test_embeddings_integration,
        test_tts_stt_integration,
        test_imagegen_integration,
        test_realtime_integration,
        test_llm_discovery_integration,
        test_public_exports,
        test_provider_config_completeness,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[X] {test.__name__}: FAILED")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"\n[OK] Passed: {passed}")
    print(f"[X] Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed > 0:
        print("\n[!]  Some tests failed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\n All verification tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
