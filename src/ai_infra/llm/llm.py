"""LLM class for direct model interaction.

This module provides the LLM class for simple chat-based interactions
without agent/tool capabilities.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from ai_infra.callbacks import CallbackManager, Callbacks

from ai_infra.llm.base import BaseLLM
from ai_infra.llm.tools import apply_output_gate, apply_output_gate_async
from ai_infra.llm.utils.error_handler import translate_provider_error
from ai_infra.llm.utils.logging_hooks import (
    ErrorContext,
    RequestContext,
    ResponseContext,
)
from ai_infra.llm.utils.structured import coerce_structured_result, is_pydantic_schema

from .utils import make_messages as _make_messages
from .utils import sanitize_model_kwargs
from .utils import with_retry as _with_retry_util


class LLM(BaseLLM):
    """Direct model convenience interface (no agent graph).

    The LLM class provides a simple API for chat-based interactions
    with language models. Use this when you don't need tool calling.

    Example - Basic usage:
        ```python
        llm = LLM()
        response = llm.chat("What is the capital of France?")
        print(response.content)  # "Paris is the capital of France."
        ```

    Example - With structured output:
        ```python
        from pydantic import BaseModel

        class Answer(BaseModel):
            city: str
            country: str

        llm = LLM()
        result = llm.chat(
            "What is the capital of France?",
            output_schema=Answer,
        )
        print(result.city)  # "Paris"
        ```

    Example - Streaming tokens:
        ```python
        llm = LLM()
        async for token, meta in llm.stream_tokens("Tell me a story"):
            print(token, end="", flush=True)
        ```
    """

    def __init__(
        self,
        *,
        callbacks: Callbacks | (CallbackManager | None) = None,
    ):
        """Initialize LLM with optional callbacks.

        Args:
            callbacks: Callback handler(s) for observing LLM events.
                Receives events for LLM calls (start, end, error, tokens).
                Can be a single Callbacks instance or a CallbackManager.
                Example: callbacks=MyCallbacks() or callbacks=CallbackManager([...])
        """
        super().__init__()
        # Use shared normalize_callbacks utility
        from ai_infra.callbacks import normalize_callbacks

        self._callbacks: CallbackManager | None = normalize_callbacks(callbacks)

    # =========================================================================
    # Discovery API - Static methods for provider/model discovery
    # =========================================================================

    @staticmethod
    def list_providers() -> list[str]:
        """
        List all supported provider names.

        Returns:
            List of provider names: ["openai", "anthropic", "google_genai", "xai"]

        Example:
            >>> LLM.list_providers()
            ['openai', 'anthropic', 'google_genai', 'xai']
        """
        from ai_infra.llm.providers.discovery import list_providers

        return list_providers()

    @staticmethod
    def list_configured_providers() -> list[str]:
        """
        List providers that have API keys configured.

        Returns:
            List of provider names with configured API keys.

        Example:
            >>> LLM.list_configured_providers()
            ['openai', 'anthropic']  # Only if these have API keys set
        """
        from ai_infra.llm.providers.discovery import list_configured_providers

        return list_configured_providers()

    @staticmethod
    def list_models(provider: str, *, refresh: bool = False) -> list[str]:
        """
        List available models for a specific provider.

        Fetches models dynamically from the provider's API.
        Results are cached for 1 hour by default.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            refresh: Force refresh from API, bypassing cache

        Returns:
            List of model IDs available from the provider.

        Raises:
            ValueError: If provider is not supported.
            RuntimeError: If provider is not configured (no API key).

        Example:
            >>> LLM.list_models("openai")
            ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...]
        """
        from ai_infra.llm.providers.discovery import list_models

        return list_models(provider, refresh=refresh)

    @staticmethod
    def list_all_models(*, refresh: bool = False) -> dict[str, list[str]]:
        """
        List models for all configured providers.

        Args:
            refresh: Force refresh from API, bypassing cache

        Returns:
            Dict mapping provider name to list of model IDs.

        Example:
            >>> LLM.list_all_models()
            {
                'openai': ['gpt-4o', 'gpt-4o-mini', ...],
                'anthropic': ['claude-sonnet-4-20250514', ...],
            }
        """
        from ai_infra.llm.providers.discovery import list_all_models

        return list_all_models(refresh=refresh)

    @staticmethod
    def is_provider_configured(provider: str) -> bool:
        """
        Check if a provider has its API key configured.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            True if the provider's API key environment variable is set.

        Example:
            >>> LLM.is_provider_configured("openai")
            True
        """
        from ai_infra.llm.providers.discovery import is_provider_configured

        return is_provider_configured(provider)

    # =========================================================================
    # Chat methods
    # =========================================================================

    def chat(
        self,
        user_msg: str,
        provider: str | None = None,
        model_name: str | None = None,
        system: str | None = None,
        extra: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        images: list[str | bytes | Path] | None = None,
        audio: Any | None = None,
        audio_output: Any | None = None,
        **model_kwargs,
    ):
        """Send a chat message and get a response.

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            extra: Extra options (e.g., {"retry": {"max_attempts": 3}})
            output_schema: Pydantic model for structured output
            output_method: How to extract structured output
            images: Optional list of images (URLs, bytes, or file paths) for vision
            audio: Optional audio input (URL, bytes, or file path) for audio understanding
            audio_output: Optional AudioOutput config to get audio response from model
            **model_kwargs: Additional model kwargs

        Returns:
            Response message or structured output if output_schema provided
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # If audio_output is requested, merge modalities into model_kwargs
        if audio_output is not None:
            from ai_infra.llm.multimodal.audio_output import AudioOutput

            if isinstance(audio_output, AudioOutput):
                model_kwargs.update(audio_output.to_openai_modalities())
            elif isinstance(audio_output, dict):
                model_kwargs.update(audio_output)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        self._logging_hooks.call_request_sync(request_ctx)
        start_time = time.time()

        # Build messages for callback (simplified)
        messages_for_callback = [{"role": "user", "content": user_msg}]
        if system:
            messages_for_callback.insert(0, {"role": "system", "content": system})

        # Fire LLM start callback
        if self._callbacks:
            from ai_infra.callbacks import LLMStartEvent

            self._callbacks.on_llm_start(
                LLMStartEvent(
                    provider=provider,
                    model=model_name or "",
                    messages=messages_for_callback,
                )
            )

        try:
            # PROMPT method uses shared helper
            if output_schema is not None and output_method == "prompt":
                res = self._prompt_structured_sync(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                # otherwise: existing structured (json_mode/function_calling/json_schema) or plain
                if output_schema is not None:
                    # output_method is not "prompt" here (handled above)
                    method_cast: Literal["json_schema", "json_mode", "function_calling"] | None = (
                        output_method  # type: ignore[assignment]
                    )
                    model = self.with_structured_output(
                        provider,
                        model_name,
                        output_schema,
                        method=method_cast,
                        **model_kwargs,
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(
                    user_msg, system, images=images, audio=audio, provider=provider
                )

                def _call():
                    return model.invoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = _call() if not retry_cfg else self._run_with_retry_sync(_call, retry_cfg)

            # Normalize Gemini's list content format to string
            # Gemini returns: [{'type': 'text', 'text': '...'}] instead of a string
            content = getattr(res, "content", None)
            if isinstance(content, list):
                normalized = self._extract_text_from_content(content)
                if normalized is not None:
                    # res is AIMessage at runtime (has content); typed as BaseModel for flexibility
                    res.content = normalized  # type: ignore[attr-defined]

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            self._logging_hooks.call_response_sync(response_ctx)

            # Fire LLM end callback
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                # Extract token usage if available
                usage = getattr(res, "usage_metadata", None)
                self._callbacks.on_llm_end(
                    LLMEndEvent(
                        provider=provider,
                        model=model_name or "",
                        response=getattr(res, "content", str(res)),
                        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                        latency_ms=duration_ms,
                    )
                )

            if output_schema is not None and is_pydantic_schema(output_schema):
                # is_pydantic_schema confirms output_schema is type[BaseModel]
                assert isinstance(output_schema, type)
                return coerce_structured_result(output_schema, res)

            try:
                return apply_output_gate(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            self._logging_hooks.call_error_sync(error_ctx)

            # Fire LLM error callback
            if self._callbacks:
                from ai_infra.callbacks import LLMErrorEvent

                self._callbacks.on_llm_error(
                    LLMErrorEvent(
                        provider=provider,
                        model=model_name or "",
                        error=e,
                        latency_ms=duration_ms,
                    )
                )

            # Translate provider error to ai-infra error
            raise translate_provider_error(e, provider=provider, model=model_name) from e

    async def achat(
        self,
        user_msg: str,
        provider: str | None = None,
        model_name: str | None = None,
        system: str | None = None,
        extra: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        output_method: (
            Literal["json_schema", "json_mode", "function_calling", "prompt"] | None
        ) = "prompt",
        images: list[str | bytes | Path] | None = None,
        audio: Any | None = None,
        audio_output: Any | None = None,
        **model_kwargs,
    ):
        """Async version of chat().

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            extra: Extra options (e.g., {"retry": {"max_attempts": 3}})
            output_schema: Pydantic model for structured output
            output_method: How to extract structured output
            images: Optional list of images (URLs, bytes, or file paths) for vision
            audio: Optional audio input (URL, bytes, or file path) for audio understanding
            audio_output: Optional AudioOutput config to get audio response from model
            **model_kwargs: Additional model kwargs

        Returns:
            Response message or structured output if output_schema provided
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        # If audio_output is requested, merge modalities into model_kwargs
        if audio_output is not None:
            from ai_infra.llm.multimodal.audio_output import AudioOutput

            if isinstance(audio_output, AudioOutput):
                model_kwargs.update(audio_output.to_openai_modalities())
            elif isinstance(audio_output, dict):
                model_kwargs.update(audio_output)

        # Create request context for logging hooks
        request_ctx = RequestContext(
            user_msg=user_msg,
            system=system,
            provider=provider,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        await self._logging_hooks.call_request_async(request_ctx)
        start_time = time.time()

        # Build messages for callback (simplified)
        messages_for_callback = [{"role": "user", "content": user_msg}]
        if system:
            messages_for_callback.insert(0, {"role": "system", "content": system})

        # Fire LLM start callback (async)
        if self._callbacks:
            from ai_infra.callbacks import LLMStartEvent

            await self._callbacks.on_llm_start_async(
                LLMStartEvent(
                    provider=provider,
                    model=model_name or "",
                    messages=messages_for_callback,
                )
            )

        try:
            if output_schema is not None and output_method == "prompt":
                res = await self._prompt_structured_async(
                    user_msg=user_msg,
                    system=system,
                    provider=provider,
                    model_name=model_name,
                    schema=output_schema,
                    extra=extra,
                    model_kwargs=model_kwargs,
                )
            else:
                if output_schema is not None:
                    # output_method is not "prompt" here (handled above)
                    method_cast: Literal["json_schema", "json_mode", "function_calling"] | None = (
                        output_method  # type: ignore[assignment]
                    )
                    model = self.with_structured_output(
                        provider,
                        model_name,
                        output_schema,
                        method=method_cast,
                        **model_kwargs,
                    )
                else:
                    model = self.set_model(provider, model_name, **model_kwargs)

                messages = _make_messages(
                    user_msg, system, images=images, audio=audio, provider=provider
                )

                async def _call():
                    return await model.ainvoke(messages)

                retry_cfg = (extra or {}).get("retry") if extra else None
                res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())

            # Normalize Gemini's list content format to string
            # Gemini returns: [{'type': 'text', 'text': '...'}] instead of a string
            content = getattr(res, "content", None)
            if isinstance(content, list):
                normalized = self._extract_text_from_content(content)
                if normalized is not None:
                    # res is AIMessage at runtime (has content); typed as BaseModel for flexibility
                    res.content = normalized  # type: ignore[attr-defined]

            # Call response hook
            duration_ms = (time.time() - start_time) * 1000
            response_ctx = ResponseContext(
                request=request_ctx,
                response=res,
                duration_ms=duration_ms,
                token_usage=getattr(res, "usage_metadata", None),
            )
            await self._logging_hooks.call_response_async(response_ctx)

            # Fire LLM end callback (async)
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                # Extract token usage if available
                usage = getattr(res, "usage_metadata", None)
                await self._callbacks.on_llm_end_async(
                    LLMEndEvent(
                        provider=provider,
                        model=model_name or "",
                        response=getattr(res, "content", str(res)),
                        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                        latency_ms=duration_ms,
                    )
                )

            if output_schema is not None and is_pydantic_schema(output_schema):
                # is_pydantic_schema confirms output_schema is type[BaseModel]
                assert isinstance(output_schema, type)
                return coerce_structured_result(output_schema, res)

            try:
                return await apply_output_gate_async(res, self._hitl)
            except Exception:
                return res

        except Exception as e:
            # Call error hook
            duration_ms = (time.time() - start_time) * 1000
            error_ctx = ErrorContext(
                request=request_ctx,
                error=e,
                duration_ms=duration_ms,
            )
            await self._logging_hooks.call_error_async(error_ctx)

            # Fire LLM error callback (async)
            if self._callbacks:
                from ai_infra.callbacks import LLMErrorEvent

                await self._callbacks.on_llm_error_async(
                    LLMErrorEvent(
                        provider=provider,
                        model=model_name or "",
                        error=e,
                        latency_ms=duration_ms,
                    )
                )

            # Translate provider error to ai-infra error
            raise translate_provider_error(e, provider=provider, model=model_name) from e

    async def stream_tokens(
        self,
        user_msg: str,
        provider: str | None = None,
        model_name: str | None = None,
        system: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        images: list[str | bytes | Path] | None = None,
        **model_kwargs,
    ):
        """Stream tokens from the model.

        Args:
            user_msg: The user's message
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)
            system: Optional system message
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            images: Optional list of images (URLs, bytes, or file paths) for vision
            **model_kwargs: Additional model kwargs

        Yields:
            Tuple of (token, metadata) for each token
        """
        sanitize_model_kwargs(model_kwargs)

        # Resolve provider and model (auto-detect if not specified)
        provider, model_name = self._resolve_provider_and_model(provider, model_name)

        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if top_p is not None:
            model_kwargs["top_p"] = top_p
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model = self.set_model(provider, model_name, **model_kwargs)
        messages = _make_messages(user_msg, system, images=images, provider=provider)

        # Build messages for callback (simplified)
        messages_for_callback = [{"role": "user", "content": user_msg}]
        if system:
            messages_for_callback.insert(0, {"role": "system", "content": system})

        # Fire LLM start callback before streaming
        if self._callbacks:
            from ai_infra.callbacks import LLMStartEvent

            await self._callbacks.on_llm_start_async(
                LLMStartEvent(
                    provider=provider,
                    model=model_name or "",
                    messages=messages_for_callback,
                )
            )

        start_time = time.time()

        try:
            async for event in model.astream(messages):
                text = getattr(event, "content", None)
                if text is None:
                    text = getattr(event, "delta", None) or getattr(event, "text", None)
                if text is None:
                    text = str(event)

                # Fire LLM token callback for each token
                if self._callbacks and text:
                    from ai_infra.callbacks import LLMTokenEvent

                    await self._callbacks.on_llm_token_async(
                        LLMTokenEvent(
                            provider=provider,
                            model=model_name or "",
                            token=text,
                        )
                    )

                meta = {"raw": event}
                yield text, meta

            # Fire LLM end callback after streaming completes
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                duration_ms = (time.time() - start_time) * 1000
                await self._callbacks.on_llm_end_async(
                    LLMEndEvent(
                        provider=provider,
                        model=model_name or "",
                        response="[streamed]",
                        latency_ms=duration_ms,
                    )
                )
        except Exception as e:
            # Fire LLM error callback on streaming error
            if self._callbacks:
                from ai_infra.callbacks import LLMErrorEvent

                duration_ms = (time.time() - start_time) * 1000
                await self._callbacks.on_llm_error_async(
                    LLMErrorEvent(
                        provider=provider,
                        model=model_name or "",
                        error=e,
                        latency_ms=duration_ms,
                    )
                )
            raise
