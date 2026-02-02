"""Base LLM class with shared configuration and utilities.

This module provides the BaseLLM class that serves as the foundation
for both LLM (direct model interface) and Agent (tool-using agent).

Phase 16.5.8: Added provider auto-detection from model name patterns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ai_infra.llm.defaults import DEFAULT_MODELS
from ai_infra.llm.providers.discovery import get_default_provider
from ai_infra.llm.tools import HITLConfig
from ai_infra.llm.utils.logging_hooks import LoggingHooks
from ai_infra.llm.utils.runtime_bind import ModelRegistry
from ai_infra.llm.utils.structured import (
    build_structured_messages,
    coerce_from_text_or_fragment,
    structured_mode_call_async,
    structured_mode_call_sync,
    validate_or_raise,
)

from .utils import with_retry as _with_retry_util


def _infer_provider_from_model(model_name: str) -> str | None:
    """Infer provider from model name patterns.

    Enables auto-detection of the appropriate provider based on well-known
    model naming conventions. This allows users to specify just the model
    name without explicitly providing the provider.

    Args:
        model_name: Model name string (e.g., "claude-sonnet-4-20250514").

    Returns:
        Provider name if pattern matches, None otherwise.

    Examples:
        >>> _infer_provider_from_model("claude-sonnet-4-20250514")
        'anthropic'
        >>> _infer_provider_from_model("gpt-4o")
        'openai'
        >>> _infer_provider_from_model("my-custom-model")
        None
    """
    model_lower = model_name.lower()

    # Anthropic models (Claude family)
    if model_lower.startswith(("claude-", "claude_")):
        return "anthropic"

    # OpenAI models (GPT, o-series, DALL-E, Whisper, TTS)
    if model_lower.startswith(("gpt-", "o1-", "o3-", "o4-", "dall-e", "whisper", "tts-")):
        return "openai"

    # Google models (Gemini, PaLM)
    if model_lower.startswith(("gemini-", "palm-")):
        return "google_genai"

    # xAI models (Grok)
    if model_lower.startswith("grok-"):
        return "xai"

    return None


class BaseLLM:
    """Base class for LLM and Agent with shared configuration and utilities."""

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.registry = ModelRegistry()
        self.tools: list[Any] = []
        self._hitl = HITLConfig()
        self._logging_hooks = LoggingHooks()
        self.require_explicit_tools: bool = False

    # shared configuration / policies
    def set_global_tools(self, tools: list[Any]):
        self.tools = tools or []

    def require_tools_explicit(self, required: bool = True):
        self.require_explicit_tools = required

    def set_logging_hooks(
        self,
        *,
        on_request=None,
        on_response=None,
        on_error=None,
        on_request_async=None,
        on_response_async=None,
        on_error_async=None,
    ):
        """Configure request/response logging hooks.

        Args:
            on_request: Callback(RequestContext) called before model invocation
            on_response: Callback(ResponseContext) called after successful response
            on_error: Callback(ErrorContext) called when an error occurs
            on_request_async: Async version of on_request
            on_response_async: Async version of on_response
            on_error_async: Async version of on_error

        Example:
            ```python
            import logging
            logger = logging.getLogger(__name__)

            llm = LLM()
            llm.set_logging_hooks(
                on_request=lambda ctx: logger.info("Request to %s/%s", ctx.provider, ctx.model_name),
                on_response=lambda ctx: logger.info("Response in %.2fms", ctx.duration_ms),
                on_error=lambda ctx: logger.error("Error: %s", ctx.error),
            )
            ```
        """
        self._logging_hooks.set(
            on_request=on_request,
            on_response=on_response,
            on_error=on_error,
            on_request_async=on_request_async,
            on_response_async=on_response_async,
            on_error_async=on_error_async,
        )
        return self

    def set_hitl(
        self,
        *,
        on_model_output=None,
        on_tool_call=None,
        on_model_output_async=None,
        on_tool_call_async=None,
    ):
        self._hitl.set(
            on_model_output=on_model_output,
            on_tool_call=on_tool_call,
            on_model_output_async=on_model_output_async,
            on_tool_call_async=on_tool_call_async,
        )

    @staticmethod
    def make_sys_gate(autoapprove: bool = False):
        def gate(tool_name: str, args: dict):
            if autoapprove:
                return {"action": "pass"}
            print(f"\nTool request: {tool_name}\nArgs: {args}")
            try:
                ans = input("Approve? [y]es / [b]lock: ").strip().lower()
            except EOFError:
                return {"action": "block", "replacement": "[auto-block: no input]"}
            if ans.startswith("y"):
                return {"action": "pass"}
            return {"action": "block", "replacement": "[blocked by user]"}

        return gate

    # model registry
    def set_model(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **(kwargs or {}))

    def _get_or_create(self, provider: str, model_name: str, **kwargs):
        return self.registry.get_or_create(provider, model_name, **kwargs)

    def get_model(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        **model_kwargs,
    ):
        """
        Get the underlying LangChain chat model for direct use.

        Provides access to the raw LangChain model for advanced use cases
        that require direct model interaction.

        Args:
            provider: Provider name (e.g., "openai", "anthropic"). Auto-detected if None.
            model_name: Model name (e.g., "gpt-4o"). Uses provider default if None.
            **model_kwargs: Additional kwargs passed to the model constructor.

        Returns:
            LangChain BaseChatModel instance.

        Raises:
            ValueError: If no provider is specified and none can be auto-detected.

        Example:
            >>> llm = LLM()
            >>> model = llm.get_model()  # Auto-detect provider
            >>> model = llm.get_model("anthropic", "claude-3-5-sonnet-latest")
            >>> # Use LangChain model directly
            >>> response = model.invoke([HumanMessage(content="Hello")])
        """
        # Resolve provider and model (auto-detect if not specified)
        resolved_provider, resolved_model = self._resolve_provider_and_model(provider, model_name)
        return self.registry.get_or_create(resolved_provider, resolved_model, **model_kwargs)

    def with_structured_output(
        self,
        provider: str,
        model_name: str,
        schema: type[BaseModel] | dict[str, Any],
        *,
        method: Literal["json_schema", "json_mode", "function_calling"] | None = "json_mode",
        **model_kwargs,
    ):
        model = self.registry.get_or_create(provider, model_name, **model_kwargs)
        try:
            # Pass method through if provided (LangChain 0.3 supports this)
            return model.with_structured_output(
                schema, **({} if method is None else {"method": method})
            )
        except Exception as e:  # pragma: no cover
            self._logger.warning(
                "[LLM] Structured output unavailable; provider=%s model=%s schema=%s error=%s",
                provider,
                model_name,
                getattr(schema, "__name__", type(schema)),
                e,
                exc_info=True,
            )
            return model

    def _resolve_provider_and_model(
        self,
        provider: str | None,
        model_name: str | None,
    ) -> tuple[str, str]:
        """Resolve provider and model, auto-detecting from model name or environment.

        Resolution order (Phase 16.5.8):
        1. Parse "provider/model" format if present (e.g., "anthropic/claude-sonnet-4")
        2. Infer provider from model name patterns (e.g., "claude-*" -> anthropic)
        3. Fall back to default provider from environment variables

        Args:
            provider: Provider name or None to auto-detect
            model_name: Model name or None to use provider's default

        Returns:
            Tuple of (provider, model_name)

        Raises:
            ValueError: If no provider is specified and none can be auto-detected

        Examples:
            >>> llm._resolve_provider_and_model(None, "claude-sonnet-4-20250514")
            ('anthropic', 'claude-sonnet-4-20250514')
            >>> llm._resolve_provider_and_model(None, "anthropic/claude-sonnet-4")
            ('anthropic', 'claude-sonnet-4')
        """
        # Step 1: Parse "provider/model" format if present
        if provider is None and model_name is not None and "/" in model_name:
            parts = model_name.split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                provider = parts[0]
                model_name = parts[1]
                self._logger.debug(
                    "Parsed provider/model format: provider=%s model=%s",
                    provider,
                    model_name,
                )

        # Step 2: Infer provider from model name patterns
        if provider is None and model_name is not None:
            inferred = _infer_provider_from_model(model_name)
            if inferred:
                provider = inferred
                self._logger.info(
                    "Auto-detected provider '%s' for model '%s'",
                    provider,
                    model_name,
                )

        # Step 3: Fall back to default provider from environment
        if provider is None:
            provider = get_default_provider()
            if provider is None:
                raise ValueError(
                    "No LLM provider configured. Set one of these environment variables: "
                    "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or XAI_API_KEY. "
                    "Or explicitly pass provider='openai' (etc.) to the method."
                )

        # Use default model for provider if not specified
        if model_name is None:
            model_name = DEFAULT_MODELS.get(provider, "gpt-4o-mini")

        return provider, model_name

    def _extract_text_from_content(self, content: str | list) -> str | None:
        """Extract text from content that may be a string or list of content blocks.

        Anthropic returns streaming content as a list of content blocks like:
            [{'text': 'Hello', 'type': 'text', 'index': 0}]

        Other providers return content as a plain string.

        Args:
            content: Either a string or a list of content block dicts

        Returns:
            Extracted text string, or None if no text content found
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from content blocks (Anthropic format)
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
                elif isinstance(block, str):
                    texts.append(block)
            return "".join(texts) if texts else None
        # Fallback: convert to string
        return str(content) if content else None

    def _run_with_retry_sync(self, fn, retry_cfg):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            self._logger.warning(
                "[LLM] chat() retry config ignored due to running loop; use achat()."
            )
            return fn()

        async def _acall():
            return fn()

        return asyncio.run(_with_retry_util(_acall, **retry_cfg))

    # ========== PROMPT method helpers (shared by chat/achat) ==========
    def _prompt_structured_sync(
        self,
        *,
        user_msg: str,
        system: str | None,
        provider: str,
        model_name: str,
        schema: type[BaseModel] | dict[str, Any],
        extra: dict[str, Any] | None,
        model_kwargs: dict[str, Any],
    ) -> BaseModel:
        model = self.set_model(provider, model_name, **model_kwargs)
        messages: list[BaseMessage] = build_structured_messages(
            schema=schema, user_msg=user_msg, system_preamble=system
        )

        def _call():
            return model.invoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = _call() if not retry_cfg else self._run_with_retry_sync(_call, retry_cfg)
        content = getattr(res, "content", None) or str(res)

        # Try direct/fragment validation
        coerced = coerce_from_text_or_fragment(schema, content)
        if coerced is not None:
            if isinstance(coerced, BaseModel):
                return coerced
            # coerced is a dict - validate it
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")

        # Final fallback: provider structured mode (json_mode)
        try:
            result = structured_mode_call_sync(
                self.with_structured_output,
                provider,
                model_name,
                schema,
                messages,
                model_kwargs,
            )
            if isinstance(result, BaseModel):
                return result
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")
        except Exception:
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")

    async def _prompt_structured_async(
        self,
        *,
        user_msg: str,
        system: str | None,
        provider: str,
        model_name: str,
        schema: type[BaseModel] | dict[str, Any],
        extra: dict[str, Any] | None,
        model_kwargs: dict[str, Any],
    ) -> BaseModel:
        """Async variant of prompt-only structured output with robust JSON fallback."""
        model = self.set_model(provider, model_name, **model_kwargs)
        messages: list[BaseMessage] = build_structured_messages(
            schema=schema, user_msg=user_msg, system_preamble=system
        )

        async def _call():
            return await model.ainvoke(messages)

        retry_cfg = (extra or {}).get("retry") if extra else None
        res = await (_with_retry_util(_call, **retry_cfg) if retry_cfg else _call())
        content = getattr(res, "content", None) or str(res)

        # Handle Gemini's list content format: [{'type': 'text', 'text': '...'}]
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts) if text_parts else str(content)

        # Try direct/fragment validation
        coerced = coerce_from_text_or_fragment(schema, content)
        if coerced is not None:
            if isinstance(coerced, BaseModel):
                return coerced
            # coerced is a dict - validate it
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")

        # Final fallback: provider structured mode (json_mode)
        try:
            result = await structured_mode_call_async(
                self.with_structured_output,
                provider,
                model_name,
                schema,
                messages,
                model_kwargs,
            )
            if isinstance(result, BaseModel):
                return result
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")
        except Exception:
            validated = validate_or_raise(schema, content)
            if isinstance(validated, BaseModel):
                return validated
            raise ValueError(f"Expected BaseModel, got {type(validated)}")
