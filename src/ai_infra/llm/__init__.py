from ai_infra.llm.agent import (
    Agent,
    CompiledSubAgent,
    FilesystemMiddleware,
    SubAgent,
    SubAgentMiddleware,
)
from ai_infra.llm.agents.copilot import (
    CopilotAgent,
    CopilotEvent,
    CopilotResult,
    PermissionDeniedError,
    PermissionMode,
    copilot_tool,
)

# Phase 6.8 - Streaming
from ai_infra.llm.auth import (
    PROVIDER_ENV_VARS,
    add_provider_mapping,
    atemporary_api_key,
    get_provider_env_var,
    temporary_api_key,
)
from ai_infra.llm.base import BaseLLM
from ai_infra.llm.defaults import MODEL, PROVIDER
from ai_infra.llm.llm import LLM

# Phase 6.5 - Context Management (Primary API)
# Phase 6.4 - Memory Management
from ai_infra.llm.memory import (
    ContextResult,
    MemoryItem,
    MemoryStore,
    count_tokens,
    count_tokens_approximate,
    fit_context,
)
from ai_infra.llm.multimodal import (
    STT,
    TTS,
    AudioFormat,
    AudioOutput,
    AudioResponse,
    TranscriptionResult,
    Voice,
)
from ai_infra.llm.providers import Providers
from ai_infra.llm.realtime import (
    AudioChunk,
    RealtimeConfig,
    RealtimeError,
    RealtimeVoice,
    TranscriptDelta,
    VADMode,
    VoiceSession,
    realtime_voice,
)
from ai_infra.llm.session import (
    PendingAction,
    ResumeDecision,
    SessionResult,
    SessionStorage,
    generate_session_id,
    memory,
    postgres,
    sqlite,
)
from ai_infra.llm.streaming import (
    StreamConfig,
    StreamEvent,
    filter_event_for_visibility,
    should_emit_event,
)
from ai_infra.llm.tools import (
    ApprovalEvent,
    ApprovalEvents,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalRule,
    MultiApprovalRequest,
    ToolExecutionConfig,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    console_approval_handler,
    create_rule_based_handler,
    create_selective_handler,
    tools_from_functions,
)
from ai_infra.llm.utils.error_handler import (
    get_supported_kwargs,
    translate_provider_error,
    validate_kwargs,
)
from ai_infra.llm.utils.logging_hooks import (
    ErrorContext,
    LoggingHooks,
    RequestContext,
    ResponseContext,
)
from ai_infra.llm.utils.settings import ModelSettings
from ai_infra.llm.workspace import Workspace, workspace

__all__ = [
    "LLM",
    "Agent",
    "BaseLLM",
    "ModelSettings",
    "Providers",
    "PROVIDER",
    "MODEL",
    "tools_from_functions",
    # Multimodal
    "TTS",
    "STT",
    "AudioFormat",
    "AudioOutput",
    "AudioResponse",
    "TranscriptionResult",
    "Voice",
    # Error handling
    "translate_provider_error",
    "get_supported_kwargs",
    "validate_kwargs",
    # Logging hooks
    "LoggingHooks",
    "RequestContext",
    "ResponseContext",
    "ErrorContext",
    # Tool execution config and errors
    "ToolExecutionConfig",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    # Approval/HITL
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalRule",
    "MultiApprovalRequest",
    "console_approval_handler",
    "create_selective_handler",
    "create_rule_based_handler",
    # Events/Observability
    "ApprovalEvent",
    "ApprovalEvents",
    # Session management
    "SessionResult",
    "SessionStorage",
    "PendingAction",
    "ResumeDecision",
    "memory",
    "postgres",
    "sqlite",
    "generate_session_id",
    # DeepAgents types
    "SubAgent",
    "CompiledSubAgent",
    "SubAgentMiddleware",
    "FilesystemMiddleware",
    # Workspace abstraction
    "Workspace",
    "workspace",
    # Realtime Voice API
    "RealtimeVoice",
    "realtime_voice",
    "RealtimeConfig",
    "VADMode",
    "AudioChunk",
    "TranscriptDelta",
    "VoiceSession",
    "RealtimeError",
    # Phase 6.5 - Context Management (Primary API)
    "fit_context",
    "ContextResult",
    # Phase 6.4 - Memory Management
    "count_tokens",
    "count_tokens_approximate",
    "MemoryStore",
    "MemoryItem",
    # Phase 6.8 - Streaming
    "StreamEvent",
    "StreamConfig",
    "should_emit_event",
    "filter_event_for_visibility",
    # Phase 6.8 - Auth helpers
    "temporary_api_key",
    "atemporary_api_key",
    "add_provider_mapping",
    "get_provider_env_var",
    "PROVIDER_ENV_VARS",
    # CopilotAgent
    "CopilotAgent",
    "CopilotEvent",
    "CopilotResult",
    "PermissionDeniedError",
    "PermissionMode",
    "copilot_tool",
]
