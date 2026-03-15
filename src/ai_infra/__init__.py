import os

from dotenv import find_dotenv, load_dotenv

if not os.environ.get("AI_INFRA_ENV_LOADED"):
    load_dotenv(find_dotenv(usecwd=True))
    os.environ["AI_INFRA_ENV_LOADED"] = "1"

from ai_infra.callbacks import (  # Events; Built-in callbacks
    CallbackManager,
    Callbacks,
    GraphNodeEndEvent,
    GraphNodeErrorEvent,
    GraphNodeStartEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    LLMTokenEvent,
    LoggingCallbacks,
    MCPConnectEvent,
    MCPDisconnectEvent,
    MCPLoggingEvent,
    MCPProgressEvent,
    MetricsCallbacks,
    PrintCallbacks,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
)
from ai_infra.embeddings import Embeddings, MultimodalEmbeddings, VectorStore
from ai_infra.embeddings.vectorstore import Document, SearchResult

# Cross-cutting concerns
from ai_infra.errors import (
    AIInfraError,
    ConfigurationError,
    MCPError,
    OpenAPIError,
    ProviderError,
    ValidationError,
)

# Evaluation framework (Phase 11)
from ai_infra.eval import (
    Case,
    ContainsExpected,
    Dataset,
    EvaluationReport,
    Evaluator,
    EvaluatorContext,
    IsInstance,
    LengthInRange,
    RAGFaithfulness,
    SemanticSimilarity,
    ToolUsageEvaluator,
    evaluate_agent,
    evaluate_agent_async,
    evaluate_retriever,
    evaluate_retriever_async,
)
from ai_infra.graph import Graph
from ai_infra.imagegen import GeneratedImage, ImageGen, ImageGenProvider
from ai_infra.llm import (  # Realtime Voice API
    LLM,
    STT,
    TTS,
    Agent,
    AudioFormat,
    AudioOutput,
    AudioResponse,
    CopilotAgent,
    CopilotEvent,
    CopilotResult,
    PermissionDeniedError,
    PermissionMode,
    RealtimeConfig,
    RealtimeVoice,
    TranscriptionResult,
    VADMode,
    Voice,
    Workspace,
    copilot_tool,
    realtime_voice,
    workspace,
)

# Phase 6.8 - Streaming
from ai_infra.llm.auth import atemporary_api_key, temporary_api_key

# Phase 6.4 - Memory Management
# Phase 6.5 - Unified Context Management
from ai_infra.llm.memory import (
    ContextResult,
    MemoryItem,
    MemoryStore,
    count_tokens,
    count_tokens_approximate,
    fit_context,
)
from ai_infra.llm.personas import Persona
from ai_infra.llm.providers import Providers
from ai_infra.llm.streaming import StreamConfig, StreamEvent
from ai_infra.llm.tools.custom.retriever import (
    create_retriever_tool,
    create_retriever_tool_async,
)
from ai_infra.logging import configure_logging, get_logger
from ai_infra.mcp import (
    CachingInterceptor,
    MCPClient,
    MCPResource,
    MCPSecuritySettings,
    MCPServer,
    MCPToolCallRequest,
    PromptInfo,
    RateLimitInterceptor,
    ResourceInfo,
    RetryInterceptor,
    ToolCallInterceptor,
    clear_mcp_cache,
    load_mcp_tools_cached,
)
from ai_infra.mcp.server.tools import mcp_from_functions

# Provider Registry (Phase 4.11)
from ai_infra.providers import (
    CapabilityConfig,
    ProviderCapability,
    ProviderConfig,
    ProviderRegistry,
    get_provider,
    is_provider_configured,
    list_providers,
    list_providers_for_capability,
)
from ai_infra.replay import MemoryStorage, SQLiteStorage, WorkflowRecorder, replay
from ai_infra.retriever import Chunk as RetrieverChunk
from ai_infra.retriever import Retriever
from ai_infra.retriever import SearchResult as RetrieverSearchResult
from ai_infra.tools import (
    ProgressEvent,
    ProgressStream,
    progress,
    tool,
    tool_exclude,
    tools_from_models,
    tools_from_models_sql,
    tools_from_object,
    tools_from_object_with_properties,
)
from ai_infra.tracing import TracingCallbacks, configure_tracing, get_tracer, trace

# Validation
from ai_infra.validation import (
    validate_llm_params,
    validate_output,
    validate_provider,
    validate_temperature,
)

__all__ = [
    # Core
    "LLM",
    "Agent",
    "Graph",
    "MCPServer",
    "MCPClient",
    "Providers",
    "mcp_from_functions",
    # MCP Security Settings
    "MCPSecuritySettings",
    "TransportSecuritySettings",
    # MCP Advanced Features (Phase 6.6)
    "ToolCallInterceptor",
    "MCPToolCallRequest",
    "CachingInterceptor",
    "RetryInterceptor",
    "RateLimitInterceptor",
    "PromptInfo",
    "MCPResource",
    "ResourceInfo",
    # Provider Registry (Phase 4.11)
    "ProviderRegistry",
    "ProviderCapability",
    "ProviderConfig",
    "CapabilityConfig",
    "get_provider",
    "list_providers",
    "list_providers_for_capability",
    "is_provider_configured",
    # Multimodal (TTS, STT, Audio)
    "TTS",
    "STT",
    "AudioFormat",
    "AudioOutput",
    "AudioResponse",
    "TranscriptionResult",
    "Voice",
    # Embeddings
    "Embeddings",
    "MultimodalEmbeddings",
    "VectorStore",
    "Document",
    "SearchResult",
    # Retriever
    "Retriever",
    "RetrieverSearchResult",  # SearchResult from retriever (has to_dict, convenience props)
    "RetrieverChunk",  # Chunk model from retriever
    "create_retriever_tool",
    "create_retriever_tool_async",
    # Errors
    "AIInfraError",
    "ProviderError",
    "MCPError",
    "OpenAPIError",
    "ValidationError",
    "ConfigurationError",
    # Logging
    "configure_logging",
    "get_logger",
    # Callbacks (Unified Callback System - Phase 6.7)
    "Callbacks",
    "CallbackManager",
    # Callback Events
    "LLMStartEvent",
    "LLMEndEvent",
    "LLMErrorEvent",
    "LLMTokenEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "ToolErrorEvent",
    "MCPConnectEvent",
    "MCPDisconnectEvent",
    "MCPProgressEvent",
    "MCPLoggingEvent",
    "GraphNodeStartEvent",
    "GraphNodeEndEvent",
    "GraphNodeErrorEvent",
    # Built-in Callbacks
    "LoggingCallbacks",
    "MetricsCallbacks",
    "PrintCallbacks",
    # Tracing
    "get_tracer",
    "configure_tracing",
    "trace",
    "TracingCallbacks",
    # Validation
    "validate_llm_params",
    "validate_output",
    "validate_provider",
    "validate_temperature",
    # Image Generation
    "ImageGen",
    "GeneratedImage",
    "ImageGenProvider",
    # Phase 4.7 - Zero Friction Integrations
    "tools_from_models",
    "tools_from_models_sql",
    "Persona",
    "replay",
    "WorkflowRecorder",
    "MemoryStorage",
    "SQLiteStorage",
    "progress",
    "ProgressStream",
    "ProgressEvent",
    # Phase 0.1 - Object Tools (tools_from_object)
    "tool",
    "tool_exclude",
    "tools_from_object",
    "tools_from_object_with_properties",
    # Phase 4.8 - Unified Workspace Architecture
    "Workspace",
    "workspace",
    # Phase 4.10 - Realtime Voice API
    "RealtimeVoice",
    "realtime_voice",
    "RealtimeConfig",
    "VADMode",
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
    # Phase 6.8 - Auth helpers
    "temporary_api_key",
    "atemporary_api_key",
    # Phase 6.8 - MCP tool loading
    "load_mcp_tools_cached",
    "clear_mcp_cache",
    # Evaluation Framework (Phase 11)
    "Case",
    "Dataset",
    "Evaluator",
    "EvaluatorContext",
    "EvaluationReport",
    "IsInstance",
    "SemanticSimilarity",
    "ToolUsageEvaluator",
    "RAGFaithfulness",
    "ContainsExpected",
    "LengthInRange",
    "evaluate_agent",
    "evaluate_agent_async",
    "evaluate_retriever",
    "evaluate_retriever_async",
    # CopilotAgent
    "CopilotAgent",
    "CopilotEvent",
    "CopilotResult",
    "PermissionMode",
    "PermissionDeniedError",
    "copilot_tool",
]
