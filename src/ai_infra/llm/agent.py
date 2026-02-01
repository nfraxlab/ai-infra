"""Agent class for tool-using LLM agents.

This module provides the Agent class for running LLM agents with tools,
including support for sessions, human-in-the-loop approval, streaming,
and DeepAgents mode for autonomous multi-step task execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

if TYPE_CHECKING:
    from ai_infra.callbacks import CallbackManager, Callbacks
    from ai_infra.llm.streaming import StreamConfig
    from ai_infra.llm.workspace import Workspace

from ai_infra.llm.agents.callbacks import (
    wrap_tool_with_callbacks as _wrap_tool_with_callbacks_impl,
)

# =============================================================================
# DeepAgents Types (imported from agents submodule)
# =============================================================================
from ai_infra.llm.agents.deep import (
    AgentMiddleware,
    CompiledSubAgent,
    FilesystemMiddleware,
    SubAgent,
    SubAgentMiddleware,
)
from ai_infra.llm.agents.deep import (
    build_deep_agent as _build_deep_agent_impl,
)
from ai_infra.llm.base import BaseLLM
from ai_infra.llm.session import (
    ResumeDecision,
    SessionConfig,
    SessionResult,
    SessionStorage,
    generate_session_id,
    get_pending_action,
    is_paused,
)
from ai_infra.llm.tools import (
    ApprovalConfig,
    ToolExecutionConfig,
    apply_output_gate,
    apply_output_gate_async,
    wrap_tool_for_approval,
    wrap_tool_for_hitl,
    wrap_tool_with_execution_config,
)
from ai_infra.llm.tools.approval import ApprovalHandler, AsyncApprovalHandler
from ai_infra.llm.tools.tool_controls import ToolCallControls
from ai_infra.llm.utils.error_handler import translate_provider_error
from ai_infra.llm.utils.runtime_bind import (
    make_agent_with_context as rb_make_agent_with_context,
)

from .utils import arun_with_fallbacks as _arun_fallbacks_util
from .utils import is_valid_response as _is_valid_response
from .utils import merge_overrides as _merge_overrides
from .utils import run_with_fallbacks as _run_fallbacks_util
from .utils import with_retry as _with_retry_util

# Export DeepAgent types
__all__ = [
    "Agent",
    "AgentMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
]


class Agent(BaseLLM):
    """Agent-oriented interface (tool calling, streaming updates, fallbacks).

    The Agent class provides a simple API for running LLM agents with tools.
    Tools can be plain Python functions, LangChain tools, or MCP tools.

    Example - Basic usage:
        ```python
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny, 72°F"

        # Simple usage with tools
        agent = Agent(tools=[get_weather])
        result = agent.run("What's the weather in NYC?")
        ```

    Example - With session memory (conversations persist):
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(tools=[...], session=memory())

        # Conversation 1 - remembered
        agent.run("I'm Bob", session_id="user-123")
        agent.run("What's my name?", session_id="user-123")  # Knows "Bob"

        # Different session - fresh start
        agent.run("What's my name?", session_id="user-456")  # Doesn't know
        ```

    Example - Pause and resume (HITL):
        ```python
        from ai_infra.llm.session import memory

        agent = Agent(
            tools=[dangerous_tool],
            session=memory(),
            pause_before=["dangerous_tool"],  # Pause before this tool
        )

        result = agent.run("Delete file.txt", session_id="task-1")

        if result.paused:
            # Show user what's pending, get approval
            print(result.pending_action)

            # Resume with decision
            result = agent.resume(session_id="task-1", approved=True)
        ```

    Example - Production with Postgres:
        ```python
        from ai_infra.llm.session import postgres

        agent = Agent(
            tools=[...],
            session=postgres("postgresql://..."),
        )
        # Sessions persist across restarts
        ```

    Example - Human approval (sync, per-request):
        ```python
        agent = Agent(
            tools=[dangerous_tool],
            require_approval=True,  # Console prompt for approval
        )
        ```

    Example - DeepAgents mode (autonomous multi-step tasks):
        ```python
        from ai_infra.llm import Agent
        from ai_infra.llm.session import memory

        # Define specialized agents
        researcher = Agent(
            name="researcher",
            description="Searches and analyzes code",
            system="You are a code research assistant.",
            tools=[search_codebase],
        )

        writer = Agent(
            name="writer",
            description="Writes and edits documentation",
            system="You are a technical writer.",
        )

        # Create a deep agent that can delegate to subagents
        agent = Agent(
            deep=True,
            session=memory(),
            subagents=[researcher, writer],  # Agents auto-convert to subagents
        )

        # The agent can now autonomously:
        # - Read/write/edit files
        # - Execute shell commands
        # - Delegate to subagents
        # - Maintain todo lists
        result = agent.run("Refactor the auth module to use JWT tokens")
        ```
    """

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        tools: list[Any] | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        *,
        # Agent identity (used when this Agent is passed as a subagent)
        name: str | None = None,
        description: str | None = None,
        system: str | None = None,
        # Callbacks for observability
        callbacks: Callbacks | CallbackManager | None = None,
        # Tool execution config
        on_tool_error: Literal["return_error", "retry", "abort"] = "return_error",
        tool_timeout: float | None = None,
        validate_tool_results: bool = False,
        max_tool_retries: int = 1,
        # Approval config
        require_approval: bool | list[str] | Callable[[str, dict[str, Any]], bool] = False,
        approval_handler: ApprovalHandler | AsyncApprovalHandler | None = None,
        # Session config (for persistence and pause/resume)
        session: SessionStorage | None = None,
        pause_before: list[str] | None = None,
        pause_after: list[str] | None = None,
        # DeepAgents mode (autonomous multi-step task execution)
        deep: bool = False,
        subagents: list[Agent | SubAgent] | None = None,
        middleware: Sequence[AgentMiddleware] | None = None,
        response_format: Any | None = None,
        context_schema: type[Any] | None = None,
        use_longterm_memory: bool = False,
        # Workspace configuration
        workspace: str | Path | Workspace | None = None,
        # Safety limits
        recursion_limit: int = 50,
        **model_kwargs,
    ):
        """Initialize an Agent with optional tools and provider settings.

        Args:
            tools: List of tools (functions, LangChain tools, or MCP tools)
            provider: LLM provider (auto-detected if None)
            model_name: Model name (uses provider default if None)

            Agent Identity (for use as subagent):
                name: Agent name (required when used as a subagent)
                description: What this agent does (used by parent to decide delegation)
                system: System prompt / instructions for this agent

            Callbacks (observability):
                callbacks: Callback handler(s) for observing agent events.
                    Receives events for LLM calls (start, end, error, tokens)
                    and tool executions (start, end, error).
                    Can be a single Callbacks instance or a CallbackManager.
                    Example: callbacks=MyCallbacks() or callbacks=CallbackManager([...])

            Tool Execution:
                on_tool_error: How to handle tool execution errors:
                    - "return_error": Return error message to agent (default, allows recovery)
                    - "retry": Retry the tool call up to max_tool_retries times
                    - "abort": Re-raise the exception and stop execution
                tool_timeout: Timeout in seconds per tool call (None = no timeout)
                validate_tool_results: Validate tool results match return type annotations
                max_tool_retries: Max retry attempts when on_tool_error="retry" (default 1)

            Human Approval:
                require_approval: Tools that require human approval:
                    - False: No approval needed (default)
                    - True: All tools need approval
                    - List[str]: Only specified tools need approval
                    - Callable: Function(tool_name, args) -> bool for dynamic approval
                approval_handler: Custom approval handler function:
                    - If None and require_approval is True, uses console prompts
                    - Can be sync or async function taking ApprovalRequest

            Session & Persistence:
                session: Session storage backend for conversation memory and pause/resume.
                    Use memory() for development, postgres() for production.
                    Example: session=memory()
                pause_before: Tool names to pause before executing (requires session).
                    The agent will return a SessionResult with paused=True.
                pause_after: Tool names to pause after executing (requires session).

            DeepAgents Mode (autonomous multi-step tasks):
                deep: Enable DeepAgents mode for autonomous task execution.
                    When True, the agent has built-in tools for file operations
                    (ls, read_file, write_file, edit_file, glob, grep, execute),
                    todo management, and subagent orchestration.
                subagents: List of agents for delegation. Can be Agent instances
                    (automatically converted) or SubAgent dicts. Agent instances
                    must have name and description set.
                middleware: Additional middleware to apply to the deep agent.
                response_format: Structured output format for agent responses.
                context_schema: Schema for the deep agent context.
                use_longterm_memory: Enable long-term memory (requires session with store).

            Workspace Configuration:
                workspace: Workspace configuration for file operations. Can be:
                    - String/Path: Directory to sandbox file operations to
                    - Workspace: Full workspace config with mode ("virtual", "sandboxed", "full")
                    For deep agents, configures the filesystem backend.
                    For regular agents, configures proj_mgmt tools.
                    Example: workspace=".", workspace=Workspace(".", mode="sandboxed")

            Safety Limits:
                recursion_limit: Maximum number of agent iterations (default: 50).
                    Prevents infinite loops when agent keeps calling tools without
                    making progress. This is a critical safety measure to prevent
                    runaway token costs. Raise only if you have monitoring in place.

            **model_kwargs: Additional kwargs passed to the model
        """
        super().__init__()
        self._name = name
        self._description = description
        self._system = system
        self._default_provider = provider
        self._default_model_name = model_name
        self._default_model_kwargs = model_kwargs
        self._tool_execution_config = ToolExecutionConfig(
            on_error=on_tool_error,
            max_retries=max_tool_retries,
            timeout=tool_timeout,
            validate_results=validate_tool_results,
        )

        # Callbacks for observability - use shared normalize_callbacks utility
        from ai_infra.callbacks import normalize_callbacks

        self._callbacks: CallbackManager | None = normalize_callbacks(callbacks)

        # DeepAgents mode config
        self._deep = deep
        self._subagents = self._convert_subagents(subagents) if subagents else None
        self._middleware = middleware
        self._response_format = response_format
        self._context_schema = context_schema
        self._use_longterm_memory = use_longterm_memory

        # Safety limits
        self._recursion_limit = recursion_limit

        # Workspace configuration
        self._workspace: Workspace | None = None
        if workspace is not None:
            if isinstance(workspace, (str, Path)):
                from ai_infra.llm.workspace import Workspace as WorkspaceClass

                self._workspace = WorkspaceClass(workspace)
            else:
                self._workspace = workspace

            # Configure proj_mgmt tools for regular agents
            self._workspace.configure_proj_mgmt()

        # Persona support (set by from_persona)
        self._tool_filter: Callable[[str], bool] | None = None
        self._persona: Any | None = None  # Persona object if loaded

        # Set up approval config
        self._approval_config: ApprovalConfig | None = None
        if require_approval or approval_handler:
            # Determine if handler is async
            if approval_handler and asyncio.iscoroutinefunction(approval_handler):
                self._approval_config = ApprovalConfig(
                    require_approval=require_approval if require_approval else True,
                    approval_handler_async=approval_handler,
                )
            else:
                self._approval_config = ApprovalConfig(
                    require_approval=require_approval if require_approval else True,
                    approval_handler=approval_handler,  # type: ignore
                )

        # Set up session config for persistence and pause/resume
        self._session_config: SessionConfig | None = None
        if session:
            self._session_config = SessionConfig(
                storage=session,
                pause_before=pause_before or [],
                pause_after=pause_after or [],
            )

        if tools:
            self.set_global_tools(tools)

    def _wrap_tool_with_callbacks(self, tool: Any, callbacks: CallbackManager) -> Any:
        """Wrap a tool to fire callback events on start/end/error.

        For BaseTool subclasses, we wrap the _run/_arun methods directly.
        For plain functions, we wrap the function itself.

        Args:
            tool: The tool to wrap (can be a function or LangChain tool)
            callbacks: CallbackManager to dispatch events to

        Returns:
            Wrapped tool that fires callback events
        """
        return _wrap_tool_with_callbacks_impl(tool, callbacks)

    @classmethod
    def from_persona(
        cls,
        path: str | None = None,
        *,
        persona: Any | None = None,
        name: str | None = None,
        prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        deny: list[str] | None = None,
        approve: list[str] | None = None,
        **kwargs,
    ) -> Agent:
        """
        Create an Agent from a persona configuration.

        Personas define agent behavior, allowed tools, and safety constraints
        in a declarative format. Load from YAML files or configure inline.

        Args:
            path: Path to YAML persona file. If provided, loads configuration
                from file. File fields can be overridden by other arguments.
            persona: Pre-built Persona object. Alternative to path.
            name: Persona name (for logging/debugging)
            prompt: System prompt defining agent behavior
            allowed_tools: List of allowed tool names (whitelist). If provided,
                only these tools are available to the agent.
            deny: List of denied tool names (blacklist). These tools
                are blocked even if passed to the agent.
            approve: List of tools requiring human approval before execution.
                Maps to Agent's require_approval parameter.
            **kwargs: Additional Agent kwargs (provider, model_name, tools, etc.)
                The 'tools' kwarg passes actual tool objects to the Agent.

        Returns:
            Configured Agent instance

        Example - From YAML file:
            ```python
            # personas/analyst.yaml:
            # name: analyst
            # prompt: You are a data analyst...
            # tools: [query_database, create_chart]
            # deny: [delete_record, drop_table]
            # approve: [send_email]
            # temperature: 0.3

            agent = Agent.from_persona("personas/analyst.yaml")
            ```

        Example - Inline configuration:
            ```python
            agent = Agent.from_persona(
                name="analyst",
                prompt="You are a data analyst. Be precise and data-driven.",
                allowed_tools=["query_database", "create_chart"],
                deny=["delete_record", "drop_table"],
                approve=["send_email"],
                temperature=0.3,
            )
            ```

        Example - With Persona object and tools:
            ```python
            persona = Persona(name="db-admin", prompt="You manage databases.")
            tools = tools_from_models(User, Product)
            agent = Agent.from_persona(persona=persona, tools=tools)
            ```
        """
        from ai_infra.llm.personas import Persona as PersonaCls
        from ai_infra.llm.personas import build_tool_filter

        # Load persona from object, file, or build from kwargs
        if persona is not None:
            # Use the provided Persona object
            # Apply any overrides
            if name:
                persona.name = name
            if prompt:
                persona.prompt = prompt
            if allowed_tools:
                persona.tools = allowed_tools
            if deny:
                persona.deny = deny
            if approve:
                persona.approve = approve
        elif path:
            persona = PersonaCls.from_yaml(path)
            # Override with explicit arguments
            if name:
                persona.name = name
            if prompt:
                persona.prompt = prompt
            if allowed_tools:
                persona.tools = allowed_tools
            if deny:
                persona.deny = deny
            if approve:
                persona.approve = approve
        else:
            persona = PersonaCls(
                name=name or "custom",
                prompt=prompt or "",
                tools=allowed_tools,
                deny=deny,
                approve=approve,
            )

        # Build tool filter function
        tool_filter = build_tool_filter(persona.tools, persona.deny)

        # Extract persona's provider/model overrides
        agent_kwargs = {}
        if persona.provider:
            agent_kwargs["provider"] = persona.provider
        if persona.model_name:
            agent_kwargs["model_name"] = persona.model_name
        if persona.temperature is not None:
            agent_kwargs["temperature"] = persona.temperature
        if persona.max_tokens is not None:
            agent_kwargs["max_tokens"] = persona.max_tokens

        # Merge with caller-provided kwargs (caller takes precedence)
        agent_kwargs.update(kwargs)

        # Set up approval for persona.approve tools
        if persona.approve:
            agent_kwargs["require_approval"] = persona.approve

        # Create agent with system prompt
        # NOTE: agent_kwargs types are validated at runtime via persona config
        agent = cls(
            system=persona.prompt,
            name=persona.name,
            **agent_kwargs,  # type: ignore[arg-type]
        )

        # Store tool filter for runtime filtering
        agent._tool_filter = tool_filter
        agent._persona = persona

        return agent

    def run(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        system: str | None = None,
        session_id: str | None = None,
        **model_kwargs,
    ) -> str | SessionResult:
        """Run the agent with a simple prompt and return the response.

        Args:
            prompt: User prompt/message
            provider: Override provider (uses default if None)
            model_name: Override model (uses default if None)
            tools: Override tools (uses global tools if None)
            system: Optional system message
            session_id: Session ID for conversation persistence (requires session=...)
            **model_kwargs: Additional model kwargs

        Returns:
            str: The agent's final text response (if no session configured)
            SessionResult: Rich result with pause state (if session configured)

        Example - Basic:
            ```python
            agent = Agent(tools=[get_weather])
            result = agent.run("What's the weather in NYC?")
            print(result)  # "The weather in NYC is Sunny, 72°F"
            ```

        Example - With session:
            ```python
            from ai_infra.llm.session import memory

            agent = Agent(tools=[...], session=memory())
            result = agent.run("Hello", session_id="user-123")
            print(result.content)
            ```
        """
        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Merge model kwargs
        eff_kwargs = {**self._default_model_kwargs, **model_kwargs}

        # Get config for session
        config = None
        eff_session_id = session_id or generate_session_id()
        if self._session_config:
            config = self._session_config.get_config(eff_session_id)

        import time as _time

        start_time = _time.time()

        # Use DeepAgent if deep=True
        if self._deep:
            deep_agent = self._build_deep_agent(
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                system=system,
            )
            result = deep_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
            # Fire callback for deep agent path
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                input_tokens, output_tokens, total_tokens = self._extract_token_usage(result)
                self._callbacks.on_llm_end(
                    LLMEndEvent(
                        provider=eff_provider,
                        model=eff_model or "",
                        response=self._extract_text_content(result),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=(_time.time() - start_time) * 1000,
                    )
                )
        else:
            # Build messages
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Run agent
            result = self.run_agent(
                messages=messages,
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                model_kwargs=eff_kwargs,
                config=config,
            )

        # If session is configured, return SessionResult
        if self._session_config:
            return self._make_session_result(result, eff_session_id)

        # Extract text content from result (legacy behavior)
        return self._extract_text_content(result)

    def _extract_text_content(self, result: Any) -> str:
        """Extract text content from agent result."""
        if hasattr(result, "get") and "messages" in result:
            # LangGraph agent output format
            msgs = result["messages"]
            if msgs:
                last_msg = msgs[-1]
                return str(getattr(last_msg, "content", str(last_msg)))
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    def _extract_token_usage(self, result: Any) -> tuple[int | None, int | None, int | None]:
        """Extract token usage from agent result.

        LangGraph agents return messages with usage_metadata on AIMessage.
        We sum up usage from all AI messages in the response.

        Args:
            result: Agent result containing messages

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens), None if not available
        """
        if not hasattr(result, "get") or "messages" not in result:
            return None, None, None

        msgs = result.get("messages", [])
        if not msgs:
            return None, None, None

        total_input = 0
        total_output = 0
        total_tokens = 0
        has_usage = False

        for msg in msgs:
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                has_usage = True
                # usage_metadata can be dict or object
                if isinstance(usage, dict):
                    total_input += usage.get("input_tokens", 0) or 0
                    total_output += usage.get("output_tokens", 0) or 0
                    total_tokens += usage.get("total_tokens", 0) or 0
                else:
                    total_input += getattr(usage, "input_tokens", 0) or 0
                    total_output += getattr(usage, "output_tokens", 0) or 0
                    total_tokens += getattr(usage, "total_tokens", 0) or 0

        if has_usage:
            return total_input, total_output, total_tokens
        return None, None, None

    def _convert_subagents(self, subagents: list[Agent | Any]) -> list[Any]:
        """Convert Agent instances to SubAgent format.

        This allows users to pass Agent instances directly to the subagents
        parameter, and they will be automatically converted to the SubAgent
        format expected by deepagents.

        Args:
            subagents: List of Agent instances or SubAgent dicts

        Returns:
            List of SubAgent dicts
        """
        converted = []
        for agent in subagents:
            if isinstance(agent, Agent):
                # Convert Agent to SubAgent format
                if not agent._name:
                    raise ValueError(
                        "Agent used as subagent must have 'name' set. "
                        "Example: Agent(name='researcher', description='...', ...)"
                    )
                if not agent._description:
                    raise ValueError(
                        "Agent used as subagent must have 'description' set. "
                        "Example: Agent(name='researcher', description='Researches topics', ...)"
                    )

                subagent_dict: dict[str, Any] = {
                    "name": agent._name,
                    "description": agent._description,
                    "system_prompt": agent._system or "",
                    "tools": list(agent.tools) if agent.tools else [],
                }

                # Add optional model if specified
                if agent._default_provider or agent._default_model_name:
                    # Build model string or pass model kwargs
                    if agent._default_model_name:
                        subagent_dict["model"] = agent._default_model_name

                converted.append(subagent_dict)
            else:
                # Already a SubAgent dict, pass through
                converted.append(agent)
        return converted

    def _make_session_result(self, result: Any, session_id: str) -> SessionResult:
        """Convert agent result to SessionResult."""
        # Check if paused
        paused = is_paused(result)
        pending = get_pending_action(result) if paused else None

        # Extract messages
        messages = []
        if hasattr(result, "get") and "messages" in result:
            messages = result["messages"]

        # Extract content
        content = ""
        if not paused:
            content = self._extract_text_content(result)

        return SessionResult(
            content=content,
            paused=paused,
            pending_action=pending,
            session_id=session_id,
            messages=messages,
        )

    async def arun(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        system: str | None = None,
        session_id: str | None = None,
        **model_kwargs,
    ) -> str | SessionResult:
        """Async version of run().

        Args:
            prompt: User prompt/message
            provider: Override provider (uses default if None)
            model_name: Override model (uses default if None)
            tools: Override tools (uses global tools if None)
            system: Optional system message
            session_id: Session ID for conversation persistence (requires session=...)
            **model_kwargs: Additional model kwargs

        Returns:
            str: The agent's final text response (if no session configured)
            SessionResult: Rich result with pause state (if session configured)
        """
        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Merge model kwargs
        eff_kwargs = {**self._default_model_kwargs, **model_kwargs}

        # Get config for session
        config = None
        eff_session_id = session_id or generate_session_id()
        if self._session_config:
            config = self._session_config.get_config(eff_session_id)

        # Use DeepAgent if deep=True
        import time as _time

        start_time = _time.time()

        if self._deep:
            deep_agent = self._build_deep_agent(
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                system=system,
            )
            result = await deep_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
            # Fire callback for deep agent path
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                input_tokens, output_tokens, total_tokens = self._extract_token_usage(result)
                self._callbacks.on_llm_end(
                    LLMEndEvent(
                        provider=eff_provider,
                        model=eff_model or "",
                        response=self._extract_text_content(result),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=(_time.time() - start_time) * 1000,
                    )
                )
        else:
            # Build messages
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Run agent
            result = await self.arun_agent(
                messages=messages,
                provider=eff_provider,
                model_name=eff_model,
                tools=tools,
                model_kwargs=eff_kwargs,
                config=config,
            )

        # If session is configured, return SessionResult
        if self._session_config:
            return self._make_session_result(result, eff_session_id)

        # Extract text content from result (legacy behavior)
        return self._extract_text_content(result)

    async def astream(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        system: str | None = None,
        visibility: Literal["minimal", "standard", "detailed", "debug"] = "standard",
        stream_mode: str | list[str] = "messages",
        config: dict[str, Any] | None = None,
        stream_config: StreamConfig | None = None,
        **model_kwargs,
    ):
        """Stream agent responses as normalized, typed events.

        This is the recommended way to stream agent responses in applications.
        Unlike astream_agent_tokens() which yields raw LangChain message chunks,
        this method yields clean StreamEvent objects ready for any framework.

        Args:
            prompt: User message/prompt
            provider: LLM provider (uses default if None)
            model_name: Model name (uses default if None)
            tools: Override tools (uses global tools if None)
            system: System prompt

            visibility: Event detail level
                - "minimal": Only response tokens
                - "standard": + tool names and timing (default)
                - "detailed": + tool arguments
                - "debug": + tool result previews

            stream_mode: LangGraph stream mode(s) - passed through to underlying graph
                Supports: "messages", "values", "updates", "custom", "debug"

            config: Full LangGraph RunnableConfig for advanced use cases
                - config["configurable"]["thread_id"] for persistence
                - config["tags"] for tracing
                - Any other LangGraph config options

            stream_config: Advanced streaming configuration (StreamConfig)
            **model_kwargs: Passed through to LLM (temperature, max_tokens, etc.)

        Yields:
            StreamEvent objects with typed data

        Example - Basic usage:
            ```python
            agent = Agent(tools=[search_docs])

            async for event in agent.astream("What is the refund policy?"):
                if event.type == "token":
                    print(event.content, end="", flush=True)
            ```

        Example - With visibility:
            ```python
            async for event in agent.astream(
                "Search for authentication docs",
                visibility="detailed",  # Include tool arguments
            ):
                if event.type == "tool_start":
                    print(f"Calling {event.tool} with {event.arguments}")
            ```

        Example - FastAPI SSE endpoint:
            ```python
            @app.post("/chat")
            async def chat(message: str):
                async def generate():
                    async for event in agent.astream(message):
                        yield f"data: {json.dumps(event.to_dict())}\\n\\n"
                return StreamingResponse(generate(), media_type="text/event-stream")
            ```
        """
        import json
        import time as time_module

        from langchain_core.messages import AIMessageChunk, ToolMessage

        from ai_infra.llm.streaming import (
            StreamConfig,
            StreamEvent,
            filter_event_for_visibility,
            should_emit_event,
        )

        # Use stream_config or create default
        cfg = stream_config or StreamConfig(visibility=visibility)
        eff_visibility = cfg.visibility

        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Merge model kwargs
        eff_kwargs = {**self._default_model_kwargs, **model_kwargs}

        # Build messages
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # State for tool call accumulation
        pending_tool_calls: dict[str, float] = {}  # tool_call_id -> start_time
        emitted_tool_starts: set = set()  # Track which tool_starts we've already emitted
        accumulating_tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, args_str}
        tools_called = 0

        # Emit "thinking" event at start
        if cfg.include_thinking and should_emit_event("thinking", eff_visibility):
            yield StreamEvent(type="thinking", model=eff_model)

        # Stream tokens
        async for token, meta in self.astream_agent_tokens(
            messages=messages,
            provider=eff_provider,
            model_name=eff_model,
            tools=tools,
            model_kwargs=eff_kwargs,
            config=config,
        ):
            # Process AIMessageChunk for tool calls
            if isinstance(token, AIMessageChunk):
                tool_call_chunks = getattr(token, "tool_call_chunks", []) or []
                tool_calls = getattr(token, "tool_calls", []) or []

                # Process tool_call_chunks - accumulate args until complete
                for tc in tool_call_chunks:
                    if isinstance(tc, dict):
                        tc_index = tc.get("index", 0)
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args_chunk = tc.get("args", "")
                    else:
                        tc_index = getattr(tc, "index", 0)
                        tc_id = getattr(tc, "id", "")
                        tc_name = getattr(tc, "name", "")
                        tc_args_chunk = getattr(tc, "args", "")

                    # Initialize accumulator for this tool call index
                    if tc_index not in accumulating_tool_calls:
                        accumulating_tool_calls[tc_index] = {
                            "id": tc_id or "",
                            "name": tc_name or "",
                            "args_str": "",
                        }

                    # Accumulate data
                    acc = accumulating_tool_calls[tc_index]
                    if tc_id:
                        acc["id"] = tc_id
                    if tc_name:
                        acc["name"] = tc_name
                    if tc_args_chunk:
                        acc["args_str"] += tc_args_chunk

                    # Try to emit tool_start when complete
                    acc_id = acc["id"] or acc["name"] or str(tc_index)
                    acc_name = acc["name"]
                    args_str = acc["args_str"]

                    # Skip if no name yet or already emitted
                    if not acc_name or acc_id in emitted_tool_starts:
                        continue

                    # Try to parse args as JSON
                    tc_args: dict[str, Any] | None = None
                    if args_str.strip():
                        try:
                            tc_args = json.loads(args_str)
                            # Successfully parsed - emit tool_start
                            if cfg.include_tool_events and should_emit_event(
                                "tool_start", eff_visibility
                            ):
                                # Debug: Log what we're emitting
                                import logging

                                logger = logging.getLogger(__name__)
                                logger.info(
                                    f" Emitting tool_start: tool={acc_name}, visibility={eff_visibility}, has_args={tc_args is not None}, args={tc_args}"
                                )

                                event = StreamEvent(
                                    type="tool_start",
                                    tool=acc_name,
                                    tool_id=acc_id,
                                    arguments=(
                                        tc_args if eff_visibility in ("detailed", "debug") else None
                                    ),
                                )
                                yield filter_event_for_visibility(event, eff_visibility)

                            emitted_tool_starts.add(acc_id)
                            pending_tool_calls[acc_id] = time_module.time()
                            tools_called += 1
                        except json.JSONDecodeError:
                            # Args still incomplete - keep accumulating
                            pass
                    elif not args_str:
                        # Args not yet complete - DON'T emit tool_start yet
                        # Wait for fully-formed tool_calls which have complete arguments
                        # This prevents emitting tool_start with null arguments
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.debug(
                            f" Tool {acc_name} has empty args_str - waiting for complete tool_calls"
                        )
                        pass

                # Also handle fully-formed tool_calls
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id") or tc.get("name", "")
                        tc_name = tc.get("name", "unknown")
                        tc_args = tc.get("args", {})
                    else:
                        tc_id = getattr(tc, "id", "") or getattr(tc, "name", "")
                        tc_name = getattr(tc, "name", "unknown")
                        tc_args = getattr(tc, "args", {})

                    # Skip if no name, no args (incomplete chunk), or already emitted
                    # Empty args {} means LLM hasn't streamed the arguments yet
                    if not tc_name or tc_name == "unknown" or tc_id in emitted_tool_starts:
                        continue
                    if not tc_args or (isinstance(tc_args, dict) and len(tc_args) == 0):
                        # Skip incomplete tool calls with empty args (streaming in progress)
                        continue

                    # Parse args if string
                    if isinstance(tc_args, str) and tc_args.strip():
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            tc_args = {}
                    elif not isinstance(tc_args, dict):
                        tc_args = {}

                    if cfg.include_tool_events and should_emit_event("tool_start", eff_visibility):
                        event = StreamEvent(
                            type="tool_start",
                            tool=tc_name,
                            tool_id=tc_id,
                            arguments=tc_args if eff_visibility in ("detailed", "debug") else None,
                        )
                        yield filter_event_for_visibility(event, eff_visibility)

                    emitted_tool_starts.add(tc_id)
                    pending_tool_calls[tc_id] = time_module.time()
                    tools_called += 1

            # Handle tool results (ToolMessage)
            if isinstance(token, ToolMessage):
                tc_id = getattr(token, "tool_call_id", "")
                tc_name = getattr(token, "name", "tool")

                # Calculate latency
                start_time = pending_tool_calls.pop(tc_id, time_module.time())
                latency_ms = (time_module.time() - start_time) * 1000

                # Clear the accumulator for this tool's index
                idx_to_remove = None
                for idx, acc in accumulating_tool_calls.items():
                    if acc.get("id") == tc_id:
                        idx_to_remove = idx
                        break
                if idx_to_remove is not None:
                    del accumulating_tool_calls[idx_to_remove]

                # Emit tool_end
                if cfg.include_tool_events and should_emit_event("tool_end", eff_visibility):
                    # Get result from tool - may be string or structured dict
                    raw_result = token.content

                    # Detect structured results (from create_retriever_tool(structured=True))
                    is_structured = (
                        isinstance(raw_result, dict)
                        and "results" in raw_result
                        and "query" in raw_result
                    )

                    # Determine result output based on visibility
                    result_output: str | dict[str, Any] | None = None
                    result_structured = False
                    if eff_visibility in ("detailed", "debug"):
                        if is_structured:
                            # raw_result is confirmed to be a dict via is_structured check
                            result_output = raw_result  # type: ignore[assignment]
                            result_structured = True
                        else:
                            result_output = str(raw_result)

                    # Create truncated preview (debug only, text results only)
                    preview = None
                    if eff_visibility == "debug" and not is_structured:
                        result_str = str(raw_result)
                        if len(result_str) > cfg.tool_result_preview_length:
                            preview = result_str[: cfg.tool_result_preview_length] + "..."
                        else:
                            preview = result_str

                    event = StreamEvent(
                        type="tool_end",
                        tool=tc_name,
                        tool_id=tc_id,
                        latency_ms=round(latency_ms, 1),
                        result=result_output,
                        result_structured=result_structured,
                        preview=preview,
                    )
                    yield filter_event_for_visibility(event, eff_visibility)

                continue

            # Stream AI response content as token events
            content: str | None = None
            if isinstance(token, AIMessageChunk) and token.content:
                # Content can be str or list (Anthropic returns list of content blocks)
                content = self._extract_text_from_content(token.content)
            elif hasattr(token, "content") and token.content:
                content = self._extract_text_from_content(token.content)
            elif isinstance(token, str) and token:
                content = token

            if content and should_emit_event("token", eff_visibility):
                yield StreamEvent(type="token", content=content)

        # Emit done event
        if should_emit_event("done", eff_visibility):
            yield StreamEvent(type="done", tools_called=tools_called)

    def resume(
        self,
        session_id: str,
        *,
        approved: bool = True,
        modified_args: dict[str, Any] | None = None,
        reason: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> str | SessionResult:
        """Resume a paused agent with a decision.

        Args:
            session_id: The session ID of the paused agent
            approved: Whether to approve the pending action
            modified_args: Modified arguments (optional, if approved)
            reason: Reason for the decision
            provider: Override provider
            model_name: Override model

        Returns:
            str or SessionResult depending on session configuration

        Example:
            ```python
            # Agent was paused
            result = agent.run("Delete file.txt", session_id="task-1")

            if result.paused:
                # Resume with approval
                result = agent.resume(session_id="task-1", approved=True)
            ```
        """
        if not self._session_config:
            raise ValueError("resume() requires session= to be configured")

        from langgraph.types import Command

        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Build resume command
        decision = ResumeDecision(
            approved=approved,
            modified_args=modified_args,
            reason=reason,
        )

        config = self._session_config.get_config(session_id)

        # Get compiled agent
        agent, context = self._make_agent_with_context(
            eff_provider,
            eff_model,
            tools=None,  # Use global tools
            model_kwargs=self._default_model_kwargs,
        )

        # Resume with Command - inject recursion_limit into config
        merged_config = self._merge_recursion_limit_config(context, config)
        result = agent.invoke(
            Command(resume=decision.model_dump()),
            context=context,
            config=merged_config,
        )

        return self._make_session_result(result, session_id)

    async def aresume(
        self,
        session_id: str,
        *,
        approved: bool = True,
        modified_args: dict[str, Any] | None = None,
        reason: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> str | SessionResult:
        """Async version of resume().

        Args:
            session_id: The session ID of the paused agent
            approved: Whether to approve the pending action
            modified_args: Modified arguments (optional, if approved)
            reason: Reason for the decision
            provider: Override provider
            model_name: Override model

        Returns:
            str or SessionResult depending on session configuration
        """
        if not self._session_config:
            raise ValueError("aresume() requires session= to be configured")

        from langgraph.types import Command

        # Resolve provider and model
        eff_provider = provider or self._default_provider
        eff_model = model_name or self._default_model_name
        eff_provider, eff_model = self._resolve_provider_and_model(eff_provider, eff_model)

        # Build resume command
        decision = ResumeDecision(
            approved=approved,
            modified_args=modified_args,
            reason=reason,
        )

        config = self._session_config.get_config(session_id)

        # Get compiled agent
        agent, context = self._make_agent_with_context(
            eff_provider,
            eff_model,
            tools=None,  # Use global tools
            model_kwargs=self._default_model_kwargs,
        )

        # Resume with Command
        result = await agent.ainvoke(
            Command(resume=decision.model_dump()),
            context=context,
            config=config,
        )

        return self._make_session_result(result, session_id)

    def _make_agent_with_context(
        self,
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        # Capture callbacks for use in wrapper closure
        callbacks = self._callbacks

        # Build a composite tool wrapper that applies execution config, approval, HITL, and callbacks
        def _wrap_tool(t: Any) -> Any:
            # 1. Apply execution config (error handling, timeout, validation)
            wrapped = wrap_tool_with_execution_config(t, self._tool_execution_config)

            # 2. Apply new approval workflow if configured (recommended)
            if self._approval_config:
                wrapped = wrap_tool_for_approval(wrapped, self._approval_config)

            # 3. Apply legacy HITL if configured (for backward compatibility)
            if self._hitl.on_tool_call or self._hitl.on_tool_call_async:
                wrapped = wrap_tool_for_hitl(wrapped, self._hitl)

            # 4. Apply callback wrapper for observability
            if callbacks:
                wrapped = self._wrap_tool_with_callbacks(wrapped, callbacks)

            return wrapped

        # Extract session config if available
        checkpointer = None
        store = None
        interrupt_before = None
        interrupt_after = None
        if self._session_config:
            checkpointer = self._session_config.storage.get_checkpointer()
            store = self._session_config.storage.get_store()
            interrupt_before = self._session_config.pause_before or None
            interrupt_after = self._session_config.pause_after or None

        return rb_make_agent_with_context(
            self.registry,
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra=extra,
            model_kwargs=model_kwargs,
            tool_controls=tool_controls,
            require_explicit_tools=self.require_explicit_tools,
            global_tools=self.tools,
            # Apply execution config, approval, and HITL wrappers
            hitl_tool_wrapper=_wrap_tool,
            logger=self._logger,
            # Session/checkpoint config
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            # Safety limits - stored in context.extra for runtime config injection
            recursion_limit=self._recursion_limit,
        )

    def _merge_recursion_limit_config(
        self,
        context: Any,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge recursion_limit from context into config for runtime safety limits.

        LangGraph requires recursion_limit to be passed at runtime to invoke()/astream()
        via the config dict, NOT to create_react_agent().

        Args:
            context: ModelSettings context containing recursion_limit in extra
            config: User-provided config dict (may be None)

        Returns:
            Config dict with recursion_limit set
        """
        result = dict(config) if config else {}
        # Get recursion_limit from context.extra (set by make_agent_with_context)
        recursion_limit = None
        if hasattr(context, "extra") and context.extra:
            recursion_limit = context.extra.get("recursion_limit")
        # Only set if not already in config and we have a value
        if "recursion_limit" not in result and recursion_limit is not None:
            result["recursion_limit"] = recursion_limit
        return result

    def _build_deep_agent(
        self,
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        system: str | None = None,
    ) -> Any:
        """Build a DeepAgents agent for autonomous multi-step task execution.

        This method creates a deep agent using LangChain's deepagents package,
        which provides built-in file tools (ls, read, write, edit, glob, grep, execute),
        todo management, and subagent orchestration.

        Args:
            provider: LLM provider
            model_name: Model name
            tools: Additional tools (added to built-in deep agent tools)
            system: System prompt / additional instructions

        Returns:
            Compiled DeepAgent graph
        """
        # Get model instance from registry
        model = self._get_model_for_deep_agent(provider, model_name)

        # Merge global tools with provided tools
        all_tools = list(self.tools) if self.tools else []
        if tools:
            all_tools.extend(tools)

        # Delegate to the extracted build_deep_agent function
        return _build_deep_agent_impl(
            model=model,
            workspace=self._workspace,
            session_config=self._session_config,
            tools=all_tools if all_tools else None,
            system=system,
            middleware=self._middleware,
            subagents=self._subagents,
            response_format=self._response_format,
            context_schema=self._context_schema,
        )

    def _get_model_for_deep_agent(self, provider: str, model_name: str | None = None) -> Any:
        """Get a LangChain chat model instance for deep agent.

        Args:
            provider: LLM provider
            model_name: Model name

        Returns:
            BaseChatModel instance
        """
        # Resolve provider and model
        eff_provider, eff_model = self._resolve_provider_and_model(provider, model_name)

        # Get or create model from registry (same as get_model())
        return self.registry.get_or_create(eff_provider, eff_model, **self._default_model_kwargs)

    async def arun_agent(
        self,
        messages: list[dict[str, Any]],
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )

        # Fire LLM start callback
        if self._callbacks:
            from ai_infra.callbacks import LLMStartEvent

            self._callbacks.on_llm_start(
                LLMStartEvent(
                    provider=provider,
                    model=model_name or "",
                    messages=messages,
                    tools=[
                        {"name": getattr(t, "name", str(t))} for t in (tools or self.tools or [])
                    ],
                )
            )

        start_time = time.time()

        async def _call():
            return await agent.ainvoke({"messages": messages}, context=context, config=config)

        try:
            retry_cfg = (extra or {}).get("retry") if extra else None
            if retry_cfg:
                res = await _with_retry_util(_call, **retry_cfg)
            else:
                res = await _call()

            # Fire LLM end callback with token usage
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                input_tokens, output_tokens, total_tokens = self._extract_token_usage(res)
                self._callbacks.on_llm_end(
                    LLMEndEvent(
                        provider=provider,
                        model=model_name or "",
                        response=self._extract_text_content(res),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
        except Exception as e:
            # Fire LLM error callback
            if self._callbacks:
                from ai_infra.callbacks import LLMErrorEvent

                self._callbacks.on_llm_error(
                    LLMErrorEvent(
                        provider=provider,
                        model=model_name or "",
                        error=e,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
            # Translate provider errors to ai-infra errors
            raise translate_provider_error(e, provider=provider, model=model_name) from e
        ai_msg = await apply_output_gate_async(res, self._hitl)
        return ai_msg

    def run_agent(
        self,
        messages: list[dict[str, Any]],
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )

        # Fire LLM start callback
        if self._callbacks:
            from ai_infra.callbacks import LLMStartEvent

            self._callbacks.on_llm_start(
                LLMStartEvent(
                    provider=provider,
                    model=model_name or "",
                    messages=messages,
                    tools=[
                        {"name": getattr(t, "name", str(t))} for t in (tools or self.tools or [])
                    ],
                )
            )

        # Inject recursion_limit into config for safety
        merged_config = self._merge_recursion_limit_config(context, config)

        start_time = time.time()
        try:
            res = agent.invoke({"messages": messages}, context=context, config=merged_config)

            # Fire LLM end callback with token usage
            if self._callbacks:
                from ai_infra.callbacks import LLMEndEvent

                input_tokens, output_tokens, total_tokens = self._extract_token_usage(res)
                self._callbacks.on_llm_end(
                    LLMEndEvent(
                        provider=provider,
                        model=model_name or "",
                        response=self._extract_text_content(res),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
        except Exception as e:
            # Fire LLM error callback
            if self._callbacks:
                from ai_infra.callbacks import LLMErrorEvent

                self._callbacks.on_llm_error(
                    LLMErrorEvent(
                        provider=provider,
                        model=model_name or "",
                        error=e,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                )
            # Translate provider errors to ai-infra errors
            raise translate_provider_error(e, provider=provider, model=model_name) from e
        ai_msg = apply_output_gate(res, self._hitl)
        return ai_msg

    async def arun_agent_stream(
        self,
        messages: list[dict[str, Any]],
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        stream_mode: str | Sequence[str] = ("updates", "values"),
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)

        # Inject recursion_limit into config for safety
        merged_config = self._merge_recursion_limit_config(context, config)

        # Track token index for callbacks
        token_index = 0

        if modes == ["messages"]:
            async for token, meta in agent.astream(
                {"messages": messages},
                context=context,
                config=merged_config,
                stream_mode="messages",
            ):
                # Fire token callback
                if self._callbacks and hasattr(token, "content") and token.content:
                    from ai_infra.callbacks import LLMTokenEvent

                    self._callbacks.on_llm_token(
                        LLMTokenEvent(
                            provider=provider,
                            model=model_name or "",
                            token=str(token.content),
                            index=token_index,
                        )
                    )
                    token_index += 1
                yield token, meta
            return
        last_values = None
        async for mode, chunk in agent.astream(
            {"messages": messages},
            context=context,
            config=merged_config,
            stream_mode=modes,
        ):
            if mode == "values":
                last_values = chunk
                continue
            else:
                yield mode, chunk
        if last_values is not None:
            gated_values = await apply_output_gate_async(last_values, self._hitl)
            yield "values", gated_values

    async def astream_agent_tokens(
        self,
        messages: list[dict[str, Any]],
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        agent, context = self._make_agent_with_context(
            provider, model_name, tools, extra, model_kwargs, tool_controls
        )
        # Inject recursion_limit into config for safety
        merged_config = self._merge_recursion_limit_config(context, config)

        token_index = 0
        async for token, meta in agent.astream(
            {"messages": messages},
            context=context,
            config=merged_config,
            stream_mode="messages",
        ):
            # Fire token callback
            if self._callbacks and hasattr(token, "content") and token.content:
                from ai_infra.callbacks import LLMTokenEvent

                self._callbacks.on_llm_token(
                    LLMTokenEvent(
                        provider=provider,
                        model=model_name or "",
                        token=str(token.content),
                        index=token_index,
                    )
                )
                token_index += 1
            yield token, meta

    def agent(
        self,
        provider: str,
        model_name: str | None = None,
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        return self._make_agent_with_context(provider, model_name, tools, extra, model_kwargs)

    # ---------- fallbacks (sync) ----------
    def run_with_fallbacks(
        self,
        messages: list[dict[str, Any]],
        candidates: list[tuple[str, str]],
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        def _run_single(provider: str, model_name: str, overrides: dict[str, Any]):
            eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls = _merge_overrides(
                extra, model_kwargs, tools, tool_controls, overrides
            )
            return self.run_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=eff_tools,
                extra=eff_extra,
                model_kwargs=eff_model_kwargs,
                tool_controls=eff_tool_controls,
                config=config,
            )

        return _run_fallbacks_util(
            candidates=candidates,
            run_single=_run_single,
            validate=_is_valid_response,
        )

    # ---------- fallbacks (async) ----------
    async def arun_with_fallbacks(
        self,
        messages: list[dict[str, Any]],
        candidates: list[tuple[str, str]],
        tools: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tool_controls: ToolCallControls | dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        async def _run_single(provider: str, model_name: str, overrides: dict[str, Any]):
            eff_extra, eff_model_kwargs, eff_tools, eff_tool_controls = _merge_overrides(
                extra, model_kwargs, tools, tool_controls, overrides
            )
            return await self.arun_agent(
                messages=messages,
                provider=provider,
                model_name=model_name,
                tools=eff_tools,
                extra=eff_extra,
                model_kwargs=eff_model_kwargs,
                tool_controls=eff_tool_controls,
                config=config,
            )

        return await _arun_fallbacks_util(
            candidates=candidates,
            run_single_async=_run_single,
            validate=_is_valid_response,
        )
