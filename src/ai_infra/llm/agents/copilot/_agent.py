"""CopilotAgent — computer automation agent backed by the GitHub Copilot CLI."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from ai_infra.llm.agents.copilot._events import CopilotEvent, CopilotResult
from ai_infra.llm.agents.copilot._guard import (
    HAS_COPILOT,
    CopilotClient,
    SubprocessConfig,
    _missing_copilot,
)
from ai_infra.llm.agents.copilot._permissions import (
    _DESTRUCTIVE_TOOLS,
    _READ_ONLY_TOOLS,
    PermissionMode,
)
from ai_infra.llm.agents.copilot._tools import _CopilotTool

if TYPE_CHECKING:
    from ai_infra.callbacks import Callbacks

logger = logging.getLogger(__name__)

# PermissionHandler type alias (re-imported for use in annotations)
PermissionHandler = Callable[[str, dict[str, Any]], bool]


class CopilotAgent:
    """Computer automation agent backed by the GitHub Copilot CLI runtime.

    Delegates autonomous task execution to the Copilot CLI, which manages
    its own LLM calls, planning loop, and tool use. This is not an LLM
    provider — it is a runtime you direct with plain-language tasks.

    The agent lazily starts the CLI process on first use and reuses it
    across multiple ``run()`` / ``stream()`` calls. Use the async context
    manager for explicit lifecycle control.

    Example — one-liner task::

        from ai_infra import CopilotAgent

        result = await CopilotAgent(cwd="/my/project").run(
            "Add docstrings to every public function"
        )
        print(result.content)

    Example — multi-turn session (conversation memory)::

        agent = CopilotAgent()
        r1 = await agent.run("Explore the auth module", session_id="s1")
        r2 = await agent.run("Now add tests for it", session_id="s1")

    Example — streaming with live tool output::

        async for event in agent.stream("Run the test suite and fix failures"):
            if event.type == "token":
                print(event.content, end="", flush=True)
            elif event.type == "tool_start":
                print(f"\\n→ {event.tool}")
            elif event.type == "tool_output":
                print(event.content, end="", flush=True)

    Example — custom tool::

        @copilot_tool
        async def lookup_issue(id: str) -> str:
            "Fetch an issue from Linear."
            return await linear.get(id)

        agent = CopilotAgent(tools=[lookup_issue])
        await agent.run("Fix the bug described in issue LIN-42")

    Example — read-only analysis (no writes, no shell)::

        agent = CopilotAgent(permissions=PermissionMode.READ_ONLY)
        result = await agent.run("Audit for OWASP Top 10 vulnerabilities")

    Example — BYOK (no GitHub Copilot subscription required)::

        agent = CopilotAgent(
            model="gpt-4.1",
            provider={
                "type": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        )

    Example — with ai-infra callbacks (LoggingCallbacks, MetricsCallbacks, etc.)::

        from ai_infra.callbacks import LoggingCallbacks
        agent = CopilotAgent(callbacks=LoggingCallbacks())

    Example — context manager for explicit cleanup::

        async with CopilotAgent(cwd="/my/project") as agent:
            result = await agent.run("Refactor the database layer")

    Args:
        model: Model to use (e.g. ``"claude-sonnet-4-5"``, ``"gpt-4.1"``).
            When using BYOK ``provider``, this is required. Otherwise the
            Copilot CLI selects the default model.
        cwd: Working directory for the CLI process. Defaults to the current
            directory. Set this to the project root when automating file tasks.
        github_token: GitHub token for authentication. Takes priority over
            the ``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` environment variables
            and the CLI's stored login credentials.
        provider: BYOK provider configuration dict for non-GitHub LLM access.
            Supports ``"openai"``, ``"azure"``, and ``"anthropic"`` types.
            When set, ``model`` is required.
            Example: ``{"type": "openai", "base_url": "...", "api_key": "..."}``
        tools: Python callables decorated with ``@copilot_tool``, or raw
            Copilot SDK ``Tool`` objects, to expose to the agent.
        permissions: Controls which tools the agent may invoke.
            Pass a ``PermissionMode`` enum value for built-in policies, or
            a callable ``(tool_name: str, arguments: dict) -> bool`` for
            custom logic. Default: ``PermissionMode.AUTO_APPROVE``.
        skill_dirs: Directories containing skill subdirectories (each with a
            ``SKILL.md`` file). Skills are prompt modules injected as context.
        mcp_servers: MCP server configurations keyed by server name.
            Example: ``{"filesystem": {"type": "local", "command": "npx",
            "args": ["-y", "@mcp/server-filesystem", "/tmp"], "tools": ["*"]}}``
        custom_agents: Named sub-agent definitions for automatic task delegation.
            Each entry: ``{"name": str, "prompt": str, "tools": list[str]}``.
        system_message: Override the default system prompt for the session.
        streaming: Enable streaming delta events from the CLI (required for
            ``stream()``). Default: ``True``.
        infinite_context: Enable automatic context compaction when the
            context window approaches capacity. Default: ``True``.
        callbacks: ai-infra ``Callbacks`` instance (e.g. ``LoggingCallbacks()``,
            ``MetricsCallbacks()``). Copilot events are bridged to the standard
            ``ToolStartEvent``, ``ToolEndEvent``, ``LLMTokenEvent`` callbacks.
        reasoning_effort: Chain-of-thought depth for models that support it.
            One of ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``.
            Call ``list_models()`` to check which models accept this option.
        on_user_input: Async callable invoked when the agent needs to ask the
            user a question (enables Copilot's ``ask_user`` tool). Receives
            ``(request, invocation)`` where ``request["question"]`` is the
            question text and ``request.get("choices")`` is an optional list.
            Must return ``{"answer": str, "wasFreeform": bool}``.
        hooks: Power-user hook dict for session lifecycle interception. Keys
            are hook names (``"on_session_start"``, ``"on_user_prompt_submitted"``,
            ``"on_post_tool_use"``, ``"on_session_end"``, ``"on_error_occurred"``).
            Any ``"on_pre_tool_use"`` key here is merged with—and takes
            precedence over—the built-in permission hook derived from
            ``permissions``.
        telemetry: OpenTelemetry config for the CLI subprocess.
            Example: ``{"otlp_endpoint": "http://localhost:4318"}``.
            Providing this dict enables tracing — no separate flag required.
        external_server: URL of an already-running Copilot CLI server
            (e.g. ``"localhost:3000"``). When set, a new subprocess is not
            spawned and the ``cli_path`` / ``env`` / ``github_token`` params
            are ignored.
        cli_path: Absolute path to a specific ``copilot`` CLI binary.
            Defaults to the bundled binary from the SDK.
        env: Extra environment variables for the CLI subprocess.
            Merged with the inherited process environment.
        disabled_skills: Skill names to exclude when ``skill_dirs`` are
            provided (e.g. ``["slow-research", "web-fetch"]``).
        initial_agent: Name of the custom agent to pre-select when the
            session starts. Must match a ``name`` in ``custom_agents``.
            Equivalent to calling ``agent.select()`` after creation but
            avoids the extra round-trip.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        cwd: str | None = None,
        github_token: str | None = None,
        provider: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        permissions: PermissionMode | PermissionHandler = PermissionMode.AUTO_APPROVE,
        skill_dirs: list[str] | None = None,
        mcp_servers: dict[str, Any] | None = None,
        custom_agents: list[dict[str, Any]] | None = None,
        system_message: str | None = None,
        streaming: bool = True,
        infinite_context: bool = True,
        callbacks: Callbacks | None = None,
        # ---- extended config ----
        reasoning_effort: str | None = None,
        on_user_input: Callable | None = None,
        hooks: dict[str, Callable] | None = None,
        telemetry: dict[str, Any] | None = None,
        external_server: str | None = None,
        cli_path: str | None = None,
        env: dict[str, str] | None = None,
        disabled_skills: list[str] | None = None,
        initial_agent: str | None = None,
    ) -> None:
        if not HAS_COPILOT:
            _missing_copilot()

        self._model = model
        self._cwd = cwd
        self._github_token = github_token
        self._provider = provider
        self._raw_tools = tools or []
        self._permissions = permissions
        self._skill_dirs = skill_dirs
        self._mcp_servers = mcp_servers
        self._custom_agents = custom_agents
        self._system_message = system_message
        self._streaming = streaming
        self._infinite_context = infinite_context
        self._callbacks = callbacks
        self._reasoning_effort = reasoning_effort
        self._on_user_input = on_user_input
        self._extra_hooks = hooks
        self._telemetry = telemetry
        self._external_server = external_server
        self._cli_path = cli_path
        self._env = env
        self._disabled_skills = disabled_skills
        self._initial_agent = initial_agent

        self._client: Any = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._started:
            return

        if self._external_server is not None:
            # Connect to an already-running Copilot CLI server
            try:
                from copilot import ExternalServerConfig  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError(
                    "ExternalServerConfig requires 'github-copilot-sdk'. "
                    "Install with: pip install 'ai-infra[copilot]'"
                )
            server_config = ExternalServerConfig(url=self._external_server)
        else:
            server_config = SubprocessConfig(
                cwd=self._cwd,
                github_token=self._github_token,
                **({"cli_path": self._cli_path} if self._cli_path else {}),
                **({"env": self._env} if self._env else {}),
                **({"telemetry": self._telemetry} if self._telemetry else {}),
            )

        self._client = CopilotClient(server_config)
        await self._client.start()
        self._started = True
        logger.debug("CopilotAgent: CLI process started (cwd=%s)", self._cwd)

    async def stop(self) -> None:
        """Stop the underlying Copilot CLI process.

        Called automatically when using the async context manager. Call
        manually when you are done with a long-lived agent instance.
        """
        if self._started and self._client is not None:
            await self._client.stop()
            self._started = False
            logger.debug("CopilotAgent: CLI process stopped")

    async def __aenter__(self) -> CopilotAgent:
        await self._ensure_started()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_sdk_tools(self) -> list[Any]:
        sdk_tools = []
        for t in self._raw_tools:
            if isinstance(t, _CopilotTool):
                sdk_tools.append(t.to_sdk_tool())
            else:
                # Accept raw SDK Tool objects or any object with a .to_sdk_tool() method
                sdk_tools.append(t)
        return sdk_tools

    def _build_permission_hook(self) -> dict[str, Any] | None:
        """Build the hooks dict for on_pre_tool_use based on permission policy."""
        perm = self._permissions

        if perm == PermissionMode.AUTO_APPROVE or isinstance(perm, PermissionMode) is False:
            # Custom callable: wrap it in a pre-tool hook
            if callable(perm):
                handler = perm

                async def _custom_pre_hook(input_data: Any, _inv: Any) -> dict[str, Any]:
                    tool_name = input_data.get("toolName", "")
                    args = input_data.get("toolArgs") or {}
                    allowed = handler(tool_name, args)
                    if allowed:
                        return {"permissionDecision": "allow"}
                    return {
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Tool '{tool_name}' blocked by permission handler.",
                    }

                return {"on_pre_tool_use": _custom_pre_hook}
            return None  # AUTO_APPROVE: no hook needed

        if perm == PermissionMode.READ_ONLY:

            async def _read_only(input_data: Any, _inv: Any) -> dict[str, Any]:
                tool_name = input_data.get("toolName", "")
                if tool_name in _READ_ONLY_TOOLS or not tool_name:
                    return {"permissionDecision": "allow"}
                return {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"CopilotAgent is in READ_ONLY mode. '{tool_name}' is not permitted."
                    ),
                }

            return {"on_pre_tool_use": _read_only}

        if perm == PermissionMode.INTERACTIVE:

            async def _interactive(input_data: Any, _inv: Any) -> dict[str, Any]:
                tool_name = input_data.get("toolName", "")
                if tool_name in _DESTRUCTIVE_TOOLS:
                    return {"permissionDecision": "ask"}
                return {"permissionDecision": "allow"}

            return {"on_pre_tool_use": _interactive}

        if perm == PermissionMode.DENY_ALL:

            async def _deny_all(input_data: Any, _inv: Any) -> dict[str, Any]:
                tool_name = input_data.get("toolName", "unknown")
                return {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"CopilotAgent is in DENY_ALL mode. '{tool_name}' blocked."
                    ),
                }

            return {"on_pre_tool_use": _deny_all}

        return None

    def _build_session_config(self, session_id: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "on_permission_request": lambda _req, _inv: {"kind": "approved"},
            "streaming": self._streaming,
            "infinite_sessions": {"enabled": self._infinite_context},
        }
        if self._model:
            config["model"] = self._model
        if session_id:
            config["session_id"] = session_id
        if self._provider:
            config["provider"] = self._provider
        if self._system_message:
            config["system_message"] = {"content": self._system_message}
        if self._skill_dirs:
            config["skill_directories"] = self._skill_dirs
        if self._disabled_skills:
            config["disabled_skills"] = self._disabled_skills
        if self._mcp_servers:
            config["mcp_servers"] = self._mcp_servers
        if self._custom_agents:
            config["custom_agents"] = self._custom_agents
        if self._reasoning_effort:
            config["reasoning_effort"] = self._reasoning_effort
        if self._on_user_input:
            config["on_user_input_request"] = self._on_user_input
        if self._initial_agent:
            config["agent"] = self._initial_agent

        sdk_tools = self._build_sdk_tools()
        if sdk_tools:
            config["tools"] = sdk_tools

        # Merge permission hook with any user-supplied hooks
        perm_hooks = self._build_permission_hook()
        merged_hooks: dict[str, Any] = {}
        if perm_hooks:
            merged_hooks.update(perm_hooks)
        if self._extra_hooks:
            merged_hooks.update(self._extra_hooks)
        if merged_hooks:
            config["hooks"] = merged_hooks

        return config

    # ------------------------------------------------------------------
    # Callback bridge helpers
    # ------------------------------------------------------------------

    def _fire_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        if self._callbacks is None:
            return
        try:
            from ai_infra.callbacks import ToolStartEvent

            mgr = self._callbacks if hasattr(self._callbacks, "on_tool_start") else None
            if mgr:
                mgr.on_tool_start(ToolStartEvent(tool_name=tool_name, arguments=args))
        except Exception:
            pass

    def _fire_tool_end(self, tool_name: str, result: Any, latency_ms: float) -> None:
        if self._callbacks is None:
            return
        try:
            from ai_infra.callbacks import ToolEndEvent

            mgr = self._callbacks if hasattr(self._callbacks, "on_tool_end") else None
            if mgr:
                mgr.on_tool_end(
                    ToolEndEvent(tool_name=tool_name, result=result, latency_ms=latency_ms)
                )
        except Exception:
            pass

    def _fire_token(self, token: str) -> None:
        if self._callbacks is None:
            return
        try:
            from ai_infra.callbacks import LLMTokenEvent

            mgr = self._callbacks if hasattr(self._callbacks, "on_llm_token") else None
            if mgr:
                mgr.on_llm_token(
                    LLMTokenEvent(provider="copilot", model=self._model or "copilot", token=token)
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> CopilotResult:
        """Run a task and return the final result.

        The Copilot CLI autonomously determines which tools to invoke, in
        what order, and when the task is complete. This call blocks until
        the agent becomes idle (all tool calls finished, final response
        delivered).

        Args:
            prompt: Plain-language task description.
            session_id: Optional session ID to resume a prior conversation.
                If omitted, a fresh session is started.

        Returns:
            ``CopilotResult`` with the assistant's final response and metadata.

        Example::

            agent = CopilotAgent(cwd="/my/project")
            result = await agent.run("Fix all mypy errors")
            print(result.content)
        """
        started_at = time.monotonic()
        content = ""
        tools_called = 0
        actual_session_id = session_id or ""

        async for event in self._stream_internal(
            prompt, session_id=session_id, attachments=attachments
        ):
            if event.type == "token":
                content += event.content
            elif event.type == "tool_end":
                tools_called += 1
            elif event.type == "done":
                if event.content and not actual_session_id:
                    actual_session_id = event.content

        return CopilotResult(
            content=content,
            session_id=actual_session_id,
            tools_called=tools_called,
            duration_ms=(time.monotonic() - started_at) * 1000,
        )

    async def stream(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[CopilotEvent]:
        """Run a task and yield typed events as they arrive.

        Yields ``CopilotEvent`` instances in real time: text tokens as they
        stream, tool execution start/output/end, agent intent updates, and
        a final ``done`` event when the turn completes.

        Args:
            prompt: Plain-language task description.
            session_id: Optional session ID to resume a prior conversation.

        Yields:
            ``CopilotEvent`` instances in chronological order.

        Example::

            async for event in agent.stream("Refactor the auth module"):
                if event.type == "token":
                    print(event.content, end="", flush=True)
                elif event.type == "tool_start":
                    print(f"\\n→ {event.tool}({event.arguments})")
                elif event.type == "tool_output":
                    print(event.content, end="", flush=True)
                elif event.type == "tool_end":
                    print(f"  [{event.latency_ms:.0f}ms]")
        """
        async for event in self._stream_internal(
            prompt, session_id=session_id, attachments=attachments
        ):
            yield event

    async def _stream_internal(
        self,
        prompt: str,
        *,
        session_id: str | None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[CopilotEvent]:
        """Internal: bridge Copilot SDK callback events to an async iterator."""
        await self._ensure_started()

        queue: asyncio.Queue[CopilotEvent | None] = asyncio.Queue()
        tool_start_times: dict[str, float] = {}

        def _translate(raw: Any) -> CopilotEvent | None:
            event_type = str(raw.type.value) if hasattr(raw.type, "value") else str(raw.type)
            data = raw.data if hasattr(raw, "data") else {}

            def _get(attr: str, default: Any = "") -> Any:
                if hasattr(data, attr):
                    return getattr(data, attr) or default
                if isinstance(data, dict):
                    return data.get(attr, default)
                return default

            if event_type == "assistant.message_delta":
                token = _get("delta_content")
                if token:
                    self._fire_token(token)
                    return CopilotEvent(type="token", content=token)

            elif event_type == "assistant.message":
                # Final complete message — only surface if we are not streaming
                # (in streaming mode tokens were already delivered via delta events)
                if not self._streaming:
                    content = _get("content")
                    if content:
                        self._fire_token(content)
                        return CopilotEvent(type="token", content=content)

            elif event_type == "assistant.reasoning_delta":
                delta = _get("delta_content")
                if delta:
                    return CopilotEvent(
                        type="reasoning_delta", content=delta, reasoning_id=_get("reasoning_id")
                    )

            elif event_type == "assistant.reasoning":
                content = _get("content")
                if content:
                    return CopilotEvent(
                        type="reasoning", content=content, reasoning_id=_get("reasoning_id")
                    )

            elif event_type == "assistant.intent":
                intent = _get("intent")
                if intent:
                    return CopilotEvent(type="intent", content=intent)

            elif event_type == "tool.execution_start":
                tool_name = _get("tool_name")
                call_id = _get("tool_call_id", tool_name)
                args = _get("arguments", {})
                tool_start_times[call_id] = time.monotonic()
                self._fire_tool_start(tool_name, args if isinstance(args, dict) else {})
                return CopilotEvent(
                    type="tool_start",
                    tool=tool_name,
                    arguments=args if isinstance(args, dict) else {},
                )

            elif event_type == "tool.execution_partial_result":
                output = _get("partial_output")
                tool_name = _get("tool_name") or _get("tool_call_id", "")
                if output:
                    return CopilotEvent(type="tool_output", tool=tool_name, content=output)

            elif event_type == "tool.execution_complete":
                call_id = _get("tool_call_id", "")
                tool_name = _get("tool_name") or call_id
                start = tool_start_times.pop(call_id, time.monotonic())
                latency_ms = (time.monotonic() - start) * 1000
                result_obj = _get("result", {})
                if hasattr(result_obj, "content"):
                    result_str = result_obj.content or ""
                elif isinstance(result_obj, dict):
                    result_str = result_obj.get("content", "")
                else:
                    result_str = str(result_obj) if result_obj else ""
                self._fire_tool_end(tool_name, result_str, latency_ms)
                return CopilotEvent(
                    type="tool_end",
                    tool=tool_name,
                    result=result_str,
                    latency_ms=latency_ms,
                )

            elif event_type == "assistant.turn_start":
                return CopilotEvent(type="turn_start", turn_id=_get("turn_id"))

            elif event_type == "assistant.turn_end":
                return CopilotEvent(type="turn_end", turn_id=_get("turn_id"))

            elif event_type == "assistant.usage":
                return CopilotEvent(
                    type="usage",
                    input_tokens=int(_get("input_tokens", 0)),
                    output_tokens=int(_get("output_tokens", 0)),
                    cost=float(_get("cost", 0.0)),
                    content=_get("model", ""),
                )

            elif event_type == "tool.execution_progress":
                progress = _get("progress_message")
                tool_name = _get("tool_name") or _get("tool_call_id", "")
                if progress:
                    return CopilotEvent(type="tool_output", tool=tool_name, content=progress)

            elif event_type == "session.task_complete":
                return CopilotEvent(type="task_complete", content=_get("summary", ""))

            elif event_type == "session.context_changed":
                return CopilotEvent(
                    type="context",
                    cwd=_get("cwd"),
                    branch=_get("branch"),
                )

            elif event_type == "session.compaction_start":
                return CopilotEvent(type="compaction", compaction_phase="start")

            elif event_type == "session.compaction_complete":
                return CopilotEvent(
                    type="compaction",
                    compaction_phase="complete",
                    tokens_removed=int(_get("tokens_removed", 0)),
                    content=_get("summary_content", ""),
                )

            elif event_type in (
                "subagent.selected",
                "subagent.started",
                "subagent.completed",
                "subagent.failed",
                "subagent.deselected",
            ):
                phase = event_type.split(".", 1)[-1]  # e.g. "started"
                return CopilotEvent(
                    type="subagent",
                    subagent_name=_get("name") or _get("agent_name", ""),
                    subagent_phase=phase,
                )

            elif event_type == "session.error":
                return CopilotEvent(
                    type="error",
                    error=f"[{_get('error_type', 'error')}] {_get('message')}",
                )

            elif event_type == "session.idle":
                return CopilotEvent(type="done")

            return None

        def _on_event(raw: Any) -> None:
            translated = _translate(raw)
            if translated is not None:
                if translated.type == "done":
                    queue.put_nowait(translated)
                    queue.put_nowait(None)  # sentinel
                else:
                    queue.put_nowait(translated)

        session_config = self._build_session_config(session_id)
        session = await self._client.create_session(session_config)
        session.on(_on_event)

        try:
            send_payload: dict[str, Any] = {"prompt": prompt}
            if attachments:
                send_payload["attachments"] = attachments
            await session.send(send_payload)

            while True:
                item = await queue.get()
                if item is None:
                    break
                # Stash the actual session_id on the done event for run() to read back
                if item.type == "done" and not item.content:
                    try:
                        item.content = getattr(session, "session_id", "") or ""
                    except Exception:
                        pass
                yield item
        finally:
            await session.disconnect()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """Return all models available to this agent.

        Useful for inspecting what models the Copilot CLI supports for the
        current authentication context.

        Returns:
            List of model identifier strings (e.g. ``["gpt-4.1", "claude-sonnet-4-5"]``).
        """
        await self._ensure_started()
        models = await self._client.list_models()
        if isinstance(models, list):
            return [m if isinstance(m, str) else getattr(m, "id", str(m)) for m in models]
        return []

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all persisted sessions on this machine.

        Each entry is a dict with at minimum ``sessionId`` and ``createdAt``
        fields. Useful for session management UIs and cleanup routines.

        Returns:
            List of session metadata dicts.

        Example::

            sessions = await agent.list_sessions()
            for s in sessions:
                print(s["sessionId"], s.get("createdAt"))
        """
        await self._ensure_started()
        result = await self._client.list_sessions()
        if isinstance(result, list):
            return [s if isinstance(s, dict) else vars(s) for s in result]
        return []

    async def delete_session(self, session_id: str) -> None:
        """Permanently delete a persisted session and all its on-disk data.

        This is irreversible — the session cannot be resumed after deletion.
        To release in-memory resources while preserving state for later
        resumption, disconnect the session instead (handled automatically
        by ``run()`` / ``stream()``).

        Args:
            session_id: The session ID to delete.

        Example::

            await agent.delete_session("user-alice-pr-review-42")
        """
        await self._ensure_started()
        await self._client.delete_session(session_id)


__all__ = ["CopilotAgent"]
