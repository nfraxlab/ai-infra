"""Unit tests for the CopilotAgent package.

Covers all sub-modules:
    _guard.py       - Optional dependency detection
    _permissions.py - PermissionMode enum and PermissionDeniedError
    _events.py      - CopilotEvent and CopilotResult dataclasses
    _tools.py       - copilot_tool decorator and _CopilotTool
    _agent.py       - CopilotAgent class (construction, config building,
                      permission hooks, session config, callback bridge,
                      and core API methods all fully mocked)

No external processes or network calls are made.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.agents.copilot._events import CopilotEvent, CopilotResult
from ai_infra.llm.agents.copilot._guard import HAS_COPILOT, _missing_copilot
from ai_infra.llm.agents.copilot._permissions import (
    _DESTRUCTIVE_TOOLS,
    _READ_ONLY_TOOLS,
    PermissionDeniedError,
    PermissionMode,
)
from ai_infra.llm.agents.copilot._tools import _CopilotTool, copilot_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(**kwargs: Any):
    """Return a CopilotAgent with the SDK guard bypassed."""
    from ai_infra.llm.agents.copilot._agent import CopilotAgent

    with patch("ai_infra.llm.agents.copilot._agent.HAS_COPILOT", True):
        return CopilotAgent(**kwargs)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


class TestGuard:
    def test_has_copilot_is_bool(self):
        assert isinstance(HAS_COPILOT, bool)

    def test_missing_copilot_raises(self):
        with pytest.raises(ImportError, match="github-copilot-sdk"):
            _missing_copilot()

    def test_placeholder_client_raises_without_sdk(self):
        """CopilotClient placeholder raises ImportError when SDK absent."""
        from ai_infra.llm.agents.copilot._guard import CopilotClient as _PC

        with patch("ai_infra.llm.agents.copilot._guard.HAS_COPILOT", False):
            # Directly instantiate the placeholder class (before real import)
            # We verify _missing_copilot is wired in via its raised message.
            with pytest.raises((ImportError, TypeError)):
                _PC.__init__(_PC.__new__(_PC))


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------


class TestPermissionMode:
    def test_enum_values(self):
        assert PermissionMode.AUTO_APPROVE == "auto-approve"
        assert PermissionMode.READ_ONLY == "read-only"
        assert PermissionMode.INTERACTIVE == "interactive"
        assert PermissionMode.DENY_ALL == "deny-all"

    def test_is_str_subclass(self):
        assert isinstance(PermissionMode.AUTO_APPROVE, str)

    def test_read_only_tools_frozenset(self):
        assert "read_file" in _READ_ONLY_TOOLS
        assert "grep" in _READ_ONLY_TOOLS
        assert "bash" not in _READ_ONLY_TOOLS

    def test_destructive_tools_frozenset(self):
        assert "bash" in _DESTRUCTIVE_TOOLS
        assert "write_file" in _DESTRUCTIVE_TOOLS
        assert "read_file" not in _DESTRUCTIVE_TOOLS

    def test_frozensets_are_disjoint(self):
        assert _READ_ONLY_TOOLS.isdisjoint(_DESTRUCTIVE_TOOLS)


class TestPermissionDeniedError:
    def test_stores_tool_name(self):
        err = PermissionDeniedError("bash")
        assert err.tool_name == "bash"

    def test_message_includes_tool(self):
        err = PermissionDeniedError("bash")
        assert "bash" in str(err)

    def test_optional_reason(self):
        err = PermissionDeniedError("git_push", reason="production branch")
        assert "production branch" in str(err)

    def test_is_runtime_error(self):
        assert isinstance(PermissionDeniedError("x"), RuntimeError)


# ---------------------------------------------------------------------------
# Event / result dataclass tests
# ---------------------------------------------------------------------------


class TestCopilotEvent:
    def test_minimal_construction(self):
        ev = CopilotEvent(type="token")
        assert ev.type == "token"
        assert ev.content == ""
        assert ev.tool == ""
        assert ev.error == ""
        assert ev.latency_ms == 0.0
        assert ev.arguments == {}

    def test_token_event(self):
        ev = CopilotEvent(type="token", content="Hello")
        assert ev.content == "Hello"

    def test_tool_start_event(self):
        ev = CopilotEvent(type="tool_start", tool="bash", arguments={"cmd": "ls"})
        assert ev.tool == "bash"
        assert ev.arguments == {"cmd": "ls"}

    def test_tool_end_event(self):
        ev = CopilotEvent(type="tool_end", tool="bash", result="file.py", latency_ms=42.5)
        assert ev.result == "file.py"
        assert ev.latency_ms == 42.5

    def test_usage_event(self):
        ev = CopilotEvent(
            type="usage", input_tokens=100, output_tokens=50, cost=0.002, content="gpt-4.1"
        )
        assert ev.input_tokens == 100
        assert ev.output_tokens == 50
        assert ev.cost == pytest.approx(0.002)

    def test_compaction_event(self):
        ev = CopilotEvent(type="compaction", compaction_phase="complete", tokens_removed=1024)
        assert ev.compaction_phase == "complete"
        assert ev.tokens_removed == 1024

    def test_subagent_event(self):
        ev = CopilotEvent(type="subagent", subagent_name="tester", subagent_phase="started")
        assert ev.subagent_name == "tester"
        assert ev.subagent_phase == "started"

    def test_reasoning_event(self):
        ev = CopilotEvent(type="reasoning", content="thinking...", reasoning_id="r1")
        assert ev.reasoning_id == "r1"

    def test_turn_event(self):
        ev = CopilotEvent(type="turn_start", turn_id="t42")
        assert ev.turn_id == "t42"

    def test_context_event(self):
        ev = CopilotEvent(type="context", cwd="/tmp", branch="main")
        assert ev.cwd == "/tmp"
        assert ev.branch == "main"

    def test_timestamp_auto_set(self):
        before = time.time()
        ev = CopilotEvent(type="done")
        after = time.time()
        assert before <= ev.timestamp <= after

    def test_arguments_default_is_independent(self):
        ev1 = CopilotEvent(type="tool_start")
        ev2 = CopilotEvent(type="tool_start")
        ev1.arguments["x"] = 1
        assert "x" not in ev2.arguments


class TestCopilotResult:
    def test_required_fields(self):
        r = CopilotResult(content="done", session_id="abc")
        assert r.content == "done"
        assert r.session_id == "abc"

    def test_default_optional_fields(self):
        r = CopilotResult(content="ok", session_id="s1")
        assert r.tools_called == 0
        assert r.duration_ms == 0.0

    def test_with_all_fields(self):
        r = CopilotResult(content="ok", session_id="s2", tools_called=5, duration_ms=1234.5)
        assert r.tools_called == 5
        assert r.duration_ms == pytest.approx(1234.5)


# ---------------------------------------------------------------------------
# copilot_tool decorator tests
# ---------------------------------------------------------------------------


class TestCopilotTool:
    def test_bare_decorator(self):
        @copilot_tool
        def fetch(url: str) -> str:
            "Fetch a URL."
            return url

        assert isinstance(fetch, _CopilotTool)
        assert fetch.name == "fetch"
        assert fetch.description == "Fetch a URL."
        assert fetch.skip_permission is False
        assert fetch.overrides_built_in_tool is False

    def test_explicit_description(self):
        @copilot_tool(description="Custom desc")
        def my_fn() -> str:
            return ""

        assert my_fn.description == "Custom desc"

    def test_explicit_name(self):
        @copilot_tool(name="renamed")
        def original() -> str:
            return ""

        assert original.name == "renamed"

    def test_skip_permission_flag(self):
        @copilot_tool(skip_permission=True)
        def safe_read() -> str:
            return ""

        assert safe_read.skip_permission is True

    def test_overrides_built_in(self):
        @copilot_tool(overrides_built_in_tool=True, name="edit_file")
        def custom_edit(path: str, content: str) -> str:
            return ""

        assert custom_edit.overrides_built_in_tool is True

    def test_name_defaults_to_func_name(self):
        @copilot_tool
        def my_special_tool() -> None:
            pass

        assert my_special_tool.name == "my_special_tool"

    def test_fn_stored(self):
        def raw():
            return 42

        wrapped = copilot_tool(raw)
        assert wrapped.fn is raw

    def test_to_sdk_tool_raises_without_sdk(self):
        @copilot_tool
        def fn() -> str:
            return ""

        with patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", False):
            with pytest.raises(ImportError):
                fn.to_sdk_tool()

    def test_to_sdk_tool_builds_sdk_object(self):
        """to_sdk_tool() calls Tool() with correct kwargs."""
        mock_tool_instance = MagicMock()
        mock_tool_cls = MagicMock(return_value=mock_tool_instance)

        @copilot_tool(description="Count words")
        def word_count(text: str) -> int:
            return len(text.split())

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            result = word_count.to_sdk_tool()

        assert result is mock_tool_instance
        call_kwargs = mock_tool_cls.call_args.kwargs
        assert call_kwargs["name"] == "word_count"
        assert call_kwargs["description"] == "Count words"
        assert "text" in call_kwargs["parameters"]["properties"]
        assert "text" in call_kwargs["parameters"]["required"]

    def test_to_sdk_tool_skip_permission(self):
        mock_tool_cls = MagicMock()

        @copilot_tool(skip_permission=True)
        def fn() -> str:
            return ""

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            fn.to_sdk_tool()

        assert mock_tool_cls.call_args.kwargs.get("skip_permission") is True

    def test_to_sdk_tool_overrides_built_in(self):
        mock_tool_cls = MagicMock()

        @copilot_tool(overrides_built_in_tool=True, name="edit_file")
        def fn(path: str, content: str) -> str:
            return ""

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            fn.to_sdk_tool()

        assert mock_tool_cls.call_args.kwargs.get("overrides_built_in_tool") is True

    @pytest.mark.asyncio
    async def test_sdk_tool_handler_sync_fn(self):
        """Handler wraps a sync function so it can be awaited."""
        mock_tool_cls = MagicMock()

        @copilot_tool
        def echo(msg: str) -> str:
            return f"echo: {msg}"

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            echo.to_sdk_tool()

        handler = mock_tool_cls.call_args.kwargs["handler"]
        result = await handler({"arguments": {"msg": "hi"}})
        assert result == "echo: hi"

    @pytest.mark.asyncio
    async def test_sdk_tool_handler_async_fn(self):
        """Handler awaits async functions correctly."""
        mock_tool_cls = MagicMock()

        @copilot_tool
        async def fetch(url: str) -> str:
            return f"fetched: {url}"

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            fetch.to_sdk_tool()

        handler = mock_tool_cls.call_args.kwargs["handler"]
        result = await handler({"arguments": {"url": "http://x.com"}})
        assert result == "fetched: http://x.com"


# ---------------------------------------------------------------------------
# CopilotAgent construction tests
# ---------------------------------------------------------------------------


class TestCopilotAgentInit:
    def test_default_params(self):
        agent = _make_agent()
        assert agent._model is None
        assert agent._cwd is None
        assert agent._permissions == PermissionMode.AUTO_APPROVE
        assert agent._streaming is True
        assert agent._infinite_context is True
        assert agent._callbacks is None
        assert agent._raw_tools == []
        assert agent._started is False

    def test_custom_params_stored(self):
        agent = _make_agent(
            model="gpt-4.1",
            cwd="/project",
            streaming=False,
            infinite_context=False,
            permissions=PermissionMode.READ_ONLY,
            reasoning_effort="high",
            initial_agent="refactor-bot",
        )
        assert agent._model == "gpt-4.1"
        assert agent._cwd == "/project"
        assert agent._streaming is False
        assert agent._infinite_context is False
        assert agent._permissions == PermissionMode.READ_ONLY
        assert agent._reasoning_effort == "high"
        assert agent._initial_agent == "refactor-bot"

    def test_raises_without_sdk(self):
        from ai_infra.llm.agents.copilot._agent import CopilotAgent

        with patch("ai_infra.llm.agents.copilot._agent.HAS_COPILOT", False):
            with pytest.raises(ImportError, match="github-copilot-sdk"):
                CopilotAgent()

    def test_tools_list_stored(self):
        @copilot_tool
        def fn() -> str:
            return ""

        agent = _make_agent(tools=[fn])
        assert len(agent._raw_tools) == 1

    def test_env_and_cli_path_stored(self):
        agent = _make_agent(env={"MYVAR": "1"}, cli_path="/usr/local/bin/copilot")
        assert agent._env == {"MYVAR": "1"}
        assert agent._cli_path == "/usr/local/bin/copilot"


# ---------------------------------------------------------------------------
# _build_permission_hook tests
# ---------------------------------------------------------------------------


class TestBuildPermissionHook:
    def test_auto_approve_returns_none(self):
        agent = _make_agent(permissions=PermissionMode.AUTO_APPROVE)
        assert agent._build_permission_hook() is None

    @pytest.mark.asyncio
    async def test_read_only_allows_read_tools(self):
        agent = _make_agent(permissions=PermissionMode.READ_ONLY)
        hooks = agent._build_permission_hook()
        assert hooks is not None
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "read_file"}, None)
        assert result["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_read_only_denies_write_tools(self):
        agent = _make_agent(permissions=PermissionMode.READ_ONLY)
        hooks = agent._build_permission_hook()
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "bash"}, None)
        assert result["permissionDecision"] == "deny"
        assert "READ_ONLY" in result["permissionDecisionReason"]

    @pytest.mark.asyncio
    async def test_interactive_allows_safe_tools(self):
        agent = _make_agent(permissions=PermissionMode.INTERACTIVE)
        hooks = agent._build_permission_hook()
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "read_file"}, None)
        assert result["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_interactive_asks_for_destructive(self):
        agent = _make_agent(permissions=PermissionMode.INTERACTIVE)
        hooks = agent._build_permission_hook()
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "bash"}, None)
        assert result["permissionDecision"] == "ask"

    @pytest.mark.asyncio
    async def test_deny_all_blocks_everything(self):
        agent = _make_agent(permissions=PermissionMode.DENY_ALL)
        hooks = agent._build_permission_hook()
        hook_fn = hooks["on_pre_tool_use"]
        for tool in ("read_file", "bash", "glob", "git_push"):
            result = await hook_fn({"toolName": tool}, None)
            assert result["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_custom_callable_allow(self):
        def allow_all(tool_name: str, args: dict) -> bool:
            return True

        agent = _make_agent(permissions=allow_all)
        hooks = agent._build_permission_hook()
        assert hooks is not None
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "bash", "toolArgs": {}}, None)
        assert result["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_custom_callable_deny(self):
        def deny_rm(tool_name: str, args: dict) -> bool:
            return tool_name != "bash"

        agent = _make_agent(permissions=deny_rm)
        hooks = agent._build_permission_hook()
        hook_fn = hooks["on_pre_tool_use"]
        result = await hook_fn({"toolName": "bash", "toolArgs": {}}, None)
        assert result["permissionDecision"] == "deny"
        assert "bash" in result["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# _build_session_config tests
# ---------------------------------------------------------------------------


class TestBuildSessionConfig:
    def test_required_keys_always_present(self):
        agent = _make_agent()
        config = agent._build_session_config(None)
        assert "streaming" in config
        assert "infinite_sessions" in config
        assert "on_permission_request" in config
        assert config["streaming"] is True
        assert config["infinite_sessions"] == {"enabled": True}

    def test_session_id_included_when_provided(self):
        agent = _make_agent()
        config = agent._build_session_config("sess-abc")
        assert config["session_id"] == "sess-abc"

    def test_session_id_omitted_when_none(self):
        agent = _make_agent()
        config = agent._build_session_config(None)
        assert "session_id" not in config

    def test_model_included(self):
        agent = _make_agent(model="gpt-4.1")
        config = agent._build_session_config(None)
        assert config["model"] == "gpt-4.1"

    def test_provider_included(self):
        provider = {"type": "openai", "api_key": "k"}
        agent = _make_agent(provider=provider)
        config = agent._build_session_config(None)
        assert config["provider"] == provider

    def test_system_message_wrapped(self):
        agent = _make_agent(system_message="You are helpful.")
        config = agent._build_session_config(None)
        assert config["system_message"] == {"content": "You are helpful."}

    def test_mcp_servers_included(self):
        mcp = {"fs": {"type": "local", "command": "npx"}}
        agent = _make_agent(mcp_servers=mcp)
        config = agent._build_session_config(None)
        assert config["mcp_servers"] == mcp

    def test_custom_agents_included(self):
        agents = [{"name": "tester", "prompt": "write tests", "tools": []}]
        agent = _make_agent(custom_agents=agents)
        config = agent._build_session_config(None)
        assert config["custom_agents"] == agents

    def test_initial_agent_maps_to_agent_key(self):
        agent = _make_agent(
            custom_agents=[{"name": "bot", "prompt": "", "tools": []}], initial_agent="bot"
        )
        config = agent._build_session_config(None)
        assert config["agent"] == "bot"

    def test_reasoning_effort_included(self):
        agent = _make_agent(reasoning_effort="high")
        config = agent._build_session_config(None)
        assert config["reasoning_effort"] == "high"

    def test_skill_dirs_included(self):
        agent = _make_agent(skill_dirs=["/skills"])
        config = agent._build_session_config(None)
        assert config["skill_directories"] == ["/skills"]

    def test_disabled_skills_included(self):
        agent = _make_agent(disabled_skills=["slow-research"])
        config = agent._build_session_config(None)
        assert config["disabled_skills"] == ["slow-research"]

    def test_hooks_merged_with_permission_hooks(self):
        user_hooks = {"on_session_start": lambda: None}
        agent = _make_agent(
            permissions=PermissionMode.READ_ONLY,
            hooks=user_hooks,
        )
        config = agent._build_session_config(None)
        merged = config["hooks"]
        # permission hook adds on_pre_tool_use; user hook adds on_session_start
        assert "on_pre_tool_use" in merged
        assert "on_session_start" in merged

    def test_user_hooks_take_precedence_over_permission_hook(self):
        def my_hook(a, b):
            return {}

        user_hooks = {"on_pre_tool_use": my_hook}
        agent = _make_agent(
            permissions=PermissionMode.READ_ONLY,
            hooks=user_hooks,
        )
        config = agent._build_session_config(None)
        # user hook should overwrite the built-in one
        assert config["hooks"]["on_pre_tool_use"] is my_hook

    def test_no_tools_key_when_empty(self):
        agent = _make_agent()
        config = agent._build_session_config(None)
        assert "tools" not in config


# ---------------------------------------------------------------------------
# _build_sdk_tools tests
# ---------------------------------------------------------------------------


class TestBuildSdkTools:
    def test_empty_list(self):
        agent = _make_agent()
        assert agent._build_sdk_tools() == []

    def test_copilot_tool_converted(self):
        mock_sdk = MagicMock()

        @copilot_tool
        def fn() -> str:
            return ""

        fn.to_sdk_tool = MagicMock(return_value=mock_sdk)
        agent = _make_agent(tools=[fn])
        result = agent._build_sdk_tools()
        assert result == [mock_sdk]
        fn.to_sdk_tool.assert_called_once()

    def test_raw_sdk_tool_passed_through(self):
        raw = MagicMock()
        agent = _make_agent(tools=[raw])
        result = agent._build_sdk_tools()
        assert result == [raw]


# ---------------------------------------------------------------------------
# Callback bridge tests
# ---------------------------------------------------------------------------


class TestCallbackBridge:
    def test_fire_tool_start_silent_without_callbacks(self):
        agent = _make_agent()
        # Should not raise
        agent._fire_tool_start("bash", {"cmd": "ls"})

    def test_fire_tool_end_silent_without_callbacks(self):
        agent = _make_agent()
        agent._fire_tool_end("bash", "output", 50.0)

    def test_fire_token_silent_without_callbacks(self):
        agent = _make_agent()
        agent._fire_token("hello")

    def test_fire_tool_start_calls_on_tool_start(self):
        from ai_infra.callbacks import Callbacks, ToolStartEvent

        events = []

        class Tracker(Callbacks):
            def on_tool_start(self, ev: ToolStartEvent) -> None:
                events.append(ev)

        agent = _make_agent(callbacks=Tracker())
        agent._fire_tool_start("grep", {"pattern": "TODO"})
        assert len(events) == 1
        assert events[0].tool_name == "grep"

    def test_fire_tool_end_calls_on_tool_end(self):
        from ai_infra.callbacks import Callbacks, ToolEndEvent

        events = []

        class Tracker(Callbacks):
            def on_tool_end(self, ev: ToolEndEvent) -> None:
                events.append(ev)

        agent = _make_agent(callbacks=Tracker())
        agent._fire_tool_end("grep", "result", 10.0)
        assert len(events) == 1
        assert events[0].tool_name == "grep"
        assert events[0].latency_ms == 10.0

    def test_fire_token_calls_on_llm_token(self):
        from ai_infra.callbacks import Callbacks, LLMTokenEvent

        tokens = []

        class Tracker(Callbacks):
            def on_llm_token(self, ev: LLMTokenEvent) -> None:
                tokens.append(ev.token)

        agent = _make_agent(model="gpt-4.1", callbacks=Tracker())
        agent._fire_token("Hello")
        assert tokens == ["Hello"]


# ---------------------------------------------------------------------------
# _translate() event mapping tests
# ---------------------------------------------------------------------------


class TestTranslate:
    """Test the internal _translate() method via _stream_internal's _translate closure."""

    def _get_translate(self, **kwargs):
        """Retrieve the _translate closure from _stream_internal."""
        agent = _make_agent(**kwargs)

        # We grab _translate by inspecting the async generator's frame locals
        # after one step. Use a simpler approach: patch the method and extract
        # the closure by starting the generator and letting it reach _translate.
        # Easier: just call _stream_internal with a mocked client and capture events.
        return agent

    def _make_raw(self, event_type: str, **data_fields) -> MagicMock:
        raw = MagicMock()
        raw.type.value = event_type
        data = MagicMock()
        for k, v in data_fields.items():
            setattr(data, k, v)
        # Make missing attrs return ""
        data.__class__ = type(
            "Data",
            (),
            dict(data_fields.items()),
        )
        raw.data = data
        return raw

    @pytest.mark.asyncio
    async def test_message_delta_emits_token(self):
        agent, events = await _run_mock_session(
            [_raw("assistant.message_delta", delta_content="Hi")]
        )
        assert any(e.type == "token" and e.content == "Hi" for e in events)

    @pytest.mark.asyncio
    async def test_tool_execution_start_emits_tool_start(self):
        agent, events = await _run_mock_session(
            [
                _raw(
                    "tool.execution_start",
                    tool_name="bash",
                    tool_call_id="c1",
                    arguments={"cmd": "ls"},
                )
            ]
        )
        assert any(e.type == "tool_start" and e.tool == "bash" for e in events)

    @pytest.mark.asyncio
    async def test_tool_execution_complete_emits_tool_end(self):
        agent, events = await _run_mock_session(
            [
                _raw("tool.execution_start", tool_name="bash", tool_call_id="c1", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c1",
                    result=_result("output"),
                ),
            ]
        )
        assert any(e.type == "tool_end" and e.tool == "bash" for e in events)

    @pytest.mark.asyncio
    async def test_tool_partial_result_emits_tool_output(self):
        agent, events = await _run_mock_session(
            [_raw("tool.execution_partial_result", tool_name="bash", partial_output="line 1\n")]
        )
        assert any(e.type == "tool_output" for e in events)

    @pytest.mark.asyncio
    async def test_tool_execution_progress_emits_tool_output(self):
        agent, events = await _run_mock_session(
            [
                _raw(
                    "tool.execution_progress",
                    tool_name="bash",
                    tool_call_id="c1",
                    progress_message="done",
                )
            ]
        )
        assert any(e.type == "tool_output" for e in events)

    @pytest.mark.asyncio
    async def test_assistant_intent_emits_intent(self):
        agent, events = await _run_mock_session(
            [_raw("assistant.intent", intent="Exploring codebase")]
        )
        assert any(e.type == "intent" and e.content == "Exploring codebase" for e in events)

    @pytest.mark.asyncio
    async def test_session_context_changed_emits_context(self):
        agent, events = await _run_mock_session(
            [_raw("session.context_changed", cwd="/proj", branch="main")]
        )
        assert any(e.type == "context" and e.cwd == "/proj" for e in events)

    @pytest.mark.asyncio
    async def test_session_error_emits_error(self):
        agent, events = await _run_mock_session(
            [_raw("session.error", error_type="timeout", message="request timed out")]
        )
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_assistant_usage_emits_usage(self):
        agent, events = await _run_mock_session(
            [
                _raw(
                    "assistant.usage",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.01,
                    model="gpt-4.1",
                )
            ]
        )
        assert any(e.type == "usage" and e.input_tokens == 100 for e in events)

    @pytest.mark.asyncio
    async def test_compaction_start_emits_compaction(self):
        agent, events = await _run_mock_session([_raw("session.compaction_start")])
        assert any(e.type == "compaction" and e.compaction_phase == "start" for e in events)

    @pytest.mark.asyncio
    async def test_compaction_complete_emits_compaction(self):
        agent, events = await _run_mock_session(
            [_raw("session.compaction_complete", tokens_removed=512, summary_content="compressed")]
        )
        ev = next(e for e in events if e.type == "compaction" and e.compaction_phase == "complete")
        assert ev.tokens_removed == 512

    @pytest.mark.asyncio
    async def test_subagent_events(self):
        for phase in ("selected", "started", "completed", "failed", "deselected"):
            agent, events = await _run_mock_session([_raw(f"subagent.{phase}", name="tester")])
            ev = next((e for e in events if e.type == "subagent"), None)
            assert ev is not None
            assert ev.subagent_phase == phase

    @pytest.mark.asyncio
    async def test_task_complete_emits_task_complete(self):
        agent, events = await _run_mock_session([_raw("session.task_complete", summary="All done")])
        assert any(e.type == "task_complete" and e.content == "All done" for e in events)

    @pytest.mark.asyncio
    async def test_turn_start_and_end(self):
        agent, events = await _run_mock_session(
            [
                _raw("assistant.turn_start", turn_id="t1"),
                _raw("assistant.turn_end", turn_id="t1"),
            ]
        )
        types = {e.type for e in events}
        assert "turn_start" in types
        assert "turn_end" in types

    @pytest.mark.asyncio
    async def test_reasoning_delta(self):
        agent, events = await _run_mock_session(
            [_raw("assistant.reasoning_delta", delta_content="thinking...", reasoning_id="r1")]
        )
        assert any(e.type == "reasoning_delta" and e.content == "thinking..." for e in events)

    @pytest.mark.asyncio
    async def test_reasoning_full(self):
        agent, events = await _run_mock_session(
            [_raw("assistant.reasoning", content="full thought", reasoning_id="r1")]
        )
        assert any(e.type == "reasoning" and e.content == "full thought" for e in events)

    @pytest.mark.asyncio
    async def test_unknown_event_type_ignored(self):
        agent, events = await _run_mock_session([_raw("unknown.mystery_event")])
        # Only the sentinel done event from idle is expected
        non_done = [e for e in events if e.type != "done"]
        assert all(e.type != "mystery" for e in non_done)

    @pytest.mark.asyncio
    async def test_done_event_from_session_idle(self):
        agent, events = await _run_mock_session([])
        assert any(e.type == "done" for e in events)


# ---------------------------------------------------------------------------
# CopilotAgent.run() tests
# ---------------------------------------------------------------------------


class TestCopilotAgentRun:
    @pytest.mark.asyncio
    async def test_run_aggregates_tokens(self):
        agent, result = await _run_agent_run(
            [
                _raw("assistant.message_delta", delta_content="Hello "),
                _raw("assistant.message_delta", delta_content="World"),
            ]
        )
        assert result.content == "Hello World"

    @pytest.mark.asyncio
    async def test_run_counts_tools(self):
        agent, result = await _run_agent_run(
            [
                _raw("tool.execution_start", tool_name="bash", tool_call_id="c1", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c1",
                    result=_result(""),
                ),
                _raw("tool.execution_start", tool_name="grep", tool_call_id="c2", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="grep",
                    tool_call_id="c2",
                    result=_result(""),
                ),
            ]
        )
        assert result.tools_called == 2

    @pytest.mark.asyncio
    async def test_run_captures_session_id(self):
        agent, result = await _run_agent_run([], session_id_on_done="my-session")
        assert result.session_id == "my-session"

    @pytest.mark.asyncio
    async def test_run_preserves_provided_session_id(self):
        agent, result = await _run_agent_run([], provided_session_id="existing-sess")
        assert result.session_id == "existing-sess"

    @pytest.mark.asyncio
    async def test_run_duration_positive(self):
        agent, result = await _run_agent_run([])
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# CopilotAgent utility methods tests
# ---------------------------------------------------------------------------


class TestCopilotAgentUtils:
    @pytest.mark.asyncio
    async def test_list_models_returns_strings(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=["gpt-4.1", "claude-sonnet-4-5"])
        agent._client = mock_client
        agent._started = True

        models = await agent.list_models()
        assert models == ["gpt-4.1", "claude-sonnet-4-5"]

    @pytest.mark.asyncio
    async def test_list_models_converts_objects(self):
        agent = _make_agent()
        model_obj = MagicMock()
        model_obj.id = "gpt-4.1"
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=[model_obj])
        agent._client = mock_client
        agent._started = True

        models = await agent.list_models()
        assert models == ["gpt-4.1"]

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=[])
        agent._client = mock_client
        agent._started = True

        assert await agent.list_models() == []

    @pytest.mark.asyncio
    async def test_list_sessions_returns_dicts(self):
        agent = _make_agent()
        sessions = [{"sessionId": "s1", "createdAt": "2026-01-01"}]
        mock_client = AsyncMock()
        mock_client.list_sessions = AsyncMock(return_value=sessions)
        agent._client = mock_client
        agent._started = True

        result = await agent.list_sessions()
        assert result == sessions

    @pytest.mark.asyncio
    async def test_list_sessions_converts_objects(self):
        agent = _make_agent()
        sess_obj = MagicMock(spec=["sessionId"])
        sess_obj.sessionId = "s2"
        mock_client = AsyncMock()
        mock_client.list_sessions = AsyncMock(return_value=[sess_obj])
        agent._client = mock_client
        agent._started = True

        result = await agent.list_sessions()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete_session_delegates(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        agent._client = mock_client
        agent._started = True

        await agent.delete_session("s-abc")
        mock_client.delete_session.assert_awaited_once_with("s-abc")

    @pytest.mark.asyncio
    async def test_stop_stops_client(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        agent._client = mock_client
        agent._started = True

        await agent.stop()
        mock_client.stop.assert_awaited_once()
        assert agent._started is False

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        agent._client = mock_client
        agent._started = False

        await agent.stop()
        mock_client.stop.assert_not_awaited()


# ---------------------------------------------------------------------------
# Top-level package re-export tests
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_all_symbols_accessible_from_package(self):
        from ai_infra.llm.agents.copilot import (
            HAS_COPILOT,
            SCRATCHPAD_TOOL_NAMES,
            CopilotAgent,
            CopilotEvent,
            CopilotResult,
            PermissionDeniedError,
            PermissionMode,
            copilot_tool,
            create_scratchpad_tools,
        )

        assert CopilotAgent is not None
        assert CopilotEvent is not None
        assert CopilotResult is not None
        assert isinstance(HAS_COPILOT, bool)
        assert PermissionMode is not None
        assert PermissionDeniedError is not None
        assert copilot_tool is not None
        assert SCRATCHPAD_TOOL_NAMES is not None
        assert create_scratchpad_tools is not None

    def test_symbols_accessible_from_agents_init(self):
        from ai_infra.llm.agents import (
            CopilotAgent,
        )

        assert CopilotAgent is not None


# ===========================================================================
# Shared test helpers
# ===========================================================================


def _result(content: str) -> MagicMock:
    """Build a fake tool result object with a .content attribute."""
    r = MagicMock()
    r.content = content
    return r


def _raw(event_type: str, **fields) -> MagicMock:
    """Build a fake raw SDK event."""
    raw = MagicMock()
    raw.type = MagicMock()
    raw.type.value = event_type

    class _Data:
        pass

    data = _Data()
    for k, v in fields.items():
        setattr(data, k, v)
    raw.data = data
    return raw


def _make_mock_session(raw_events: list[MagicMock], session_id_on_done: str = "") -> MagicMock:
    """Build a mock Copilot session that fires raw_events then session.idle."""
    session = MagicMock()
    session.session_id = session_id_on_done

    _callback_ref: list = []

    def _on(cb):
        _callback_ref.append(cb)

    session.on = _on
    session.send = AsyncMock()
    session.disconnect = AsyncMock()

    async def _fire_events_after_send(*args, **kwargs):
        cb = _callback_ref[0] if _callback_ref else None
        if cb:
            for ev in raw_events:
                cb(ev)
            # Always terminate with session.idle
            idle = _raw("session.idle")
            cb(idle)

    session.send.side_effect = _fire_events_after_send
    return session


async def _run_mock_session(
    raw_events: list[MagicMock],
    session_id_on_done: str = "",
    provided_session_id: str | None = None,
) -> tuple[Any, list[CopilotEvent]]:
    """Run _stream_internal on a mocked client and collect all events."""
    agent = _make_agent()

    mock_session = _make_mock_session(raw_events, session_id_on_done)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)
    agent._client = mock_client
    agent._started = True

    events: list[CopilotEvent] = []
    async for ev in agent._stream_internal("test prompt", session_id=provided_session_id):
        events.append(ev)

    return agent, events


async def _run_agent_run(
    raw_events: list[MagicMock],
    session_id_on_done: str = "",
    provided_session_id: str | None = None,
) -> tuple[Any, CopilotResult]:
    agent = _make_agent()

    mock_session = _make_mock_session(raw_events, session_id_on_done)
    mock_client = AsyncMock()
    mock_client.create_session = AsyncMock(return_value=mock_session)
    agent._client = mock_client
    agent._started = True

    result = await agent.run("test prompt", session_id=provided_session_id)
    return agent, result


# ===========================================================================
# Scratchpad tools tests
# ===========================================================================


class TestScratchpadTools:
    """Tests for create_scratchpad_tools() and SCRATCHPAD_TOOL_NAMES."""

    def test_create_returns_four_tools(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools = create_scratchpad_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {
            "scratchpad_think",
            "scratchpad_plan",
            "scratchpad_reflect",
            "scratchpad_read",
        }

    def test_tool_names_constant_matches(self):
        from ai_infra.llm.agents.copilot import SCRATCHPAD_TOOL_NAMES, create_scratchpad_tools

        tools = create_scratchpad_tools()
        assert {t.name for t in tools} == set(SCRATCHPAD_TOOL_NAMES)

    def test_all_tools_skip_permission(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        for t in create_scratchpad_tools():
            assert t.skip_permission is True, f"{t.name} should skip_permission"

    def test_think_records_and_read_returns(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools = create_scratchpad_tools()
        by_name = {t.name: t for t in tools}

        result = by_name["scratchpad_think"].fn(thought="Let me analyze the bug")
        assert result == "Recorded."

        read_result = by_name["scratchpad_read"].fn()
        assert "Let me analyze the bug" in read_result

    def test_plan_records(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools = create_scratchpad_tools()
        by_name = {t.name: t for t in tools}

        by_name["scratchpad_plan"].fn(plan="1. Read code\n2. Fix bug\n3. Test")
        read = by_name["scratchpad_read"].fn()
        assert "1. Read code" in read

    def test_reflect_records(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools = create_scratchpad_tools()
        by_name = {t.name: t for t in tools}

        by_name["scratchpad_reflect"].fn(reflection="The approach worked well")
        read = by_name["scratchpad_read"].fn()
        assert "approach worked well" in read

    def test_read_empty_scratchpad(self):
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools = create_scratchpad_tools()
        by_name = {t.name: t for t in tools}

        result = by_name["scratchpad_read"].fn()
        assert result == "(scratchpad is empty)"

    def test_independent_instances(self):
        """Each create_scratchpad_tools() call gets its own buffer."""
        from ai_infra.llm.agents.copilot import create_scratchpad_tools

        tools_a = create_scratchpad_tools()
        tools_b = create_scratchpad_tools()

        by_name_a = {t.name: t for t in tools_a}
        by_name_b = {t.name: t for t in tools_b}

        by_name_a["scratchpad_think"].fn(thought="session A only")
        read_b = by_name_b["scratchpad_read"].fn()
        assert read_b == "(scratchpad is empty)"
