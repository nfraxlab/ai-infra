"""Integration tests for CopilotAgent.

These tests verify the full request/response lifecycle using a fully mocked
Copilot SDK client — no real Copilot CLI process or network connection is
required.

Each test exercises a realistic scenario: multi-turn sessions, permission
enforcement, custom tool invocation, streaming UI patterns, BYOK config,
skill/MCP wiring, and lifecycle management.

Run with: pytest tests/integration/test_copilot_agent.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_infra.llm.agents.copilot import (
    CopilotAgent,
    CopilotEvent,
    CopilotResult,
    PermissionMode,
    copilot_tool,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _result_obj(content: str) -> MagicMock:
    r = MagicMock()
    r.content = content
    return r


def _raw(event_type: str, **fields) -> MagicMock:
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


class FakeCopilotSession:
    """A minimal in-process stand-in for the SDK session object."""

    def __init__(self, events_to_fire: list[MagicMock], session_id: str = "sess-001") -> None:
        self._events = events_to_fire
        self.session_id = session_id
        self._callback = None
        self.sent_payloads: list[dict] = []
        self.disconnected = False

    def on(self, cb) -> None:
        self._callback = cb

    async def send(self, payload: dict) -> None:
        self.sent_payloads.append(payload)
        if self._callback:
            for ev in self._events:
                self._callback(ev)
            # Terminate with session.idle
            self._callback(_raw("session.idle"))

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeCopilotClient:
    """A minimal in-process stand-in for the SDK CopilotClient."""

    def __init__(
        self,
        sessions: list[FakeCopilotSession] | None = None,
        models: list[str] | None = None,
    ) -> None:
        self._sessions = sessions or []
        self._session_index = 0
        self._models = models or ["gpt-4.1", "claude-sonnet-4-5"]
        self.started = False
        self.stopped = False
        self.list_sessions_data: list[dict] = []
        self.deleted_sessions: list[str] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def create_session(self, config: dict) -> FakeCopilotSession:
        session = self._sessions[self._session_index]
        self._session_index += 1
        return session

    async def list_models(self) -> list[str]:
        return self._models

    async def list_sessions(self) -> list[dict]:
        return self.list_sessions_data

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)


def _make_agent_with_client(
    fake_client: FakeCopilotClient,
    **kwargs,
) -> CopilotAgent:
    """Construct a CopilotAgent that uses fake_client, bypassing the CLI guard."""
    with (
        patch("ai_infra.llm.agents.copilot._agent.HAS_COPILOT", True),
        patch("ai_infra.llm.agents.copilot._agent.CopilotClient", return_value=fake_client),
        patch("ai_infra.llm.agents.copilot._agent.SubprocessConfig"),
    ):
        agent = CopilotAgent(**kwargs)
    # Inject the fake client so _ensure_started uses it (it will call CopilotClient())
    # We override by pre-starting with the fake directly.
    agent._client = fake_client
    agent._started = True
    return agent


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestCopilotAgentLifecycle:
    @pytest.mark.asyncio
    async def test_agent_starts_on_first_run(self):
        fake_client = FakeCopilotClient(sessions=[FakeCopilotSession([])])

        with (
            patch("ai_infra.llm.agents.copilot._agent.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._agent.CopilotClient", return_value=fake_client),
            patch("ai_infra.llm.agents.copilot._agent.SubprocessConfig"),
        ):
            agent = CopilotAgent()
            assert agent._started is False
            await agent.run("hello")
            assert fake_client.started is True

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self):
        fake_client = FakeCopilotClient(sessions=[])

        with (
            patch("ai_infra.llm.agents.copilot._agent.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._agent.CopilotClient", return_value=fake_client),
            patch("ai_infra.llm.agents.copilot._agent.SubprocessConfig"),
        ):
            async with CopilotAgent() as agent:
                assert agent._started is True
                assert fake_client.started is True

        assert fake_client.stopped is True

    @pytest.mark.asyncio
    async def test_stop_marks_not_started(self):
        fake_client = FakeCopilotClient()
        agent = _make_agent_with_client(fake_client)

        await agent.stop()
        assert agent._started is False
        assert fake_client.stopped is True

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        fake_client = FakeCopilotClient()
        agent = _make_agent_with_client(fake_client)

        await agent.stop()
        await agent.stop()  # second call should not raise


# ---------------------------------------------------------------------------
# run() integration tests
# ---------------------------------------------------------------------------


class TestCopilotAgentRunIntegration:
    @pytest.mark.asyncio
    async def test_simple_task_returns_content(self):
        session = FakeCopilotSession(
            [
                _raw("assistant.message_delta", delta_content="All done."),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        result = await agent.run("Add type hints")

        assert isinstance(result, CopilotResult)
        assert result.content == "All done."

    @pytest.mark.asyncio
    async def test_result_counts_tool_calls(self):
        session = FakeCopilotSession(
            [
                _raw("tool.execution_start", tool_name="bash", tool_call_id="c1", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c1",
                    result=_result_obj("ok"),
                ),
                _raw(
                    "tool.execution_start", tool_name="write_file", tool_call_id="c2", arguments={}
                ),
                _raw(
                    "tool.execution_complete",
                    tool_name="write_file",
                    tool_call_id="c2",
                    result=_result_obj("ok"),
                ),
                _raw("tool.execution_start", tool_name="bash", tool_call_id="c3", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c3",
                    result=_result_obj("ok"),
                ),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        result = await agent.run("Run and fix tests")

        assert result.tools_called == 3

    @pytest.mark.asyncio
    async def test_session_id_captured_from_done_event(self):
        session = FakeCopilotSession([], session_id="auto-sess-xyz")
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        result = await agent.run("Summarise repo")

        # The done event's content should be captured as session_id
        assert result.session_id == "auto-sess-xyz"

    @pytest.mark.asyncio
    async def test_provided_session_id_preserved(self):
        session = FakeCopilotSession([], session_id="other-id")
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        result = await agent.run("Continue", session_id="my-existing-session")

        assert result.session_id == "my-existing-session"

    @pytest.mark.asyncio
    async def test_prompt_sent_to_session(self):
        session = FakeCopilotSession([])
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        await agent.run("Fix lint errors")

        assert len(session.sent_payloads) == 1
        assert session.sent_payloads[0]["prompt"] == "Fix lint errors"

    @pytest.mark.asyncio
    async def test_attachments_forwarded(self):
        session = FakeCopilotSession([])
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        attachments = [{"type": "file", "path": "main.py"}]
        await agent.run("Review this file", attachments=attachments)

        assert session.sent_payloads[0].get("attachments") == attachments

    @pytest.mark.asyncio
    async def test_session_disconnected_after_run(self):
        session = FakeCopilotSession([])
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        await agent.run("nothing")

        assert session.disconnected is True

    @pytest.mark.asyncio
    async def test_duration_ms_positive(self):
        session = FakeCopilotSession([])
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        result = await agent.run("nothing")

        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_multi_turn_session_reuse(self):
        """Two run() calls with the same session_id thread correctly."""
        session1 = FakeCopilotSession([_raw("assistant.message_delta", delta_content="First")])
        session2 = FakeCopilotSession([_raw("assistant.message_delta", delta_content="Second")])
        fake_client = FakeCopilotClient(sessions=[session1, session2])
        agent = _make_agent_with_client(fake_client)

        r1 = await agent.run("Tell me about auth", session_id="conv-1")
        r2 = await agent.run("Now write tests", session_id="conv-1")

        assert r1.content == "First"
        assert r2.content == "Second"


# ---------------------------------------------------------------------------
# stream() integration tests
# ---------------------------------------------------------------------------


class TestCopilotAgentStreamIntegration:
    @pytest.mark.asyncio
    async def test_stream_yields_all_event_types(self):
        """A full simulated turn yields a representative mix of event types."""
        session = FakeCopilotSession(
            [
                _raw("assistant.turn_start", turn_id="t1"),
                _raw("assistant.intent", intent="Exploring files"),
                _raw(
                    "tool.execution_start",
                    tool_name="glob",
                    tool_call_id="c0",
                    arguments={"pattern": "*.py"},
                ),
                _raw(
                    "tool.execution_complete",
                    tool_name="glob",
                    tool_call_id="c0",
                    result=_result_obj("a.py\nb.py"),
                ),
                _raw("assistant.message_delta", delta_content="Found two files."),
                _raw(
                    "assistant.usage", input_tokens=120, output_tokens=30, cost=0.0, model="gpt-4.1"
                ),
                _raw("assistant.turn_end", turn_id="t1"),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        events: list[CopilotEvent] = []
        async for ev in agent.stream("List Python files"):
            events.append(ev)

        types = {e.type for e in events}
        assert "turn_start" in types
        assert "intent" in types
        assert "tool_start" in types
        assert "tool_end" in types
        assert "token" in types
        assert "usage" in types
        assert "turn_end" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_stream_token_content_order(self):
        """Tokens arrive in the order the SDK delivers them."""
        session = FakeCopilotSession(
            [
                _raw("assistant.message_delta", delta_content="Hello"),
                _raw("assistant.message_delta", delta_content=" "),
                _raw("assistant.message_delta", delta_content="World"),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        tokens = [ev.content for ev in [e async for e in agent.stream("hi")] if ev.type == "token"]
        assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_stream_tool_latency_populated(self):
        """tool_end.latency_ms is non-negative."""
        session = FakeCopilotSession(
            [
                _raw("tool.execution_start", tool_name="bash", tool_call_id="c1", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c1",
                    result=_result_obj("ok"),
                ),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        events = [ev async for ev in agent.stream("run tests")]
        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self):
        session = FakeCopilotSession([_raw("assistant.message_delta", delta_content="ok")])
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        events = [ev async for ev in agent.stream("go")]
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stream_error_event_passthrough(self):
        session = FakeCopilotSession(
            [
                _raw("session.error", error_type="rate_limit", message="Too many requests"),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        events = [ev async for ev in agent.stream("hi")]
        err = next((e for e in events if e.type == "error"), None)
        assert err is not None
        assert "rate_limit" in err.error or "Too many requests" in err.error

    @pytest.mark.asyncio
    async def test_stream_compaction_events(self):
        session = FakeCopilotSession(
            [
                _raw("session.compaction_start"),
                _raw(
                    "session.compaction_complete", tokens_removed=2048, summary_content="compressed"
                ),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]))
        events = [ev async for ev in agent.stream("long task")]
        starts = [e for e in events if e.type == "compaction" and e.compaction_phase == "start"]
        completes = [
            e for e in events if e.type == "compaction" and e.compaction_phase == "complete"
        ]
        assert len(starts) == 1
        assert len(completes) == 1
        assert completes[0].tokens_removed == 2048


# ---------------------------------------------------------------------------
# Permission mode integration tests
# ---------------------------------------------------------------------------


class TestPermissionIntegration:
    @pytest.mark.asyncio
    async def test_read_only_session_config_has_hook(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.READ_ONLY,
        )
        config = agent._build_session_config(None)
        assert "hooks" in config
        assert "on_pre_tool_use" in config["hooks"]

    @pytest.mark.asyncio
    async def test_auto_approve_session_config_no_pre_tool_hook(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.AUTO_APPROVE,
        )
        config = agent._build_session_config(None)
        # No hooks key at all, or no on_pre_tool_use
        hooks = config.get("hooks", {})
        assert "on_pre_tool_use" not in hooks

    @pytest.mark.asyncio
    async def test_read_only_permission_hook_denies_bash(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.READ_ONLY,
        )
        config = agent._build_session_config(None)
        hook = config["hooks"]["on_pre_tool_use"]
        result = await hook({"toolName": "bash"}, None)
        assert result["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_deny_all_permission_hook_denies_read(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.DENY_ALL,
        )
        config = agent._build_session_config(None)
        hook = config["hooks"]["on_pre_tool_use"]
        result = await hook({"toolName": "read_file"}, None)
        assert result["permissionDecision"] == "deny"


# ---------------------------------------------------------------------------
# Custom tool integration tests
# ---------------------------------------------------------------------------


class TestCustomToolIntegration:
    @pytest.mark.asyncio
    async def test_custom_tool_registered_in_session_config(self):
        mock_tool_cls = MagicMock(return_value=MagicMock())

        @copilot_tool(description="Look up an issue")
        async def get_issue(id: str) -> str:
            return f"issue {id}"

        agent = _make_agent_with_client(
            FakeCopilotClient(),
            tools=[get_issue],
        )

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            config = agent._build_session_config(None)

        assert "tools" in config
        assert len(config["tools"]) == 1

    @pytest.mark.asyncio
    async def test_custom_tool_handler_callable_at_runtime(self):
        """The tool handler registered in the SDK is actually callable."""
        captured_handler = []
        mock_tool_cls = MagicMock(
            side_effect=lambda **kw: captured_handler.append(kw["handler"]) or MagicMock()
        )

        @copilot_tool
        async def echo(message: str) -> str:
            return f"echo: {message}"

        agent = _make_agent_with_client(FakeCopilotClient(), tools=[echo])

        with (
            patch("ai_infra.llm.agents.copilot._tools.HAS_COPILOT", True),
            patch("ai_infra.llm.agents.copilot._tools.Tool", mock_tool_cls),
        ):
            agent._build_session_config(None)

        assert len(captured_handler) == 1
        result = await captured_handler[0]({"arguments": {"message": "hello"}})
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_raw_sdk_tool_passed_directly(self):
        """Raw SDK Tool objects are forwarded without conversion."""
        raw_sdk_tool = MagicMock()
        agent = _make_agent_with_client(FakeCopilotClient(), tools=[raw_sdk_tool])
        config = agent._build_session_config(None)
        assert config["tools"][0] is raw_sdk_tool


# ---------------------------------------------------------------------------
# BYOK / provider config integration tests
# ---------------------------------------------------------------------------


class TestBYOKIntegration:
    def test_provider_stored_correctly(self):
        provider_cfg = {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            model="gpt-4.1",
            provider=provider_cfg,
        )
        config = agent._build_session_config(None)
        assert config["provider"] == provider_cfg
        assert config["model"] == "gpt-4.1"

    def test_anthropic_provider(self):
        provider_cfg = {"type": "anthropic", "api_key": "test-key"}
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            model="claude-sonnet-4-5",
            provider=provider_cfg,
        )
        config = agent._build_session_config(None)
        assert config["provider"]["type"] == "anthropic"


# ---------------------------------------------------------------------------
# Skills / MCP / sub-agent config tests
# ---------------------------------------------------------------------------


class TestAdvancedConfig:
    def test_skill_dirs_wired_into_session_config(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            skill_dirs=["/agent/skills"],
        )
        config = agent._build_session_config(None)
        assert config["skill_directories"] == ["/agent/skills"]

    def test_disabled_skills_wired(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            skill_dirs=["/skills"],
            disabled_skills=["slow-research"],
        )
        config = agent._build_session_config(None)
        assert config["disabled_skills"] == ["slow-research"]

    def test_mcp_servers_wired(self):
        mcp = {"fs": {"type": "local", "command": "npx", "args": [], "tools": ["*"]}}
        agent = _make_agent_with_client(FakeCopilotClient(), mcp_servers=mcp)
        config = agent._build_session_config(None)
        assert config["mcp_servers"] == mcp

    def test_custom_agents_wired(self):
        agents_def = [{"name": "tester", "prompt": "write tests", "tools": ["write_file"]}]
        agent = _make_agent_with_client(FakeCopilotClient(), custom_agents=agents_def)
        config = agent._build_session_config(None)
        assert config["custom_agents"] == agents_def

    def test_initial_agent_maps_to_agent_key(self):
        agents_def = [{"name": "tester", "prompt": "write tests", "tools": []}]
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            custom_agents=agents_def,
            initial_agent="tester",
        )
        config = agent._build_session_config(None)
        assert config["agent"] == "tester"

    def test_system_message_wrapped(self):
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            system_message="Never modify production data.",
        )
        config = agent._build_session_config(None)
        assert config["system_message"]["content"] == "Never modify production data."

    def test_infinite_context_disabled(self):
        agent = _make_agent_with_client(FakeCopilotClient(), infinite_context=False)
        config = agent._build_session_config(None)
        assert config["infinite_sessions"] == {"enabled": False}


# ---------------------------------------------------------------------------
# Session management integration tests
# ---------------------------------------------------------------------------


class TestSessionManagementIntegration:
    @pytest.mark.asyncio
    async def test_list_models_delegates(self):
        fake_client = FakeCopilotClient(models=["gpt-4.1", "o4-mini"])
        agent = _make_agent_with_client(fake_client)
        models = await agent.list_models()
        assert "gpt-4.1" in models
        assert "o4-mini" in models

    @pytest.mark.asyncio
    async def test_list_sessions_delegates(self):
        fake_client = FakeCopilotClient()
        fake_client.list_sessions_data = [
            {"sessionId": "s1", "createdAt": "2026-01-01"},
            {"sessionId": "s2", "createdAt": "2026-02-01"},
        ]
        agent = _make_agent_with_client(fake_client)
        sessions = await agent.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["sessionId"] == "s1"

    @pytest.mark.asyncio
    async def test_delete_session_removes_from_client(self):
        fake_client = FakeCopilotClient()
        agent = _make_agent_with_client(fake_client)
        await agent.delete_session("s-dead")
        assert "s-dead" in fake_client.deleted_sessions


# ---------------------------------------------------------------------------
# Callback bridge integration tests
# ---------------------------------------------------------------------------


class TestCallbackBridgeIntegration:
    @pytest.mark.asyncio
    async def test_tool_start_callback_fired_during_run(self):
        from ai_infra.callbacks import Callbacks, ToolStartEvent

        tool_events = []

        class Tracker(Callbacks):
            def on_tool_start(self, ev: ToolStartEvent) -> None:
                tool_events.append(ev)

        session = FakeCopilotSession(
            [
                _raw(
                    "tool.execution_start",
                    tool_name="bash",
                    tool_call_id="c1",
                    arguments={"cmd": "ls"},
                ),
                _raw(
                    "tool.execution_complete",
                    tool_name="bash",
                    tool_call_id="c1",
                    result=_result_obj("ok"),
                ),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]), callbacks=Tracker())
        await agent.run("list files")

        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "bash"

    @pytest.mark.asyncio
    async def test_token_callback_fired_during_run(self):
        from ai_infra.callbacks import Callbacks, LLMTokenEvent

        tokens = []

        class Tracker(Callbacks):
            def on_llm_token(self, ev: LLMTokenEvent) -> None:
                tokens.append(ev.token)

        session = FakeCopilotSession(
            [
                _raw("assistant.message_delta", delta_content="Hello"),
                _raw("assistant.message_delta", delta_content=" World"),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]), callbacks=Tracker())
        await agent.run("say hi")

        assert tokens == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_tool_end_callback_fired_with_latency(self):
        from ai_infra.callbacks import Callbacks, ToolEndEvent

        end_events = []

        class Tracker(Callbacks):
            def on_tool_end(self, ev: ToolEndEvent) -> None:
                end_events.append(ev)

        session = FakeCopilotSession(
            [
                _raw("tool.execution_start", tool_name="grep", tool_call_id="c1", arguments={}),
                _raw(
                    "tool.execution_complete",
                    tool_name="grep",
                    tool_call_id="c1",
                    result=_result_obj("found"),
                ),
            ]
        )
        agent = _make_agent_with_client(FakeCopilotClient(sessions=[session]), callbacks=Tracker())
        await agent.run("search")

        assert len(end_events) == 1
        assert end_events[0].tool_name == "grep"
        assert end_events[0].latency_ms >= 0


# ---------------------------------------------------------------------------
# User input hook integration test
# ---------------------------------------------------------------------------


class TestUserInputHookIntegration:
    def test_on_user_input_wired_into_session_config(self):
        async def my_input_handler(request, invocation):
            return {"answer": "yes", "wasFreeform": True}

        agent = _make_agent_with_client(
            FakeCopilotClient(),
            on_user_input=my_input_handler,
        )
        config = agent._build_session_config(None)
        assert config["on_user_input_request"] is my_input_handler


# ---------------------------------------------------------------------------
# Hooks merging integration test
# ---------------------------------------------------------------------------


class TestHooksMergingIntegration:
    def test_extra_hooks_merged_with_permission_hooks(self):
        session_start_fn = MagicMock()
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.READ_ONLY,
            hooks={"on_session_start": session_start_fn, "on_session_end": MagicMock()},
        )
        config = agent._build_session_config(None)
        hooks = config["hooks"]
        assert "on_pre_tool_use" in hooks  # from READ_ONLY
        assert hooks["on_session_start"] is session_start_fn

    def test_user_on_pre_tool_use_overrides_permission_hook(self):
        my_hook = AsyncMock(return_value={"permissionDecision": "allow"})
        agent = _make_agent_with_client(
            FakeCopilotClient(),
            permissions=PermissionMode.DENY_ALL,
            hooks={"on_pre_tool_use": my_hook},
        )
        config = agent._build_session_config(None)
        assert config["hooks"]["on_pre_tool_use"] is my_hook
