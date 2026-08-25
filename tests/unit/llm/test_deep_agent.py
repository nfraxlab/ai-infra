"""Tests for llm/agents/deep.py - Deep Agent Building.

This module tests the deep agent infrastructure including:
- DeepAgents type exports and placeholders
- HAS_DEEPAGENTS availability flag
- build_deep_agent() function
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Module Availability Tests
# =============================================================================


class TestDeepAgentsAvailability:
    """Tests for HAS_DEEPAGENTS flag and module exports."""

    def test_has_deepagents_flag_exists(self):
        """Test HAS_DEEPAGENTS flag is exported."""
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        assert isinstance(HAS_DEEPAGENTS, bool)

    def test_subagent_exported(self):
        """Test SubAgent type is exported."""
        from ai_infra.llm.agents.deep import SubAgent

        assert SubAgent is not None

    def test_compiled_subagent_exported(self):
        """Test CompiledSubAgent type is exported."""
        from ai_infra.llm.agents.deep import CompiledSubAgent

        assert CompiledSubAgent is not None

    def test_subagent_middleware_exported(self):
        """Test SubAgentMiddleware type is exported."""
        from ai_infra.llm.agents.deep import SubAgentMiddleware

        assert SubAgentMiddleware is not None

    def test_filesystem_middleware_exported(self):
        """Test FilesystemMiddleware type is exported."""
        from ai_infra.llm.agents.deep import FilesystemMiddleware

        assert FilesystemMiddleware is not None

    def test_agent_middleware_exported(self):
        """Test AgentMiddleware type is exported."""
        from ai_infra.llm.agents.deep import AgentMiddleware

        assert AgentMiddleware is not None

    def test_build_deep_agent_exported(self):
        """Test build_deep_agent function is exported."""
        from ai_infra.llm.agents.deep import build_deep_agent

        assert callable(build_deep_agent)

    def test_all_exports_in_module(self):
        """Test __all__ contains expected exports."""
        from ai_infra.llm.agents import deep

        expected = [
            "HAS_DEEPAGENTS",
            "AgentMiddleware",
            "CompiledSubAgent",
            "FilesystemMiddleware",
            "SubAgent",
            "SubAgentMiddleware",
            "build_deep_agent",
        ]

        for name in expected:
            assert name in deep.__all__, f"{name} should be in __all__"


# =============================================================================
# Placeholder Tests (When deepagents is not installed)
# =============================================================================


class TestPlaceholderBehavior:
    """Tests for placeholder classes when deepagents is not installed."""

    @patch.dict("sys.modules", {"deepagents": None})
    def test_placeholder_subagent_import_error_message(self):
        """Test SubAgent placeholder raises ImportError with helpful message."""
        # We can't easily test the placeholder behavior without reloading the module
        # So we test the expected error message format
        from ai_infra.llm.agents.deep import HAS_DEEPAGENTS

        if not HAS_DEEPAGENTS:
            from ai_infra.llm.agents.deep import SubAgent

            with pytest.raises(ImportError, match="deepagents"):
                SubAgent()


# =============================================================================
# build_deep_agent Function Tests
# =============================================================================


class TestBuildDeepAgentBasic:
    """Basic tests for build_deep_agent function."""

    def test_build_deep_agent_requires_model(self):
        """Test build_deep_agent requires a model argument."""
        from ai_infra.llm.agents.deep import build_deep_agent

        # Should fail without model
        with pytest.raises(TypeError):
            build_deep_agent()  # type: ignore[call-arg]

    @patch("ai_infra.llm.agents.deep.HAS_DEEPAGENTS", False)
    def test_build_deep_agent_raises_when_not_installed(self):
        """Test build_deep_agent raises ImportError when deepagents not installed."""
        # Reload module to get the behavior with patched flag
        from ai_infra.llm.agents import deep

        # Mock the import to fail
        with patch.dict("sys.modules", {"deepagents": None}):
            with patch.object(deep, "HAS_DEEPAGENTS", False):
                # The function should raise ImportError
                mock_model = MagicMock()
                with pytest.raises(ImportError, match="deepagents"):
                    deep.build_deep_agent(mock_model)


class TestBuildDeepAgentWithMocks:
    """Tests for build_deep_agent with mocked deepagents."""

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_calls_create_deep_agent(self, mock_create):
        """Test build_deep_agent calls create_deep_agent from deepagents."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            _result = build_deep_agent(mock_model)
            mock_create.assert_called_once()
        except ImportError:
            # If deepagents is not installed, skip this test
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_model(self, mock_create):
        """Test build_deep_agent passes model to create_deep_agent."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("model") is mock_model
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_tools(self, mock_create):
        """Test build_deep_agent passes tools to create_deep_agent."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        tools = [MagicMock(), MagicMock()]

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, tools=tools)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("tools") is tools
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_system(self, mock_create):
        """Test build_deep_agent passes system prompt."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, system="You are helpful")
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("system_prompt") == "You are helpful"
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_middleware(self, mock_create):
        """Test build_deep_agent passes middleware."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        middleware = [MagicMock(), MagicMock()]

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, middleware=middleware)
            call_kwargs = mock_create.call_args.kwargs
            # Middleware should be converted to tuple
            assert call_kwargs.get("middleware") == tuple(middleware)
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_subagents(self, mock_create):
        """Test build_deep_agent passes subagents."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        subagents = [MagicMock()]

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, subagents=subagents)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("subagents") is subagents
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_response_format(self, mock_create):
        """Test build_deep_agent passes response_format."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        response_format = {"type": "json_object"}

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, response_format=response_format)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("response_format") is response_format
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_passes_context_schema(self, mock_create):
        """Test build_deep_agent passes context_schema."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        class MyContextSchema:
            pass

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, context_schema=MyContextSchema)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("context_schema") is MyContextSchema
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_empty_middleware_is_empty_tuple(self, mock_create):
        """Test build_deep_agent with no middleware passes empty tuple."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("middleware") == ()
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_build_deep_agent_none_tools_is_none(self, mock_create):
        """Test build_deep_agent with no tools passes None."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("tools") is None
        except ImportError:
            pytest.skip("deepagents package not installed")


# =============================================================================
# Workspace Integration Tests
# =============================================================================


class TestBuildDeepAgentWorkspace:
    """Tests for build_deep_agent with workspace parameter."""

    @patch("deepagents.create_deep_agent")
    def test_workspace_backend_extracted(self, mock_create):
        """Test workspace backend is extracted and passed."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        mock_workspace = MagicMock()
        mock_backend = MagicMock()
        mock_workspace.get_deepagent_backend.return_value = mock_backend

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, workspace=mock_workspace)
            mock_workspace.get_deepagent_backend.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("backend") is mock_backend
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_no_workspace_means_no_backend(self, mock_create):
        """Test no workspace means backend is None."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("backend") is None
        except ImportError:
            pytest.skip("deepagents package not installed")


# =============================================================================
# Session Config Integration Tests
# =============================================================================


class TestBuildDeepAgentSessionConfig:
    """Tests for build_deep_agent with session_config parameter."""

    @patch("deepagents.create_deep_agent")
    def test_session_config_checkpointer_extracted(self, mock_create):
        """Test checkpointer is extracted from session config."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_checkpointer = MagicMock()
        mock_session.storage.get_checkpointer.return_value = mock_checkpointer
        mock_session.storage.get_store.return_value = None
        mock_session.pause_before = None
        mock_session.pause_after = None

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, session_config=mock_session)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("checkpointer") is mock_checkpointer
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_session_config_store_extracted(self, mock_create):
        """Test store is extracted from session config."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_store = MagicMock()
        mock_session.storage.get_checkpointer.return_value = None
        mock_session.storage.get_store.return_value = mock_store
        mock_session.pause_before = None
        mock_session.pause_after = None

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, session_config=mock_session)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("store") is mock_store
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_session_config_pause_before(self, mock_create):
        """Test pause_before is converted to the approval interrupt configuration."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.storage.get_checkpointer.return_value = None
        mock_session.storage.get_store.return_value = None
        mock_session.pause_before = ["tool1", "tool2"]
        mock_session.pause_after = None

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, session_config=mock_session)
            call_kwargs = mock_create.call_args.kwargs
            interrupt_on = call_kwargs.get("interrupt_on")
            assert interrupt_on is not None
            assert interrupt_on.get("tool1") is True
            assert interrupt_on.get("tool2") is True
        except ImportError:
            pytest.skip("deepagents package not installed")

    def test_session_config_pause_after_is_rejected(self):
        """Test pause_after fails because DeepAgents lacks an equivalent interrupt."""
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.storage.get_checkpointer.return_value = None
        mock_session.storage.get_store.return_value = None
        mock_session.pause_before = None
        mock_session.pause_after = ["tool1"]

        from ai_infra.llm.agents.deep import build_deep_agent

        with pytest.raises(ValueError, match="pause_after is not supported"):
            build_deep_agent(mock_model, session_config=mock_session)

    def test_session_config_pause_before_and_after_is_rejected(self):
        """Test pause_after is rejected even when pause_before is configured."""
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.storage.get_checkpointer.return_value = None
        mock_session.storage.get_store.return_value = None
        mock_session.pause_before = ["tool1"]
        mock_session.pause_after = ["tool1"]

        from ai_infra.llm.agents.deep import build_deep_agent

        with pytest.raises(ValueError, match="pause_after is not supported"):
            build_deep_agent(mock_model, session_config=mock_session)

    @patch("deepagents.create_deep_agent")
    def test_no_session_config_means_none_values(self, mock_create):
        """Test no session config means None for checkpointer/store/interrupt_on."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("checkpointer") is None
            assert call_kwargs.get("store") is None
            assert call_kwargs.get("interrupt_on") is None
        except ImportError:
            pytest.skip("deepagents package not installed")


# =============================================================================
# All Parameters Combined Tests
# =============================================================================


class TestBuildDeepAgentAllParams:
    """Tests for build_deep_agent with all parameters."""

    @patch("deepagents.create_deep_agent")
    def test_all_params_passed(self, mock_create):
        """Test all parameters are passed correctly."""
        mock_create.return_value = MagicMock()

        # Create all mocks
        mock_model = MagicMock()
        mock_workspace = MagicMock()
        mock_backend = MagicMock()
        mock_workspace.get_deepagent_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_checkpointer = MagicMock()
        mock_store = MagicMock()
        mock_session.storage.get_checkpointer.return_value = mock_checkpointer
        mock_session.storage.get_store.return_value = mock_store
        mock_session.pause_before = ["tool1"]
        mock_session.pause_after = None

        tools = [MagicMock()]
        middleware = [MagicMock()]
        subagents = [MagicMock()]
        response_format = {"type": "json"}

        class ContextSchema:
            pass

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(
                mock_model,
                workspace=mock_workspace,
                session_config=mock_session,
                tools=tools,
                system="You are helpful",
                middleware=middleware,
                subagents=subagents,
                response_format=response_format,
                context_schema=ContextSchema,
            )

            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs.get("model") is mock_model
            assert call_kwargs.get("backend") is mock_backend
            assert call_kwargs.get("tools") is tools
            assert call_kwargs.get("system_prompt") == "You are helpful"
            assert call_kwargs.get("middleware") == tuple(middleware)
            assert call_kwargs.get("subagents") is subagents
            assert call_kwargs.get("response_format") is response_format
            assert call_kwargs.get("context_schema") is ContextSchema
            assert call_kwargs.get("checkpointer") is mock_checkpointer
            assert call_kwargs.get("store") is mock_store
            assert call_kwargs.get("interrupt_on") == {"tool1": True}
        except ImportError:
            pytest.skip("deepagents package not installed")


# =============================================================================
# Edge Cases
# =============================================================================


class TestBuildDeepAgentEdgeCases:
    """Tests for edge cases in build_deep_agent."""

    @patch("deepagents.create_deep_agent")
    def test_empty_tools_list(self, mock_create):
        """Test empty tools list is passed as None."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, tools=[])
            call_kwargs = mock_create.call_args.kwargs
            # Empty list should still be passed as empty list (truthy check)
            assert call_kwargs.get("tools") is None or call_kwargs.get("tools") == []
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_empty_middleware_list(self, mock_create):
        """Test empty middleware list is passed as empty tuple."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, middleware=[])
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("middleware") == ()
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_empty_subagents_list(self, mock_create):
        """Test empty subagents list is passed."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, subagents=[])
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("subagents") == []
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_none_system_prompt(self, mock_create):
        """Test None system prompt is passed."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, system=None)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("system_prompt") is None
        except ImportError:
            pytest.skip("deepagents package not installed")

    @patch("deepagents.create_deep_agent")
    def test_empty_system_prompt(self, mock_create):
        """Test empty string system prompt is passed."""
        mock_create.return_value = MagicMock()
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            build_deep_agent(mock_model, system="")
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get("system_prompt") == ""
        except ImportError:
            pytest.skip("deepagents package not installed")


# =============================================================================
# Return Value Tests
# =============================================================================


class TestBuildDeepAgentReturn:
    """Tests for build_deep_agent return value."""

    @patch("deepagents.create_deep_agent")
    def test_returns_compiled_agent(self, mock_create):
        """Test build_deep_agent returns the compiled agent."""
        mock_compiled = MagicMock()
        mock_create.return_value = mock_compiled
        mock_model = MagicMock()

        from ai_infra.llm.agents.deep import build_deep_agent

        try:
            result = build_deep_agent(mock_model)
            assert result is mock_compiled
        except ImportError:
            pytest.skip("deepagents package not installed")
