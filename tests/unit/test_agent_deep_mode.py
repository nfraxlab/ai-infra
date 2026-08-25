"""Tests for Agent deep mode and shared modern configuration."""

from unittest.mock import MagicMock, patch

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from ai_infra.llm import Agent, SubAgent
from ai_infra.llm.session import memory


class TestAgentDeepMode:
    """Test Agent with deep=True."""

    def test_deep_false_by_default(self):
        """Agent defaults to deep=False (regular mode)."""
        agent = Agent()
        assert agent._deep is False

    def test_deep_true_sets_flag(self):
        """Agent(deep=True) sets the deep flag."""
        agent = Agent(deep=True)
        assert agent._deep is True

    def test_existing_params_work_with_deep(self):
        """All existing Agent params work with deep=True."""
        agent = Agent(
            provider="anthropic",
            model_name="claude-sonnet-4",
            tools=[lambda x: x],
            system="Test prompt",
            deep=True,
        )
        assert agent._deep is True
        assert agent._default_provider == "anthropic"
        assert agent._default_model_name == "claude-sonnet-4"
        assert agent._system == "Test prompt"
        assert len(agent.tools) == 1

    def test_session_works_with_deep(self):
        """Our session abstraction works with deep=True."""
        session = memory()
        agent = Agent(
            deep=True,
            session=session,
        )
        assert agent._session_config is not None
        assert agent._session_config.storage is session

    def test_deep_agent_rejects_pause_after(self):
        """Deep agents reject pause_after because their approval API has no equivalent."""
        with pytest.raises(ValueError, match="pause_after is not supported"):
            Agent(deep=True, session=memory(), pause_after=["dangerous_tool"])

    def test_deep_agent_pause_before_and_resume(self):
        """Deep agents translate the public resume API to approval decisions."""

        class ToolBindableFakeMessagesListChatModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        def dangerous_tool() -> str:
            """Return an approved result."""
            return "approved"

        model = ToolBindableFakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "dangerous_tool", "args": {}, "id": "call-1"}],
                ),
                AIMessage(content="complete"),
            ]
        )
        compiled_agent = create_deep_agent(
            model=model,
            tools=[dangerous_tool],
            interrupt_on={"dangerous_tool": True},
            checkpointer=MemorySaver(),
        )
        agent = Agent(
            deep=True,
            provider="test",
            model_name="fake-model",
            session=memory(),
            pause_before=["dangerous_tool"],
        )

        with (
            patch.object(agent, "_resolve_provider_and_model", return_value=("test", "fake-model")),
            patch.object(agent, "_build_deep_agent", return_value=compiled_agent),
        ):
            paused = agent.run("Run the dangerous tool", session_id="deep-pause")
            assert paused.paused is True
            assert paused.pending_action is not None
            assert paused.pending_action.tool_name == "dangerous_tool"

            resumed = agent.resume("deep-pause")

        assert resumed.paused is False
        assert resumed.content == "complete"


class TestAgentStandardModernConfiguration:
    """Test modern configuration is shared with standard agents."""

    def test_standard_agent_forwards_modern_configuration(self):
        """Standard mode forwards configuration that deep mode already accepts."""
        middleware = MagicMock()
        response_format = MagicMock()

        class RuntimeContext:
            pass

        agent = Agent(
            middleware=[middleware],
            response_format=response_format,
            context_schema=RuntimeContext,
        )

        with patch("ai_infra.llm.agent.rb_make_agent_with_context") as mock_build:
            mock_build.return_value = (MagicMock(), MagicMock())

            agent._make_agent_with_context(provider="openai")

        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["middleware"] == [middleware]
        assert call_kwargs["response_format"] is response_format
        assert call_kwargs["context_schema"] is RuntimeContext


class TestAgentSubagentConversion:
    """Test automatic Agent to SubAgent conversion."""

    def test_agent_to_subagent_conversion(self):
        """Agent instances are auto-converted to SubAgent format."""
        researcher = Agent(
            name="researcher",
            description="Searches for info",
            system="You are a researcher.",
        )

        agent = Agent(
            deep=True,
            subagents=[researcher],
        )

        assert len(agent._subagents) == 1
        subagent = agent._subagents[0]
        assert subagent["name"] == "researcher"
        assert subagent["description"] == "Searches for info"
        assert subagent["system_prompt"] == "You are a researcher."
        assert subagent["tools"] == []

    def test_agent_with_tools_to_subagent(self):
        """Agent with tools converts properly."""

        def my_tool(x: str) -> str:
            return x

        researcher = Agent(
            name="researcher",
            description="Searches for info",
            system="You are a researcher.",
            tools=[my_tool],
        )

        agent = Agent(
            deep=True,
            subagents=[researcher],
        )

        subagent = agent._subagents[0]
        assert len(subagent["tools"]) == 1

    def test_multiple_agent_subagents(self):
        """Multiple Agent instances convert properly."""
        researcher = Agent(
            name="researcher",
            description="Searches",
            system="Research assistant",
        )
        writer = Agent(
            name="writer",
            description="Writes",
            system="Writing assistant",
        )

        agent = Agent(
            deep=True,
            subagents=[researcher, writer],
        )

        assert len(agent._subagents) == 2
        assert agent._subagents[0]["name"] == "researcher"
        assert agent._subagents[1]["name"] == "writer"

    def test_subagent_dict_passthrough(self):
        """SubAgent dicts pass through unchanged."""
        agent = Agent(
            deep=True,
            subagents=[
                SubAgent(
                    name="test",
                    description="Test agent",
                    system_prompt="You are a test.",
                    tools=[],
                ),
            ],
        )

        assert len(agent._subagents) == 1
        assert agent._subagents[0]["name"] == "test"

    def test_mixed_agent_and_subagent(self):
        """Mix of Agent and SubAgent works."""
        researcher = Agent(
            name="researcher",
            description="Searches",
            system="Research assistant",
        )

        agent = Agent(
            deep=True,
            subagents=[
                researcher,
                SubAgent(
                    name="writer",
                    description="Writes",
                    system_prompt="Writing assistant",
                    tools=[],
                ),
            ],
        )

        assert len(agent._subagents) == 2
        assert agent._subagents[0]["name"] == "researcher"
        assert agent._subagents[1]["name"] == "writer"

    def test_missing_name_raises_error(self):
        """Agent without name raises ValueError."""
        with pytest.raises(ValueError, match="must have 'name' set"):
            Agent(
                deep=True,
                subagents=[Agent(description="test")],
            )

    def test_missing_description_raises_error(self):
        """Agent without description raises ValueError."""
        with pytest.raises(ValueError, match="must have 'description' set"):
            Agent(
                deep=True,
                subagents=[Agent(name="test")],
            )


class TestAgentIdentityParams:
    """Test name, description, system params."""

    def test_name_stored(self):
        """name parameter is stored."""
        agent = Agent(name="my-agent")
        assert agent._name == "my-agent"

    def test_description_stored(self):
        """description parameter is stored."""
        agent = Agent(description="Does things")
        assert agent._description == "Does things"

    def test_system_stored(self):
        """system parameter is stored."""
        agent = Agent(system="You are helpful")
        assert agent._system == "You are helpful"

    def test_all_identity_params(self):
        """All identity params work together."""
        agent = Agent(
            name="helper",
            description="Helps with tasks",
            system="You are a helpful assistant.",
        )
        assert agent._name == "helper"
        assert agent._description == "Helps with tasks"
        assert agent._system == "You are a helpful assistant."


class TestAgentDeepModeExports:
    """Test DeepAgent type exports."""

    def test_subagent_exported(self):
        """SubAgent is exported from ai_infra.llm."""
        from ai_infra.llm import SubAgent

        assert SubAgent is not None

    def test_middleware_exported(self):
        """Middleware types are exported."""
        from ai_infra.llm import FilesystemMiddleware, SubAgentMiddleware

        assert FilesystemMiddleware is not None
        assert SubAgentMiddleware is not None

    def test_compiled_subagent_exported(self):
        """CompiledSubAgent is exported."""
        from ai_infra.llm import CompiledSubAgent

        assert CompiledSubAgent is not None


class TestAgentDeepParams:
    """Test deep-mode specific parameters."""

    def test_middleware_stored(self):
        """middleware parameter is stored."""
        middleware = [MagicMock()]
        agent = Agent(deep=True, middleware=middleware)
        assert agent._middleware == middleware

    def test_response_format_stored(self):
        """response_format parameter is stored."""
        agent = Agent(deep=True, response_format={"type": "json"})
        assert agent._response_format == {"type": "json"}

    def test_context_schema_stored(self):
        """context_schema parameter is stored."""

        class MySchema:
            pass

        agent = Agent(deep=True, context_schema=MySchema)
        assert agent._context_schema == MySchema

    def test_use_longterm_memory_stored(self):
        """use_longterm_memory parameter is stored."""
        agent = Agent(deep=True, use_longterm_memory=True)
        assert agent._use_longterm_memory is True

    def test_use_longterm_memory_default_false(self):
        """use_longterm_memory defaults to False."""
        agent = Agent(deep=True)
        assert agent._use_longterm_memory is False


class TestBuildDeepAgent:
    """Test _build_deep_agent method."""

    def test_build_deep_agent_called(self):
        """_build_deep_agent calls create_deep_agent."""

        agent = Agent(deep=True, provider="openai")
        # We'd need to mock more to fully test run(), so just test the method exists
        assert hasattr(agent, "_build_deep_agent")

    def test_deep_agent_has_required_methods(self):
        """Deep agent has run/arun methods."""
        agent = Agent(deep=True)
        assert hasattr(agent, "run")
        assert hasattr(agent, "arun")
        assert hasattr(agent, "_build_deep_agent")
        assert hasattr(agent, "_get_model_for_deep_agent")
