"""Tests for ai_infra.graph module."""

import pytest
from typing_extensions import TypedDict

from ai_infra.graph import Graph, RetryPolicy, TimeoutPolicy
from ai_infra.graph.models import ConditionalEdge, Edge

# ============================================================================
# Test State Types
# ============================================================================


class SimpleState(TypedDict):
    input: str
    output: str


class WorkflowState(TypedDict):
    input: str
    analysis: str
    decision: bool
    output: str


class CounterState(TypedDict):
    count: int
    history: list


# ============================================================================
# Test Node Functions
# ============================================================================


def analyze_node(state: dict) -> dict:
    """Simple analysis node."""
    return {"analysis": f"analyzed: {state.get('input', '')}"}


def decide_node(state: dict) -> dict:
    """Decision node based on analysis."""
    should_execute = "good" in state.get("analysis", "")
    return {"decision": should_execute}


def execute_node(state: dict) -> dict:
    """Execution node."""
    return {"output": f"executed with: {state.get('analysis', '')}"}


def increment_node(state: dict) -> dict:
    """Increment counter."""
    count = state.get("count", 0) + 1
    history = list(state.get("history", [])) + [count]
    return {"count": count, "history": history}


def double_node(state: dict) -> dict:
    """Double the count."""
    return {"count": state.get("count", 0) * 2}


async def async_node(state: dict) -> dict:
    """Async node for testing."""
    return {"output": f"async: {state.get('input', '')}"}


# ============================================================================
# 3.3.1 Zero-Config Graph Building Tests
# ============================================================================


class TestZeroConfigGraphBuilding:
    """Tests for 3.3.1 Zero-Config Graph Building."""

    def test_simple_dict_based_graph(self):
        """Test simple graph definition with dict-based nodes."""
        graph = Graph(
            nodes={
                "analyze": analyze_node,
                "execute": execute_node,
            },
            edges=[
                ("analyze", "execute"),
            ],
        )

        result = graph.run({"input": "test data"})
        # LangGraph merges state, so we check for the final output
        assert "output" in result
        assert "analyzed: test data" in result["output"]

    def test_explicit_entry_point(self):
        """Test graph with explicit entry point."""
        graph = Graph(
            nodes={
                "step1": lambda s: {"step": 1},
                "step2": lambda s: {"step": 2},
            },
            edges=[
                ("step1", "step2"),
            ],
            entry="step1",
        )

        result = graph.run({})
        assert result["step"] == 2

    def test_conditional_edge_with_tuple(self):
        """Test conditional edge using 3-tuple syntax."""
        graph = Graph(
            nodes={
                "check": lambda s: {"should_continue": s.get("input") == "go"},
                "process": lambda s: {"output": "processed"},
            },
            edges=[
                ("check", "process", lambda s: s.get("should_continue", False)),
            ],
        )

        # Should continue
        result = graph.run({"input": "go"})
        assert result.get("output") == "processed"

        # Should not continue (ends early)
        result = graph.run({"input": "stop"})
        assert result.get("output") is None

    def test_node_from_lambda(self):
        """Test that lambdas work as nodes."""
        graph = Graph(
            nodes={
                "double": lambda state: {"value": state.get("value", 0) * 2},
            },
            edges=[],  # Single node, auto START/END
        )

        result = graph.run({"value": 5})
        assert result["value"] == 10

    def test_node_from_function(self):
        """Test that regular functions work as nodes."""
        graph = Graph(
            nodes={
                "inc": increment_node,
            },
            edges=[],
        )

        result = graph.run({"count": 0, "history": []})
        assert result["count"] == 1

    def test_node_from_method(self):
        """Test that instance methods work as nodes."""

        class Processor:
            def __init__(self, multiplier: int):
                self.multiplier = multiplier

            def process(self, state: dict) -> dict:
                return {"value": state.get("value", 0) * self.multiplier}

        processor = Processor(3)
        graph = Graph(
            nodes={
                "multiply": processor.process,
            },
            edges=[],
        )

        result = graph.run({"value": 4})
        assert result["value"] == 12

    def test_multi_node_chain(self):
        """Test a chain of multiple nodes."""

        # Use nodes that accumulate state rather than replace
        def analyze(state):
            return {
                "analysis": f"analyzed: {state.get('input', '')}",
                "input": state.get("input"),
            }

        def decide(state):
            should_execute = "good" in state.get("analysis", "")
            return {"decision": should_execute, "analysis": state.get("analysis")}

        def execute(state):
            return {
                "output": f"executed: {state.get('analysis', '')}",
                "decision": state.get("decision"),
            }

        graph = Graph(
            nodes={
                "analyze": analyze,
                "decide": decide,
                "execute": execute,
            },
            edges=[
                ("analyze", "decide"),
                ("decide", "execute"),
            ],
            entry="analyze",
        )

        result = graph.run({"input": "good data"})
        assert result["decision"] is True
        assert "executed" in result["output"]


# ============================================================================
# 3.3.2 Full Flexibility: LangGraph Power Tests
# ============================================================================


class TestLangGraphPower:
    """Tests for 3.3.2 Full Flexibility: LangGraph Power."""

    def test_checkpointer_support(self):
        """Test that checkpointer parameter is accepted."""
        # Just test that it doesn't error - actual persistence requires MemorySaver
        graph = Graph(
            nodes={"node": lambda s: {"done": True}},
            edges=[],
            checkpointer=None,  # Explicitly None
        )
        result = graph.run({})
        assert result["done"] is True

    def test_store_support(self):
        """Test that store parameter is accepted."""
        graph = Graph(
            nodes={"node": lambda s: {"done": True}},
            edges=[],
            store=None,
        )
        result = graph.run({})
        assert result["done"] is True

    def test_interrupt_before_parameter(self):
        """Test that interrupt_before is accepted."""
        graph = Graph(
            nodes={
                "safe": lambda s: {"step": "safe"},
                "dangerous": lambda s: {"step": "dangerous"},
            },
            edges=[("safe", "dangerous")],
            interrupt_before=["dangerous"],
        )
        # Graph should compile without error
        assert graph._interrupt_before == ["dangerous"]

    def test_interrupt_after_parameter(self):
        """Test that interrupt_after is accepted."""
        graph = Graph(
            nodes={
                "checkpoint": lambda s: {"step": "checkpoint"},
                "continue": lambda s: {"step": "continue"},
            },
            edges=[("checkpoint", "continue")],
            interrupt_after=["checkpoint"],
        )
        assert graph._interrupt_after == ["checkpoint"]

    def test_access_underlying_langgraph(self):
        """Test that graph property exposes underlying LangGraph."""
        graph = Graph(
            nodes={"node": lambda s: {"done": True}},
            edges=[],
        )

        # Access underlying compiled graph
        assert graph.graph is not None
        # Should have invoke method
        assert hasattr(graph.graph, "invoke")
        assert hasattr(graph.graph, "ainvoke")


class TestGraphResiliencePolicies:
    """Tests for LangGraph-native resilience policy forwarding."""

    @pytest.mark.asyncio
    async def test_per_node_retry_policy_retries_failed_node(self):
        """A per-node RetryPolicy is forwarded to LangGraph execution."""
        attempts = 0

        async def flaky_node(state: dict) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("retry me")
            return {"attempts": attempts}

        timeout = TimeoutPolicy(run_timeout=1)
        graph = Graph(
            nodes={"flaky": flaky_node},
            edges=[],
            node_policies={
                "flaky": {
                    "retry_policy": RetryPolicy(
                        initial_interval=0,
                        jitter=False,
                        max_attempts=2,
                        retry_on=ValueError,
                    ),
                    "timeout": timeout,
                }
            },
        )

        assert (await graph.arun({}))["attempts"] == 2
        assert attempts == 2
        assert graph._node_policies["flaky"]["timeout"] is timeout

    def test_node_defaults_apply_to_all_nodes(self):
        """A graph-wide RetryPolicy is applied when a node has no override."""
        attempts = 0

        def flaky_node(state: dict) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("retry me")
            return {"attempts": attempts}

        graph = Graph(
            nodes={"flaky": flaky_node},
            edges=[],
            node_defaults={
                "retry_policy": RetryPolicy(
                    initial_interval=0,
                    jitter=False,
                    max_attempts=2,
                    retry_on=RuntimeError,
                )
            },
        )

        assert graph.run({})["attempts"] == 2
        assert attempts == 2

    def test_node_policies_reject_unknown_node_and_option(self):
        """Invalid policies fail at graph construction rather than being ignored."""
        with pytest.raises(ValueError, match="unknown nodes: missing"):
            Graph(
                nodes={"present": lambda state: state},
                edges=[],
                node_policies={"missing": {}},
            )

        with pytest.raises(ValueError, match="unsupported policy options: unsupported"):
            Graph(
                nodes={"present": lambda state: state},
                edges=[],
                node_defaults={"unsupported": True},
            )


# ============================================================================
# 3.3.3 Production Ready: State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for 3.3.3 Production Ready: State Management."""

    def test_state_schema_alias(self):
        """Test that state_schema works as alias for state_type."""
        graph = Graph(
            nodes={"node": lambda s: {"output": "done"}},
            edges=[],
            state_schema=SimpleState,
        )

        assert graph.state_type is SimpleState

    def test_legacy_state_type_still_works(self):
        """Test backward compatibility with state_type parameter."""
        graph = Graph(
            nodes={"node": lambda s: {"output": "done"}},
            edges=[],
            state_type=SimpleState,
        )

        assert graph.state_type is SimpleState

    def test_default_state_type_is_dict(self):
        """Test that state_type defaults to dict when not provided."""
        graph = Graph(
            nodes={"node": lambda s: {"output": "done"}},
            edges=[],
        )

        assert graph.state_type is dict

    def test_validate_state_missing_required_keys(self):
        """Test that validate_state catches missing required keys."""
        graph = Graph(
            nodes={"node": lambda s: {"output": "done"}},
            edges=[],
            state_schema=SimpleState,
            validate_state=True,
        )

        # Missing 'input' and 'output' keys
        with pytest.raises(ValueError, match="missing required keys"):
            graph.run({})

    def test_validate_state_type_mismatch(self):
        """Test that validate_state catches type mismatches."""
        graph = Graph(
            nodes={"node": lambda s: {"output": "done"}},
            edges=[],
            state_schema=SimpleState,
            validate_state=True,
        )

        # 'input' should be str, not int
        with pytest.raises(ValueError, match="expected str"):
            graph.run({"input": 123, "output": "test"})

    def test_validate_state_passes_valid_state(self):
        """Test that validation passes for valid state."""
        graph = Graph(
            nodes={"node": lambda s: s},
            edges=[],
            state_schema=SimpleState,
            validate_state=True,
        )

        result = graph.run({"input": "test", "output": "result"})
        assert result["input"] == "test"

    def test_typed_state_with_workflow(self):
        """Test type-safe state through a workflow."""
        graph = Graph(
            nodes={
                "analyze": analyze_node,
                "decide": decide_node,
                "execute": execute_node,
            },
            edges=[
                ("analyze", "decide"),
                ("decide", "execute"),
            ],
            state_schema=WorkflowState,
        )

        result = graph.run({"input": "good test"})
        assert "analysis" in result
        assert "decision" in result
        assert "output" in result


# ============================================================================
# Legacy API Compatibility Tests
# ============================================================================


class TestLegacyAPICompatibility:
    """Tests for backward compatibility with legacy API."""

    def test_node_definitions_alias(self):
        """Test that node_definitions still works (legacy API)."""
        graph = Graph(
            node_definitions={"node": lambda s: {"done": True}},
            edges=[],
        )

        result = graph.run({})
        assert result["done"] is True

    def test_edge_objects(self):
        """Test that Edge objects still work."""
        graph = Graph(
            nodes={
                "a": lambda s: {"step": "a"},
                "b": lambda s: {"step": "b"},
            },
            edges=[
                Edge(start="a", end="b"),
            ],
        )

        result = graph.run({})
        # Final step should be "b"
        assert result["step"] == "b"

    def test_conditional_edge_objects(self):
        """Test that ConditionalEdge objects still work."""

        def router(state):
            return "yes" if state.get("go") else "no"

        graph = Graph(
            nodes={
                "start": lambda s: {"started": True, "go": s.get("go")},
                "yes": lambda s: {"result": "yes"},
                "no": lambda s: {"result": "no"},
            },
            edges=[
                ConditionalEdge(start="start", router_fn=router, targets=["yes", "no"]),
            ],
        )

        # Test yes path
        result = graph.run({"go": True})
        assert result["result"] == "yes"

        # Test no path
        result = graph.run({"go": False})
        assert result["result"] == "no"


# ============================================================================
# Graph Analysis Tests
# ============================================================================


class TestGraphAnalysis:
    """Tests for graph analysis and debugging features."""

    def test_analyze_returns_structure(self):
        """Test that analyze() returns graph structure."""
        graph = Graph(
            nodes={
                "a": lambda s: {"a": True},
                "b": lambda s: {"b": True},
            },
            edges=[("a", "b")],
            state_schema=SimpleState,
        )

        structure = graph.analyze()
        assert structure.node_count == 2
        assert "a" in structure.nodes
        assert "b" in structure.nodes
        assert structure.state_type_name == "SimpleState"

    def test_describe_returns_dict(self):
        """Test that describe() returns a dict."""
        graph = Graph(
            nodes={"node": lambda s: s},
            edges=[],
        )

        desc = graph.describe()
        assert isinstance(desc, dict)
        assert "node_count" in desc

    def test_get_arch_diagram(self):
        """Test that get_arch_diagram returns mermaid diagram."""
        graph = Graph(
            nodes={
                "a": lambda s: s,
                "b": lambda s: s,
            },
            edges=[("a", "b")],
        )

        diagram = graph.get_arch_diagram()
        assert isinstance(diagram, str)
        # Mermaid diagrams typically start with graph or flowchart
        assert "graph" in diagram.lower() or "flowchart" in diagram.lower() or "---" in diagram


# ============================================================================
# Async Tests
# ============================================================================


class TestAsyncGraph:
    """Tests for async graph execution."""

    @pytest.mark.asyncio
    async def test_arun_basic(self):
        """Test basic async execution."""
        graph = Graph(
            nodes={"node": async_node},
            edges=[],
        )

        result = await graph.arun({"input": "test"})
        assert result["output"] == "async: test"

    @pytest.mark.asyncio
    async def test_astream_values(self):
        """Test async streaming values."""
        graph = Graph(
            nodes={
                "a": lambda s: {"step": 1},
                "b": lambda s: {"step": 2},
            },
            edges=[("a", "b")],
        )

        values = []
        async for chunk in graph.astream_values({}):
            values.append(chunk)

        assert len(values) >= 1
        # Final state should have step=2
        assert values[-1]["step"] == 2


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_nodes_raises(self):
        """Test that missing nodes parameter raises."""
        with pytest.raises(ValueError, match=r"nodes.*required"):
            Graph(edges=[])

    def test_missing_edges_raises(self):
        """Test that missing edges parameter raises."""
        with pytest.raises(ValueError, match=r"edges.*required"):
            Graph(nodes={"node": lambda s: s})

    def test_invalid_entry_raises(self):
        """Test that invalid entry node raises."""
        with pytest.raises(ValueError, match=r"Entry node.*not a known node"):
            Graph(
                nodes={"a": lambda s: s},
                edges=[],
                entry="nonexistent",
            )

    def test_invalid_edge_target_raises(self):
        """Test that invalid edge target raises."""
        with pytest.raises(ValueError, match="not a known node"):
            Graph(
                nodes={"a": lambda s: s},
                edges=[("a", "nonexistent")],
            )

    def test_invalid_conditional_fn_raises(self):
        """Test that non-callable in 3-tuple raises."""
        with pytest.raises(ValueError, match="must be a callable"):
            Graph(
                nodes={"a": lambda s: s, "b": lambda s: s},
                edges=[("a", "b", "not a function")],
            )
