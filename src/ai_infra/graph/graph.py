from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from ai_infra.graph.models import GraphStructure
from ai_infra.graph.utils import (
    build_edges_enhanced,
    make_hook,
    make_trace_fn,
    make_trace_wrapper,
    normalize_initial_state,
    normalize_node_definitions,
    normalize_stream_mode,
    validate_state_against_schema,
    wrap_node,
)

_NODE_POLICY_KEYS = frozenset(
    {"retry_policy", "cache_policy", "error_handler", "timeout", "trace_policy"}
)
_NODE_DEFAULT_KEYS = _NODE_POLICY_KEYS - {"trace_policy"}


class Graph:
    """
    Production-ready workflow graph with zero-config building.

    Supports multiple initialization patterns:

    1. Simple dict-based (GOAL: minimal boilerplate):
       ```python
       graph = Graph(
           nodes={
               "analyze": analyze_func,
               "decide": decide_func,
               "execute": execute_func,
           },
           edges=[
               ("analyze", "decide"),
               ("decide", "execute", lambda state: state["should_execute"]),
           ],
           entry="analyze",
       )
       ```

    2. Any callable as node:
       ```python
       graph = Graph(nodes={
           "func": my_function,
           "lambda": lambda state: {"output": state["input"] * 2},
           "method": my_instance.method,
           "agent": my_agent.run,
       })
       ```

    3. Full graph configuration:
       ```python
       graph = Graph(
           nodes={...},
           edges=[...],
           checkpointer=MemorySaver(),
           interrupt_before=["dangerous_node"],
           interrupt_after=["checkpoint_node"],
           node_defaults={"retry_policy": RetryPolicy(max_attempts=3)},
           node_policies={"dangerous_node": {"timeout": 30}},
       )
       ```

    4. Type-safe state:
       ```python
       class WorkflowState(TypedDict):
           input: str
           output: str

       graph = Graph[WorkflowState](
           nodes={...},
           state_schema=WorkflowState,
           validate_state=True,
       )
       ```
    """

    def __init__(
        self,
        *,
        # New simplified API
        nodes: dict[str, Any] | Sequence | None = None,
        edges: Sequence | None = None,
        entry: str | None = None,
        state_schema: type | None = None,
        validate_state: bool = False,
        # LangGraph power features
        checkpointer=None,
        store=None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        node_defaults: Mapping[str, Any] | None = None,
        node_policies: Mapping[str, Mapping[str, Any]] | None = None,
        # Legacy API (backward compatible)
        state_type: type | None = None,
        node_definitions: Sequence | dict | None = None,
    ):
        # Handle API aliases (new names preferred, legacy still works)
        nodes = nodes or node_definitions
        state_type = state_schema or state_type

        if nodes is None:
            raise ValueError("nodes (or node_definitions) is required")
        if edges is None:
            raise ValueError("edges is required")

        # Default state_type to dict if not provided
        if state_type is None:
            state_type = dict

        if not (
            isinstance(state_type, type)
            and (issubclass(state_type, dict) or hasattr(state_type, "__annotations__"))
        ):
            raise ValueError("state_schema must be a TypedDict or dict subclass")
        self.state_type = state_type
        self._validate_state = validate_state

        # Normalize node definitions
        normalized_nodes: dict[str, Any] = normalize_node_definitions(nodes)
        self.node_definitions = list(normalized_nodes.items())
        node_names = list(normalized_nodes.keys())

        # Build edges with enhanced tuple support (including conditional)
        regular_edges, conditional_edges = build_edges_enhanced(node_names, edges, entry)
        self.edges = regular_edges
        self.conditional_edges = conditional_edges

        # LangGraph features
        self._checkpointer = checkpointer
        self._store = store
        self._interrupt_before = interrupt_before or []
        self._interrupt_after = interrupt_after or []
        self._node_defaults = self._normalize_node_policy_options(
            node_defaults,
            allowed_keys=_NODE_DEFAULT_KEYS,
            label="node_defaults",
        )
        self._node_policies = self._normalize_node_policies(node_policies, node_names)

        # Build and compile graph
        self.graph = self._build_graph().compile(
            checkpointer=self._checkpointer,
            store=self._store,
            interrupt_before=self._interrupt_before or None,
            interrupt_after=self._interrupt_after or None,
        )

    def _build_graph(self, node_items=None, sync: bool = False) -> StateGraph:
        wf = StateGraph(self.state_type)  # type: ignore[type-var]
        if self._node_defaults:
            wf.set_node_defaults(**self._node_defaults)
        node_items = node_items or self.node_definitions
        for name, fn in node_items:
            wf.add_node(name, wrap_node(fn, sync), **self._node_policies.get(name, {}))
        for start, router_fn, path_map in self.conditional_edges:
            wf.add_conditional_edges(start, wrap_node(router_fn, sync), path_map)
        for start, end in self.edges:
            wf.add_edge(start, end)
        return wf

    @staticmethod
    def _normalize_node_policy_options(
        options: Mapping[str, Any] | None,
        *,
        allowed_keys: frozenset[str],
        label: str,
    ) -> dict[str, Any]:
        if options is None:
            return {}
        if not isinstance(options, Mapping):
            raise ValueError(f"{label} must be a mapping of native LangGraph policy options")

        unknown_keys = set(options) - allowed_keys
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise ValueError(f"{label} contains unsupported policy options: {keys}")
        return dict(options)

    def _normalize_node_policies(
        self,
        policies: Mapping[str, Mapping[str, Any]] | None,
        node_names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        if policies is None:
            return {}
        if not isinstance(policies, Mapping):
            raise ValueError("node_policies must map node names to policy options")

        unknown_nodes = set(policies) - set(node_names)
        if unknown_nodes:
            names = ", ".join(sorted(unknown_nodes))
            raise ValueError(f"node_policies contains unknown nodes: {names}")

        return {
            name: self._normalize_node_policy_options(
                options,
                allowed_keys=_NODE_POLICY_KEYS,
                label=f"node_policies[{name!r}]",
            )
            for name, options in policies.items()
        }

    def _prepare_run(
        self,
        initial_state=None,
        *,
        config=None,
        on_enter=None,
        on_exit=None,
        trace=None,
        sync: bool = False,
        **kwargs,
    ):
        initial_state = normalize_initial_state(initial_state, kwargs)

        # Validate state if enabled
        if self._validate_state and initial_state is not None:
            validate_state_against_schema(initial_state, self.state_type)

        # fast path: no hooks -> use cached compiled graph
        if not (on_enter or on_exit or trace):
            compiled = (
                self._build_graph(sync=sync).compile(
                    checkpointer=self._checkpointer,
                    store=self._store,
                    interrupt_before=self._interrupt_before or None,
                    interrupt_after=self._interrupt_after or None,
                )
                if sync
                else self.graph
            )
            return compiled, initial_state, config

        # else, patch & recompile
        on_enter_fn = make_hook(on_enter, sync=sync)
        on_exit_fn = make_hook(on_exit, sync=sync)
        trace_fn = make_trace_fn(trace, sync=sync)
        patched_nodes = [
            (
                name,
                make_trace_wrapper(
                    name, wrap_node(fn, sync), on_enter_fn, on_exit_fn, trace_fn, sync
                ),
            )
            for name, fn in self.node_definitions
        ]
        compiled = self._build_graph(node_items=patched_nodes, sync=sync).compile(
            checkpointer=self._checkpointer,
            store=self._store,
            interrupt_before=self._interrupt_before or None,
            interrupt_after=self._interrupt_after or None,
        )
        return compiled, initial_state, config

    # ---- invoke -----------------------------------------------------------------
    async def arun(
        self,
        initial_state=None,
        *,
        config=None,
        on_enter=None,
        on_exit=None,
        trace=None,
        **kwargs,
    ) -> Any:
        compiled, initial_state, config = self._prepare_run(
            initial_state,
            config=config,
            on_enter=on_enter,
            on_exit=on_exit,
            trace=trace,
            sync=False,
            **kwargs,
        )
        return (
            await compiled.ainvoke(initial_state, config=config)
            if config is not None
            else await compiled.ainvoke(initial_state)
        )

    def run(
        self,
        initial_state=None,
        *,
        config=None,
        on_enter=None,
        on_exit=None,
        trace=None,
        **kwargs,
    ) -> Any:
        compiled, initial_state, config = self._prepare_run(
            initial_state,
            config=config,
            on_enter=on_enter,
            on_exit=on_exit,
            trace=trace,
            sync=True,
            **kwargs,
        )
        return (
            compiled.invoke(initial_state, config=config)
            if config is not None
            else compiled.invoke(initial_state)
        )

    # ---- streaming ---------------------------------------------------------------
    async def astream(
        self, initial_state=None, *, config=None, stream_mode=("updates", "values")
    ) -> AsyncIterator[tuple[str, Any]]:
        stream_mode = normalize_stream_mode(stream_mode)
        compiled, initial_state, config = self._prepare_run(
            initial_state, config=config, sync=False
        )
        async for mode, chunk in compiled.astream(
            initial_state, config=config, stream_mode=stream_mode
        ):
            yield mode, chunk

    def stream(
        self, initial_state=None, *, config=None, stream_mode=("updates", "values")
    ) -> Iterator[tuple[str, Any]]:
        stream_mode = normalize_stream_mode(stream_mode)
        compiled, initial_state, config = self._prepare_run(initial_state, config=config, sync=True)
        for mode, chunk in compiled.stream(initial_state, config=config, stream_mode=stream_mode):
            yield mode, chunk

    async def astream_values(self, initial_state=None, *, config=None):
        async for _, chunk in self.astream(initial_state, config=config, stream_mode="values"):
            yield chunk

    def stream_values(self, initial_state=None, *, config=None):
        for _, chunk in self.stream(initial_state, config=config, stream_mode="values"):
            yield chunk

    # ---- analysis / debug --------------------------------------------------------
    def analyze(self) -> GraphStructure:
        nodes = [name for name, _ in self.node_definitions]
        entry_points = [end for start, end in self.edges if start == START] or nodes[:1]
        exit_points = [start for start, end in self.edges if end == END]
        conditional_edges_data = (
            [
                {
                    "start": start,
                    "router_function": getattr(router_fn, "__name__", str(router_fn)),
                    "path_options": list(path_map.keys()),
                }
                for start, router_fn, path_map in self.conditional_edges
            ]
            if self.conditional_edges
            else None
        )
        state_schema = {
            key: getattr(value, "__name__", str(value))
            for key, value in getattr(self.state_type, "__annotations__", {}).items()
        }
        # reachability
        reachable = set(entry_points)
        edges_map: dict[str, list[str]] = {start: [] for start, _ in self.edges}
        for start, end in self.edges:
            edges_map.setdefault(start, []).append(end)
        queue = list(entry_points)
        while queue:
            node = queue.pop(0)
            for nbr in edges_map.get(node, []):
                if nbr not in reachable and nbr not in (START, END):
                    reachable.add(nbr)
                    queue.append(nbr)
        unreachable = [n for n in nodes if n not in reachable]

        return GraphStructure(
            state_type_name=self.state_type.__name__,
            state_schema=state_schema,
            node_count=len(nodes),
            nodes=nodes,
            edge_count=len(self.edges),
            edges=self.edges,
            conditional_edge_count=len(self.conditional_edges) if self.conditional_edges else 0,
            conditional_edges=conditional_edges_data,
            entry_points=entry_points,
            exit_points=exit_points,
            has_memory=self._checkpointer is not None,
            unreachable=unreachable,
        )

    def describe(self) -> dict:
        return self.analyze().model_dump()

    def get_state(self, config):
        return self.graph.get_state(config)

    def get_state_history(self, config):
        return list(self.graph.get_state_history(config))

    def get_arch_diagram(self) -> str:
        return self.graph.get_graph().draw_mermaid()
