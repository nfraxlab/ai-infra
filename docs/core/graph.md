# Graph Class

> Stateful workflows with typed state and conditional branching.

## Quick Start

```python
from ai_infra import Graph
from typing import TypedDict

class State(TypedDict):
    messages: list
    count: int

def increment(state: State) -> State:
    return {"count": state["count"] + 1}

def check_done(state: State) -> str:
    return "done" if state["count"] >= 3 else "continue"

graph = Graph(State)
graph.add_node("increment", increment)
graph.add_conditional_edges("increment", check_done, {
    "continue": "increment",
    "done": "__end__"
})
graph.set_entry_point("increment")

result = graph.run({"messages": [], "count": 0})
print(result["count"])  # 3
```

---

## Creating Graphs

### Define State

Use TypedDict to define your workflow state:

```python
from typing import TypedDict, Annotated
from operator import add

class ChatState(TypedDict):
    messages: Annotated[list, add]  # Messages accumulate
    context: str
    iteration: int
```

### Create Graph

```python
from ai_infra import Graph

graph = Graph(ChatState)
```

---

## Adding Nodes

Nodes are functions that transform state:

```python
def fetch_context(state: ChatState) -> ChatState:
    """Fetch relevant context."""
    context = search_database(state["messages"][-1])
    return {"context": context}

def generate_response(state: ChatState) -> ChatState:
    """Generate AI response using context."""
    response = llm.chat(
        state["messages"],
        system=f"Context: {state['context']}"
    )
    return {"messages": [{"role": "assistant", "content": response}]}

graph.add_node("fetch_context", fetch_context)
graph.add_node("generate_response", generate_response)
```

---

## Adding Edges

### Simple Edges

```python
# Linear flow: fetch -> generate -> end
graph.add_edge("fetch_context", "generate_response")
graph.add_edge("generate_response", "__end__")
```

### Conditional Edges

```python
def should_continue(state: ChatState) -> str:
    """Decide next step based on state."""
    if state["iteration"] >= 5:
        return "end"
    if needs_more_context(state):
        return "fetch_more"
    return "respond"

graph.add_conditional_edges(
    "check_state",
    should_continue,
    {
        "end": "__end__",
        "fetch_more": "fetch_context",
        "respond": "generate_response",
    }
)
```

---

## Entry and Exit Points

```python
# Set where the graph starts
graph.set_entry_point("fetch_context")

# Compile the graph
compiled = graph.compile()
```

---

## Running Graphs

### Synchronous

```python
initial_state = {
    "messages": [{"role": "user", "content": "Hello"}],
    "context": "",
    "iteration": 0,
}

result = graph.run(initial_state)
print(result["messages"])
```

### Asynchronous

```python
result = await graph.arun(initial_state)
```

### Streaming

```python
for event in graph.stream(initial_state):
    print(f"Node: {event['node']}")
    print(f"State: {event['state']}")
```

---

## Resilience Policies

Pass ai-infra graph policy objects through `node_defaults` to protect every
node, then use `node_policies` only where a node needs an override. Per-node
values take precedence over graph defaults.

```python
from ai_infra.graph import RetryPolicy, TimeoutPolicy

graph = Graph(
    nodes={
        "fetch": fetch_context,
        "respond": generate_response,
    },
    edges=[("fetch", "respond")],
    node_defaults={
        "retry_policy": RetryPolicy(
            initial_interval=0.5,
            max_attempts=3,
        ),
    },
    node_policies={
        "fetch": {
            "timeout": TimeoutPolicy(run_timeout=15),
        },
    },
)
```

`node_defaults` accepts `retry_policy`, `cache_policy`, `error_handler`, and
`timeout`. `node_policies` additionally accepts `trace_policy`. Timeout policies
require async nodes because synchronous Python work cannot be safely cancelled.

---

## With Agent Nodes

Integrate agents into graph nodes:

```python
from ai_infra import Graph, Agent

agent = Agent(tools=[search_tool, calculate_tool])

def agent_node(state: State) -> State:
    """Run agent as a graph node."""
    result = agent.run(state["messages"][-1]["content"])
    return {"messages": [{"role": "assistant", "content": result}]}

graph.add_node("agent", agent_node)
```

---

## Parallel Execution

Run nodes in parallel:

```python
from ai_infra import Graph

def task_a(state):
    return {"result_a": do_task_a()}

def task_b(state):
    return {"result_b": do_task_b()}

def combine(state):
    return {"final": f"{state['result_a']} + {state['result_b']}"}

graph = Graph(State)
graph.add_node("task_a", task_a)
graph.add_node("task_b", task_b)
graph.add_node("combine", combine)

# Both tasks run after start
graph.add_edge("__start__", "task_a")
graph.add_edge("__start__", "task_b")

# Combine waits for both
graph.add_edge("task_a", "combine")
graph.add_edge("task_b", "combine")
```

---

## Checkpointing

Save and resume graph state:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

graph = Graph(State)
# ... add nodes and edges ...

compiled = graph.compile(checkpointer=checkpointer)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = compiled.invoke(initial_state, config)

# Later, resume from checkpoint
result = compiled.invoke({"messages": [new_message]}, config)
```

---

## Visualization

Generate graph diagrams:

```python
# Get Mermaid diagram
mermaid = graph.get_mermaid()
print(mermaid)

# Or ASCII representation
print(graph.draw_ascii())
```

---

## Error Handling

```python
from ai_infra import Graph
from ai_infra.errors import AIInfraError

try:
    result = graph.run(initial_state)
except AIInfraError as e:
    print(f"Graph execution failed: {e}")
```

---

## Best Practices

1. **Keep nodes focused** - Each node should do one thing
2. **Use typed state** - TypedDict helps catch errors
3. **Handle edge cases** - Conditional edges should cover all paths
4. **Add logging** - Use callbacks for observability
5. **Test incrementally** - Test each node independently

---

## See Also

- [Agent](agents.md) - Use agents as graph nodes
- [LLM](llm.md) - LLM calls within nodes
- [Tracing](../infrastructure/tracing.md) - Observability for graphs
