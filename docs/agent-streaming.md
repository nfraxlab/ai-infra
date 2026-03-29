# Rich Agent Stream Events

> New event types for building agent UIs: usage tracking, turn lifecycle,
> intent display, and task checklists.

These events extend the base [streaming guide](streaming.md) with richer
metadata for applications that need to visualize agent state.

---

## Usage Events

Track token consumption and cost per LLM call.

**When they fire**: After each LLM call completes — before `tool_end` when
tools are used, and before `done` for the final response.

**Visibility**: `detailed` or `debug` only.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Input tokens consumed |
| `output_tokens` | `int` | Output tokens produced |
| `cost` | `float \| None` | Estimated cost in USD (requires pricing config) |
| `model` | `str \| None` | Model name |

### Example: Token Accounting

```python
from ai_infra import Agent
from ai_infra.llm.streaming import StreamConfig

config = StreamConfig(
    visibility="detailed",
    cost_per_input_token=0.000003,   # $3 per 1M input tokens
    cost_per_output_token=0.000015,  # $15 per 1M output tokens
)

total_cost = 0.0
async for event in agent.astream("Summarize the auth module", stream_config=config):
    if event.type == "usage":
        total_cost += event.cost or 0.0
        print(f"  Tokens: {event.input_tokens} in / {event.output_tokens} out")
        if event.cost:
            print(f"  Cost: ${event.cost:.6f}")
    elif event.type == "token":
        print(event.content, end="", flush=True)

print(f"\nTotal cost: ${total_cost:.6f}")
```

### Example: FastAPI Cost Header

```python
@app.post("/chat")
async def chat(message: str):
    total_input = 0
    total_output = 0

    async def generate():
        nonlocal total_input, total_output
        async for event in agent.astream(message, stream_config=config):
            if event.type == "usage":
                total_input += event.input_tokens or 0
                total_output += event.output_tokens or 0
            yield f"data: {json.dumps(event.to_dict())}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Token-Count": f"{total_input}+{total_output}"},
    )
```

---

## Turn Lifecycle Events

Track agent reasoning/tool-calling iterations for UI state transitions.

**When they fire**: `turn_start` emits when the LLM begins producing output
for a new iteration. `turn_end` emits when all tools for that iteration
complete, or when the agent finishes a direct-answer turn.

**Visibility**: `standard` or higher.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `turn_id` | `int` | Turn counter, 1-indexed. Increments each iteration. |
| `tools_called` | `int \| None` | Number of tools called in this turn (`turn_end` only). |

### Turn Sequence

A typical multi-tool agent flow:

```
turn_start (turn_id=1)
  reasoning → tool_start → tool_end
turn_end (turn_id=1, tools_called=1)

turn_start (turn_id=2)
  reasoning → tool_start → tool_end → tool_start → tool_end
turn_end (turn_id=2, tools_called=2)

turn_start (turn_id=3)
  token token token ...
turn_end (turn_id=3, tools_called=0)
done
```

### Example: UI State Machine

```python
async for event in agent.astream("Refactor the auth module"):
    match event.type:
        case "turn_start":
            ui.show_spinner(f"Turn {event.turn_id}...")
        case "turn_end":
            if event.tools_called:
                ui.update_status(f"Used {event.tools_called} tools")
            ui.hide_spinner()
        case "token":
            ui.append_text(event.content)
        case "done":
            ui.finalize()
```

### Example: Turn Progress Bar

```python
turns = []
async for event in agent.astream(prompt, stream_config=config):
    if event.type == "turn_start":
        turns.append({"id": event.turn_id, "tools": 0})
    elif event.type == "turn_end":
        turns[-1]["tools"] = event.tools_called or 0
        progress = f"Turn {event.turn_id} complete ({event.tools_called} tools)"
        yield {"type": "progress", "message": progress, "turns": len(turns)}
```

---

## Intent Events

Human-readable description of what the agent is doing, derived from tool calls.

**When they fire**: Immediately after each `tool_start`, if the tool name
maps to a known description. Consecutive duplicate intents are deduplicated.

**Visibility**: `standard` or higher.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Human-readable intent (e.g., "Searching codebase") |

### Built-in Intent Map

| Tool Name | Intent |
|-----------|--------|
| `search` | Searching codebase |
| `grep_search` | Searching for text patterns |
| `read_file` | Reading files |
| `write_file` | Writing files |
| `run_python` | Running Python code |
| `run_shell` | Running shell command |
| `create_visualization` | Creating visualization |

### Example: Agent Status Indicator

```python
async for event in agent.astream("Fix the login bug"):
    match event.type:
        case "intent":
            ui.set_status(event.content)  # "Reading files"
        case "tool_start":
            ui.show_tool_badge(event.tool)
        case "token":
            ui.clear_status()
            ui.append_text(event.content)
```

### Custom Intent Map

Override or extend the built-in map via `StreamConfig.tool_intent_map`:

```python
config = StreamConfig(
    tool_intent_map={
        "search_docs": "Searching documentation",
        "query_db": "Querying database",
        "run_tests": "Running test suite",
        "deploy": "Deploying to staging",
    }
)
async for event in agent.astream(prompt, stream_config=config):
    if event.type == "intent":
        print(f"Agent: {event.content}")
```

When `tool_intent_map` is set, it fully replaces the built-in map.

---

## Todo Events

Task checklist updates from the agent's todo/task management tool.

**When they fire**: After `tool_end` for the configured todo tool
(default: `manage_todos`). Only emitted when the tool result contains
parseable todo items.

**Visibility**: `standard` or higher.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `todo_items` | `list[dict]` | List of task items, each with `id`, `title`, `status` |

### Supported Result Formats

The parser handles multiple formats:

```python
# Direct list
[{"id": 1, "title": "Fetch data", "status": "completed"}, ...]

# Dict with items/todos/todoList/todo_items key
{"items": [{"id": 1, "title": "Fetch data", "status": "completed"}, ...]}
{"todos": [...]}
{"todoList": [...]}
```

### Example: Render Task Checklist

```python
async for event in agent.astream("Plan the migration"):
    if event.type == "todo":
        for item in event.todo_items:
            icon = "[x]" if item["status"] == "completed" else "[ ]"
            print(f"  {icon} {item['title']}")
    elif event.type == "token":
        print(event.content, end="", flush=True)
```

### Example: React Task List (Frontend)

```typescript
function TaskList({ todoItems }: { todoItems: TodoItem[] }) {
  return (
    <ul>
      {todoItems.map((item) => (
        <li key={item.id} className={item.status === "completed" ? "done" : ""}>
          <input type="checkbox" checked={item.status === "completed"} readOnly />
          {item.title}
        </li>
      ))}
    </ul>
  );
}

// In your SSE handler:
if (event.type === "todo") {
  setTodoItems(event.todo_items);
}
```

### Custom Todo Tool Name

If your agent uses a different tool name for task management:

```python
config = StreamConfig(todo_tool_name="update_tasks")
async for event in agent.astream(prompt, stream_config=config):
    ...
```

---

## Migration Guide: CopilotEvent to StreamEvent

`Agent.astream()` emits `StreamEvent` objects. `CopilotAgent.stream()` emits
`CopilotEvent` objects. If you are migrating from `CopilotAgent` to `Agent`
(or consuming both), use this mapping.

### Type Mapping

| CopilotEvent Type | StreamEvent Type | Notes |
|-------------------|------------------|-------|
| `token` | `token` | Same semantics |
| `tool_start` | `tool_start` | Same semantics |
| `tool_output` | *(no equivalent)* | CopilotAgent streams tool output incrementally |
| `tool_end` | `tool_end` | StreamEvent adds `result_structured` field |
| `intent` | `intent` | Same semantics |
| `reasoning` | `reasoning` | CopilotEvent gets from CLI; StreamEvent uses buffer classifier |
| `reasoning_delta` | *(no equivalent)* | StreamEvent uses `stream_reasoning_immediately=True` instead |
| `usage` | `usage` | Same fields; StreamEvent requires `detailed` visibility |
| `turn_start` | `turn_start` | Same semantics |
| `turn_end` | `turn_end` | Same semantics |
| `todo` | `todo` | Same semantics |
| `done` | `done` | StreamEvent adds `tools_called` count |
| `error` | `error` | Same semantics |
| `context` | *(Agent-exclusive)* | CopilotAgent only: cwd, branch info |
| `subagent` | *(Agent-exclusive)* | CopilotAgent only: sub-agent lifecycle |
| `compaction` | *(Agent-exclusive)* | CopilotAgent only: context compaction |
| `task_complete` | *(Agent-exclusive)* | CopilotAgent only: autonomous task completion |

### StreamEvent-Only Features

| Feature | StreamEvent | CopilotEvent |
|---------|------------|--------------|
| `result_structured` | Yes | No |
| `preview` (debug) | Yes | No |
| Visibility levels | Yes (minimal/standard/detailed/debug) | No |
| `thinking` event | Yes | No |
| `stream_reasoning_immediately` | Yes | No |
| `to_dict()` serialization | Yes | No |

### CopilotAgent-Exclusive Events

These event types exist only in `CopilotEvent` and have no `StreamEvent`
equivalent:

- **`context`**: Workspace context (cwd, branch). Emitted at stream start.
- **`subagent`**: Sub-agent spawn/complete lifecycle. For multi-agent orchestration.
- **`compaction`**: Context window compaction. Emitted when history is summarized.
- **`task_complete`**: Autonomous task finished. Emitted when `CopilotAgent.run_task()` completes.

If your UI depends on these, continue using `CopilotAgent.stream()` for
those features.
