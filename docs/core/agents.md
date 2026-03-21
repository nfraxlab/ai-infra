# Agent Class

> Tool-calling agents with human-in-the-loop support and provider fallbacks.

## Quick Start

```python
from ai_infra import Agent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

agent = Agent(tools=[get_weather])
result = agent.run("What's the weather in San Francisco?")
print(result)  # "The weather in San Francisco is 72°F and sunny."
```

---

## Creating Agents

### With Function Tools

```python
from ai_infra import Agent

def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

agent = Agent(tools=[search, calculate])
result = agent.run("What is 15 * 23?")
```

### With Pydantic Schema Tools

```python
from pydantic import BaseModel, Field
from ai_infra import Agent

class SendEmail(BaseModel):
    """Send an email to a recipient."""
    to: str = Field(description="Email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")

def send_email_handler(to: str, subject: str, body: str) -> str:
    # Actually send email...
    return f"Email sent to {to}"

agent = Agent(tools=[SendEmail])
# Register handler
agent.register_tool_handler("SendEmail", send_email_handler)
```

---

## Running Agents

### Synchronous

```python
agent = Agent(tools=[my_tool])

# Simple run
result = agent.run("Do something")

# With messages
result = agent.run_agent(
    messages=[{"role": "user", "content": "Do something"}],
    provider="openai",
    model_name="gpt-4o",
)
```

### Asynchronous

```python
import asyncio

async def main():
    agent = Agent(tools=[my_tool])
    result = await agent.arun("Do something async")
    print(result)

asyncio.run(main())
```

---

## Streaming Responses

Use `agent.astream()` to stream typed `StreamEvent` objects.

### Basic Usage

```python
from ai_infra import Agent

agent = Agent(tools=[search_docs])
async for event in agent.astream("What is the refund policy?"):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Visibility Levels

- `minimal`: tokens only
- `standard`: tokens + tool names + reasoning (default)
- `detailed`: include tool arguments and full tool results
- `debug`: include tool result previews

### Reasoning Events

By default, text the model emits before or between tool calls is
classified as `reasoning` events instead of `token` events. This
lets you show the agent's "thinking" separately from its answer:

```python
async for event in agent.astream("Analyse the auth module"):
    if event.type == "reasoning":
        show_thinking(event.content)  # Pre-tool narration
    elif event.type == "token":
        print(event.content, end="", flush=True)  # Final answer
```

Disable with `stream_config=StreamConfig(include_reasoning=False)`.

```python
async for event in agent.astream("Search docs", visibility="detailed"):
    if event.type == "tool_start":
        print(f"Calling {event.tool} with {event.arguments}")
```

### With LangGraph Config

Pass LangGraph config for persistence and tracing:

```python
config = {
    "configurable": {"thread_id": "user-123"},
    "tags": ["production"],
}

async for event in agent.astream("Continue", config=config):
    ...
```

### Framework Examples

FastAPI SSE:

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    async def generate():
        async for event in agent.astream(req.message, visibility=req.visibility):
            yield f"data: {event.to_dict()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

WebSocket:

```python
@router.websocket("/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    async for event in agent.astream("Hello"):
        await ws.send_json(event.to_dict())
```

---

## Human-in-the-Loop (HITL)

Require human approval before tool execution:

```python
from ai_infra import Agent

def delete_file(path: str) -> str:
    """Delete a file from disk."""
    # Dangerous operation!
    os.remove(path)
    return f"Deleted {path}"

agent = Agent(tools=[delete_file])

# Wrap dangerous tools with HITL
agent.enable_hitl(
    tools=["delete_file"],
    approval_callback=lambda tool, args: input(f"Allow {tool}({args})? [y/n]: ") == "y"
)

result = agent.run("Delete temp.txt")
# User will be prompted before deletion
```

### Async HITL for Web Apps

```python
from ai_infra import Agent

# Store pending approvals
pending_approvals = {}

async def request_approval(tool_name: str, args: dict, request_id: str):
    """Called when approval needed - notify frontend."""
    pending_approvals[request_id] = {"tool": tool_name, "args": args}
    # WebSocket/SSE to frontend...

async def wait_for_approval(request_id: str) -> bool:
    """Wait for user response from frontend."""
    while request_id in pending_approvals:
        await asyncio.sleep(0.1)
    return pending_approvals.pop(f"{request_id}_result", False)

agent = Agent(tools=[dangerous_tool])
agent.enable_hitl_async(
    tools=["dangerous_tool"],
    request_approval=request_approval,
    wait_for_approval=wait_for_approval,
)
```

---

## Provider Fallbacks

Automatically try backup providers on failure:

```python
agent = Agent(tools=[my_tool])

result = agent.run_with_fallbacks(
    messages=[{"role": "user", "content": "Hello"}],
    candidates=[
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-20250514"),
        ("google_genai", "gemini-2.0-flash"),
    ]
)
```

### Async Fallbacks

```python
result = await agent.arun_with_fallbacks(
    messages=[{"role": "user", "content": "Hello"}],
    candidates=[
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-20250514"),
    ]
)
```

---

## Deep Mode

Enable autonomous multi-step operations with file access:

```python
agent = Agent(deep=True)

result = agent.run("""
    1. Read the file config.yaml
    2. Update the version to 2.0.0
    3. Save the changes
""")
```

Deep mode automatically provides:
- File read/write tools
- Code execution (sandboxed)
- Multi-step planning

See [Deep Agent](../features/deep-agent.md) for details.

---

## Subagents

Delegate tasks to specialized agents:

```python
# Create specialized agents
researcher = Agent(tools=[search_tool], name="researcher")
writer = Agent(tools=[write_tool], name="writer")

# Main agent can delegate
main_agent = Agent(
    tools=[researcher, writer],  # Agents become tools!
)

result = main_agent.run("""
    Research the latest AI news, then write a summary.
""")
```

---

## Structured Output

Get typed responses from agents:

```python
from pydantic import BaseModel
from ai_infra import Agent

class TaskResult(BaseModel):
    success: bool
    message: str
    data: dict

agent = Agent(tools=[my_tool])
result = agent.run(
    "Process the data",
    response_model=TaskResult
)
print(result.success)  # True
```

---

## Configuration

```python
agent = Agent(
    tools=[my_tool],

    # Provider settings
    provider="openai",
    model="gpt-4o",

    # Agent behavior
    max_iterations=10,
    verbose=True,

    # Model settings
    temperature=0.7,
    max_tokens=4096,
)
```

---

## MCP Tools

Use tools from MCP servers:

```python
from ai_infra import Agent, MCPClient

# Connect to MCP server
async with MCPClient("http://localhost:8080") as client:
    tools = await client.list_tools()

    agent = Agent(tools=tools)
    result = await agent.arun("Use the MCP tools")
```

See [MCP Client](../mcp/client.md) for details.

---

## Error Handling

```python
from ai_infra import Agent
from ai_infra.errors import (
    AIInfraError,
    ToolExecutionError,
    ToolTimeoutError,
)

agent = Agent(tools=[my_tool])

try:
    result = agent.run("Do something")
except ToolExecutionError as e:
    print(f"Tool {e.tool_name} failed: {e.message}")
except ToolTimeoutError as e:
    print(f"Tool {e.tool_name} timed out after {e.timeout}s")
except AIInfraError as e:
    print(f"Agent error: {e}")
```

## See Also

- [LLM](llm.md) - Chat without tools
- [Deep Agent](../features/deep-agent.md) - Autonomous agents
- [Personas](../features/personas.md) - Config-driven agent behavior
- [MCP Client](../mcp/client.md) - External tools via MCP
