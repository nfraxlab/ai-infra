# Streaming Guide

> Stream agent responses with typed events for real-time UIs.

## Quick Start

Get streaming working in 3 lines:

```python
from ai_infra import Agent

agent = Agent(tools=[my_tool])

# Basic streaming - print tokens as they arrive
async for event in agent.astream("What is the weather in NYC?"):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Complete FastAPI SSE Example

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ai_infra import Agent
import json

app = FastAPI()
agent = Agent(tools=[search_docs])

@app.post("/chat")
async def chat(message: str):
    async def generate():
        async for event in agent.astream(message, visibility="detailed"):
            yield f"data: {json.dumps(event.to_dict())}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### WebSocket Example

```python
from fastapi import WebSocket

@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    message = await websocket.receive_text()

    async for event in agent.astream(message):
        await websocket.send_json(event.to_dict())

    await websocket.close()
```

---

## StreamEvent Reference

`StreamEvent` fields (None when not applicable):
- `type`: thinking | reasoning | token | tool_start | tool_end | done | error
- `content`: token text or reasoning narration
- `tool` / `tool_id`: tool name and call ID
- `arguments`: tool args (visibility detailed+)
- `result`: **FULL tool result** (visibility detailed+) - for parsing/processing
- `result_structured`: True if result is a structured dict (not text)
- `preview`: truncated tool result (visibility=debug) - for UI display
- `latency_ms`: tool latency
- `model`: model name (thinking)
- `tools_called`: total tools (done)
- `error`: error message
- `timestamp`: event timestamp

### Reasoning Events

When `include_reasoning=True` (the default), text the model emits before
its first tool call or between consecutive tool calls is classified as
**reasoning** rather than tokens. This gives consumers a "thinking out
loud" stream to show in a collapsed panel — reasoning is not part of
the final answer.

```python
async for event in agent.astream("Analyse the auth module"):
    if event.type == "reasoning":
        # Pre-tool or inter-tool narration
        show_in_scratchpad(event.content)
    elif event.type == "token":
        # Final answer text
        print(event.content, end="", flush=True)
```

If the model emits more than `reasoning_token_limit` characters (default
300) without calling a tool, the buffer is reclassified as answer tokens
so direct-answer responses stream normally.

Serialize with `event.to_dict()`.

### Agent vs CopilotAgent Reasoning

The reasoning classification described above applies to `Agent.astream()`.
`CopilotAgent.stream()` emits `CopilotEvent` objects (not `StreamEvent`)
and gets reasoning events directly from the CLI runtime rather than from a
buffer-based classifier. See the
[Copilot Agent guide](features/copilot-agent.md) for `CopilotEvent` types.

### Structured Results

When a tool returns a structured dict (e.g., `create_retriever_tool(structured=True)`), the event indicates this:

```python
async for event in agent.astream("Search for auth"):
    if event.type == "tool_end" and event.result_structured:
        # event.result is a dict, not a string
        results = event.result["results"]
        query = event.result["query"]
        for r in results:
            print(f"{r['source']}: {r['score']}")
    elif event.type == "tool_end":
        # event.result is a string
        print(event.result)
```

**`to_dict()` output differs based on `result_structured`:**

```python
# Text result (result_structured=False)
{"type": "tool_end", "result": "text output...", ...}

# Structured result (result_structured=True)
{"type": "tool_end", "structured_result": {"results": [...], "query": "..."}, ...}
```

## Visibility Levels

Control what data is included in streaming events:

### `minimal`
- Response tokens only
- No thinking, tool events, or metadata
- Cleanest output for simple UIs

### `standard` (default)
- Response tokens
- Tool names and timing
- Thinking indicator
- **No** tool arguments or results

### `detailed`
- Everything in `standard`
- **Tool arguments** (inputs)
- **FULL tool results** (outputs) <- NEW!
- For applications that need to parse tool outputs
- Example: Create clickable links from search results

### `debug`
- Everything in `detailed`
- **Truncated preview** (500 chars) for quick UI display
- For development/debugging

## Tool Result Fields

Two fields for different use cases:

| Field | Visibility | Purpose | Example |
|-------|-----------|---------|---------|
| `result` | `detailed+` | **Full output** for parsing | Multi-result search output |
| `preview` | `debug` only | **Truncated** for UI display | First 500 chars |

## StreamConfig reference

`StreamConfig` controls visibility and tool handling:
- `visibility`: minimal | standard | detailed | debug (default: standard)
- `include_thinking`: emit initial thinking event
- `include_reasoning`: classify pre/inter-tool text as reasoning events (default: True)
- `include_tool_events`: emit tool_start/tool_end
- `reasoning_token_limit`: max chars buffered as reasoning before reclassifying as tokens (default: 300)
- `tool_result_preview_length`: max preview length (debug)
- `deduplicate_tool_starts`: avoid duplicate starts per tool call

Pass via `agent.astream(..., stream_config=StreamConfig(visibility="detailed"))`.

### Disabling Reasoning Classification

If you do not need the reasoning/token split, disable it:

```python
from ai_infra.llm.streaming import StreamConfig

config = StreamConfig(include_reasoning=False)
async for event in agent.astream(prompt, stream_config=config):
    # All text arrives as "token" events — no "reasoning" events emitted
    ...
```

## BYOK helper (temporary keys)

Use user-provided API keys for a single request:

```python
from ai_infra import Agent, atemporary_api_key

async with atemporary_api_key("openai", user_key):
    async for event in agent.astream(prompt):
        yield event.to_dict()
```

## MCP tool loader (cached)

Load and cache MCP tools once, with optional force refresh:

```python
from ai_infra import Agent, load_mcp_tools_cached

tools = await load_mcp_tools_cached("http://localhost:8000/mcp")
agent = Agent(tools=tools)
```

## Examples

### Basic streaming

```python
async for event in agent.astream("What is the refund policy?"):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Visibility control

```python
async for event in agent.astream("Search docs", visibility="detailed"):
    if event.type == "tool_start":
        print(event.arguments)
```

### With LangGraph config

```python
config = {
    "configurable": {"thread_id": "user-123"},
    "tags": ["production"],
}
async for event in agent.astream("Continue our conversation", config=config):
    ...
```

### FastAPI SSE endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
agent = Agent(tools=[search_docs])

@app.post("/chat")
async def chat(message: str, provider: str, api_key: str):
    async def generate():
        async with atemporary_api_key(provider, api_key):
            async for event in agent.astream(message):
                yield f"data: {event.to_dict()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### WebSocket endpoint

```python
from fastapi import WebSocket

@router.websocket("/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    async for event in agent.astream("Hello"):
        await ws.send_json(event.to_dict())
```

### Cached MCP tools + BYOK

```python
MCP_URL = "http://localhost:8000/mcp"

async def stream_with_mcp(message: str, api_key: str):
    tools = await load_mcp_tools_cached(MCP_URL)
    agent = Agent(tools=tools)
    async with atemporary_api_key("openai", api_key):
        async for event in agent.astream(message, visibility="debug"):
            yield event
```

## Real-World Use Case: Clickable Tool Results

**Problem**: Chat UI needs to show clickable links to documentation pages from MCP `search_docs` tool results.

**Tool output format**:
```
### Result 1 (svc-infra: auth.md)
[snippet about authentication]
---
### Result 2 (ai-infra: core/llm.md)
[snippet about LLM usage]
```

**Solution**: Use `detailed` visibility to get full results, parse, and create links:

```python
import re
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ai_infra import Agent, atemporary_api_key, load_mcp_tools_cached

app = FastAPI()

def parse_doc_results(result: str) -> list[dict]:
    """Parse search_docs output into structured data."""
    pattern = r"### Result \d+ \((.+?): (.+?)\)"
    matches = re.finditer(pattern, result)
    return [
        {"package": m.group(1), "path": m.group(2)}
        for m in matches
    ]

@app.post("/chat")
async def chat(message: str, provider: str, api_key: str):
    tools = await load_mcp_tools_cached("http://localhost:8000/mcp")
    agent = Agent(tools=tools)

    async def generate():
        async with atemporary_api_key(provider, api_key):
            async for event in agent.astream(message, visibility="detailed"):
                # Regular events pass through
                if event.type != "tool_end":
                    yield f"data: {json.dumps(event.to_dict())}\n\n"
                    continue

                # Parse tool results for search tools
                if "search" in event.tool and event.result:
                    docs = parse_doc_results(event.result)

                    # Emit custom event with structured data
                    yield f"data: {json.dumps({
                        'type': 'tool_results',
                        'tool': event.tool,
                        'docs': docs,  # [{"package": "svc-infra", "path": "auth.md"}, ...]
                        'latency_ms': event.latency_ms,
                    })}\n\n"
                else:
                    # Regular tool_end event
                    yield f"data: {json.dumps(event.to_dict())}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Frontend** (TypeScript/React):
```typescript
// Handle tool results in chat UI
if (event.type === "tool_results") {
    event.docs.forEach((doc: {package: string, path: string}) => {
        const url = `/${doc.package}/${doc.path.replace('.md', '')}`;
        // Create clickable chip/link in UI
        showClickableChip(doc.package, doc.path, url);
    });
}
```

**Key benefits**:
- [OK] Full tool results available at `detailed` visibility
- [OK] Parse once in backend, send structured data to frontend
- [OK] Frontend receives clean, clickable links
- [OK] No string truncation issues (unlike `debug` preview)
