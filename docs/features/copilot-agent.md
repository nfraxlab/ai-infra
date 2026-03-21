# Copilot Agent

> Computer automation agent backed by the GitHub Copilot CLI runtime.

## Quick Start

```python
from ai_infra import CopilotAgent

agent = CopilotAgent(cwd="/path/to/project")
result = await agent.run("Add type hints to all public functions in src/")
print(result.content)
```

---

## Overview

`CopilotAgent` delegates autonomous task execution to the GitHub Copilot CLI.
Unlike `Agent` — which calls LLMs through ai-infra's provider abstractions —
`CopilotAgent` drives the full Copilot CLI runtime.

The LLM and tool selection are managed internally by the CLI. Developers
provide a plain-language task; the agent determines the steps.

Built-in CLI capabilities:

- Shell and bash execution
- File system operations (read, write, edit, glob, grep)
- Git operations (status, diff, commit, branch management)
- Web fetch and search
- Multi-step planning with automatic context compaction
- Sub-agent orchestration via `custom_agents`

### When to Use CopilotAgent vs Agent

| Situation | Recommended |
|-----------|-------------|
| File edits, git ops, shell automation | `CopilotAgent` |
| Custom LLM provider or model control | `Agent` |
| Structured output or tool schemas | `Agent` |
| Open-ended project tasks (code, docs, tests) | `CopilotAgent` |
| Embedding in an existing ai-infra graph | `Agent` |

---

## Installation

`CopilotAgent` requires the optional `copilot` extra:

```bash
pip install 'ai-infra[copilot]'
```

The GitHub Copilot CLI must also be installed and authenticated:

```bash
# macOS / Linux via npm
npm install -g @github/copilot-cli

# Authenticate
gh auth login
gh copilot --version
```

A GitHub Copilot subscription is required unless you supply BYOK credentials
via the `provider` parameter.

---

## Basic Usage

### Single task

```python
from ai_infra import CopilotAgent

result = await CopilotAgent(cwd="/my/project").run(
    "Fix all mypy type errors in src/"
)
print(result.content)
print(f"Tools called: {result.tools_called}")
print(f"Session ID:   {result.session_id}")
```

### Multi-turn sessions

Resume a conversation by passing the `session_id` from a previous result:

```python
agent = CopilotAgent(cwd="/my/project")

r1 = await agent.run("Explore the auth module and summarise it")
r2 = await agent.run("Now add unit tests for the JWT logic", session_id=r1.session_id)
r3 = await agent.run("Run the tests and fix any failures", session_id=r1.session_id)
```

### Async context manager

Use the context manager when you need to sequence multiple tasks and want
explicit process teardown guarantees:

```python
async with CopilotAgent(cwd="/my/project") as agent:
    await agent.run("Generate a CHANGELOG from git history")
    await agent.run("Update the version in pyproject.toml")
```

---

## Streaming

`CopilotAgent.stream()` yields `CopilotEvent` instances in real time.

```python
async for event in agent.stream("Run the test suite and fix failures"):
    if event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "tool_start":
        print(f"\n→ {event.tool}({event.arguments})")
    elif event.type == "tool_output":
        print(event.content, end="", flush=True)
    elif event.type == "tool_end":
        print(f"  [{event.latency_ms:.0f}ms]")
    elif event.type == "done":
        print("\nDone.")
```

### Event reference

| `event.type` | Fired when | Key fields |
|---|---|---|
| `token` | Text chunk streams from the model | `content` |
| `tool_start` | A tool begins executing | `tool`, `arguments` |
| `tool_output` | Partial stdout from a long-running tool | `tool`, `content` |
| `tool_end` | Tool execution completes | `tool`, `result`, `latency_ms` |
| `intent` | Agent describes what it is about to do | `content` |
| `context` | Working directory or branch changes | `cwd`, `branch` |
| `reasoning` | Full chain-of-thought block (supported models) | `content`, `reasoning_id` |
| `reasoning_delta` | Streaming reasoning chunk | `content`, `reasoning_id` |
| `usage` | Per-LLM-call token accounting | `input_tokens`, `output_tokens`, `cost` |
| `turn_start` | Agent begins a reasoning turn | `turn_id` |
| `turn_end` | Agent finishes a reasoning turn | `turn_id` |
| `subagent` | Sub-agent lifecycle (selected / started / completed) | `subagent_name`, `subagent_phase` |
| `todo` | Agent updated its task checklist | `todo_items` |
| `compaction` | Context window compressed (infinite sessions) | `compaction_phase`, `tokens_removed` |
| `task_complete` | Agent considers the task done | `content` (summary) |
| `done` | Session turn fully complete | `content` (session ID) |
| `error` | An error occurred | `error` |

---

## Custom Tools

Register Python functions as tools the agent can call during a task.
Type hints generate the JSON schema automatically.

```python
from ai_infra import CopilotAgent, copilot_tool

@copilot_tool
async def get_issue(id: str) -> str:
    "Fetch an issue from Linear."
    return await linear_client.get(id)

@copilot_tool(description="Post a comment on a pull request", skip_permission=True)
async def post_comment(pr_number: int, body: str) -> str:
    return await github_client.post_comment(pr_number, body)

agent = CopilotAgent(tools=[get_issue, post_comment])
result = await agent.run("Fix the bug described in issue LIN-84 and open a PR")
```

### Overriding built-in tools

Replace a built-in Copilot CLI tool with your own implementation:

```python
@copilot_tool(
    name="edit_file",
    description="Project-aware file editor with lint validation",
    overrides_built_in_tool=True,
)
async def edit_file(path: str, content: str) -> str:
    validate_lint(path, content)
    Path(path).write_text(content)
    return f"Written {path}"

agent = CopilotAgent(tools=[edit_file])
```

---

## Scratchpad Tools

Give the agent explicit tools for recording its thinking, making the
reasoning process visible to the UI.

```python
from ai_infra import CopilotAgent
from ai_infra.llm.agents.copilot import create_scratchpad_tools, SCRATCHPAD_TOOL_NAMES

scratchpad = create_scratchpad_tools()
agent = CopilotAgent(tools=[*scratchpad, *my_other_tools])
```

This registers four tools:

| Tool | Purpose |
|------|---------|
| `scratchpad_think` | Record free-form reasoning and analysis |
| `scratchpad_plan` | Lay out a numbered plan or next steps |
| `scratchpad_reflect` | Look back at progress, note corrections |
| `scratchpad_read` | Read back all recorded scratchpad entries |

All four share a single in-memory buffer created per
`create_scratchpad_tools()` call, so concurrent sessions get independent
scratchpads.

### Routing scratchpad events in the UI

When streaming, scratchpad tool calls appear as `tool_start` / `tool_end`
events. Use `SCRATCHPAD_TOOL_NAMES` to detect them and route their
content to a collapsible panel instead of the main tool timeline:

```python
from ai_infra.llm.agents.copilot import SCRATCHPAD_TOOL_NAMES

async for event in agent.stream("Fix the failing tests"):
    if event.type == "tool_start" and event.tool in SCRATCHPAD_TOOL_NAMES:
        args = event.arguments or {}
        thinking = args.get("thought") or args.get("plan") or args.get("reflection") or ""
        show_in_scratchpad(thinking)
    elif event.type == "tool_start":
        show_tool_running(event.tool, event.arguments)
    elif event.type == "token":
        print(event.content, end="", flush=True)
```

---

## Permissions

Control which built-in tools the agent is allowed to invoke.

```python
from ai_infra import CopilotAgent, PermissionMode

# Default — all tools approved
agent = CopilotAgent(permissions=PermissionMode.AUTO_APPROVE)

# Analysis only — no writes, no shell
agent = CopilotAgent(permissions=PermissionMode.READ_ONLY)
result = await agent.run("Audit for OWASP Top 10 vulnerabilities")

# Pause before destructive actions (bash, write_file, git_commit, etc.)
agent = CopilotAgent(permissions=PermissionMode.INTERACTIVE)

# Block everything — dry run / replay
agent = CopilotAgent(permissions=PermissionMode.DENY_ALL)
```

### Custom permission handler

Provide a callable `(tool_name: str, arguments: dict) -> bool` for
fine-grained logic:

```python
def my_policy(tool_name: str, arguments: dict) -> bool:
    if tool_name == "bash":
        cmd = arguments.get("command", "")
        return not cmd.startswith("rm")
    return True

agent = CopilotAgent(permissions=my_policy)
```

---

## BYOK (Bring Your Own Key)

Use a non-GitHub LLM without a Copilot subscription:

```python
import os
from ai_infra import CopilotAgent

agent = CopilotAgent(
    model="gpt-4.1",
    provider={
        "type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ["OPENAI_API_KEY"],
    },
)
```

Supported provider types: `"openai"`, `"azure"`, `"anthropic"`.

---

## Skills

Inject prompt modules from SKILL.md files to shape the agent's behaviour:

```python
agent = CopilotAgent(
    skill_dirs=["/path/to/agent/skills"],
    disabled_skills=["slow-research"],
)
```

Each directory may contain multiple skill subdirectories, each with a
`SKILL.md` file. Skills are loaded as context at session creation.

---

## MCP Servers

Attach Model Context Protocol servers to extend the agent's tool set:

```python
agent = CopilotAgent(
    mcp_servers={
        "filesystem": {
            "type": "local",
            "command": "npx",
            "args": ["-y", "@mcp/server-filesystem", "/workspace"],
            "tools": ["*"],
        }
    }
)
```

---

## Sub-agents

Define specialised sub-agents for automatic task delegation:

```python
agent = CopilotAgent(
    custom_agents=[
        {
            "name": "test-writer",
            "prompt": "You are an expert at writing pytest tests. Only write tests, never edit source.",
            "tools": ["read_file", "write_file", "glob"],
        },
        {
            "name": "security-reviewer",
            "prompt": "Review code for security vulnerabilities. Never modify files.",
            "tools": ["read_file", "grep", "glob"],
        },
    ],
    # Pre-select an agent for this session
    initial_agent="test-writer",
)
```

Sub-agent lifecycle events are exposed via `CopilotEvent` with
`event.type == "subagent"`.

---

## Callbacks

Bridge Copilot events to the standard ai-infra `Callbacks` interface:

```python
from ai_infra import CopilotAgent
from ai_infra.callbacks import LoggingCallbacks, MetricsCallbacks

agent = CopilotAgent(
    callbacks=LoggingCallbacks(),  # or MetricsCallbacks(), or a custom Callbacks subclass
)
```

Copilot tool start/end and token events are automatically forwarded to
`on_tool_start`, `on_tool_end`, and `on_llm_token` callback handlers.

---

## Session Management

```python
agent = CopilotAgent()

# List all persisted sessions on this machine
sessions = await agent.list_sessions()
for s in sessions:
    print(s["sessionId"], s.get("createdAt"))

# Delete a session and all its on-disk data (irreversible)
await agent.delete_session("user-alice-pr-review-42")
```

---

## Model Discovery

```python
models = await agent.list_models()
# ["gpt-4.1", "claude-sonnet-4-5", "o4-mini", ...]
```

---

## Reasoning Effort

For models that support chain-of-thought depth control:

```python
agent = CopilotAgent(
    model="claude-sonnet-4-5",
    reasoning_effort="high",  # "low" | "medium" | "high" | "xhigh"
)
```

---

## Telemetry

Enable OpenTelemetry tracing for the CLI subprocess:

```python
agent = CopilotAgent(
    telemetry={"otlp_endpoint": "http://localhost:4318"},
)
```

---

## Connecting to an External Server

If you manage the CLI process separately (e.g. in a sidecar container):

```python
agent = CopilotAgent(external_server="localhost:3000")
```

When `external_server` is set, `cli_path`, `env`, and `github_token` are
ignored and no subprocess is started.

---

## API Reference

### `CopilotAgent`

```python
CopilotAgent(
    *,
    model: str | None = None,
    cwd: str | None = None,
    github_token: str | None = None,
    provider: dict | None = None,
    tools: list | None = None,
    permissions: PermissionMode | Callable = PermissionMode.AUTO_APPROVE,
    skill_dirs: list[str] | None = None,
    mcp_servers: dict | None = None,
    custom_agents: list[dict] | None = None,
    system_message: str | None = None,
    streaming: bool = True,
    infinite_context: bool = True,
    callbacks: Callbacks | None = None,
    reasoning_effort: str | None = None,
    on_user_input: Callable | None = None,
    hooks: dict[str, Callable] | None = None,
    telemetry: dict | None = None,
    external_server: str | None = None,
    cli_path: str | None = None,
    env: dict[str, str] | None = None,
    disabled_skills: list[str] | None = None,
    initial_agent: str | None = None,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(prompt, *, session_id, attachments)` | `CopilotResult` | Run a task, return final result |
| `stream(prompt, *, session_id, attachments)` | `AsyncIterator[CopilotEvent]` | Run a task, yield live events |
| `stop()` | `None` | Stop the CLI subprocess |
| `list_models()` | `list[str]` | List available models |
| `list_sessions()` | `list[dict]` | List persisted sessions |
| `delete_session(session_id)` | `None` | Delete a persisted session |

### `CopilotResult`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Assistant's final text response |
| `session_id` | `str` | Session identifier for resumption |
| `tools_called` | `int` | Number of tool invocations |
| `duration_ms` | `float` | Total wall-clock time in milliseconds |

### `PermissionMode`

| Value | Behaviour |
|-------|-----------|
| `AUTO_APPROVE` | All tools approved (default) |
| `READ_ONLY` | Read-only tools only; no writes or shell |
| `INTERACTIVE` | Destructive tools require explicit approval |
| `DENY_ALL` | No tools allowed (dry-run mode) |

### `CopilotEvent`

All fields default to empty/zero when not applicable to a given event type.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Event type (see [Event reference](#event-reference)) |
| `content` | `str` | Text content (tokens, reasoning, tool output) |
| `tool` | `str` | Tool name |
| `arguments` | `dict` | Tool input arguments |
| `result` | `str` | Tool execution result |
| `cwd` | `str` | Working directory (context events) |
| `branch` | `str` | Git branch (context events) |
| `error` | `str` | Error message |
| `latency_ms` | `float` | Tool execution latency |
| `timestamp` | `float` | Event timestamp (auto-set) |
| `reasoning_id` | `str` | Groups related reasoning chunks |
| `subagent_name` | `str` | Sub-agent identifier |
| `subagent_phase` | `str` | Sub-agent lifecycle phase |
| `tokens_removed` | `int` | Tokens removed during compaction |
| `compaction_phase` | `str` | `"start"` or `"complete"` |
| `input_tokens` | `int` | LLM input tokens (usage events) |
| `output_tokens` | `int` | LLM output tokens (usage events) |
| `cost` | `float` | Estimated cost (usage events) |
| `turn_id` | `str` | Reasoning turn identifier |
| `todo_items` | `list[dict]` | Task checklist items (todo events) |
