# Contributing to ai-infra

Thank you for your interest in contributing to ai-infra! This document provides guidelines for contributing.

## [!] AI Safety Warning

**ai-infra controls AI/LLM systems. Bugs here can cause runaway costs, security breaches, or system crashes.**

Before contributing, please read the quality standards in [.github/copilot-instructions.md](.github/copilot-instructions.md).

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Git

### Development Setup

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/ai-infra.git
cd ai-infra

# Add upstream remote
git remote add upstream https://github.com/nfraxlab/ai-infra.git

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run tests
pytest -q

# Run linting
ruff check

# Run type checking
mypy src
```

## AI Safety Requirements

### Recursion Limits

**All agent loops MUST have recursion limits:**

```python
# [OK] Correct - Explicit limit
agent = create_react_agent(llm, tools, recursion_limit=50)

# [X] WRONG - Infinite loop = infinite cost
agent = create_react_agent(llm, tools)
```

### Tool Result Truncation

**Always truncate tool results before sending to LLM:**

```python
# [OK] Correct
result = tool.run()
if len(result) > max_chars:
    result = result[:max_chars] + "\n[TRUNCATED]"

# [X] WRONG - Could blow context window
result = tool.run()  # Could be 100MB
messages.append({"role": "tool", "content": result})
```

### No Code Execution

**Never use eval() or pickle.load() on untrusted data:**

```python
# [X] WRONG - Arbitrary code execution
new_args = eval(user_input)

# [OK] Correct - Safe parsing
import ast
new_args = ast.literal_eval(user_input)
```

### Prompt Injection Protection

**Sanitize external content:**

```python
# [OK] Correct
tool_desc = sanitize_description(mcp_server.get_tool_description())

# [X] WRONG - Could contain "IGNORE PREVIOUS INSTRUCTIONS"
system_prompt += mcp_server.get_tool_description()
```

## Development Workflow

### Quick Start (Recommended)

Use `make pr` for the fastest workflow:

```bash
# 1. Make your code changes
# 2. Create a PR with one command:
make pr m="feat: add your feature"
```

### Two-Mode Workflow

**Mode A: Start a new PR** (on default branch OR with `new=1`)
```bash
# On main -> creates new branch + PR, stays on new branch
make pr m="feat: add streaming support"

# On feature branch -> split commits into new PR
make pr m="feat: split this work" new=1
```

**Mode B: Update current PR** (on feature branch)
```bash
# Add more commits to existing PR
make pr m="fix: address review feedback"

# Sync with main before pushing (rebase + force-push)
make pr m="fix: sync and update" sync=1
```

### All Options

| Option | Example | Description |
|--------|---------|-------------|
| `m=` | `m="feat: add X"` | Commit message (required, conventional commits) |
| `sync=1` | `sync=1` | Rebase on base branch before pushing |
| `new=1` | `new=1` | Force create new PR from current HEAD |
| `b=` | `b="my-branch"` | Use explicit branch name |
| `draft=1` | `draft=1` | Create PR as draft |
| `base=` | `base=develop` | Target different base branch |
| `FORCE=1` | `FORCE=1` | Skip conventional commit validation |

### Examples

```bash
# Basic: create PR from main
make pr m="feat: add MCP tool support"

# Add commits to existing PR
make pr m="fix: handle edge case"

# Sync with main, then push
make pr m="refactor: clean up" sync=1

# Create draft PR
make pr m="feat: work in progress" draft=1

# Target a release branch
make pr m="fix: hotfix" base=release-v1

# Split work into new PR from feature branch
make pr m="feat: extract this part" new=1

# Batch commits before PR
make commit m="feat: add base class"
make commit m="feat: add implementation"
make pr m="feat: complete streaming support"
```

### Manual Workflow

If you prefer manual git commands:

```bash
# 1. Sync your fork with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create a branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# - Add recursion limits to all loops
# - Truncate tool results
# - Add timeouts to external calls
# - Test streaming cancellation

# 4. Run quality checks
ruff format
ruff check
mypy src
pytest -q

# 5. Commit and push to your fork
git add -A
git commit -m "feat: your feature"
git push origin feature/your-feature-name

# 6. Open a PR from your fork to nfraxlab/ai-infra on GitHub
```

### Batching Multiple Commits

For related changes, batch commits before creating a PR:

```bash
make commit m="feat: add base class"
make commit m="feat: add implementation"
make pr m="feat: complete feature"
```

## Code Standards

### Type Hints

All functions must have complete type hints:

```python
async def chat(
    messages: list[Message],
    model: str = "gpt-4",
    max_tokens: int = 1000,
) -> ChatResponse:
    ...
```

### Testing

Test LLM integrations with mocks:

```python
@pytest.fixture
def mock_llm():
    return MockLLM(responses=["Test response"])

def test_agent_respects_limit(mock_llm):
    agent = create_agent(mock_llm, recursion_limit=5)
    # Verify agent stops at limit
```

## Project Structure

```
ai-infra/
├── src/ai_infra/      # Main package
│   ├── llm/           # LLM providers
│   ├── graph/         # LangGraph wrapper
│   ├── mcp/           # MCP client/server
│   └── cli/           # CLI tools
├── tests/
├── docs/
└── examples/
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format. This enables automated CHANGELOG generation.

**Format:** `type(scope): description`

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding or updating tests
- `ci:` CI/CD changes
- `chore:` Maintenance tasks

**Examples:**
```
feat: add streaming support for agents
fix: handle timeout in MCP client
docs: update getting-started guide
refactor: extract callback normalization to shared utility
test: add unit tests for memory module
```

**Bad examples (will be grouped as "Other Changes"):**
```
Refactor code for improved readability  <- Missing type prefix!
updating docs                           <- Missing type prefix!
bug fix                                 <- Missing type prefix!
```

### PR Title Enforcement

A GitHub Action automatically ensures your PR title reflects the highest-priority commit type:

1. Scans all commits in the PR for conventional commit prefixes
2. Auto-updates the PR title if needed (e.g., `docs:` -> `feat:` if there's a `feat:` commit)
3. Passes with a warning after update

**Priority order:** `feat!` > `feat` > `fix` > `perf` > `refactor` > `docs` > `chore` > `test` > `ci` > `build`

This ensures squash-merge commits trigger the correct semantic-release.

## Deprecation Guidelines

When removing or changing public APIs, follow our [Deprecation Policy](DEPRECATION.md).

### When to Deprecate vs Remove

- **Deprecate first** if the feature has any external users
- **Immediate removal** only for security vulnerabilities (see DEPRECATION.md)
- **Never remove** without at least 2 minor versions of deprecation warnings

### How to Add Deprecation Warnings

Use the `@deprecated` decorator:

```python
from ai_infra.utils.deprecation import deprecated

@deprecated(
    version="1.2.0",
    reason="Use new_function() instead",
    removal_version="1.4.0"
)
def old_function():
    ...
```

For deprecated parameters:

```python
from ai_infra.utils.deprecation import deprecated_parameter

def my_function(new_param: str, old_param: str | None = None):
    if old_param is not None:
        deprecated_parameter(
            name="old_param",
            version="1.2.0",
            reason="Use new_param instead"
        )
        new_param = old_param
    ...
```

### Documentation Requirements

When deprecating a feature, you must:

1. Add `@deprecated` decorator or call `deprecated_parameter()`
2. Update the docstring with deprecation notice
3. Add entry to "Deprecated Features Registry" in DEPRECATION.md
4. Add entry to CHANGELOG.md under "Deprecated" section

### Migration Guide Requirements

For significant deprecations, create a migration guide:

1. Create `docs/migrations/v{version}.md`
2. Explain what changed and why
3. Provide before/after code examples
4. Link from the deprecation warning message

## Required Checks Before PR

- [ ] No `eval()` on any input
- [ ] Recursion limits on all agent loops
- [ ] Tool results truncated
- [ ] Timeouts on external calls
- [ ] `ruff check` passes
- [ ] `mypy src` passes
- [ ] `pytest` passes
- [ ] Deprecations follow the deprecation policy

Thank you for contributing!
