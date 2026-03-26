# Contributing to Unified Agent Gateway

Thank you for taking the time to contribute! This document covers everything you need to go from idea to merged PR.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Linting and Formatting](#linting-and-formatting)
- [Project Structure](#project-structure)
- [Adding a New Provider](#adding-a-new-provider)
- [Adding a New Tool Source](#adding-a-new-tool-source)
- [Adding a New Context Source](#adding-a-new-context-source)
- [Pull Request Process](#pull-request-process)
- [Commit Message Style](#commit-message-style)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

---

## Ways to Contribute

- **Bug reports** — use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Feature requests** — use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- **New provider adapters** — Cohere, Bedrock, Azure OpenAI, …
- **New tool sources** — gRPC, GraphQL, database queries, …
- **New context sources** — vector stores, custom KV, document stores, …
- **Documentation** — improve README, API spec, architecture doc
- **Tests** — increase coverage, add integration scenario tests
- **Performance** — profiling, connection pooling, caching

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_GITHUB_USERNAME/unified-agent-gateway.git
cd unified-agent-gateway

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev deps
pip install -e ".[dev]"

# 4. Set up pre-commit hooks (auto-formats on every commit)
pre-commit install

# 5. Copy env template
cp .env.example .env
# Fill in at least one API key if you want integration tests

# 6. Verify everything passes
make test
```

Python **3.10+** is required. We test on 3.10, 3.11, and 3.12.

---

## Running Tests

```bash
# Full suite (no live API keys required)
pytest

# Skip integration tests (default on CI)
pytest -m "not integration"

# Single file
pytest tests/test_api.py -v

# With coverage
pytest --cov=. --cov-report=term-missing
```

All 100+ unit tests run offline — they mock the provider SDKs and MCP transport layer.

Integration tests (marked `@pytest.mark.integration`) hit live APIs and require real keys in `.env`. They are excluded from CI by default.

---

## Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Pre-commit hooks run
automatically on every commit if you ran `pre-commit install` during setup.

```bash
# Check (same checks CI runs)
make lint

# Auto-fix and format
make format
```

Or manually:

```bash
ruff check .
ruff check --fix .
ruff format .
```

CI will fail on ruff errors. Run `make lint` before opening a PR.

---

## Project Structure

```
api/          FastAPI endpoints — do not put business logic here
config/       Settings models (Pydantic) — all env var definitions live here
context/      Context registry + adapters (ContextForge, RAG, static, KV)
core/         Normalized data types, agent loop, durable execution primitives
docs/         Architecture and API spec markdown
postman/      Postman collection
providers/    LLM adapters (one file per provider)
runtime/      Router, bootstrap, SSE helpers
tests/        pytest suite
tools/        Tool registry, MCP loader, inline MCP HTTP client
```

Keep each layer's responsibilities clean:
- **Providers** normalize ↔ provider-native — they must not know about MCP, contexts, or routing.
- **AgentLoop** owns the tool-calling lifecycle — it must not know about HTTP or SSE.
- **API** wires everything together — it should stay thin.

---

## Adding a New Provider

1. Create `providers/myprovider.py`:

```python
from providers.base import BaseProvider
from core.types import NormalizedMessage, NormalizedResponse, StreamEvent, ToolDefinition
from typing import AsyncIterator, List, Optional

class MyProvider(BaseProvider):
    name = "myprovider"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.model = model
        # initialise SDK client here

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> NormalizedResponse:
        # translate → call → translate back
        ...

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        # yield StreamEvent(type="chunk", delta="..."), ..., StreamEvent(type="done")
        ...
```

2. Register in `runtime/router.py`:

```python
from providers.myprovider import MyProvider
PROVIDERS["myprovider"] = MyProvider
```

3. Add defaults in `resolve_provider_config` if it needs a dedicated key/model env var.

4. Add tests in `tests/test_myprovider.py` — mock the SDK, test message mapping and stream accumulation.

5. Update [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`README.md`](README.md).

---

## Adding a New Tool Source

1. Add a variant to `ToolSource` in `tools/registry.py` (optional — `ToolSource.HTTP` or `ToolSource.PYTHON` covers most cases).

2. Register via `ToolRegistry.register()`:

```python
registry.register(
    name="my_tool",
    description="…",
    json_schema={…},
    source=ToolSource.PYTHON,   # or your new variant
    handler=my_async_function,
    metadata={"custom": "data"},
)
```

3. Wire it in `runtime/bootstrap.py` if it should be globally available.

---

## Adding a New Context Source

1. Optionally add a variant to `ContextSource` in `context/registry.py`.

2. Create an async fetch function and register it:

```python
async def my_fetcher(**kwargs) -> str:
    # fetch and return text
    ...

registry.register(RegisteredContext(
    name="my_source",
    source=ContextSource.RAG,   # or your new variant
    fetch=my_fetcher,
    required=False,
    max_chars=4000,
))
```

3. Wire it in `runtime/bootstrap.py`.

---

## Pull Request Process

1. **Open an issue first** for any significant change (new provider, architectural change, breaking API change).
2. Fork the repo and create a branch: `git checkout -b feat/my-feature`.
3. Make your changes, add tests, update docs.
4. Run `make lint && make test` — both must pass.
5. Push and open a PR against `main`.
6. Fill in the PR template — describe what, why, and how to test.
7. A maintainer will review; address feedback with new commits (no force-push until approved).
8. Once approved and CI is green, the PR will be squash-merged.

### PR checklist

- [ ] `pytest` passes with no new failures
- [ ] `ruff check .` clean
- [ ] New public functions/classes have docstrings
- [ ] New behaviour is tested (unit test at minimum)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docs updated if API shape changed

---

## Commit Message Style

```
type(scope): short imperative description

Optional longer explanation (wrap at 72 chars).

Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`

Examples:
- `feat(providers): add Mistral adapter`
- `fix(agent-loop): handle empty tool results gracefully`
- `docs(api-spec): document runtime.mcp_namespaces field`
- `test(api): add BYOK 403 scenario`

---

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/unified-agent-gateway/discussions) or reach out via the issue tracker.
