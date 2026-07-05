# Contributing to Open Agent

Thanks for your interest in Open Agent! This document explains how to set up a
development environment and contribute changes back to the project. Open Agent
is an open-source Agentic RAG autonomous work assistant — see the
[README](./README.md) for an overview of features and architecture.

Contributions of all kinds are welcome: bug reports, feature requests, code,
docs, and examples.

## Development Environment Setup

You need **Python 3.10+** and `git`. The project uses
[`hatchling`](https://hatch.pypa.io/) as its build backend.

1. **Clone your fork** (replace `your-username` with your GitHub username):

   ```bash
   git clone https://github.com/your-username/open-agent.git
   cd open-agent
   ```

2. **Install the package in editable mode with dev extras**:

   ```bash
   pip install -e ".[dev]"
   ```

   This installs `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, and `mypy`.
   For the full optional stack (RAG backends, server, MCP, browser, etc.) install
   the `all` extra instead: `pip install -e ".[all]"`.

3. **(Optional) Browser tool** — the headless-browser tool requires Playwright.
   After installing the `browser` extra, download the Chromium runtime:

   ```bash
   pip install -e ".[browser]"
   playwright install chromium
   ```

4. **Verify the install** by running the test suite:

   ```bash
   pytest tests/
   ```

## Code Style

Consistency is enforced with two tools. Run them before opening a pull request.

- **[ruff](https://docs.astral.sh/ruff/)** — formatter and linter. Line length
  is **100** (configured in `pyproject.toml`). Run:

  ```bash
  ruff check src tests
  ruff format src tests   # optional: auto-format
  ```

- **[mypy](http://mypy-lang.org/)** — static type checker, run in strict mode
  against the source tree:

  ```bash
  mypy src
  ```

Additional conventions:

- Every Python module should start with `from __future__ import annotations`
  so that type hints are evaluated lazily (PEP 563).
- Follow the existing naming and layout you see in `src/open_agent/`.
- Public APIs should have docstrings (the codebase uses Google-style
  docstrings with reStructuredText Sphinx directives).
## Testing

- **Unit tests** live in `tests/unit/` and **integration tests** in
  `tests/integration/`. Run the whole suite with:

  ```bash
  pytest tests/
  ```

- Async tests are configured with `asyncio_mode = "auto"` (see
  `pyproject.toml`), so `async def test_...` functions run on an event loop
  automatically — no `@pytest.mark.asyncio` decorator needed.

- Integration tests for the FastAPI server use
  [`TestClient`](https://fastapi.tiangolo.com/reference/testclient/) /
  `httpx` to exercise the HTTP and WebSocket endpoints end-to-end. These may
  require the `server` extra (`pip install -e ".[server]"`).

- When adding a feature or fixing a bug, add a focused test alongside it. Aim
  for tests that run fast and do not hit real LLM or network endpoints — mock
  the model with `unittest.mock.AsyncMock` or the shared `MockModel` fixture
  in `tests/conftest.py`.

## Pull Request Process

1. **Fork** the repository on GitHub and clone your fork locally.
2. **Create a branch** from `main` named after the change, e.g.
   `feature/rag-reranker` or `fix/shell-sandbox-glob`.
3. **Make your changes**, keeping commits focused and the code style clean
   (see above).
4. **Run the checks** locally before pushing:

   ```bash
   ruff check src tests
   mypy src
   pytest tests/
   ```

5. **Push** your branch and **open a Pull Request** against `main`. Fill in
   the PR template (or describe clearly) what changed and why. Reference any
   related issue (e.g. `Closes #42`).
6. **Address review feedback** — push additional commits to the same branch.
   Prefer keeping the history readable; maintainers may squash on merge.

### Commit Message Convention

Use the **imperative mood** for the subject line, as if giving a command:
"Add", "Fix", "Refactor", "Update", "Remove". Keep the subject under ~72
characters and capitalize it. Optionally add a body explaining the *why*.

Good examples:

- `Add shell sandbox pattern matching`
- `Fix retry backoff for Anthropic rate limits`
- `Refactor retriever to share the embedding cache`
- `Update README quick-start with the new CLI flags`

Avoid vague messages like `fix` or `update stuff`.

## Reporting Bugs & Requesting Features

Use [GitHub Issues](https://github.com/123comux/open-agent/issues) for both
bug reports and feature requests.

- **Bug reports**: include the Open Agent version, Python version, OS, the
  exact command or code that reproduces the problem, and the full error
  traceback. A minimal reproducer script is hugely appreciated.
- **Feature requests**: describe the use case and the problem you are trying
  to solve, not just the proposed solution. Screenshots or example commands
  help.

Security-sensitive issues should **not** be filed publicly — please see the
security policy (or contact a maintainer privately) before disclosing.

## License

Open Agent is released under the **MIT License** (see
[LICENSE](./LICENSE)). By contributing, you agree that your contributions
will be licensed under the same MIT terms. If a file you touch carries a
copyright header, keep it consistent; otherwise no per-file header is
required.

---

Happy hacking! If you get stuck, open an issue or start a discussion — we are
glad to help.