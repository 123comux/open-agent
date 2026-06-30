# Contributing to Open Agent

Thanks for your interest in contributing to Open Agent! This document covers how
to set up a development environment, run the tests, follow the code style, and
submit pull requests.

## Fork and Clone

1. **Fork** the repository on GitHub (top-right **Fork** button).
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/open-agent.git
   cd open-agent
   ```

3. Add the upstream remote so you can stay in sync:

   ```bash
   git remote add upstream https://github.com/your-org/open-agent.git
   ```

4. Create a feature branch for your work:

   ```bash
   git switch -c feat/my-new-feature
   ```

   Branch from the latest `main`:

   ```bash
   git fetch upstream
   git switch main
   git merge --ff-only upstream/main
   git switch -c feat/my-new-feature
   ```

## Set Up a Development Environment

Open Agent targets **Python 3.10, 3.11, and 3.12**. We recommend using a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install the package together with the dev tooling and all optional extras:

```bash
make dev          # equivalent to: pip install -e ".[dev,all]"
```

This pulls in `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, and `mypy`, plus
the `rag`, `server`, and `mcp` runtime extras so the full test suite can run.

## Run the Tests

```bash
make test         # runs pytest (config in pyproject.toml)
```

To see coverage:

```bash
pytest --cov=open_agent --cov-report=term-missing
```

Tests live under `tests/` and are split into `unit/`, `integration/`, `tools/`,
and `e2e/`. New code should come with corresponding tests in the matching
folder.

## Code Style

We use **[ruff](https://docs.astral.sh/ruff/)** for linting/formatting and
**[mypy](https://mypy-lang.org/)** for static type checking. Configuration for
both lives in `pyproject.toml`.

Before you push, make sure all three pass:

```bash
make lint         # ruff check src tests
make typecheck    # mypy src/open_agent
make test         # pytest
```

Conventions:

- Line length is **100** characters.
- The enabled ruff rule sets are `E`, `F`, `I`, `N`, `W`, `UP`.
- `mypy` runs in **strict** mode — avoid `Any` and `# type: ignore` unless
  absolutely necessary (and explain why in a comment if you must).
- Keep imports sorted (ruff `I` will handle this; run `ruff check --fix`).
- Write docstrings for public modules, classes, and functions.

## Submit a Pull Request

1. **Commit your changes** using clear, conventional commit messages, e.g.
   `feat(rag): add PDF document loader`, `fix(cli): handle empty input`, or
   `docs: clarify config table`.
2. **Rebase** onto the latest `main` to avoid merge commits where possible:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Push** to your fork:

   ```bash
   git push origin feat/my-new-feature
   ```

4. **Open a pull request** against `main`. In the PR description:
   - Summarize **what** changed and **why**.
   - Reference any related issues (`Closes #123`).
   - Note any breaking changes or migration steps.
5. **CI must be green.** GitHub Actions runs `ruff`, `mypy`, and `pytest` across
   Python 3.10 / 3.11 / 3.12. Fix any failures before requesting review.
6. Address review feedback by pushing additional commits to the same branch
   (avoid force-pushing after review unless asked).

## Reporting Issues

If you find a bug or have a feature request, please [open an
issue](https://github.com/your-org/open-agent/issues) and include:

- A clear title and description.
- Steps to reproduce (for bugs), including your Python version, OS, and
  `OPEN_AGENT_` config (redact any API keys).
- Expected vs. actual behavior.
- Relevant logs or stack traces.

## Code of Conduct

Be respectful and constructive. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/). Harassment or
disrespectful behavior will not be tolerated.

---

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](../LICENSE).
