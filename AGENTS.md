# Repository Guidelines

## Project Structure & Module Organization

This repository is a generic LangChain/LangGraph agent demo skeleton. Core source code lives under `src/`. `src/agent.py` assembles the agent, tools, checkpointer, store, and middleware. `src/runtime.py` defines request-scoped context shared across tools and middleware.

Feature areas are module-owned: `src/memory/`, `src/skills/`, `src/summarization/`, and `src/observability/` each contain their own middleware or helpers. Business tools belong in `src/tools/`; do not register skill or memory tools there. Prompt helpers live in `src/prompt/`, and configuration loading/schema code lives in `src/config/`.

Tests are in `tests/` and generally mirror the module they cover, for example `tests/test_skill_middleware.py` and `tests/test_observability_events.py`. Local skill assets use `skills/<skill-name>/SKILL.md`, with optional `references/`, `scripts/`, and `assets/` subdirectories.

## Build, Test, and Development Commands

- `.venv/bin/python main.py` runs the local console demo for streaming, tool calls, and reasoning display.
- `.venv/bin/python -m pytest` runs the full test suite.
- `.venv/bin/python -m pytest tests/test_observability_events.py -q` runs a focused test file during iteration.
- `pip install -e ".[dev]"` installs the package and pytest dependency into the active environment when rebuilding locally.

## Coding Style & Naming Conventions

Use Python 3.12+ syntax, 4-space indentation, type annotations for public helpers, and small module-owned functions. Keep `__init__.py` files limited to imports and public exports; implementation belongs in dedicated module files.

Use descriptive snake_case names for functions, variables, and test cases. Middleware builders should follow `build_<feature>_middleware()`. Keep `main.py` as a thin demo entrypoint, not a place for business logic.

## Testing Guidelines

Use `pytest`. Add or update tests with every behavior change. Prefer focused tests for loaders, middleware, runtime context, and logging helpers. Test names should describe behavior, such as `test_skill_tools_log_rejected_file_reads`.

For observability tests, assert event names and safe metadata, not full prompts, model outputs, memory values, or skill file contents.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat: ...`, `refactor: ...`, `fix: ...`, `test: ...`, or `docs: ...`. Keep commits scoped to one logical change.

Before opening a PR, run `.venv/bin/python -m pytest`. PR descriptions should summarize the change, list validation commands, mention config or schema changes, and note any new extension points.

## Security & Configuration Tips

Use `config.example.yaml` as the safe template and keep local secrets in `config.yaml` or environment variables. Do not log full prompts, tool argument values, model responses, memory values, skill contents, API keys, tokens, passwords, or authorization headers.
