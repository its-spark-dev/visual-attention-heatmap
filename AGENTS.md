# Repository Guidelines

## Project Structure & Module Organization
- `core/` holds the reusable attention engine and feature implementations.
- `core/features/` is reserved for independent, self-contained feature modules.
- `README.md` explains the project scope and goals; `DESIGN.md` captures architectural rules.
- No application layer, datasets, or visualization utilities are checked in yet; keep those outside `core/`.

## Build, Test, and Development Commands
- This repository does not currently define build or run scripts.
- When adding execution entry points, document them here and in `README.md` (example: `python -m your_module`).
- If you introduce tooling (e.g., `make`, `uv`, `poetry`, `pipenv`), add the exact commands and required versions.

## Coding Style & Naming Conventions
- Language: Python. Use 4-space indentation and follow PEP 8.
- Keep modules small and focused; feature modules should expose a single clear responsibility.
- Naming:
  - Files: lowercase with underscores (e.g., `center_bias.py`).
  - Classes: `CamelCase` (e.g., `CenterBiasFeature`).
  - Functions: `snake_case` (e.g., `normalize_map`).
- No formatter/linter is configured yet; avoid reformatting unrelated code.

## Testing Guidelines
- Tests use `pytest` and live under `tests/` (e.g., `tests/test_center_bias.py`).
- Keep tests deterministic; feature outputs should be normalized to `[0, 1]` as stated in `DESIGN.md`.

## Running Tests
- Create a virtual environment with `uv venv .venv`.
- Install dependencies: `uv pip install --python .venv/bin/python pytest numpy`.
- Run the suite: `.venv/bin/python -m pytest -q`.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative, sentence-case messages (e.g., “Add design document…”).
- Keep commits scoped to one change; avoid mixing refactors with behavior changes.
- PRs should include:
  - A concise summary of the change and rationale.
  - Notes on how the change aligns with the core rules in `DESIGN.md`.
  - Any new commands, dependencies, or configuration steps.
  - Screenshots or sample outputs when visual artifacts are affected.

## Architecture Notes
- The core must remain image-type agnostic and purely rule-based.
- Feature modules must be independent, deterministic, and normalized before fusion.
- Presets, UI, I/O, or semantic heuristics belong outside `core/`.
