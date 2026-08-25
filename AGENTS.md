# Repository Guidelines

## Project Structure & Module Organization

Source code lives in `palinstrophy/`. `turbo_simulator.py` contains the DNS solver and backend-specific SciPy, CuPy, and MLX paths; `turbo_main.py` provides the PySide6 GUI; `turbo_wrapper.py` exposes the solver as a Python API. Post-processing and saved-case utilities are in the remaining package modules. Reference screenshots belong in `images/`, while profiling, packaging, correctness, and platform helpers live in `tools/`. Project metadata and locked dependencies are defined by `pyproject.toml` and `uv.lock`. There is currently no dedicated `tests/` directory.

## Build, Test, and Development Commands

- `uv sync --dev` installs the Python 3.13 environment and development tools from the lockfile.
- `uv run turbulence` launches the GUI with automatic backend selection.
- `SCIPYTURBO_SEED=4895 uv run sim 128 10000 10 200 0.05 cpu 50 KM3 CHECK` runs a reproducible, small CPU comparison of both time integrators.
- `uv run sim 256 10000 10 1001 0.75 auto 100 KM3` exercises the normal solver CLI.
- `uv build` creates source and wheel distributions.
- `uv run python -m compileall palinstrophy` performs a quick syntax/import compilation check.

Use small grids for routine validation; production-sized simulations consume substantial memory and time.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase module constants. Add type annotations to new public APIs and short docstrings where behavior or numerical assumptions are not obvious. Keep backend branches numerically equivalent and avoid unnecessary host/device transfers. No formatter or linter is configured, so preserve the surrounding style and keep imports organized.

## Testing Guidelines

No automated test or coverage framework is configured. Every change should at least pass compilation and a small CPU solver run. Numerical or backend changes should also use `CHECK` and, when available, run on the affected CUDA or MLX hardware. Exercise GUI changes interactively and include screenshots for visible layout changes. If introducing tests, place them under `tests/` with names such as `test_turbo_wrapper.py` and document any new test dependency.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Fuse MLX diagnostics` and often include measured performance in parentheses. Keep each commit focused. Pull requests should explain the motivation, affected backends, validation commands, and numerical or performance impact. Link relevant issues and attach before/after screenshots for GUI changes or benchmark figures for optimization work.
